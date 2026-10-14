"""Exact resource-feasible MinSum/MaxSum selection for the two K12 arms."""

from __future__ import annotations

import math
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from multilingual_megatron.experiments.scaling.language_composition.selection.common import (
    PRIMARY_AXES,
    ResourceStats,
    SelectionError,
    language_set_sha256,
    mixture_stats,
    resource_violations,
)
from multilingual_megatron.experiments.scaling.language_composition.selection.distances import DistanceBundle, mean_pairwise_distance


MIP_GAP_NUMERICAL_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SolverSettings:
    """Explicit non-scientific controls for the two independent solves."""

    arm_workers: int = 1
    solver_threads: int = 1
    show_progress: bool = False

    def validate(self) -> None:
        if self.arm_workers not in {1, 2}:
            raise SelectionError("arm_workers must be 1 or 2")
        if self.solver_threads < 1:
            raise SelectionError("solver_threads must be at least 1")


def _resource_rows(
    *,
    languages: Sequence[str],
    availability: Mapping[str, int],
    pairs: Sequence[tuple[int, int]],
    policy: Mapping[str, Any],
    reference: ResourceStats,
) -> list[tuple[dict[int, float], float, float]]:
    """Linearize English share, effective K, and no-cycling constraints."""

    count = len(languages)
    pools = np.asarray([availability[language] for language in languages], dtype=float)
    required = str(policy["required_language"])
    english_index = languages.index(required)
    alpha = float(policy["alpha"])
    relative = pools**alpha / pools[english_index] ** alpha
    english_tolerance = float(policy["english_probability_absolute_tolerance"])
    rows = [
        (
            {index: float(relative[index]) for index in range(count)},
            1.0 / (reference.english_probability + english_tolerance),
            1.0 / (reference.english_probability - english_tolerance),
        )
    ]

    squared_sum = {index: float(relative[index] ** 2) for index in range(count)}
    for pair_index, (first, second) in enumerate(pairs):
        squared_sum[count + pair_index] = float(
            2.0 * relative[first] * relative[second]
        )
    denominator = {index: float(relative[index] ** 2) for index in range(count)}
    effective_tolerance = float(
        policy["effective_language_count_absolute_tolerance"]
    )
    effective_bounds = (
        (reference.effective_language_count - effective_tolerance, 0.0, math.inf),
        (reference.effective_language_count + effective_tolerance, -math.inf, 0.0),
    )
    for effective, lower, upper in effective_bounds:
        rows.append(
            (
                {
                    index: value - effective * denominator.get(index, 0.0)
                    for index, value in squared_sum.items()
                },
                lower,
                upper,
            )
        )

    total_tokens = float(policy["total_tokens"])
    for selected_index, available in enumerate(pools):
        coefficients = {
            index: -float(available / total_tokens * relative[index])
            for index in range(count)
        }
        coefficients[selected_index] += float(relative[selected_index])
        rows.append((coefficients, -math.inf, 0.0))
    return rows


def _solve_arm(
    arm: str,
    objective: np.ndarray,
    integrality: np.ndarray,
    constraint_matrix: Any,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: SolverSettings,
) -> dict[str, Any]:
    options: dict[str, object] = {
        "disp": settings.show_progress,
        "mip_rel_gap": 0.0,
        "presolve": True,
        "random_seed": 0,
        "threads": settings.solver_threads,
    }
    with warnings.catch_warnings():
        # SciPy deliberately forwards these two valid HiGHS controls while
        # warning that they are outside scipy.optimize.milp's small allowlist.
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        result = milp(
            -objective if arm == "diverse" else objective,
            integrality=integrality,
            bounds=Bounds(np.zeros(len(objective)), np.ones(len(objective))),
            constraints=LinearConstraint(constraint_matrix, lower, upper),
            options=options,
        )
    gap = getattr(result, "mip_gap", None)
    if (
        result.status != 0
        or result.x is None
        or gap is None
        or abs(float(gap)) > MIP_GAP_NUMERICAL_TOLERANCE
    ):
        raise SelectionError(
            f"{arm} MILP did not prove optimality: {result.message}; "
            f"status={result.status}, mip_gap={gap}, has_solution={result.x is not None}"
        )
    return {
        "arm": arm,
        "solution": np.asarray(result.x, dtype=float),
        "trace": {
            "arm": arm,
            "status": int(result.status),
            "message": str(result.message),
            "mip_gap": float(gap),
            "mip_gap_numerical_tolerance": MIP_GAP_NUMERICAL_TOLERANCE,
            "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
            "solver_objective": float(result.fun),
            "solver_threads": settings.solver_threads,
            "random_seed": 0,
        },
    }


def _arm_record(
    *,
    name: str,
    languages: Sequence[str],
    indices: Sequence[int],
    bundle: DistanceBundle,
    availability: Mapping[str, int],
    policy: Mapping[str, Any],
    reference_stats: ResourceStats,
) -> dict[str, Any]:
    members = tuple(languages[index] for index in indices)
    stats = mixture_stats(
        members,
        availability,
        alpha=float(policy["alpha"]),
        total_tokens=int(policy["total_tokens"]),
        required_language=str(policy["required_language"]),
    )
    violations = resource_violations(
        stats,
        reference=reference_stats,
        english_tolerance=float(policy["english_probability_absolute_tolerance"]),
        effective_language_count_tolerance=float(
            policy["effective_language_count_absolute_tolerance"]
        ),
    )
    return {
        "name": name,
        "languages": list(members),
        "language_set_sha256": language_set_sha256(members),
        "composite_score": mean_pairwise_distance(bundle.composite, indices),
        "axis_scores": {
            axis: mean_pairwise_distance(bundle.matrices[axis], indices)
            for axis in PRIMARY_AXES
        },
        "resource_stats": stats.to_dict(),
        "resource_violations": violations,
    }


def select_language_sets(
    *,
    spec: Mapping[str, Any],
    languages: Sequence[str],
    reference_languages: Sequence[str],
    availability: Mapping[str, int],
    bundle: DistanceBundle,
    settings: SolverSettings = SolverSettings(),
) -> dict[str, Any]:
    """Independently minimize and maximize mean pairwise composite distance."""

    settings.validate()
    if tuple(languages) != bundle.languages:
        raise SelectionError("Optimizer language order differs from distance order")
    policy = spec["selection_policy"]
    count = len(languages)
    set_size = int(policy["set_size"])
    required = str(policy["required_language"])
    if required not in languages or not set(reference_languages) <= set(languages):
        raise SelectionError("Required or reference language is outside the frame")
    if len(reference_languages) != set_size:
        raise SelectionError("Reference size differs from selected-arm size")

    reference_stats = mixture_stats(
        reference_languages,
        availability,
        alpha=float(policy["alpha"]),
        total_tokens=int(policy["total_tokens"]),
        required_language=required,
    )
    pairs = [
        (first, second)
        for first in range(count)
        for second in range(first + 1, count)
    ]
    variable_count = count + len(pairs)
    rows: list[tuple[dict[int, float], float, float]] = [
        ({index: 1.0 for index in range(count)}, set_size, set_size),
        ({languages.index(required): 1.0}, 1.0, 1.0),
    ]
    for pair_index, (first, second) in enumerate(pairs):
        pair = count + pair_index
        rows.extend(
            [
                ({pair: 1.0, first: -1.0}, -math.inf, 0.0),
                ({pair: 1.0, second: -1.0}, -math.inf, 0.0),
                ({pair: 1.0, first: -1.0, second: -1.0}, -1.0, math.inf),
            ]
        )
    pair_count = set_size * (set_size - 1) // 2
    rows.append(
        (
            {count + pair_index: 1.0 for pair_index in range(len(pairs))},
            pair_count,
            pair_count,
        )
    )
    for language_index in range(count):
        coefficients = {language_index: -float(set_size - 1)}
        for pair_index, pair in enumerate(pairs):
            if language_index in pair:
                coefficients[count + pair_index] = 1.0
        rows.append((coefficients, 0.0, 0.0))
    rows.extend(
        _resource_rows(
            languages=languages,
            availability=availability,
            pairs=pairs,
            policy=policy,
            reference=reference_stats,
        )
    )

    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    lower = np.empty(len(rows), dtype=float)
    upper = np.empty(len(rows), dtype=float)
    for row_index, (coefficients, row_lower, row_upper) in enumerate(rows):
        lower[row_index], upper[row_index] = row_lower, row_upper
        for column, value in coefficients.items():
            matrix[row_index, column] = value

    objective = np.zeros(variable_count, dtype=float)
    for pair_index, (first, second) in enumerate(pairs):
        objective[count + pair_index] = (
            bundle.composite[first, second] / pair_count
        )
    integrality = np.r_[np.ones(count, dtype=int), np.zeros(len(pairs), dtype=int)]
    shared_arguments = (
        objective,
        integrality,
        matrix.tocsr(),
        lower,
        upper,
        settings,
    )
    arms = ("compact", "diverse")
    if settings.arm_workers == 1:
        outcomes = {arm: _solve_arm(arm, *shared_arguments) for arm in arms}
    else:
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {
                arm: executor.submit(_solve_arm, arm, *shared_arguments)
                for arm in arms
            }
            outcomes = {arm: future.result() for arm, future in futures.items()}

    records: dict[str, dict[str, Any]] = {}
    for arm in arms:
        solution = outcomes[arm]["solution"]
        indices = [
            index for index, value in enumerate(solution[:count]) if value > 0.5
        ]
        if len(indices) != set_size:
            raise SelectionError(
                f"{arm} MILP selected {len(indices)} languages, expected {set_size}"
            )
        records[arm] = _arm_record(
            name=arm,
            languages=languages,
            indices=indices,
            bundle=bundle,
            availability=availability,
            policy=policy,
            reference_stats=reference_stats,
        )
        solver_objective = abs(float(outcomes[arm]["trace"]["solver_objective"]))
        if not math.isclose(
            solver_objective,
            float(records[arm]["composite_score"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise SelectionError(f"{arm} solver objective does not replay")

    reference_indices = [languages.index(language) for language in reference_languages]
    records["reference"] = _arm_record(
        name="reference",
        languages=languages,
        indices=reference_indices,
        bundle=bundle,
        availability=availability,
        policy=policy,
        reference_stats=reference_stats,
    )
    compact = records["compact"]
    reference = records["reference"]
    diverse = records["diverse"]

    directional: dict[str, dict[str, Any]] = {}
    score_triplets = {
        "composite": (
            compact["composite_score"],
            reference["composite_score"],
            diverse["composite_score"],
        ),
        **{
            axis: (
                compact["axis_scores"][axis],
                reference["axis_scores"][axis],
                diverse["axis_scores"][axis],
            )
            for axis in PRIMARY_AXES
        },
    }
    for axis, (compact_score, reference_score, diverse_score) in score_triplets.items():
        directional[axis] = {
            "passed": compact_score < reference_score < diverse_score,
            "compact": compact_score,
            "reference": reference_score,
            "diverse": diverse_score,
            "compact_to_reference_margin": reference_score - compact_score,
            "reference_to_diverse_margin": diverse_score - reference_score,
        }
    directional_passed = all(gate["passed"] for gate in directional.values())

    promotion = spec.get("promotion_policy", {})
    expected_hashes = promotion.get("expected_language_set_sha256", {})
    membership_gates = {
        arm: {
            "passed": records[arm]["language_set_sha256"] == expected_hashes.get(arm),
            "expected": expected_hashes.get(arm),
            "observed": records[arm]["language_set_sha256"],
        }
        for arm in arms
    }
    membership_passed = all(gate["passed"] for gate in membership_gates.values())
    resource_passed = all(not records[name]["resource_violations"] for name in records)
    solver_passed = all(
        outcome["trace"]["status"] == 0
        and abs(outcome["trace"]["mip_gap"]) <= MIP_GAP_NUMERICAL_TOLERANCE
        for outcome in outcomes.values()
    )

    shared = sorted(set(compact["languages"]) & set(diverse["languages"]))
    unchanged_pairs = len(shared) * (len(shared) - 1) // 2
    return {
        "status": (
            "passed"
            if directional_passed
            and membership_passed
            and resource_passed
            and solver_passed
            else "blocked"
        ),
        "method": {
            "objective": "mean within-set pairwise composite distance",
            "compact": "independent exact minimum",
            "diverse": "independent exact maximum",
            "pair_count_per_arm": pair_count,
            "axes": list(PRIMARY_AXES),
            "axis_weights": {axis: 0.25 for axis in PRIMARY_AXES},
        },
        "reference": reference,
        "compact": compact,
        "diverse": diverse,
        "gates": {
            "all_directional_gates_passed": directional_passed,
            "directional": directional,
            "expected_memberships_passed": membership_passed,
            "expected_memberships": membership_gates,
            "resource_constraints_passed": resource_passed,
            "solver_optimality_passed": solver_passed,
        },
        "overlap": {
            "shared_languages": shared,
            "shared_count": len(shared),
            "jaccard": len(shared)
            / len(set(compact["languages"]) | set(diverse["languages"])),
            "unchanged_internal_pairs": unchanged_pairs,
            "changed_internal_pairs_per_arm": pair_count - unchanged_pairs,
            "pair_count_per_arm": pair_count,
        },
        "solver_trace": [outcomes[arm]["trace"] for arm in arms],
    }
