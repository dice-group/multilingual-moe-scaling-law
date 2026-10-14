"""Frozen D15 holdouts, paired one-SE selection, and D16 memberships."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
from .data import EXPECTED_READOUTS, EXPECTED_TRAJECTORIES, HOLDOUT_MANIFEST_SCHEMA_VERSION, PAIRED_BOOTSTRAP_REPLICATES, PAIRED_BOOTSTRAP_SEED, RANDOM_HOLDOUT_SEED, RANDOM_HOLDOUT_TRAJECTORIES, RESAMPLING_MANIFEST_SCHEMA_VERSION, UNCERTAINTY_REPLICATES, UNCERTAINTY_SUBSAMPLE_SEED, UNCERTAINTY_SUBSET_TRAJECTORIES, FitRow, Stage1FitContractError, canonical_json_sha256
PRIMARY_FOLD_IDS = ('largest_N_act', 'E32', 'K200_common_languages')
MODEL_ORDER = ('M0', 'M1', 'M2')
GENERALIZATION_HOLDOUT_SCHEMA = 'stage1_fit_generalization_holdouts/v1'
LARGEST_COMPUTE_VIEW = 'largest_logical_compute_top20'

class Stage1FitSelectionError(Stage1FitContractError):
    """Raised when a D15 fold or paired statistic is malformed."""

@dataclass(frozen=True)
class TrajectoryFold:
    fold_id: str
    role: str
    train_trajectory_ids: tuple[str, ...]
    test_trajectory_ids: tuple[str, ...]
    score_languages: tuple[str, ...]
    excluded_test_languages: tuple[str, ...]
    membership_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class TerminalContinuation:
    role: str
    train_readout_ids: tuple[str, ...]
    test_readout_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]
    membership_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class HoldoutManifest:
    input_sha256: str
    primary_folds: tuple[TrajectoryFold, ...]
    terminal_continuation: TerminalContinuation
    random_holdout: TrajectoryFold
    lowest_loss_extrapolation: TrajectoryFold
    random_seed: int
    lowest_loss_definition: str
    manifest_sha256: str
    schema_version: str = HOLDOUT_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['primary_folds'] = [fold.to_dict() for fold in self.primary_folds]
        payload['terminal_continuation'] = self.terminal_continuation.to_dict()
        payload['random_holdout'] = self.random_holdout.to_dict()
        payload['lowest_loss_extrapolation'] = self.lowest_loss_extrapolation.to_dict()
        return payload

@dataclass(frozen=True)
class PairedOneSESelection:
    selected_model: str
    observed_best_model: str
    observed_aggregate_rmse: Mapping[str, float]
    paired_standard_error_vs_best: Mapping[str, float]
    eligibility_threshold: Mapping[str, float]
    eligible_models: tuple[str, ...]
    bootstrap_difference_percentiles_10_90: Mapping[str, tuple[float, float]]
    fold_order: tuple[str, ...]
    bootstrap_replicates: int
    seed: int
    bootstrap_sampling: str
    bootstrap_population_trajectories: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['observed_aggregate_rmse'] = dict(self.observed_aggregate_rmse)
        payload['paired_standard_error_vs_best'] = dict(self.paired_standard_error_vs_best)
        payload['eligibility_threshold'] = dict(self.eligibility_threshold)
        payload['bootstrap_difference_percentiles_10_90'] = {name: list(values) for name, values in self.bootstrap_difference_percentiles_10_90.items()}
        return payload

def _membership_hash(values: Sequence[str]) -> str:
    return canonical_json_sha256(list(sorted((str(value) for value in values))))

def _trajectory_design(rows: Sequence[FitRow]) -> dict[str, tuple[float, int, int]]:
    design: dict[str, tuple[float, int, int]] = {}
    for row in rows:
        value = (float(row.N_act), int(row.E), int(row.K))
        previous = design.setdefault(row.trajectory_id, value)
        if previous != value:
            raise Stage1FitSelectionError(f'trajectory {row.trajectory_id} changes N_act/E/K')
    if not design:
        raise Stage1FitSelectionError('cannot build holdouts from empty rows')
    return design

def _fold(*, fold_id: str, role: str, population: Sequence[str], test_ids: Sequence[str], rows: Sequence[FitRow]) -> TrajectoryFold:
    population_set = set(population)
    test = tuple(sorted((str(value) for value in test_ids)))
    test_set = set(test)
    if not test or not test_set < population_set:
        raise Stage1FitSelectionError(f'{fold_id} must hold out a non-empty proper trajectory subset')
    train = tuple(sorted(population_set - test_set))
    train_languages = {row.language for row in rows if row.trajectory_id in set(train)}
    test_languages = {row.language for row in rows if row.trajectory_id in test_set}
    score_languages = tuple(sorted(train_languages & test_languages))
    excluded = tuple(sorted(test_languages - train_languages))
    if not score_languages:
        raise Stage1FitSelectionError(f'{fold_id} has no estimable test languages')
    membership = {'fold_id': fold_id, 'train_trajectory_ids': list(train), 'test_trajectory_ids': list(test), 'score_languages': list(score_languages), 'excluded_test_languages': list(excluded)}
    return TrajectoryFold(fold_id=fold_id, role=role, train_trajectory_ids=train, test_trajectory_ids=test, score_languages=score_languages, excluded_test_languages=excluded, membership_sha256=canonical_json_sha256(membership))

def _terminal_continuation(rows: Sequence[FitRow]) -> TerminalContinuation:
    maximum: dict[str, float] = {}
    for row in rows:
        maximum[row.trajectory_id] = max(maximum.get(row.trajectory_id, -math.inf), float(row.D_total))
    train_readouts = {row.readout_id for row in rows if float(row.D_total) < maximum[row.trajectory_id]}
    test_readouts = {row.readout_id for row in rows if float(row.D_total) == maximum[row.trajectory_id]}
    for trajectory_id in maximum:
        if not any((row.trajectory_id == trajectory_id and row.readout_id in train_readouts for row in rows)):
            raise Stage1FitSelectionError(f'trajectory {trajectory_id} lacks an earlier continuation readout')
        terminal_ids = {row.readout_id for row in rows if row.trajectory_id == trajectory_id and float(row.D_total) == maximum[trajectory_id]}
        if len(terminal_ids) != 1:
            raise Stage1FitSelectionError(f'trajectory {trajectory_id} has {len(terminal_ids)} terminal readouts')
    payload = {'train_readout_ids': sorted(train_readouts), 'test_readout_ids': sorted(test_readouts), 'trajectory_ids': sorted(maximum)}
    return TerminalContinuation(role='dependent_within_trajectory_continuation_diagnostic', train_readout_ids=tuple(payload['train_readout_ids']), test_readout_ids=tuple(payload['test_readout_ids']), trajectory_ids=tuple(payload['trajectory_ids']), membership_sha256=canonical_json_sha256(payload))

def _random_trajectory_ids(trajectory_ids: Sequence[str], *, count: int, seed: int) -> tuple[str, ...]:
    import numpy as np
    population = tuple(sorted((str(value) for value in trajectory_ids)))
    if len(population) != len(set(population)):
        raise Stage1FitSelectionError('random-holdout population has duplicates')
    if not 0 < count < len(population):
        raise Stage1FitSelectionError('invalid random-holdout trajectory count')
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.choice(len(population), size=count, replace=False)
    return tuple(sorted((population[int(index)] for index in indices)))

def _lowest_loss_trajectory_ids(rows: Sequence[FitRow], *, count: int) -> tuple[str, ...]:
    maximum: dict[str, float] = {}
    for row in rows:
        maximum[row.trajectory_id] = max(maximum.get(row.trajectory_id, -math.inf), float(row.D_total))
    terminal_losses: dict[str, list[float]] = {name: [] for name in maximum}
    for row in rows:
        if float(row.D_total) == maximum[row.trajectory_id]:
            terminal_losses[row.trajectory_id].append(float(row.adjusted_loss))
    if not 0 < count < len(terminal_losses):
        raise Stage1FitSelectionError('invalid lowest-loss trajectory count')
    ranked = sorted(((sum(values) / len(values), trajectory_id) for trajectory_id, values in terminal_losses.items() if values))
    if len(ranked) != len(terminal_losses):
        raise Stage1FitSelectionError('missing terminal loss for trajectory')
    return tuple(sorted((trajectory_id for _, trajectory_id in ranked[:count])))

def build_holdout_manifest(rows: Sequence[FitRow], *, input_sha256: str, random_count: int=RANDOM_HOLDOUT_TRAJECTORIES, random_seed: int=RANDOM_HOLDOUT_SEED, lowest_loss_count: int=30, require_canonical_population: bool=True) -> HoldoutManifest:
    """Freeze the three primary and three explicitly secondary D15 views."""
    design = _trajectory_design(rows)
    population = tuple(sorted(design))
    if require_canonical_population and len(population) != EXPECTED_TRAJECTORIES:
        raise Stage1FitSelectionError(f'expected {EXPECTED_TRAJECTORIES} trajectories, got {len(population)}')
    largest_n = max((value[0] for value in design.values()))
    largest_n_ids = [name for name, value in design.items() if value[0] == largest_n]
    e32_ids = [name for name, value in design.items() if value[1] == 32]
    k200_ids = [name for name, value in design.items() if value[2] == 200]
    primary = (_fold(fold_id='largest_N_act', role='primary_independent_trajectory_extrapolation', population=population, test_ids=largest_n_ids, rows=rows), _fold(fold_id='E32', role='primary_independent_trajectory_extrapolation', population=population, test_ids=e32_ids, rows=rows), _fold(fold_id='K200_common_languages', role='primary_independent_trajectory_extrapolation_common_languages_only', population=population, test_ids=k200_ids, rows=rows))
    random_ids = _random_trajectory_ids(population, count=random_count, seed=random_seed)
    lowest_ids = _lowest_loss_trajectory_ids(rows, count=lowest_loss_count)
    unsigned = {'schema_version': HOLDOUT_MANIFEST_SCHEMA_VERSION, 'input_sha256': input_sha256, 'primary_folds': [fold.to_dict() for fold in primary], 'terminal_continuation': _terminal_continuation(rows).to_dict(), 'random_holdout': _fold(fold_id='random_13_of_64', role='secondary_atlas_comparability_diagnostic', population=population, test_ids=random_ids, rows=rows).to_dict(), 'lowest_loss_extrapolation': _fold(fold_id='lowest_terminal_macro_loss_30', role='secondary_joint_moe_outcome_selected_diagnostic', population=population, test_ids=lowest_ids, rows=rows).to_dict(), 'random_seed': random_seed, 'lowest_loss_definition': 'Rank whole trajectories by terminal equal-language mean adjusted loss; freeze the 30 lowest-loss trajectories. Secondary only.'}
    terminal = _terminal_continuation(rows)
    random_fold = _fold(fold_id='random_13_of_64', role='secondary_atlas_comparability_diagnostic', population=population, test_ids=random_ids, rows=rows)
    lowest_fold = _fold(fold_id='lowest_terminal_macro_loss_30', role='secondary_joint_moe_outcome_selected_diagnostic', population=population, test_ids=lowest_ids, rows=rows)
    unsigned['terminal_continuation'] = terminal.to_dict()
    unsigned['random_holdout'] = random_fold.to_dict()
    unsigned['lowest_loss_extrapolation'] = lowest_fold.to_dict()
    return HoldoutManifest(input_sha256=input_sha256, primary_folds=primary, terminal_continuation=terminal, random_holdout=random_fold, lowest_loss_extrapolation=lowest_fold, random_seed=random_seed, lowest_loss_definition=unsigned['lowest_loss_definition'], manifest_sha256=canonical_json_sha256(unsigned))

def build_generalization_holdout_manifest(rows: Sequence[FitRow], *, input_sha256: str,
    source_holdouts: Mapping[str, str], require_canonical_population: bool=True) -> dict[str, Any]:
    """Seal the dependent top-20% logical-compute readout diagnostic."""
    if set(source_holdouts) != {'path', 'file_sha256', 'manifest_sha256'}: raise Stage1FitSelectionError('source holdout binding is malformed')
    readouts: dict[str, tuple[str, int, int]] = {}; languages: dict[str, set[str]] = {}
    for row in rows:
        if (not math.isfinite(float(row.N_act)) or not float(row.N_act).is_integer()
                or not math.isfinite(float(row.D_total)) or not float(row.D_total).is_integer()):
            raise Stage1FitSelectionError('logical compute requires exact integer N_act and D_total')
        value = (str(row.trajectory_id), int(row.N_act), int(row.D_total))
        previous = readouts.setdefault(str(row.readout_id), value)
        if previous != value: raise Stage1FitSelectionError(f'readout {row.readout_id} changes trajectory, N_act, or D_total')
        languages.setdefault(str(row.readout_id), set()).add(str(row.language))
    if not readouts or (require_canonical_population and len(readouts) != EXPECTED_READOUTS):
        raise Stage1FitSelectionError(f'expected {EXPECTED_READOUTS} readouts, got {len(readouts)}')
    if len(readouts) % 5: raise Stage1FitSelectionError('top-20% split requires a readout count divisible by five')
    target = len(readouts) // 5; compute = {name: 6 * value[1] * value[2] for name, value in readouts.items()}
    cutoff = sorted(compute.values(), reverse=True)[target - 1]; test = sorted(name for name, value in compute.items() if value >= cutoff)
    if len(test) != target: raise Stage1FitSelectionError('logical-compute tie crosses the frozen 20% boundary')
    train = sorted(set(readouts) - set(test)); test_trajectories = sorted({readouts[name][0] for name in test}); train_trajectories = sorted({readouts[name][0] for name in train})
    if not set(test_trajectories) <= set(train_trajectories): raise Stage1FitSelectionError('every high-compute test trajectory must retain an earlier training readout')
    train_languages = set().union(*(languages[name] for name in train)); test_languages = set().union(*(languages[name] for name in test))
    if not test_languages <= train_languages: raise Stage1FitSelectionError('high-compute test languages must be estimable from training readouts')
    membership = {'fold_id': LARGEST_COMPUTE_VIEW, 'role': 'dependent_within_trajectory_logical_compute_diagnostic',
        'train_trajectory_ids': train_trajectories, 'test_trajectory_ids': test_trajectories, 'score_languages': sorted(test_languages),
        'train_readout_ids': train, 'test_readout_ids': test}
    membership['membership_sha256'] = canonical_json_sha256(membership)
    unsigned = {'schema_version': GENERALIZATION_HOLDOUT_SCHEMA, 'input_sha256': input_sha256, 'source_holdouts': dict(source_holdouts),
        'logical_compute': {'formula': 'F=6*N_act*D_total', 'population_unit': 'unique_readout', 'test_fraction': 0.2,
            'test_count': target, 'train_count': len(train), 'threshold_flops': cutoff,
            'tie_policy': 'include_complete_boundary_tie_or_fail'}, LARGEST_COMPUTE_VIEW: membership}
    return {**unsigned, 'manifest_sha256': canonical_json_sha256(unsigned)}

def trajectory_mean_squared_errors(observed: Sequence[float], predicted: Sequence[float], trajectory_ids: Sequence[str]) -> dict[str, float]:
    """Reduce row residuals to the independent trajectory unit."""
    import numpy as np
    y = np.asarray(observed, dtype=np.float64)
    prediction = np.asarray(predicted, dtype=np.float64)
    ids = np.asarray(trajectory_ids, dtype=object)
    if y.ndim != 1 or y.shape != prediction.shape or y.shape != ids.shape or (not y.size):
        raise Stage1FitSelectionError('metric arrays must be non-empty aligned vectors')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise Stage1FitSelectionError('metric arrays must be finite')
    result: dict[str, float] = {}
    for trajectory_id in sorted({str(value) for value in ids}):
        selected = ids == trajectory_id
        result[trajectory_id] = float(np.mean((prediction[selected] - y[selected]) ** 2))
    return result

def trajectory_balanced_rmse(observed: Sequence[float], predicted: Sequence[float], trajectory_ids: Sequence[str]) -> float:
    errors = trajectory_mean_squared_errors(observed, predicted, trajectory_ids)
    return math.sqrt(sum(errors.values()) / len(errors))

def trajectory_balanced_r2(observed: Sequence[float], predicted: Sequence[float], trajectory_ids: Sequence[str]) -> float | None:
    """Return D15 trajectory-weighted R2, or ``None`` for a constant response."""
    import numpy as np
    y = np.asarray(observed, dtype=np.float64)
    prediction = np.asarray(predicted, dtype=np.float64)
    ids = np.asarray(trajectory_ids, dtype=object)
    if y.ndim != 1 or y.shape != prediction.shape or y.shape != ids.shape or (not y.size):
        raise Stage1FitSelectionError('metric arrays must be non-empty aligned vectors')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise Stage1FitSelectionError('metric arrays must be finite')
    unique = sorted({str(value) for value in ids})
    counts = {name: int(np.sum(ids == name)) for name in unique}
    weights = np.asarray([1.0 / (len(unique) * counts[str(name)]) for name in ids], dtype=np.float64)
    mean = float(np.dot(weights, y))
    total = float(np.dot(weights, (y - mean) ** 2))
    if total == 0.0:
        return None
    residual = float(np.dot(weights, (prediction - y) ** 2))
    return 1.0 - residual / total

def aggregate_equal_fold_rmse(fold_rmse: Mapping[str, float]) -> float:
    if not fold_rmse:
        raise Stage1FitSelectionError('cannot aggregate zero folds')
    values = [float(value) for value in fold_rmse.values()]
    if any((not math.isfinite(value) or value < 0.0 for value in values)):
        raise Stage1FitSelectionError('fold RMSE values must be finite and non-negative')
    return sum(values) / len(values)

def _validate_candidate_errors(candidate_fold_trajectory_mse: Mapping[str, Mapping[str, Mapping[str, float]]], model_order: Sequence[str], fold_order: Sequence[str] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    candidates = tuple((name for name in model_order if name in candidate_fold_trajectory_mse))
    if set(candidates) != set(candidate_fold_trajectory_mse):
        raise Stage1FitSelectionError('candidate errors must contain only and all declared ordered models')
    if not candidates:
        raise Stage1FitSelectionError('no candidate errors supplied')
    first_folds = set(candidate_fold_trajectory_mse[candidates[0]])
    folds = tuple(fold_order) if fold_order is not None else tuple(sorted(first_folds))
    if set(folds) != first_folds or not folds:
        raise Stage1FitSelectionError('fold order does not match candidate errors')
    for candidate in candidates:
        if set(candidate_fold_trajectory_mse[candidate]) != set(folds):
            raise Stage1FitSelectionError('candidates do not share identical folds')
    for fold in folds:
        reference_ids = set(candidate_fold_trajectory_mse[candidates[0]][fold])
        if not reference_ids:
            raise Stage1FitSelectionError(f'fold {fold} has zero trajectories')
        for candidate in candidates:
            values = candidate_fold_trajectory_mse[candidate][fold]
            if set(values) != reference_ids:
                raise Stage1FitSelectionError(f'candidates do not share trajectories in fold {fold}')
            if any((not math.isfinite(float(value)) or float(value) < 0.0 for value in values.values())):
                raise Stage1FitSelectionError(f'fold {fold} contains invalid trajectory MSE')
    return (candidates, folds)

def select_model_paired_one_standard_error(candidate_fold_trajectory_mse: Mapping[str, Mapping[str, Mapping[str, float]]], *, model_order: Sequence[str]=MODEL_ORDER, fold_order: Sequence[str] | None=None, replicates: int=PAIRED_BOOTSTRAP_REPLICATES, seed: int=PAIRED_BOOTSTRAP_SEED) -> PairedOneSESelection:
    """Apply D15's equal-fold, paired whole-trajectory one-SE rule.

    The primary folds overlap.  Each bootstrap draw therefore assigns one
    multiplicity to every trajectory in the union of their test sets and reuses
    that multiplicity wherever the trajectory occurs.  Drawing each fold
    independently would incorrectly treat overlapping observations as
    independent.
    """
    import numpy as np
    if replicates < 2:
        raise Stage1FitSelectionError('paired bootstrap needs at least two replicates')
    candidates, folds = _validate_candidate_errors(candidate_fold_trajectory_mse, model_order, fold_order)
    observed: dict[str, float] = {}
    for candidate in candidates:
        fold_scores = {fold: math.sqrt(sum(candidate_fold_trajectory_mse[candidate][fold].values()) / len(candidate_fold_trajectory_mse[candidate][fold])) for fold in folds}
        observed[candidate] = aggregate_equal_fold_rmse(fold_scores)
    observed_best = min(candidates, key=lambda name: (observed[name], candidates.index(name)))
    generator = np.random.Generator(np.random.PCG64(seed))
    bootstrapped = {candidate: np.empty(replicates, dtype=np.float64) for candidate in candidates}
    ordered_ids = {fold: tuple(sorted(candidate_fold_trajectory_mse[candidates[0]][fold])) for fold in folds}
    population = tuple(sorted({trajectory_id for identifiers in ordered_ids.values() for trajectory_id in identifiers}))
    population_index = {trajectory_id: index for index, trajectory_id in enumerate(population)}
    for replicate in range(replicates):
        # A vanishingly rare draw can assign zero mass to one entire fold.  Such
        # a draw has no fold RMSE, so deterministically redraw the whole union.
        for _ in range(10_000):
            sampled_indices = generator.integers(0, len(population), size=len(population))
            multiplicities = np.bincount(sampled_indices, minlength=len(population))
            fold_masses = {fold: int(sum(multiplicities[population_index[name]] for name in ordered_ids[fold]))
                for fold in folds}
            if all(value > 0 for value in fold_masses.values()):
                break
        else:
            raise Stage1FitSelectionError('could not draw a non-empty shared trajectory bootstrap for every fold')
        aggregate = {candidate: 0.0 for candidate in candidates}
        for fold in folds:
            identifiers = ordered_ids[fold]
            for candidate in candidates:
                values = candidate_fold_trajectory_mse[candidate][fold]
                sampled_mse = sum(float(values[name]) * multiplicities[population_index[name]] for name in identifiers) / fold_masses[fold]
                aggregate[candidate] += math.sqrt(sampled_mse) / len(folds)
        for candidate in candidates:
            bootstrapped[candidate][replicate] = aggregate[candidate]
    standard_errors: dict[str, float] = {}
    percentiles: dict[str, tuple[float, float]] = {}
    thresholds: dict[str, float] = {}
    best_values = bootstrapped[observed_best]
    minimum = observed[observed_best]
    for candidate in candidates:
        differences = bootstrapped[candidate] - best_values
        standard_errors[candidate] = float(np.std(differences, ddof=1))
        lower, upper = np.percentile(differences, [10.0, 90.0])
        percentiles[candidate] = (float(lower), float(upper))
        thresholds[candidate] = minimum + standard_errors[candidate]
    eligible = tuple((candidate for candidate in candidates if observed[candidate] <= thresholds[candidate]))
    if not eligible:
        raise Stage1FitSelectionError('one-SE rule produced no eligible model')
    return PairedOneSESelection(selected_model=eligible[0], observed_best_model=observed_best, observed_aggregate_rmse=observed, paired_standard_error_vs_best=standard_errors, eligibility_threshold=thresholds, eligible_models=eligible, bootstrap_difference_percentiles_10_90=percentiles, fold_order=folds, bootstrap_replicates=replicates, seed=seed, bootstrap_sampling='one multinomial draw over the union of held-out trajectories; shared multiplicities across overlapping folds', bootstrap_population_trajectories=len(population))
class Stage1FitUncertaintyError(Stage1FitContractError):
    """Raised when the frozen D16 membership contract is incomplete."""

@dataclass(frozen=True)
class UncertaintySubset:
    replicate_id: int
    trajectory_ids: tuple[str, ...]
    membership_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ResamplingManifest:
    input_sha256: str
    population_trajectory_ids: tuple[str, ...]
    subsets: tuple[UncertaintySubset, ...]
    bit_generator: str
    seed: int
    subset_size: int
    sampling: str
    manifest_sha256: str
    schema_version: str = RESAMPLING_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['subsets'] = [subset.to_dict() for subset in self.subsets]
        return payload

def generate_uncertainty_subsets(trajectory_ids: Sequence[str], *, replicates: int=UNCERTAINTY_REPLICATES, subset_size: int=UNCERTAINTY_SUBSET_TRAJECTORIES, seed: int=UNCERTAINTY_SUBSAMPLE_SEED, require_canonical_population: bool=True) -> tuple[UncertaintySubset, ...]:
    """Generate the frozen 100 independent 51-of-64 PCG64 subsets."""
    import numpy as np
    population = tuple(sorted((str(value) for value in trajectory_ids)))
    if len(population) != len(set(population)):
        raise Stage1FitUncertaintyError('uncertainty population has duplicates')
    if require_canonical_population and len(population) != EXPECTED_TRAJECTORIES:
        raise Stage1FitUncertaintyError(f'expected {EXPECTED_TRAJECTORIES} trajectories, got {len(population)}')
    if replicates < 1 or not 0 < subset_size < len(population):
        raise Stage1FitUncertaintyError('invalid uncertainty replicate/subset count')
    generator = np.random.Generator(np.random.PCG64(seed))
    subsets: list[UncertaintySubset] = []
    for replicate_id in range(replicates):
        indices = generator.choice(len(population), size=subset_size, replace=False)
        membership = tuple(sorted((population[int(index)] for index in indices)))
        subsets.append(UncertaintySubset(replicate_id=replicate_id, trajectory_ids=membership, membership_sha256=canonical_json_sha256(list(membership))))
    return tuple(subsets)

def build_resampling_manifest(trajectory_ids: Sequence[str], *, input_sha256: str, replicates: int=UNCERTAINTY_REPLICATES, subset_size: int=UNCERTAINTY_SUBSET_TRAJECTORIES, seed: int=UNCERTAINTY_SUBSAMPLE_SEED, require_canonical_population: bool=True) -> ResamplingManifest:
    population = tuple(sorted((str(value) for value in trajectory_ids)))
    subsets = generate_uncertainty_subsets(population, replicates=replicates, subset_size=subset_size, seed=seed, require_canonical_population=require_canonical_population)
    unsigned = {'schema_version': RESAMPLING_MANIFEST_SCHEMA_VERSION, 'input_sha256': input_sha256, 'population_trajectory_ids': list(population), 'subsets': [subset.to_dict() for subset in subsets], 'bit_generator': 'PCG64', 'seed': seed, 'subset_size': subset_size, 'sampling': 'without_replacement_within_each_replicate'}
    return ResamplingManifest(input_sha256=input_sha256, population_trajectory_ids=population, subsets=subsets, bit_generator='PCG64', seed=seed, subset_size=subset_size, sampling='without_replacement_within_each_replicate', manifest_sha256=canonical_json_sha256(unsigned))
