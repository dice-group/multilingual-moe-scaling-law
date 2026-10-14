from __future__ import annotations

import hashlib
import json
from pathlib import Path
import string
import sys
import tempfile
import unittest
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import Stage1ReleaseQAError
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import build_language_dimensions
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import build_validation_cohorts
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import reconstruct_and_verify_aggregates
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import validate_duplicate_candidate_routing
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import validate_routing_rows
from multilingual_megatron.experiments.scaling.stage1_release.validation_qa import validate_validation_identities


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ordered_hash(languages: list[str]) -> str:
    digest = hashlib.sha256()
    for language in languages:
        digest.update(language.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _production_languages() -> list[str]:
    prefixes = [
        first + second + third
        for first in string.ascii_lowercase
        for second in string.ascii_lowercase
        for third in string.ascii_lowercase
    ]
    return [f"{prefix}_Latn" for prefix in prefixes[:200]]


def _language_repo(root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    languages = _production_languages()
    for size in (1, 12, 50, 200):
        members = languages[:size]
        _write_json(
            root / "configs" / "language_groupings" / f"scaling_fw2_v2_k{size:03d}.json",
            {
                "schema_version": 2,
                "language_set_id": f"test_K{size}",
                "language_set_sha256": _ordered_hash(members),
                "0": {
                    "group": f"k{size:03d}",
                    "languages": members,
                },
            },
        )
    first_prefix = languages[0].split("_")[0]
    second_prefix = languages[1].split("_")[0]
    _write_json(
        root / "configs" / "scaling_law" / "multilingual_moe_v1" / "language_sets.json",
        {
            "study_id": "legacy_test",
            "sets": {
                "K200": {
                    "languages": [
                        {
                            "id": f"{first_prefix}_Latn",
                            "family": "Family One",
                            "name": "Language One",
                            "resource_availability": "HIGH",
                            "script": "Latn",
                        },
                        {
                            "id": f"{second_prefix}_Cyrl",
                            "family": "Family Two",
                            "name": "Language Two",
                            "resource_availability": "LOW",
                            "script": "Cyrl",
                        },
                    ]
                }
            },
        },
    )
    corpus = [
        {
            "language": language,
            "dataset_release_id": "test_release",
            "exact_unique_language_tokens_including_eod": 1000 + index,
            "unique_language_tokens_lower_bound_including_eod": 900 + index,
            "train_status": "ready",
            "train_selected_tokens": 1000,
            "validation_status": "ready",
            "validation_selected_tokens": 100,
            "complete_source_scan": True,
            "corpus_manifest_sha256": hashlib.sha256(b"corpus").hexdigest(),
        }
        for index, language in enumerate(languages)
    ]
    return languages, corpus


def _language_row(
    *,
    readout_id: str,
    language: str,
    role: str,
    nll_sum: float,
) -> dict[str, Any]:
    token_count = 2
    adjusted_nll_sum = nll_sum - 2.0
    target_sha = hashlib.sha256(f"{language}:targets".encode()).hexdigest()
    unigram_sha = hashlib.sha256(f"{language}:unigram".encode()).hexdigest()
    input_sha = _payload_sha256(
        {
            "schema": "multilingual_validation_language_loss/v1",
            "language": language,
            "nll_sum_hex": float(nll_sum).hex(),
            "token_count": token_count,
            "target_token_ids_sha256": target_sha,
            "unigram_counts_sha256": unigram_sha,
        }
    )
    validation_input_identity_sha = _payload_sha256(
        {
            "schema": "multilingual_validation_input_identity/v1",
            "language": language,
            "token_count": token_count,
            "target_token_ids_sha256": target_sha,
            "unigram_counts_sha256": unigram_sha,
            "vocab_size": 16,
            "special_token_ids": [0, 1],
        }
    )
    return {
        "readout_id": readout_id,
        "language": language,
        "metric_role": role,
        "token_count": token_count,
        "nll_sum": nll_sum,
        "raw_nll": nll_sum / token_count,
        "adjusted_nll_sum": adjusted_nll_sum,
        "atlas_tao_unigram_adjusted_loss": adjusted_nll_sum / token_count,
        "target_token_ids_sha256": target_sha,
        "unigram_counts_sha256": unigram_sha,
        "input_sha256": input_sha,
        "input_sha256_semantics": (
            "outcome_inclusive_multilingual_validation_language_loss_v1_record_hash"
        ),
        "validation_input_identity_sha256": validation_input_identity_sha,
        "vocab_size": 16,
        "special_token_ids_json": "[0,1]",
    }


def _aggregate_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "status": "empty",
            "aggregate_language_count": 0,
            "aggregate_token_count": 0,
            "equal_language_adjusted_loss": None,
            "equal_language_raw_nll": None,
            "token_micro_adjusted_loss": None,
            "token_micro_raw_nll": None,
            "languages_json": "[]",
        }
    tokens = sum(row["token_count"] for row in rows)
    ordered = sorted(rows, key=lambda row: row["language"])
    return {
        "status": "complete",
        "aggregate_language_count": len(rows),
        "aggregate_token_count": tokens,
        "equal_language_adjusted_loss": sum(
            row["atlas_tao_unigram_adjusted_loss"] for row in rows
        )
        / len(rows),
        "equal_language_raw_nll": sum(row["raw_nll"] for row in rows) / len(rows),
        "token_micro_adjusted_loss": sum(row["adjusted_nll_sum"] for row in rows)
        / tokens,
        "token_micro_raw_nll": sum(row["nll_sum"] for row in rows) / tokens,
        "languages_json": json.dumps(
            [row["language"] for row in ordered],
            separators=(",", ":"),
        ),
    }


def _aggregate_rows(
    readout_id: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trained = [row for row in rows if row["metric_role"] == "trained"]
    zero_shot = [row for row in rows if row["metric_role"] == "zero_shot"]
    result = [
        {
            "readout_id": readout_id,
            "aggregate_kind": "metric_role",
            "cohort": "trained_languages",
            **_aggregate_values(trained),
            "membership_sha256": None,
            "missing_languages_json": "[]",
            "reason": None,
        },
        {
            "readout_id": readout_id,
            "aggregate_kind": "metric_role",
            "cohort": "zero_shot_languages",
            **_aggregate_values(zero_shot),
            "membership_sha256": None,
            "missing_languages_json": "[]",
            "reason": None,
        },
    ]
    trained_values = _aggregate_values(trained)
    result.extend(
        (
            {
                "readout_id": readout_id,
                "aggregate_kind": "declared_metric",
                "cohort": "primary",
                "status": "complete",
                "metric_name": (
                    "trained_equal_language_atlas_tao_unigram_adjusted_loss"
                ),
                "metric_value": trained_values["equal_language_adjusted_loss"],
                "aggregate_language_count": len(trained),
                "aggregate_token_count": None,
                "equal_language_adjusted_loss": None,
                "equal_language_raw_nll": None,
                "token_micro_adjusted_loss": None,
                "token_micro_raw_nll": None,
                "languages_json": trained_values["languages_json"],
                "membership_sha256": None,
                "missing_languages_json": "[]",
                "reason": None,
            },
            {
                "readout_id": readout_id,
                "aggregate_kind": "declared_metric",
                "cohort": "secondary",
                "status": "complete",
                "metric_name": "trained_equal_language_raw_nll",
                "metric_value": trained_values["equal_language_raw_nll"],
                "aggregate_language_count": len(trained),
                "aggregate_token_count": None,
                "equal_language_adjusted_loss": None,
                "equal_language_raw_nll": None,
                "token_micro_adjusted_loss": None,
                "token_micro_raw_nll": None,
                "languages_json": trained_values["languages_json"],
                "membership_sha256": None,
                "missing_languages_json": "[]",
                "reason": None,
            },
        )
    )
    by_language = {row["language"]: row for row in rows}
    fixed = {
        "english": ["eng_Latn"],
        "k12": ["eng_Latn", "fra_Latn"],
        "k50": ["deu_Latn", "eng_Latn", "fra_Latn"],
    }
    trained_languages = set(row["language"] for row in trained)
    for cohort, members in fixed.items():
        membership_sha = _payload_sha256({"name": cohort, "languages": members})
        missing = sorted(set(members) - trained_languages)
        if missing:
            result.append(
                {
                    "readout_id": readout_id,
                    "aggregate_kind": "fixed_common_cohort",
                    "cohort": cohort,
                    "status": "not_applicable",
                    "metric_name": None,
                    "metric_value": None,
                    "aggregate_language_count": None,
                    "aggregate_token_count": None,
                    "equal_language_adjusted_loss": None,
                    "equal_language_raw_nll": None,
                    "token_micro_adjusted_loss": None,
                    "token_micro_raw_nll": None,
                    "languages_json": json.dumps(members, separators=(",", ":")),
                    "membership_sha256": membership_sha,
                    "missing_languages_json": json.dumps(
                        missing, separators=(",", ":")
                    ),
                    "reason": "not_all_cohort_languages_were_trained",
                }
            )
        else:
            result.append(
                {
                    "readout_id": readout_id,
                    "aggregate_kind": "fixed_common_cohort",
                    "cohort": cohort,
                    **_aggregate_values([by_language[item] for item in members]),
                    "metric_name": None,
                    "metric_value": None,
                    "membership_sha256": membership_sha,
                    "missing_languages_json": "[]",
                    "reason": None,
                }
            )
    return result


class Stage1AdReleaseValidationQATest(unittest.TestCase):
    def test_language_dimension_uses_current_script_and_discloses_legacy_join(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            languages, corpus = _language_repo(Path(temporary))

            result = build_language_dimensions(Path(temporary), corpus)

        self.assertEqual(len(result.languages), 200)
        self.assertEqual(len(result.memberships), 263)
        self.assertEqual(len(result.sources), 5)
        first = result.languages[0]
        second = result.languages[1]
        self.assertEqual(first["language"], languages[0])
        self.assertEqual(first["legacy_metadata_join_method"], "unique_iso639_3_prefix")
        self.assertEqual(first["name"], "Language One")
        self.assertFalse(first["stale_script_mismatch"])
        self.assertEqual(second["production_script"], "Latn")
        self.assertEqual(second["legacy_script"], "Cyrl")
        self.assertTrue(second["stale_script_mismatch"])
        self.assertEqual(
            second["exact_unique_language_tokens_including_eod"],
            1001,
        )
        self.assertEqual(
            result.languages[2]["legacy_metadata_join_method"],
            "no_legacy_iso639_3_prefix",
        )

    def test_reconstructs_every_aggregate_and_rejects_changed_mean(self) -> None:
        readout = "prod_eval_a_e1_k002_d1"
        rows = [
            _language_row(
                readout_id=readout,
                language="eng_Latn",
                role="trained",
                nll_sum=4.0,
            ),
            _language_row(
                readout_id=readout,
                language="fra_Latn",
                role="trained",
                nll_sum=6.0,
            ),
            _language_row(
                readout_id=readout,
                language="deu_Latn",
                role="zero_shot",
                nll_sum=8.0,
            ),
        ]
        aggregates = _aggregate_rows(readout, rows)

        cohorts = build_validation_cohorts(aggregates)
        result = reconstruct_and_verify_aggregates(rows, aggregates)

        self.assertEqual(len(cohorts), 7)
        self.assertEqual(result.summary["status"], "pass")
        self.assertEqual(result.summary["verified_aggregate_count"], 7)
        self.assertEqual(result.summary["max_absolute_error"], 0.0)

        changed = [dict(row) for row in aggregates]
        changed[0]["equal_language_raw_nll"] += 0.1
        with self.assertRaisesRegex(
            Stage1ReleaseQAError,
            "independently reconstructed",
        ):
            reconstruct_and_verify_aggregates(rows, changed)

    def test_validation_identity_reconstructs_input_hash_and_detects_drift(
        self,
    ) -> None:
        rows = [
            _language_row(
                readout_id=readout,
                language=language,
                role="trained",
                nll_sum=nll,
            )
            for readout, nll in (("readout_1", 4.0), ("readout_2", 5.0))
            for language in ("eng_Latn", "fra_Latn")
        ]

        result = validate_validation_identities(rows)

        self.assertEqual(result.summary["outcome_inclusive_record_hashes_verified"], 4)
        self.assertEqual(result.summary["validation_input_identity_hashes_verified"], 4)
        self.assertEqual(result.summary["language_count"], 2)
        changed = [dict(row) for row in rows]
        changed[-1]["target_token_ids_sha256"] = hashlib.sha256(
            b"different"
        ).hexdigest()
        changed[-1]["input_sha256"] = _payload_sha256(
            {
                "schema": "multilingual_validation_language_loss/v1",
                "language": changed[-1]["language"],
                "nll_sum_hex": float(changed[-1]["nll_sum"]).hex(),
                "token_count": changed[-1]["token_count"],
                "target_token_ids_sha256": changed[-1]["target_token_ids_sha256"],
                "unigram_counts_sha256": changed[-1]["unigram_counts_sha256"],
            }
        )
        changed[-1]["validation_input_identity_sha256"] = _payload_sha256(
            {
                "schema": "multilingual_validation_input_identity/v1",
                "language": changed[-1]["language"],
                "token_count": changed[-1]["token_count"],
                "target_token_ids_sha256": changed[-1]["target_token_ids_sha256"],
                "unigram_counts_sha256": changed[-1]["unigram_counts_sha256"],
                "vocab_size": changed[-1]["vocab_size"],
                "special_token_ids": [0, 1],
            }
        )
        with self.assertRaisesRegex(Stage1ReleaseQAError, "input drift"):
            validate_validation_identities(changed)

    def test_routing_dimensions_and_assignment_conservation(self) -> None:
        configs = [
            {
                "ladder": "A",
                "experts": 4,
                "num_layers": 2,
                "num_moe_experts_runtime": 4,
                "moe_router_topk": 1,
            }
        ]
        rows = [
            {
                "readout_id": "readout",
                "language": "eng_Latn",
                "ladder": "A",
                "experts": 4,
                "layer_index": layer,
                "topk": 1,
                "expected_assignments": 10,
                "kept_assignments": 10,
                "dropped_assignments": 0,
                "dropped_assignment_rate": 0.0,
                "forwarded_tokens": 10,
                "expert_token_counts_json": "[1,2,3,4]",
            }
            for layer in range(2)
        ]

        result = validate_routing_rows(rows, configs)

        self.assertEqual(result.summary["routing_row_count"], 2)
        self.assertEqual(result.summary["dropped_assignments"], 0)
        changed = [dict(row) for row in rows]
        changed[0]["expert_token_counts_json"] = "[1,2,3,3]"
        with self.assertRaisesRegex(Stage1ReleaseQAError, "sum mismatch"):
            validate_routing_rows(changed, configs)

    def test_duplicate_candidate_routing_requires_identical_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_rows = []
            for candidate in ("canonical", "recovery"):
                job = root / candidate / "jobs" / "prod_eval_c_e4_k001_d1"
                language_path = job / "languages" / "eng_Latn.json"
                routing = {
                    "applicable": True,
                    "schema": "multilingual_moe_routing_diagnostics/v1",
                    "layers": [
                        {
                            "layer_index": 0,
                            "expected_assignments": 10,
                            "kept_assignments": 10,
                            "dropped_assignments": 0,
                            "expert_token_counts": [1, 2, 3, 4],
                        }
                    ],
                }
                _write_json(language_path, {"moe_routing": routing})
                language_sha = hashlib.sha256(language_path.read_bytes()).hexdigest()
                manifest_path = job / "evaluation_manifest.json"
                _write_json(
                    manifest_path,
                    {
                        "languages": {
                            "eng_Latn": {
                                "path": str(language_path),
                                "sha256": language_sha,
                            }
                        }
                    },
                )
                candidate_rows.append(
                    {
                        "readout_id": "prod_eval_c_e4_k001_d1",
                        "ledger_valid": True,
                        "candidate_manifest_path": str(manifest_path),
                        "candidate_manifest_sha256": hashlib.sha256(
                            manifest_path.read_bytes()
                        ).hexdigest(),
                    }
                )

            result = validate_duplicate_candidate_routing(candidate_rows)
            self.assertEqual(result.summary["duplicate_readout_count"], 1)
            self.assertEqual(result.summary["candidate_count"], 2)
            self.assertEqual(result.summary["language_routing_payload_count"], 2)

            recovery_manifest = Path(candidate_rows[1]["candidate_manifest_path"])
            recovery = json.loads(recovery_manifest.read_text())
            recovery_language = Path(recovery["languages"]["eng_Latn"]["path"])
            changed = json.loads(recovery_language.read_text())
            changed["moe_routing"]["layers"][0]["expert_token_counts"] = [4, 3, 2, 1]
            _write_json(recovery_language, changed)
            recovery["languages"]["eng_Latn"]["sha256"] = hashlib.sha256(
                recovery_language.read_bytes()
            ).hexdigest()
            _write_json(recovery_manifest, recovery)
            candidate_rows[1]["candidate_manifest_sha256"] = hashlib.sha256(
                recovery_manifest.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(
                Stage1ReleaseQAError,
                "different routing evidence",
            ):
                validate_duplicate_candidate_routing(candidate_rows)


if __name__ == "__main__":
    unittest.main()
