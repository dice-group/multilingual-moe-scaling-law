from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from multilingual_megatron.pipeline.stages.final_loss.validation_metrics import (  # noqa: E402
    MissingRequiredLanguageError,
    UnigramReference,
    build_validation_report,
    compute_language_losses,
    compute_language_losses_from_array,
    verify_report_hash,
)


def _cohorts() -> dict[str, list[str]]:
    additions = [f"lang_{index:03d}" for index in range(1, 50)]
    return {
        "english": ["eng_Latn"],
        "k12": ["eng_Latn", *additions[:11]],
        "k50": ["eng_Latn", *additions],
    }


def _loss(language: str, *, nll: float = 2.0, tokens: list[int] | None = None):
    reference = UnigramReference.from_counts(
        language=language,
        vocab_size=5,
        counts={0: 3, 1: 1},
        special_token_ids=[4],
    )
    target_ids = tokens if tokens is not None else [0]
    return compute_language_losses(
        language=language,
        nll_sum=nll * len(target_ids),
        target_token_ids=target_ids,
        unigram=reference,
    )


class ValidationMetricsTest(unittest.TestCase):
    def test_unigram_adjustment_uses_pseudo_count_one_for_unseen_ids(self) -> None:
        reference = UnigramReference.from_counts(
            language="eng_Latn",
            vocab_size=5,
            counts={1: 1, 0: 3},
            special_token_ids=[4],
        )

        result = compute_language_losses(
            language="eng_Latn",
            nll_sum=6.0,
            target_token_ids=[0, 2, 2],
            unigram=reference,
        )

        # Non-special IDs 0..3 have effective counts [3, 1, 1, 1].
        expected_unigram_sum = math.log(3 / 6) + 2 * math.log(1 / 6)
        self.assertEqual(reference.smoothed_token_count, 6)
        self.assertEqual(result.token_count, 3)
        self.assertEqual(result.unseen_target_token_count, 2)
        self.assertAlmostEqual(result.raw_nll, 2.0)
        self.assertAlmostEqual(result.unigram_log_probability_sum, expected_unigram_sum)
        self.assertAlmostEqual(
            result.atlas_tao_adjusted_loss,
            2.0 + expected_unigram_sum / 3,
        )

        # Count-map insertion order and iterable type must not change hashes.
        same_reference = UnigramReference.from_counts(
            language="eng_Latn",
            vocab_size=5,
            counts={0: 3, 1: 1},
            special_token_ids=(4,),
        )
        same_result = compute_language_losses(
            language="eng_Latn",
            nll_sum=6,
            target_token_ids=(token for token in (0, 2, 2)),
            unigram=same_reference,
        )
        self.assertEqual(reference.counts_sha256, same_reference.counts_sha256)
        self.assertEqual(result.input_sha256, same_result.input_sha256)

    def test_k1_primary_excludes_required_zero_shot_languages(self) -> None:
        english = _loss("eng_Latn", nll=2.0, tokens=[0, 0])
        zero_shot = _loss("lang_001", nll=7.0, tokens=[0])

        report = build_validation_report(
            run_id="k1_run",
            observations=[zero_shot, english],
            trained_languages=["eng_Latn"],
            required_languages=["eng_Latn", "lang_001"],
            cohorts=_cohorts(),
        )

        self.assertEqual(report["coverage"]["zero_shot_languages"], ["lang_001"])
        roles = {row["language"]: row["role"] for row in report["languages"]}
        self.assertEqual(roles, {"eng_Latn": "trained", "lang_001": "zero_shot"})
        primary = report["aggregates"]["primary"]
        self.assertEqual(primary["languages"], ["eng_Latn"])
        self.assertEqual(primary["language_count"], 1)
        self.assertAlmostEqual(primary["value"], english.atlas_tao_adjusted_loss)
        secondary = report["aggregates"]["secondary"]
        self.assertEqual(secondary["name"], "trained_equal_language_raw_nll")
        self.assertEqual(secondary["languages"], ["eng_Latn"])
        self.assertEqual(secondary["language_count"], 1)
        self.assertAlmostEqual(secondary["value"], english.raw_nll)
        self.assertEqual(secondary["target_domain"], "non_special_validation_tokens")
        self.assertIn("exclude EOD as a scored target", report["metric_spec"]["eod_context_policy"])
        self.assertIn("score every real non-special target", report["metric_spec"]["final_partial_sequence_policy"])
        self.assertEqual(
            report["aggregates"]["fixed_common_cohorts"]["k12"]["status"],
            "not_applicable",
        )
        self.assertTrue(verify_report_hash(report))

        report["aggregates"]["primary"]["value"] = 123.0
        self.assertFalse(verify_report_hash(report))

    def test_missing_required_language_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            MissingRequiredLanguageError,
            "missing required validation languages: lang_001",
        ):
            build_validation_report(
                run_id="incomplete",
                observations=[_loss("eng_Latn")],
                trained_languages=["eng_Latn", "lang_001"],
                required_languages=["eng_Latn", "lang_001"],
                cohorts=_cohorts(),
            )

    def test_complete_k50_emits_macro_micro_and_all_fixed_cohorts(self) -> None:
        cohorts = _cohorts()
        observations = [
            _loss(language, nll=float(index + 1), tokens=[0] * (index % 3 + 1))
            for index, language in enumerate(cohorts["k50"])
        ]
        report = build_validation_report(
            run_id="k50_run",
            observations=reversed(observations),
            trained_languages=reversed(cohorts["k50"]),
            required_languages=cohorts["k50"],
            cohorts=cohorts,
        )

        trained = report["aggregates"]["trained_languages"]
        self.assertEqual(trained["language_count"], 50)
        self.assertAlmostEqual(
            report["aggregates"]["secondary"]["value"],
            trained["equal_language_raw_nll"],
        )
        self.assertNotEqual(
            trained["equal_language_raw_nll"],
            trained["token_micro_raw_nll"],
        )
        fixed = report["aggregates"]["fixed_common_cohorts"]
        self.assertEqual(fixed["english"]["status"], "complete")
        self.assertEqual(fixed["k12"]["language_count"], 12)
        self.assertEqual(fixed["k50"]["language_count"], 50)
        self.assertEqual(report["aggregates"]["zero_shot_languages"]["status"], "empty")
        self.assertTrue(verify_report_hash(report))

    def test_special_validation_target_is_rejected(self) -> None:
        reference = UnigramReference.from_counts(
            language="eng_Latn",
            vocab_size=3,
            counts={0: 2},
            special_token_ids=[2],
        )
        with self.assertRaisesRegex(ValueError, "mask it before summing NLL"):
            compute_language_losses(
                language="eng_Latn",
                nll_sum=1.0,
                target_token_ids=[2],
                unigram=reference,
            )

    def test_vectorized_loss_reduction_preserves_exact_statistics_and_hashes(self) -> None:
        reference = UnigramReference.from_counts(
            language="eng_Latn",
            vocab_size=8,
            counts={0: 5, 1: 2, 3: 1},
            special_token_ids=[6, 7],
        )
        targets = [0, 3, 1, 0, 2, 2, 5, 1, 0]
        generic = compute_language_losses(
            language="eng_Latn",
            nll_sum=27.5,
            target_token_ids=targets,
            unigram=reference,
        )
        vectorized = compute_language_losses_from_array(
            language="eng_Latn",
            nll_sum=27.5,
            target_token_ids=np.asarray(targets, dtype=np.int64),
            unigram=reference,
        )
        self.assertEqual(generic.to_dict(), vectorized.to_dict())


if __name__ == "__main__":
    unittest.main()
