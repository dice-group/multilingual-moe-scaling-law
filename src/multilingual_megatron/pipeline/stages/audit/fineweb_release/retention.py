"""Retention policy for auditable FineWeb-v2 scientific releases."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _retention_policy(
    release_root: Path, *, adoption: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    corpus = release_root / "corpus"
    required_paths = [
        str(corpus / "source_lock"),
        str(corpus / "bootstrap"),
        str(corpus / "tokenizer_sentence_acquisition_plan.json"),
        str(corpus / "tokenizer_bootstrap"),
        str(corpus / "tokenizer_sentences"),
        str(corpus / "tokenizer"),
        str(corpus / "token_calibration.json"),
        str(corpus / "sampling_plan_32b.json"),
        str(corpus / "production_bootstrap"),
        str(corpus / "release_32b"),
        str(corpus / "tokenization_work"),
        str(corpus / "tokenized"),
    ]
    reason = (
        "The tokenized manifests reference prefixes under tokenization_work, and the "
        "three bootstrap generations provide the immutable pilot, tokenizer-acquisition, "
        "and production lineage. Cleanup before result acceptance would break auditability."
    )
    if adoption is not None:
        required_paths.extend(
            [
                str(corpus / "k12_migration" / "adoption_manifest.json"),
                str(Path(str(adoption["predecessor_root"])).resolve()),
            ]
        )
        reason += (
            " The K12 successor also contains hash-bound provenance and verified immutable "
            "copies from the sealed predecessor, so that predecessor must be retained through "
            "acceptance."
        )
    return {
        "policy": "retain_until_all_A_B_v2_results_and_reproducibility_audits_are_accepted",
        "automatic_cleanup": False,
        "required_paths": required_paths,
        "reason": reason,
    }
