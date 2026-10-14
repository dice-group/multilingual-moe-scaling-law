"""Immutable contracts for FineWeb v2 bootstrap acquisition."""

from __future__ import annotations

import re
from dataclasses import dataclass

SCHEMA_VERSION = 1
TOOL_ID = "fineweb_ab_v2_bootstrap_v1"
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
OUTPUT_NAMES = ("train.jsonl", "validation_reserve.jsonl", "calibration.jsonl")


@dataclass(frozen=True, slots=True)
class BootstrapSettings:
    seed: int = 42
    # Candidate reserve only. The exact final validation target remains 0.20
    # of unique final-token inventory and is selected after tokenizer training.
    validation_fraction: float = 0.30
    target_unique_documents: int = 10_000
    target_utf8_bytes: int = 64 * 1024 * 1024
    calibration_documents: int = 1_000

    def validate(self) -> None:
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be strictly between zero and one")
        if int(self.target_unique_documents) <= 0:
            raise ValueError("target_unique_documents must be positive")
        if int(self.target_utf8_bytes) <= 0:
            raise ValueError("target_utf8_bytes must be positive")
        if int(self.calibration_documents) <= 0:
            raise ValueError("calibration_documents must be positive")


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    dataset: str
    revision: str
    subset: str
    is_english: bool

    @property
    def hf_root(self) -> str:
        return f"hf://datasets/{self.dataset}@{self.revision}"


@dataclass(frozen=True, slots=True)
class SourceAvailability:
    documents: int
    utf8_bytes: int
    inventory_path: str
    inventory_sha256: str
