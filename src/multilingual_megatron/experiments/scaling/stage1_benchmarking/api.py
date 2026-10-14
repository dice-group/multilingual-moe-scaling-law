"""Small experiment-layer facade for the Stage-1 benchmark stage."""

from multilingual_megatron.experiments.scaling.stage1_benchmarking.contracts import (
    BenchmarkProfile,
    METRICS_SCHEMA,
    _atomic_json,
    _load_json,
    _payload_sha256,
    _resolve,
    _sha256_file,
    _verify_payload_hash,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.replay import (
    aggregate_scores,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.requests import (
    freeze_requests,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.score_specs import (
    build_score_specs,
)

__all__ = [
    "BenchmarkProfile",
    "METRICS_SCHEMA",
    "_atomic_json",
    "_load_json",
    "_payload_sha256",
    "_resolve",
    "_sha256_file",
    "_verify_payload_hash",
    "aggregate_scores",
    "build_score_specs",
    "freeze_requests",
]
