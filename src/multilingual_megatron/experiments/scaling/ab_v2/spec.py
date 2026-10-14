"""Compatibility entry point for the neutral A/B-v2 scientific contract."""

from multilingual_megatron.contracts.ab_v2 import (
    DEFAULT_SPEC,
    EXPECTED_K,
    LANGUAGE_ID_RE,
    SpecValidationError,
    language_set_sha256,
    main,
    validate_ab_v2_spec,
)

__all__ = [
    "DEFAULT_SPEC",
    "EXPECTED_K",
    "LANGUAGE_ID_RE",
    "SpecValidationError",
    "language_set_sha256",
    "validate_ab_v2_spec",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
