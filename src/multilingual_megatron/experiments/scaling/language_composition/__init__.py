"""Language-composition selection, data preparation, training, and analysis."""

__all__ = ["PRIMARY_AXES", "SelectionError", "language_set_sha256"]


def __getattr__(name):
    # Analysis-only exports must not import the selection solver or its data stack.
    if name in __all__:
        from .selection import common
        return getattr(common, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
