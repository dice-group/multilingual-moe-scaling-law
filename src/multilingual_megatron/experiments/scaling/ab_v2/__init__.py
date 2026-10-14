"""Explicit, stage-owned implementation modules for scaling workflows.

``api`` is the only public workflow surface.  ``variant`` carries the A/B,
C, or D execution contract explicitly; implementation modules never mutate
each other's policy at import time.
"""

# Importing the base policy registers the default used by owner-level helpers
# in direct unit tests. Public calls still pass a concrete variant explicitly.
from .policy import AB_POLICY, AB_VARIANT

__all__ = ["AB_POLICY", "AB_VARIANT"]
