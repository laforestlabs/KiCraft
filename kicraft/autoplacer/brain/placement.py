"""PlacementSolver and PlacementScorer -- backward-compatible re-export hub.

The implementation is split across focused modules:
- ``placement_utils``  -- shared geometry helpers
- ``placement_scorer`` -- PlacementScorer
- ``placement_solver`` -- PlacementSolver

Import from this module for backward compatibility; prefer direct imports
from the sub-modules in new code.
"""

from __future__ import annotations

# Re-export everything that callers currently import from placement
from .placement_scorer import PlacementScorer  # noqa: F401
from .placement_solver import PlacementSolver  # noqa: F401
from .placement_utils import (  # noqa: F401
    _bbox_overlap,
    _bbox_overlap_amount,
    _bbox_overlap_xy,
    _effective_bbox,
    _pad_half_extents,
    _swap_pad_positions,
    _update_pad_positions,
    compute_min_board_size,
)

__all__ = [
    "PlacementScorer",
    "PlacementSolver",
    "_bbox_overlap",
    "_bbox_overlap_amount",
    "_bbox_overlap_xy",
    "_effective_bbox",
    "_pad_half_extents",
    "_swap_pad_positions",
    "_update_pad_positions",
    "compute_min_board_size",
]
