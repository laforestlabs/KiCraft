"""Single-source freshness + lookup for every rendered PNG the GUI surfaces.

Before this module, five places owned different parts of "is this
PNG fresh for this leaf in this run?":

- ``pipeline_graph._load_render_floor``    -> ``.experiments/run_started_at``
- ``pipeline_graph._load_run_phase``       -> ``.experiments/run_phase``
- ``pipeline_graph._mtime_passes``         -> mtime predicate
- ``pins.is_pinned`` / ``pins.read_pins``  -> ``pins.json``
- ``layout_editor.render.RENDERER_VERSION``  -> sidecar version

and every consumer (the monitor, the manual layout, the inspect tool)
re-implemented its own fallback ladder over those signals. ``RenderIndex``
collapses the freshness predicates into one object; ``parent_render`` /
``leaf_render`` / ``round_renders`` then answer the "what should I show
the user?" questions in one call.

Lifetime: build a fresh ``RenderIndex`` at the top of each
``gather_pipeline_state`` tick. The four signals it gates on
(``run_started_at``, ``run_phase``, ``pins.json``, the per-PNG mtime)
are all on disk and cheap to re-read; we deliberately do NOT cache
across ticks so a run that starts mid-render is picked up immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RenderIndex:
    """Bundle of freshness signals + lookup helpers.

    Build via ``RenderIndex.load(experiments_dir)`` so the constructor
    stays trivial; freshness signals are read from disk inside ``load``.
    Field semantics:

    - ``experiments_dir``: project ``.experiments/`` root.
    - ``render_floor``: mtime threshold (epoch seconds). Files older
      than this are stale -- they belong to a prior run. ``None`` when
      no run has started in this project.
    - ``run_phase``: ``"leaves_only" | "parents_only" | "full" | None``.
      Controls the pin bypass: in ``parents_only`` the leaves are not
      re-solved so pinned-leaf renders stay valid even when their
      mtimes precede the run start.
    - ``pinned_rounds``: ``{leaf_key: round_index}`` from ``pins.json``.
    """

    experiments_dir: Path
    render_floor: float | None
    run_phase: str | None
    pinned_rounds: dict[str, int]

    @classmethod
    def load(cls, experiments_dir: Path) -> "RenderIndex":
        return cls(
            experiments_dir=experiments_dir,
            render_floor=_load_render_floor(experiments_dir),
            run_phase=_load_run_phase(experiments_dir),
            pinned_rounds=_load_pinned_rounds(experiments_dir),
        )

    # ------------------------------------------------------------------
    # Freshness predicate
    # ------------------------------------------------------------------

    def mtime_passes(self, path: Path, *, floor: float | None = None) -> bool:
        """True when ``path`` exists and its mtime >= ``floor`` (or no
        floor is in effect). Use ``floor=None`` to inherit ``render_floor``;
        pass ``floor=0.0`` to disable the gate entirely (pinned-leaf
        bypass in ``parents_only``)."""
        try:
            mt = path.stat().st_mtime
        except OSError:
            return False
        effective = self.render_floor if floor is None else floor
        return effective is None or mt >= effective

    def leaf_floor(self, leaf_key: str) -> float | None:
        """The mtime floor that applies to a specific leaf. Returns
        ``None`` (bypass) when the leaf is pinned AND the run is in
        ``parents_only`` phase -- the leaf is not being re-solved so
        its renders from a prior run stay valid. Returns ``render_floor``
        otherwise."""
        if self.run_phase == "parents_only" and leaf_key in self.pinned_rounds:
            return None
        return self.render_floor

    def is_pinned(self, leaf_key: str) -> bool:
        return leaf_key in self.pinned_rounds

    def pinned_round(self, leaf_key: str) -> int | None:
        return self.pinned_rounds.get(leaf_key)

    # ------------------------------------------------------------------
    # Render lookup
    # ------------------------------------------------------------------

    def leaf_render(
        self, leaf_dir: Path, leaf_key: str, *, round_index: int | None = None
    ) -> Path | None:
        """Best fresh render PNG for one leaf.

        With ``round_index`` set, returns that round's routed snapshot
        (or pre-route if routed is missing) -- the per-round scrubber's
        thumbnail. Without ``round_index``, returns the pinned round's
        snapshot when pinned, otherwise the canonical
        ``routed_front_all.png`` (or pre-route fallback). Returns
        ``None`` when no fresh candidate exists.
        """
        renders = leaf_dir / "renders"
        floor = self.leaf_floor(leaf_key)
        if round_index is not None:
            for name in (
                f"round_{round_index:04d}_routed_front_all.png",
                f"round_{round_index:04d}_pre_route_front_all.png",
            ):
                p = renders / name
                if self.mtime_passes(p, floor=floor):
                    return p
            return None
        pinned = self.pinned_round(leaf_key)
        if pinned is not None:
            for name in (
                f"round_{pinned:04d}_routed_front_all.png",
                f"round_{pinned:04d}_pre_route_front_all.png",
            ):
                p = renders / name
                if self.mtime_passes(p, floor=floor):
                    return p
        for name in (
            "routed_front_all.png",
            "pre_route_front_all.png",
            "routed_copper_both.png",
            "pre_route_copper_both.png",
        ):
            p = renders / name
            if self.mtime_passes(p, floor=floor):
                return p
        return None

    def round_renders(
        self, leaf_dir: Path, leaf_key: str, round_index: int
    ) -> tuple[Path | None, Path | None]:
        """Routed and pre-route renders for one specific round of one
        leaf. Used by the per-round scrubber in the leaf detail panel."""
        renders = leaf_dir / "renders"
        floor = self.leaf_floor(leaf_key)
        routed = renders / f"round_{round_index:04d}_routed_front_all.png"
        pre = renders / f"round_{round_index:04d}_pre_route_front_all.png"
        return (
            routed if self.mtime_passes(routed, floor=floor) else None,
            pre if self.mtime_passes(pre, floor=floor) else None,
        )

    def parent_render(
        self,
        *,
        round_index: int | None = None,
        prefer_routed: bool = True,
        preview_paths: dict[str, Any] | None = None,
    ) -> Path | None:
        """Best fresh parent render PNG for the monitor's root node.

        Probes, in priority order:
        1. The per-round dir under ``hierarchical_autoexperiment/round_NNNN/``
           when ``round_index`` is provided.
        2. The live ``preview_paths`` recorded in ``run_status``
           (``parent_routed_preview`` / ``parent_stamped_preview``).
        3. ``hierarchical_pipeline/`` canonical fallback.
        4. Newest matching PNG across ``subcircuits/*/renders/`` (last
           resort when the metadata pointers are missing).

        Within each layer, ``prefer_routed`` controls whether to surface
        the post-route render or the pre-route stamp first -- routed when
        the round succeeded, stamped when it failed (so the user sees
        the geometry that was rejected, not a missing icon).
        """
        names_order = (
            ("parent_routed.png", "parent_stamped.png")
            if prefer_routed
            else ("parent_stamped.png", "parent_routed.png")
        )

        if round_index is not None and round_index > 0:
            round_dir = (
                self.experiments_dir
                / "hierarchical_autoexperiment"
                / f"round_{round_index:04d}"
            )
            for name in names_order:
                p = round_dir / name
                if self.mtime_passes(p):
                    return p

        if preview_paths:
            preferred_key = (
                "parent_routed_preview" if prefer_routed else "parent_stamped_preview"
            )
            fallback_key = (
                "parent_stamped_preview" if prefer_routed else "parent_routed_preview"
            )
            for key in (preferred_key, fallback_key):
                cand = preview_paths.get(key)
                if not cand:
                    continue
                p = Path(str(cand))
                if self.mtime_passes(p):
                    return p

        hp = self.experiments_dir / "hierarchical_pipeline"
        for name in names_order:
            p = hp / name
            if self.mtime_passes(p):
                return p

        # Last-resort scan when the per-stage metadata pointers are
        # missing entirely (e.g. acceptance-gate rejection that bailed
        # before writing run_status preview_paths).
        sub_root = self.experiments_dir / "subcircuits"
        if not sub_root.exists():
            return None
        best_mtime = -1.0
        best: Path | None = None
        for child in sub_root.iterdir():
            if not child.is_dir():
                continue
            for name in names_order:
                cand = child / "renders" / name
                if not self.mtime_passes(cand):
                    continue
                try:
                    mt = cand.stat().st_mtime
                except OSError:
                    continue
                if mt > best_mtime:
                    best_mtime, best = mt, cand
        return best


# Loaders kept module-private so callers go through RenderIndex.load.

def _load_render_floor(experiments_dir: Path) -> float | None:
    try:
        return float((experiments_dir / "run_started_at").read_text().strip())
    except (OSError, ValueError):
        return None


def _load_run_phase(experiments_dir: Path) -> str | None:
    try:
        return (experiments_dir / "run_phase").read_text().strip() or None
    except OSError:
        return None


def _load_pinned_rounds(experiments_dir: Path) -> dict[str, int]:
    try:
        from kicraft.autoplacer.brain import pins as _pins
        manifest = _pins.read_pins(experiments_dir)
    except Exception:
        return {}
    out: dict[str, int] = {}
    for leaf_key, record in (manifest.get("pinned_leaves") or {}).items():
        try:
            out[str(leaf_key)] = int(record.get("round"))
        except (TypeError, ValueError):
            continue
    return out
