"""Single source of truth for KiCraft board-artifact locations and freshness.

WHY THIS MODULE EXISTS
----------------------
Board artifacts (``.kicad_pcb``) for a project live under
``<project_dir>/.experiments/subcircuits/<slug>/`` with a handful of canonical
names (placed vs routed, parent vs leaf) plus per-round and per-candidate
snapshots. Historically every consumer hard-coded those filenames and
re-implemented "find the current board" with subtly different precedence:

  * ``_find_placed_parent`` returned the *routed* board first even though its
    caller (``replay --no-route``) wanted the placement -- so a placement-only
    run silently promoted a STALE routed board from a previous run;
  * ``shutil.copy2`` preserved the source mtime on promote, so the staleness was
    invisible to the usual "did the file change?" check;
  * fallbacks used ``sorted(glob)[-1]`` (alphabetical) or ``iterdir`` first-match
    (arbitrary OS order) instead of "most recently produced".

That divergence cost a multi-hour debugging session (see
``docs/bug-replay-no-route-promotes-stale-routed-board.md``). This module
centralizes the fix so there is ONE place that knows the layout:

  * filename constants (:data:`PARENT_ROUTED`, :data:`PARENT_PLACED`, ...);
  * one INTENT-BASED resolver, :func:`resolve_parent_board` -- ``kind="placed"``
    NEVER falls back to a routed board, so placement-only iteration can't be
    fooled;
  * a run-scoped freshness predicate, :func:`produced_by_this_run`, so a board
    the current run did not produce is never silently promoted;
  * the ``<stem>.provenance.json`` record (:func:`write_promote_provenance` /
    :func:`read_provenance`) and the ``kicraft artifacts`` query both read from
    here, so agents have ONE truthful answer to "where is the current board and
    is it fresh?".

See ``docs/ARTIFACTS.md`` for the full artifact map and the agent-facing
contract. INVARIANT: provenance lives only in JSON sidecars -- it is NEVER
written into a ``.kicad_pcb`` (geometry goldens stay byte-stable).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Literal

# --- canonical artifact filenames (the ONLY place these literals are defined) --
PARENT_ROUTED = "parent_routed.kicad_pcb"
PARENT_PLACED = "parent_placed.kicad_pcb"
LEAF_ROUTED = "leaf_routed.kicad_pcb"
LEAF_PLACED = "leaf_placed.kicad_pcb"
LEAF_ILLEGAL = "leaf_illegal_pre_stamp.kicad_pcb"
METADATA_JSON = "metadata.json"
SEARCH_DIR = "_search"

# Richest-first leaf fallback tiers, mirroring the historical _find_best_leaf_board
# precedence: a routed leaf, then a routed round snapshot, then a placed leaf, then
# a placed round snapshot, then even a legality-REJECTED placement (preview only).
_LEAF_TIERS: tuple[str, ...] = (
    f"*/{LEAF_ROUTED}",
    f"*/round_*_{LEAF_ROUTED}",
    f"*/{LEAF_PLACED}",
    f"*/round_*_{LEAF_PLACED}",
    f"*/{LEAF_ILLEGAL}",
)

# Provenance / run-identity env vars. KICRAFT_RUN_ID matches the convention in
# kicraft/parts_library/query_log.py (set by the web driver), so a board's
# provenance correlates with the part-lookup logs from the same design run.
ENV_RUN_ID = "KICRAFT_RUN_ID"
ENV_RUN_STARTED_AT = "KICRAFT_RUN_STARTED_AT"

ParentKind = Literal["routed", "placed"]


# --- directory layout ---------------------------------------------------------

def artifact_root(project_dir: Path | str) -> Path:
    """``<project_dir>/.experiments/subcircuits`` -- the subcircuit artifact root.

    Mirrors ``kicraft.autoplacer.brain.subcircuit_artifacts.artifact_root_dir``;
    inlined to keep this module import-light (scripts and the CLI query import it).
    """
    return Path(project_dir) / ".experiments" / "subcircuits"


def _is_parent_dir(d: Path) -> bool:
    return d.is_dir() and (
        (d / PARENT_ROUTED).is_file() or (d / PARENT_PLACED).is_file()
    )


def _dir_recency(d: Path) -> float:
    """Most-recent-write time of a parent artifact dir: the max mtime across its
    parent boards AND its metadata.json. Using the BOARD mtimes (not just
    metadata.json) is what makes this correct for ``--no-route`` runs, where the
    placement board is freshly re-saved but metadata.json is NOT rewritten (it is
    only persisted on a routed compose)."""
    best = 0.0
    for name in (PARENT_ROUTED, PARENT_PLACED, METADATA_JSON):
        p = d / name
        try:
            best = max(best, p.stat().st_mtime)
        except OSError:
            continue
    return best


def latest_parent_artifact_dir(project_dir: Path | str) -> Path | None:
    """The parent subcircuit dir most recently written by the layout engine.

    Selects by :func:`_dir_recency` (max board/metadata mtime), which is
    deterministic and correct even when stale parent dirs from earlier
    hierarchies accumulate -- unlike the historical ``iterdir`` first-match
    (arbitrary OS order) in ``solve_hierarchy._find_parent_artifact``.
    """
    root = artifact_root(project_dir)
    if not root.is_dir():
        return None
    candidates = [d for d in root.iterdir() if _is_parent_dir(d)]
    if not candidates:
        return None
    return max(candidates, key=_dir_recency)


# --- intent-based board resolution --------------------------------------------

def resolve_parent_board(
    project_dir: Path | str, *, kind: ParentKind
) -> Path | None:
    """Resolve the parent board for an explicit INTENT.

    ``kind="routed"`` -> ``parent_routed.kicad_pcb`` only (else ``None``).
    ``kind="placed"`` -> ``parent_placed.kicad_pcb`` only; it NEVER
        falls back to the routed board. This is the fix for the
        ``replay --no-route`` stale-board bug: a placement-only run asks for
        ``kind="placed"`` and therefore can never receive a routed board left
        over from a previous run.
    """
    name = PARENT_ROUTED if kind == "routed" else PARENT_PLACED
    art = latest_parent_artifact_dir(project_dir)
    if art is not None:
        board = art / name
        if board.is_file():
            return board
    # Fallback: scan for the file anywhere under the project, newest by mtime
    # (NOT sorted()[-1], which is alphabetical and only correct by accident).
    hits = [p for p in Path(project_dir).glob(f"**/{name}") if p.is_file()]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def resolve_best_leaf_board(project_dir: Path | str) -> Path | None:
    """Richest single-leaf board, as a LAST-RESORT preview when the parent
    compose produced no parent board at all (rc6). Routed leaf > routed round
    snapshot > placed leaf > placed round snapshot > legality-rejected placement;
    within a tier, most recently written. Mirrors the historical
    ``_find_best_leaf_board`` (already mtime-correct)."""
    root = artifact_root(project_dir)
    if not root.is_dir():
        return None
    for pattern in _LEAF_TIERS:
        hits = [p for p in root.glob(pattern) if p.is_file()]
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)
    return None


# --- run identity & freshness -------------------------------------------------

def ensure_run_context() -> tuple[str, float]:
    """Idempotently establish ``(run_id, run_started_at)`` for THIS process and
    export them so the layout/compose/route subprocesses inherit the same
    identity (they all do ``os.environ.copy()`` or pass no ``env=``).

    Honors a ``KICRAFT_RUN_ID`` the web driver already injected, unifying board
    provenance with the parts-query run correlation. Call once at the entry of
    each command that promotes a board (build / replay / manual-route) BEFORE the
    layout subprocess is spawned.
    """
    rid = os.environ.get(ENV_RUN_ID)
    if not rid:
        rid = uuid.uuid4().hex[:12]
        os.environ[ENV_RUN_ID] = rid
    started = os.environ.get(ENV_RUN_STARTED_AT)
    started_f: float | None = None
    if started:
        try:
            started_f = float(started)
        except ValueError:
            started_f = None
    if started_f is None:
        started_f = time.time()
        os.environ[ENV_RUN_STARTED_AT] = repr(started_f)
    return rid, started_f


def current_run_id() -> str | None:
    return os.environ.get(ENV_RUN_ID) or None


def current_run_started_at() -> float | None:
    raw = os.environ.get(ENV_RUN_STARTED_AT)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def board_run_id(board: Path) -> str | None:
    """The ``run_id`` that produced ``board``, read from the sibling
    ``metadata.json`` in the board's artifact dir, or ``None`` if unrecorded.

    Note: ``metadata.json`` is only (re)written on a ROUTED compose, so a
    freshly *placed* board (``--no-route``) usually has a stale or absent
    ``run_id`` here. That is why :func:`produced_by_this_run` only trusts a
    POSITIVE match and otherwise falls back to mtime.
    """
    meta = board.parent / METADATA_JSON
    if not meta.is_file():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rid = data.get("run_id") if isinstance(data, dict) else None
    return rid if isinstance(rid, str) and rid else None


def produced_by_this_run(
    board: Path, *, run_id: str | None, run_started_at: float | None
) -> bool:
    """Did the current run produce ``board``?

    Decision order, chosen so a fresh board is never wrongly rejected:

    1. POSITIVE run_id match -> ``True`` (authoritative; immune to clock skew /
       mtime-preserving copies). We trust a match but never *reject* on a
       mismatch, because ``metadata.json`` (the run_id source) is not rewritten
       on a stamp-only ``--no-route`` run while the board itself IS re-saved.
    2. Otherwise fall back to mtime: ``board.mtime >= run_started_at``. Reliable
       because every board write (``board.Save()``, KiCad Routing Tools save) refreshes
       the mtime. ``>=`` (not ``>``) tolerates a same-second boundary.

    With no run context at all (run_started_at is None and no run_id), returns
    ``True`` -- callers without a run context (ad-hoc tooling) get the legacy
    permissive behavior rather than a spurious failure.
    """
    if run_id is not None and board_run_id(board) == run_id:
        return True
    if run_started_at is None:
        return run_id is None  # no run context to judge against -> permit
    try:
        return board.stat().st_mtime >= run_started_at
    except OSError:
        return False


# --- promote provenance sidecar ------------------------------------------------

def provenance_path(pcb: Path) -> Path:
    """``<stem>.provenance.json`` next to the promoted ``<stem>.kicad_pcb``."""
    return pcb.with_suffix(".provenance.json")


def pre_promote_seed_path(project_dir: Path | str) -> Path:
    """``.experiments/pre_promote_seed.kicad_pcb`` -- the full-component seed
    board snapshotted just before a promote overwrites ``<stem>.kicad_pcb``.

    An rc6 build deliberately promotes the best PARTIAL board as the project
    preview (no-fallback-previews), which destroys the only board ``replay``
    can re-solve leaves from. The build tail writes this snapshot at promote
    time; ``replay`` restores it when provenance says the board on disk is a
    partial (KC-9G4YPT GAP 2)."""
    return Path(project_dir) / ".experiments" / "pre_promote_seed.kicad_pcb"


def file_md5(path: Path) -> str | None:
    """Content fingerprint of a board, for provenance + the ``kicraft artifacts``
    query. md5 here is an identity fingerprint, not security-sensitive."""
    try:
        h = hashlib.md5()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def write_promote_provenance(
    pcb: Path,
    *,
    run_id: str | None,
    run_started_at: float | None,
    source_board: Path,
    source_kind: str,
    fresh: bool,
) -> Path:
    """Record what THIS run promoted to ``<stem>.kicad_pcb``. The authoritative,
    agent-facing answer to "which run produced the board on disk, and from what".
    Written in the parent process at promote time."""
    payload = {
        "schema_version": "promote-provenance-v1",
        "run_id": run_id,
        "run_started_at": run_started_at,
        "promoted_at": time.time(),
        "promoted_pcb": str(pcb),
        "md5": file_md5(pcb),
        "source_board": str(source_board),
        "source_kind": source_kind,
        "source_run_id": board_run_id(source_board),
        "source_mtime": _safe_mtime(source_board),
        "fresh": bool(fresh),
    }
    out = provenance_path(pcb)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out)  # atomic
    return out


def read_provenance(pcb: Path) -> dict | None:
    p = provenance_path(pcb)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None
