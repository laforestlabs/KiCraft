"""Resumable design session over the (already re-entrant) stage driver.

The web worker used to be fire-and-forget: one tempdir, run every stage, persist,
exit. This module wraps `drive_chain` so a design can be:

- resumed from a partial state.json (run the stages whose slots are still empty),
- re-driven from an edited stage (run that stage's downstream),
- parked on a blocking clarifying question and continued later with the answer.

It owns only the LLM-driven schematic stages (DESIGN_STAGES). The deterministic
build (synth/place/route/fab) stays in the caller, which runs it once a session
reports status "ok".
"""
from __future__ import annotations

import json
from pathlib import Path

from .stage_driver import DESIGN_STAGES, drive_chain
from .storage import _state_path

# The deterministic build sub-phases, in pipeline order after DESIGN_STAGES.
# Their status is always derived from artifacts (sheets / board / fab zip), never
# persisted: artifacts cannot go stale against themselves.
BUILD_PHASES = ("synthesize", "place_route", "fab")


def _stage_done(stage: str, state: dict) -> bool:
    """Whether `stage`'s contribution to the state is already present. wiring is
    not a standalone slot: it populates bom.connections / bom.no_connect_pins."""
    if stage == "wiring":
        bom = state.get("bom") or {}
        return bool(bom.get("connections")) or bool(bom.get("no_connect_pins"))
    return state.get(stage) is not None


def remaining_stages(state: dict) -> list[str]:
    """DESIGN_STAGES from the first incomplete stage onward (a hole re-runs the
    tail). Empty when every schematic stage is satisfied."""
    stages = list(DESIGN_STAGES)
    for i, stage in enumerate(stages):
        if not _stage_done(stage, state):
            return stages[i:]
    return []


def derive_stage_statuses(state: dict, *, project_status: str | None = None,
                          sheets_exist: bool = False,
                          synth_checks_failed: bool = False,
                          pcb_ready: bool = False,
                          zip_ok: bool = False) -> dict[str, str]:
    """Map every pipeline phase to its durable status, for restoring the GUI's
    stage tabs on a reopened project: 'pending' | 'parked' | 'done' | 'failed'.

    Design stages read the persisted stage_status block (written by the stage
    driver at commit/fail time), falling back to slot presence for legacy
    projects that predate it. An unanswered open question marks its stage
    'parked'. The build phases are derived from artifact signals the caller
    reads from the workspace, gated on the design being complete so leftover
    artifacts from a build that predates an edit don't count. 'active' is never
    produced here: only a live event stream knows a stage is running.
    """
    ss = state.get("stage_status") or {}
    out: dict[str, str] = {}
    for s in DESIGN_STAGES:
        e = ss.get(s)
        if isinstance(e, dict) and e.get("ok") is True:
            out[s] = "done"
        elif isinstance(e, dict) and e.get("ok") is False:
            out[s] = "failed"
        elif _stage_done(s, state):
            out[s] = "done"  # legacy project predating stage_status
        else:
            out[s] = "pending"
    for q in state.get("open_questions") or []:
        s = q.get("stage")
        if not q.get("answer") and out.get(s) not in (None, "done"):
            out[s] = "parked"

    design_complete = all(out[s] == "done" for s in DESIGN_STAGES)
    failed = project_status == "failed"
    synth_ok = design_complete and sheets_exist and not synth_checks_failed
    out["synthesize"] = ("done" if synth_ok
                         else "failed" if design_complete and failed
                         else "pending")
    # A fab-acceptable board can still carry non-blocking warnings (e.g. a
    # minor, fraction-of-a-mm courtyard clip): the build succeeded and exported
    # the package + 3D model, but the gap is surfaced as a yellow 'warning'
    # rather than a green 'done'.
    has_warnings = bool((state.get("artifacts") or {}).get("build_warnings"))
    # A produced board means place/route succeeded; a failed build localizes to
    # the FAB gate, not here -- so "done"/"warning" (board exists) outrank
    # "failed" exactly as the original done-first precedence did.
    out["place_route"] = ("warning" if design_complete and pcb_ready and has_warnings
                          else "done" if design_complete and pcb_ready
                          else "failed" if synth_ok and failed
                          else "pending")
    # failed-with-a-board outranks zip_ok: after a failed (re)build the board
    # on disk is the failed candidate, so any surviving zip from an earlier
    # successful build is stale -- the tab must read failed, not done.
    out["fab"] = ("failed" if design_complete and pcb_ready and failed
                  else "warning" if design_complete and zip_ok and has_warnings
                  else "done" if design_complete and zip_ok
                  else "pending")
    return out


def downstream_stages(stage: str) -> list[str]:
    """The stages after `stage` (what editing `stage` invalidates and must re-run)."""
    stages = list(DESIGN_STAGES)
    if stage not in stages:
        return []
    return stages[stages.index(stage) + 1:]


def read_state(ws) -> dict:
    """Best-effort load of a committed state.json (or {} if absent). Resolves the
    workspace (``.kicraft``), durable (``kicraft``), or legacy (top-level) layout
    via ``_state_path``; the workspace writers below keep their explicit
    ``.kicraft`` path, which they always run against."""
    p = _state_path(Path(ws))
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def commit_slot(ws, stage: str, slot: dict, brief: str = "", project_stem=None):
    """Commit an edited slot to the workspace state.json via the deterministic CLI
    (which re-validates it). Returns (ok, out); out carries `errors` on rejection."""
    from .stage_driver import _commit, _stamp_stage_status
    state_path = Path(ws) / ".kicraft" / "state.json"
    ok, out = _commit(stage, dict(slot), state_path, brief, project_stem, Path(ws))
    if ok:  # a manual edit is a zero-cost commit; the stage is (re)done
        _stamp_stage_status(state_path, stage, True)
    return ok, out


def record_answers(ws, stage: str, answers: list[dict]) -> None:
    """Stamp the user's answers onto the stage's open_questions in state.json (for
    the record; the answers are also injected into the re-run prompt)."""
    state_path = Path(ws) / ".kicraft" / "state.json"
    sj = read_state(ws)
    by_text = {a.get("text"): a.get("answer") for a in (answers or [])}
    for q in sj.get("open_questions") or []:
        if q.get("stage") == stage and q.get("text") in by_text:
            q["answer"] = by_text[q["text"]]
    state_path.write_text(json.dumps(sj, indent=2) + "\n", encoding="utf-8")


def null_downstream(ws, stage: str) -> list[str]:
    """Null the slots downstream of `stage` (and drop their open_questions) so an
    edit cannot leave stale data behind; the caller then re-drives them. wiring's
    data lives in the bom slot, so clearing it empties bom.connections /
    no_connect_pins. Returns the stages cleared."""
    state_path = Path(ws) / ".kicraft" / "state.json"
    sj = read_state(ws)
    cleared = downstream_stages(stage)
    for s in cleared:
        if s == "wiring":
            bom = sj.get("bom")
            if isinstance(bom, dict):
                bom["connections"] = []
                bom["no_connect_pins"] = []
        else:
            sj[s] = None
        ss = sj.get("stage_status")
        if isinstance(ss, dict):  # the stage's recorded outcome is stale too
            ss.pop(s, None)
    sj["open_questions"] = [q for q in (sj.get("open_questions") or [])
                            if q.get("stage") not in cleared]
    state_path.write_text(json.dumps(sj, indent=2) + "\n", encoding="utf-8")
    return cleared


def run_session(ws, brief: str, stages, answers=None, instruction=None,
                client=None, progress=None, run_id=None, core_defaults=None) -> dict:
    """Drive `stages` over the workspace's state.json.

    Returns {status, results, guard, questions, last_stage} where status is:
    - "ok"             every stage committed,
    - "failed"         a stage could not commit within its retry budget,
    - "awaiting_input" a stage parked on a blocking clarifying question.

    `answers` / `instruction` apply to the first stage (the one being resumed or
    edited); downstream stages re-draft cleanly from the updated state.
    `core_defaults` is the core-components registry rows (admin-curated default
    parts) to surface in the architecture/bom prompts; fetched fresh per run,
    never persisted in state.json.
    """
    stages = list(stages)
    if not stages:
        return {"status": "ok", "results": [], "guard": None,
                "questions": None, "last_stage": None}
    results, guard, state_path = drive_chain(
        stages, brief, Path(ws), progress=progress, client=client,
        answers=answers, instruction=instruction, run_id=run_id,
        core_defaults=core_defaults)
    last = results[-1] if results else None
    if last and last.get("needs_input"):
        status = "awaiting_input"
    elif results and all(r.get("commit_ok") for r in results):
        status = "ok"
    else:
        status = "failed"
    return {"status": status, "results": results, "guard": guard,
            "state_path": state_path,
            "questions": (last.get("questions") if last else None),
            "last_stage": (last.get("stage") if last else None)}
