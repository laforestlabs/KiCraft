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
import re
from pathlib import Path

from kicraft.fsutil import atomic_write_text

from .stage_driver import DESIGN_STAGES, drive_chain
from .storage import _state_path

# The deterministic build sub-phases, in pipeline order after DESIGN_STAGES.
# Their status is always derived from artifacts (sheets / board / fab zip), never
# persisted: artifacts cannot go stale against themselves.
BUILD_PHASES = ("synthesize", "place_route", "electrical_review", "fab")


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
    # Electrical review: the post-wiring review writes its durable outcome to
    # stage_status (like the design stages) and its findings to the top-level
    # review_findings slot (artifacts.review_findings on legacy projects). A
    # blocker the re-drive did not clear reads as a yellow 'warning' (the run
    # proceeded; the gap is recorded), never a red failure. No stage_status
    # entry (review skipped / pre-R3 project) stays 'pending'.
    er = ss.get("electrical_review")
    findings = (state.get("review_findings")
                or (state.get("artifacts") or {}).get("review_findings") or [])
    has_blocker = any(isinstance(f, dict) and f.get("severity") == "blocker"
                      for f in findings)
    if isinstance(er, dict) and er.get("ok") is True:
        out["electrical_review"] = "warning" if has_blocker else "done"
    elif isinstance(er, dict) and er.get("ok") is False:
        out["electrical_review"] = "failed"
    elif findings:
        # Legacy build-tail review: persisted findings but no stage_status.
        out["electrical_review"] = "warning" if has_blocker else "done"
    elif design_complete and zip_ok and not failed:
        # Legacy successful build with no recorded review outcome: the
        # (build-tail) review gate passed or was disabled — either way it did
        # not block, so a finished project's review tab must not sit gray.
        out["electrical_review"] = "done"
    else:
        out["electrical_review"] = "pending"
    return out


def downstream_stages(stage: str) -> list[str]:
    """The stages after `stage` (what editing `stage` invalidates and must re-run)."""
    stages = list(DESIGN_STAGES)
    if stage not in stages:
        return []
    return stages[stages.index(stage) + 1:]


def read_state(ws) -> dict:
    """Best-effort load of a committed state.json (or {} if absent). The path is
    always ``<ws>/.kicraft/state.json`` via ``_state_path`` -- one layout, no
    fallback (Phase 4a; see CLAUDE.md "Storage model")."""
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


def _read_state_for_update(ws) -> dict | None:
    """read_state for read-modify-write callers: returns None (refuse) when
    state.json exists but cannot be parsed, instead of the {} read_state
    hands to render paths -- writing that {} back would wipe every committed
    slot with no error surfaced anywhere."""
    p = _state_path(Path(ws))
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def record_answers(ws, stage: str, answers: list[dict]) -> None:
    """Stamp the user's answers onto the stage's open_questions in state.json (for
    the record; the answers are also injected into the re-run prompt)."""
    state_path = Path(ws) / ".kicraft" / "state.json"
    sj = _read_state_for_update(ws)
    if sj is None:  # unreadable state: skip the stamp, never write {} over it
        return
    by_text = {a.get("text"): a.get("answer") for a in (answers or [])}
    for q in sj.get("open_questions") or []:
        if q.get("stage") == stage and q.get("text") in by_text:
            q["answer"] = by_text[q["text"]]
    atomic_write_text(state_path, json.dumps(sj, indent=2) + "\n")


def null_downstream(ws, stage: str) -> list[str]:
    """Null the slots downstream of `stage` (and drop their open_questions) so an
    edit cannot leave stale data behind; the caller then re-drives them. wiring's
    data lives in the bom slot, so clearing it empties bom.connections /
    no_connect_pins. Returns the stages cleared."""
    state_path = Path(ws) / ".kicraft" / "state.json"
    sj = _read_state_for_update(ws)
    if sj is None:
        # Unreadable committed state: fail LOUD rather than write {} back
        # (which would wipe every slot) or silently skip the clear (which
        # would leave stale downstream data behind an edit).
        raise RuntimeError(f"state.json unreadable in {ws}; cannot edit stages")
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
    atomic_write_text(state_path, json.dumps(sj, indent=2) + "\n")
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


# --------------------------------------------------------------------------- #
# BOM self-repair (shared by the web app and the self-eval driver)
# --------------------------------------------------------------------------- #
BOM_RECONCILE_TARGET = "bom"
# Total re-drive budget per project. Real deficit CHAINS exist -- a reconcile
# pass adds parts and wiring then finds the next GENUINE deficit (07-10 batch
# runs 13/22: nRF52840 DCCH cap, DRV8833 charge-pump cap; 07-13 batch run_10:
# an RP2040 VREG_VOUT cap) -- so a single-shot guard made every chain >= 2
# unwinnable by construction (fix-plan N3). Three links covers every chain
# observed; a stuck loop is cut earlier by the no-change check below.
BOM_RECONCILE_MAX_PASSES = 3


def bom_reconcile_instruction(questions) -> str:
    """Turn wiring's ``reconcile_target="bom"`` deficit note(s) into a BOM-stage
    instruction that adds the missing parts. Each question's text is already a
    precise "add N of X for pins Y" statement (per the wiring spec)."""
    lines = [str(q.get("text", "")).strip()
             for q in questions if str(q.get("text", "")).strip()]
    body = "\n- ".join(lines)
    return (
        "The wiring stage could not finish because the BOM is missing supporting "
        "parts its ICs require. Add the parts described below: give each a fresh, "
        "unique ref, the correct value and footprint, the same sheet as the IC it "
        "serves, and list it in that IC's ic_groups entry. Then re-emit the FULL "
        "BOM. Do NOT ask the user and do NOT drop any part already present -- just "
        "provision what's missing:\n- " + body
    )


def bom_reconcile_deficits(res: dict) -> list[dict]:
    """The ``reconcile_target="bom"`` deficit questions from a wiring park, or []."""
    if res.get("status") != "awaiting_input" or res.get("last_stage") != "wiring":
        return []
    return [
        q for q in (res.get("questions") or [])
        if q.get("reconcile_target") == BOM_RECONCILE_TARGET
    ]


# A wiring deficit note names the passive it needs in a stereotyped form:
# "requires a 1uF capacitor to GND", "Add two 10k resistors (0402/0603)",
# "needs a 0.1uF capacitor between BOOT (pin1) and PH (pin8)". The value +
# kind pair is enough to provision the PART deterministically -- wiring only
# parks because the part doesn't exist; connecting it is wiring's own job on
# the re-drive (self-eval 2026-07-17 T4: 3 of 34 briefs died with the model
# never applying exactly this add across 3 LLM passes).
_PASSIVE_ASK_RE = re.compile(
    r"(?:\b(a|an|one|two|three|four|\d+)\s+)?"
    r"(\d+(?:\.\d+)?\s?(?:[pnumµ]F|[kM](?:Ω|ohm)?\b|Ω|ohm\b|R\b))"
    r"[^.;,]*?\b(capacitor|resistor|inductor)s?\b",
    re.IGNORECASE,
)
_QTY_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4}
_SHEET_RE = re.compile(r"on the ([A-Z][A-Z0-9 _/-]*?) sheet\b")
_KIND_PREFIX = {"capacitor": "C", "resistor": "R", "inductor": "L"}
_KIND_SYMBOL = {"capacitor": "Device:C", "resistor": "Device:R",
                "inductor": "Device:L"}
_KIND_FOOTPRINT = {
    "capacitor": "Capacitor_SMD:C_0603_1608Metric",
    "resistor": "Resistor_SMD:R_0603_1608Metric",
    "inductor": "Inductor_SMD:L_0603_1608Metric",
}


def _norm_value(v: str) -> str:
    """Loose value identity: '0.1µF' == '0.1uF', '10kΩ' == '10k'."""
    s = (v or "").strip().lower().replace("µ", "u").replace("Ω", "")
    s = s.replace("ohm", "").replace(" ", "")
    return s


def parse_passive_deficits(texts: list[str]) -> list[dict]:
    """Extract fully-specified passive asks from deficit prose. Returns
    ``[{kind, value, qty, sheet}]``; anything the regex can't read with
    confidence is simply absent (the caller falls back to the LLM pass)."""
    asks: list[dict] = []
    for text in texts:
        t = str(text or "")
        # "on the X sheet" pins the sheet; "on the X and Y sheets" is
        # ambiguous per-part -- leave None (donor/default sheet applies).
        m_sheet = _SHEET_RE.search(t)
        sheet = (
            m_sheet.group(1).strip()
            if m_sheet and " sheets" not in t
            else None
        )
        for m in _PASSIVE_ASK_RE.finditer(t):
            qty_tok = (m.group(1) or "a").lower()
            qty = _QTY_WORDS.get(qty_tok)
            if qty is None:
                try:
                    qty = max(1, min(8, int(qty_tok)))
                except ValueError:
                    qty = 1
            asks.append({
                "kind": m.group(3).lower(),
                "value": m.group(2).strip(),
                "qty": qty,
                "sheet": sheet,
            })
    return asks


def _next_ref(parts: list[dict], prefix: str) -> str:
    used = set()
    for p in parts:
        r = str(p.get("ref") or "")
        if r.startswith(prefix) and r[len(prefix):].isdigit():
            used.add(int(r[len(prefix):]))
    n = 1
    while n in used:
        n += 1
    return f"{prefix}{n}"


def _catalog_passive(kind: str, value: str) -> dict | None:
    """Offline-catalog pick for a jellybean passive: in-stock, single-element,
    value-matched, Basic-preferred. None when the catalog can't answer."""
    try:
        from kicraft.parts_library import jlcparts
        if not jlcparts.available():
            return None
        fp = _KIND_FOOTPRINT[kind]
        kw = jlcparts.bom_keyword(value, fp)
        if not kw:
            return None
        cands = jlcparts.search(kw) or []
        if not cands:
            relaxed = jlcparts.relax_keyword(kw)
            cands = jlcparts.search(relaxed) if relaxed else []
        ok = [
            c for c in cands
            if (c.get("stock") or 0) > 0
            and not jlcparts.is_multi_element_array(c)
            and jlcparts.chip_value_matches(value, c)
        ]
        ok.sort(key=lambda c: (c.get("type") != "Basic", -(c.get("stock") or 0)))
        return ok[0] if ok else None
    except Exception:
        return None


def apply_deterministic_bom_adds(ws, deficits: list[dict]) -> list[str]:
    """Provision parseable passive deficits directly into ``state.bom.parts``.

    For each ask, clone a same-value donor part already in the BOM (same real
    LCSC sourcing, proven orderable) or fall back to an offline-catalog pick.
    Returns the added refs ([] = nothing applied; caller uses the LLM pass).
    Parts are ADDED only -- nothing existing is touched -- and wiring is
    re-driven by the caller to connect them (ic_groups membership follows from
    wiring's own commit, as with any model-added part)."""
    asks = parse_passive_deficits(
        [str(q.get("text", "")) for q in deficits]
    )
    if not asks:
        return []
    try:
        state_path = _state_path(Path(ws))
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    bom = state.get("bom") or {}
    parts = bom.get("parts")
    if not isinstance(parts, list) or not parts:
        return []

    # Ungrouped same-value parts = an earlier deterministic pass already
    # provisioned this ask and wiring STILL parks on it -- stop re-adding
    # (that's a stuck loop for the LLM path to report, not a parts shortfall).
    grouped: set[str] = set()
    for members in (bom.get("ic_groups") or {}).values():
        if isinstance(members, list):
            grouped.update(str(r) for r in members)

    # A sheet name scraped from prose is only usable if it actually exists --
    # the emitter hard-fails on unknown sheets, and a bad sheet baked into
    # state.bom.parts is unfixable by the wiring re-drive.
    known_sheets = {
        str(p.get("sheet") or "").strip().upper(): str(p.get("sheet"))
        for p in parts if p.get("sheet")
    }

    added: list[str] = []
    for ask in asks:
        ask_sheet = known_sheets.get(
            str(ask["sheet"] or "").strip().upper()
        )
        want = _norm_value(ask["value"])
        prefix = _KIND_PREFIX[ask["kind"]]
        same_value = [
            p for p in parts
            if str(p.get("ref", "")).startswith(prefix)
            and _norm_value(str(p.get("value", ""))) == want
        ]
        ungrouped = [p for p in same_value
                     if str(p.get("ref")) not in grouped]
        if len(ungrouped) >= ask["qty"]:
            continue
        donor = next(
            (p for p in same_value if p.get("sourcing_note") or p.get("mpn")),
            None,
        )
        pick = None if donor is not None else _catalog_passive(
            ask["kind"], ask["value"]
        )
        if donor is None and pick is None:
            continue
        for _ in range(ask["qty"] - len(ungrouped)):
            ref = _next_ref(parts, prefix)
            if donor is not None:
                entry = dict(donor)
                entry["ref"] = ref
                if ask_sheet:
                    entry["sheet"] = ask_sheet
            else:
                entry = {
                    "ref": ref,
                    "value": ask["value"].replace("µ", "u"),
                    "symbol": _KIND_SYMBOL[ask["kind"]],
                    "footprint": _KIND_FOOTPRINT[ask["kind"]],
                    "sheet": ask_sheet or str(parts[0].get("sheet") or ""),
                    "mpn": pick.get("model"),
                    "datasheet": None,
                    "sourcing_note": f"LCSC {pick.get('lcsc')}",
                    "side": None,
                    "source_leaf": None,
                }
            parts.append(entry)
            added.append(ref)
    if not added:
        return []
    try:
        # Atomic like every other state.json commit: three processes read this
        # file as their IPC contract, and a mid-write kill must never leave it
        # truncated.
        atomic_write_text(
            state_path, json.dumps(state, indent=2) + "\n"
        )
    except Exception:
        return []
    return added


def _bom_signature(ws) -> tuple[int, frozenset] | None:
    """Identity of the committed BOM (part count + ref set), for detecting a
    reconcile pass that changed nothing. ``None`` when the state is unreadable
    -- the caller then treats the pass as a change (fail open: a transient
    read problem must not cut a genuine deficit chain short)."""
    try:
        state = json.loads(_state_path(Path(ws)).read_text(encoding="utf-8"))
        parts = (state.get("bom") or {}).get("parts") or []
        refs = frozenset(str(p.get("ref")) for p in parts if isinstance(p, dict))
        return (len(parts), refs)
    except Exception:
        return None


def maybe_bom_reconcile(
    ws, brief, res, *, progress=None, run_id=None, core_defaults=None,
    client=None, reconcile_passes: int = 0,
) -> tuple[dict, int]:
    """Re-drive ``[bom, wiring]`` once when wiring parked on a BOM parts shortfall.

    Wiring tags a deficit park with ``reconcile_target="bom"``: it needs parts the
    BOM lacks, which wiring itself cannot add. Plain-answering that park loops
    forever (all 5 synthesis deaths in the 07-10 batch were this), so re-run
    bom+wiring with the concrete shortfall instead.

    Budgeted at ``BOM_RECONCILE_MAX_PASSES`` total passes (callers loop while the
    pass count advances -- deficit chains are real, see the constant's comment),
    and a pass that changes NOTHING in the committed BOM exhausts the budget
    immediately: that is a stuck loop, not a chain. Returns
    ``(new_or_original_res, total_passes)``. Shared by ``server/web.py`` and
    ``kicraft/eval/self_eval.py`` (WS6)."""
    if reconcile_passes >= BOM_RECONCILE_MAX_PASSES:
        return res, reconcile_passes
    deficits = bom_reconcile_deficits(res)
    if not deficits:
        return res, reconcile_passes
    # Deterministic first (self-eval 2026-07-17 T4): a fully-specified passive
    # ask needs no model round-trip to provision -- clone a same-value donor
    # already in the BOM (or an offline-catalog pick) and re-drive WIRING only.
    # Anything unparsed falls through to the LLM bom+wiring pass, which also
    # remains the stuck-loop reporter.
    added = apply_deterministic_bom_adds(ws, deficits)
    if added:
        if progress is not None:
            progress({"kind": "build_log",
                      "text": f"[bom-reconcile] deterministically provisioned "
                              f"{', '.join(added)} from the wiring deficit note "
                              f"(pass {reconcile_passes + 1}/"
                              f"{BOM_RECONCILE_MAX_PASSES}); re-driving wiring "
                              "to connect them (no model BOM pass)"})
        rr = run_session(
            ws, brief, ["wiring"],
            instruction=(
                "The missing supporting parts from your deficit note were "
                f"added to the BOM as {', '.join(added)} (same value/footprint "
                "sourcing as specified). Re-emit the FULL wiring, connecting "
                "each of them exactly as your note described. Do NOT ask the "
                "user and do NOT park on the same deficit again."
            ),
            progress=progress, run_id=run_id, core_defaults=core_defaults,
            client=client,
        )
        return rr, reconcile_passes + 1
    if progress is not None:
        progress({"kind": "build_log",
                  "text": f"[bom-reconcile] wiring flagged a BOM parts shortfall; "
                          f"re-driving bom+wiring (pass {reconcile_passes + 1}/"
                          f"{BOM_RECONCILE_MAX_PASSES}) to add the missing parts "
                          "(not asking the user)"})
    before = _bom_signature(ws)
    rr = run_session(
        ws, brief, ["bom", "wiring"],
        instruction=bom_reconcile_instruction(deficits),
        progress=progress, run_id=run_id, core_defaults=core_defaults, client=client,
    )
    passes = reconcile_passes + 1
    after = _bom_signature(ws)
    if before is not None and after is not None and after == before:
        if progress is not None:
            progress({"kind": "build_log",
                      "text": "[bom-reconcile] the pass changed nothing in the "
                              "committed BOM -- stopping reconcile (stuck loop, "
                              "not a deficit chain)"})
        passes = BOM_RECONCILE_MAX_PASSES
    return rr, passes
