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

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


from kicraft.design.stage_state import (
    downstream_stages as _downstream_stages,
    invalidate_downstream,
)
from kicraft.fsutil import atomic_write_text

from .stage_pipeline import DESIGN_STAGES, drive_chain
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


def derive_stage_statuses(
    state: dict,
    *,
    project_status: str | None = None,
    sheets_exist: bool = False,
    synth_checks_failed: bool = False,
    pcb_ready: bool = False,
    zip_ok: bool = False,
) -> dict[str, str]:
    """Restore ``pending|parked|done|warning|failed`` pipeline statuses.

    A committed candidate advances even with semantic diagnostics. Legacy
    committed entries lacking semantic fields remain ``done``.
    """
    ss = state.get("stage_status") or {}
    out: dict[str, str] = {}
    for s in DESIGN_STAGES:
        e = ss.get(s)
        if isinstance(e, dict) and e.get("ok") is True:
            diagnostics = e.get("diagnostics") or []
            out[s] = (
                "warning"
                if (
                    e.get("semantic_clean") is False
                    or e.get("repair_required") is True
                    or bool(diagnostics)
                )
                else "done"
            )
        elif isinstance(e, dict) and e.get("ok") is False:
            out[s] = "failed"
        elif _stage_done(s, state):
            out[s] = "done"  # legacy project predating stage_status
        else:
            out[s] = "pending"
    for q in state.get("open_questions") or []:
        s = q.get("stage")
        if not q.get("answer") and out.get(s) not in (None, "done", "warning"):
            out[s] = "parked"

    design_complete = all(out[s] in {"done", "warning"} for s in DESIGN_STAGES)
    failed = project_status == "failed"
    fab_gated = any(
        isinstance(entry, dict)
        and (
            entry.get("fab_safe") is False
            or any(
                isinstance(diagnostic, dict) and diagnostic.get("severity") == "fab_gate"
                for diagnostic in entry.get("diagnostics") or []
            )
        )
        for entry in ss.values()
    )
    artifacts = state.get("artifacts") or {}
    pcb_errors = artifacts.get("pcb_errors") or []
    route_error = any(
        isinstance(error, dict) and error.get("stage") == "place_route" for error in pcb_errors
    )
    verify_error = any(
        isinstance(error, dict) and error.get("stage") == "verify" for error in pcb_errors
    )
    synth_ok = design_complete and sheets_exist and not synth_checks_failed
    out["synthesize"] = (
        "done" if synth_ok else "failed" if design_complete and failed else "pending"
    )
    # A produced board means place/route succeeded unless the durable payload
    # explicitly records a place/route terminal error. A verify-only error keeps
    # the produced candidate inspectable as done/warning here.
    has_warnings = bool(artifacts.get("build_warnings"))
    out["place_route"] = (
        "failed"
        if design_complete and route_error
        else "warning"
        if design_complete and pcb_ready and has_warnings
        else "done"
        if design_complete and pcb_ready
        else "failed"
        if synth_ok and failed
        else "pending"
    )
    # failed-with-a-board outranks zip_ok: after a failed (re)build the board
    # on disk is the failed candidate, so any surviving zip is stale. Either
    # terminal PCB error invalidates the fab package.
    out["fab"] = (
        "failed"
        if design_complete and fab_gated
        else "failed"
        if design_complete and pcb_ready and (failed or route_error or verify_error)
        else "warning"
        if design_complete and zip_ok and has_warnings
        else "done"
        if design_complete and zip_ok
        else "pending"
    )
    # Electrical review: the post-wiring review writes its durable outcome to
    # stage_status (like the design stages) and its findings to the top-level
    # review_findings slot (artifacts.review_findings on legacy projects). A
    # blocker the re-drive did not clear reads as a yellow 'warning' (the run
    # proceeded; the gap is recorded), never a red failure. No stage_status
    # entry (review skipped / pre-R3 project) stays 'pending'.
    er = ss.get("electrical_review")
    findings = (
        state.get("review_findings") or (state.get("artifacts") or {}).get("review_findings") or []
    )
    has_blocker = any(isinstance(f, dict) and f.get("severity") == "blocker" for f in findings)
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
    return list(_downstream_stages(stage))


def read_state(ws) -> dict:
    """Best-effort load of the canonical ``<ws>/.kicraft/state.json``."""
    p = _state_path(Path(ws))
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def commit_slot(ws, stage: str, slot: dict, brief: str = "", project_stem=None):
    """Commit an edited slot to the workspace state.json via the deterministic CLI
    (which re-validates it). Returns (ok, out); out carries `errors` on rejection."""
    from .stage_state_io import commit_stage, stamp_stage_status

    state_path = Path(ws) / ".kicraft" / "state.json"
    ok, out = commit_stage(stage, dict(slot), state_path, brief, project_stem, Path(ws))
    if ok:  # a manual edit is a zero-cost commit; the stage is (re)done
        stamp_stage_status(state_path, stage, True)
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
    cleared = list(invalidate_downstream(sj, stage))
    atomic_write_text(state_path, json.dumps(sj, indent=2) + "\n")
    return cleared


def run_session(
    ws,
    brief: str,
    stages,
    answers=None,
    instruction=None,
    client=None,
    progress=None,
    run_id=None,
    core_defaults=None,
    auto_default_questions: bool | None = None,
) -> dict:
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
        return {"status": "ok", "results": [], "guard": None, "questions": None, "last_stage": None}
    results, guard, state_path = drive_chain(
        stages,
        brief,
        Path(ws),
        progress=progress,
        client=client,
        answers=answers,
        instruction=instruction,
        run_id=run_id,
        core_defaults=core_defaults,
        auto_default_questions=auto_default_questions,
    )
    last = results[-1] if results else None
    if last and last.get("needs_input"):
        status = "awaiting_input"
    elif results and all(r.get("commit_ok") for r in results):
        status = "ok"
    else:
        status = "failed"
    return {
        "status": status,
        "results": results,
        "guard": guard,
        "state_path": state_path,
        "questions": (last.get("questions") if last else None),
        "last_stage": (last.get("stage") if last else None),
        "failure_kind": (last.get("failure_kind") if last else None),
        "retryable": bool(last and last.get("failure_kind") == "provider_rate_limited"),
        "retry_action": (
            "retry_stage" if last and last.get("failure_kind") == "provider_rate_limited" else None
        ),
    }


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


# A wiring deficit that asks for a swap, not an addition: "Replace U1
# (ADuM1301ARWZ) with two ADuM1201ARZ ...". The add-only instruction below
# actively forbade the drop ("do NOT drop any part already present"), so the
# BOM stage obeyed the instruction over the deficit and recommitted an
# identical BOM until the stuck-loop guard killed the run (self-eval
# 2026-07-27 run_08).
_REPLACE_ASK_RE = re.compile(
    r"\breplace\s+(?:the\s+)?[A-Z]{1,4}\d+[A-Z0-9]*\b[\s\S]{0,120}?\bwith\b",
    re.IGNORECASE,
)


def bom_reconcile_instruction(questions) -> str:
    """Turn wiring's ``reconcile_target="bom"`` deficit note(s) into a BOM-stage
    instruction that applies them. Each question's text is already a precise
    "add N of X for pins Y" (or "replace REF with ...") statement per the wiring
    spec; the framing must permit exactly what the deficit asks -- add-only when
    it only adds, replacement-permitting when it names a swap."""
    lines = [str(q.get("text", "")).strip() for q in questions if str(q.get("text", "")).strip()]
    body = "\n- ".join(lines)
    if any(_REPLACE_ASK_RE.search(ln) for ln in lines):
        return (
            "The wiring stage could not finish because the BOM does not provide "
            "the parts it needs. Apply the change(s) described below EXACTLY: "
            "where a replacement is named, remove ONLY the named part(s) and add "
            "the named successor(s) (fresh unique refs, correct value and "
            "footprint, the same sheet as the IC each serves, ic_groups updated "
            "to match); where an addition is named, add it the same way. Keep "
            "every part the note does not name. Then re-emit the FULL BOM. Do "
            "NOT ask the user:\n- " + body
        )
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
        q for q in (res.get("questions") or []) if q.get("reconcile_target") == BOM_RECONCILE_TARGET
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
    r"(\d+(?:\.\d+)?[\s-]?(?:[pnumµ]F|[kM](?:Ω|ohm)?\b|Ω|ohm\b|R\b))"
    r"[^.;,]*?\b(cap(?:acitor)?|resistor|inductor)s?\b",
    re.IGNORECASE,
)
# Kind-then-value supplement: "Add a bottom resistor (e.g., 3.9k, typical for
# 3.3V output)" names the value AFTER the noun, which the value-first regex
# above cannot see -- self-eval 2026-07-27 run_22 died with exactly this
# single-resistor ask falling through to an LLM pass that never applied it.
# Scanned ONLY inside sentences containing an ask-verb (see
# ``_ASK_VERB_RE``), so descriptive prose like "the BOM has only a top
# resistor R3 (12k)" can never provision a duplicate.
_PASSIVE_KIND_VALUE_RE = re.compile(
    r"(?:\b(a|an|one|two|three|four|\d+)\s+)?"
    r"(?:[a-z-]+\s+){0,2}?(cap(?:acitor)?|resistor|inductor)s?\b"
    r"(?:(?!\b(?:and|or)\b)[^.;]){0,40}?"
    r"(\d+(?:\.\d+)?[\s-]?(?:[pnumµ]F|[kM](?:Ω|ohm)?\b|Ω|ohm\b|R\b))",
    re.IGNORECASE,
)
_ASK_VERB_RE = re.compile(
    r"\b(add|need(?:s|ed)?|require(?:s|d)?|missing|provision|include)\b",
    re.IGNORECASE,
)
# "e.g."/"i.e." would break the sentence split below ("(e.g., 3.9k" spans two
# periods); decimals are protected by the (?!\d) lookahead instead.
_ABBREV_DOT_RE = re.compile(r"\b([ei])\.\s?([ge])\.,?", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"[;.](?!\d)")
# Part nouns the deterministic passive-add can NEVER provision. When a deficit
# note asks for one of these alongside parseable passives, "added something"
# must not read as "added everything": board 639's note asked for a crystal +
# load caps + a u.FL -- the caps were added, wiring was re-driven with "do NOT
# park on this again", and the committed board had the load caps wired to a
# crystal that never existed (2026-07-19 review §5.8).
_NON_PASSIVE_ASK_RE = re.compile(
    r"\b(crystal|oscillator|resonator|antenna|u\.?fl|connector|header|jack|"
    r"socket|receptacle|diode|led\b|transistor|mosfet|regulator|switch|"
    r"button|fuse|ferrite|choke|varistor|tvs|test[ _-]?(?:point|pad)|"
    r"screw[ _-]?terminals?|terminal[ _-]?blocks?)\b",
    re.IGNORECASE,
)
_QTY_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4}
_SHEET_RE = re.compile(r"\b(?:on|to) the ([A-Z][A-Z0-9 _/-]*?) sheet\b")
_KIND_PREFIX = {"capacitor": "C", "resistor": "R", "inductor": "L"}
_KIND_ALIAS = {"cap": "capacitor"}
_KIND_SYMBOL = {"capacitor": "Device:C", "resistor": "Device:R", "inductor": "Device:L"}
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


def _ask_qty(qty_tok: str | None) -> int:
    tok = (qty_tok or "a").lower()
    qty = _QTY_WORDS.get(tok)
    if qty is None:
        try:
            qty = max(1, min(8, int(tok)))
        except ValueError:
            qty = 1
    return qty


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
        sheet = m_sheet.group(1).strip() if m_sheet and " sheets" not in t else None
        seen: set[tuple[str, str]] = set()
        for m in _PASSIVE_ASK_RE.finditer(t):
            kind, value = m.group(3).lower(), m.group(2).strip()
            kind = _KIND_ALIAS.get(kind, kind)
            seen.add((kind, _norm_value(value)))
            asks.append(
                {
                    "kind": kind,
                    "value": value,
                    "qty": _ask_qty(m.group(1)),
                    "sheet": sheet,
                }
            )
        # Kind-then-value supplement, scoped to ask-verb sentences and deduped
        # against the value-first pass so the same ask never provisions twice.
        for sent in _SENTENCE_SPLIT_RE.split(_ABBREV_DOT_RE.sub(r"\1\2", t)):
            if not _ASK_VERB_RE.search(sent):
                continue
            for m in _PASSIVE_KIND_VALUE_RE.finditer(sent):
                kind, value = m.group(2).lower(), m.group(3).strip()
                kind = _KIND_ALIAS.get(kind, kind)
                key = (kind, _norm_value(value))
                if key in seen:
                    continue
                seen.add(key)
                asks.append(
                    {
                        "kind": kind,
                        "value": value,
                        "qty": _ask_qty(m.group(1)),
                        "sheet": sheet,
                    }
                )
    return asks


def _next_ref(parts: list[dict], prefix: str) -> str:
    used = set()
    for p in parts:
        r = str(p.get("ref") or "")
        if r.startswith(prefix) and r[len(prefix) :].isdigit():
            used.add(int(r[len(prefix) :]))
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
            c
            for c in cands
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
    asks = parse_passive_deficits([str(q.get("text", "")) for q in deficits])
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
        for p in parts
        if p.get("sheet")
    }

    added: list[str] = []
    for ask in asks:
        ask_sheet = known_sheets.get(str(ask["sheet"] or "").strip().upper())
        want = _norm_value(ask["value"])
        prefix = _KIND_PREFIX[ask["kind"]]
        same_value = [
            p
            for p in parts
            if str(p.get("ref", "")).startswith(prefix)
            and _norm_value(str(p.get("value", ""))) == want
        ]
        ungrouped = [
            p for p in same_value if p.get("reconcile_added") and str(p.get("ref")) not in grouped
        ]
        if len(ungrouped) >= ask["qty"]:
            continue
        donor = next(
            (p for p in same_value if p.get("sourcing_note") or p.get("mpn")),
            None,
        )
        pick = None if donor is not None else _catalog_passive(ask["kind"], ask["value"])
        if donor is None and pick is None:
            continue
        for _ in range(ask["qty"] - len(ungrouped)):
            ref = _next_ref(parts, prefix)
            if donor is not None:
                entry = dict(donor)
                entry["ref"] = ref
                entry["reconcile_added"] = True
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
                    "reconcile_added": True,
                }
            parts.append(entry)
            added.append(ref)
    if not added:
        return []
    try:
        # Atomic like every other state.json commit: three processes read this
        # file as their IPC contract, and a mid-write kill must never leave it
        # truncated.
        atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")
    except Exception:
        return []
    return added


def _unfulfilled_asks(ws, deficits) -> list[dict]:
    """Parsed passive asks with no ``reconcile_added`` matching part committed.

    Non-empty => the deterministic pass did not fully satisfy the deficit (an
    ask has no ``reconcile_added`` same-kind/same-value part in the committed
    BOM). Fails open: on any read error every parsed ask is returned, so the
    caller falls through to the LLM bom+wiring pass rather than falsely
    claiming the deficit was fulfilled."""
    asks = parse_passive_deficits([str(q.get("text", "")) for q in (deficits or [])])
    if not asks:
        return []
    try:
        state = json.loads(_state_path(Path(ws)).read_text(encoding="utf-8"))
        parts = (state.get("bom") or {}).get("parts")
    except Exception:
        return asks
    if not isinstance(parts, list):
        return asks
    unfulfilled: list[dict] = []
    for ask in asks:
        prefix = _KIND_PREFIX.get(ask["kind"])
        if prefix is None:
            unfulfilled.append(ask)
            continue
        want = _norm_value(ask["value"])
        if not any(
            isinstance(p, dict)
            and p.get("reconcile_added")
            and str(p.get("ref", "")).startswith(prefix)
            and _norm_value(str(p.get("value", ""))) == want
            for p in parts
        ):
            unfulfilled.append(ask)
    return unfulfilled


# Deficit identity for the stuck-loop test: the set of component refs the
# note names plus the passive kinds it asks for. Models reword the same
# deficit between passes ("3.3k to 4.7k" -> "3.9k, typical"), so raw-text
# equality would misread a reworded repeat as a new deficit; refs+kinds are
# stable across rewordings while genuinely different deficits (screw terminals
# vs a BOOT0 strap) differ.
_REF_TOKEN_RE = re.compile(r"\b[A-Z]{1,4}\d{1,4}\b")


def _deficit_key(questions) -> tuple[frozenset, frozenset]:
    texts = [str(q.get("text", "")) for q in (questions or [])]
    refs = frozenset(m.group(0) for t in texts for m in _REF_TOKEN_RE.finditer(t))
    kinds = frozenset(a["kind"] for a in parse_passive_deficits(texts))
    return (refs, kinds)


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
    ws,
    brief,
    res,
    *,
    progress=None,
    run_id=None,
    core_defaults=None,
    client=None,
    reconcile_passes: int = 0,
    auto_default_questions: bool | None = None,
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
    # Partial fulfillment guard: the deterministic pass only provisions
    # regex-parseable R/C/L asks. If the note ALSO names a part class it can
    # never add (crystal, connector, ...) OR a parseable passive ask was not
    # actually provisioned (skipped by a pre-existing part, or an abbreviation
    # the parser could not fully resolve), the wiring-only re-drive below would
    # command "do NOT park on this deficit again" while the deficit is still
    # real -- fall through to the LLM bom+wiring pass instead (the
    # already-added passives are committed and preserved).
    _texts = " ".join(str(q.get("text", "")) for q in deficits)
    _unfulfilled_nonpassive = _NON_PASSIVE_ASK_RE.search(_texts)
    _unfulfilled_passive = _unfulfilled_asks(ws, deficits)
    if added and (_unfulfilled_nonpassive or _unfulfilled_passive):
        if progress is not None:
            _remainders = []
            if _unfulfilled_nonpassive:
                _remainders.append(f"non-passive part(s) ({_unfulfilled_nonpassive.group(0)!r})")
            if _unfulfilled_passive:
                _remainders.append(
                    "unfulfilled passive ask(s) ("
                    + ", ".join(f"{a['kind']} {a['value']}" for a in _unfulfilled_passive)
                    + ")"
                )
            progress(
                {
                    "kind": "build_log",
                    "text": f"[bom-reconcile] deterministically provisioned "
                    f"{', '.join(added)}, but the deficit also asks "
                    f"for {' and '.join(_remainders)} the deterministic "
                    "pass cannot add -- falling through to the LLM "
                    "bom+wiring pass for the remainder",
                }
            )
        added = []
    if added:
        if progress is not None:
            progress(
                {
                    "kind": "build_log",
                    "text": f"[bom-reconcile] deterministically provisioned "
                    f"{', '.join(added)} from the wiring deficit note "
                    f"(pass {reconcile_passes + 1}/"
                    f"{BOM_RECONCILE_MAX_PASSES}); re-driving wiring "
                    "to connect them (no model BOM pass)",
                }
            )
        rr = run_session(
            ws,
            brief,
            ["wiring"],
            instruction=(
                "The missing supporting parts from your deficit note were "
                f"added to the BOM as {', '.join(added)} (same value/footprint "
                "sourcing as specified). Re-emit the FULL wiring, connecting "
                "each of them exactly as your note described. Do NOT ask the "
                "user and do NOT park on the same deficit again."
            ),
            progress=progress,
            run_id=run_id,
            core_defaults=core_defaults,
            client=client,
            auto_default_questions=auto_default_questions,
        )
        return rr, reconcile_passes + 1
    if progress is not None:
        progress(
            {
                "kind": "build_log",
                "text": f"[bom-reconcile] wiring flagged a BOM parts shortfall; "
                f"re-driving bom+wiring (pass {reconcile_passes + 1}/"
                f"{BOM_RECONCILE_MAX_PASSES}) to add the missing parts "
                "(not asking the user)",
            }
        )
    before = _bom_signature(ws)
    rr = run_session(
        ws,
        brief,
        ["bom", "wiring"],
        instruction=bom_reconcile_instruction(deficits),
        progress=progress,
        run_id=run_id,
        core_defaults=core_defaults,
        client=client,
        auto_default_questions=auto_default_questions,
    )
    passes = reconcile_passes + 1
    after = _bom_signature(ws)
    if before is not None and after is not None and after == before:
        # A changed-nothing pass is a stuck loop ONLY when wiring re-parks on
        # the SAME deficit. When it re-parks on a NEW one the chain advanced
        # (wiring revealed the next genuine need) and killing the whole budget
        # here denies that new deficit even its deterministic-add attempt --
        # self-eval 2026-07-27 run_24 died with a trivially clonable BOOT0
        # strap ask exactly this way.
        new_defs = bom_reconcile_deficits(rr)
        if new_defs and _deficit_key(new_defs) != _deficit_key(deficits):
            if progress is not None:
                progress(
                    {
                        "kind": "build_log",
                        "text": "[bom-reconcile] the pass changed nothing in "
                        "the committed BOM, but wiring re-parked on a "
                        "DIFFERENT deficit -- treating as an advancing "
                        "chain, not a stuck loop",
                    }
                )
        else:
            if progress is not None:
                progress(
                    {
                        "kind": "build_log",
                        "text": "[bom-reconcile] the pass changed nothing in the "
                        "committed BOM -- stopping reconcile (stuck loop, "
                        "not a deficit chain)",
                    }
                )
            passes = BOM_RECONCILE_MAX_PASSES
        return rr, passes
    # The BOM changed; if wiring still re-parks on the SAME deficit, the model
    # changed something irrelevant -- spend one pointed retry that quotes the
    # unmet ask verbatim before the caller burns another generic pass
    # (2026-07-27 run_22: two "ok" BOM passes never added the asked-for FB
    # resistor).
    new_defs = bom_reconcile_deficits(rr)
    if (
        new_defs
        and passes < BOM_RECONCILE_MAX_PASSES
        and _deficit_key(new_defs) == _deficit_key(deficits)
    ):
        if progress is not None:
            progress(
                {
                    "kind": "build_log",
                    "text": f"[bom-reconcile] the pass changed the BOM but did "
                    f"NOT resolve the deficit; pointed retry (pass "
                    f"{passes + 1}/{BOM_RECONCILE_MAX_PASSES}) naming "
                    "the unmet ask",
                }
            )
        before2 = after
        rr = run_session(
            ws,
            brief,
            ["bom", "wiring"],
            instruction=(
                "Your previous BOM pass changed the BOM but did NOT resolve the "
                "deficit below -- the named part(s) are STILL missing. Apply the "
                "note literally this time. " + bom_reconcile_instruction(deficits)
            ),
            progress=progress,
            run_id=run_id,
            core_defaults=core_defaults,
            client=client,
            auto_default_questions=auto_default_questions,
        )
        passes += 1
        after2 = _bom_signature(ws)
        retry_defs = bom_reconcile_deficits(rr)
        if (
            before2 is not None
            and after2 is not None
            and after2 == before2
            and retry_defs
            and _deficit_key(retry_defs) == _deficit_key(deficits)
        ):
            if progress is not None:
                progress(
                    {
                        "kind": "build_log",
                        "text": "[bom-reconcile] pointed retry changed nothing "
                        "either -- stopping reconcile (stuck loop)",
                    }
                )
            passes = BOM_RECONCILE_MAX_PASSES
    return rr, passes


# --------------------------------------------------------------------------- #
# Shared build-to-design recovery (phase D)
#
# Web, headless generation, and batch self-evaluation previously differed here:
# web had a single ERC-only wiring retry, the other two had none. The build
# worker still only runs the deterministic build and returns its exit code plus
# durable evidence; THIS module owns the design revision decision and cannot be
# a second autonomous designer.
#
# The recovery budget lives in ``stage_status["build_recovery"]`` (see
# ``models.BuildRecoveryEvent``): a resumed or re-entered run reads the same
# durable counter, so a cap can never reset by re-entry.
# --------------------------------------------------------------------------- #
BUILD_RECOVERY_STATUS_KEY = "build_recovery"
# Total design-revision actions (model re-drives) one project state may spend on
# build feedback. Mirrors BOM_RECONCILE_MAX_PASSES' philosophy: enough for a
# genuine chain, cut short by the unchanged/repeated/oscillating checks below.
BUILD_RECOVERY_MAX_ATTEMPTS = 3
# Bound the persisted history so a pathological project cannot grow state.json
# without limit; the newest rows are the ones the policy reads.
_BUILD_RECOVERY_MAX_EVENTS = 12

# Build exit codes (kicraft.design.cli_app): 5 = synthesis gate (ERC and the
# other deterministic checks), 6 = place/route or tooling abort, 7 = verify/DRC
# rejection with a produced board. 0 is a completed build. Everything else is a
# setup/tooling/process failure with no attributable design evidence.
_INFRA_BUILD_RCS = frozenset({1, 2, 3, 4, 8})

# Gate code -> owning choice. Codes are the stable ``§9.x`` ids carried in the
# deterministic check name; the choice that can actually change the failing fact
# belongs to the stage named here. Wiring owns nets/connections/no-connects and
# their electrical coverage; BOM owns parts, footprints, support circuits,
# quantities, sourcing; architecture owns sheets, ownership, realization
# topology, and reviewed recipe selection; "compiler" means a reviewed
# deterministic implementation owns the fact and the model may not re-author it.
_BUILD_OWNER_BY_GATE = {
    "9.9": "wiring",
    "9.10": "wiring",
    "9.11": "wiring",
    "9.12": "wiring",
    "9.14": "wiring",
    "9.15": "wiring",
    "9.16": "wiring",
    "9.17": "wiring",
    "9.18": "wiring",
    "9.19": "wiring",
    "9.20": "wiring",
    "9.31": "wiring",
    "9.36": "wiring",
    "9.2": "bom",
    "9.6": "bom",
    "9.13": "bom",
    "9.21": "bom",
    "9.23": "bom",
    "9.25": "bom",
    "9.26": "bom",
    "9.27": "bom",
    "9.28": "bom",
    "9.29": "bom",
    "9.32": "bom",
    "9.33": "bom",
    "9.34": "bom",
    "9.35": "bom",
    "9.3": "architecture",
    "9.4": "architecture",
    "9.7": "architecture",
    "9.8": "architecture",
    "9.22": "architecture",
    "9.24": "architecture",
    "9.42": "architecture",
    # §9.43/§9.44 both read a datum the architecture stage declares: the address
    # strap a requirement demands for its converter, and the voltage a typed
    # rail puts on an analog input. The owning repair is that assignment, not
    # the reviewed part that honestly reports its limit.
    "9.43": "architecture",
    "9.44": "architecture",
    "9.37": "compiler",
    "9.38": "compiler",
    "9.39": "compiler",
    "9.40": "compiler",
    "9.41": "compiler",
}
# Checks that validate a REVIEWED implementation's own construction. A failure
# whose named refs are recipe/lowerer-owned is a protected compiler defect: the
# model may not be asked to re-author the recipe's pins or parts.
_COMPILER_OWNED_GATES = frozenset({"9.37", "9.38", "9.39", "9.40", "9.41"})
# Checks whose failure means the REQUIRED implementation is not constructible
# within the allowed envelope (part availability, footprint reality, physical
# realization, unsurfaced substitution). Those may justify a permitted reviewed
# alternative -- and nothing at all when the user fixed the part/limit.
_AVAILABILITY_GATES = frozenset({"9.13", "9.23", "9.26", "9.28", "9.33", "9.34", "9.35", "9.42"})
# Built-in PCB error codes that name deterministic design defects we can act on
# by revising an allowed choice (never by relaxing DRC or rewriting geometry).
_LAYOUT_DESIGN_CODES = frozenset(
    {
        "unconnected",
        "shorts",
        "courtyards_overlap",
        "keepout_intrusion",
        "missing_refs",
        "malformed_board_geometry",
        "empty_board",
        "layout_failure",
        "verification_failed",
    }
)
# Tooling/transport failures: never a reason to change the circuit.
_LAYOUT_INFRA_CODES = frozenset(
    {"drc_timeout", "drc_unavailable", "drc_failed", "board_missing"}
)

_BUILD_GATE_RE = re.compile(r"^§?\s*(\d+\.\d+)\b")
_PIN_TOKEN_RE = re.compile(r"\b([A-Z]{1,4}\d{1,4})\s*[.\s]\s*(?:pin\s*)?(\d{1,3}[A-Za-z0-9_]*)\b")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")

# Recovery actions, and the choice invalidation each one requires. The second
# element is the LAST stage kept: ``null_downstream`` clears everything after it,
# so intent and functional_spec always survive and wiring-only repair keeps the
# committed BOM parts (its own connections are cleared).
_RECOVERY_PLAN = {
    "repair_wiring": ("bom", ("wiring",)),
    "backtrack_bom": ("bom", ("bom", "wiring")),
    "backtrack_architecture": ("functional_spec", ("architecture", "bom", "wiring")),
    "try_reviewed_alternative": ("bom", ("bom", "wiring")),
}
_ARCHITECTURE_ALTERNATIVE = ("functional_spec", ("architecture", "bom", "wiring"))


@dataclass(frozen=True)
class BuildFailure:
    """Attributable evidence from one failed deterministic build."""

    kind: str  # design_defect | compiler_defect | capability_gap | infrastructure | unattributable
    action: str  # repair_wiring | backtrack_bom | backtrack_architecture | try_reviewed_alternative | rebuild | none
    owner_stage: str | None
    fingerprint: str
    reason: str
    evidence: tuple[str, ...] = ()
    gate_codes: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()
    diagnostics: tuple[dict, ...] = ()
    user_required: bool = False


@dataclass
class BuildRecoveryState:
    """Durable, re-entry-proof recovery budget and history."""

    ok: bool | None = None
    failure_kind: str | None = None
    run_id: str | None = None
    attempts: int = 0
    max_attempts: int = BUILD_RECOVERY_MAX_ATTEMPTS
    # False when no durable record exists yet, so the caller's explicit
    # ``max_attempts`` still applies to a fresh project state.
    present: bool = False
    choice_fingerprints: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    @property
    def failure_fingerprints(self) -> set[str]:
        return {
            str(event.get("failure_fingerprint"))
            for event in self.events
            if event.get("failure_fingerprint")
        }


def build_recovery_state_path(ws) -> Path:
    return _state_path(Path(ws))


def read_build_recovery(ws) -> BuildRecoveryState:
    """Load the durable recovery budget from ``stage_status[build_recovery]``."""
    try:
        state = json.loads(build_recovery_state_path(ws).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return BuildRecoveryState()
    entry = (state.get("stage_status") or {}).get(BUILD_RECOVERY_STATUS_KEY)
    if not isinstance(entry, dict):
        return BuildRecoveryState()
    events = [event for event in (entry.get("recovery_events") or []) if isinstance(event, dict)]
    choices = [fp for fp in (entry.get("recovery_choice_fingerprints") or []) if isinstance(fp, str)]
    max_attempts = entry.get("recovery_max_attempts")
    return BuildRecoveryState(
        ok=entry.get("ok"),
        failure_kind=entry.get("failure_kind"),
        run_id=entry.get("recovery_run_id"),
        attempts=int(entry.get("recovery_attempts") or 0),
        max_attempts=(
            int(max_attempts) if isinstance(max_attempts, int) else BUILD_RECOVERY_MAX_ATTEMPTS
        ),
        present=True,
        choice_fingerprints=choices,
        events=events,
    )


def write_build_recovery(
    ws,
    *,
    ok: bool,
    failure_kind: str | None = None,
    run_id: str | None = None,
    attempts: int | None = None,
    max_attempts: int | None = None,
    choice_fingerprints: list[str] | None = None,
    events: list[dict] | None = None,
    diagnostics: list[dict] | None = None,
) -> dict:
    """Persist the recovery record under the existing durable status carrier.

    Read-modify-write over the raw state.json so a concurrent writer of another
    stage_status key is preserved and the record round-trips through
    ``models.StageStatus`` (the typed fields live there, not in a side database).
    """
    state_path = build_recovery_state_path(ws)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    block = state.get("stage_status")
    if not isinstance(block, dict):
        block = {}
    prior = block.get(BUILD_RECOVERY_STATUS_KEY)
    prior = prior if isinstance(prior, dict) else {}
    merged_attempts = int(prior.get("recovery_attempts") or 0) if attempts is None else attempts
    merged_max = (
        int(prior.get("recovery_max_attempts") or BUILD_RECOVERY_MAX_ATTEMPTS)
        if max_attempts is None
        else max_attempts
    )
    merged_choices = (
        list(prior.get("recovery_choice_fingerprints") or [])
        if choice_fingerprints is None
        else list(choice_fingerprints)
    )
    merged_events = (
        list(prior.get("recovery_events") or []) if events is None else list(events)
    )[-_BUILD_RECOVERY_MAX_EVENTS:]
    entry: dict = {
        "ok": bool(ok),
        "finished_at": _utc_now(),
        "repair_required": not ok,
        "repair_attempted": bool(merged_attempts),
        "repair_adopted": bool(merged_attempts) and bool(ok),
        "diagnostics": list(diagnostics or []),
        "recovery_attempts": merged_attempts,
        "recovery_max_attempts": merged_max,
        "recovery_choice_fingerprints": merged_choices,
        "recovery_events": merged_events,
    }
    if run_id:
        entry["recovery_run_id"] = run_id
    elif prior.get("recovery_run_id"):
        entry["recovery_run_id"] = prior["recovery_run_id"]
    if failure_kind:
        entry["failure_kind"] = failure_kind
    block[BUILD_RECOVERY_STATUS_KEY] = entry
    state["stage_status"] = block
    state_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")
    return entry


def _utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_build_evidence(ws) -> tuple[list[dict], list[dict]]:
    """Deterministic build failure evidence: (synthesis diagnostics, pcb errors).

    Synthesis rows are normalized StageDiagnostic dicts whose ``gate_codes``
    carry the stable ``§9.x`` id. PCB rows are the durable
    ``artifacts.pcb_errors`` payload. Both are the carriers build already writes;
    recovery invents no parallel exception hierarchy.
    """
    ws = Path(ws)
    diagnostics: list[dict] = []
    checks_path = ws / ".kicraft" / "synthesis_check.json"
    try:
        summary = json.loads(checks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = None
    if isinstance(summary, dict):
        for check in summary.get("checks") or []:
            if not isinstance(check, dict) or check.get("ok"):
                continue
            name = str(check.get("name") or "")
            offenders = [str(o) for o in (check.get("offenders") or []) if str(o).strip()]
            code = _gate_code(name)
            diagnostics.append(
                {
                    "code": f"build_gate_{(code or 'unknown').replace('.', '_')}",
                    "severity": "repair_required",
                    "message": (str(check.get("message") or "").strip() or name)[:400],
                    "evidence": offenders[:20],
                    "gate_codes": [code] if code else [],
                }
            )
    pcb_errors: list[dict] = []
    try:
        state = json.loads(build_recovery_state_path(ws).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    artifacts = state.get("artifacts") if isinstance(state, dict) else None
    if isinstance(artifacts, dict):
        for error in artifacts.get("pcb_errors") or []:
            if isinstance(error, dict):
                pcb_errors.append(error)
    return diagnostics, pcb_errors


def _gate_code(name: str) -> str | None:
    match = _BUILD_GATE_RE.match(name.strip())
    return match.group(1) if match else None


def _evidence_tokens(*texts) -> tuple[str, ...]:
    tokens: set[str] = set()
    for text in texts:
        for chunk in text if isinstance(text, (list, tuple)) else [text]:
            value = str(chunk or "")
            tokens.update(match.group(0) for match in _REF_TOKEN_RE.finditer(value))
            tokens.update(
                f"{m.group(1)}.{m.group(2)}" for m in _PIN_TOKEN_RE.finditer(value)
            )
            code = _gate_code(value)
            if code:
                tokens.add(code)
    return tuple(sorted(tokens))


def _failure_fingerprint(gate_codes, refs, texts) -> str:
    """Stable identity of a build failure: gate id + named ref/pin tokens.

    Deliberately NOT raw prose: a reworded report of the same unresolved defect
    must still read as unchanged, while a genuinely different defect (different
    gate or different named parts/pins) reads as new.
    """
    payload = json.dumps(
        {
            "gates": sorted(set(gate_codes)),
            "refs": sorted(set(refs)),
            "tokens": sorted(set(_evidence_tokens(*texts))),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized(text: str) -> str:
    return _NON_ALNUM_RE.sub("", str(text or "").upper())


def user_required_parts(state: dict) -> set[str]:
    """Normalized tokens of the parts the user named explicitly.

    ``intent.named_parts`` is the stage that records every exact MPN/IC/module
    the brief named; those may never be silently swapped for an easier
    implementation.
    """
    intent = state.get("intent") if isinstance(state, dict) else None
    named = (intent or {}).get("named_parts") or []
    return {token for token in (_normalized(p) for p in named) if len(token) >= 3}


def user_required_requirement_ids(state: dict) -> set[str]:
    """Original obligation ids the user stated, plus protected identities."""
    if not isinstance(state, dict):
        return set()
    ids: set[str] = set()
    for slot in ("intent", "functional_spec"):
        payload = state.get(slot) or {}
        for row in payload.get("obligations") or []:
            if isinstance(row, dict) and row.get("original_obligation_id"):
                ids.add(str(row["original_obligation_id"]))
    architecture = state.get("architecture") or {}
    ids.update(str(item) for item in architecture.get("protected_identities") or [])
    return ids


def _part_ownership(state: dict) -> dict[str, str]:
    """ref -> resolution_source ("recipe"/"lowerer"/"llm"/"reuse"/"")."""
    bom = (state.get("bom") or {}) if isinstance(state, dict) else {}
    out: dict[str, str] = {}
    for part in bom.get("parts") or []:
        if isinstance(part, dict) and part.get("ref"):
            out[str(part["ref"])] = str(part.get("resolution_source") or "")
    return out


def _part_values(state: dict) -> dict[str, str]:
    """ref -> committed part value/MPN, for matching user-named parts."""
    bom = (state.get("bom") or {}) if isinstance(state, dict) else {}
    out: dict[str, str] = {}
    for part in bom.get("parts") or []:
        if isinstance(part, dict) and part.get("ref"):
            out[str(part["ref"])] = " ".join(
                str(part.get(key) or "") for key in ("value", "mpn")
            ).strip()
    return out


def classify_build_failure(ws, *, rc: int | None = None) -> BuildFailure:
    """Attribute one failed build to the smallest choice that can change it."""
    ws = Path(ws)
    diagnostics, pcb_errors = read_build_evidence(ws)
    try:
        state = json.loads(build_recovery_state_path(ws).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    if not isinstance(state, dict):
        state = {}

    gate_codes = [
        code for diag in diagnostics for code in (diag.get("gate_codes") or []) if code
    ]
    texts = [str(diag.get("message") or "") for diag in diagnostics]
    texts += [str(item) for diag in diagnostics for item in (diag.get("evidence") or [])]
    pcb_codes = [str(error.get("code") or "") for error in pcb_errors]
    texts += [str(error.get("title") or "") + " " + str(error.get("explanation") or "")
              for error in pcb_errors]
    texts += [str(item) for error in pcb_errors for item in (error.get("details") or [])]
    refs = _evidence_tokens(*texts)
    refs = tuple(token for token in refs if not _gate_code(token))

    fingerprint = _failure_fingerprint(gate_codes or pcb_codes, refs, texts)
    if not diagnostics and not pcb_errors:
        kind = "infrastructure" if rc in _INFRA_BUILD_RCS else "unattributable"
        return BuildFailure(
            kind=kind,
            action="none",
            owner_stage=None,
            fingerprint=fingerprint,
            reason=(
                "the build failed without attributable deterministic evidence"
                + (f" (exit {rc})" if rc is not None else "")
            ),
            diagnostics=tuple(diagnostics),
            refs=refs,
        )

    ownership = _part_ownership(state)
    named = user_required_parts(state)
    required_ids = user_required_requirement_ids(state)
    evidence_refs = tuple(ref for ref in refs if ref in ownership)
    # The requirement is user-required when the evidence names a user-supplied
    # part (by the part's committed value, or by literal text) or when the
    # failure scope intersects the user's obligations / protected identities.
    values = _part_values(state)
    user_pinned = any(_normalized(values.get(ref, "")) in named for ref in evidence_refs)
    if not user_pinned:
        user_pinned = any(token in _normalized(text) for token in named for text in texts)
    if not user_pinned and required_ids:
        diag_ids = {
            str(diag["requirement_id"]) for diag in diagnostics if diag.get("requirement_id")
        }
        user_pinned = bool(diag_ids & required_ids)

    owner = None
    for code in gate_codes:
        owner = _BUILD_OWNER_BY_GATE.get(code)
        if owner:
            break
    if owner is None and pcb_errors:
        owner = "bom"

    # (1) A reviewed implementation failing its own construction check is a
    # protected compiler defect when the named refs are recipe/lowerer-owned.
    compiler_owned = any(
        ownership.get(ref) in ("recipe", "lowerer") for ref in evidence_refs
    )
    if compiler_owned and (
        set(gate_codes) & _COMPILER_OWNED_GATES
        or (owner == "compiler")
    ):
        return BuildFailure(
            kind="compiler_defect",
            action="none",
            owner_stage="compiler",
            fingerprint=fingerprint,
            reason=(
                "a reviewed deterministic implementation failed its own construction "
                f"check ({', '.join(sorted(set(gate_codes))) or 'reviewed invariant'})"
            ),
            evidence=tuple(texts[:20]),
            gate_codes=tuple(gate_codes),
            refs=refs,
            diagnostics=tuple(diagnostics),
            user_required=user_pinned,
        )

    # (2) An immutable user-required part/limit with no safe constructible
    # implementation is a capability gap: state it, never loop on it.
    if user_pinned and (set(gate_codes) & _AVAILABILITY_GATES):
        return BuildFailure(
            kind="capability_gap",
            action="none",
            owner_stage=None,
            fingerprint=fingerprint,
            reason=(
                "a user-required part/limit has no constructible implementation within "
                "the allowed envelope"
            ),
            evidence=tuple(texts[:20]),
            gate_codes=tuple(gate_codes),
            refs=refs,
            requirement_ids=tuple(sorted(required_ids))[:20],
            diagnostics=tuple(diagnostics),
            user_required=True,
        )
    if user_pinned and all(code in _LAYOUT_INFRA_CODES for code in pcb_codes) and pcb_codes:
        return BuildFailure(
            kind="infrastructure",
            action="none",
            owner_stage=None,
            fingerprint=fingerprint,
            reason="the place/route or verification tooling failed; not a circuit change",
            evidence=tuple(texts[:20]),
            refs=refs,
            diagnostics=tuple(diagnostics),
        )

    # (3) Placement/routing/verification failure: revise an ALLOWED choice only.
    if pcb_errors and not gate_codes:
        design_codes = [code for code in pcb_codes if code in _LAYOUT_DESIGN_CODES]
        if not design_codes and all(code in _LAYOUT_INFRA_CODES for code in pcb_codes):
            return BuildFailure(
                kind="infrastructure",
                action="none",
                owner_stage=None,
                fingerprint=fingerprint,
                reason="the place/route or verification tooling failed; not a circuit change",
                evidence=tuple(texts[:20]),
                refs=refs,
                diagnostics=tuple(diagnostics),
            )
        if not evidence_refs:
            return BuildFailure(
                kind="infrastructure",
                action="none",
                owner_stage=None,
                fingerprint=fingerprint,
                reason=(
                    "the layout failed without a named component/net to attribute to a "
                    "design choice"
                ),
                evidence=tuple(texts[:20]),
                refs=refs,
                diagnostics=tuple(diagnostics),
            )
        if user_pinned:
            return BuildFailure(
                kind="capability_gap",
                action="none",
                owner_stage=None,
                fingerprint=fingerprint,
                reason=(
                    "the user-required geometry/parts cannot be placed and routed within "
                    "the allowed layout freedom"
                ),
                evidence=tuple(texts[:20]),
                refs=refs,
                diagnostics=tuple(diagnostics),
                user_required=True,
            )
        if all(ownership.get(ref) in ("recipe", "lowerer") for ref in evidence_refs):
            return BuildFailure(
                kind="compiler_defect",
                action="none",
                owner_stage="compiler",
                fingerprint=fingerprint,
                reason=(
                    "reviewed deterministic parts could not be laid out; the constraint is "
                    "internal, not a free model choice"
                ),
                evidence=tuple(texts[:20]),
                refs=refs,
                diagnostics=tuple(diagnostics),
            )
        # Exhausted routing may only revise a proven allowed choice with concrete
        # evidence -- never arbitrary geometry and never a relaxed DRC gate.
        return BuildFailure(
            kind="design_defect",
            action="try_reviewed_alternative",
            owner_stage="bom",
            fingerprint=fingerprint,
            reason=(
                "attributable place/route constraint on model-selected parts "
                f"({', '.join(design_codes) or 'layout'})"
            ),
            evidence=tuple(texts[:20]),
            gate_codes=tuple(gate_codes),
            refs=refs,
            diagnostics=tuple(diagnostics),
        )

    # (4) Ordinary design defect: repair the smallest owning choice.
    if owner == "compiler":
        return BuildFailure(
            kind="compiler_defect",
            action="none",
            owner_stage="compiler",
            fingerprint=fingerprint,
            reason="a reviewed deterministic invariant failed; not model-repairable",
            evidence=tuple(texts[:20]),
            gate_codes=tuple(gate_codes),
            refs=refs,
            diagnostics=tuple(diagnostics),
        )
    if owner is None:
        return BuildFailure(
            kind="unattributable",
            action="none",
            owner_stage=None,
            fingerprint=fingerprint,
            reason="the failing check has no owning stage mapping",
            evidence=tuple(texts[:20]),
            gate_codes=tuple(gate_codes),
            refs=refs,
            diagnostics=tuple(diagnostics),
        )
    if set(gate_codes) & _AVAILABILITY_GATES:
        return BuildFailure(
            kind="design_defect",
            action="try_reviewed_alternative",
            owner_stage="architecture" if owner == "architecture" else "bom",
            fingerprint=fingerprint,
            reason=(
                "the selected implementation is not constructible; a permitted reviewed "
                f"alternative must be resolved ({', '.join(sorted(set(gate_codes) & _AVAILABILITY_GATES))})"
            ),
            evidence=tuple(texts[:20]),
            gate_codes=tuple(gate_codes),
            refs=refs,
            diagnostics=tuple(diagnostics),
        )
    action = {
        "wiring": "repair_wiring",
        "bom": "backtrack_bom",
        "architecture": "backtrack_architecture",
    }[owner]
    return BuildFailure(
        kind="design_defect",
        action=action,
        owner_stage=owner,
        fingerprint=fingerprint,
        reason=f"{owner}-owned deterministic constraint ({', '.join(sorted(set(gate_codes)))})",
        evidence=tuple(texts[:20]),
        gate_codes=tuple(gate_codes),
        refs=refs,
        diagnostics=tuple(diagnostics),
    )


def _owner_choice_fingerprint(ws, owner: str | None) -> str | None:
    """Identity of the owning CHOICE, for the unchanged/oscillation guards.

    Only the stages that own a *choice* (the BOM's parts and the architecture's
    requirements/recipe selections) have one. Wiring is the derived connection
    set of a choice, not a choice itself: its identity legitimately changes when
    the repair invalidates and re-derives it, so an unchanged-wiring check would
    be meaningless. A repeated wiring failure is instead caught by the identical
    failure fingerprint.
    """
    if owner not in ("bom", "architecture"):
        return None
    try:
        state = json.loads(build_recovery_state_path(ws).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if owner == "bom":
        bom = state.get("bom") or {}
        payload = {
            "parts": sorted(
                f"{p.get('ref')}|{p.get('symbol')}|{p.get('footprint')}|{p.get('value')}"
                for p in (bom.get("parts") or [])
                if isinstance(p, dict)
            ),
            "ic_groups": bom.get("ic_groups") or {},
        }
    else:
        architecture = state.get("architecture") or {}
        payload = {
            "requirements": sorted(
                str(r.get("id")) for r in (architecture.get("requirements") or []) if isinstance(r, dict)
            ),
            "recipes": sorted(
                f"{s.get('recipe')}@{s.get('instance')}"
                for s in (architecture.get("recipe_selections") or [])
                if isinstance(s, dict)
            ),
            "sheets": sorted(
                str(s.get("name")) for s in (architecture.get("sheets") or []) if isinstance(s, dict)
            ),
        }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _snapshot_state(ws) -> bytes | None:
    """Bytes of the current accepted state.json, for an adopt-or-restore probe."""
    try:
        return build_recovery_state_path(ws).read_bytes()
    except OSError:
        return None


def _restore_state(ws, snapshot: bytes | None) -> None:
    """Put the last accepted state back after a revision that did not commit."""
    if snapshot is None:
        return
    try:
        atomic_write_text(build_recovery_state_path(ws), snapshot.decode("utf-8"))
    except (OSError, UnicodeDecodeError):
        pass


def _budget_allows_recovery(client) -> bool:
    """Whether the shared ceilings leave room for one more design revision.

    Reads the SAME persistent guard the calls themselves consult; a mock/replay
    client without a guard imposes no ceiling (it spends nothing).
    """
    guard = getattr(client, "guard", None)
    if guard is None or not hasattr(guard, "status"):
        return True
    try:
        status = guard.status() or {}
    except Exception:  # noqa: BLE001 - a guard hiccup must not fake a refusal
        return True
    if status.get("kill_switch"):
        return False
    # A ceiling of 0/None means that scope is UNCONFIGURED (the mock/replay
    # guard reports all-zero ceilings and spends nothing) -- not exhausted.
    for remaining_key, ceiling_key in (
        ("daily_remaining_usd", "daily_ceiling_usd"),
        ("total_remaining_usd", "total_ceiling_usd"),
    ):
        ceiling = status.get(ceiling_key)
        remaining = status.get(remaining_key)
        if not isinstance(ceiling, (int, float)) or ceiling <= 0:
            continue
        if isinstance(remaining, (int, float)) and remaining <= 0.0:
            return False
    return True


def _recovery_instruction(failure: BuildFailure) -> str:
    """Concrete, constraint-scoped instruction naming the evidence and owner."""
    lines = [f"- {line}" for line in list(failure.evidence)[:12]]
    body = "\n".join(lines) or f"- {failure.reason}"
    scope = {
        "repair_wiring": "Fix ONLY the connections/no-connect pins that violate these constraints",
        "backtrack_bom": "Fix ONLY the parts these constraints name",
        "backtrack_architecture": "Revise ONLY the architecture choice these constraints name",
        "try_reviewed_alternative": (
            "Resolve a different COMPATIBLE reviewed implementation for the named choice"
        ),
    }.get(failure.action, "Resolve the constraints")
    return (
        f"The deterministic build failed with these concrete constraints:\n{body}\n"
        f"{scope}; keep every other requirement, net, and part unchanged. "
        "Do NOT drop required parts, do NOT relax any electrical or fabrication limit, "
        "and do NOT ask the user."
    )


def _emit_recovery_event(progress, *, action: str, reason: str, outcome: str,
                         requirement_ids=(), evidence=(), failure: BuildFailure | None = None) -> None:
    if progress is None:
        return
    progress(
        {
            "kind": "recovery_action",
            "action": action,
            "reason": reason,
            "requirement_ids": list(requirement_ids),
            "evidence": list(evidence)[:20],
            "outcome": outcome,
            "failure_kind": failure.kind if failure is not None else None,
            "owner_stage": failure.owner_stage if failure is not None else None,
        }
    )
    progress({"kind": "build_log", "text": f"[recover] {action}: {reason}"[:500]})


def _record_recovery_event(
    ws,
    *,
    failure: BuildFailure,
    action: str,
    outcome: str,
    choice_fingerprint: str | None,
    run_id: str | None,
    ok: bool,
    failure_kind: str | None,
    attempts: int,
    state: BuildRecoveryState,
    choice_fingerprints: list[str] | None = None,
) -> BuildRecoveryState:
    event = {
        "action": action,
        "reason": failure.reason,
        "outcome": outcome,
        "failure_fingerprint": failure.fingerprint,
        "requirement_ids": list(failure.requirement_ids),
        "evidence": list(failure.evidence)[:20],
        "diagnostics": list(failure.diagnostics),
    }
    if choice_fingerprint:
        event["choice_fingerprint"] = choice_fingerprint
    events = [*state.events, event]
    # The policy list is the set of choices already tried as a repair starting
    # point; the event's own choice_fingerprint is provenance and must not be
    # folded into it, or a re-entry would refuse its own starting choice.
    choices = list(state.choice_fingerprints if choice_fingerprints is None else choice_fingerprints)
    write_build_recovery(
        ws,
        ok=ok,
        failure_kind=failure_kind,
        run_id=run_id,
        attempts=attempts,
        max_attempts=state.max_attempts,
        choice_fingerprints=choices,
        events=events,
        diagnostics=list(failure.diagnostics),
    )
    return BuildRecoveryState(
        ok=ok,
        failure_kind=failure_kind,
        run_id=run_id or state.run_id,
        attempts=attempts,
        max_attempts=state.max_attempts,
        choice_fingerprints=choices,
        events=events,
    )


def run_build_recovery(
    ws,
    brief: str,
    build,
    *,
    progress=None,
    run_id: str | None = None,
    client=None,
    core_defaults=None,
    auto_default_questions: bool | None = None,
    redrive=None,
    max_attempts: int = BUILD_RECOVERY_MAX_ATTEMPTS,
) -> dict:
    """Build once, then apply the shared bounded build-to-design policy.

    ``build`` is the caller's deterministic build worker (web build queue or the
    batch runner); it returns the build exit code and writes the durable
    evidence. This function owns the design revision: it may re-drive the
    smallest owning stage through ``run_session`` (or the caller's ``redrive``),
    invalidating only that stage's downstream, and rebuilds. It never edits
    geometry, never relaxes a gate, and never restarts the budget.

    Returns ``{rc, attempts, max_attempts, status, failure_kind, events,
    needs_input, questions}`` where status is ``ok`` (a build completed),
    ``recovered`` (a revision rebuilt successfully), ``blocked``,
    ``exhausted``, or ``awaiting_input``.
    """
    ws = Path(ws)
    state = read_build_recovery(ws)
    max_attempts = int(state.max_attempts) if state.present else int(max_attempts)
    attempts = state.attempts
    seen_failures = state.failure_fingerprints
    seen_choices = set(state.choice_fingerprints)
    rc = build()
    if rc == 0:
        write_build_recovery(
            ws,
            ok=True,
            run_id=run_id,
            attempts=attempts,
            max_attempts=max_attempts,
            choice_fingerprints=list(seen_choices),
            events=state.events,
        )
        return {
            "rc": rc,
            "attempts": attempts,
            "max_attempts": max_attempts,
            "status": "ok" if attempts == 0 else "recovered",
            "failure_kind": None,
            "events": state.events,
            "needs_input": False,
            "questions": [],
        }

    while True:
        failure = classify_build_failure(ws, rc=rc)
        if failure.kind in ("infrastructure", "unattributable"):
            state = _record_recovery_event(
                ws, failure=failure, action="none", outcome="blocked",
                choice_fingerprint=None, run_id=run_id, ok=False,
                failure_kind=failure.kind, attempts=attempts, state=state,
            )
            _emit_recovery_event(
                progress, action="none", reason=failure.reason, outcome="blocked", failure=failure
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "blocked", "failure_kind": failure.kind,
                "events": state.events, "needs_input": False, "questions": [],
            }
        if failure.kind in ("compiler_defect", "capability_gap"):
            state = _record_recovery_event(
                ws, failure=failure, action="none", outcome="exhausted",
                choice_fingerprint=None, run_id=run_id, ok=False,
                failure_kind=failure.kind, attempts=attempts, state=state,
            )
            _emit_recovery_event(
                progress, action="none", reason=failure.reason, outcome="exhausted",
                requirement_ids=failure.requirement_ids, evidence=failure.evidence, failure=failure,
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "exhausted", "failure_kind": failure.kind,
                "events": state.events, "needs_input": False, "questions": [],
            }
        if attempts >= max_attempts:
            reason = f"recovery budget exhausted ({attempts}/{max_attempts})"
            state = _record_recovery_event(
                ws, failure=failure, action=failure.action, outcome="exhausted",
                choice_fingerprint=None, run_id=run_id, ok=False,
                failure_kind="recovery_exhausted", attempts=attempts, state=state,
            )
            _emit_recovery_event(
                progress, action=failure.action, reason=reason, outcome="exhausted",
                evidence=failure.evidence, failure=failure,
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "exhausted", "failure_kind": "recovery_exhausted",
                "events": state.events, "needs_input": False, "questions": [],
            }
        if failure.fingerprint in seen_failures:
            reason = "the same build failure repeated; no identical request is re-issued"
            state = _record_recovery_event(
                ws, failure=failure, action=failure.action, outcome="exhausted",
                choice_fingerprint=None, run_id=run_id, ok=False,
                failure_kind="recovery_repeated", attempts=attempts, state=state,
            )
            _emit_recovery_event(
                progress, action=failure.action, reason=reason, outcome="exhausted",
                evidence=failure.evidence, failure=failure,
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "exhausted", "failure_kind": "recovery_repeated",
                "events": state.events, "needs_input": False, "questions": [],
            }
        if not _budget_allows_recovery(client):
            reason = "remaining project/global budget cannot cover another design revision"
            state = _record_recovery_event(
                ws, failure=failure, action=failure.action, outcome="exhausted",
                choice_fingerprint=None, run_id=run_id, ok=False,
                failure_kind="budget_refused", attempts=attempts, state=state,
            )
            _emit_recovery_event(
                progress, action=failure.action, reason=reason, outcome="exhausted",
                evidence=failure.evidence, failure=failure,
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "exhausted", "failure_kind": "budget_refused",
                "events": state.events, "needs_input": False, "questions": [],
            }

        owner = failure.owner_stage
        invalidate_from, stages = _RECOVERY_PLAN.get(failure.action, (None, ()))
        if failure.action == "try_reviewed_alternative" and owner == "architecture":
            invalidate_from, stages = _ARCHITECTURE_ALTERNATIVE
        if not stages:
            state = _record_recovery_event(
                ws, failure=failure, action="none", outcome="blocked",
                choice_fingerprint=None, run_id=run_id, ok=False,
                failure_kind="unattributable", attempts=attempts, state=state,
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "blocked", "failure_kind": "unattributable",
                "events": state.events, "needs_input": False, "questions": [],
            }

        choice_before = _owner_choice_fingerprint(ws, owner)
        if choice_before is not None and choice_before in seen_choices:
            reason = (
                "this choice was already tried and did not resolve the failure "
                "(oscillation); terminating instead of repeating it"
            )
            state = _record_recovery_event(
                ws, failure=failure, action=failure.action, outcome="exhausted",
                choice_fingerprint=choice_before, run_id=run_id, ok=False,
                failure_kind="recovery_oscillation", attempts=attempts, state=state,
            )
            _emit_recovery_event(
                progress, action=failure.action, reason=reason, outcome="exhausted",
                evidence=failure.evidence, failure=failure,
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "exhausted", "failure_kind": "recovery_oscillation",
                "events": state.events, "needs_input": False, "questions": [],
            }
        # The choice we are about to leave is recorded as tried-and-insufficient,
        # so returning to it later (A -> B -> A) is refused rather than repeated.
        if choice_before is not None:
            seen_choices.add(choice_before)

        # The last accepted design stays on disk until a revision validates: the
        # revision is probed in place (the stage driver only commits a candidate
        # that passed its gates), and any non-committing outcome restores the
        # snapshot. Reuses the existing state.json commit mechanism -- no second
        # transaction store.
        snapshot = _snapshot_state(ws)
        try:
            null_downstream(ws, invalidate_from)
        except RuntimeError as exc:
            state = _record_recovery_event(
                ws, failure=failure, action=failure.action, outcome="blocked",
                choice_fingerprint=choice_before, run_id=run_id, ok=False,
                failure_kind="state_unreadable", attempts=attempts, state=state,
            )
            _emit_recovery_event(
                progress, action=failure.action, reason=str(exc), outcome="blocked",
                evidence=failure.evidence, failure=failure,
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "blocked", "failure_kind": "state_unreadable",
                "events": state.events, "needs_input": False, "questions": [],
            }

        instruction = _recovery_instruction(failure)
        _emit_recovery_event(
            progress, action=failure.action, reason=failure.reason, outcome="applied",
            requirement_ids=failure.requirement_ids, evidence=failure.evidence, failure=failure,
        )
        if redrive is not None:
            result = redrive(list(stages), instruction)
        else:
            result = run_session(
                ws,
                brief,
                list(stages),
                instruction=instruction,
                client=client,
                progress=progress,
                run_id=run_id,
                core_defaults=core_defaults,
                auto_default_questions=auto_default_questions,
            )
        status = (result or {}).get("status")
        if status == "awaiting_input":
            _restore_state(ws, snapshot)
            state = _record_recovery_event(
                ws, failure=failure, action=failure.action, outcome="blocked",
                choice_fingerprint=choice_before, run_id=run_id, ok=False,
                failure_kind="awaiting_input", attempts=attempts, state=state,
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "awaiting_input", "failure_kind": "awaiting_input",
                "events": state.events, "needs_input": True,
                "questions": (result or {}).get("questions") or [],
            }
        if status != "ok":
            _restore_state(ws, snapshot)
            state = _record_recovery_event(
                ws, failure=failure, action=failure.action, outcome="blocked",
                choice_fingerprint=choice_before, run_id=run_id, ok=False,
                failure_kind=(result or {}).get("failure_kind") or "redrive_failed",
                attempts=attempts, state=state,
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "blocked",
                "failure_kind": (result or {}).get("failure_kind") or "redrive_failed",
                "events": state.events, "needs_input": False, "questions": [],
            }

        attempts += 1
        choice_after = _owner_choice_fingerprint(ws, owner)
        seen_failures.add(failure.fingerprint)
        state = _record_recovery_event(
            ws, failure=failure, action=failure.action, outcome="applied",
            choice_fingerprint=choice_after, run_id=run_id, ok=False,
            failure_kind=None, attempts=attempts, state=state,
            choice_fingerprints=list(seen_choices),
        )
        if choice_after is not None and choice_after == choice_before:
            reason = (
                "the owning choice did not change after the repair; "
                "stopping instead of rebuilding an unchanged candidate"
            )
            state = _record_recovery_event(
                ws, failure=failure, action=failure.action, outcome="exhausted",
                choice_fingerprint=choice_after, run_id=run_id, ok=False,
                failure_kind="recovery_stalled", attempts=attempts, state=state,
            )
            _emit_recovery_event(
                progress, action=failure.action, reason=reason, outcome="exhausted",
                evidence=failure.evidence, failure=failure,
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "exhausted", "failure_kind": "recovery_stalled",
                "events": state.events, "needs_input": False, "questions": [],
            }
        rc = build()
        if rc == 0:
            write_build_recovery(
                ws, ok=True, run_id=run_id, attempts=attempts,
                max_attempts=max_attempts, choice_fingerprints=list(seen_choices),
                events=state.events,
            )
            _emit_recovery_event(
                progress, action=failure.action,
                reason=f"recovered: {failure.reason}", outcome="applied",
                requirement_ids=failure.requirement_ids, failure=failure,
            )
            return {
                "rc": rc, "attempts": attempts, "max_attempts": max_attempts,
                "status": "recovered", "failure_kind": None,
                "events": state.events, "needs_input": False, "questions": [],
            }

