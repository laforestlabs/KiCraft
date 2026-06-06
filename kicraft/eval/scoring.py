"""Source-agnostic scoring: Class-C dimension scorers, script gates, and the
finalize math (weighted total, gate caps, grade band).

Everything here operates on a generic ``m`` metrics dict and the parsed rubric,
never on files directly. That is what lets one implementation serve both the
offline harness (which builds ``m`` from a harvested ``claude`` transcript) and
the web app (which builds the same ``m`` shape from its own ``events.jsonl`` +
artifact tree). A front-end supplies ``m`` with these sub-dicts:

    state, synth, erc, generated, perm, transcript, latency, token_usage,
    expected_question_band   (see kicraft.eval.artifacts + each collector)

The ``transcript`` sub-dict is a logical run-trace, not literally a ``claude``
transcript: the web collector synthesises one (present / crashes / failed_commits
/ ask_questions / synth_attempts) from its event stream so these scorers run
unchanged.
"""
from __future__ import annotations

import datetime as dt

CANONICAL_STAGES = 5  # intent, functional_spec, architecture, bom, wiring


# --------------------------------------------------------------------------- #
# time helpers (shared by the metrics collectors)
# --------------------------------------------------------------------------- #
def _parse_ts(s: str | None):
    if not s:
        return None
    s = s.strip()
    try:
        if s.endswith("Z") and "T" in s and "-" not in s.split("T")[0][4:]:
            # compact UTC like 20260524T225920Z
            return dt.datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_latency_min(transcript: dict, state: dict, synth: dict) -> tuple[float | None, bool]:
    """Return (minutes, is_approximate). Prefer transcript (consistent tz)."""
    if transcript.get("present"):
        a = _parse_ts(transcript.get("first_ts"))
        b = _parse_ts(transcript.get("synth_ts") or transcript.get("last_ts"))
        if a and b and b > a:
            return round((b - a).total_seconds() / 60, 1), False
    # fallback: history start -> synth checked_at (tz-mismatched -> approximate)
    a = _parse_ts(state.get("history_first_ts"))
    b = _parse_ts(synth.get("checked_at"))
    if a and b:
        if a.tzinfo is None:
            a = a.replace(tzinfo=dt.timezone.utc)
        if b.tzinfo is None:
            b = b.replace(tzinfo=dt.timezone.utc)
        mins = (b - a).total_seconds() / 60
        if mins >= 0:
            return round(mins, 1), True
    return None, True


# --------------------------------------------------------------------------- #
# Class-C dimension scorers  ->  (level|None, partial, rationale)
# --------------------------------------------------------------------------- #
def score_pipeline_completion(m) -> tuple[int | None, bool, str]:
    st = m["state"]
    if not st.get("present"):
        return 0, False, "no state.json"
    slots = st["slots"]
    if not any(slots.values()) or (slots.get("intent") and sum(slots.values()) == 1):
        return 0, False, "no slots beyond intent"
    if not (st["all_slots"] and st["wiring_done"]):
        return 1, False, "incomplete: missing a slot or bom.connections"
    if not m["generated"]["synthesized"]:
        return 2, False, "all slots + wiring, but not synthesized"
    status = m["synth"].get("status")
    if status != "ok":
        return 3, False, f"synthesized but synthesis_check.status={status!r}"
    return 4, False, "synthesized, status ok, files present"


def score_computing_cleanliness(m) -> tuple[int | None, bool, str]:
    erc = m["erc"]
    synth = m["synth"]
    tr = m["transcript"]
    errors = erc.get("errors")
    warnings = erc.get("warnings")
    failed = synth.get("failed_count")
    crashed = bool(tr.get("crashes")) if tr.get("present") else False

    if not m["generated"]["synthesized"]:
        if crashed:
            return 0, False, "synthesis-blocking crash (traceback in transcript)"
        return 2, True, "synthesis not reached; cleanliness unconfirmed (partial)"

    # synthesized: prefer ERC counts; fall back to synth_check failures
    if crashed or (errors is not None and errors > 10):
        return 0, False, f"crash={crashed}, erc_errors={errors}"
    if (errors is not None and errors >= 1) or (failed is not None and failed >= 2):
        return 1, False, f"erc_errors={errors}, failed_checks={failed}"
    if failed == 1:
        return 2, False, "exactly 1 failed synthesis check"
    if errors is None and failed is None:
        return 2, True, "synthesized but no ERC/check signal found (partial)"
    if (warnings or 0) > 0:
        return 3, False, f"clean errors/checks; {warnings} ERC warnings"
    return 4, False, "0 errors, 0 failed checks, 0 warnings"


def score_convergence(m) -> tuple[int | None, bool, str]:
    tr = m["transcript"]
    if tr.get("present"):
        err_recommits = tr.get("failed_commits", 0)
        level = {0: 4, 1: 3, 2: 2, 3: 1}.get(err_recommits, 0)
        return level, False, f"{err_recommits} failed/error-driven commit(s) in transcript"
    extra = max(0, m["state"].get("history_len", 0) - CANONICAL_STAGES)
    if extra == 0:
        return 4, True, "history==5 canonical stages; no transcript to confirm (partial)"
    level = max(0, 4 - extra)
    return level, True, f"{extra} extra history commit(s); cannot classify error vs user-driven without transcript (partial)"


def score_latency(m) -> tuple[int | None, bool, str]:
    mins, approx = m["latency"]
    if mins is None:
        return None, True, "no usable timestamps (transcript absent); unscored"
    # The fallback (history -> synth checked_at) is tz-mismatched and, on archived
    # multi-session records, can span days. Don't let an untrustworthy absolute
    # value drive the score: leave it for the observer to read off the transcript.
    if approx and mins > 60:
        return None, True, (f"fallback latency {mins} min implausible "
                            f"(tz-mismatch / multi-session archive); unscored, use transcript")
    for lvl, hi in ((4, 8), (3, 15), (2, 30), (1, 60)):
        if mins <= hi:
            return lvl, approx, f"{mins} min{' (approx, tz-mismatched fallback)' if approx else ''}"
    return 0, approx, f"{mins} min (>60)"


def score_friction(m) -> tuple[int | None, bool, str]:
    tr = m["transcript"]
    perm = m["perm"]
    band = m["expected_question_band"]  # (lo, hi) or None
    excess = perm["excess"]
    q = tr.get("ask_questions") if tr.get("present") else None

    # question component vs band
    q_state = "unknown"
    if q is not None and band is not None:
        lo, hi = band
        if lo <= q <= hi:
            q_state = "in_band"
        elif abs(q - lo) <= 1 or abs(q - hi) <= 1:
            q_state = "near_band"
        else:
            q_state = "out_of_band"

    partial = q is None or band is None
    # combine with permission excess
    if q_state == "out_of_band" and excess > 3:
        return 0, partial, f"questions out of band (asked {q}, band {band}) and {excess} excess prompts"
    if q_state == "out_of_band" or excess > 3:
        return 1, partial, f"q={q} band={band} ({q_state}); excess_prompts={excess}"
    if q_state == "near_band" or 2 <= excess <= 3:
        return 2, partial, f"q={q} band={band} ({q_state}); excess_prompts={excess}"
    if excess <= 1 and q_state in ("in_band", "unknown"):
        if q_state == "in_band" and excess == 0:
            return 4, partial, f"questions in band ({q}), zero excess prompts"
        return 3, partial, f"q={q} band={band} ({q_state}); excess_prompts={excess}"
    return 2, True, f"q={q} band={band} ({q_state}); excess_prompts={excess} (partial)"


CLASS_C_SCORERS = {
    "pipeline_completion": score_pipeline_completion,
    "computing_error_cleanliness": score_computing_cleanliness,
    "convergence_efficiency": score_convergence,
    "latency": score_latency,
    "interaction_friction": score_friction,
}


# --------------------------------------------------------------------------- #
# gates (script-detectable)
# --------------------------------------------------------------------------- #
def eval_script_gates(m, rubric) -> list[dict]:
    fired = []
    caps = {g["id"]: g["cap"] for g in rubric["gates"]}
    erc_errors = m["erc"].get("errors")
    if erc_errors is not None and erc_errors >= 1:
        fired.append({"id": "erc_errors", "cap": caps["erc_errors"], "by": "script",
                      "why": f"{erc_errors} ERC error(s)"})
    # synthesis_broken only on positive evidence of a failed attempt
    tr = m["transcript"]
    attempted = (tr.get("present") and tr.get("synth_attempts", 0) > 0)
    if attempted and not m["generated"]["synthesized"]:
        fired.append({"id": "synthesis_broken", "cap": caps["synthesis_broken"], "by": "script",
                      "why": "synthesize attempted (transcript) but no project files produced"})
    return fired


# --------------------------------------------------------------------------- #
# report assembly (shared by every front-end)
# --------------------------------------------------------------------------- #
def dim_by_id(rubric):
    return {d["id"]: d for d in rubric["dimensions"]}


def metrics_block(m) -> dict:
    """The schema ``metrics`` block from an ``m`` dict. Front-end-neutral: the web
    collector populates the same sub-dicts (with perm excess 0 and a synthesised
    transcript) so this maps identically for both paths."""
    tr, st, synth, erc = m["transcript"], m["state"], m["synth"], m["erc"]
    return {
        "synthesized": m["generated"]["synthesized"],
        "generated_files": m["generated"],
        "synthesis_status": synth.get("status"),
        "failed_checks": synth.get("failed_checks"),
        "erc_errors": erc.get("errors"),
        "erc_warnings": erc.get("warnings"),
        "latency_min": m["latency"][0],
        "latency_approx": m["latency"][1],
        "user_questions": tr.get("ask_questions") if tr.get("present") else None,
        "stage_commit_calls": tr.get("stage_commit_calls") if tr.get("present") else None,
        "failed_commits": tr.get("failed_commits") if tr.get("present") else None,
        "crashes": tr.get("crashes") if tr.get("present") else None,
        "history_len": st.get("history_len"),
        "open_questions": st.get("open_questions"),
        "bom_parts": st.get("bom_parts"),
        "permission_floor": m["perm"]["count"],
        "permission_excess": m["perm"]["excess"],
        "expected_question_band": m["expected_question_band"],
        "transcript_present": tr.get("present", False),
        "token_usage": m.get("token_usage"),
    }


def score_class_c_dims(m, rubric) -> dict:
    """Build the report ``dimensions`` map: Class-C scored by script, Class-J left
    null for an observer (human in the harness, the LLM judge on the web)."""
    report_dims = {}
    for d in rubric["dimensions"]:
        did = d["id"]
        if d["class"] == "C":
            level, partial, why = CLASS_C_SCORERS[did](m)
            report_dims[did] = {"class": "C", "weight": d["weight"], "level": level,
                                "partial": partial, "by": "script", "rationale": why}
        else:
            report_dims[did] = {"class": "J", "weight": d["weight"], "level": None,
                                "partial": False, "by": "observer", "rationale": ""}
    return report_dims


def grade_for(score: float, rubric) -> dict:
    for band in rubric["bands"]:
        if score >= band["min"]:
            return {"grade": band["grade"], "verdict": band["verdict"]}
    return {"grade": "F", "verdict": "BROKEN"}


def finalize_report(report: dict, rubric: dict) -> dict:
    """Total a fully-graded report in place: weighted = sum(weight*level/4), capped
    by the lowest triggered gate, mapped to a grade band. Raises ValueError if any
    dimension is still ungraded (so a partial judge result fails loudly)."""
    dims = report["dimensions"]
    missing = [k for k, v in dims.items() if v.get("level") is None]
    if missing:
        raise ValueError(f"cannot finalize: {len(missing)} dimension(s) ungraded: "
                         f"{', '.join(missing)}")
    points = sum(v["weight"] * v["level"] / 4 for v in dims.values())
    weighted = round(points, 1)

    fired = list(report.get("gates", {}).get("triggered", []))
    caps = [g["cap"] for g in fired]
    final = round(min([weighted] + caps), 1)
    g = grade_for(final, rubric)

    report["score"] = {
        "weighted": weighted,
        "final": final,
        "grade": g["grade"],
        "verdict": g["verdict"],
        "gates_applied": [{"id": x["id"], "cap": x["cap"]} for x in fired],
        "pending_dimensions": [],
    }
    report["finalized_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return report
