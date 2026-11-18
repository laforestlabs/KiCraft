"""Frozen Surprise Me product-path campaigns, separate from the stage corpus."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import time
import uuid

from kicraft.cli.web_cost_report import load_rows, load_stage_attempts
from kicraft.design.lowering import lowerer_summaries
from kicraft.design.recipes import recipe_summaries
from kicraft.design.stage_semantics import DETECTOR_VERSION
from kicraft.fsutil import atomic_write_text
from kicraft.server.client import make_client
from kicraft.server.config import Settings
from kicraft.server.examples import BRIEF_SEED_MODULO, BRIEF_TEMPLATES, generate_brief
from kicraft.server.session import run_session
from kicraft.server.stage_contracts import source_obligation_rows, validate_obligation_retention

STAGES = ("intent", "functional_spec", "architecture")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _save(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _family(seed):
    return random.Random(seed).choice(BRIEF_TEMPLATES)[0]


def freeze_smoke_cohort(original_briefs=None):
    """Freeze the original project briefs and two previously unrepresented draws."""
    if original_briefs is None:
        originals = Path.home() / ".kicraft" / "projects" / "1"
        original_briefs = [(originals / str(project) / "brief.txt").read_text(encoding="utf-8")
                           for project in (1169, 1170, 1171)]
    if len(original_briefs) != 3:
        raise ValueError("smoke cohort requires all three original briefs")
    seeds = [82, 83, 84]
    families = {_family(seed) for seed in seeds}
    seed = 86
    while len(seeds) < 5:
        family = _family(seed)
        if family not in families:
            seeds.append(seed)
            families.add(family)
        seed += 1
    briefs = list(original_briefs) + [generate_brief(seed) for seed in seeds[3:]]
    return [{"seed": seed, "family": _family(seed), "brief": brief,
             "brief_sha256": hashlib.sha256(brief.encode()).hexdigest()}
            for seed, brief in zip(seeds, briefs)]


def _core_defaults(settings):
    if not settings.enable_core_defaults:
        return None
    uri = settings.users_db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(
            "SELECT * FROM core_components WHERE enabled=1 ORDER BY category, sort_order, id")]
        for row in rows:
            row["enabled"] = bool(row["enabled"])
        return rows


def _identity(settings):
    root = Path(__file__).resolve().parents[2]
    source = {}
    for base in (root / "kicraft", root / ".agents" / "skills" / "kicraft"):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".md", ".yaml", ".yml", ".txt"}:
                source[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Positive allowlist: adding a future credential to Settings must never
    # silently publish it in a campaign manifest.
    approved = {
        "model", "design_profile", "escalation_profile", "provider_fallback_profile",
        "backend", "max_tokens_per_call", "daily_usd_ceiling", "total_usd_ceiling",
        "project_llm_budget_usd", "kill_switch", "request_timeout_s",
        "llm_max_retries", "llm_retry_backoff_s", "design_temperature",
        "serialization_retries", "serialization_escape_temperature",
        "serialization_max_tokens", "stage_output_limits", "collection_bounds",
        "stage_semantics", "contract_ladder", "design_reasoning_tokens",
        "reasoning_max_tokens", "reasoning_repeat_window", "reasoning_repeat_threshold",
        "provider_order", "provider_allow_fallbacks", "max_price_prompt",
        "max_price_completion", "enable_prompt_cache", "enable_core_defaults",
        "enable_draft_audit", "draft_audit_model", "draft_audit_confidence",
        "enable_name_reconciliation", "enable_electrical_review", "review_model",
        "review_reasoning_tokens", "review_reasoning_effort", "review_max_tokens",
        "review_provider_order", "review_max_price_prompt", "review_max_price_completion",
        "review_corroboration", "review_temperature",
    }
    public = {key: value for key, value in asdict(settings).items() if key in approved}
    defaults = _core_defaults(settings)
    return {"source_sha256": _hash(source), "settings": public,
            "provider_endpoint_sha256": hashlib.sha256(settings.base_url.encode()).hexdigest(),
            "core_defaults": defaults, "core_defaults_sha256": _hash(defaults),
            "recipes": recipe_summaries(), "lowerers": lowerer_summaries(),
            "detector_version": DETECTOR_VERSION}


def freeze_campaign(out, *, count=300, seed=20261118, parallel=3, budget_usd=10.0,
                    settings=None, carryover=None):
    if count < 1 or parallel < 1 or parallel > 3 or not math.isfinite(budget_usd) or not 0 < budget_usd <= 35:
        raise ValueError("invalid campaign size, parallelism or budget")
    settings = settings or Settings.from_env()
    if not 0 < settings.project_llm_budget_usd <= .60 or not 0 < settings.daily_usd_ceiling <= 20 or not 0 < settings.total_usd_ceiling <= 250:
        raise ValueError("production spending caps must be enabled and no greater than .60/20/250")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("campaign already frozen")
    carried_entries = {}
    carried_from = None
    if carryover is not None:
        prior = Path(carryover)
        with (prior / "runner.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("cannot carry over an active campaign") from exc
            old_manifest = json.loads((prior / "manifest.json").read_text())
            with _ledger(prior) as old_ledger:
                if old_ledger.get("sealed") or old_ledger["manifest_sha256"] != _hash(old_manifest):
                    raise ValueError("prior campaign sealed or manifest mismatch")
                carried_from = {"manifest_sha256": _hash(old_manifest), "ledger_sha256": _hash(old_ledger)}
                for key, entry in old_ledger["entries"].items():
                    carried_entries[f"carryover:{old_manifest['campaign_id']}:{key}"] = {
                        "kind": "carryover", "status": entry["status"],
                        "amount_usd": entry["amount_usd"], "source_kind": entry["kind"],
                        "source_campaign_id": old_manifest["campaign_id"], "source_entry_id": key,
                        "source_entry": entry}
                if sum(e["amount_usd"] for e in carried_entries.values()) > budget_usd + 1e-9:
                    raise ValueError("carried exposure exceeds campaign budget")
                old_ledger["sealed"] = True
    rng = random.Random(seed)
    trials = []
    for index in range(count):
        trial_seed = rng.randrange(BRIEF_SEED_MODULO)
        brief = generate_brief(trial_seed)
        trials.append({"id": f"trial-{index:03d}", "seed": trial_seed, "family": _family(trial_seed),
                       "brief": brief, "brief_sha256": hashlib.sha256(brief.encode()).hexdigest()})
    manifest = {"schema_version": 1, "campaign_id": uuid.uuid4().hex, "seed": seed,
                "count": count, "parallel": parallel, "budget_usd": budget_usd,
                "identity": _identity(settings), "trials": trials, "smoke_cohort": freeze_smoke_cohort(),
                "carryover": carried_from}
    for trial in trials:
        path = out / "trials" / trial["id"] / "brief.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, trial["brief"])
    _save(out / "manifest.json", manifest)
    _save(out / "ledger.json", {"manifest_sha256": _hash(manifest), "entries": carried_entries})
    return manifest


@contextmanager
def _ledger(out):
    out = Path(out)
    with (out / "ledger.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = json.loads((out / "ledger.json").read_text())
        yield doc
        _save(out / "ledger.json", doc)


def reserve_spend(out, entry_id, amount, *, kind="preparation"):
    """Reserve maximum exposure before starting a preparation or trial run."""
    manifest = json.loads((Path(out) / "manifest.json").read_text())
    amount = float(amount)
    if not math.isfinite(amount) or amount <= 0 or amount > manifest["identity"]["settings"]["project_llm_budget_usd"]:
        raise ValueError("invalid reservation")
    with _ledger(out) as ledger:
        if ledger.get("sealed") or ledger["manifest_sha256"] != _hash(manifest) or entry_id in ledger["entries"]:
            raise ValueError("manifest mismatch or duplicate reservation")
        exposure = sum(float(e["amount_usd"]) for e in ledger["entries"].values())
        if exposure + amount > manifest["budget_usd"] + 1e-9:
            return False
        ledger["entries"][entry_id] = {"kind": kind, "status": "reserved", "amount_usd": amount}
    return True


def settle_spend(out, entry_id, actual_usd):
    """Settle a reserved run with its authoritative ledger cost; never reuse its ID."""
    actual_usd = float(actual_usd)
    with _ledger(out) as ledger:
        entry = ledger["entries"][entry_id]
        if ledger.get("sealed"):
            raise ValueError("campaign has been sealed for carryover")
        if entry["status"] != "reserved" or not math.isfinite(actual_usd) or not 0 <= actual_usd <= entry["amount_usd"] + 1e-9:
            raise ValueError("invalid settlement")
        entry.update(status="settled", amount_usd=actual_usd)


def _attempts_cover_receipts(calls, attempts):
    """A duplicate outcome cannot stand in for a different billed POST."""
    def key(row):
        return tuple(row.get(field) for field in ("stage", "attempt", "call_mode", "model"))

    receipts = Counter()
    for call in calls:
        meta = call["meta"]
        if meta.get("phase") == "decision_not_dispatched":
            continue
        identity = key({**meta, "model": call.get("model")})
        if any(value is None for value in identity):
            return False
        receipts[identity] += 1
    outcomes = Counter(key(attempt) for attempt in attempts)
    return bool(receipts) and all(outcomes[identity] >= count for identity, count in receipts.items())


def _run_trial(out, trial, defaults, ledger_path, *, settings=None):
    workspace = Path(out) / "work" / trial["id"]
    workspace.mkdir(parents=True, exist_ok=False)
    run_id = f"surprise-{Path(out).name}-{trial['id']}-{uuid.uuid4().hex}"
    events = workspace / "events.jsonl"
    frozen_brief = (Path(out) / "trials" / trial["id"] / "brief.txt").read_text(encoding="utf-8")
    if frozen_brief != trial["brief"] or hashlib.sha256(frozen_brief.encode()).hexdigest() != trial["brief_sha256"]:
        raise ValueError("frozen brief identity mismatch")
    atomic_write_text(workspace / "brief.txt", frozen_brief)
    prompt_sections = []
    start = time.monotonic()
    def progress(event):
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, default=str) + "\n")
        if event.get("kind") == "stage_done" and event.get("stage") in STAGES:
            state_path = workspace / ".kicraft" / "state.json"
            if state_path.is_file():
                snapshot = workspace / "snapshots" / (event["stage"] + ".json")
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(snapshot, state_path.read_text(encoding="utf-8"))
        if event.get("kind") in {"prompt_sections", "prompt_size"}:
            prompt_sections.append(event)
    result = None
    try:
        result = run_session(workspace, frozen_brief, STAGES, progress=progress, run_id=run_id,
                             core_defaults=defaults, auto_default_questions=True,
                             client=make_client(settings or Settings.from_env()))
        failure = result.get("failure_kind") or (None if result.get("status") == "ok" else result.get("status"))
    except Exception:
        failure = "runner_exception"
    calls = [r for r in load_rows(ledger_path) if r["meta"].get("run_id") == run_id]
    provider_receipts = [r for r in calls if r["meta"].get("phase") != "decision_not_dispatched"]
    attempts = [r for r in load_stage_attempts(ledger_path) if r.get("run_id") == run_id]
    cost = sum(r["cost_usd"] for r in calls)
    with sqlite3.connect(Path(ledger_path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        exposure_rows = [row[0] for row in conn.execute(
            "SELECT ceiling_usd FROM provider_exposure WHERE run_id=?", (run_id,))]
    valid_exposure = bool(exposure_rows) and all(
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        for value in exposure_rows
    )
    exposure = sum(exposure_rows) if valid_exposure else (0 if not exposure_rows else None)
    valid_receipts = all(
        isinstance(r["cost_usd"], (int, float)) and math.isfinite(r["cost_usd"])
        and r["cost_usd"] >= 0 for r in calls
    )
    outstanding_bound = cost + exposure if valid_receipts and valid_exposure else None
    stages = {r.get("stage"): r for r in (result or {}).get("results", [])}
    committed = all(stages.get(stage, {}).get("commit_ok") for stage in STAGES)
    semantic_clean = all(
        stages.get(stage, {}).get("semantic_clean") is True
        and not stages.get(stage, {}).get("repair_required")
        for stage in STAGES
    )
    retained = False
    if committed:
        try:
            state = json.loads((workspace / ".kicraft" / "state.json").read_text(encoding="utf-8"))
            for stage, source_slots in (("functional_spec", ("intent",)), ("architecture", ("intent", "functional_spec"))):
                payload = state[stage]
                validate_obligation_retention(stage, payload, state)
                expected = source_obligation_rows(state, slots=source_slots)
                actual = source_obligation_rows(state, slots=(stage,))
                if _hash(expected) != _hash(actual):
                    raise ValueError("committed obligation rows differ from source")
            retained = True
        except (ValueError, KeyError, OSError, TypeError):
            retained = False
    provider_attempts = [a for a in attempts if a.get("call_mode") != "deterministic_commit"]
    logged = (bool(provider_receipts) and exposure == 0 and valid_receipts and bool(stages) and _attempts_cover_receipts(calls, provider_attempts)
              and all(any(a.get("stage") == stage for a in provider_attempts) for stage in stages)
              and (workspace / ".kicraft" / "state.json").is_file() and events.is_file())
    architecture = bool(stages.get("architecture", {}).get("commit_ok"))
    return {"id": trial["id"], "run_id": run_id, "family": trial["family"],
            "status": "finished", "complete_path": bool(committed and semantic_clean and retained and logged and not failure),
            "architecture_committed": architecture, "architecture_attempted": "architecture" in stages,
            "failure_kind": failure or ("semantic_rejected" if committed and not semantic_clean else None) or ("obligation_retention" if committed and not retained else None) or (None if logged else "missing_attempt_evidence"),
            "obligations_retained": retained,
            "semantic_clean": semantic_clean,
            "stage_outcomes": {stage: {"commit_ok": bool(row.get("commit_ok")),
                                         "failure_kind": row.get("failure_kind"),
                                         "semantic_clean": row.get("semantic_clean"),
                                         "repair_required": row.get("repair_required"),
                                         "diagnostics": row.get("diagnostics"),
                                         "diagnostic": row.get("diagnostic")}
                               for stage, row in stages.items()},
            "attempt_count": len(provider_attempts),
            "architecture_provider_calls": sum(a.get("stage") == "architecture" for a in provider_attempts),
            "architecture_attempts": sum(
                a.get("stage") == "architecture"
                and a.get("call_mode") not in {"draft_audit", "name_reconciliation"}
                for a in provider_attempts
            ),
            "cost_receipted": bool(provider_receipts) and exposure == 0 and valid_receipts,
            "outstanding_bound_usd": outstanding_bound,
            "provider_exposure_usd": exposure,
            "attempt_ids": [{"stage": a.get("stage"), "attempt": a.get("attempt"),
                             "request_id": a.get("request_id"), "outcome": a.get("outcome"),
                             "diagnostic_codes": a.get("diagnostic_codes"),
                             "commit_gate_codes": a.get("commit_gate_codes")} for a in attempts],
            "cost_usd": cost,
            "providers": sorted({str(r.get("provider")) for r in calls}),
            "models": sorted({str(r.get("model")) for r in calls}),
            "prompt_sections": prompt_sections if prompt_sections else None,
            "input_tokens": sum(r["input_tokens"] for r in calls), "wall_s": time.monotonic() - start}


def campaign_report(manifest, ledger):
    from kicraft.eval.stage_reliability import exact_zero_failure_lower_bound

    entries = ledger["entries"]
    rows = [entries.get(t["id"], {}).get("result", {"id": t["id"], "family": t["family"], "status": "missing"}) for t in manifest["trials"]]
    successful = []
    failures = Counter()
    for row in rows:
        if row.get("complete_path") is True and row.get("semantic_clean") is True:
            successful.append(row)
            continue
        failure = row.get("failure_kind")
        if not failure and row.get("complete_path"):
            failure = "semantic_rejected" if row.get("semantic_clean") is False else "missing_semantic_evidence"
        failures[failure or row.get("status") or "missing"] += 1
    passed = len(successful)
    attempted = [r for r in rows if r.get("architecture_attempted")]
    def percentile(key, fraction):
        values = sorted(r[key] for r in rows if isinstance(r.get(key), (int, float))
                        and (key != "cost_usd" or r.get("cost_receipted") is True))
        return values[math.ceil((len(values)-1)*fraction)] if values else None
    return {"count": len(rows), "complete_path_passes": passed,
            "settled_spend_usd": sum(e["amount_usd"] for e in entries.values() if e["status"] == "settled"),
            "reserved_exposure_usd": sum(e["amount_usd"] for e in entries.values() if e["status"] == "reserved"),
            "preparation_and_carryover_usd": sum(e["amount_usd"] for e in entries.values() if e["kind"] != "trial"),
            "campaign_exposure_usd": sum(e["amount_usd"] for e in entries.values()),
            "campaign_budget_usd": manifest.get("budget_usd"),
            "architecture_attempted": len(attempted), "architecture_passes": sum(bool(r.get("architecture_committed")) for r in attempted),
            "release_passed": len(rows) == 300 and passed == 300,
            "zero_failure_lower_bound": exact_zero_failure_lower_bound(len(rows)) if passed == len(rows) and rows else None,
            "failures": dict(failures),
            "families": dict(Counter(r["family"] for r in rows)),
            "first_call_success": sum(r.get("architecture_attempts") == 1 for r in successful),
            "repaired_success": sum(r.get("architecture_attempts", 0) > 1 for r in successful),
            "p50_cost_usd": percentile("cost_usd", .5), "p95_cost_usd": percentile("cost_usd", .95),
            "p50_wall_s": percentile("wall_s", .5), "p95_wall_s": percentile("wall_s", .95),
            "rows": rows}


def _resume_campaign_locked(out, *, settings=None):
    out = Path(out)
    settings = settings or Settings.from_env()
    manifest = json.loads((out / "manifest.json").read_text())
    if _hash(_identity(settings)) != _hash(manifest["identity"]):
        raise ValueError("campaign code, settings, defaults or catalog identity drift")
    with _ledger(out) as ledger:
        if ledger.get("sealed") or ledger["manifest_sha256"] != _hash(manifest):
            raise ValueError("campaign manifest identity mismatch")
        # A previously started interrupted run is a failure, not an unstarted trial.
        for trial in manifest["trials"]:
            entry = ledger["entries"].get(trial["id"])
            if entry and entry["status"] == "reserved" and "result" not in entry:
                entry["result"] = {"id": trial["id"], "family": trial["family"], "status": "interrupted", "failure_kind": "interrupted"}
        _save(out / "report.json", campaign_report(manifest, ledger))
    with ThreadPoolExecutor(max_workers=manifest["parallel"]) as pool:
        pending = {}
        next_index = 0
        exhausted = False
        while pending or not exhausted:
            while len(pending) < manifest["parallel"] and not exhausted:
                if next_index == len(manifest["trials"]):
                    exhausted = True
                    break
                trial = manifest["trials"][next_index]
                with _ledger(out) as ledger:
                    already = trial["id"] in ledger["entries"]
                if already:
                    next_index += 1
                    continue
                amount = settings.project_llm_budget_usd
                if not reserve_spend(out, trial["id"], amount, kind="trial"):
                    exhausted = not pending
                    break
                next_index += 1
                pending[pool.submit(
                    _run_trial, out, trial, manifest["identity"]["core_defaults"],
                    settings.ledger_path, settings=settings,
                )] = trial
            if pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    trial = pending.pop(future)
                    try:
                        row = future.result()
                        if row["cost_receipted"] and (not math.isfinite(row["cost_usd"]) or
                                                       not 0 <= row["cost_usd"] <= settings.project_llm_budget_usd + 1e-9):
                            raise ValueError("unusable cost receipt")
                    except Exception:
                        row = {"id": trial["id"], "family": trial["family"], "status": "failed", "failure_kind": "missing_cost_or_evidence"}
                    with _ledger(out) as ledger:
                        if row.get("cost_receipted"):
                            ledger["entries"][trial["id"]].update(status="settled", amount_usd=row["cost_usd"])
                        else:
                            row["cost_receipted"] = False
                            row["complete_path"] = False
                            row["failure_kind"] = row.get("failure_kind") or "missing_cost_or_evidence"
                            bound = row.get("outstanding_bound_usd")
                            entry = ledger["entries"][trial["id"]]
                            if (isinstance(bound, (int, float)) and not isinstance(bound, bool)
                                    and math.isfinite(bound) and 0 < bound <= entry["amount_usd"]):
                                entry["amount_usd"] = bound
                        ledger["entries"][trial["id"]]["result"] = row
                        _save(out / "report.json", campaign_report(manifest, ledger))
    with _ledger(out) as ledger:
        report = campaign_report(manifest, ledger)
    _save(out / "report.json", report)
    return report


def resume_campaign(out, *, settings=None):
    # Do not let a second process mark live reservations as interrupted.
    with (Path(out) / "runner.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("campaign runner already active") from exc
        return _resume_campaign_locked(out, settings=settings)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze or resume a Surprise Me reliability campaign")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--campaign-budget-usd", type=float, default=10)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--carryover", type=Path, help="seal and carry prior campaign exposure")
    args = parser.parse_args(argv)
    if args.resume:
        if args.out or args.freeze_only or args.carryover:
            parser.error("--resume cannot be combined with --out or --freeze-only")
        report = resume_campaign(args.resume)
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
        return 0 if report["release_passed"] else 1
    if not args.out or not args.freeze_only:
        parser.error("new campaigns require --out and --freeze-only; resume separately")
    manifest = freeze_campaign(args.out, count=args.count, seed=args.seed, parallel=args.parallel,
                               budget_usd=args.campaign_budget_usd, carryover=args.carryover)
    print(f"frozen {len(manifest['trials'])} trials sha256={_hash(manifest)}")
    return 0
