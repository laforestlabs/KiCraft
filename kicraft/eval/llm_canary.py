"""Guarded fixed-cohort production LLM canary runner."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

from kicraft.build_slots import host_cpu_count, slot_count, slots_dir
from kicraft.cli.model_preflight import preflight_role
from kicraft.server.config import DESIGN_PROFILES, Settings
from kicraft.server.session import DESIGN_STAGES
from kicraft.server.spend_guard import SpendGuard
from kicraft.tuning.benchmark import BENCHMARK_PROMPTS

COHORT = (
    "r2r-dac",
    "usb-a-power-splitter",
    "stm32-min",
    "nrf52-beacon",
    "dual-rail-supply",
    "encoder-oled-panel",
    "can-node",
    "servo-driver-16",
    "round-led-ring",
)
REFERENCE_BATCH = Path("/home/kicraft/.kicraft/self_eval/20260825T033602Z")
ENVELOPE_USD = 0.70
REPO_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _git(*args: str, text: bool = True):
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=text, stderr=subprocess.STDOUT, timeout=30
    )


def _checkout_identity() -> dict:
    commit = _git("rev-parse", "HEAD").strip()
    raw = _git("status", "--porcelain=v1", "-z", text=False)
    entries = [entry for entry in raw.split(b"\0") if entry]
    paths: list[str] = []
    for entry in entries:
        decoded = entry.decode("utf-8", "surrogateescape")
        path = decoded[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    paths = sorted(set(paths))
    if paths:
        raise RuntimeError(
            "checkout has uncommitted runtime/config changes: " + ", ".join(paths)
        )
    return {
        "commit": commit,
        "dirty_paths": [],
        "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _cohort() -> list[dict]:
    by_slug = {entry["slug"]: entry for entry in BENCHMARK_PROMPTS}
    if len(by_slug) != len(BENCHMARK_PROMPTS):
        raise RuntimeError("BENCHMARK_PROMPTS contains duplicate slugs")
    missing = [slug for slug in COHORT if slug not in by_slug]
    if missing:
        raise RuntimeError("fixed cohort missing from BENCHMARK_PROMPTS: " + ", ".join(missing))
    rows = []
    archetypes = set()
    for slug in COHORT:
        entry = by_slug[slug]
        archetype = entry["archetype"]
        if archetype in archetypes:
            raise RuntimeError(f"fixed cohort repeats archetype {archetype!r}")
        archetypes.add(archetype)
        rows.append(
            {
                "slug": slug,
                "archetype": archetype,
                "brief_sha256": _sha_bytes(entry["brief"].encode("utf-8")),
            }
        )
    if len(archetypes) != 9:
        raise RuntimeError("fixed cohort does not cover exactly nine archetypes")
    return rows


def _resolved_roles(settings: Settings) -> tuple[dict, dict]:
    name = settings.design_profile
    if name not in DESIGN_PROFILES:
        raise RuntimeError(f"designer profile {name!r} is not a named DESIGN_PROFILES entry")
    profile = DESIGN_PROFILES[name]
    expected = {
        "model": str(profile["model"]),
        "provider_order": list(profile["provider_order"]),
        "max_price_prompt": float(profile["max_price_prompt"]),
        "max_price_completion": float(profile["max_price_completion"]),
    }
    actual = {
        "model": settings.model,
        "provider_order": list(settings.provider_order),
        "max_price_prompt": settings.max_price_prompt,
        "max_price_completion": settings.max_price_completion,
    }
    if actual != expected:
        raise RuntimeError(f"resolved designer settings do not match profile {name!r}")
    judge_settings = settings.for_judge()
    if not settings.eval_judge_model or not judge_settings.provider_order:
        raise RuntimeError("judge model and provider route must resolve explicitly")
    if judge_settings.max_price_prompt <= 0 or judge_settings.max_price_completion <= 0:
        raise RuntimeError("judge price caps must be finite and positive")
    designer = {"profile": name, **actual}
    judge = {
        "model": settings.eval_judge_model,
        "provider_order": list(judge_settings.provider_order),
        "max_price_prompt": judge_settings.max_price_prompt,
        "max_price_completion": judge_settings.max_price_completion,
    }
    return designer, judge


def _policy_rows(settings: Settings) -> dict:
    rows = {}
    for stage in DESIGN_STAGES:
        policy = settings.design_stage_policy(stage, 4096)
        rows[stage] = {
            "normal_max_tokens": policy.normal_max_tokens,
            "normal_reasoning": policy.normal_reasoning,
            "serialization_max_tokens": policy.serialization_max_tokens,
            "serialization_retries": policy.serialization_retries,
            "collection_bounds": [
                {
                    "field": bound.field,
                    "total": bound.total,
                    "per_group": bound.per_group,
                    "group_key": bound.group_key,
                }
                for bound in policy.collection_bounds
            ],
            "reasoning_guard": (
                {
                    "name": policy.reasoning_guard.name,
                    "hard_max_tokens": policy.reasoning_guard.hard_max_tokens,
                    "repetition_enabled": policy.reasoning_guard.repetition_enabled,
                    "repeat_window": policy.reasoning_guard.repeat_window,
                    "repeat_threshold": policy.reasoning_guard.repeat_threshold,
                    "wall_stall_s": policy.reasoning_guard.wall_stall_s,
                }
                if policy.reasoning_guard
                else None
            ),
        }
    return rows


def probe_build_slots() -> list[int]:
    """Return occupied global slot indexes after a nonblocking all-slot probe."""
    count = slot_count()
    if count <= 0:
        return []
    directory = slots_dir()
    directory.mkdir(parents=True, exist_ok=True)
    acquired: list[tuple[int, int]] = []
    occupied: list[int] = []
    try:
        for index in range(count):
            fd = os.open(directory / f"slot_{index}.lock", os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                occupied.append(index)
            else:
                acquired.append((index, fd))
    finally:
        for _, fd in acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    return occupied


def _require_headroom(status: dict, required: float) -> None:
    if status.get("kill_switch"):
        raise RuntimeError("spend kill switch is engaged")
    if float(status.get("daily_remaining_usd", 0)) + 1e-9 < required:
        raise RuntimeError(f"daily spend headroom does not cover ${required:.2f} campaign envelope")
    if float(status.get("total_remaining_usd", 0)) + 1e-9 < required:
        raise RuntimeError(f"total spend headroom does not cover ${required:.2f} campaign envelope")


def _sanitize_preflight(result: dict) -> dict:
    forbidden = {"api_key", "authorization", "reply_head", "raw", "text", "reasoning"}

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key.lower() not in forbidden}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(result)


def _manifest_identity(settings: Settings, campaign_id: str) -> dict:
    designer, judge = _resolved_roles(settings)
    reference_summary = REFERENCE_BATCH / "summary.json"
    if not reference_summary.is_file():
        raise RuntimeError(f"reference summary missing: {reference_summary}")
    return {
        "campaign_id": campaign_id,
        "reference": {
            "batch": str(REFERENCE_BATCH),
            "summary_sha256": _sha_file(reference_summary),
        },
        "cohort": _cohort(),
        "checkout": _checkout_identity(),
        "host": {
            "cpu_count": host_cpu_count(),
            "global_build_slots": slot_count(),
            "canary_build_slots": 1,
            "exclusive_host": False,
        },
        "designer": designer,
        "judge": judge,
        "design_stage_policy": _policy_rows(settings),
        "judge_max_tokens": settings.eval_judge_max_tokens,
        "envelope_usd": ENVELOPE_USD,
    }


def _update_manifest(path: Path, *, status: str, error: dict | None = None, **fields) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(fields)
    payload["run_status"] = status
    payload["operational_error"] = error
    payload["updated_at"] = _now()
    _json_write(path, payload)
    return payload


def _typed_error(kind: str, exc: BaseException, settings: Settings | None = None) -> dict:
    message = str(exc)[:600]
    if settings and settings.api_key:
        message = message.replace(settings.api_key, "[REDACTED]")
    return {"kind": kind, "message": message}


def _batch_argv(batch: Path, *, resume: bool) -> list[str]:
    command = [sys.executable, "-m", "kicraft.eval.self_eval"]
    if resume:
        command.extend(["--resume", str(batch)])
    command.extend(
        [
            "--only",
            ",".join(COHORT),
            "--out",
            str(batch),
            "--repeats",
            "1",
            "--parallel",
            "1",
            "--build-slots",
            "1",
            "--build-timeout",
            "2400",
        ]
    )
    return command


def _tee_subprocess(argv: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            argv,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=15)
            raise


def _invoke_analysis(batch: Path) -> int:
    from . import llm_analysis

    return llm_analysis.analyze_batch(
        batch,
        baseline=REFERENCE_BATCH,
        ledger=None,
        projects_dir=None,
    )


def _campaign_spend(ledger: Path, campaign_id: str) -> float:
    if not ledger.is_file():
        return 0.0
    with sqlite3.connect(ledger) as conn:
        rows = conn.execute("SELECT cost_usd, meta FROM spend").fetchall()
    total = 0.0
    for cost, raw_meta in rows:
        try:
            meta = json.loads(raw_meta) if raw_meta else {}
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(meta, dict) and meta.get("campaign_id") == campaign_id:
            total += float(cost or 0.0)
    return total


def _run_new(batch: Path) -> int:
    settings: Settings | None = None
    manifest_path = batch / "canary_manifest.json"
    if batch.exists() and any(batch.iterdir()):
        print(f"error: output directory is not empty: {batch}", file=sys.stderr)
        return 2
    batch.mkdir(parents=True, exist_ok=True)
    campaign_id = f"llm-canary-{uuid.uuid4()}"
    try:
        settings = Settings.from_env()
        identity = _manifest_identity(settings, campaign_id)
        occupied = probe_build_slots()
        if occupied:
            raise RuntimeError(f"production build slot(s) occupied: {occupied}")
        guard = SpendGuard(settings)
        before = guard.status()
        _require_headroom(before, ENVELOPE_USD)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        payload = {
            "schema_version": 1,
            "created_at": _now(),
            "immutable": locals().get("identity", {"campaign_id": campaign_id}),
            "run_status": "preflight_failed",
            "operational_error": _typed_error("prerequisite", exc, settings),
        }
        _json_write(manifest_path, payload)
        print(f"error: {payload['operational_error']['message']}", file=sys.stderr)
        return 2

    designer_path = batch / "preflight-designer.json"
    judge_path = batch / "preflight-judge.json"
    try:
        designer_result = preflight_role(
            settings,
            role="designer",
            model=settings.model,
            meta_ctx={"campaign_id": campaign_id},
        )
        _json_write(designer_path, _sanitize_preflight(designer_result))
        judge_settings = settings.for_judge()
        judge_result = preflight_role(
            judge_settings,
            role="judge",
            model=settings.eval_judge_model,
            meta_ctx={"campaign_id": campaign_id},
        )
        _json_write(judge_path, _sanitize_preflight(judge_result))
        after = guard.status()
        preflight_cost = float((designer_result.get("smoke") or {}).get("cost_usd") or 0.0) + float(
            (judge_result.get("smoke") or {}).get("cost_usd") or 0.0
        )
        _require_headroom(after, max(0.0, ENVELOPE_USD - preflight_cost))
        if not designer_result.get("ok") or not judge_result.get("ok"):
            raise RuntimeError("designer or judge preflight failed")
        payload = {
            "schema_version": 1,
            "created_at": _now(),
            "updated_at": _now(),
            "immutable": identity,
            "run_status": "ready",
            "operational_error": None,
            "spend": {"before_preflight": before, "after_preflight": after},
            "preflights": {
                "designer": {"path": designer_path.name, "sha256": _sha_file(designer_path)},
                "judge": {"path": judge_path.name, "sha256": _sha_file(judge_path)},
            },
        }
        _json_write(manifest_path, payload)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        if not manifest_path.exists():
            _json_write(
                manifest_path,
                {
                    "schema_version": 1,
                    "created_at": _now(),
                    "immutable": identity,
                    "spend": {"before_preflight": before, "after_preflight": guard.status()},
                    "preflights": {},
                    "run_status": "preflight_failed",
                    "operational_error": _typed_error("preflight", exc, settings),
                },
            )
        else:
            _update_manifest(
                manifest_path,
                status="preflight_failed",
                error=_typed_error("preflight", exc, settings),
            )
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return _run_batch(batch, manifest_path, resume=False, settings=settings)


def _validate_resume(batch: Path, settings: Settings) -> dict:
    manifest_path = batch / "canary_manifest.json"
    campaign_path = batch / "campaign_manifest.json"
    if not manifest_path.is_file() or not campaign_path.is_file():
        raise RuntimeError("resume requires canary_manifest.json and campaign_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    immutable = manifest.get("immutable") or {}
    campaign_id = immutable.get("campaign_id")
    if not campaign_id:
        raise RuntimeError("canary manifest lacks campaign_id")
    expected = _manifest_identity(settings, campaign_id)
    if immutable != expected:
        raise RuntimeError("canary frozen identity differs from current checkout/model/policy")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    frozen = campaign.get("immutable") or {}
    if frozen.get("code_revision") != immutable["checkout"]["commit"]:
        raise RuntimeError("campaign and canary checkout identities disagree")
    if frozen.get("design_model") != immutable["designer"]["model"]:
        raise RuntimeError("campaign and canary designer models disagree")
    if frozen.get("judge_model") != immutable["judge"]["model"]:
        raise RuntimeError("campaign and canary judge models disagree")
    if [row.get("slug") for row in frozen.get("corpus") or []] != list(COHORT):
        raise RuntimeError("campaign cohort differs from fixed canary cohort")
    if frozen.get("repeats") != 1 or immutable.get("envelope_usd") != ENVELOPE_USD:
        raise RuntimeError("campaign repeats or envelope differs from fixed canary")
    return manifest


def _run_batch(batch: Path, manifest_path: Path, *, resume: bool, settings: Settings) -> int:
    try:
        occupied = probe_build_slots()
        if occupied:
            raise RuntimeError(f"production build slot(s) occupied: {occupied}")
        if resume:
            manifest = _validate_resume(batch, settings)
            campaign_id = manifest["immutable"]["campaign_id"]
            remaining = max(0.0, ENVELOPE_USD - _campaign_spend(settings.ledger_path, campaign_id))
            _require_headroom(SpendGuard(settings).status(), remaining)
        _update_manifest(manifest_path, status="ready", error=None)
        rc = _tee_subprocess(_batch_argv(batch, resume=resume), batch / "canary.log")
    except KeyboardInterrupt:
        _update_manifest(
            manifest_path,
            status="batch_interrupted",
            error={"kind": "interrupted", "message": "canary subprocess interrupted"},
        )
        return 130
    except BaseException as exc:
        _update_manifest(
            manifest_path,
            status="batch_failed",
            error=_typed_error("harness", exc, settings),
        )
        print(f"error: {exc}", file=sys.stderr)
        return 2

    status = "batch_complete" if rc == 0 else "batch_failed"
    _update_manifest(
        manifest_path,
        status=status,
        error=None if rc == 0 else {"kind": "subprocess_exit", "message": f"self-eval exited {rc}"},
        batch_exit_code=rc,
    )
    try:
        analysis_rc = _invoke_analysis(batch)
    except BaseException as exc:
        _update_manifest(
            manifest_path,
            status="batch_failed" if rc else "batch_complete",
            error=_typed_error("analysis", exc, settings),
        )
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        return 2
    if rc != 0:
        return 2
    return analysis_rc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="start a fresh fixed nine-slug canary")
    run.add_argument("--out", required=True, type=Path)
    resume = sub.add_parser("resume", help="resume one interrupted canary directory")
    resume.add_argument("batch_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_new(args.out.resolve())
    settings = Settings.from_env()
    batch = args.batch_dir.resolve()
    try:
        _validate_resume(batch, settings)
    except BaseException as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return _run_batch(batch, batch / "canary_manifest.json", resume=True, settings=settings)


if __name__ == "__main__":
    raise SystemExit(main())
