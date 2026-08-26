"""Stage-chain, replay, budget-client, and full-pipeline sequencing."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .client import CappedOpenRouterClient, make_client
from .config import Settings
from .spend_guard import BudgetExceeded, SpendGuard
from .stage_runtime import _stage_max_retries, _stage_max_tokens, drive_stage
from .stage_state_io import KICRAFT, run_design_cli

DESIGN_STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring")
SUPPORTED_STAGES = DESIGN_STAGES

def drive_chain(
    stages,
    brief,
    workspace,
    max_tokens=4096,
    max_retries=2,
    on_stage=None,
    progress=None,
    client=None,
    answers=None,
    instruction=None,
    run_id=None,
    core_defaults=None,
):
    ws = Path(workspace)
    (ws / ".kicraft").mkdir(parents=True, exist_ok=True)
    state_path = ws / ".kicraft" / "state.json"
    if client is None:
        # make_client() builds Settings.from_env() itself for the live client, and
        # skips it entirely for the mock (which needs no OPENROUTER_API_KEY).
        client = make_client()
    base_ctx = {"run_id": run_id} if run_id else {}
    results = []
    for i, stage in enumerate(stages):
        # answers/instruction belong to the stage being resumed or edited, which
        # is the first stage of this chain; downstream stages re-draft cleanly.
        r = drive_stage(
            client,
            stage,
            brief,
            state_path,
            ws,
            _stage_max_tokens(stage, max_tokens),
            _stage_max_retries(stage, max_retries),
            progress=progress,
            answers=(answers if i == 0 else None),
            instruction=(instruction if i == 0 else None),
            meta_ctx=base_ctx,
            core_defaults=core_defaults,
        )
        results.append(r)
        if on_stage:
            on_stage(r)
        cost = r.get("cost_usd")
        cstr = f"${cost:.6f}" if isinstance(cost, (int, float)) else "n/a"
        tag = "ok  " if r.get("commit_ok") else "FAIL"
        extra = f" rounds={r['rounds']}" if r.get("rounds") else ""
        if r.get("tool_calls") is not None:
            extra += f" tools={r['tool_calls']}"
        line = f"  [{tag}] {stage:<16} cost={cstr}  attempts={r.get('attempts', '-')}{extra}"
        if not r.get("commit_ok"):
            line += f"\n         -> {r.get('error') or r.get('commit')}"
            if r.get("reply_head"):
                line += f"\n         reply_head: {r['reply_head']!r}"
        if r.get("needs_input"):
            line += "\n         -> parked: awaiting a clarifying answer from the user"
        print(line)
        if not r.get("commit_ok") or r.get("needs_input"):
            break
    return results, client.guard.status(), str(state_path)


class _BudgetGuard:
    """Wrap a SpendGuard with a per-run USD ceiling on top of the global ones.

    ``preflight()`` (called before every model completion) refuses once this
    run's delta past the snapshot reaches ``budget_usd``. Granularity is one
    completion, so a run may overshoot by at most a single call. Everything
    else (record / record_stage / status / spent_*) delegates to the base.
    """

    def __init__(self, base: SpendGuard, budget_usd: float):
        self._base = base
        self._budget = float(budget_usd)
        self._start = base.spent_total()

    def _delta(self) -> float:
        return self._base.spent_total() - self._start

    def preflight(self) -> None:
        self._base.preflight()
        if self._delta() >= self._budget:
            raise BudgetExceeded(
                f"run budget ${self._budget:.2f} exhausted (spent ${self._delta():.4f})"
            )

    def __getattr__(self, name):
        return getattr(self._base, name)


def make_budget_client(budget_usd: float = 0.25):
    """A client whose guard additionally refuses once THIS run spends
    ``budget_usd`` (on top of the global daily/total ceilings). Mock/replay
    mode spends $0 and returns the plain mock client (no budget needed)."""
    if os.environ.get("KICRAFT_LLM_MODE", "live").strip().lower() in ("mock", "replay"):
        return make_client()
    settings = Settings.from_env()
    guard = SpendGuard(settings)
    if budget_usd and budget_usd > 0:
        guard = _BudgetGuard(guard, budget_usd)
    return CappedOpenRouterClient(settings, guard=guard)


def run_pipeline(
    brief,
    workspace,
    stages=DESIGN_STAGES,
    budget_usd=0.25,
    max_tokens=4096,
    max_retries=2,
    build=True,
    quality="good",
    progress=None,
    core_defaults=None,
    client=None,
) -> dict:
    """Full end-to-end run: drive the LLM design stages (budget-capped), then —
    if every stage committed — run the deterministic build. This is the harness
    for testing LLM-prompt / guardrail changes against a real board."""
    client = client or make_budget_client(budget_usd)
    results, guard, state_path = drive_chain(
        list(stages),
        brief,
        workspace,
        max_tokens=max_tokens,
        max_retries=max_retries,
        client=client,
        progress=progress,
        core_defaults=core_defaults,
    )
    all_committed = len(results) == len(stages) and all(r.get("commit_ok") for r in results)
    build_rc = None
    if build and all_committed:
        build_rc = run_design_cli(
            KICRAFT
            + ["build", ".kicraft/state.json", "generated", "--no-archive", "--quality", quality],
            cwd=Path(workspace),
        ).returncode
    return {
        "stages": results,
        "all_committed": all_committed,
        "guard": guard,
        "state_path": str(state_path),
        "build_rc": build_rc,
    }


def drive_replay(
    state_path,
    stage,
    budget_usd=0.25,
    max_retries=2,
    progress=None,
    core_defaults=None,
    client=None,
) -> dict:
    """Re-run ONE design stage from a frozen, already-committed state.json — the
    LLM-side repro harness for prompt/guardrail changes (mirrors ``cli_app
    replay`` for the deterministic place/route stages). Copies the state into a
    temp workspace (the source is never mutated), reads the brief from it, and
    drives ``stage`` with a budget-capped client."""
    src = Path(state_path).expanduser().resolve()
    if not src.is_file():
        return {"error": f"state.json not found: {src}"}
    try:
        state = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"error": f"could not read {src}: {e}"}
    brief = ((state.get("intent") or {}).get("goal") or "").strip()
    if not brief:
        brief_txt = src.parent.parent / "brief.txt"
        if brief_txt.is_file():
            brief = brief_txt.read_text(encoding="utf-8").strip()
    if not brief:
        return {"error": f"no brief recoverable from {src} (intent.goal or brief.txt)"}
    if stage not in SUPPORTED_STAGES:
        return {"error": f"unsupported stage {stage!r}; supported: {list(SUPPORTED_STAGES)}"}

    tmp = Path(tempfile.mkdtemp(prefix="kc-replay-"))
    (tmp / ".kicraft").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp / ".kicraft" / "state.json")

    client = client or make_budget_client(budget_usd)
    results, guard, spath = drive_chain(
        [stage],
        brief,
        tmp,
        max_retries=max_retries,
        client=client,
        progress=progress,
        core_defaults=core_defaults,
    )
    return {
        "brief": brief,
        "workspace": str(tmp),
        "state_path": str(spath),
        "stage": results[0] if results else None,
        "guard": guard,
        "all_committed": bool(results) and results[0].get("commit_ok"),
    }
