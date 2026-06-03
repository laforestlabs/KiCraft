"""Capped OpenRouter chat client: every model call enforces B0 cost-safety.

Call flow: `SpendGuard.preflight()` (kill switch + ceilings) -> bounded request
(`max_tokens`) -> `SpendGuard.record()` (actual cost). All product model spend
goes through this class, so the caps cannot be bypassed by a new code path.
"""
from __future__ import annotations

import requests

from .config import Settings
from .spend_guard import SpendGuard

# Conservative fallback prices (USD per million tokens, input/output). OpenRouter
# normally returns the real cost; this is used only if it omits it, and it errs
# high so a missing cost never lets the spend ceiling under-count actual spend.
_FALLBACK_PRICES = {
    "deepseek": (1.0, 2.0),
    "haiku": (1.0, 5.0),
    "sonnet": (3.0, 15.0),
    "opus": (15.0, 75.0),
}
_FALLBACK_DEFAULT = (10.0, 30.0)  # unknown model: assume expensive


def estimate_cost(model: str, input_tokens, output_tokens) -> float:
    mid = (model or "").lower()
    inp, out = _FALLBACK_DEFAULT
    for fam, price in _FALLBACK_PRICES.items():
        if fam in mid:
            inp, out = price
            break
    return ((input_tokens or 0) * inp + (output_tokens or 0) * out) / 1_000_000.0


class CappedOpenRouterClient:
    def __init__(self, settings: Settings | None = None, guard: SpendGuard | None = None):
        self.s = settings or Settings.from_env()
        self.guard = guard or SpendGuard(self.s)

    def chat(self, messages: list[dict], model: str | None = None,
             max_tokens: int | None = None, temperature: float = 0.2) -> dict:
        # 1) Hard cap check BEFORE any spend.
        self.guard.preflight()

        model = model or self.s.model
        max_tokens = max_tokens or self.s.max_tokens_per_call
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,     # 2) bound the cost of this single call
            "temperature": temperature,
            "usage": {"include": True},   # ask OpenRouter to return the real cost
        }
        headers = {
            "Authorization": f"Bearer {self.s.api_key}",
            "Content-Type": "application/json",
            "X-Title": "KiCraft",
        }
        resp = requests.post(f"{self.s.base_url}/chat/completions", headers=headers,
                             json=body, timeout=self.s.request_timeout_s)
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage") or {}
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")
        cost = float(usage.get("cost") or 0.0)
        if cost <= 0.0:  # 3) never record 0 for real spend, or the ceiling under-counts
            cost = estimate_cost(model, in_tok, out_tok)
        self.guard.record(model, in_tok, out_tok, cost, meta="chat")

        choices = data.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        finish = choices[0].get("finish_reason") if choices else None
        # content can be null (e.g. reasoning-only turns); coerce to str so callers
        # never get None back. Surface reasoning/finish_reason for visibility.
        text = msg.get("content") or ""
        return {"text": text, "reasoning": msg.get("reasoning"), "finish_reason": finish,
                "model": model, "usage": usage, "cost_usd": cost,
                "guard": self.guard.status()}
