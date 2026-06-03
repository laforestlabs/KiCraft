"""Capped OpenRouter chat client: every model call enforces B0 cost-safety.

Call flow per completion: `SpendGuard.preflight()` (kill switch + ceilings) ->
bounded request (`max_tokens`) -> `SpendGuard.record()` (actual cost). Both the
plain `chat()` and the `chat_with_tools()` loop go through `_complete()`, so the
caps cannot be bypassed by a new code path.
"""
from __future__ import annotations

import json

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

    def _complete(self, body: dict) -> tuple[dict, float]:
        """One capped completion: preflight -> POST -> record actual cost.

        Returns (response_json, cost_usd). Every spend in the product flows
        through here, so a single enforcement point covers chat and tool use.
        """
        self.guard.preflight()  # hard cap check BEFORE any spend
        payload = {**body, "usage": {"include": True}}
        payload.setdefault("model", self.s.model)
        payload.setdefault("max_tokens", self.s.max_tokens_per_call)
        resp = requests.post(
            f"{self.s.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.s.api_key}",
                     "Content-Type": "application/json", "X-Title": "KiCraft"},
            json=payload, timeout=self.s.request_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        in_tok, out_tok = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cost = float(usage.get("cost") or 0.0)
        if cost <= 0.0:  # never record 0 for real spend, or the ceiling under-counts
            cost = estimate_cost(payload["model"], in_tok, out_tok)
        self.guard.record(payload["model"], in_tok, out_tok, cost, meta=body.get("_meta", "chat"))
        return data, cost

    def chat(self, messages, model=None, max_tokens=None, temperature=0.2) -> dict:
        body = {"messages": messages, "temperature": temperature}
        if model:
            body["model"] = model
        if max_tokens:
            body["max_tokens"] = max_tokens
        data, cost = self._complete(body)
        choices = data.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        finish = choices[0].get("finish_reason") if choices else None
        return {"text": msg.get("content") or "", "reasoning": msg.get("reasoning"),
                "finish_reason": finish, "model": data.get("model"),
                "usage": data.get("usage") or {}, "cost_usd": cost,
                "guard": self.guard.status()}

    def chat_with_tools(self, messages, tools, executor, model=None, max_tokens=None,
                        temperature=0.2, max_rounds=12) -> dict:
        """Tool-use loop. `tools` = OpenAI tool specs; `executor(name, args) -> str`.

        Mutates `messages` in place (appends each assistant turn and the tool
        results) so a caller can continue the same conversation afterwards.
        Returns the final assistant text once the model stops calling tools (or
        when max_rounds is hit). Every round is a capped completion.
        """
        total_cost = 0.0
        n_tool_calls = 0
        for rnd in range(max_rounds):
            body = {"messages": messages, "tools": tools, "tool_choice": "auto",
                    "temperature": temperature, "_meta": "tools"}
            if model:
                body["model"] = model
            if max_tokens:
                body["max_tokens"] = max_tokens
            data, cost = self._complete(body)
            total_cost += cost
            choices = data.get("choices") or []
            msg = (choices[0].get("message") or {}) if choices else {}

            assistant = {"role": "assistant", "content": msg.get("content")}
            tcs = msg.get("tool_calls") or []
            if tcs:
                assistant["tool_calls"] = tcs
            messages.append(assistant)

            if not tcs:
                return {"text": msg.get("content") or "", "cost_usd": total_cost,
                        "rounds": rnd + 1, "tool_calls": n_tool_calls,
                        "guard": self.guard.status()}

            for tc in tcs:
                n_tool_calls += 1
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = executor(name, args)
                except Exception as e:  # surface tool errors to the model, don't crash
                    result = f"tool error: {e}"
                messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                                 "name": name, "content": str(result)[:4000]})

        return {"text": "", "cost_usd": total_cost, "rounds": max_rounds,
                "tool_calls": n_tool_calls, "guard": self.guard.status(),
                "max_rounds_hit": True}
