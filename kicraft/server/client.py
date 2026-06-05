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

    def _provider_block(self) -> dict | None:
        """OpenRouter `provider` routing block (cost safety). Prefers the caching
        backend(s) in `provider_order`, allows bounded fallbacks, and caps the
        per-Mtok price so no single call can hit the expensive-backend tail."""
        prov: dict = {}
        if self.s.provider_order:
            prov["order"] = list(self.s.provider_order)
        prov["allow_fallbacks"] = self.s.provider_allow_fallbacks
        mp = {}
        if self.s.max_price_prompt > 0:
            mp["prompt"] = self.s.max_price_prompt
        if self.s.max_price_completion > 0:
            mp["completion"] = self.s.max_price_completion
        if mp:
            prov["max_price"] = mp
        return prov or None

    @staticmethod
    def _apply_cache_control(messages: list) -> None:
        """Mark the system prompt (the large, stable spec+schema prefix that is
        re-sent on every tool round and retry) with an ephemeral cache breakpoint
        in OpenAI content-parts form. Honored by caching providers, ignored by the
        rest; DeepSeek caches automatically regardless. Idempotent (skips content
        that is already structured)."""
        for m in messages:
            if m.get("role") != "system":
                continue
            content = m.get("content")
            if isinstance(content, str) and content:
                m["content"] = [{"type": "text", "text": content,
                                 "cache_control": {"type": "ephemeral"}}]
            break

    def _stream(self, body: dict, on_delta=None) -> tuple[dict, float]:
        """One capped streaming completion (SSE).

        Calls on_delta({"reasoning"|"content": <partial text>}) as tokens arrive,
        accumulates content / reasoning / tool_calls from the deltas, and records
        the real cost from the final usage chunk. Returns (assembled_message,
        cost). preflight() runs before any spend, so the caps still apply.
        """
        self.guard.preflight()  # hard cap check BEFORE any spend
        # Internal "_"-prefixed keys (_meta, _meta_ctx) are control data, not API
        # fields: keep them out of the request body sent to OpenRouter.
        meta_phase = body.get("_meta", "stream")
        meta_ctx = body.get("_meta_ctx") if isinstance(body.get("_meta_ctx"), dict) else {}
        payload = {k: v for k, v in body.items() if not k.startswith("_")}
        payload.update({"stream": True, "stream_options": {"include_usage": True},
                        "usage": {"include": True}})
        payload.setdefault("model", self.s.model)
        payload.setdefault("max_tokens", self.s.max_tokens_per_call)
        prov = self._provider_block()
        if prov:
            payload["provider"] = prov
        if self.s.enable_prompt_cache and isinstance(payload.get("messages"), list):
            self._apply_cache_control(payload["messages"])
        content, reasoning = [], []
        tool_calls: dict = {}
        finish = None
        provider = None
        usage: dict = {}
        with requests.post(
            f"{self.s.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.s.api_key}",
                     "Content-Type": "application/json", "X-Title": "KiCraft"},
            json=payload, timeout=self.s.request_timeout_s, stream=True,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("provider"):
                    provider = chunk["provider"]
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    if delta.get("reasoning"):
                        reasoning.append(delta["reasoning"])
                        if on_delta:
                            on_delta({"reasoning": delta["reasoning"]})
                    if delta.get("content"):
                        content.append(delta["content"])
                        if on_delta:
                            on_delta({"content": delta["content"]})
                    for tcd in delta.get("tool_calls") or []:
                        slot = tool_calls.setdefault(tcd.get("index", 0),
                                                     {"id": None, "name": "", "args": ""})
                        if tcd.get("id"):
                            slot["id"] = tcd["id"]
                        fn = tcd.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]

        msg = {"role": "assistant", "content": "".join(content) or None,
               "reasoning": "".join(reasoning) or None, "finish_reason": finish}
        if tool_calls:
            msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["args"]}}
                for tc in (tool_calls[i] for i in sorted(tool_calls))]

        in_tok, out_tok = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        cost = float(usage.get("cost") or 0.0)
        if cost <= 0.0:  # never record 0 for real spend, or the ceiling under-counts
            cost = estimate_cost(payload["model"], in_tok, out_tok)
        rec_meta = {"phase": meta_phase, "provider": provider, "finish_reason": finish,
                    "cached_tokens": int(cached or 0), **meta_ctx}
        self.guard.record(payload["model"], in_tok, out_tok, cost, meta=rec_meta)
        return msg, cost

    @staticmethod
    def _delta_progress(progress):
        """Wrap a progress callback to forward streaming token deltas."""
        def on_delta(d):
            if not progress:
                return
            if d.get("reasoning"):
                progress({"kind": "reasoning_delta", "text": d["reasoning"]})
            elif d.get("content"):
                progress({"kind": "answer_delta", "text": d["content"]})
        return on_delta

    def chat(self, messages, model=None, max_tokens=None, temperature=0.2, progress=None,
             meta_ctx=None) -> dict:
        body = {"messages": messages, "temperature": temperature}
        if model:
            body["model"] = model
        if max_tokens:
            body["max_tokens"] = max_tokens
        if meta_ctx:
            body["_meta_ctx"] = meta_ctx
        msg, cost = self._stream(body, on_delta=self._delta_progress(progress))
        return {"text": msg.get("content") or "", "reasoning": msg.get("reasoning"),
                "finish_reason": msg.get("finish_reason"), "model": None,
                "usage": {}, "cost_usd": cost, "guard": self.guard.status()}

    def chat_with_tools(self, messages, tools, executor, model=None, max_tokens=None,
                        temperature=0.2, max_rounds=12, progress=None, meta_ctx=None) -> dict:
        """Tool-use loop. `tools` = OpenAI tool specs; `executor(name, args) -> str`.

        Mutates `messages` in place (appends each assistant turn and the tool
        results) so a caller can continue the same conversation afterwards.
        `progress(event)` is called as work happens with events of kind
        "reasoning" / "tool" / "tool_result" / "answer". Every round is a capped
        completion.
        """
        total_cost = 0.0
        n_tool_calls = 0
        seen: dict[str, int] = {}  # (name, args) signature -> times requested
        on_delta = self._delta_progress(progress)
        for rnd in range(max_rounds):
            if rnd == max_rounds - 1:
                # Final tool round: ask the model to finish on THIS call. It keeps
                # tool_choice="auto", so the prompt-cache prefix still matches the loop
                # and the call is cache-warm. If the model complies we return via the
                # normal no-tool-calls path below and skip the separate forced-final,
                # which sets tool_choice="none", a request shape the serving provider
                # does not prompt-cache (observed cache_tok=0), making it the single
                # most expensive call in a maxed-out BOM stage.
                messages.append({"role": "user", "content":
                                 "This is your FINAL tool round. Stop calling tools and "
                                 "output ONLY the final JSON answer now."})
            body = {"messages": messages, "tools": tools, "tool_choice": "auto",
                    "parallel_tool_calls": True, "temperature": temperature,
                    "_meta": "tools", "_meta_ctx": {**(meta_ctx or {}), "round": rnd}}
            if model:
                body["model"] = model
            if max_tokens:
                body["max_tokens"] = max_tokens
            msg, cost = self._stream(body, on_delta=on_delta)
            total_cost += cost

            assistant = {"role": "assistant", "content": msg.get("content")}
            tcs = msg.get("tool_calls") or []
            if tcs:
                assistant["tool_calls"] = tcs
            messages.append(assistant)

            if not tcs:
                return {"text": msg.get("content") or "", "cost_usd": total_cost,
                        "rounds": rnd + 1, "tool_calls": n_tool_calls,
                        "finish_reason": msg.get("finish_reason"),
                        "guard": self.guard.status()}

            for tc in tcs:
                n_tool_calls += 1
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if progress:
                    progress({"kind": "tool", "name": name, "args": args})
                try:
                    result = executor(name, args)
                except Exception as e:  # surface tool errors to the model, don't crash
                    result = f"tool error: {e}"
                # Break identical-call thrash: a weak model can repeat the exact same
                # failing call for rounds on end. The result will not change, so tell
                # it to stop and converge instead of silently re-running.
                sig = name + "|" + json.dumps(args, sort_keys=True)
                seen[sig] = seen.get(sig, 0) + 1
                if seen[sig] >= 2:
                    result = (f"NOTE: you have already made this identical call "
                              f"{seen[sig]} times; the result will not change. Stop "
                              f"repeating it: change the arguments or output your final "
                              f"JSON answer now.\n{result}")
                if progress:
                    progress({"kind": "tool_result", "name": name, "output": str(result)[:600]})
                messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                                 "name": name, "content": str(result)[:4000]})

        # Tool-round budget exhausted. Returning empty text here reads upstream as
        # "no JSON in reply" (a silent, expensive failure). Instead force one final
        # tool-free completion so the model commits to an answer we can parse.
        messages.append({"role": "user", "content":
                         "You have used your entire tool-call budget. Do NOT call any "
                         "more tools. Output your final answer now as a single JSON "
                         "object only."})
        body = {"messages": messages, "tools": tools, "tool_choice": "none",
                "temperature": temperature, "_meta": "tools-final",
                "_meta_ctx": {**(meta_ctx or {}), "round": "final"}}
        if model:
            body["model"] = model
        if max_tokens:
            body["max_tokens"] = max_tokens
        msg, cost = self._stream(body, on_delta=on_delta)
        total_cost += cost
        messages.append({"role": "assistant", "content": msg.get("content")})
        return {"text": msg.get("content") or "", "cost_usd": total_cost,
                "rounds": max_rounds, "tool_calls": n_tool_calls,
                "finish_reason": msg.get("finish_reason"),
                "guard": self.guard.status(), "max_rounds_hit": True,
                "forced_final": True}
