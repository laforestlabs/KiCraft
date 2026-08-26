"""Capped OpenRouter chat client: every model call enforces B0 cost-safety.

Call flow per completion: `SpendGuard.preflight()` (kill switch + ceilings) ->
bounded request (`max_tokens`) -> `SpendGuard.record()` (actual cost). Both the
plain `chat()` and the `chat_with_tools()` loop go through `_complete()`, so the
caps cannot be bypassed by a new code path.
"""

from __future__ import annotations

import json
import os
import time

import requests

from .config import CollectionBound, ReasoningGuardPolicy, Settings
from .spend_guard import SpendGuard

# Transient failures worth a bounded retry (before any token is streamed): all
# 5xx + 429 (rate limit) on the HTTP status, plus connection resets / timeouts.
_RETRY_NETWORK_EXC = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

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

# Tool-loop convergence caps. A weak model (e.g. deepseek-flash) fills the whole
# round budget re-verifying parts it already resolved, repeating identical
# lookups for rounds on end. Once it has reused enough cached results -- or made
# enough total tool calls -- stop offering tools and force the final JSON.
_MAX_REDUNDANT_TOOL_CALLS = 3
_MAX_TOTAL_TOOL_CALLS = 16

# Reasoning-loop breaker: a reasoning model can burn its whole output budget
# re-deriving one decision and emit NO content. max_tokens does NOT bound
# DeepSeek's reasoning channel, so the client enforces its own in-stream ceiling
# and repetition fingerprint. See docs/plans/reasoning-loop-breaker.md.
# Recent-buffer size (chars) used for the repetition fingerprint: large enough to
# hold several copies of the ~700-char block a stuck model repeats verbatim.
_REASONING_RECENT_CHARS = 4096


class _StreamingCollectionGuard:
    """Count direct members of bounded top-level JSON arrays incrementally."""

    def __init__(self, bounds: tuple[CollectionBound, ...]):
        self._counters = [_TopLevelArrayCounter(bound) for bound in bounds]

    def consume(self, text: str) -> tuple[str, dict | None]:
        for index, char in enumerate(text):
            for counter in self._counters:
                overflow = counter.consume(char)
                if overflow is not None:
                    return text[:index], overflow
        return text, None

    def counts(self) -> dict[str, int]:
        return {counter.bound.field: counter.count for counter in self._counters}


class _TopLevelArrayCounter:
    """Streaming lexer for one direct child array of the root JSON object."""

    def __init__(self, bound: CollectionBound):
        self.bound = bound
        self.stack: list[str] = []
        self.in_string = False
        self.escape = False
        self.capture_string = False
        self.string_buf = ""
        self.completed_string: str | None = None
        self.awaiting_array = False
        self.target_depth: int | None = None
        self.expect_member = False
        self.count = 0
        self.member_buf: list[str] | None = None
        self.group_counts: dict[str, int] = {}

    def _finish_member(self) -> dict | None:
        if self.member_buf is None:
            return None
        raw = "".join(self.member_buf).strip()
        self.member_buf = None
        if not raw or self.bound.per_group is None or self.bound.group_key is None:
            return None
        try:
            member = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(member, dict) or self.bound.group_key not in member:
            return None
        group = str(member[self.bound.group_key])
        observed = self.group_counts.get(group, 0) + 1
        self.group_counts[group] = observed
        if observed <= self.bound.per_group:
            return None
        return {
            "field": self.bound.field,
            "observed_count": observed,
            "configured_total": self.bound.per_group,
            "limit_scope": "group",
            "group_key": self.bound.group_key,
            "group_value": group,
        }

    def consume(self, char: str) -> dict | None:
        direct_target = self.target_depth is not None and len(self.stack) == self.target_depth
        ends_member = (
            self.member_buf is not None and not self.in_string and direct_target and char in ",]"
        )
        if ends_member:
            overflow = self._finish_member()
            if overflow is not None:
                return overflow
        elif self.member_buf is not None:
            self.member_buf.append(char)

        if self.in_string:
            if self.escape:
                self.escape = False
                if self.capture_string:
                    self.string_buf += char
                return None
            if char == "\\":
                self.escape = True
                return None
            if char == '"':
                self.in_string = False
                self.completed_string = self.string_buf if self.capture_string else None
                self.capture_string = False
                self.string_buf = ""
                return None
            if self.capture_string and len(self.string_buf) <= len(self.bound.field):
                self.string_buf += char
            return None

        direct_target = self.target_depth is not None and len(self.stack) == self.target_depth
        if self.expect_member and direct_target and not char.isspace() and char != "]":
            observed = self.count + 1
            if observed > self.bound.total:
                return {
                    "field": self.bound.field,
                    "observed_count": observed,
                    "configured_total": self.bound.total,
                }
            self.count = observed
            self.expect_member = False
            self.member_buf = [char]

        if char == '"':
            self.in_string = True
            self.capture_string = self.stack == ["{"]
            self.string_buf = ""
            return None
        if char.isspace():
            return None
        if char == ":":
            self.awaiting_array = self.stack == ["{"] and self.completed_string == self.bound.field
            self.completed_string = None
            return None
        self.completed_string = None
        if char == "{":
            self.stack.append("{")
        elif char == "[":
            self.stack.append("[")
            if self.awaiting_array and self.stack == ["{", "["]:
                self.target_depth = len(self.stack)
                self.expect_member = True
            self.awaiting_array = False
        elif char == "}":
            if self.stack and self.stack[-1] == "{":
                self.stack.pop()
        elif char == "]":
            if self.target_depth is not None and len(self.stack) == self.target_depth:
                self.target_depth = None
                self.expect_member = False
            if self.stack and self.stack[-1] == "[":
                self.stack.pop()
        elif char == "," and direct_target:
            self.expect_member = True
        elif self.awaiting_array:
            self.awaiting_array = False
        return None


def make_client(settings: Settings | None = None):
    """Construct the active chat client, honoring ``KICRAFT_LLM_MODE``.

    Default (unset / anything but mock|replay) returns the real
    ``CappedOpenRouterClient`` -- so this is a prod no-op. ``mock``/``replay``
    return a ``MockClient`` that replays a recorded per-stage transcript at $0,
    for load/stress testing the pipeline without spend (and without an API key:
    the mock never reads ``settings.api_key``). The import is lazy so the
    loadtest package is never pulled into the hot path in production.
    """
    mode = os.environ.get("KICRAFT_LLM_MODE", "live").strip().lower()
    if mode in ("mock", "replay"):
        from kicraft.loadtest.mockllm import MockClient

        return MockClient(settings)
    return CappedOpenRouterClient(settings or Settings.from_env())


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
                m["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            break

    def _open_stream(self, payload: dict):
        """POST and return a streamed Response with a non-retryable status.

        Retries TRANSIENT failures (HTTP >=500 / 429, connection reset, timeout)
        with exponential backoff BEFORE any token is consumed, so a one-off 503
        no longer drops the call. Once a 2xx response is returned, streaming
        proceeds in ``_stream`` and is never retried (it could double-emit). Other
        4xx errors raise immediately via ``raise_for_status`` (not transient)."""
        url = f"{self.s.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.s.api_key}",
            "Content-Type": "application/json",
            "X-Title": "KiCraft",
        }
        max_retries = max(0, int(getattr(self.s, "llm_max_retries", 0)))
        backoff = float(getattr(self.s, "llm_retry_backoff_s", 1.0))
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.s.request_timeout_s,
                    stream=True,
                )
                if resp.status_code >= 500 or resp.status_code == 429:
                    transient = requests.exceptions.HTTPError(
                        f"{resp.status_code} {resp.reason}", response=resp
                    )
                    resp.close()
                    raise transient
                resp.raise_for_status()  # other 4xx: not transient -> propagate
                # OpenRouter's SSE stream is UTF-8 but sends `text/event-stream`
                # with NO charset, so requests falls back to ISO-8859-1 for
                # `iter_lines(decode_unicode=True)` -- which turns a UTF-8 `µ`
                # (0xC2 0xB5) into "Âµ" and every other multibyte char into
                # mojibake that then lands verbatim in the BOM value / state.json.
                # Pin UTF-8 so the decoded stream matches the bytes on the wire.
                resp.encoding = "utf-8"
                return resp
            except (*_RETRY_NETWORK_EXC, requests.exceptions.HTTPError) as e:
                # Only 5xx/429 HTTPErrors reach here as retryable; a 4xx
                # raise_for_status raised above is also an HTTPError, but it was
                # already returned... so distinguish: a 4xx has a response with a
                # client-error code and must NOT retry.
                code = getattr(getattr(e, "response", None), "status_code", None)
                is_http = isinstance(e, requests.exceptions.HTTPError)
                if is_http and code is not None and code < 500 and code != 429:
                    raise
                if attempt >= max_retries:
                    raise
                time.sleep(backoff * (2**attempt))

    @staticmethod
    def _reasoning_abort_reason(
        policy: ReasoningGuardPolicy | None,
        reasoning_chars: int,
        content_chars: int,
        reasoning_recent: str,
        stream_t0: float,
    ) -> str | None:
        """Return the first policy limit crossed by a reasoning-only stream."""
        if policy is None or content_chars:
            return None
        if reasoning_chars > policy.hard_max_tokens * 4:
            return "hard_ceiling"
        if (time.monotonic() - stream_t0) > policy.wall_stall_s:
            return "wall_stall"
        if not policy.repetition_enabled:
            return None
        window = policy.repeat_window
        if (
            len(reasoning_recent) >= window
            and reasoning_recent.count(reasoning_recent[-window:]) >= policy.repeat_threshold
        ):
            return "repetition"
        return None

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
        reasoning_guard = body.get("_reasoning_guard")
        if not isinstance(reasoning_guard, ReasoningGuardPolicy):
            reasoning_guard = None
        collection_bounds = body.get("_collection_bounds")
        if not isinstance(collection_bounds, tuple):
            collection_bounds = ()
        payload = {k: v for k, v in body.items() if not k.startswith("_")}
        payload.update(
            {"stream": True, "stream_options": {"include_usage": True}, "usage": {"include": True}}
        )
        payload.setdefault("model", self.s.model)
        payload.setdefault("max_tokens", self.s.max_tokens_per_call)
        prov = self._provider_block()
        if prov:
            payload["provider"] = prov
        if self.s.enable_prompt_cache and isinstance(payload.get("messages"), list):
            self._apply_cache_control(payload["messages"])
        # Mid-stream retry: _open_stream retries transient failures only up to
        # the 2xx header; a connection dropped DURING iter_lines (e.g.
        # "Connection broken: InvalidChunkLength" -- live board 625) used to
        # propagate out and permanently fail the whole design run. Nothing has
        # been committed at that point, so the safe recovery is to discard the
        # partial buffers and re-POST the identical payload from scratch. The
        # only cost is cosmetic: on_delta already streamed the discarded
        # partials to the UI, so the viewer sees the reasoning restart.
        max_stream_retries = max(0, int(getattr(self.s, "llm_max_retries", 0)))
        stream_backoff = float(getattr(self.s, "llm_retry_backoff_s", 1.0))
        for stream_attempt in range(max_stream_retries + 1):
            content, reasoning = [], []
            tool_calls: dict = {}
            finish = None
            provider = None
            usage: dict = {}
            loop_abort_reason = None
            collection_limit = None
            reasoning_chars = 0
            content_chars = 0
            reasoning_recent = ""
            collection_guard = _StreamingCollectionGuard(collection_bounds)
            stream_t0 = time.monotonic()
            try:
                resp = self._open_stream(payload)
                with resp:
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
                        if chunk.get("error"):
                            error = chunk["error"]
                            detail = error.get("message") if isinstance(error, dict) else str(error)
                            raise requests.exceptions.HTTPError(
                                f"OpenRouter stream error: {detail}"
                            )
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
                                reasoning_chars += len(delta["reasoning"])
                                reasoning_recent = (reasoning_recent + delta["reasoning"])[
                                    -_REASONING_RECENT_CHARS:
                                ]
                                if on_delta:
                                    on_delta({"reasoning": delta["reasoning"]})
                            if delta.get("content"):
                                accepted, overflow = collection_guard.consume(delta["content"])
                                if accepted:
                                    content.append(accepted)
                                    content_chars += len(accepted)
                                    if on_delta:
                                        on_delta({"content": accepted})
                                if overflow is not None:
                                    collection_limit = {
                                        **overflow,
                                        "emitted_content_chars": content_chars,
                                    }
                                    break
                            for tcd in delta.get("tool_calls") or []:
                                slot = tool_calls.setdefault(
                                    tcd.get("index", 0), {"id": None, "name": "", "args": ""}
                                )
                                if tcd.get("id"):
                                    slot["id"] = tcd["id"]
                                fn = tcd.get("function") or {}
                                if fn.get("name"):
                                    slot["name"] = fn["name"]
                                if fn.get("arguments"):
                                    slot["args"] += fn["arguments"]
                            loop_abort_reason = self._reasoning_abort_reason(
                                reasoning_guard,
                                reasoning_chars,
                                content_chars,
                                reasoning_recent,
                                stream_t0,
                            )
                            if loop_abort_reason:
                                break
                        if loop_abort_reason or collection_limit:
                            break
            except _RETRY_NETWORK_EXC:
                if stream_attempt >= max_stream_retries:
                    raise
                time.sleep(stream_backoff * (2**stream_attempt))
                continue
            break  # stream ended cleanly or by a client-owned policy abort

        if loop_abort_reason:
            finish = "reasoning_loop"
        elif collection_limit:
            finish = "collection_limit"
        msg = {
            "role": "assistant",
            "content": "".join(content) or None,
            "reasoning": "".join(reasoning) or None,
            "finish_reason": finish,
        }
        if loop_abort_reason:
            msg["loop_detected"] = True
            msg["loop_abort_reason"] = loop_abort_reason
        if collection_limit:
            msg["collection_limit"] = collection_limit
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["args"]},
                }
                for tc in (tool_calls[i] for i in sorted(tool_calls))
            ]
        # Completion telemetry for chat()/chat_with_tools() callers: provider
        # usage fields when supplied, content/reasoning character counts, the
        # requested max_tokens, and the selected reasoning policy. Null-safe
        # for mocks and legacy providers (None is the "not supplied" value).
        msg["provider"] = provider
        msg["usage"] = dict(usage) if usage else None
        msg["content_chars"] = content_chars
        msg["reasoning_chars"] = reasoning_chars
        msg["requested_max_tokens"] = payload.get("max_tokens")
        msg["reasoning_policy"] = payload.get("reasoning")
        msg["reasoning_policy_name"] = reasoning_guard.name if reasoning_guard else None
        msg["collection_counts"] = collection_guard.counts()

        in_tok, out_tok = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if loop_abort_reason or collection_limit:
            # A client-owned abort precedes the final usage chunk. Estimate both
            # prompt and partial output so the spend ceiling remains conservative.
            prompt_chars = len(
                json.dumps(payload.get("messages") or [], ensure_ascii=False, separators=(",", ":"))
            )
            in_tok = in_tok or max(1, prompt_chars // 4)
            out_tok = out_tok or max(1, (reasoning_chars + content_chars) // 4)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        cost = float(usage.get("cost") or 0.0)
        if cost <= 0.0:  # never record 0 for real spend, or the ceiling under-counts
            cost = estimate_cost(payload["model"], in_tok, out_tok)
        response_policy = (payload.get("response_format") or {}).get("json_schema") or {}
        rec_meta = {
            "phase": meta_phase,
            "profile": getattr(self.s, "design_profile", "custom"),
            "provider": provider,
            "finish_reason": finish,
            "cached_tokens": int(cached or 0),
            "loop_detected": bool(loop_abort_reason),
            "loop_abort_reason": loop_abort_reason,
            "reasoning_policy_name": reasoning_guard.name if reasoning_guard else None,
            "response_policy_name": response_policy.get("name"),
            "reasoning_chars": reasoning_chars,
            "content_chars": content_chars,
            "max_tokens": payload.get("max_tokens"),
            "reasoning_policy": payload.get("reasoning"),
            "collection_limit": collection_limit,
            **meta_ctx,
            "collection_counts": collection_guard.counts(),
        }
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

    def chat(
        self,
        messages,
        model=None,
        max_tokens=None,
        temperature=0.2,
        progress=None,
        meta_ctx=None,
        reasoning=None,
        reasoning_guard=None,
        collection_bounds=(),
        response_format=None,
    ) -> dict:
        body = {"messages": messages, "temperature": temperature}
        if model:
            body["model"] = model
        if max_tokens:
            body["max_tokens"] = max_tokens
        if meta_ctx:
            body["_meta_ctx"] = meta_ctx
        # OpenRouter unified reasoning control (the "thinking budget"), e.g.
        # {"max_tokens": 8000} or {"effort": "high"}. Passed straight through to
        # the provider; harmless to omit.
        if reasoning:
            body["reasoning"] = reasoning
        if reasoning_guard:
            body["_reasoning_guard"] = reasoning_guard
        if collection_bounds:
            body["_collection_bounds"] = tuple(collection_bounds)
        if response_format:
            body["response_format"] = response_format
        msg, cost = self._stream(body, on_delta=self._delta_progress(progress))
        return {
            "text": msg.get("content") or "",
            "reasoning": msg.get("reasoning"),
            "finish_reason": msg.get("finish_reason"),
            "model": model or self.s.model,
            "usage": msg.get("usage") or {},
            "cost_usd": cost,
            "guard": self.guard.status(),
            "loop_detected": bool(msg.get("loop_detected")),
            "loop_abort_reason": msg.get("loop_abort_reason"),
            "reasoning_policy_name": msg.get("reasoning_policy_name"),
            "collection_limit": msg.get("collection_limit"),
            "provider": msg.get("provider"),
            "profile": getattr(self.s, "design_profile", "custom"),
            "max_tokens": msg.get("requested_max_tokens"),
            "reasoning_policy": msg.get("reasoning_policy"),
            "content_chars": msg.get("content_chars"),
            "collection_counts": msg.get("collection_counts") or {},
            "reasoning_chars": msg.get("reasoning_chars"),
        }

    def chat_with_tools(
        self,
        messages,
        tools,
        executor,
        model=None,
        max_tokens=None,
        temperature=0.2,
        max_rounds=12,
        progress=None,
        meta_ctx=None,
        reasoning=None,
        reasoning_guard=None,
        collection_bounds=(),
        response_format=None,
    ) -> dict:
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
        cache: dict[str, str] = {}  # signature -> first result (reused on repeats)
        redundant = 0  # identical calls served from cache
        force_final = False  # thrash detected -> hard-stop tools next round
        on_delta = self._delta_progress(progress)
        for rnd in range(max_rounds):
            last_round = rnd == max_rounds - 1
            final_response = force_final or last_round
            if final_response:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "This is the FINAL tool round and FINAL response. Stop "
                            "calling tools and output ONLY the schema-bound JSON answer now."
                        ),
                    }
                )
            body = {
                "messages": messages,
                "tools": tools,
                "tool_choice": "none" if final_response else "auto",
                "parallel_tool_calls": True,
                "temperature": temperature,
                "_meta": "tools",
                "_meta_ctx": {**(meta_ctx or {}), "round": rnd},
            }
            if reasoning_guard:
                body["_reasoning_guard"] = reasoning_guard
            if collection_bounds:
                body["_collection_bounds"] = tuple(collection_bounds)
            if model:
                body["model"] = model
            if max_tokens:
                body["max_tokens"] = max_tokens
            if reasoning:
                body["reasoning"] = reasoning
            if response_format and final_response:
                body["response_format"] = response_format
            msg, cost = self._stream(body, on_delta=on_delta)
            total_cost += cost

            assistant = {"role": "assistant", "content": msg.get("content")}
            tcs = msg.get("tool_calls") or []
            if tcs:
                assistant["tool_calls"] = tcs
            messages.append(assistant)

            if msg.get("loop_detected"):
                # Reasoning loop in a tool round: stop tools, drop reasoning, and
                # force the final JSON on the next round.
                force_final = True
                reasoning = {"enabled": False}
                continue

            if not tcs and response_format and not final_response:
                # Tool rounds are ordinary calls. Discard any unconstrained final
                # prose/JSON and buy exactly one tool-free schema-bound response.
                force_final = True
                reasoning = {"enabled": False}
                continue
            if not tcs:
                return {
                    "text": msg.get("content") or "",
                    "cost_usd": total_cost,
                    "rounds": rnd + 1,
                    "tool_calls": n_tool_calls,
                    "finish_reason": msg.get("finish_reason"),
                    "guard": self.guard.status(),
                    "loop_detected": bool(msg.get("loop_detected")),
                    "loop_abort_reason": msg.get("loop_abort_reason"),
                    "reasoning_policy_name": msg.get("reasoning_policy_name"),
                    "collection_limit": msg.get("collection_limit"),
                    "provider": msg.get("provider"),
                    "usage": msg.get("usage") or {},
                    "max_tokens": msg.get("requested_max_tokens"),
                    "reasoning_policy": msg.get("reasoning_policy"),
                    "content_chars": msg.get("content_chars"),
                    "collection_counts": msg.get("collection_counts") or {},
                    "reasoning_chars": msg.get("reasoning_chars"),
                }
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
                # Break identical-call thrash. A weak model repeats the exact same
                # call for rounds on end; the result cannot change, so reuse the
                # cached one instead of re-running the tool (saves the subprocess)
                # and tell it to converge.
                sig = name + "|" + json.dumps(args, sort_keys=True)
                seen[sig] = seen.get(sig, 0) + 1
                if sig in cache:
                    result = cache[sig]
                    redundant += 1
                else:
                    try:
                        result = executor(name, args)
                    except Exception as e:  # surface tool errors, don't crash
                        result = f"tool error: {e}"
                    cache[sig] = result
                if seen[sig] >= 3:
                    # Hard cutoff: the 2nd repeat already got the notice + full
                    # payload, and a reflexive re-verifier repeats anyway (live
                    # board 635: 4 identical lookups, the last right after its
                    # own "now write the final JSON"). From the 3rd repeat the
                    # steer REPLACES the payload -- there is nothing new to
                    # re-read, and each repeat is a paid round trip
                    # (2026-07-19 review §5.7).
                    result = (
                        f"NOTE: identical call repeated ({seen[sig]}x); the "
                        f"result was already provided twice and will not "
                        f"change. It is withheld this time. Use the answer "
                        f"you already have and output the final JSON now."
                    )
                elif seen[sig] >= 2:
                    result = (
                        f"NOTE: identical call repeated ({seen[sig]}x); the cached "
                        f"result is reused and will not change. Stop verifying and "
                        f"output the final JSON now.\n{result}"
                    )
                # Too many redundant or total tool calls -> stop offering tools next
                # round and force the model to commit to an answer.
                if redundant >= _MAX_REDUNDANT_TOOL_CALLS or n_tool_calls >= _MAX_TOTAL_TOOL_CALLS:
                    force_final = True
                if progress:
                    progress({"kind": "tool_result", "name": name, "output": str(result)[:600]})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "name": name,
                        "content": str(result)[:4000],
                    }
                )

        # Tool-round budget exhausted. Returning empty text here reads upstream as
        # "no JSON in reply" (a silent, expensive failure). Instead force one final
        # tool-free completion so the model commits to an answer we can parse.
        messages.append(
            {
                "role": "user",
                "content": "You have used your entire tool-call budget. Do NOT call any "
                "more tools. Output your final answer now as a single JSON "
                "object only.",
            }
        )
        body = {
            "messages": messages,
            "tools": tools,
            "tool_choice": "none",
            "temperature": temperature,
            "_meta": "tools-final",
            "_meta_ctx": {**(meta_ctx or {}), "round": "final"},
        }
        if reasoning_guard:
            body["_reasoning_guard"] = reasoning_guard
        if collection_bounds:
            body["_collection_bounds"] = tuple(collection_bounds)
        if model:
            body["model"] = model
        if max_tokens:
            body["max_tokens"] = max_tokens
        if reasoning:
            body["reasoning"] = reasoning
        if response_format:
            body["response_format"] = response_format
        msg, cost = self._stream(body, on_delta=on_delta)
        total_cost += cost
        messages.append({"role": "assistant", "content": msg.get("content")})
        return {
            "text": msg.get("content") or "",
            "cost_usd": total_cost,
            "rounds": max_rounds,
            "tool_calls": n_tool_calls,
            "finish_reason": msg.get("finish_reason"),
            "guard": self.guard.status(),
            "max_rounds_hit": True,
            "forced_final": True,
            "loop_detected": bool(msg.get("loop_detected")),
            "loop_abort_reason": msg.get("loop_abort_reason"),
            "reasoning_policy_name": msg.get("reasoning_policy_name"),
            "collection_limit": msg.get("collection_limit"),
            "provider": msg.get("provider"),
            "usage": msg.get("usage") or {},
            "max_tokens": msg.get("requested_max_tokens"),
            "reasoning_policy": msg.get("reasoning_policy"),
            "content_chars": msg.get("content_chars"),
            "collection_counts": msg.get("collection_counts") or {},
            "reasoning_chars": msg.get("reasoning_chars"),
        }
