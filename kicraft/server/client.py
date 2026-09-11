"""Capped OpenRouter chat client: every model call enforces B0 cost-safety.

Call flow per completion: `SpendGuard.preflight()` (kill switch + ceilings) ->
bounded request (`max_tokens`) -> `SpendGuard.record()` (actual cost). Both the
plain `chat()` and the `chat_with_tools()` loop go through `_complete()`, so the
caps cannot be bypassed by a new code path.
"""

from __future__ import annotations
import json
import os
import re
import time
from dataclasses import replace


import requests

from .config import CollectionBound, DESIGN_PROFILES, ReasoningGuardPolicy, Settings
from .spend_guard import SpendGuard

# Transient failures worth a bounded retry (before any token is streamed): all
# 5xx + 429 (rate limit) on the HTTP status, plus connection resets / timeouts.
_RETRY_NETWORK_EXC = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def classify_provider_exception(exc: BaseException) -> dict:
    """Return redacted stable provider/transport facts without response bodies."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    headers = getattr(response, "headers", {}) or {}
    request_id = headers.get("x-request-id") or headers.get("x-openrouter-request-id")
    error_code = None
    if response is not None:
        try:
            payload = response.json()
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                raw_code = error.get("code")
                if isinstance(raw_code, (str, int)):
                    error_code = str(raw_code)[:96]
        except (ValueError, TypeError):
            pass
    text = str(exc).lower()
    if status is None and isinstance(exc, requests.exceptions.HTTPError):
        match = re.search(r"\b([1-5][0-9]{2})\b", text)
        if match:
            status = int(match.group(1))
    if status == 429:
        kind = "provider_rate_limited"
    elif status is not None and status >= 500:
        kind = "provider_upstream_5xx"
    elif status in {401, 403}:
        kind = "provider_auth"
    elif status is not None and 400 <= status < 500:
        if "response_format" in text or "response format" in text:
            kind = "provider_response_format_rejected"
        elif any(word in text for word in ("tool", "reasoning", "schema", "capability")):
            kind = "provider_capability_rejected"
        else:
            kind = "provider_request_rejected"
    elif isinstance(exc, requests.exceptions.Timeout):
        kind = "transport_timeout"
    elif isinstance(exc, requests.exceptions.ChunkedEncodingError) or "stream" in text:
        kind = "transport_stream_interrupted"
    elif isinstance(exc, requests.exceptions.ConnectionError):
        kind = "transport_connection"
    elif isinstance(exc, requests.exceptions.HTTPError):
        kind = "transport_stream_interrupted" if response is None else "provider_unknown"
    else:
        kind = "provider_unknown"
    return {
        "failure_kind": kind,
        "http_status": status,
        "error_code": error_code,
        "request_id": str(request_id)[:128] if request_id else None,
    }


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
    """Enforce JSON syntax, collection bounds and forbidden schema properties."""

    def __init__(self, bounds: tuple[CollectionBound, ...], response_format=None):
        self._counters = [_TopLevelArrayCounter(bound) for bound in bounds]
        schema = ((response_format or {}).get("json_schema") or {}).get("schema")
        structured = bool(bounds) or isinstance(schema, dict) or (response_format or {}).get("type") in {
            "json_schema", "json_object"
        }
        self._properties = _StreamingPropertyGuard(schema) if structured else None

    def consume(self, text: str) -> tuple[str, dict | None]:
        for index, char in enumerate(text):
            if self._properties is not None:
                violation = self._properties.consume(char)
                if violation is not None:
                    return text[:index], violation
            for counter in self._counters:
                overflow = counter.consume(char)
                if overflow is not None:
                    return text[:index], overflow
        return text, None

    def counts(self) -> dict[str, int]:
        return {counter.bound.field: counter.count for counter in self._counters}


class _StreamingPropertyGuard:
    """Incrementally validate JSON grammar, decoding only bounded key tokens.

    Schema projection is deliberately conservative: unions retain every branch,
    and unsupported constraints remain the final schema validator's job.
    """

    _KEY_LIMIT = 4096

    def __init__(self, schema):
        self.schema = schema
        self.stack: list[dict] = []
        self.in_string = False
        self.escape = False
        self.unicode_digits = 0
        self.key_token: list[str] | None = None
        self.string_is_key = False
        self.number = None
        self.literal = ""
        self.literal_index = 0
        self.root_state = "value"
        self.wrapper = "start"
        self.fence_prefix = ""
        self.fenced = False
        self.offset = -1
        self.line = 1
        self.column = 0

    def _child(self, schema, key, seen=()):
        if not isinstance(schema, dict):
            return schema if isinstance(schema, bool) else True
        constraints = []
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/") and ref not in seen:
            target = self.schema
            try:
                for part in ref[2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
            except (KeyError, TypeError):
                target = True
            constraints.append(self._child(target, key, (*seen, ref)))
        for combinator in ("anyOf", "oneOf"):
            if isinstance(schema.get(combinator), list):
                branches = [self._child(branch, key, seen) for branch in schema[combinator]]
                constraints.append(self._combine(branches, "anyOf"))
        for branch in schema.get("allOf", []):
            constraints.append(self._child(branch, key, seen))
        if isinstance(key, int):
            prefix = schema.get("prefixItems", [])
            items = schema.get("items", True)
            if key < len(prefix):
                constraints.append(prefix[key])
            elif isinstance(items, list):
                constraints.append(
                    items[key] if key < len(items) else schema.get("additionalItems", True)
                )
            else:
                constraints.append(items)
        else:
            properties = schema.get("properties", {})
            matches = [properties[key]] if key in properties else []
            for pattern, value in schema.get("patternProperties", {}).items():
                try:
                    if re.search(pattern, key):
                        matches.append(value)
                except re.error:
                    # A regex dialect we cannot interpret must not close a branch.
                    matches.append(True)
            constraints.append(
                self._combine(matches, "allOf")
                if matches
                else schema.get("additionalProperties", True)
            )
        return self._combine(constraints, "allOf")

    @staticmethod
    def _combine(branches, operator):
        decisive, neutral = (False, True) if operator == "allOf" else (True, False)
        if any(branch is decisive for branch in branches):
            return decisive
        remaining = [branch for branch in branches if branch is not neutral]
        if not remaining:
            return neutral
        return remaining[0] if len(remaining) == 1 else {operator: remaining}

    def _value(self):
        if not self.stack:
            return self.schema, "$"
        parent = self.stack[-1]
        key = parent["key"] if parent["kind"] == "{" else parent["index"]
        if key is None:
            return True, parent["path"] + "[?]"
        suffix = (
            f"[{key}]"
            if isinstance(key, int)
            else ("." + key if key.isidentifier() else "[" + json.dumps(key) + "]")
        )
        return self._child(parent["schema"], key), parent["path"] + suffix

    def _syntax(self, error: str) -> dict:
        return {
            "limit_scope": "syntax",
            "field": self.stack[-1]["path"] if self.stack else "$",
            "syntax_error": error,
            "character_offset": self.offset,
            "line": self.line,
            "column": self.column,
        }

    def _finish_value(self):
        if self.stack:
            self.stack[-1]["state"] = "comma_or_end"
        else:
            self.root_state = "end"

    def _string(self, char: str) -> dict | None:
        if self.key_token is not None:
            if len(self.key_token) < self._KEY_LIMIT:
                self.key_token.append(char)
            else:
                # Oversized keys remain syntactically checked but cannot safely
                # participate in schema projection or duplicate detection.
                self.key_token = None
        if self.unicode_digits:
            if char not in "0123456789abcdefABCDEF":
                return self._syntax("Expected a hexadecimal digit in Unicode escape")
            self.unicode_digits -= 1
        elif self.escape:
            self.escape = False
            if char == "u":
                self.unicode_digits = 4
            elif char not in '"\\/bfnrt':
                return self._syntax("Invalid JSON string escape")
        elif char == "\\":
            self.escape = True
        elif char == '"':
            self.in_string = False
            if self.string_is_key:
                parent = self.stack[-1]
                key = json.loads("".join(self.key_token)) if self.key_token is not None else None
                parent["key"] = key
                parent["state"] = "colon"
                if key is not None:
                    if key in parent["keys"]:
                        return self._syntax("Duplicate object key " + json.dumps(key))
                    parent["keys"].add(key)
                    if self._child(parent["schema"], key) is False:
                        return {"limit_scope": "property", "field": parent["path"], "property": key}
            else:
                self._finish_value()
            self.key_token = None
        elif ord(char) < 0x20:
            return self._syntax("Unescaped control character in JSON string")
        return None

    def _number(self, char: str) -> dict | None:
        state = self.number
        digit = "0" <= char <= "9"
        if state == "sign" and digit:
            self.number = "zero" if char == "0" else "integer"
        elif state in {"integer", "fraction", "exponent"} and digit:
            pass
        elif state in {"zero", "integer"} and char == ".":
            self.number = "dot"
        elif state in {"zero", "integer", "fraction"} and char in "eE":
            self.number = "e"
        elif state == "e" and char in "+-":
            self.number = "exponent_sign"
        elif state in {"dot", "e", "exponent_sign"} and digit:
            self.number = "fraction" if state == "dot" else "exponent"
        elif state in {"zero", "integer", "fraction", "exponent"} and char in " \t\r\n,]}":
            self.number = None
            self._finish_value()
        else:
            return self._syntax("Invalid JSON number")
        return None

    def consume(self, char: str) -> dict | None:
        self.offset += 1
        self.column += 1
        violation = self._consume(char)
        if char == "\n":
            self.line += 1
            self.column = 0
        return violation

    def _consume(self, char: str) -> dict | None:
        if self.in_string:
            return self._string(char)
        if self.literal:
            if char != self.literal[self.literal_index]:
                return self._syntax("Invalid JSON literal; expected " + self.literal)
            self.literal_index += 1
            if self.literal_index == len(self.literal):
                self.literal = ""
                self._finish_value()
            return None
        if self.number is not None:
            violation = self._number(char)
            if violation is not None or self.number is not None:
                return violation
            # The delimiter terminates the number AND belongs to its container.

        if self.wrapper in {"start", "opening"}:
            # The final extractor permits leading prose and optional bare/json
            # fences. Before a root container starts, a later object can still
            # salvage that preamble. Never return here once JSON has begun.
            if char in "{[":
                self.fenced |= self.fence_prefix in {"```", "```json"}
                self.wrapper = "body"
            else:
                if self.wrapper == "opening":
                    if self.fence_prefix in {"```", "```json"} and char.isspace():
                        self.fenced = True
                        self.wrapper = "start"
                        self.fence_prefix = ""
                    elif "```json".startswith(self.fence_prefix + char):
                        self.fence_prefix += char
                    else:
                        self.wrapper = "start"
                        self.fence_prefix = ""
                elif char == "`":
                    self.wrapper = "opening"
                    self.fence_prefix = "`"
                return None

        if not self.stack and self.root_state == "end":
            if self.wrapper == "closed":
                # _extract_json ignores prose outside a complete fenced object.
                return None
            if self.wrapper == "closing":
                if char != "`":
                    return self._syntax("Expected a closing JSON markdown fence")
                self.fence_prefix += char
                if self.fence_prefix == "```":
                    self.wrapper = "closed"
                return None
            if char.isspace():
                return None
            if self.fenced and char == "`":
                self.wrapper = "closing"
                self.fence_prefix = "`"
                return None
            return self._syntax("Unexpected content after the JSON value")
        if char in " \t\r\n":
            return None

        parent = self.stack[-1] if self.stack else None
        state = parent["state"] if parent else self.root_state
        if parent:
            end = "}" if parent["kind"] == "{" else "]"
            if char == end and state in {"key_or_end", "value_or_end", "comma_or_end"}:
                self.stack.pop()
                self._finish_value()
                return None
            if state == "comma_or_end":
                if char != ",":
                    return self._syntax("Expected ',' or " + repr(end) + " after a value")
                parent["state"] = "key" if parent["kind"] == "{" else "value"
                parent["key"] = None
                parent["index"] += 1
                return None
            if state == "colon":
                if char != ":":
                    return self._syntax("Expected ':' after an object key")
                parent["state"] = "value"
                return None
            if state in {"key", "key_or_end"}:
                if char != '"':
                    return self._syntax("Expected a quoted object key")
                self.in_string = True
                self.string_is_key = True
                self.key_token = ['"']
                return None

        if char == '"':
            self.in_string = True
            self.string_is_key = False
        elif char in "{[":
            schema, path = self._value()
            self.stack.append(
                {
                    "kind": char,
                    "schema": schema,
                    "path": path,
                    "state": "key_or_end" if char == "{" else "value_or_end",
                    "key": None,
                    "index": 0,
                    "keys": set() if char == "{" else None,
                }
            )
        elif char in "tfn":
            self.literal = "true" if char == "t" else ("false" if char == "f" else "null")
            self.literal_index = 1
        elif char == "-" or "0" <= char <= "9":
            self.number = "sign" if char == "-" else ("zero" if char == "0" else "integer")
        else:
            return self._syntax("Expected a JSON value")
        return None


def _complete_bounded_wiring_json(content: str, limit: dict | None) -> str | None:
    """Close a valid top-level `pins` prefix when the exact unit bound is reached."""
    if not limit or limit.get("field") != "pins" or limit.get("limit_scope"):
        return None
    prefix = content.rstrip()
    if not prefix.endswith(","):
        return None
    completed = prefix[:-1] + "]}"
    try:
        payload = json.loads(completed)
    except json.JSONDecodeError:
        return None
    pins = payload.get("pins") if isinstance(payload, dict) else None
    if not isinstance(pins, list) or len(pins) != int(limit.get("configured_total") or 0):
        return None
    return completed


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
        self.unique_values: set[tuple[str, ...]] = set()
        # Each ASCII field-name character can be encoded as a six-byte JSON
        # Unicode escape. Longer tokens cannot name this bounded collection.
        self.field_token_limit = 6 * len(bound.field) + 2

    def _finish_member(self) -> dict | None:
        if self.member_buf is None:
            return None
        raw = "".join(self.member_buf).strip()
        self.member_buf = None
        if not raw:
            return None
        try:
            member = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(member, dict):
            return None
        if self.bound.unique_keys and all(key in member for key in self.bound.unique_keys):
            values = tuple(str(member[key]) for key in self.bound.unique_keys)
            if values in self.unique_values:
                return {
                    "field": self.bound.field,
                    "observed_count": self.count,
                    "configured_total": self.bound.total,
                    "limit_scope": "duplicate",
                    "unique_keys": list(self.bound.unique_keys),
                    "duplicate_values": list(values),
                }
            self.unique_values.add(values)
        if self.bound.per_group is None or self.bound.group_key is None:
            return None
        if self.bound.group_key not in member:
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
            if self.capture_string:
                if len(self.string_buf) < self.field_token_limit:
                    self.string_buf += char
                else:
                    self.capture_string = False
                    self.string_buf = ""
            if self.escape:
                self.escape = False
                return None
            if char == "\\":
                self.escape = True
                return None
            if char == '"':
                self.in_string = False
                self.completed_string = None
                if self.capture_string:
                    try:
                        self.completed_string = json.loads(self.string_buf)
                    except json.JSONDecodeError:
                        pass
                self.capture_string = False
                self.string_buf = ""
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
            if self.bound.unique_keys or self.bound.group_key is not None:
                self.member_buf = [char]

        if char == '"':
            self.in_string = True
            self.capture_string = self.stack == ["{"]
            self.string_buf = '"' if self.capture_string else ""
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

    def with_design_profile(self, profile_name: str) -> "CappedOpenRouterClient":
        """Clone this route while retaining the run's one shared spend guard."""
        profile = DESIGN_PROFILES[profile_name]
        settings = replace(
            self.s,
            model=str(profile["model"]),
            design_profile=profile_name,
            provider_order=list(profile["provider_order"]),
            provider_allow_fallbacks=False,
            max_price_prompt=float(profile["max_price_prompt"]),
            max_price_completion=float(profile["max_price_completion"]),
        )
        return CappedOpenRouterClient(settings, guard=self.guard)

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

    def _configured_call_ceiling_usd(
        self,
        payload: dict,
        reasoning_guard: ReasoningGuardPolicy | None,
    ) -> float:
        """Conservative price-cap reservation for one outbound completion."""
        prompt_chars = len(
            json.dumps(
                payload.get("messages") or [],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        prompt_tokens = max(1, (prompt_chars + 3) // 4)
        output_tokens = max(0, int(payload.get("max_tokens") or 0))
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("enabled") is not False:
            reasoning_tokens = reasoning.get("max_tokens")
            if reasoning_tokens is None and reasoning_guard is not None:
                reasoning_tokens = reasoning_guard.hard_max_tokens
            output_tokens += max(0, int(reasoning_tokens or 0))
        prompt_price = max(0.0, float(getattr(self.s, "max_price_prompt", 0.0) or 0.0))
        completion_price = max(
            0.0,
            float(getattr(self.s, "max_price_completion", 0.0) or 0.0),
        )
        return (prompt_tokens * prompt_price + output_tokens * completion_price) / 1_000_000

    def _estimated_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate using the selected provider profile before model-family fallback."""
        prompt_price = max(
            0.0,
            float(getattr(self.s, "max_price_prompt", 0.0) or 0.0),
        )
        completion_price = max(
            0.0,
            float(getattr(self.s, "max_price_completion", 0.0) or 0.0),
        )
        if prompt_price or completion_price:
            return (input_tokens * prompt_price + output_tokens * completion_price) / 1_000_000
        return estimate_cost(model, input_tokens, output_tokens)

    def _open_stream(
        self,
        payload: dict,
        *,
        run_id: str | None = None,
        call_ceiling_usd: float = 0.0,
    ):
        """POST and return a streamed Response with a non-retryable status.

        Retries 5xx/network failures before any token is consumed. Rate limits
        are returned immediately to the top-level retry action; consuming local
        retries on 429 extends the provider's rolling window. Authenticated and
        other non-transient 4xx failures are never retried.
        """
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
                self.guard.preflight(call_ceiling_usd, run_id)
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.s.request_timeout_s,
                    stream=True,
                )
                if resp.status_code == 429:
                    limited = requests.exceptions.HTTPError(
                        f"{resp.status_code} {resp.reason}", response=resp
                    )
                    resp.close()
                    raise limited
                if resp.status_code >= 500:
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
                # Only 5xx HTTP errors and network failures are retried here.
                # Every 4xx, including 429, belongs to the top-level action.
                code = getattr(getattr(e, "response", None), "status_code", None)
                is_http = isinstance(e, requests.exceptions.HTTPError)
                if is_http and code is not None and code < 500:
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
        call_ceiling_usd = self._configured_call_ceiling_usd(payload, reasoning_guard)
        run_id = meta_ctx.get("run_id")
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
            received_content_chars = 0
            reasoning_recent = ""
            collection_guard = _StreamingCollectionGuard(
                collection_bounds, payload.get("response_format")
            )
            stream_t0 = time.monotonic()
            try:
                resp = self._open_stream(
                    payload,
                    run_id=str(run_id) if run_id else None,
                    call_ceiling_usd=call_ceiling_usd,
                )
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
                            status = error.get("code") if isinstance(error, dict) else None
                            try:
                                status = int(status)
                            except (TypeError, ValueError):
                                status = None
                            error_response = None
                            if status is not None and 100 <= status <= 599:
                                error_response = requests.Response()
                                error_response.status_code = status
                            stream_error = requests.exceptions.HTTPError(
                                f"OpenRouter stream error: {detail}",
                                response=error_response,
                            )
                            stream_error._kicraft_in_band = True
                            raise stream_error
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
                                received_content_chars += len(delta["content"])
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
            except (*_RETRY_NETWORK_EXC, requests.exceptions.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if isinstance(exc, requests.exceptions.HTTPError) and not getattr(
                    exc, "_kicraft_in_band", False
                ):
                    raise
                if isinstance(exc, requests.exceptions.HTTPError) and status == 429:
                    raise
                if (
                    isinstance(exc, requests.exceptions.HTTPError)
                    and status is not None
                    and status != 429
                    and status < 500
                ):
                    raise
                prompt_chars = len(
                    json.dumps(
                        payload.get("messages") or [],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                retry_input_tokens = max(1, prompt_chars // 4)
                retry_output_tokens = max(1, (reasoning_chars + content_chars) // 4)
                retry_cost = self._estimated_cost(
                    payload["model"], retry_input_tokens, retry_output_tokens
                )
                self.guard.record(
                    payload["model"],
                    retry_input_tokens,
                    retry_output_tokens,
                    retry_cost,
                    meta={
                        "phase": meta_phase,
                        "finish_reason": "stream_retry_discarded",
                        "http_status": status,
                        "content_chars": content_chars,
                        "reasoning_chars": reasoning_chars,
                        **meta_ctx,
                    },
                )
                if stream_attempt >= max_stream_retries:
                    raise
                time.sleep(stream_backoff * (2**stream_attempt))
                continue
            break  # stream ended cleanly or by a client-owned policy abort

        content_text = "".join(content)
        bounded_completion = _complete_bounded_wiring_json(
            content_text,
            collection_limit,
        )
        if bounded_completion is not None:
            content_text = bounded_completion
            content_chars = len(content_text)
            finish = "stop"
        elif loop_abort_reason:
            finish = "reasoning_loop"
        elif collection_limit:
            finish = "collection_limit"
        msg = {
            "role": "assistant",
            "content": content_text or None,
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
            # The provider generated the entire last delta, even the suffix
            # discarded by the guard. Charge that paid partial output as well.
            out_tok = out_tok or max(1, (reasoning_chars + received_content_chars) // 4)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        cost = float(usage.get("cost") or 0.0)
        if cost <= 0.0:  # never record 0 for real spend, or the ceiling under-counts
            cost = self._estimated_cost(payload["model"], in_tok, out_tok)
        response_policy = (payload.get("response_format") or {}).get("json_schema") or {}
        rec_meta = {
            "phase": meta_phase,
            "profile": getattr(self.s, "design_profile", "custom"),
            "provider": provider,
            "finish_reason": finish,
            "cached_tokens": int(cached or 0),
            "loop_detected": bool(loop_abort_reason),
            "bounded_collection_completed": bounded_completion is not None,
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
            if response_format:
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

            if not tcs or msg.get("collection_limit"):
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
