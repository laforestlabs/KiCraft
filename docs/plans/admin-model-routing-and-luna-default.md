# Admin model routing + OpenAI Luna default — 2026-09-11

**Status:** proposed (for a separate implementing agent)
**Audience:** the agent implementing this plan
**Source:** operator decision, 2026-09-11 (revised: no escalation tier)

## Decision

1. Add an admin routing page on KiCraft.io that configures the active design
   model (a "switch") plus model behavior: max tokens, project budget, and
   thinking/reasoning.
2. Change the design default back to **OpenRouter**, routed to an **OpenAI
   "Luna"** model (native structured output). This becomes the reliability path.
3. Keep **DeepSeek direct** as the one alternative, using DeepSeek's official
   JSON-output guidance (the word "json" in the prompt + a concrete schema
   example to copy) and a `json_object` response format.

**No escalation, no hidden failover.** There are exactly two profiles — `luna`
and `deepseek` — and the one you pick is the one that runs. Escalation and
provider fallback are disabled (empty), so a run never silently switches models
inside itself. Simplicity is a requirement: no "pro" tier, no recovery ladder.

The hard cost/safety boundaries (API keys, kill switch, daily/total USD
ceilings) stay in `.env`. The admin page may only move the routing + behavior
knobs, never the secrets or the safety ceilings.

## Background (why)

DeepSeek's own API does **not** implement OpenAI structured outputs
(`response_format: {"type": "json_schema"}`). It returns HTTP 400
`"This response_format type is unavailable now"`; only `json_object` exists.
KiCraft's design stages (architecture, BOM, wiring) depend on strict JSON-schema
enforcement, which is the reliability mechanism those complex nested slots rely
on. Evidence gathered 2026-09-11:

- `json_schema` -> DeepSeek 400 "unavailable now".
- `json_object` -> accepted, but requires the word "json" in the prompt.
- Real single-brief canary with a `json_object` fallback: `intent` and
  `functional_spec` passed; `architecture` failed 3/3 with `invalid_schema`
  ("provider response did not satisfy the required JSON schema").

Conclusion: OpenAI models (native structured output) are the reliable default;
DeepSeek stays as a cost profile that runs without the hard guarantee.

## Current working-tree state (read this first)

There is **uncommitted** prototype work already in the working tree from the
earlier DeepSeek-direct attempt. The implementing agent must reconcile it, not
redo it blindly:

- `kicraft/server/config.py` — added `Settings.backend` ("openrouter" |
  "deepseek"), `Settings.deepseek_api_key`, and `DESIGN_PROFILES` entries with
  `backend`/`base_url` keys. `from_env()` resolves backend/base_url/key from the
  profile and requires `DEEPSEEK_API_KEY` for a deepseek backend.
  `for_review()`/`for_judge()` reset to OpenRouter. `redacted()` exposes
  `backend`. **The flash profile currently points at DeepSeek — this is the part
  to rework into the two-profile `luna`/`deepseek` set below.**
- `kicraft/server/client.py` — backend-aware client: gates the OpenRouter
  `provider` routing block, `usage.include`, `X-Title`, and cache-breakpoint to
  OpenRouter; uses the deepseek key for a deepseek backend; translates the
  OpenRouter `reasoning` dict to DeepSeek's `thinking` toggle (default
  disabled); reads `reasoning_content`/`prompt_cache_hit_tokens`; passes
  `reasoning_content` back across tool rounds; and **translates
  `response_format: json_schema` -> `json_object` for the deepseek backend while
  keeping the full schema for the in-stream guard and telemetry.** **This is
  correct and reusable; keep it.**
- `kicraft/server/smoketest.py`, `.env.example`, and tests (`tests/conftest.py`,
  `tests/test_client_provider.py`, `tests/test_stage_driver_retry.py`) were also
  touched. Keep the test additions that pin backend behavior; adjust the
  profile-related ones to the new two-profile set.
- `.env` was **reverted** to `KICRAFT_MODEL=deepseek/deepseek-v4-flash-0731`
  (OpenRouter) so the still-running old code stays consistent; `DEEPSEEK_API_KEY`
  is present, and `KICRAFT_ESCALATION_PROFILE=pro` /
  `KICRAFT_PROVIDER_FALLBACK_PROFILE=pro` are still set. The working tree and
  `.env` are therefore currently out of sync; do not restart services until this
  plan is implemented and `.env` matches the code. **As part of this change,
  clear `KICRAFT_ESCALATION_PROFILE` and `KICRAFT_PROVIDER_FALLBACK_PROFILE` to
  empty.**

## Target design

### Named design profiles (`kicraft/server/config.py`)

Replace the current `flash`/`pro` pair with exactly two profiles:

| profile | backend | model | base_url | provider_order | prompt cap | completion cap | notes |
|---|---|---|---|---|---|---|---|
| `luna` (default) | openrouter | `openai/<luna-slug>` (verify) | `https://openrouter.ai/api/v1` | cheapest verified provider | verify | verify | native structured output |
| `deepseek` | deepseek | `deepseek-flash` | `https://api.deepseek.com` | `[]` | 0.30 | 1.20 | peak cache-miss rates, conservative |

Rules:

- `luna` is the **default** (the value `_resolved_design_profile` returns when
  nothing overrides it). Change the `Settings.model` dataclass default and the
  `.env.example` default accordingly.
- `deepseek` keeps `provider_order=[]` and the client's `json_object` fallback
  path (already implemented) is what actually runs it.
- **Escalation and provider fallback are disabled.** The routing config does not
  expose them; `.env` must set `KICRAFT_ESCALATION_PROFILE=` and
  `KICRAFT_PROVIDER_FALLBACK_PROFILE=` (empty). With those empty the existing
  stage-runtime `with_design_profile` switch never fires, so the active profile
  is final. Deleting the underlying escalation code is optional cleanup, out of
  scope.

The `_resolved_design_profile()` env-conflict check (`KICRAFT_MODEL` /
`KICRAFT_PROVIDER_ORDER` / `KICRAFT_MAX_PRICE_*` must equal the profile) stays
for env-driven selection, but must be **skipped** when the routing config (Part
2) is the authority for the active profile.

### Part 2 — admin routing page

**Persistence.** Add a small durable config file, default
`~/.kicraft/routing.json` (env-overridable as `KICRAFT_ROUTING_CONFIG`). It
holds exactly the admin-controlled knobs and nothing else:

```json
{
  "active_profile": "luna",
  "max_tokens_per_call": 4096,
  "project_llm_budget_usd": 0.10,
  "design_reasoning_tokens": 0,
  "design_temperature": 0.0
}
```

- `active_profile` must name a `DESIGN_PROFILES` entry (`luna` or `deepseek`).
  Unknown/empty/missing -> ignore the file (fall back to env), and log once.
- Every other key maps 1:1 onto an existing `Settings` field.
- The file must **never** hold secrets or the hard safety ceilings (kill switch,
  daily/total USD). Keep those `.env`-only. Document this in the file header
  comment and enforce it in the load path (ignore/refuse any key outside the
  allowlist).

**Resolution.** In `Settings.from_env()`, after env resolution:

1. Load the routing config if present.
2. If `active_profile` is set, resolve the designer profile from it instead of
   `KICRAFT_DESIGN_PROFILE`, and skip the env `KICRAFT_MODEL`/price/order
   conflict checks (the routing config wins; a stale `.env` `KICRAFT_MODEL` must
   not crash startup).
3. Overlay the behavior keys over the env-derived values (`max_tokens_per_call`,
   `project_llm_budget_usd`, `design_reasoning_tokens`, `design_temperature`).

This means a flip takes effect on the **next** `Settings.from_env()` call with no
process restart. Both the web app and the build worker call `Settings.from_env()`
per run, so both pick it up.

**UI.** New `@ui.page("/admin/routing")` in `kicraft/server/routes_admin.py`,
gated by `_require_admin()` (re-check `is_admin` in the mutating handler, matching
the existing defense-in-depth pattern). Register it in `_admin_header`'s nav
list. Layout mirroring the other admin pages (`_admin_card_style`):

- **Active model** — a select/radio of the two profiles (`luna`, `deepseek`),
  with the backend/model/base_url shown per option. Mutating = write
  `active_profile` to the routing config.
- **Max tokens per call** — number input -> `max_tokens_per_call` (bounded to the
  existing sane range, see `KICRAFT_MAX_TOKENS_PER_CALL` semantics).
- **Project LLM budget** — number -> `project_llm_budget_usd`.
- **Thinking** — the design reasoning toggle -> `design_reasoning_tokens`
  (0 = disabled; a positive number enables with that budget). Label it plainly:
  "reasoning/thinking budget".
- **Design temperature** — number -> `design_temperature` (0..1, clamp).
- Show the current **resolved** settings (via `Settings.redacted()`) so the admin
  sees the effective values, including which profile is actually active and the
  backend it resolves to.
- A "Save" that atomically writes the JSON file (write-to-temp + rename) and a
  visible "takes effect on the next design run" note.

Never render or echo the API keys. The page reads no secrets.

### Part 3 — DeepSeek JSON guidance

DeepSeek's `json_object` mode has two requirements: the word "json" must appear
in the prompt, and the model benefits from a concrete example to copy. Both are
already largely present in `kicraft/server/stage_prompts.py` (the slot prompt
says "Output ONLY a single JSON object", embeds the Pydantic schema, and appends
`_worked_example(...)`).

Tasks:

1. Confirm the deepseek profile's prompt path still carries the word "json"
   verbatim in every stage prompt (it does today; add a regression test that
   asserts `"json"` appears case-insensitively in the built prompt).
2. Confirm `_worked_example` produces a concrete *instance* (not just the schema)
   for every stage. If any stage's example is empty or schema-only, add a
   minimal concrete example for it — the complex stages (architecture, BOM,
   wiring) matter most.
3. Keep the client's `json_object` translation (already implemented): for the
   deepseek backend, `response_format` becomes `{"type": "json_object"}` while
   the full schema stays available to the in-stream property guard and the
   telemetry. Do **not** send `json_schema` to DeepSeek.

## File-by-file change list

- `kicraft/server/config.py` — rework `DESIGN_PROFILES` to the two-profile
  `luna`/`deepseek` set; default profile `luna`; add routing-config load/overlay
  in `from_env`; keep the `backend`/`deepseek_api_key`/`for_review`/`for_judge`
  work already present.
- `kicraft/server/routing_config.py` (new) — `RoutingConfig` dataclass +
  `load()`/`save()` with an allowlist and temp+rename atomic write. Pure,
  unit-testable.
- `kicraft/server/routes_admin.py` — add `/admin/routing` page + nav entry.
- `kicraft/server/client.py` — keep the backend abstraction and `json_object`
  fallback as-is; no further change expected.
- `.env.example` — default `KICRAFT_DESIGN_PROFILE=luna`, clear the escalation
  and provider-fallback fields, document `DEEPSEEK_API_KEY` as optional
  (deepseek profile only), and note the routing config path.
- `.env` (production, operator or deploy step) — clear
  `KICRAFT_ESCALATION_PROFILE` and `KICRAFT_PROVIDER_FALLBACK_PROFILE`, and set
  `KICRAFT_MODEL`/`KICRAFT_DESIGN_PROFILE` to match `luna` (or rely on the
  routing config).
- `tests/` — new tests for: profile resolution (luna default, deepseek
  optional), routing-config load/overlay/skip-on-invalid, admin page gating,
  deepseek `json_object` translation (already partially present), prompt contains
  "json" + a concrete example, and that an empty escalation/fallback profile
  means no `with_design_profile` switch.

## Verification / acceptance

- Default `Settings.from_env()` (no routing config, clean env) resolves
  `design_profile == "luna"`, `backend == "openrouter"`, model = the Luna slug.
- With `routing.json` setting `active_profile == "deepseek"`, `Settings.from_env()`
  resolves `backend == "deepseek"`, `base_url == "https://api.deepseek.com"`,
  `model == "deepseek-flash"`, and a `DEEPSEEK_API_KEY` is required.
- A deepseek `_stream` payload contains `response_format: {"type": "json_object"}`
  and never `json_schema`; the OpenRouter-only fields (`provider`, `usage`,
  `X-Title`, cache breakpoint) are absent.
- Escalation and provider fallback are empty, and no path switches profiles
  mid-run.
- `/admin/routing` loads for an admin, 403s/redirects for a non-admin, and a save
  round-trips through the routing file and is reflected in the next
  `Settings.from_env()`.
- No secret or safety ceiling is readable or writable from the page or the
  routing file.
- Full deterministic test suite green; no live provider needed except the final
  canary.

## To verify before pinning (blocking)

- **Exact OpenRouter slug for "Luna"** and its cheapest provider + price caps, and
  that it supports `response_format: json_schema` with `strict`, tools (BOM),
  streaming, and the reasoning controls the roles use. Use
  `kicraft.cli.model_preflight` / `kicraft.cli.provider_bench` against the live
  OpenRouter key before committing the `luna` profile values.
- DeepSeek peak/off-peak price caps on the current official pricing page (the
  0.30/1.20 used above are the 2026-09-11 peak cache-miss rates; re-check).

## Non-goals

- No changes to the electrical-review or Class-J judge roles (still
  `minimax/minimax-m3` via OpenRouter).
- No escalation tier, no provider fallback, no recovery ladder — two profiles
  only, and the active one is final. (Deleting the dormant escalation code is
  optional cleanup, not required.)
- No re-engineering of the stage schemas or the retry/recovery safety net.
- No changes to the hard cost/safety ceilings or secrets.
