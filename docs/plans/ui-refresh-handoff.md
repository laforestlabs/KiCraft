# KiCraft UI refresh — handoff (2026-07-01)

Refreshes the kicraft.io web UI: kills the ugly amber "ask a question" box,
introduces a real theme system (one source of truth for color), rebrands the
palette to **dark + circuit-green**, and adds depth/typography. This doc is the
map for whoever picks it up next.

## Why (the ask)

User complaints, verbatim: the clarifying-question dialogs looked "cluttered and
flat," it was "hard to tell what is asking you to do and where to enter the
info," and "the color is terribly ugly." Root causes found during investigation:

- The clarifying-question panel (`web.py`) was an **inline amber-on-brown card**
  (`#1f1300`/`#92400e`) with yellow text and a bare underline input — no signal
  for "type here," clashing with the slate-blue everywhere else.
- **No theme system existed.** ~60 neutral hex literals in `web.py` alone, plus
  more across `routes_admin.py`, `stagetabs.py`, and four `static/*.css` files —
  so "the colors are ugly" could not be fixed in one place.
- Inputs used Quasar's default `standard` variant = just an underline → the
  "where do I enter info?" problem.

## Decisions (chosen by the user)

1. **Question UX → chat-style bubbles.** Agent question as a bubble, tappable
   quick-pick chips, a clearly-outlined reply box. (Not a modal; not the old
   inline panel.)
2. **Look → dark + a branded accent → circuit-green `#4ade80`.** Its own
   identity, not stock Tailwind-blue.
3. **Scope → full UX pass** (question UX + all dialogs + theme seed + broader
   surfaces + landing).

## Architecture — how the theme works

`kicraft/server/theme.py` is the **single source of truth**:

- **Palette as Python constants** (`BRAND`, `PRIMARY`, `BG`, `SURFACE`,
  `RAISED`, `BORDER`, `TEXT`, `MUTED`, …). Import these instead of typing hex.
- **One shared stylesheet**, injected via `ui.add_head_html(..., shared=True)`
  in `theme.install()`, called once at import from `web.py`. `shared=True`
  appends to a class-level global in NiceGUI, so it lands on **every page**
  (app + admin + landing) with **zero edits** to the ~20 scattered
  `ui.dark_mode().enable()` sites.
- **Overrides Quasar's `--q-*` brand vars** (`--q-primary`, `--q-positive`,
  `--q-negative`, …) with `!important`. That's why every existing
  `.props("color=primary")` button follows the theme with no per-site change.
- **CSS custom properties** (`--kc-bg`, `--kc-surface`, `--kc-brand`, …) on
  `:root`. Inline `.style("background:var(--kc-surface)")` call sites resolve
  these, so a future palette change is a **one-file edit** in `theme.py`.

### Two greens on purpose

- `BRAND = #4ade80` — the bright signature. Used on dark surfaces: focus rings,
  links, the question card accent, active states, icons.
- `PRIMARY = #16a34a` (green-600) — what Quasar paints **behind white button
  text**. Bright `#4ade80` under white text fails contrast; `#16a34a` gives
  ~3.3:1 so filled `color=primary` buttons stay legible.

**To retune the green:** edit `BRAND` / `PRIMARY` (and the `--kc-ring` rgba) in
`theme.py`. Nothing else needs touching for in-app surfaces. The **landing** and
**chart** palettes are separate concrete-hex copies (see caveats) — update those
by hand if you want them to track.

## Palette

| Token | Hex | Role |
| --- | --- | --- |
| `BG` | `#0b0f14` | app background (near-black neutral) |
| `SURFACE` | `#12171f` | cards, dialogs, panels |
| `RAISED` | `#1a212b` | inputs, chips, raised rows |
| `BORDER` | `#232c38` | hairline borders |
| `BORDER_STRONG` | `#2e3a49` | input outlines |
| `BRAND` | `#4ade80` | signature circuit-green (accents on dark) |
| `PRIMARY` | `#16a34a` | filled buttons behind white text |
| `TEXT` / `MUTED` / `DIM` | `#e8eef5` / `#9aa7b5` / `#657085` | text tiers |
| `SUCCESS`/`WARNING`/`ERROR` | `#34d399`/`#fbbf24`/`#f87171` | semantic (kept) |

Fonts: **Inter** (UI) + **JetBrains Mono** (code/BOM) via Google Fonts, with
system-ui fallback.

## What changed (file by file)

- **`kicraft/server/theme.py`** *(new)* — palette constants, the global
  stylesheet (brand-var overrides, depth, outlined inputs + green focus ring,
  scrollbars, and the `kc-qcard` / `kc-bubble` / `kc-stage-pill` / `kc-qchip`
  chat classes), and `install()`.
- **`kicraft/server/web.py`**
  - `_theme.install()` wired in after the static mounts.
  - `build_question_panel()` rewritten as the **chat-style card** (agent bubble,
    stage pill, outlined answer field, quick-pick chips, Enter-to-submit).
    Submit/resume logic unchanged — presentation only.
  - **74** neutral hex literals → `var(--kc-*)` tokens (body bg, panels, all six
    dialogs: support, delete, clone, re-run, self-eval, apply-rules).
- **`kicraft/server/routes_admin.py`** — 31 neutral literals → tokens. Chart
  constants `_CHART_AXIS` / `_CHART_GRID` bumped to concrete new hex (**not**
  tokens — see caveat).
- **`kicraft/server/stagetabs.py`** — inspector/log-pane neutral backgrounds →
  tokens. Per-stage accent rainbow + semantic status colors **kept**.
- **`static/kc_onboarding.css`** — composer accents blue → circuit-green;
  neutrals → tokens.
- **`static/kc_follow.css`** — table text → tokens.
- **`static/kc_landing.css`** — full rebrand: `--kc-blue`/`--kc-violet` renamed
  to `--kc-brand`/`--kc-accent`, palette shifted, hero gradient now
  **green → cyan**, glows greened.

## Caveats / gotchas (read before editing)

1. **echarts can't use `var()`.** echart option dicts render to `<canvas>`,
   which does not resolve CSS custom properties. `routes_admin.py`
   `_CHART_AXIS`/`_CHART_GRID` and the chart series palettes are deliberately
   **concrete hex**. Never sweep those to `var(--kc-*)` — it silently breaks
   chart colors. This is why the admin sweep skipped line ~65.
2. **`theme.py` CSS is inside an f-string.** Every literal `{`/`}` in the CSS is
   escaped as `{{`/`}}`. If you add rules, double the braces or it won't import.
3. **Chip class collision.** `kc_onboarding.css` already owns `.kc-chip` (the
   composer's "Surprise me" suggestion chips). The question quick-picks use a
   distinct **`kc-qchip`** class to avoid stomping it.
4. **Landing scopes its own vars.** `.kc-landing { --kc-* }` shadows the global
   tokens within the marketing page subtree (same values now). Its font stack
   falls back through `var(--kc-font-sans)`.
5. **Semantic amber left intact.** The "⚠ Fabricable, with a caution" banner in
   the fab tab (`web.py` ~5327) is a real warning state — correctly still amber.

## Verification status

- ✅ All touched modules compile; `web.py` + `routes_admin.py` import; theme
  confirmed present on the shared head.
- ✅ CSS-faithful preview generated from the real `theme.py` stylesheet
  (`scratchpad/gen_theme_preview.py` → `theme_preview.html`) — shows the palette
  and before/after of the question card.
- ⚠️ **No live-app screenshot** — this environment has no headless browser
  (no playwright/selenium/chrome). The remaining check is visual-on-deploy.

## Deploy

Presentation-only changes → restart the web process only; the **build worker
does not need restarting**:

```
deploy/restart-web.sh
```

Then eyeball: a real design's clarifying-question step, a confirmation dialog,
the landing page, and an admin chart page (confirm charts still render colored).

## Follow-ups (not done)

- **Accent hues in `web.py`/`stage_diagram.py`** — `#60a5fa`/`#a78bfa` still mark
  "current leaf" vs "parent board" and the architecture-diagram node types.
  These are intentional *semantic* two-color distinctions; revisit if you want
  them on-brand (e.g. brand-green vs a cyan).
- **`kc_follow.css` scrollbar** `#334155` thumb — cosmetic, could tokenize.
- **Charts** — if the green rebrand should reach data-viz, retune the concrete
  chart palettes in `routes_admin.py` by hand (canvas, so no tokens).
- **Live visual QA** across breakpoints (`kc_mobile.css` untouched — it's
  layout-only, no color).
- Consider a light-theme variant later (all inline colors already tokenized, so
  the groundwork is mostly done).
