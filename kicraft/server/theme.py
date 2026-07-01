"""KiCraft visual theme — the single source of truth for the app's look.

Historically every color was a hard-coded hex literal scattered across
``web.py`` / ``stagetabs.py`` / ``routes_admin.py`` and four ``static/*.css``
files, so "the colors are ugly" could not be fixed in one place. This module
fixes that: the palette lives here as Python constants (import them instead of
typing ``#0f172a``), and :func:`install` injects one shared stylesheet that
themes every page at once.

Design: a refined **dark** theme with a single **circuit-green** brand accent.
Two greens on purpose — the bright ``BRAND`` (#4ade80) reads well on dark
surfaces (focus rings, links, active states, icons); the deeper ``PRIMARY``
(#16a34a) is what Quasar paints behind *white* button text, so filled
``color=primary`` buttons stay legible.

The stylesheet also overrides Quasar's ``--q-*`` brand variables, so existing
``.props("color=primary")`` / ``color=positive`` / ``color=negative`` call
sites follow the theme with no per-site edits. It is injected with
``shared=True`` (a class-level global in NiceGUI), so a single call at import
covers all pages including admin.
"""

from __future__ import annotations

from nicegui import ui

# --- palette -----------------------------------------------------------------
# Import these in Python instead of re-typing hex. CSS mirrors them as
# custom properties below; keep the two in sync.

BG = "#0b0f14"            # app background — near-black neutral (warmer than slate)
SURFACE = "#12171f"       # cards, dialogs, panels
RAISED = "#1a212b"        # inputs, chips, raised rows
BORDER = "#232c38"        # hairline borders / dividers
BORDER_STRONG = "#2e3a49"  # input outlines, emphasized edges

BRAND = "#4ade80"         # signature circuit-green (accents on dark surfaces)
BRAND_STRONG = "#22c55e"  # hover / gradient stop
PRIMARY = "#16a34a"       # filled buttons behind white text (legible green-600)

TEXT = "#e8eef5"          # primary text / headings
MUTED = "#9aa7b5"         # secondary labels
DIM = "#657085"           # hints, timestamps, tertiary info

SUCCESS = "#34d399"       # kept — reads distinct from brand at a glance
WARNING = "#fbbf24"
ERROR = "#f87171"
INFO = "#38bdf8"

# Fonts: Inter for UI (metrically close to system-ui, low layout risk),
# JetBrains Mono for the code / BOM / diagnostics panes.
FONT_SANS = ('"Inter", ui-sans-serif, system-ui, -apple-system, '
             '"Segoe UI", Roboto, sans-serif')
FONT_MONO = ('"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, '
             'Consolas, monospace')

_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@400;500;600;700&'
    'family=JetBrains+Mono:wght@400;500&display=swap">'
)

# --- stylesheet --------------------------------------------------------------
# Kept deliberately surgical: brand-var overrides + depth + a clear input box +
# a small set of reusable classes (chat bubble, chip, stage pill). Existing
# cards that carry an inline background win over the un-!important .q-card rule
# on purpose — they migrate to tokens as their call sites are touched.

_CSS = f"""
<style>
:root {{
  /* Quasar brand vars — themes every color=primary/positive/negative/... site.
     !important so we win regardless of stylesheet load order. */
  --q-primary: {PRIMARY} !important;
  --q-positive: {PRIMARY} !important;
  --q-negative: #ef4444 !important;
  --q-warning: #f59e0b !important;
  --q-info: {INFO} !important;
  --q-dark: {SURFACE} !important;
  --q-dark-page: {BG} !important;

  /* KiCraft tokens — the app's palette, referenced as var(--kc-*). */
  --kc-bg: {BG};
  --kc-surface: {SURFACE};
  --kc-raised: {RAISED};
  --kc-border: {BORDER};
  --kc-border-strong: {BORDER_STRONG};
  --kc-brand: {BRAND};
  --kc-brand-strong: {BRAND_STRONG};
  --kc-text: {TEXT};
  --kc-muted: {MUTED};
  --kc-dim: {DIM};
  --kc-success: {SUCCESS};
  --kc-warning: {WARNING};
  --kc-error: {ERROR};
  --kc-ring: rgba(74, 222, 128, 0.18);
  --kc-font-sans: {FONT_SANS};
  --kc-font-mono: {FONT_MONO};
}}

body {{ font-family: var(--kc-font-sans); }}
body.body--dark {{ background: var(--kc-bg); }}
.q-page, .q-page-container {{ background: transparent; }}
code, pre, .font-mono {{ font-family: var(--kc-font-mono); }}

/* Links pick up the brand; the landing page scopes its own link color. */
a {{ color: var(--kc-brand); }}

/* Depth: soft elevation + rounded corners. No !important, so any card with an
   inline background keeps it until its call site is migrated to tokens. */
.q-card {{
  background: var(--kc-surface);
  border: 1px solid var(--kc-border);
  border-radius: 12px;
}}
.q-dialog .q-card {{ box-shadow: 0 20px 60px rgba(0, 0, 0, 0.55); }}
.q-btn {{ border-radius: 8px; }}

/* Inputs: the #1 "where do I type?" fix. Quasar's default `standard` variant is
   just an underline; give it a real filled box with a green focus ring. Covers
   every ui.input / ui.textarea / ui.select that isn't already outlined/filled. */
.q-field--standard .q-field__control {{
  background: var(--kc-raised);
  border: 1px solid var(--kc-border-strong);
  border-radius: 8px;
  padding: 0 12px;
}}
.q-field--standard .q-field__control::before,
.q-field--standard .q-field__control::after {{ border: 0 !important; }}
.q-field--standard.q-field--focused .q-field__control {{
  border-color: var(--kc-brand);
  box-shadow: 0 0 0 3px var(--kc-ring);
}}
/* Outlined variant (used by the redesigned question box) gets the same ring. */
.q-field--outlined.q-field--focused .q-field__control:after {{
  border-color: var(--kc-brand);
  border-width: 1.5px;
}}

/* Scrollbars — quiet, on-theme. */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{
  background: #2a3644; border-radius: 8px; border: 2px solid var(--kc-bg);
}}
::-webkit-scrollbar-thumb:hover {{ background: #38485a; }}

/* ---- reusable pieces for the clarifying-question chat UX ---------------- */
.kc-qcard {{
  background: var(--kc-surface);
  border: 1px solid var(--kc-border);
  border-left: 3px solid var(--kc-brand);
  border-radius: 14px;
  box-shadow: 0 10px 34px rgba(0, 0, 0, 0.38);
}}
.kc-bubble {{
  background: var(--kc-raised);
  border: 1px solid var(--kc-border);
  border-radius: 4px 14px 14px 14px;
}}
.kc-stage-pill {{
  background: rgba(74, 222, 128, 0.10);
  color: var(--kc-brand);
  border: 1px solid rgba(74, 222, 128, 0.30);
  border-radius: 999px;
  padding: 2px 10px;
  font-weight: 600;
  letter-spacing: 0.02em;
}}
.kc-qchip.q-btn {{
  border: 1px solid var(--kc-border-strong);
  border-radius: 999px;
  color: var(--kc-muted);
  background: var(--kc-raised);
}}
.kc-qchip.q-btn:hover {{
  border-color: var(--kc-brand);
  color: var(--kc-brand);
}}
</style>
"""


def install() -> None:
    """Inject the theme (fonts + stylesheet) into every page's head.

    Idempotent-safe to call once at import; ``shared=True`` appends to a
    class-level global in NiceGUI so no page context is required.
    """
    ui.add_head_html(_FONT_LINKS, shared=True)
    ui.add_head_html(_CSS, shared=True)
