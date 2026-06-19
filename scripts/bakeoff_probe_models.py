#!/usr/bin/env python3
"""Probe whether each bakeoff slate model is routable with the ACTIVE key.

One minimal chat call per model. Prints OK (with cost) or FAIL (with the error).
Use this after changing OPENROUTER_API_KEY to see which models the key unlocks.

  # test the key already in .env
  .venv/bin/python scripts/bakeoff_probe_models.py

  # test a different key without editing .env (shell env overrides .env)
  OPENROUTER_API_KEY=sk-or-... .venv/bin/python scripts/bakeoff_probe_models.py

Then run the matrix over only the models that came back OK:
  KICRAFT_DAILY_USD_CEILING=18 .venv/bin/python scripts/bakeoff_review_models.py \
      --models flash,v4pro,minimax,qwen,glm,gemini,mistral,haiku
"""
from __future__ import annotations

import sys

from kicraft.server.client import make_client
from kicraft.server.config import Settings

# slate IDs + a couple of gemini/mistral alternates to try if the primary 404s
PROBE = [
    "deepseek/deepseek-v4-flash",   # incumbent (known-good here)
    "deepseek/deepseek-v4-pro",
    "minimax/minimax-m3",
    "qwen/qwen3.7-plus",
    "z-ai/glm-5.2",
    "google/gemini-3.5-flash",
    "google/gemini-3.1-flash-lite",
    "mistralai/mistral-medium-3-5",
    "anthropic/claude-haiku-4.5",
]


def main(argv):
    ids = argv or PROBE
    s = Settings.from_env()
    # Relax KiCraft's production cost-safety routing (fp8 provider pin +
    # $0.18/$0.35 price cap) -- that block, not OpenRouter, 404s the pricier
    # models. Spend-guard $ ceiling still applies.
    s.provider_order = []
    s.max_price_prompt = 0.0
    s.max_price_completion = 0.0
    ok = []
    for m in ids:
        c = make_client(s)
        try:
            r = c.chat([{"role": "user", "content": "Reply with the single word OK."}],
                       model=m, max_tokens=20, temperature=0)
            print(f"OK    {m:40s} text={r['text'][:15]!r} ${r['cost_usd']:.5f}")
            ok.append(m)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {m:40s} {type(e).__name__}: {str(e)[:110]}")
    print(f"\n{len(ok)}/{len(ids)} routable: {', '.join(ok) or '(none beyond errors)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
