"""Smoke test: prove the OpenRouter key works and the caps are live, for ~$0.0001.

    python -m kicraft.server.smoketest

Reads OPENROUTER_API_KEY from .env (see .env.example). Queries the prepaid key
limit, makes one bounded DeepSeek call through the capped client, and prints the
reply, token usage, cost, and the running spend status.
"""
from __future__ import annotations

import json
import sys

import requests

from .client import CappedOpenRouterClient
from .config import Settings


def _key_info(s: Settings) -> None:
    try:
        r = requests.get(f"{s.base_url}/key",
                         headers={"Authorization": f"Bearer {s.api_key}"}, timeout=30)
        if r.ok:
            d = r.json().get("data", {})
            print(f"OpenRouter key: usage=${d.get('usage')}  limit={d.get('limit')}  "
                  f"limit_remaining={d.get('limit_remaining')}  free_tier={d.get('is_free_tier')}")
        else:
            print(f"(key info HTTP {r.status_code}; skipping)")
    except requests.RequestException as e:
        print(f"(key info check skipped: {e})")


def main(argv=None) -> int:
    s = Settings.from_env()
    print("KiCraft capped-client smoke test")
    print(f"  settings: {json.dumps(s.redacted())}")
    _key_info(s)

    client = CappedOpenRouterClient(s)
    print(f"  spend before: {json.dumps(client.guard.status())}")
    res = client.chat(
        [{"role": "user", "content": "Reply with exactly: KiCraft OpenRouter link OK"}],
        max_tokens=128,
    )
    print("\n--- model reply ---")
    reply = (res.get("text") or "").strip()
    print(reply or "(no content returned)")
    if not reply and res.get("reasoning"):
        print("[reasoning-only output, truncated]:", str(res["reasoning"])[:300])
    print(f"finish_reason: {res.get('finish_reason')}")
    print("\n--- usage ---")
    print(json.dumps(res["usage"], indent=2))
    print(f"\ncost this call: ${res['cost_usd']:.6f}")
    print(f"spend after:    {json.dumps(res['guard'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
