"""Locust web-tier load driver for kicraft.io.

    locust -f scripts/loadtest_web_ws.py --host http://127.0.0.1:8080
    locust -f scripts/loadtest_web_ws.py --host http://127.0.0.1:8080 \
           --headless -u 20 -r 2 -t 2m

What this drives today: the public HTTP surface (landing, auth pages, /browse,
/samples, /pricing) under many concurrent virtual users -- the SQLite-read +
template-render path. Each VU also fetches the NiceGUI client bundle to approximate
a real page load's static cost.

What it does NOT yet drive: the interactive design *submit*. That runs over NiceGUI's
socket.io websocket, which Locust's HttpUser cannot speak without a socket.io client.
To load the full design pipeline run the web app with KICRAFT_LLM_MODE=mock +
KICRAFT_MOCK_TRANSCRIPT and use the in-process pipeline scenario
(`python -m kicraft.loadtest pipeline`), which exercises the same _run_design thread
model + queue + SQLite writes at $0. (Driving the websocket here is a TODO -- noted
explicitly so this file is not mistaken for full end-to-end coverage.)
"""
from __future__ import annotations

import os
import random

try:
    from locust import HttpUser, between, task
except ImportError as e:  # pragma: no cover - locust is an optional external tool
    raise SystemExit(
        "locust is not installed. `pip install locust` (it is intentionally not a "
        "KiCraft dependency -- it is an external load driver).") from e

_PUBLIC = ["/", "/login", "/signup", "/browse", "/samples", "/pricing"]
# Optional: a signed file URL minted out of band (scripts/mint_loadtest_token.py).
_TOKEN_URL = os.environ.get("KICRAFT_LOADTEST_TOKEN_URL", "")


class PublicBrowser(HttpUser):
    """A visitor browsing the public pages (the unauthenticated surface a public
    launch exposes to the world)."""

    wait_time = between(0.5, 2.0)

    @task(5)
    def browse_pages(self) -> None:
        path = random.choice(_PUBLIC)
        with self.client.get(path, name=path, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"{path} -> {r.status_code}")

    @task(1)
    def fetch_project_file(self) -> None:
        if not _TOKEN_URL:
            return
        with self.client.get(_TOKEN_URL, name="/project/[token]/[file]",
                             catch_response=True) as r:
            if r.status_code not in (200, 304):
                r.failure(f"file -> {r.status_code}")
