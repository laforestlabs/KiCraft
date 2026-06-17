#!/usr/bin/env python3
"""Mint a signed project-file token OUT OF BAND for web-tier load tests.

The web app serves raw KiCad files at /project/<token>/<file>, where <token> is an
HMAC over the absolute project dir signed with KICRAFT_STORAGE_SECRET (web.py
_register_project_dir). Rather than add a token-minting ROUTE to the production app
(new attack surface), this script reproduces that ~6-line HMAC locally so a load
driver (k6/Locust) can hit the file endpoint with a valid token. Run it ON THE BOX,
where the secret lives.

    .venv/bin/python scripts/mint_loadtest_token.py /home/kicraft/.kicraft/projects/1/generated/A_USB_C

Prints the token and a ready /project/<token>/<file> URL per servable file.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import sys
from pathlib import Path

# Match web.py: the same suffix whitelist the serve handler enforces.
_ALLOWED_SUFFIXES = (".kicad_pcb", ".kicad_sch", ".kicad_pro", ".net", ".zip")


def _secret() -> bytes:
    # Load .env the same way the server does (stdlib, no override) so a token
    # minted here verifies against the running app.
    try:
        from kicraft.server.config import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return os.environ.get("KICRAFT_STORAGE_SECRET", "kicraft-dev-secret").encode("utf-8")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def mint(project_dir: Path) -> str:
    payload = _b64e(str(project_dir.resolve()).encode("utf-8"))
    sig = _b64e(hmac.new(_secret(), payload.encode("ascii"), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    project_dir = Path(argv[0])
    if not project_dir.is_dir():
        print(f"not a directory: {project_dir}", file=sys.stderr)
        return 2
    base = os.environ.get("KICRAFT_LOADTEST_BASE", "http://127.0.0.1:8080").rstrip("/")
    token = mint(project_dir)
    print(f"token: {token}\n")
    files = sorted(p.name for p in project_dir.iterdir()
                   if p.is_file() and p.name.endswith(_ALLOWED_SUFFIXES))
    if not files:
        print(f"(no servable files in {project_dir})", file=sys.stderr)
    for name in files:
        print(f"{base}/project/{token}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
