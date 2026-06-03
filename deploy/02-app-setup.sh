#!/usr/bin/env bash
# Stage 2, run as the KICRAFT user from inside the cloned repo:
#   cd ~/KiCraft && bash deploy/02-app-setup.sh
# Creates the venv (with --system-site-packages so pcbnew imports), installs the
# server, verifies KiCad 9, and scaffolds .env from the template.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, regardless of where this is called from

# --system-site-packages is REQUIRED so the venv can import the system KiCad's pcbnew
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[server,design]"

# sanity: pcbnew must import inside the venv AND be KiCad 9
.venv/bin/python - <<'PY'
import pcbnew
v = pcbnew.GetBuildVersion()
print("KiCad", v)
assert v.startswith("9."), f"need KiCad 9, got {v}"
import nicegui, kicraft.server.web  # noqa: F401
print("deps OK")
PY

if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  echo
  echo ">>> Created .env (mode 600). Edit it now:   nano .env"
  echo ">>> Fill: OPENROUTER_API_KEY, KICRAFT_ACCESS_PASSWORD, KICRAFT_STORAGE_SECRET"
  echo ">>> and set KICRAFT_MAX_TOKENS_PER_CALL=4096"
  echo ">>> Then run:  bash deploy/03-service-setup.sh"
else
  echo ".env already exists; leaving it untouched. Edit if needed, then run deploy/03-service-setup.sh"
fi
