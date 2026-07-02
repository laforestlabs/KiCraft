#!/usr/bin/env bash
# Stage 3, run as the KICRAFT user from the repo root (uses sudo):
#   cd ~/KiCraft && bash deploy/03-service-setup.sh
# Installs the systemd service + Caddyfile and starts everything.
set -euo pipefail
cd "$(dirname "$0")/.."
test -f .env || { echo "ERROR: .env is missing. Run deploy/02-app-setup.sh and edit .env first."; exit 1; }
grep -q '^OPENROUTER_API_KEY=sk-' .env || { echo "ERROR: OPENROUTER_API_KEY not set in .env"; exit 1; }
grep -q '^KICRAFT_ACCESS_PASSWORD=.\+' .env || { echo "ERROR: KICRAFT_ACCESS_PASSWORD is empty in .env (the site would refuse all logins)"; exit 1; }

# --- app under systemd ---
sudo cp deploy/kicraft-web.service /etc/systemd/system/kicraft-web.service
sudo systemctl daemon-reload
sudo systemctl enable --now kicraft-web

# --- weekly offline JLC catalog refresh (stale dumps rot: KC-V8YWN8) ---
sudo cp deploy/kicraft-jlcparts-update.service /etc/systemd/system/
sudo cp deploy/kicraft-jlcparts-update.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kicraft-jlcparts-update.timer
sleep 3
curl -fsS -o /dev/null -w "local app: HTTP %{http_code}\n" http://127.0.0.1:8080/login \
  || echo "app not responding yet; check:  journalctl -u kicraft-web -e"

# --- Caddy TLS + reverse proxy ---
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy

echo
echo "=== stage 3 done ==="
echo "Watch Caddy obtain the cert:   journalctl -u caddy -f"
echo "Then browse:                   https://kicraft.io"
