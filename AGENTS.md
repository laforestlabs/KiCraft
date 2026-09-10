# AGENTS.md — operating context for this checkout

## Production box

This checkout at `/home/kicraft/KiCraft` (host `kicraft-web`, user `kicraft`) is the
**production Hetzner box serving https://kicraft.io**. It is not a dev laptop and not
a CI runner. When in doubt, verify against the live processes, not assumptions:

- Web app: NiceGUI on `127.0.0.1:8080`, fronted by Caddy (TLS) → `kicraft.io`.
- Router: KiCadRoutingTools pinned at `/home/kicraft/KiCadRoutingTools`
  (commit `3ceb773722bea67aa3685e7ee430c0c0d17ef38d`), venv at
  `/home/kicraft/krt-venv/bin/python` (see `.env` `KICRAFT_KICAD_ROUTING_TOOLS_*`).
- Secrets, spend caps, signup codes: `/home/kicraft/KiCraft/.env` (mode 600).

## Deploy (git pull on this box)

```bash
cd ~/KiCraft && git pull
.venv/bin/pip install -e ".[server,design]"
./deploy/deploy-production.sh       # mandatory canary, restart, and health verification
```

- `deploy/deploy-production.sh` is the canonical production restart path. It first
  runs `deploy/verify-design-canary.sh`, which drives all 34 curated briefs
  through the live LLM pipeline under the production `.env` profile and per-run
  budget and exits nonzero unless every brief commits all five stages. Only then
  does it restart both services and verify web HTTP 200 plus
  `[build-worker] ready`. Unit tests do not exercise the provider/budget path.
- The systemd units (`kicraft-web.service`, `kicraft-build-worker.service`) are
  **inactive**. Services run as detached `setsid nohup` processes (ppid 1) managed
  by `deploy/restart-web.sh` / `deploy/restart-build-worker.sh`. Use the scripts,
  not `systemctl`.
- Verify after restart: `curl -sf http://127.0.0.1:8080/` → 200; the build-worker
  log (`logs/kicraft_build_worker.log`) ends with `[build-worker] ready`.
