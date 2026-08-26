# Deploying KiCraft to kicraft.io (Hetzner + Porkbun)

A gated, spend-capped deployment on a single fixed-price box. The worst-case
monthly cost is the flat box rent + your prepaid OpenRouter balance, a demand
spike becomes a queue, never an invoice.

Topology: `Porkbun DNS -> Hetzner box -> Caddy (TLS) -> kicraft-web (127.0.0.1:8080) -> capped gateway`.

## 0. Prerequisites
- A Hetzner Cloud box, provisioned + hardened per the project's earlier runbook
  (non-root `kicraft` user, SSH-key-only, `ufw` allowing OpenSSH/80/443).
- The domain `kicraft.io` in your Porkbun account.
- Your OpenRouter API key (prepaid, auto top-up OFF).

## 1. Point Porkbun DNS at the box (do this first, TLS needs it)
In Porkbun -> Domain Management -> kicraft.io -> DNS:
- Delete Porkbun's default parking A/ALIAS records.
- Add `A` record: Host blank (`@`), Answer = your Hetzner IPv4, TTL 600.
- Add `AAAA` record: Host blank, Answer = your Hetzner IPv6 (optional but nice), TTL 600.
- Add `A` (or `CNAME`) record: Host `www`, Answer = the IPv4 (or `kicraft.io`).
Wait until `dig +short kicraft.io` returns your box IP before step 5 (DNS propagation, usually minutes).

## 2. System dependencies (on the box, as a sudo user)
```bash
sudo apt-get update
sudo apt-get install -y software-properties-common git python3-venv python3-pip
# KiCad 9 provides pcbnew (the Python module synthesis needs) + the stock symbol/footprint
# libraries. KiCraft emits and processes KiCad 9 files, and the Ubuntu 24.04 universe package
# is KiCad 8 (too old), so install KiCad 9 from the official KiCad PPA.
sudo add-apt-repository --yes ppa:kicad/kicad-9.0-releases
sudo apt-get update
sudo apt-get install -y kicad
# Caddy (automatic HTTPS reverse proxy)
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy
# Place/route also requires KiCad 9 plus the pinned KiCadRoutingTools checkout:
# source 0.20.2, commit 3ceb773722bea67aa3685e7ee430c0c0d17ef38d, native 0.20.1.
```

## 3. Deploy the app (as the `kicraft` user)
```bash
cd ~
git clone <your KiCraft repo URL> KiCraft
cd KiCraft
# IMPORTANT: --system-site-packages so the venv can import the system KiCad's pcbnew
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[server,design]"
# sanity: pcbnew must import inside the venv AND be KiCad 9
.venv/bin/python -c "import pcbnew; v=pcbnew.GetBuildVersion(); print('KiCad', v); assert v.startswith('9.'), 'KiCraft needs KiCad 9'; import nicegui, kicraft.server.web, kicraft.server.accounts; print('deps OK')"
```

Create `~/KiCraft/.env` (mode 600). Use `.env.example` as the template and fill in:
```bash
umask 077
cat > .env <<'ENV'
OPENROUTER_API_KEY=sk-or-...your key...
KICRAFT_DESIGN_PROFILE=flash
KICRAFT_MODEL=deepseek/deepseek-v4-flash-0731
KICRAFT_REVIEW_MODEL=minimax/minimax-m3
KICRAFT_EVAL_JUDGE_MODEL=minimax/minimax-m3
KICRAFT_DAILY_USD_CEILING=5
KICRAFT_TOTAL_USD_CEILING=50
KICRAFT_MAX_TOKENS_PER_CALL=4096
KICRAFT_SIGNUP_CODE=pick-a-strong-invite-code   # bootstrap only; mint real codes at /admin/invites
KICRAFT_STORAGE_SECRET=long-random-string-for-session-cookies
ENV
chmod 600 .env
```

## 4. Run it under systemd
```bash
sudo cp deploy/kicraft-web.service /etc/systemd/system/kicraft-web.service
sudo systemctl daemon-reload
sudo systemctl enable --now kicraft-web
systemctl status kicraft-web          # should be active (running)
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/login   # 200
```

## 5. TLS + reverse proxy with Caddy
```bash
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
journalctl -u caddy -f                 # watch it obtain the kicraft.io certificate
```
Caddy fetches a Let's Encrypt cert automatically once DNS resolves to the box and 80/443 are open.

## 6. Create your account and verify
- Browse to `https://kicraft.io` -> Sign in -> "Create an account".
- Register with your email, a password, and the `KICRAFT_SIGNUP_CODE` invite code.
- On the design page, try a brief, watch the stages stream, download the zip. It
  also appears under "Your projects" and is saved under `~/.kicraft/projects/`.
- Grant yourself a higher tier (Stripe is not wired yet, so tiers are manual):
  `~/KiCraft/.venv/bin/kicraft-accounts set-tier you@example.com max`
  (`kicraft-accounts list` shows everyone and their tier.)

## Cost-safety on the live box (recap)
- Spend is capped in code (`KICRAFT_DAILY_USD_CEILING` / `KICRAFT_TOTAL_USD_CEILING`) on top of your prepaid OpenRouter balance with auto top-up OFF. The worst case is bounded by the smaller of those.
- Kill switch: `KICRAFT_KILL_SWITCH=1` in `.env` then `sudo systemctl restart kicraft-web` halts all model calls instantly.
- Registration is invite-only by default: signups need either a code minted at `/admin/invites` (codes can grant a tier for N days or forever, carry a max-use cap, and can be disabled) or the legacy `KICRAFT_SIGNUP_CODE` env code (plain Free tier). The same page has the public-launch switch that lets the Free tier register with no code. Per-user tier quotas (free 1/week, pro 5/month, max 25/month) bound each account on top of the global ceilings.
- The box is fixed-price; load turns into a queue, not a bigger bill.

## Updating
```bash
cd ~/KiCraft && git pull
.venv/bin/pip install -e ".[server,design]"
sudo systemctl restart kicraft-web
```
