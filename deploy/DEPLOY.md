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
# KiCad provides pcbnew (the Python module synthesis needs) + the stock symbol/footprint libraries
sudo apt-get install -y kicad git python3-venv python3-pip
# Caddy (automatic HTTPS reverse proxy)
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy
# Optional, only if you later enable place/route (kicraft build): a JRE for FreeRouting
# sudo apt-get install -y default-jre
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
# sanity: pcbnew must import inside the venv
.venv/bin/python -c "import pcbnew, nicegui, kicraft.server.web; print('deps OK')"
```

Create `~/KiCraft/.env` (mode 600). Use `.env.example` as the template and fill in:
```bash
umask 077
cat > .env <<'ENV'
OPENROUTER_API_KEY=sk-or-...your key...
KICRAFT_MODEL=deepseek/deepseek-v4-flash
KICRAFT_DAILY_USD_CEILING=5
KICRAFT_TOTAL_USD_CEILING=50
KICRAFT_MAX_TOKENS_PER_CALL=4096
KICRAFT_ACCESS_PASSWORD=pick-a-strong-shared-password
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

## 6. Verify
- Browse to `https://kicraft.io` -> the access-password page.
- Enter the password -> the design page. Try a brief, watch the stages stream, download the zip.

## Cost-safety on the live box (recap)
- Spend is capped in code (`KICRAFT_DAILY_USD_CEILING` / `KICRAFT_TOTAL_USD_CEILING`) on top of your prepaid OpenRouter balance with auto top-up OFF. The worst case is bounded by the smaller of those.
- Kill switch: `KICRAFT_KILL_SWITCH=1` in `.env` then `sudo systemctl restart kicraft-web` halts all model calls instantly.
- Access is gated by `KICRAFT_ACCESS_PASSWORD`; only people you share it with can spend the balance.
- The box is fixed-price; load turns into a queue, not a bigger bill.

## Updating
```bash
cd ~/KiCraft && git pull
.venv/bin/pip install -e ".[server,design]"
sudo systemctl restart kicraft-web
```
