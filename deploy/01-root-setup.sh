#!/usr/bin/env bash
# Stage 1, run as ROOT on a fresh Ubuntu 24.04 box.
# Creates a sudo user with your SSH key, hardens SSH, locks the firewall to
# 22/80/443, and installs KiCad 9 + Caddy. Safe to re-run.
#
#   scp deploy/01-root-setup.sh root@<box-ip>:/root/
#   ssh root@<box-ip> 'bash /root/01-root-setup.sh'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a

# --- non-root sudo user, with your SSH key copied over ---
id kicraft &>/dev/null || adduser --disabled-password --gecos "" kicraft
usermod -aG sudo kicraft
install -d -m700 -o kicraft -g kicraft /home/kicraft/.ssh
cp /root/.ssh/authorized_keys /home/kicraft/.ssh/authorized_keys
chown kicraft:kicraft /home/kicraft/.ssh/authorized_keys
chmod 600 /home/kicraft/.ssh/authorized_keys

# --- harden SSH: key-only, no root login ---
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/; s/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
systemctl restart ssh

# --- box firewall: only SSH/HTTP/HTTPS inbound ---
apt-get update
apt-get install -y ufw unattended-upgrades
ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw --force enable

# --- KiCad 9 (the version KiCraft needs) + Python + git ---
apt-get install -y software-properties-common git python3-venv python3-pip
add-apt-repository --yes ppa:kicad/kicad-9.0-releases
apt-get update
apt-get install -y kicad

# --- Caddy (automatic HTTPS) ---
apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | tee /etc/apt/sources.list.d/caddy-stable.list
apt-get update
apt-get install -y caddy

echo
echo "=== stage 1 done ==="
dpkg -l kicad | tail -1
echo "Next: ssh kicraft@<box-ip>  then clone the repo and run deploy/02-app-setup.sh"
