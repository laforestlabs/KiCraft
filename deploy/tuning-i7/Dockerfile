# KiCraft auto-tuning worker — runs the CPU-bound place+route tuning loop on a
# beefy box (e.g. a homelab i7), at $0 LLM cost. Mirrors the production stack so
# routing behaves identically: Ubuntu 24.04 + KiCad 9.0 + FreeRouting 1.9.0 +
# Java + xvfb. The repo (code + committed tuning corpus) is cloned at RUN time
# into the mounted /data volume, so updating is a restart, not a rebuild.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC PIP_ROOT_USER_ACTION=ignore

# Toolchain: KiCad 9.0 (kicad-cli + pcbnew python), a JRE + xvfb for FreeRouting,
# git/python for the repo + venv. The Check-Date/Valid-Until bypass keeps `apt`
# working when the host clock is skewed (common on homelab boxes with a dead RTC
# battery) — Ubuntu re-publishes its InRelease files constantly, so even a few
# hours behind makes them look "not valid yet". Fix the host clock too.
RUN printf 'Acquire::Check-Valid-Until "false";\nAcquire::Check-Date "false";\n' \
        > /etc/apt/apt.conf.d/99no-check-date \
 && apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg curl git \
        python3 python3-venv python3-pip \
        default-jre-headless xvfb xauth fonts-dejavu-core \
 && add-apt-repository -y ppa:kicad/kicad-9.0-releases \
 && apt-get update && apt-get install -y kicad \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# FreeRouting 1.9.0 at the path kicraft.autoplacer.config expects by default.
RUN mkdir -p /root/.local/lib \
 && curl -fsSL -o /root/.local/lib/freerouting-1.9.0.jar \
      https://github.com/freerouting/freerouting/releases/download/v1.9.0/freerouting-1.9.0.jar \
 && test -s /root/.local/lib/freerouting-1.9.0.jar

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV KICRAFT_REPO=https://codeberg.org/LaForestLabs/KiCraft.git \
    KICRAFT_BRANCH=main \
    KICRAFT_DATA=/data
VOLUME ["/data"]
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["tune"]
