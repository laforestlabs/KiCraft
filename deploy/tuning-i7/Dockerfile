# KiCraft auto-tuning worker with the production KiCad Routing Tools runtime.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC PIP_ROOT_USER_ACTION=ignore

RUN printf 'Acquire::Check-Valid-Until "false";\nAcquire::Check-Date "false";\n' \
        > /etc/apt/apt.conf.d/99no-check-date \
 && apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg git \
        python3 python3-venv python3-pip \
 && add-apt-repository -y ppa:kicad/kicad-9.0-releases \
 && apt-get update && apt-get install -y kicad \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/drandyhaas/KiCadRoutingTools.git /opt/KiCadRoutingTools \
 && git -C /opt/KiCadRoutingTools checkout 3ceb773722bea67aa3685e7ee430c0c0d17ef38d \
 && python3 -m venv --system-site-packages /opt/krt-venv \
 && /opt/krt-venv/bin/pip install --no-cache-dir -r /opt/KiCadRoutingTools/requirements.txt \
 && cd /opt/KiCadRoutingTools \
 && /opt/krt-venv/bin/python build_router.py --tag v0.20.2

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV KICRAFT_REPO=https://codeberg.org/LaForestLabs/KiCraft.git \
    KICRAFT_BRANCH=main \
    KICRAFT_DATA=/data \
    KICRAFT_KICAD_ROUTING_TOOLS_PATH=/opt/KiCadRoutingTools \
    KICRAFT_KICAD_ROUTING_TOOLS_PYTHON=/opt/krt-venv/bin/python
VOLUME ["/data"]
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["tune"]
