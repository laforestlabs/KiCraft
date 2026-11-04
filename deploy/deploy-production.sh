#!/usr/bin/env bash
# Canonical production deployment: restart both services, then verify health.
#
# The live 34-brief design canary (deploy/verify-design-canary.sh) is
# deliberately NOT part of this path: while the pipeline is being stabilised a
# deploy must not be blocked by a provider/budget failure. Run it manually when
# you want the end-to-end real-provider gate:
#     ./deploy/verify-design-canary.sh
#     ./deploy/verify-design-canary.sh rp2040-min   # diagnostic subset
set -euo pipefail
cd "$(dirname "$0")/.."

./deploy/restart-web.sh
./deploy/restart-build-worker.sh

echo "production health passed: web and newly started build worker are ready"
