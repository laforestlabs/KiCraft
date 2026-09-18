#!/usr/bin/env bash
# Run a KiCraft CLI with the production .env profile already in its environment.
#
# Two tools on this box need it, for different reasons:
#   * the stage debugger (`kicraft-stage-debug`) runs with CWD = the project workspace, where a
#     CWD-relative `.env` lookup finds nothing, so it reports a missing OPENROUTER_API_KEY even
#     though the box is configured;
#   * the design CLI's build tail (`kicraft build`) reads its autoplacer/routing configuration
#     from the process environment while importing, before any `.env` it loads itself, so a
#     manual build otherwise fails with "KiCadRoutingTools is selected but
#     kicraft_routing_tools_path is unset".
#
# The services do not need this (they load `.env` themselves from the repo root); this is for
# driving a CLI against a project workspace by hand.
#
# Usage:
#   deploy/run-with-env.sh kicraft-stage-debug debug-draft --workspace . --stage intent ...
#   deploy/run-with-env.sh kicraft build .kicraft/state.json generated --no-archive
#   deploy/run-with-env.sh python -c 'import os; print(os.environ["KICRAFT_KICAD_ROUTING_TOOLS_PATH"])'
#
# The repo's venv is prepended to PATH, so console scripts and `python` resolve to it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${KICRAFT_ENV_FILE:-$REPO/.env}"

if [ "$#" -eq 0 ]; then
    echo "usage: $(basename "$0") <command> [args...]" >&2
    exit 2
fi
if [ ! -f "$ENV_FILE" ]; then
    echo "no env file at $ENV_FILE" >&2
    exit 1
fi

# Every KEY=VALUE line is exported verbatim. An already-set variable wins, matching
# kicraft.server.config.load_dotenv: an operator override on the command line stays in charge.
while IFS= read -r line; do
    case "$line" in
        ''|'#'*) continue ;;
    esac
    case "$line" in
        *=*)
            key="${line%%=*}"
            value="${line#*=}"
            if [ -z "${!key+x}" ]; then
                export "$key=$value"
            fi
            ;;
    esac
done < "$ENV_FILE"

export PATH="$REPO/.venv/bin:$PATH"
exec "$@"
