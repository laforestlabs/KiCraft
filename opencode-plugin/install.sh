#!/usr/bin/env bash
# install.sh -- copy the KiCraft start-new-project opencode plugin into a target
# KiCad project so that opencode auto-loads it from .opencode/plugins/.
#
# Usage:
#   ./install.sh <target-project-dir>
#
# The script is idempotent. Re-running it overwrites the previously installed
# plugin source with the current source from this directory.

set -euo pipefail

if [[ "${1:-}" == "" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <target-project-dir>" >&2
  echo "" >&2
  echo "Installs the KiCraft start-new-project opencode plugin into the given" >&2
  echo "directory by copying files into <target>/.opencode/plugins/ and" >&2
  echo "<target>/.opencode/commands/." >&2
  exit 1
fi

TARGET="$1"
if [[ ! -d "$TARGET" ]]; then
  echo "error: target directory does not exist: $TARGET" >&2
  exit 2
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$TARGET/.opencode/plugins/kicraft-start-project"
COMMAND_DIR="$TARGET/.opencode/commands"
PKG_FILE="$TARGET/.opencode/package.json"

mkdir -p "$PLUGIN_DIR/schema"
mkdir -p "$COMMAND_DIR"

cp "$SRC_DIR/src/index.ts"                       "$PLUGIN_DIR/index.ts"
cp "$SRC_DIR/schema/project_plan.schema.json"    "$PLUGIN_DIR/schema/project_plan.schema.json"
cp "$SRC_DIR/schema/example_plan.json"           "$PLUGIN_DIR/schema/example_plan.json"
cp "$SRC_DIR/command/kicraft-new.md"             "$COMMAND_DIR/kicraft-new.md"

# Write a minimal package.json so opencode's bun-based plugin loader can
# resolve @opencode-ai/plugin when it bundles user plugins. We do not pin
# zod here -- the plugin only uses tool.schema, which is re-exported by
# @opencode-ai/plugin.
if [[ ! -f "$PKG_FILE" ]]; then
  cat > "$PKG_FILE" <<'JSON'
{
  "name": "opencode-local-plugins",
  "version": "0.0.0",
  "private": true,
  "type": "module",
  "dependencies": {
    "@opencode-ai/plugin": "1.4.6"
  }
}
JSON
fi

echo "Installed KiCraft start-new-project plugin into:"
echo "  $PLUGIN_DIR/"
echo "  $COMMAND_DIR/kicraft-new.md"
echo "  $PKG_FILE"
echo ""
echo "Next: open opencode in $TARGET and run /kicraft-new"
