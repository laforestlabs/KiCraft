#!/usr/bin/env bash
# install.sh -- install the KiCraft start-new-project opencode plugin into a
# target KiCad project so opencode auto-discovers it.
#
# Opencode's plugin loader globs `.opencode/{plugin,plugins}/*.{ts,js}`
# (direct files only -- nested folders are NOT scanned). So the installed
# layout MUST be a single flat .ts file:
#
#   <target>/.opencode/plugins/kicraft-start-project.ts
#   <target>/.opencode/command/kicraft-new.md
#   <target>/.opencode/package.json   (only created if absent)
#
# The plugin source lives in src/index.ts during development. The dev-time
# self-test reads schema/example_plan.json relative to the source file; the
# installed plugin never runs that self-test path (process.argv[1] won't
# match) so no schema files need to be installed.
#
# Usage:
#   ./install.sh <target-project-dir>
#
# Re-running overwrites the previously installed plugin file.

set -euo pipefail

if [[ "${1:-}" == "" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <target-project-dir>" >&2
  echo "" >&2
  echo "Installs the KiCraft start-new-project opencode plugin into the given" >&2
  echo "directory as a single flat file under <target>/.opencode/plugins/." >&2
  exit 1
fi

TARGET="$1"
if [[ ! -d "$TARGET" ]]; then
  echo "error: target directory does not exist: $TARGET" >&2
  exit 2
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC="$SRC_DIR/src/index.ts"
COMMAND_SRC="$SRC_DIR/command/kicraft-new.md"

if [[ ! -f "$PLUGIN_SRC" ]]; then
  echo "error: plugin source missing: $PLUGIN_SRC" >&2
  exit 3
fi
if [[ ! -f "$COMMAND_SRC" ]]; then
  echo "error: command source missing: $COMMAND_SRC" >&2
  exit 3
fi

PLUGIN_DST_DIR="$TARGET/.opencode/plugins"
PLUGIN_DST="$PLUGIN_DST_DIR/kicraft-start-project.ts"
COMMAND_DST_DIR="$TARGET/.opencode/command"
COMMAND_DST="$COMMAND_DST_DIR/kicraft-new.md"
PKG_FILE="$TARGET/.opencode/package.json"

mkdir -p "$PLUGIN_DST_DIR"
mkdir -p "$COMMAND_DST_DIR"

cp "$PLUGIN_SRC"  "$PLUGIN_DST"
cp "$COMMAND_SRC" "$COMMAND_DST"

# Verify the installed plugin matches the loader glob. This catches future
# regressions where someone re-introduces a nested folder layout.
if ! ls "$PLUGIN_DST_DIR"/*.ts >/dev/null 2>&1; then
  echo "error: install produced no *.ts file under $PLUGIN_DST_DIR" >&2
  exit 4
fi

# Minimal package.json so opencode's bun-based loader can resolve
# @opencode-ai/plugin when bundling user plugins.
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

echo "Installed KiCraft start-new-project plugin:"
echo "  $PLUGIN_DST"
echo "  $COMMAND_DST"
echo "  $PKG_FILE"
echo ""
echo "Next: open opencode in $TARGET and run /kicraft-new"
