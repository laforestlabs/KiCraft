#!/usr/bin/env bash
# Merge a KiCraft pull request on Codeberg (Forgejo) with a merge commit and
# delete the source branch. Authenticated via ~/.netrc. Scoped to this one repo
# so the permission grant is "merge a numbered PR here", not arbitrary curl.
#
# Usage: deploy/merge-pr.sh <PR-number>
set -euo pipefail
PR="${1:?usage: merge-pr.sh <PR-number>}"
REPO="${KICRAFT_REPO:-LaForestLabs/KiCraft}"
resp="/tmp/kc_merge_${PR}.json"
code=$(curl -sS --netrc -H "Content-Type: application/json" -X POST \
  "https://codeberg.org/api/v1/repos/${REPO}/pulls/${PR}/merge" \
  -d '{"Do":"merge","delete_branch_after_merge":true}' \
  -o "$resp" -w "%{http_code}")
if [ "$code" = "200" ]; then
  echo "PR #${PR}: merged (merge commit) and source branch deleted"
else
  echo "PR #${PR}: merge FAILED (HTTP ${code})"
  cat "$resp" 2>/dev/null
  exit 1
fi
