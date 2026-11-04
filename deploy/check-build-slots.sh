#!/usr/bin/env bash
# Detect — and optionally reap — leaked host build slots and orphaned build processes.
#
# WHY THIS EXISTS
# `build_slot()` (kicraft/build_slots.py) holds an flock for the duration of a build, and it is
# released only when the holder *exits*. A build's own timeout lives in its parent, so when a
# campaign is killed (OOM, or a manual kill) its build child is left with no bound at all: if it
# is blocked rather than dead it keeps the flock forever. Measured on this box 2026-09-19: A1's
# OOM kill orphaned `python -m kicraft.design.cli_app build` (ppid 1) which held slot_0 for
# **12 h 24 m** at 1 s of CPU and 407 MB RSS, while the production build worker — which sizes its
# own concurrency from the same host gate — ran the whole time on a single slot. Nothing else
# notices, because a leaked slot is silent: the next build simply takes another slot and looks
# healthy.
#
# WHAT IT REPORTS
#   * per slot: free / held, holder pid + cmdline + age + RSS
#   * STRICT LEAK: a holder whose process is orphaned (ppid 1) and whose command line is a
#     build/route invocation. The detached services (`kicraft.server.web`,
#     `kicraft.server.build_worker`) are ppid 1 by design and never match, so they are never
#     reaped.
#   * ORPHAN: a ppid-1 build/route process holding no slot — still burning CPU and RAM.
#   * stale breadcrumb pid (the lock file records the last acquirer; flock is the truth)
#
# USAGE
#   deploy/check-build-slots.sh                     # report; exit 1 when a strict leak is present
#   deploy/check-build-slots.sh --reap              # TERM then KILL strict leaks; exit 2
#   deploy/check-build-slots.sh --json              # machine-readable report
#   deploy/check-build-slots.sh --loop 300 --reap   # supervise: check every 300 s until stopped
#
# OPTIONS
#   --reap              terminate strict leaks (TERM, then KILL after 5 s)
#   --min-age-s N       only reap/flag leaks older than N seconds (default 120, so a restart's
#                       in-flight orphans are given time to exit on their own)
#   --loop N            repeat every N seconds (implies forever; log one line per pass)
#   --json              emit JSON instead of text
#   -q                  quiet: print only problems and the final status
#
# EXIT CODES  0 clean · 1 leak found (not reaped) · 2 reaped · 3 usage/runtime error
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLOTS_DIR="${KICRAFT_BUILD_SLOTS_DIR:-$HOME/.kicraft/build_slots}"
LOG="${KICRAFT_BUILD_SLOTS_MONITOR_LOG:-$REPO/logs/build_slots_monitor.log}"

# Command lines that are build work (never the long-lived detached services).
BUILD_CMD_RE='kicraft\.design\.cli_app build|kicraft/cli/solve_subcircuits\.py|py_router/route\.py|kicraft\.design\.cli_app place'

REAP=0
JSON=0
QUIET=0
LOOP=0
MIN_AGE_S=120

while [ $# -gt 0 ]; do
    case "$1" in
        --reap) REAP=1; shift ;;
        --json) JSON=1; shift ;;
        -q) QUIET=1; shift ;;
        --loop) LOOP="${2:?--loop needs seconds}"; shift 2 ;;
        --min-age-s) MIN_AGE_S="${2:?--min-age-s needs seconds}"; shift 2 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 3 ;;
    esac
done

# Slot count: same resolution as kicraft.build_slots.slot_count() — .env is loaded by every
# KiCraft process, and this script must agree with what the running processes actually use.
slot_count() {
    local raw="" n=""
    if [ "${KICRAFT_BUILD_SLOTS+x}" ]; then
        raw="$KICRAFT_BUILD_SLOTS"
    else
        raw="$(grep -E '^KICRAFT_BUILD_SLOTS=' "$REPO/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'"'"' ')"
    fi
    if [[ "$raw" =~ ^-?[0-9]+$ ]]; then
        [ "$raw" -lt 0 ] && raw=0
        echo "$raw"; return
    fi
    n=$(( $(nproc --all) / 6 ))
    [ "$n" -lt 1 ] && n=1
    echo "$n"
}

# The helper checks /proc/locks against the file's device/inode and confirms
# that flock is held. An invisible owner or a changing file is an error.
slot_holders() { python3 "$REPO/deploy/build-slot-process.py" owners "$1"; }

ppid_of()   { ps -o ppid= -p "$1" 2>/dev/null | tr -d ' '; }
etimes_of() { ps -o etimes= -p "$1" 2>/dev/null | tr -d ' '; }
rss_of()    { ps -o rss= -p "$1" 2>/dev/null | tr -d ' '; }
cmdline_of() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | sed 's/ *$//'; }

is_build_cmd() { printf '%s' "$1" | grep -Eq "$BUILD_CMD_RE"; }

declare -a FINDINGS=()
declare -A STARTS=() UIDS=() IDENTITY_SEEN=() LOCK_FILES=()

# Capture kernel process identity, not just a reusable PID. A missing or
# malformed proc entry is never eligible for a signal.
snapshot_identity() {
    local pid="$1" line rest start uid
    local -a fields
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    IFS= read -r line < "/proc/$pid/stat" 2>/dev/null || return 1
    rest="${line##*) }"
    [[ "$rest" != "$line" ]] || return 1
    read -r -a fields <<< "$rest"
    start="${fields[19]:-}"
    uid="$(stat -c %u "/proc/$pid" 2>/dev/null)" || return 1
    [[ "$start" =~ ^[0-9]+$ && "$uid" =~ ^[0-9]+$ ]] || return 1
    printf '%s %s\n' "$start" "$uid"
}

remember_identity() {
    local pid="$1" identity start uid
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    [[ -z "${IDENTITY_SEEN[$pid]:-}" ]] || return 0
    IDENTITY_SEEN["$pid"]=1
    identity="$(snapshot_identity "$pid")" || return 1
    read -r start uid <<< "$identity"
    STARTS["$pid"]="$start"
    UIDS["$pid"]="$uid"
}

identity_still_reapable() {
    local pid="$1" identity start uid cmd
    [[ -n "${STARTS[$pid]:-}" && -n "${UIDS[$pid]:-}" ]] || return 1
    identity="$(snapshot_identity "$pid")" || return 1
    read -r start uid <<< "$identity"
    [[ "$start" = "${STARTS[$pid]}" && "$uid" = "${UIDS[$pid]}" ]] || return 1
    [ "$(ppid_of "$pid")" = 1 ] || return 1
    cmd="$(cmdline_of "$pid")"
    is_build_cmd "$cmd"
}

scan_once() {
    local total held=0 idx file pid ppid age rss cmd cls
    total="$(slot_count)"
    FINDINGS=()
    STARTS=()
    UIDS=()
    IDENTITY_SEEN=()
    LOCK_FILES=()

    if [ ! -d "$SLOTS_DIR" ]; then
        [ "$JSON" -eq 1 ] || echo "slots dir absent ($SLOTS_DIR): no build has ever taken a slot here"
        return 0
    fi

    local i
    for (( i = 0; i < total; i++ )); do
        file="$SLOTS_DIR/slot_$i.lock"
        [ -e "$file" ] || continue
        local holders
        if ! holders="$(slot_holders "$file")"; then
            [ "$JSON" -eq 0 ] && printf 'slot %s/%s  owner unknown\n' "$((i + 1))" "$total"
            FINDINGS+=("unknown|0|0|0|slot_$i owner unavailable")
            continue
        fi
        if [ -z "$holders" ]; then
            [ "$QUIET" -eq 1 ] || [ "$JSON" -eq 1 ] || printf 'slot %s/%s  free\n' "$((i + 1))" "$total"
            continue
        fi
        held=$(( held + 1 ))
        for pid in $holders; do
            remember_identity "$pid" || true
            ppid="$(ppid_of "$pid")"; age="$(etimes_of "$pid")"
            rss="$(rss_of "$pid")"; cmd="$(cmdline_of "$pid")"
            cls="busy"
            if [ "${ppid:-0}" = "1" ] && is_build_cmd "$cmd"; then
                if [ "${age:-0}" -ge "$MIN_AGE_S" ]; then
                    cls="leak"
                    LOCK_FILES["$pid"]="$file"
                    FINDINGS+=("leak|$pid|$age|$rss|$cmd")
                else
                    cls="orphan-young"
                fi
            fi
            { [ "$QUIET" -eq 1 ] || [ "$JSON" -eq 1 ]; } && [ "$cls" = "busy" ] && continue
            [ "$JSON" -eq 1 ] || printf 'slot %s/%s  %-12s pid=%s ppid=%s age=%ss rss=%sKB cmd=%s\n' \
                "$((i + 1))" "$total" "$cls" "$pid" "${ppid:-?}" "${age:-?}" "${rss:-?}" "${cmd:0:90}"
        done
    done

    # Orphaned build/route processes that hold no slot: wasted CPU/RAM all the same.
    local orphan
    while read -r orphan; do
        [ -z "$orphan" ] && continue
        remember_identity "$orphan" || true
        local oc
        oc="$(cmdline_of "$orphan")"
        is_build_cmd "$oc" || continue
        local oage orss oheld="no"
        oage="$(etimes_of "$orphan")"; orss="$(rss_of "$orphan")"
        local f
        for (( i = 0; i < total; i++ )); do
            f="$SLOTS_DIR/slot_$i.lock"
            [ -e "$f" ] || continue
            local owners
            if ! owners="$(slot_holders "$f")"; then
                oheld="unknown"
            elif printf '%s\n' "$owners" | grep -qx "$orphan"; then
                oheld="slot_$i"
                break
            fi
        done
        [ "$JSON" -eq 0 ] && printf 'orphan build process pid=%s age=%ss rss=%sKB holds=%s cmd=%s\n' \
            "$orphan" "${oage:-?}" "${orss:-?}" "$oheld" "${oc:0:80}"
        if [ "${oage:-0}" -ge "$MIN_AGE_S" ] && [ "$oheld" = "no" ]; then
            FINDINGS+=("orphan|$orphan|$oage|$orss|$oc")
        fi
    done < <(ps -eo pid=,ppid= | awk '$2==1 {print $1}')

    { [ "$QUIET" -eq 1 ] || [ "$JSON" -eq 1 ]; } || printf 'slots: %s held / %s total · strict leaks: %s\n' \
        "$held" "$total" "$(leak_count)"
    return 0
}

leak_count() {
    local n=0 row
    for row in "${FINDINGS[@]:-}"; do
        [ -z "$row" ] && continue
        case "$row" in leak\|*) n=$(( n + 1 ));; esac
    done
    echo "$n"
}

finding_count() { [ "${#FINDINGS[@]}" -eq 0 ] && echo 0 || printf '%s\n' "${#FINDINGS[@]}"; }

reap_findings() {
    local row pid failed=0
    local -A term_sent=()
    for row in "${FINDINGS[@]:-}"; do
        case "$row" in
            leak\|*|orphan\|*)
                pid="${row#*|}"; pid="${pid%%|*}"
                [[ -z "${term_sent[$pid]:-}" ]] || continue
                if identity_still_reapable "$pid"; then
                    echo "reaping pid=$pid (${row%%|*})" >&2
                    if python3 "$REPO/deploy/build-slot-process.py" signal "$pid" "${STARTS[$pid]}" "${UIDS[$pid]}" 15 "${LOCK_FILES[$pid]:-}"; then
                        term_sent["$pid"]=1
                    else
                        failed=1
                    fi
                else
                    echo "refusing to signal pid=$pid: process identity or eligibility changed" >&2
                    failed=1
                fi
                ;;
            unknown\|*) failed=1 ;;
        esac
    done
    sleep 5
    for row in "${FINDINGS[@]:-}"; do
        case "$row" in
            leak\|*|orphan\|*)
                pid="${row#*|}"; pid="${pid%%|*}"
                if [[ -n "${term_sent[$pid]:-}" ]] && identity_still_reapable "$pid"; then
                    echo "pid=$pid survived SIGTERM; sending SIGKILL" >&2
                    python3 "$REPO/deploy/build-slot-process.py" signal "$pid" "${STARTS[$pid]}" "${UIDS[$pid]}" 9 "${LOCK_FILES[$pid]:-}" || true
                fi
                ;;
        esac
    done
    [ "$failed" -eq 0 ] && [ "${#term_sent[@]}" -gt 0 ]
}

json_report() {
    local total; total="$(slot_count)"
    local rows="" row kind pid age rss cmd
    for row in "${FINDINGS[@]:-}"; do
        [ -z "$row" ] && continue
        IFS='|' read -r kind pid age rss cmd <<<"$row"
        cmd="$(printf '%s' "${cmd:0:160}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
        rows+="{\"kind\":\"$kind\",\"pid\":$pid,\"age_s\":${age:-0},\"rss_kb\":${rss:-0},\"cmd\":$cmd},"
    done
    local dir_json
    dir_json="$(printf '%s' "$SLOTS_DIR" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
    printf '{"slots_total":%s,"findings":[%s],"slots_dir":%s,"min_age_s":%s}\n' \
        "$total" "${rows%,}" "$dir_json" "$MIN_AGE_S"
}

pass() {
    scan_once
    local count; count="$(finding_count)"
    if [ "$JSON" -eq 1 ]; then json_report; fi
    if [ "$count" -gt 0 ] && [ "$REAP" -eq 1 ]; then
        if reap_findings; then return 2; else return 1; fi
    fi
    [ "$count" -gt 0 ] && return 1
    return 0
}

if [ "$LOOP" -gt 0 ]; then
    mkdir -p "$(dirname "$LOG")"
    echo "==== build-slot monitor started $(date -Is) every ${LOOP}s reap=$REAP ====" >> "$LOG"
    while true; do
        out="$(pass; echo "rc=$?")"
        rc="${out##*rc=}"
        printf '%s %s\n' "$(date -Is)" "$(printf '%s' "$out" | sed '$d' | tr '\n' ' ')" >> "$LOG"
        if [ "$rc" = "2" ]; then
            printf '%s REAPED leak(s); see lines above\n' "$(date -Is)" >> "$LOG"
        fi
        sleep "$LOOP"
    done
fi

pass
exit $?
