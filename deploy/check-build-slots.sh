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
    raw="$(grep -E '^KICRAFT_BUILD_SLOTS=' "$REPO/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"' ')"
    if [ -n "$raw" ] && [ "$raw" -ge 0 ] 2>/dev/null; then
        echo "$raw"; return
    fi
    n=$(( $(nproc) / 6 ))
    [ "$n" -lt 1 ] && n=1
    echo "$n"
}

# "held" when a non-blocking flock on the slot file fails (flock is the authoritative owner).
slot_held() {
    if command -v flock >/dev/null 2>&1; then
        ! flock -n "$1" true 2>/dev/null
    else
        return 1   # cannot tell without flock: report free
    fi
}

slot_holders() { lsof -t "$1" 2>/dev/null | sort -u; }

ppid_of()   { ps -o ppid= -p "$1" 2>/dev/null | tr -d ' '; }
etimes_of() { ps -o etimes= -p "$1" 2>/dev/null | tr -d ' '; }
rss_of()    { ps -o rss= -p "$1" 2>/dev/null | tr -d ' '; }
cmdline_of() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | sed 's/ *$//'; }

is_build_cmd() { printf '%s' "$1" | grep -Eq "$BUILD_CMD_RE"; }

declare -a FINDINGS=()

scan_once() {
    local total held=0 idx file pid ppid age rss cmd cls
    total="$(slot_count)"
    FINDINGS=()

    if [ ! -d "$SLOTS_DIR" ]; then
        echo "slots dir absent ($SLOTS_DIR): no build has ever taken a slot here"
        return 0
    fi

    local i
    for (( i = 0; i < total; i++ )); do
        file="$SLOTS_DIR/slot_$i.lock"
        [ -e "$file" ] || continue
        if ! slot_held "$file"; then
            [ "$QUIET" -eq 1 ] || [ "$JSON" -eq 1 ] || printf 'slot %s/%s  free\n' "$((i + 1))" "$total"
            continue
        fi
        held=$(( held + 1 ))
        local holders
        holders="$(slot_holders "$file")"
        if [ -z "$holders" ]; then
            [ "$JSON" -eq 0 ] && printf 'slot %s/%s  HELD (holder unknown: no lsof; flock is held)\n' "$((i + 1))" "$total"
            FINDINGS+=("unknown|0|0|0|slot_$i held with no visible holder")
            continue
        fi
        for pid in $holders; do
            ppid="$(ppid_of "$pid")"; age="$(etimes_of "$pid")"
            rss="$(rss_of "$pid")"; cmd="$(cmdline_of "$pid")"
            cls="busy"
            if [ "${ppid:-0}" = "1" ] && is_build_cmd "$cmd"; then
                if [ "${age:-0}" -ge "$MIN_AGE_S" ]; then
                    cls="leak"
                    FINDINGS+=("leak|$pid|$age|$rss|$cmd")
                else
                    cls="orphan-young"
                fi
            fi
            { [ "$QUIET" -eq 1 ] || [ "$JSON" -eq 1 ]; } && [ "$cls" = "busy" ] && continue
            printf 'slot %s/%s  %-12s pid=%s ppid=%s age=%ss rss=%sKB cmd=%s\n' \
                "$((i + 1))" "$total" "$cls" "$pid" "${ppid:-?}" "${age:-?}" "${rss:-?}" "${cmd:0:90}"
        done
    done

    # Orphaned build/route processes that hold no slot: wasted CPU/RAM all the same.
    local orphan
    while read -r orphan; do
        [ -z "$orphan" ] && continue
        local oc
        oc="$(cmdline_of "$orphan")"
        is_build_cmd "$oc" || continue
        local oage orss oheld="no"
        oage="$(etimes_of "$orphan")"; orss="$(rss_of "$orphan")"
        local f
        for (( i = 0; i < total; i++ )); do
            f="$SLOTS_DIR/slot_$i.lock"
            [ -e "$f" ] || continue
            if slot_holders "$f" | grep -qx "$orphan"; then oheld="slot_$i"; fi
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
    local row pid
    for row in "${FINDINGS[@]:-}"; do
        case "$row" in
            leak\|*|orphan\|*)
                pid="${row#*|}"; pid="${pid%%|*}"
                echo "reaping pid=$pid (${row%%|*})"
                kill -TERM "$pid" 2>/dev/null
                ;;
        esac
    done
    sleep 5
    for row in "${FINDINGS[@]:-}"; do
        case "$row" in
            leak\|*|orphan\|*)
                pid="${row#*|}"; pid="${pid%%|*}"
                if kill -0 "$pid" 2>/dev/null; then
                    echo "pid=$pid survived SIGTERM; sending SIGKILL"
                    kill -KILL "$pid" 2>/dev/null
                fi
                ;;
        esac
    done
}

json_report() {
    local total; total="$(slot_count)"
    local rows="" row kind pid age rss cmd
    for row in "${FINDINGS[@]:-}"; do
        [ -z "$row" ] && continue
        IFS='|' read -r kind pid age rss cmd <<<"$row"
        cmd="${cmd//\"/\\\"}"
        rows+="{\"kind\":\"$kind\",\"pid\":$pid,\"age_s\":${age:-0},\"rss_kb\":${rss:-0},\"cmd\":\"${cmd:0:160}\"},"
    done
    printf '{"slots_total":%s,"findings":[%s],"slots_dir":"%s","min_age_s":%s}\n' \
        "$total" "${rows%,}" "$SLOTS_DIR" "$MIN_AGE_S"
}

pass() {
    scan_once
    local count; count="$(finding_count)"
    if [ "$JSON" -eq 1 ]; then json_report; fi
    if [ "$count" -gt 0 ] && [ "$REAP" -eq 1 ]; then
        reap_findings
        return 2
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
