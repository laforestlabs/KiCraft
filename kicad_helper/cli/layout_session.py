#!/usr/bin/env python3
"""Track layout progress across iterations within a session.

Records board state snapshots at each scoring run, diffs them to classify
changes as minor tweaks vs major redesigns, and tracks cumulative token
usage. Designed so Claude can review session history and decide whether
to keep tweaking or try a fundamentally different approach.

Usage:
    # Record a new iteration (run after each score_layout.py run):
    python3 layout_session.py record <pcb> [--score-json <path>]

    # Show session summary:
    python3 layout_session.py summary

    # Compare two iterations:
    python3 layout_session.py diff <iter_a> <iter_b>
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import pcbnew

SESSION_FILE = os.path.join(os.path.dirname(__file__), "results", "session.json")


def snapshot_board(pcb_path):
    """Capture a diffable snapshot of board state."""
    board = pcbnew.LoadBoard(pcb_path)

    # Component positions and orientations
    components = {}
    for fp in board.Footprints():
        ref = fp.GetReferenceAsString()
        pos = fp.GetPosition()
        components[ref] = {
            "x": round(pcbnew.ToMM(pos.x), 2),
            "y": round(pcbnew.ToMM(pos.y), 2),
            "rotation": round(fp.GetOrientationDegrees(), 1),
            "layer": fp.GetLayerName(),
            "value": fp.GetValue(),
        }

    # Board outline
    board_rect = board.GetBoardEdgesBoundingBox()
    board_w = round(pcbnew.ToMM(board_rect.GetWidth()), 2)
    board_h = round(pcbnew.ToMM(board_rect.GetHeight()), 2)

    # Trace stats
    trace_count = 0
    via_count = 0
    total_trace_mm = 0
    for track in board.GetTracks():
        if isinstance(track, pcbnew.PCB_VIA):
            via_count += 1
        else:
            trace_count += 1
            s, e = track.GetStart(), track.GetEnd()
            total_trace_mm += math.hypot(
                pcbnew.ToMM(e.x) - pcbnew.ToMM(s.x),
                pcbnew.ToMM(e.y) - pcbnew.ToMM(s.y),
            )

    # Net count
    nets = set()
    for pad in board.GetPads():
        n = pad.GetNetname()
        if n:
            nets.add(n)

    return {
        "components": components,
        "board_w_mm": board_w,
        "board_h_mm": board_h,
        "board_area_mm2": round(board_w * board_h, 1),
        "trace_count": trace_count,
        "via_count": via_count,
        "total_trace_mm": round(total_trace_mm, 1),
        "net_count": len(nets),
    }


def diff_snapshots(a, b):
    """Diff two snapshots. Returns change classification and details."""
    changes = {
        "components_moved": [],
        "components_rotated": [],
        "components_added": [],
        "components_removed": [],
        "board_resized": False,
        "trace_delta": b["trace_count"] - a["trace_count"],
        "via_delta": b["via_count"] - a["via_count"],
        "trace_length_delta_mm": round(b["total_trace_mm"] - a["total_trace_mm"], 1),
    }

    # Board size change
    if abs(b["board_w_mm"] - a["board_w_mm"]) > 0.5 or abs(b["board_h_mm"] - a["board_h_mm"]) > 0.5:
        changes["board_resized"] = True
        changes["board_size_before"] = f"{a['board_w_mm']}x{a['board_h_mm']}mm"
        changes["board_size_after"] = f"{b['board_w_mm']}x{b['board_h_mm']}mm"

    a_comps = a["components"]
    b_comps = b["components"]

    for ref in set(list(a_comps.keys()) + list(b_comps.keys())):
        if ref not in a_comps:
            changes["components_added"].append(ref)
            continue
        if ref not in b_comps:
            changes["components_removed"].append(ref)
            continue

        ca, cb = a_comps[ref], b_comps[ref]
        dist = math.hypot(cb["x"] - ca["x"], cb["y"] - ca["y"])
        if dist > 0.1:  # moved more than 0.1mm
            changes["components_moved"].append({
                "ref": ref,
                "distance_mm": round(dist, 2),
                "from": [ca["x"], ca["y"]],
                "to": [cb["x"], cb["y"]],
            })
        rot_delta = abs(cb["rotation"] - ca["rotation"])
        if rot_delta > 0.5 and rot_delta < 359.5:
            changes["components_rotated"].append({
                "ref": ref,
                "from_deg": ca["rotation"],
                "to_deg": cb["rotation"],
            })

    # Classify change magnitude
    moved = changes["components_moved"]
    max_move = max((m["distance_mm"] for m in moved), default=0)
    avg_move = sum(m["distance_mm"] for m in moved) / len(moved) if moved else 0
    n_moved = len(moved)
    n_rotated = len(changes["components_rotated"])
    n_added = len(changes["components_added"])
    n_removed = len(changes["components_removed"])
    struct_changes = n_added + n_removed

    if struct_changes > 3 or changes["board_resized"] or max_move > 20 or n_moved > 10:
        classification = "major_redesign"
    elif max_move > 5 or n_moved > 3 or n_rotated > 2 or struct_changes > 0:
        classification = "moderate_rework"
    elif n_moved > 0 or n_rotated > 0 or abs(changes["trace_delta"]) > 5:
        classification = "minor_tweak"
    else:
        classification = "no_change"

    changes["classification"] = classification
    changes["summary"] = {
        "components_moved": n_moved,
        "max_move_mm": max_move,
        "avg_move_mm": round(avg_move, 1),
        "components_rotated": n_rotated,
        "components_added": n_added,
        "components_removed": n_removed,
    }

    return changes


def load_session():
    if os.path.isfile(SESSION_FILE):
        with open(SESSION_FILE) as f:
            return json.load(f)
    return {"iterations": [], "created": datetime.now(timezone.utc).isoformat()}


def save_session(session):
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump(session, f, indent=2)


def cmd_record(args):
    session = load_session()
    snapshot = snapshot_board(args.pcb)
    iteration_num = len(session["iterations"])

    # Load token usage from score JSON if provided
    token_usage = {}
    score = None
    if args.score_json and os.path.isfile(args.score_json):
        with open(args.score_json) as f:
            score_data = json.load(f)
        token_usage = score_data.get("token_usage", {})
        score = score_data.get("overall_score")

    # Diff against previous
    change_info = None
    if session["iterations"]:
        prev = session["iterations"][-1]["snapshot"]
        change_info = diff_snapshots(prev, snapshot)

    # Cumulative tokens
    prev_cumulative = session["iterations"][-1]["cumulative_tokens"] if session["iterations"] else 0
    iter_tokens = token_usage.get("total_tokens", 0)
    cumulative = prev_cumulative + iter_tokens

    entry = {
        "iteration": iteration_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot,
        "score": score,
        "token_usage": token_usage,
        "cumulative_tokens": cumulative,
        "changes": change_info,
    }
    session["iterations"].append(entry)
    save_session(session)

    # Print
    print(f"\n  Iteration #{iteration_num} recorded")
    if score is not None:
        print(f"  Score: {score:.1f}/100")
    print(f"  Board: {snapshot['board_w_mm']}x{snapshot['board_h_mm']}mm "
          f"({snapshot['trace_count']} traces, {snapshot['via_count']} vias)")
    print(f"  Tokens this run: {iter_tokens:,}  |  Cumulative: {cumulative:,}")

    if change_info:
        cs = change_info["summary"]
        print(f"\n  Change: {change_info['classification'].upper().replace('_', ' ')}")
        if cs["components_moved"]:
            print(f"    Moved: {cs['components_moved']} components "
                  f"(max {cs['max_move_mm']:.1f}mm, avg {cs['avg_move_mm']:.1f}mm)")
        if cs["components_rotated"]:
            print(f"    Rotated: {cs['components_rotated']} components")
        if cs["components_added"]:
            print(f"    Added: {cs['components_added']} components")
        if cs["components_removed"]:
            print(f"    Removed: {cs['components_removed']} components")
        if change_info["board_resized"]:
            print(f"    Board: {change_info['board_size_before']} -> {change_info['board_size_after']}")
        print(f"    Traces: {change_info['trace_delta']:+d}  "
              f"Vias: {change_info['via_delta']:+d}  "
              f"Length: {change_info['trace_length_delta_mm']:+.1f}mm")
    else:
        print("\n  Baseline (no previous iteration to compare)")
    print()


def cmd_summary(args):
    session = load_session()
    iters = session["iterations"]
    if not iters:
        print("No iterations recorded.")
        return

    print(f"\n  Layout Session — {len(iters)} iterations")
    print(f"  {'─' * 56}")
    print(f"  {'#':<4} {'Score':>6} {'Tokens':>8} {'Cumul':>8} {'Change':<20} {'Moved'}")
    print(f"  {'─' * 56}")

    for it in iters:
        score_str = f"{it['score']:.1f}" if it["score"] is not None else "—"
        tok = it["token_usage"].get("total_tokens", 0)
        cum = it["cumulative_tokens"]
        if it["changes"]:
            cls = it["changes"]["classification"].replace("_", " ")
            moved = it["changes"]["summary"]["components_moved"]
            moved_str = f"{moved} ({it['changes']['summary']['max_move_mm']:.0f}mm max)" if moved else "—"
        else:
            cls = "baseline"
            moved_str = "—"
        print(f"  {it['iteration']:<4} {score_str:>6} {tok:>8,} {cum:>8,} {cls:<20} {moved_str}")

    # Score trend
    scores = [it["score"] for it in iters if it["score"] is not None]
    if len(scores) >= 2:
        delta = scores[-1] - scores[0]
        print(f"\n  Score: {scores[0]:.1f} -> {scores[-1]:.1f} ({'+' if delta >= 0 else ''}{delta:.1f})")
        print(f"  Total tokens: {iters[-1]['cumulative_tokens']:,}")
        if delta != 0:
            efficiency = iters[-1]["cumulative_tokens"] / abs(delta)
            print(f"  Tokens per score point: {efficiency:,.0f}")

    # Stagnation detection
    if len(scores) >= 3:
        recent = scores[-3:]
        spread = max(recent) - min(recent)
        if spread < 1.0:
            print(f"\n  ⚠ Score stagnant ({spread:.1f} spread over last 3 runs) — consider major redesign")

    print()


def cmd_diff(args):
    session = load_session()
    iters = session["iterations"]
    a_idx, b_idx = args.iter_a, args.iter_b
    if a_idx >= len(iters) or b_idx >= len(iters):
        print(f"Invalid iteration index (have {len(iters)} iterations)")
        return

    a, b = iters[a_idx]["snapshot"], iters[b_idx]["snapshot"]
    changes = diff_snapshots(a, b)

    print(f"\n  Diff: iteration #{a_idx} -> #{b_idx}")
    print(f"  Classification: {changes['classification'].upper().replace('_', ' ')}")
    print(f"\n  Board: {a['board_w_mm']}x{a['board_h_mm']}mm -> {b['board_w_mm']}x{b['board_h_mm']}mm")
    print(f"  Traces: {a['trace_count']} -> {b['trace_count']} ({changes['trace_delta']:+d})")
    print(f"  Vias: {a['via_count']} -> {b['via_count']} ({changes['via_delta']:+d})")
    print(f"  Total trace: {a['total_trace_mm']:.1f}mm -> {b['total_trace_mm']:.1f}mm "
          f"({changes['trace_length_delta_mm']:+.1f}mm)")

    if changes["components_moved"]:
        print(f"\n  Components moved ({len(changes['components_moved'])}):")
        for m in sorted(changes["components_moved"], key=lambda x: -x["distance_mm"]):
            print(f"    {m['ref']:<8} {m['distance_mm']:6.1f}mm  "
                  f"({m['from'][0]},{m['from'][1]}) -> ({m['to'][0]},{m['to'][1]})")

    if changes["components_rotated"]:
        print(f"\n  Components rotated ({len(changes['components_rotated'])}):")
        for r in changes["components_rotated"]:
            print(f"    {r['ref']:<8} {r['from_deg']}° -> {r['to_deg']}°")

    for label, key in [("Added", "components_added"), ("Removed", "components_removed")]:
        if changes[key]:
            print(f"\n  {label}: {', '.join(changes[key])}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Track layout session progress")
    sub = parser.add_subparsers(dest="cmd")

    rec = sub.add_parser("record", help="Record a new iteration")
    rec.add_argument("pcb", help="Path to .kicad_pcb file")
    rec.add_argument("--score-json", help="Path to score result JSON for token data")

    sub.add_parser("summary", help="Show session summary")

    dif = sub.add_parser("diff", help="Compare two iterations")
    dif.add_argument("iter_a", type=int)
    dif.add_argument("iter_b", type=int)

    args = parser.parse_args()
    if args.cmd == "record":
        cmd_record(args)
    elif args.cmd == "summary":
        cmd_summary(args)
    elif args.cmd == "diff":
        cmd_diff(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
