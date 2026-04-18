#!/usr/bin/env python3
"""Compare two experiment rounds side-by-side.

Loads round detail JSONs and outputs a structured diff showing:
  - Config parameter changes with delta values
  - Score diff with direction arrows
  - Net routing status changes (newly routed / newly failed)
  - DRC diff (new vs resolved violations)
  - Phase timing comparison

Usage:
    python3 diff_rounds.py <experiments_dir> <round_a> <round_b> [--format text|json|html]
"""
from __future__ import annotations
import argparse
import json
import os
import sys


def load_round(rounds_dir: str, round_num: int) -> dict:
    path = os.path.join(rounds_dir, f"round_{round_num:04d}.json")
    if not os.path.exists(path):
        print(f"Round detail not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def diff_config(a: dict, b: dict) -> list[dict]:
    """Compare two config dicts, return list of changed params."""
    all_keys = sorted(set(list(a.keys()) + list(b.keys())))
    changes = []
    for k in all_keys:
        va = a.get(k)
        vb = b.get(k)
        if va != vb:
            delta = None
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = vb - va
            changes.append({
                "param": k,
                "round_a": va,
                "round_b": vb,
                "delta": delta,
            })
    return changes


def diff_scores(a: dict, b: dict) -> list[dict]:
    """Compare scoring fields between two rounds."""
    fields = [
        ("score", "Score"),
        ("routing.routed", "Nets Routed"),
        ("routing.failed", "Nets Failed"),
        ("routing.vias", "Vias"),
        ("routing.total_length_mm", "Trace Length (mm)"),
        ("placement.total", "Placement Score"),
        ("placement.crossover_count", "Crossovers"),
        ("placement.board_containment", "Board Containment"),
        ("placement.courtyard_overlap", "Courtyard Overlap"),
        ("timing.placement_ms", "Placement (ms)"),
        ("timing.routing_ms", "Routing (ms)"),
        ("drc.shorts", "Shorts"),
        ("drc.total", "Total DRC"),
    ]
    diffs = []
    for path, label in fields:
        va = _get_nested(a, path)
        vb = _get_nested(b, path)
        if va is None and vb is None:
            continue
        va = va or 0
        vb = vb or 0
        delta = vb - va if isinstance(va, (int, float)) else None
        arrow = ""
        if delta is not None:
            if delta > 0:
                arrow = "↑"
            elif delta < 0:
                arrow = "↓"
            else:
                arrow = "="
        diffs.append({
            "field": label,
            "round_a": va,
            "round_b": vb,
            "delta": delta,
            "arrow": arrow,
        })
    return diffs


def diff_nets(a: dict, b: dict) -> dict:
    """Compare per-net routing results between two rounds."""
    nets_a = {nr["net"]: nr for nr in a.get("per_net", [])}
    nets_b = {nr["net"]: nr for nr in b.get("per_net", [])}

    newly_routed = []
    newly_failed = []
    status_unchanged = []

    all_nets = sorted(set(list(nets_a.keys()) + list(nets_b.keys())))
    for net in all_nets:
        na = nets_a.get(net, {})
        nb = nets_b.get(net, {})
        sa = na.get("success", None)
        sb = nb.get("success", None)

        if sa is False and sb is True:
            newly_routed.append({
                "net": net,
                "old_reason": na.get("failure_reason"),
                "new_vias": nb.get("vias", 0),
                "new_length": nb.get("length_mm", 0),
            })
        elif sa is True and sb is False:
            newly_failed.append({
                "net": net,
                "reason": nb.get("failure_reason"),
            })
        elif sa == sb:
            # Check for significant changes in metrics
            if na and nb:
                via_delta = nb.get("vias", 0) - na.get("vias", 0)
                len_delta = nb.get("length_mm", 0) - na.get("length_mm", 0)
                if abs(via_delta) > 0 or abs(len_delta) > 1.0:
                    status_unchanged.append({
                        "net": net,
                        "via_delta": via_delta,
                        "length_delta": round(len_delta, 1),
                    })

    return {
        "newly_routed": newly_routed,
        "newly_failed": newly_failed,
        "metric_changes": status_unchanged,
    }


def diff_drc(a: dict, b: dict) -> dict:
    """Compare DRC violations between rounds."""
    va = a.get("drc", {})
    vb = b.get("drc", {})

    # Count diffs
    count_changes = {}
    for field in ["shorts", "unconnected", "clearance", "courtyard", "total"]:
        ca = va.get(field, 0)
        cb = vb.get(field, 0)
        if ca != cb:
            count_changes[field] = {"round_a": ca, "round_b": cb, "delta": cb - ca}

    return {"count_changes": count_changes}


def _get_nested(d: dict, path: str):
    """Get dotted-path value from nested dict."""
    parts = path.split(".")
    for p in parts:
        if isinstance(d, dict):
            d = d.get(p)
        else:
            return None
    return d


def format_text(round_a: int, round_b: int,
                config_diff: list, score_diff: list,
                net_diff: dict, drc_diff: dict) -> str:
    """Format diff as human-readable colored text."""
    lines = []
    lines.append(f"=== Round {round_a} vs Round {round_b} ===\n")

    # Config
    lines.append("--- Config Changes ---")
    if config_diff:
        for c in config_diff:
            delta_str = f" (Δ={c['delta']:+.4g})" if c["delta"] is not None else ""
            lines.append(f"  {c['param']:30s}  {c['round_a']!s:>12s} → {c['round_b']!s:<12s}{delta_str}")
    else:
        lines.append("  (no config changes)")
    lines.append("")

    # Scores
    lines.append("--- Score Comparison ---")
    for s in score_diff:
        delta_str = ""
        if s["delta"] is not None:
            sign = "+" if s["delta"] >= 0 else ""
            delta_str = f"  ({sign}{s['delta']:.2f} {s['arrow']})"
        lines.append(f"  {s['field']:25s}  {s['round_a']:>10}  →  {s['round_b']:<10}{delta_str}")
    lines.append("")

    # Net changes
    lines.append("--- Net Routing Changes ---")
    if net_diff["newly_routed"]:
        lines.append(f"  Newly routed ({len(net_diff['newly_routed'])}):")
        for n in net_diff["newly_routed"]:
            lines.append(f"    ✓ {n['net']} (was: {n['old_reason']}, now: {n['new_vias']} vias, {n['new_length']:.1f}mm)")
    if net_diff["newly_failed"]:
        lines.append(f"  Newly failed ({len(net_diff['newly_failed'])}):")
        for n in net_diff["newly_failed"]:
            lines.append(f"    ✗ {n['net']} ({n['reason']})")
    if net_diff["metric_changes"]:
        lines.append(f"  Metric changes ({len(net_diff['metric_changes'])}):")
        for n in net_diff["metric_changes"][:10]:
            lines.append(f"    ~ {n['net']} vias:{n['via_delta']:+d} length:{n['length_delta']:+.1f}mm")
    if not any([net_diff["newly_routed"], net_diff["newly_failed"], net_diff["metric_changes"]]):
        lines.append("  (no routing changes)")
    lines.append("")

    # DRC
    lines.append("--- DRC Changes ---")
    if drc_diff["count_changes"]:
        for field, vals in drc_diff["count_changes"].items():
            sign = "+" if vals["delta"] >= 0 else ""
            lines.append(f"  {field:15s}  {vals['round_a']} → {vals['round_b']} ({sign}{vals['delta']})")
    else:
        lines.append("  (no DRC changes)")

    return "\n".join(lines)


def format_json(round_a: int, round_b: int,
                config_diff: list, score_diff: list,
                net_diff: dict, drc_diff: dict) -> str:
    return json.dumps({
        "round_a": round_a,
        "round_b": round_b,
        "config_changes": config_diff,
        "score_comparison": score_diff,
        "net_routing_changes": net_diff,
        "drc_changes": drc_diff,
    }, indent=2, default=str)


def format_html(round_a: int, round_b: int,
                config_diff: list, score_diff: list,
                net_diff: dict, drc_diff: dict) -> str:
    """Generate self-contained HTML diff report."""
    rows_cfg = ""
    for c in config_diff:
        delta = f"{c['delta']:+.4g}" if c["delta"] is not None else "-"
        rows_cfg += f"<tr><td>{c['param']}</td><td>{c['round_a']}</td><td>{c['round_b']}</td><td>{delta}</td></tr>\n"

    rows_score = ""
    for s in score_diff:
        color = ""
        if s["arrow"] == "↑":
            color = ' style="color:green"'
        elif s["arrow"] == "↓":
            color = ' style="color:red"'
        delta_str = f"{s['delta']:+.2f}" if s["delta"] is not None else "-"
        rows_score += (f"<tr><td>{s['field']}</td><td>{s['round_a']}</td>"
                       f"<td>{s['round_b']}</td><td{color}>{delta_str} {s['arrow']}</td></tr>\n")

    nets_html = ""
    for n in net_diff.get("newly_routed", []):
        nets_html += f'<div class="net-ok">✓ {n["net"]} — was: {n["old_reason"]}</div>\n'
    for n in net_diff.get("newly_failed", []):
        nets_html += f'<div class="net-fail">✗ {n["net"]} — {n["reason"]}</div>\n'

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Round {round_a} vs {round_b}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
h2 {{ color: #555; margin-top: 1.5em; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.5em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f5f5f5; }}
.net-ok {{ color: green; padding: 2px 0; }}
.net-fail {{ color: red; padding: 2px 0; }}
</style></head><body>
<h1>Round {round_a} vs Round {round_b}</h1>

<h2>Config Changes</h2>
<table><tr><th>Parameter</th><th>Round {round_a}</th><th>Round {round_b}</th><th>Delta</th></tr>
{rows_cfg if rows_cfg else '<tr><td colspan="4">No config changes</td></tr>'}
</table>

<h2>Score Comparison</h2>
<table><tr><th>Metric</th><th>Round {round_a}</th><th>Round {round_b}</th><th>Change</th></tr>
{rows_score}
</table>

<h2>Net Routing Changes</h2>
{nets_html if nets_html else '<p>No routing status changes</p>'}

</body></html>"""


def main():
    parser = argparse.ArgumentParser(
        description="Compare two experiment rounds")
    parser.add_argument("experiments_dir",
                        help="Path to .experiments directory")
    parser.add_argument("round_a", type=int, help="First round number")
    parser.add_argument("round_b", type=int, help="Second round number")
    parser.add_argument("--format", "-f", default="text",
                        choices=["text", "json", "html"],
                        help="Output format (default: text)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file (default: stdout)")
    args = parser.parse_args()

    rounds_dir = os.path.join(args.experiments_dir, "rounds")
    a = load_round(rounds_dir, args.round_a)
    b = load_round(rounds_dir, args.round_b)

    config_diff = diff_config(a.get("config", {}), b.get("config", {}))
    score_diff = diff_scores(a, b)
    net_diff = diff_nets(a, b)
    drc_diff = diff_drc(a, b)

    if args.format == "json":
        output = format_json(args.round_a, args.round_b,
                             config_diff, score_diff, net_diff, drc_diff)
    elif args.format == "html":
        output = format_html(args.round_a, args.round_b,
                             config_diff, score_diff, net_diff, drc_diff)
    else:
        output = format_text(args.round_a, args.round_b,
                             config_diff, score_diff, net_diff, drc_diff)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Diff saved to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
