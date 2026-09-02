#!/usr/bin/env python3
"""Summarize token usage and estimated cost for a KiCraft design session.

Parses one or more supported agent transcripts (``.jsonl``) and sums token
usage across every assistant turn, including delegated sidechains, to report
the true start-to-finish cost of a design session.

KiCraft's deterministic pipeline (synthesize -> place -> route -> fab) spends no
LLM tokens. Token cost belongs to the agent runtime that performs the design
interview. This tool reads transcript records carrying assistant-message usage
objects so the session can be measured before pricing it.

Deliberately pure-stdlib (no kicraft, pcbnew, or pydantic imports) so the
skill-eval harness can reuse :func:`summarize_transcripts` without pulling in
heavy dependencies.

The ``token_usage`` dict it returns mirrors and extends the shape already used
in ``kicraft/cli/score_layout.py`` (``{... total_tokens ...}``).
"""
import argparse
import json
import os
import sys
from glob import glob

# ---------------------------------------------------------------------------
# Pricing.
#
# List prices in USD per million tokens (MTok), base input/output, as of
# 2026-06. Cache and web-tool prices are derived from these via the multipliers
# below. This table is a convenience for a rough cost model, NOT a billing
# contract: prices change, so verify them (or override with --prices) before
# relying on the dollar figures. Models are matched by family substring
# (opus/sonnet/haiku) so version/date suffixes in the id do not matter.
# ---------------------------------------------------------------------------
PRICES_PER_MTOK = {
    "opus": {"input": 15.0, "output": 75.0},
    "sonnet": {"input": 3.0, "output": 15.0},
    "haiku": {"input": 1.0, "output": 5.0},
}
CACHE_READ_MULT = 0.10      # cache hits bill at 0.1x base input
CACHE_WRITE_5M_MULT = 1.25  # 5-minute ephemeral cache write
CACHE_WRITE_1H_MULT = 2.0   # 1-hour ephemeral cache write
WEB_SEARCH_PER_1K = 10.0    # Anthropic server-side web search, USD per 1000 requests

UNKNOWN_MODEL = "unknown"


def price_for_model(model_id, prices=None):
    """Map a transcript model id (e.g. 'provider-model-version') to (family, price_entry).

    Matching is by family substring so version suffixes are irrelevant. Returns
    (UNKNOWN_MODEL, None) when no family matches, which marks the cost estimate
    as partial.
    """
    table = prices or PRICES_PER_MTOK
    mid = (model_id or "").lower()
    for family, price in table.items():
        if family in mid:
            return family, price
    return UNKNOWN_MODEL, None


def _new_bucket():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_5m_tokens": 0,
        "cache_creation_1h_tokens": 0,
        "web_search_requests": 0,
        "web_fetch_requests": 0,
        "turns": 0,
    }


def _iter_assistant_usage(path):
    """Yield (record, message, usage) for every assistant turn carrying usage."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            usage = msg.get("usage")
            if isinstance(usage, dict):
                yield rec, msg, usage


def _collect_unique_usage(paths):
    """Return {request_key: (model_id, usage)} de-duplicated across all paths.

    Agent runtimes may log one API call more than once. Keying by requestId
    (falling back to the message id, then a file+line tag) counts each distinct
    call exactly once while preserving sidechain calls.
    """
    seen = {}
    for fi, path in enumerate(paths):
        for li, (rec, msg, usage) in enumerate(_iter_assistant_usage(path)):
            key = rec.get("requestId") or msg.get("id") or f"{fi}:{li}"
            seen[key] = (msg.get("model"), usage)
    return seen


def _add_usage(bucket, usage):
    bucket["input_tokens"] += usage.get("input_tokens", 0) or 0
    bucket["output_tokens"] += usage.get("output_tokens", 0) or 0
    bucket["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0) or 0

    cc = usage.get("cache_creation")
    if isinstance(cc, dict) and (cc.get("ephemeral_5m_input_tokens") is not None
                                 or cc.get("ephemeral_1h_input_tokens") is not None):
        bucket["cache_creation_5m_tokens"] += cc.get("ephemeral_5m_input_tokens", 0) or 0
        bucket["cache_creation_1h_tokens"] += cc.get("ephemeral_1h_input_tokens", 0) or 0
    else:
        # No tier breakdown: attribute the lump sum to the cheaper 5m tier.
        bucket["cache_creation_5m_tokens"] += usage.get("cache_creation_input_tokens", 0) or 0

    stu = usage.get("server_tool_use")
    if isinstance(stu, dict):
        bucket["web_search_requests"] += stu.get("web_search_requests", 0) or 0
        bucket["web_fetch_requests"] += stu.get("web_fetch_requests", 0) or 0

    bucket["turns"] += 1


def _bucket_cost(bucket, price):
    """USD cost for a bucket given a base price entry, or None if price unknown."""
    if not price:
        return None
    inp = price["input"]
    cost = (
        bucket["input_tokens"] * inp
        + bucket["output_tokens"] * price["output"]
        + bucket["cache_read_tokens"] * inp * CACHE_READ_MULT
        + bucket["cache_creation_5m_tokens"] * inp * CACHE_WRITE_5M_MULT
        + bucket["cache_creation_1h_tokens"] * inp * CACHE_WRITE_1H_MULT
    ) / 1_000_000.0
    cost += bucket["web_search_requests"] / 1000.0 * WEB_SEARCH_PER_1K
    return cost


def _bucket_totals(bucket):
    cache_creation = bucket["cache_creation_5m_tokens"] + bucket["cache_creation_1h_tokens"]
    return {
        "input_tokens": bucket["input_tokens"],
        "output_tokens": bucket["output_tokens"],
        "cache_read_tokens": bucket["cache_read_tokens"],
        "cache_creation_tokens": cache_creation,
        "total_tokens": (bucket["input_tokens"] + bucket["output_tokens"]
                         + bucket["cache_read_tokens"] + cache_creation),
        "web_search_requests": bucket["web_search_requests"],
        "web_fetch_requests": bucket["web_fetch_requests"],
        "turns": bucket["turns"],
    }


def summarize_transcripts(paths, prices=None):
    """Summarize token usage across one or more transcript files.

    Returns a ``token_usage`` dict with combined totals, an ``estimated_cost_usd``,
    a per-model-family breakdown, and ``cost_known`` (False when any tokens came
    from a model with no price entry, so the cost is a lower bound).
    """
    table = prices or PRICES_PER_MTOK
    seen = _collect_unique_usage(paths)

    by_family = {}        # family -> bucket
    unknown_models = set()

    for model_id, usage in seen.values():
        family, price = price_for_model(model_id, table)
        if family == UNKNOWN_MODEL and model_id:
            unknown_models.add(model_id)
        bucket = by_family.setdefault(family, _new_bucket())
        _add_usage(bucket, usage)

    combined = _new_bucket()
    total_cost = 0.0
    cost_known = True
    by_model_out = {}
    for family, bucket in by_family.items():
        for k in combined:
            combined[k] += bucket[k]
        _, price = (family, table.get(family)) if family != UNKNOWN_MODEL else (family, None)
        cost = _bucket_cost(bucket, price)
        if cost is None:
            cost_known = False
        else:
            total_cost += cost
        entry = _bucket_totals(bucket)
        entry["estimated_cost_usd"] = round(cost, 4) if cost is not None else None
        by_model_out[family] = entry

    out = _bucket_totals(combined)
    out["estimated_cost_usd"] = round(total_cost, 4)
    out["cost_known"] = cost_known
    out["by_model"] = by_model_out
    out["transcripts"] = [os.path.basename(p) for p in paths]
    if unknown_models:
        out["unknown_models"] = sorted(unknown_models)
    return out


def summarize_transcript(path, prices=None):
    """Convenience wrapper for a single transcript file."""
    return summarize_transcripts([path], prices=prices)


def format_summary(token_usage):
    """Render a human-readable summary string for the CLI."""
    tu = token_usage
    lines = []
    lines.append("=" * 62)
    lines.append("  KiCraft session token usage")
    srcs = tu.get("transcripts", [])
    if srcs:
        lines.append(f"  Transcripts: {', '.join(srcs)}")
    lines.append("=" * 62)
    lines.append(f"  Input          {tu['input_tokens']:>14,}")
    lines.append(f"  Output         {tu['output_tokens']:>14,}")
    lines.append(f"  Cache read     {tu['cache_read_tokens']:>14,}  (billed ~0.1x input)")
    lines.append(f"  Cache write    {tu['cache_creation_tokens']:>14,}  (billed 1.25x-2x input)")
    lines.append(f"  {'-' * 58}")
    lines.append(f"  Total tokens   {tu['total_tokens']:>14,}  over {tu['turns']} API call(s)")
    if tu.get("web_search_requests"):
        lines.append(f"  Web searches   {tu['web_search_requests']:>14,}")

    cost = tu.get("estimated_cost_usd")
    if cost is not None:
        flag = "" if tu.get("cost_known", True) else "  (lower bound: unpriced model present)"
        lines.append(f"  {'-' * 58}")
        lines.append(f"  Est. cost      ${cost:>13,.4f}{flag}")

    by_model = tu.get("by_model", {})
    if len(by_model) > 1 or (by_model and next(iter(by_model)) != "opus"):
        lines.append(f"  {'-' * 58}")
        lines.append("  By model:")
        for fam, m in sorted(by_model.items()):
            c = m.get("estimated_cost_usd")
            cstr = f"${c:,.4f}" if c is not None else "n/a"
            lines.append(f"    {fam:<8} {m['total_tokens']:>12,} tok  {cstr:>12}  "
                         f"({m['turns']} call(s))")
    if tu.get("unknown_models"):
        lines.append(f"  Unpriced models: {', '.join(tu['unknown_models'])}")
    lines.append("")
    return "\n".join(lines)


def _expand_paths(raw_paths):
    """Expand directories to their *.jsonl files and globs to matches."""
    out = []
    for p in raw_paths:
        if os.path.isdir(p):
            out.extend(sorted(glob(os.path.join(p, "*.jsonl"))))
        elif any(ch in p for ch in "*?["):
            out.extend(sorted(glob(p)))
        else:
            out.append(p)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize token usage and estimated cost for a KiCraft "
                    "design session from supported agent transcript JSONL files.")
    parser.add_argument("transcript", nargs="+",
                        help="Transcript .jsonl file(s), a directory of them, or a glob. "
                             "Multiple are summed (a design that spanned sessions).")
    parser.add_argument("--json", action="store_true",
                        help="Emit the token_usage dict as JSON instead of a summary.")
    parser.add_argument("--prices", metavar="FILE",
                        help="JSON file overriding the per-MTok price table, e.g. "
                             '{"opus": {"input": 15, "output": 75}}.')
    args = parser.parse_args(argv)

    prices = None
    if args.prices:
        with open(args.prices, encoding="utf-8") as fh:
            prices = json.load(fh)

    paths = _expand_paths(args.transcript)
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        print(f"error: transcript(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    if not paths:
        print("error: no transcript files matched", file=sys.stderr)
        return 2

    token_usage = summarize_transcripts(paths, prices=prices)

    if args.json:
        print(json.dumps(token_usage, indent=2))
    else:
        print(format_summary(token_usage))

    if token_usage["turns"] == 0:
        print("warning: no assistant turns with usage found in transcript(s)",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
