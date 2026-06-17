"""Pure ECharts option-dict builders for the /admin/loadtest dashboard.

Plain data in, option dict out -- no NiceGUI/connection context -- so the chart
shapes are unit-testable, matching how the existing web.py ECharts helpers are
written. web.py feeds the returned dicts straight to ``ui.echart``.
"""
from __future__ import annotations

_AXIS = "#94a3b8"
_GRID = "#1e293b"
_TITLE = {"color": "#e2e8f0", "fontSize": 13}


def _rel_seconds(samples) -> list[float]:
    if not samples:
        return []
    t0 = samples[0]["ts"]
    return [round((s["ts"] - t0), 1) for s in samples]


def _line(name, data, color):
    return {"type": "line", "name": name, "data": list(data), "smooth": True,
            "showSymbol": False, "connectNulls": True,
            "itemStyle": {"color": color}, "lineStyle": {"color": color}}


def _base(title: str, x):
    return {
        "backgroundColor": "transparent",
        "title": {"text": title, "textStyle": _TITLE},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 22, "textStyle": {"color": _AXIS, "fontSize": 10}},
        "grid": {"left": 50, "right": 16, "top": 56, "bottom": 40},
        "xAxis": {"type": "category", "data": list(x), "name": "s",
                  "axisLabel": {"color": _AXIS, "fontSize": 9}},
        "yAxis": {"type": "value", "axisLabel": {"color": _AXIS},
                  "splitLine": {"lineStyle": {"color": _GRID}}},
    }


def host_chart(samples) -> dict:
    """CPU% / mem% / loadavg over the run -- the saturation signal."""
    x = _rel_seconds(samples)
    opt = _base("Host: CPU% · mem% · loadavg", x)
    opt["series"] = [
        _line("cpu%", [s.get("cpu_pct") for s in samples], "#60a5fa"),
        _line("mem%", [s.get("mem_pct") for s in samples], "#f472b6"),
        _line("loadavg", [s.get("loadavg") for s in samples], "#fbbf24"),
    ]
    return opt


def queue_chart(samples) -> dict:
    """Build-queue depth + running over the run."""
    x = _rel_seconds(samples)
    opt = _base("Build queue: depth · running", x)
    opt["series"] = [
        _line("queued", [s.get("queue_depth") for s in samples], "#a78bfa"),
        _line("running", [s.get("queue_running") for s in samples], "#34d399"),
    ]
    return opt


def disk_chart(samples) -> dict:
    """Free disk over the run (the Phase-0 watch signal)."""
    x = _rel_seconds(samples)
    opt = _base("Disk free (MB)", x)
    opt["series"] = [_line("disk_free_mb",
                           [s.get("disk_free_mb") for s in samples], "#22d3ee")]
    opt["legend"] = {"show": False}
    return opt


def latency_bar(summary: dict, key: str = "build") -> dict:
    """p50/p95/p99 of the build (or design) latency, in seconds."""
    block = summary.get(key) or {}
    # stored latencies are seconds for build-storm, ms via events elsewhere; the
    # summary 'build'/'wait' blocks from buildstorm are already in seconds.
    vals = [block.get("p50"), block.get("p95"), block.get("p99"), block.get("max")]
    return {
        "backgroundColor": "transparent",
        "title": {"text": f"{key} latency (s)", "textStyle": _TITLE},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 50, "right": 16, "top": 44, "bottom": 36},
        "xAxis": {"type": "category", "data": ["p50", "p95", "p99", "max"],
                  "axisLabel": {"color": _AXIS}},
        "yAxis": {"type": "value", "axisLabel": {"color": _AXIS},
                  "splitLine": {"lineStyle": {"color": _GRID}}},
        "series": [{"type": "bar", "data": [v if v is not None else 0 for v in vals],
                    "itemStyle": {"color": "#60a5fa"}}],
    }


def outcome_pie(summary: dict) -> dict:
    """Success vs nonzero-rc vs failed (build-storm) or design/build/error counts."""
    if "ok" in summary:  # build-storm shape
        pairs = [("ok", summary.get("ok", 0)),
                 ("nonzero rc", summary.get("nonzero_rc", 0)),
                 ("failed", summary.get("failed", 0))]
    else:  # pipeline shape
        pairs = [("design ok", summary.get("design_ok", 0)),
                 ("build ok", summary.get("build_ok", 0)),
                 ("errors", summary.get("errors", 0))]
    return {
        "backgroundColor": "transparent",
        "title": {"text": "Outcomes", "textStyle": _TITLE},
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0, "textStyle": {"color": _AXIS}},
        "series": [{"type": "pie", "radius": ["38%", "66%"], "center": ["50%", "46%"],
                    "data": [{"name": n, "value": v} for n, v in pairs if v],
                    "label": {"color": "#cbd5e1"}}],
    }


def slot_sweep_bar(summaries) -> dict:
    """Build wall-time (p95) vs slot count -- the saturation-knee chart."""
    labels = [str(s.get("slots")) for s in summaries]
    vals = [round((s.get("build") or {}).get("p95") or 0, 1) for s in summaries]
    return {
        "backgroundColor": "transparent",
        "title": {"text": "p95 build time vs build slots", "textStyle": _TITLE},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 50, "right": 16, "top": 44, "bottom": 40},
        "xAxis": {"type": "category", "data": labels, "name": "slots",
                  "axisLabel": {"color": _AXIS}},
        "yAxis": {"type": "value", "axisLabel": {"color": _AXIS},
                  "splitLine": {"lineStyle": {"color": _GRID}}},
        "series": [{"type": "bar", "data": vals, "itemStyle": {"color": "#f59e0b"}}],
    }
