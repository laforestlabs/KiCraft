"""Pure ECharts option-dict builders for the /admin/security dashboard."""
from __future__ import annotations

_AXIS = "#94a3b8"
_GRID = "#1e293b"
_TITLE = {"color": "#e2e8f0", "fontSize": 13}

# Severity -> bar color (matches the store's SEVERITY_ORDER vocabulary).
_SEV_COLOR = {"critical": "#ef4444", "high": "#f97316", "medium": "#f59e0b",
              "low": "#60a5fa", "info": "#94a3b8", "unknown": "#64748b"}
_SEV_ORDER = ["critical", "high", "medium", "low", "info", "unknown"]


def severity_bar(counts: dict) -> dict:
    """Open findings by severity (critical first)."""
    labels = [s for s in _SEV_ORDER if s in counts]
    return {
        "backgroundColor": "transparent",
        "title": {"text": "Open findings by severity", "textStyle": _TITLE},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 50, "right": 16, "top": 44, "bottom": 36},
        "xAxis": {"type": "category", "data": labels,
                  "axisLabel": {"color": _AXIS}},
        "yAxis": {"type": "value", "axisLabel": {"color": _AXIS},
                  "splitLine": {"lineStyle": {"color": _GRID}}},
        "series": [{"type": "bar",
                    "data": [{"value": counts[s],
                              "itemStyle": {"color": _SEV_COLOR.get(s, "#64748b")}}
                             for s in labels]}],
    }


def status_pie(status_counts: dict) -> dict:
    """Open vs acknowledged findings."""
    return {
        "backgroundColor": "transparent",
        "title": {"text": "Findings by status", "textStyle": _TITLE},
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0, "textStyle": {"color": _AXIS}},
        "series": [{"type": "pie", "radius": ["38%", "66%"], "center": ["50%", "46%"],
                    "data": [{"name": k, "value": v} for k, v in status_counts.items() if v],
                    "label": {"color": "#cbd5e1"}}],
    }
