"""Admin dashboards: tuning, self-eval, loadtest, security, user/invite management,
and core-component editing -- the ``@ui.page("/admin/...")`` surface.

Extracted from web.py (refactor roadmap Phase 3a). Importing this module REGISTERS
its routes (the ``@ui.page`` decorators run at import time), so ``web.py`` pulls it
in once near the bottom with ``from . import routes_admin``.

Shared page scaffolding (``_store``/``_current_user``/``_require_admin``) and a few
render helpers live in ``web`` and are imported below; they are resolved at request
time, and ``_store`` still reads ``web._STORE`` so the test seam is unchanged. Charts
use NiceGUI's bundled ECharts primitive (the ``server`` extra has no plotly).
"""
from __future__ import annotations
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

from nicegui import ui

from kicraft.cli.artifact_paths import LEAF_ROUTED, artifact_root

from . import billing
from .accounts import (
    CORE_COMPONENT_CATEGORIES,
    DEFAULT_TIER,
    TIERS,
    _RESET_TTL_SECONDS,
    is_admin,
)
from .config import Settings
from .host_metrics import HostMetricsStore
from .kicanvas import KiCanvasSource, KiCanvasView, kicanvas_head
from .render_serving import _register_project_dir, _resolve_project_token
from .spend_guard import SpendGuard
from .stage_driver import KICRAFT
from .storage import _discover_generated_dir
from ..parts_library import jlcparts
from kicraft.tuning.benchmark import BENCHMARK_PROMPTS

# Shared web-app helpers (see module docstring). Resolved at request time; the
# import-order contract is that web.py imports routes_admin (never the reverse first).
from .web import (
    _current_user,
    _render_scorecard,
    _render_synth_view,
    _require_admin,
    _schematic_sources,
    _signup_code,
    _store,
)


# --------------------------------------------------------------------------- #
# Admin dashboard (stats/trends + user management). Gated by _require_admin();
# charts use the ECharts primitive bundled with NiceGUI (the web server ships
# under the `server` extra, which has no plotly -- that is a `gui`-extra dep).
# --------------------------------------------------------------------------- #
# Chart colors are baked into echart option dicts, which render to <canvas> and
# cannot resolve CSS var()s — so these stay concrete hex (kept in step with the
# theme palette by hand). Everything else in this file is CSS and uses tokens.
_CHART_AXIS = "#9aa7b5"
_CHART_GRID = "#232c38"


def _echart_bar(labels, values, *, title: str, color: str = "#60a5fa") -> dict:
    """ECharts bar-chart option dict. Pure (plain lists in, dict out) so it is
    unit-testable without a UI/connection context."""
    return {
        "backgroundColor": "transparent",
        "title": {"text": title, "textStyle": {"color": "#e2e8f0", "fontSize": 13}},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 50, "right": 16, "top": 44, "bottom": 56},
        "xAxis": {"type": "category", "data": list(labels),
                  "axisLabel": {"color": _CHART_AXIS, "fontSize": 10, "rotate": 45}},
        "yAxis": {"type": "value", "axisLabel": {"color": _CHART_AXIS},
                  "splitLine": {"lineStyle": {"color": _CHART_GRID}}},
        "series": [{"type": "bar", "data": list(values),
                    "itemStyle": {"color": color}}],
    }


def _echart_line(labels, values, *, title: str, color: str = "#34d399") -> dict:
    """ECharts line/area-chart option dict (pure; see _echart_bar)."""
    opt = _echart_bar(labels, values, title=title, color=color)
    opt["series"] = [{"type": "line", "data": list(values), "smooth": True,
                      "showSymbol": False, "itemStyle": {"color": color},
                      "areaStyle": {"color": color, "opacity": 0.15}}]
    return opt


def _echart_pie(pairs, *, title: str) -> dict:
    """ECharts donut option dict from (name, value) pairs (pure; see _echart_bar)."""
    return {
        "backgroundColor": "transparent",
        "title": {"text": title, "textStyle": {"color": "#e2e8f0", "fontSize": 13}},
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0, "textStyle": {"color": _CHART_AXIS}},
        "series": [{"type": "pie", "radius": ["38%", "66%"], "center": ["50%", "46%"],
                    "data": [{"name": str(n), "value": v} for n, v in pairs],
                    "label": {"color": "#cbd5e1"}}],
    }


def _echart_multi_line(labels, series, *, title: str, y_name: str = "",
                       baseline=None, y_range=None) -> dict:
    """Multi-series line chart over a category x-axis (pure; see _echart_bar).

    ``series`` = [(name, values, color), ...]; ``baseline`` draws a dashed
    horizontal reference line (the current DEFAULT_CONFIG); ``y_range`` pins the
    y-axis to (min, max)."""
    s = []
    for i, (name, values, color) in enumerate(series):
        d = {"type": "line", "name": name, "data": list(values), "smooth": True,
             "showSymbol": False, "connectNulls": True,
             "itemStyle": {"color": color}, "lineStyle": {"color": color}}
        if i == 0 and baseline is not None:
            d["markLine"] = {
                "symbol": "none", "silent": True,
                "label": {"color": "#f59e0b", "formatter": "default", "fontSize": 9},
                "lineStyle": {"color": "#f59e0b", "type": "dashed"},
                "data": [{"yAxis": baseline}]}
        s.append(d)
    yaxis = {"type": "value", "name": y_name, "nameTextStyle": {"color": _CHART_AXIS},
             "axisLabel": {"color": _CHART_AXIS},
             "splitLine": {"lineStyle": {"color": _CHART_GRID}}}
    if y_range is not None:
        yaxis["min"], yaxis["max"] = y_range
    return {
        "backgroundColor": "transparent",
        "title": {"text": title, "textStyle": {"color": "#e2e8f0", "fontSize": 13}},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 22, "textStyle": {"color": _CHART_AXIS, "fontSize": 10}},
        "grid": {"left": 54, "right": 18, "top": 58, "bottom": 34},
        "xAxis": {"type": "category", "data": list(labels), "name": "generation",
                  "nameLocation": "middle", "nameGap": 24,
                  "nameTextStyle": {"color": _CHART_AXIS},
                  "axisLabel": {"color": _CHART_AXIS, "fontSize": 10}},
        "yAxis": yaxis,
        "series": s,
    }


def _echart_scatter(series, *, title: str, x_name: str = "", y_name: str = "") -> dict:
    """Scatter chart (pure; see _echart_bar). ``series`` = [(name, points, color,
    size), ...] where points = [[x, y], ...]."""
    s = [{"type": "scatter", "name": name, "data": list(points),
          "symbolSize": size, "itemStyle": {"color": color}}
         for name, points, color, size in series]
    return {
        "backgroundColor": "transparent",
        "title": {"text": title, "textStyle": {"color": "#e2e8f0", "fontSize": 13}},
        "tooltip": {"trigger": "item"},
        "legend": {"top": 22, "textStyle": {"color": _CHART_AXIS, "fontSize": 10}},
        "grid": {"left": 54, "right": 18, "top": 58, "bottom": 44},
        "xAxis": {"type": "value", "name": x_name, "nameLocation": "middle",
                  "nameGap": 26, "nameTextStyle": {"color": _CHART_AXIS},
                  "axisLabel": {"color": _CHART_AXIS},
                  "splitLine": {"lineStyle": {"color": _CHART_GRID}}},
        "yAxis": {"type": "value", "name": y_name, "nameTextStyle": {"color": _CHART_AXIS},
                  "axisLabel": {"color": _CHART_AXIS},
                  "splitLine": {"lineStyle": {"color": _CHART_GRID}}},
        "series": s,
    }


# Host-resource (drive / RAM / CPU) trends ---------------------------------- #
# Timescales for the host-usage charts; the default ("7d") matches the user's
# request. ``_HOST_TIMESCALE_SECONDS`` -> None means "all history".
_HOST_TIMESCALES = ["1h", "6h", "24h", "7d", "30d", "90d", "all"]
_HOST_TIMESCALE_SECONDS = {
    "1h": 3600.0, "6h": 21600.0, "24h": 86400.0,
    "7d": 604800.0, "30d": 2592000.0, "90d": 7776000.0, "all": None,
}


def _host_downsample(rows: list[dict], max_points: int = 240) -> list[dict]:
    """Mean-bucket the host series to <= ``max_points`` rows so a 30-day /
    30-second-cadence window (~86k rows) keeps a small JSON payload for the
    browser. The last ``ts`` in each bucket anchors it; metrics are averaged."""
    if len(rows) <= max_points:
        return rows
    bucket = math.ceil(len(rows) / max_points)
    keys = ("cpu_pct", "mem_pct", "disk_pct", "mem_used_mb",
            "disk_used_mb", "disk_total_mb")
    out: list[dict] = []
    for i in range(0, len(rows), bucket):
        chunk = rows[i:i + bucket]
        agg: dict = {"ts": chunk[-1]["ts"]}
        for k in keys:
            vals = [c[k] for c in chunk if c.get(k) is not None]
            agg[k] = round(sum(vals) / len(vals), 2) if vals else None
        out.append(agg)
    return out


def _host_usage_chart(rows: list[dict], key: str, *, title: str,
                      color: str, y_name: str = "", y_max: float | None = 100.0,
                      y_fmt: str | None = None) -> dict:
    """ECharts *time-axis* area line for one host metric (pure; see
    ``_echart_bar``). Points are ``[ts_ms, value]`` so ECharts picks axis ticks
    and tooltip time formatting itself -- better than a category axis for a
    multi-timescale view. ``y_max=100`` pins percent charts 0..100; pass None
    for an auto-scaled axis (drive MB shows raw magnitude)."""
    pts = [[r["ts"] * 1000.0, r[key]] for r in rows if r.get(key) is not None]
    yaxis: dict = {"type": "value", "name": y_name,
                   "nameTextStyle": {"color": _CHART_AXIS},
                   "axisLabel": {"color": _CHART_AXIS},
                   "splitLine": {"lineStyle": {"color": _CHART_GRID}}}
    if y_fmt is not None:
        yaxis["axisLabel"]["formatter"] = y_fmt
    if y_max is not None:
        yaxis["min"], yaxis["max"] = 0, y_max
    return {
        "backgroundColor": "transparent",
        "title": {"text": title, "textStyle": {"color": "#e2e8f0", "fontSize": 13}},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 48, "right": 18, "top": 44, "bottom": 32},
        "xAxis": {"type": "time",
                  "axisLabel": {"color": _CHART_AXIS, "fontSize": 10},
                  "splitLine": {"lineStyle": {"color": _CHART_GRID}}},
        "yAxis": yaxis,
        "series": [{"type": "line", "data": pts, "smooth": True,
                    "showSymbol": False, "connectNulls": True,
                    "itemStyle": {"color": color}, "lineStyle": {"color": color},
                    "areaStyle": {"color": color, "opacity": 0.15}}],
    }


def _render_host_charts(selection: str) -> None:
    """Build the three host-usage charts (drive / RAM / CPU) for the given
    timescale label. Pure on the rows read from the host-metrics store; never
    raises -- a missing/empty store shows a friendly placeholder card so the
    admin overview always renders even before the sampler has any data."""
    seconds = _HOST_TIMESCALE_SECONDS.get(selection)
    since = None if seconds is None else (time.time() - seconds)
    try:
        rows = HostMetricsStore().series(since=since)
    except Exception:
        rows = []
    rows = _host_downsample(rows)

    def _card(key: str, title: str, color: str) -> None:
        with ui.card().classes("flex-1").style(_admin_card_style()):
            if rows:
                ui.echart(_host_usage_chart(rows, key, title=title, color=color,
                         y_name="%")) \
                    .classes("w-full").style("height:240px")
            else:
                ui.label("No host metrics yet — the sampler starts with the "
                         "web server.").classes("text-xs") \
                    .style("color:#64748b;align-self:center")

    with ui.row().classes("w-full flex-wrap gap-3"):
        _card("cpu_pct", "CPU usage (%)", "#60a5fa")
        _card("mem_pct", "RAM usage (%)", "#f472b6")
        _card("disk_pct", "Drive usage (%)", "#22d3ee")



def _admin_header(active: str) -> None:
    """Shared header for the admin pages; `active` names the current sub-page."""
    with ui.header().classes("items-center justify-between") \
            .style("background:var(--kc-surface);border-bottom:1px solid var(--kc-border)"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label(f"admin · {active}").classes("text-sm").style("color:#94a3b8")
        with ui.row().classes("items-center gap-2"):
            ui.button("Overview", icon="insights",
                      on_click=lambda: ui.navigate.to("/admin")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Users", icon="group",
                      on_click=lambda: ui.navigate.to("/admin/users")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Invites", icon="vpn_key",
                      on_click=lambda: ui.navigate.to("/admin/invites")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Components", icon="memory",
                      on_click=lambda: ui.navigate.to("/admin/core-components")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Self-Eval", icon="science",
                      on_click=lambda: ui.navigate.to("/admin/self-eval")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Tuning", icon="tune",
                      on_click=lambda: ui.navigate.to("/admin/tuning")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Tidiness A/B", icon="grid_view",
                      on_click=lambda: ui.navigate.to("/admin/tidiness-ab")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Load", icon="speed",
                      on_click=lambda: ui.navigate.to("/admin/loadtest")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Security", icon="security",
                      on_click=lambda: ui.navigate.to("/admin/security")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Support", icon="support_agent",
                      on_click=lambda: ui.navigate.to("/admin/support")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Back to workspace", icon="arrow_back",
                      on_click=lambda: ui.navigate.to("/")) \
                .props("flat dense no-caps color=white").classes("text-xs")


def _admin_card_style() -> str:
    return "background:var(--kc-surface);border:1px solid var(--kc-border);min-width:380px"


# --------------------------------------------------------------------------- #
# Admin: auto-tuning results (kicraft.tuning runs). Read-only visualization of
# how the routed Pareto objective improves and how the searched parameters
# converge over CMA-ES generations. Reads each run dir (tuning.db + checkpoint)
# via kicraft.tuning.report_data; auto-refreshes so a live run updates in place.
# --------------------------------------------------------------------------- #
_TUNE_PALETTE = ["#60a5fa", "#34d399", "#fbbf24", "#f87171", "#a78bfa",
                 "#22d3ee", "#fb923c", "#4ade80", "#e879f9", "#f472b6"]


def _tuning_out_roots() -> list[Path]:
    """Every root a tuning run can live under (GUI default + repo logs)."""
    base = Path(getattr(Settings.from_env(), "projects_dir",
                        Path.home() / ".kicraft" / "projects"))
    roots = [base.parent / "tuning",
             Path(__file__).resolve().parents[2] / "logs" / "tuning"]
    out: list[Path] = []
    seen: set = set()
    for r in roots:
        try:
            rp = r.resolve()
        except OSError:
            continue
        if rp not in seen and r.is_dir():
            seen.add(rp)
            out.append(r)
    return out


def _tuning_detail_ui(d: dict) -> None:
    """Render one tuning run's stat cards + charts into the current container."""
    gens = d["gens"]
    active = d["active_params"]
    base = d["baseline"] or {}

    def _stat(label: str, value: str) -> None:
        with ui.card().style(_admin_card_style() + ";min-width:150px;padding:8px 12px"):
            ui.label(label).classes("text-xs").style("color:#64748b")
            ui.label(value).classes("text-lg font-bold").style("color:#e2e8f0")

    with ui.row().classes("w-full gap-3 flex-wrap"):
        _stat("generation", str(d["n_gens"]))
        _stat("configs tried", str(d["n_configs"]))
        _stat("train / holdout", f"{d['n_train'] or '?'} / {d['n_holdout'] or '?'}")
        _stat("scalarization", d["scalarization"] or "—")
        if base.get("fab") is not None:
            _stat("default fab-ready", f"{base['fab']:.2f}")

    if not gens:
        ui.label("Baseline still evaluating — charts populate once generation 0 "
                 "completes.").classes("text-sm").style("color:#94a3b8")

    labels = [g["gen"] for g in gens]

    def _col(metric: str, where: str):
        return [None if not g.get(where) else g[where].get(metric) for g in gens]

    if gens:
        with ui.row().classes("w-full gap-3 flex-wrap"):
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_multi_line(labels, [
                    ("train best", _col("fab", "train"), "#34d399"),
                    ("holdout", _col("fab", "holdout"), "#60a5fa")],
                    title="Fab-ready rate / generation", y_name="rate",
                    baseline=base.get("fab"), y_range=(0, 1))) \
                    .classes("w-full").style("height:260px")
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_multi_line(labels, [
                    ("train best", _col("j", "train"), "#a78bfa"),
                    ("holdout", _col("j", "holdout"), "#60a5fa")],
                    title="Objective J / generation", y_name="J")) \
                    .classes("w-full").style("height:260px")
        with ui.row().classes("w-full gap-3 flex-wrap"):
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_multi_line(labels, [
                    ("train best", _col("drc", "train"), "#f87171")],
                    title="Mean DRC (shorts+unconnected) / generation",
                    y_name="violations", baseline=base.get("drc"))) \
                    .classes("w-full").style("height:260px")
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_multi_line(labels, [
                    ("train best", _col("wall", "train"), "#22d3ee")],
                    title="Mean build time / generation", y_name="seconds",
                    baseline=base.get("wall"))) \
                    .classes("w-full").style("height:260px")

        with ui.card().classes("w-full").style(_admin_card_style()):
            series = []
            for i, p in enumerate(active):
                tr = {pt["gen"]: pt["norm"] for pt in d["param_traces"].get(p, [])}
                series.append((p, [tr.get(g) for g in labels],
                               _TUNE_PALETTE[i % len(_TUNE_PALETTE)]))
            ui.echart(_echart_multi_line(
                labels, series, title="Parameter convergence (normalized 0–1)",
                y_name="norm", y_range=(0, 1))).classes("w-full").style("height:320px")

    pts = d["points"]
    if pts:
        dom = [[p["wall"], p["fab"]] for p in pts if not p["front"] and not p["baseline"]]
        front = [[p["wall"], p["fab"]] for p in pts if p["front"] and not p["baseline"]]
        baseln = [[p["wall"], p["fab"]] for p in pts if p["baseline"]]
        with ui.card().classes("w-full").style(_admin_card_style()):
            ui.echart(_echart_scatter([
                ("evaluated", dom, "#475569", 7),
                ("Pareto front", front, "#34d399", 12),
                ("default", baseln, "#f59e0b", 15)],
                title="Pareto archive — fab-ready vs build time",
                x_name="mean build time (s)", y_name="fab-ready rate")) \
                .classes("w-full").style("height:340px")

    if active and gens:
        with ui.card().classes("w-full").style(_admin_card_style()):
            ui.label("Active params: current best vs default").classes(
                "text-sm font-bold").style("color:#94a3b8")
            with ui.row().classes("w-full gap-3 text-xs").style("color:#64748b"):
                ui.label("param").style("width:260px")
                ui.label("default").style("width:110px")
                ui.label("current best").style("width:120px")
            for p in active:
                tr = d["param_traces"].get(p, [])
                cur = tr[-1]["value"] if tr else None
                with ui.row().classes("w-full gap-3 text-xs").style(
                        "border-top:1px solid var(--kc-border);padding:3px 0"):
                    ui.label(p).classes("font-mono").style("width:260px;color:#cbd5e1")
                    ui.label(f"{d['defaults'].get(p, 0):.4g}").style(
                        "width:110px;color:#94a3b8")
                    ui.label("—" if cur is None else f"{cur:.4g}").style(
                        "width:120px;color:#e2e8f0")


@ui.page("/admin/tuning")
def admin_tuning_page():
    """Admin: list of auto-tuning runs (kicraft.tuning), newest first."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect
    ui.dark_mode().enable()
    _admin_header("tuning")
    from kicraft.tuning import report_data

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1300px"):
        ui.label("Auto-tuning runs").classes("text-2xl font-bold text-white")
        ui.label("CMA-ES tuning of the default placement/routing config against the "
                 "routed Pareto objective (fab-ready · DRC · build time).") \
            .classes("text-sm").style("color:#94a3b8")
        ui.separator().style("background:var(--kc-border);margin-top:4px")
        runs_box = ui.column().classes("w-full gap-0")

        def render_list():
            runs = [report_data.run_overview(d)
                    for d in report_data.discover_runs(_tuning_out_roots())]
            runs_box.clear()
            with runs_box:
                if not runs:
                    ui.label("No tuning runs under ~/.kicraft/tuning or logs/tuning yet.") \
                        .classes("text-xs").style("color:#64748b")
                    return
                with ui.row().classes("w-full items-center gap-3 text-xs").style(
                        "padding:4px 6px;color:#64748b"):
                    ui.label("run").style("width:210px")
                    ui.label("state").style("width:90px")
                    ui.label("gen").style("width:54px")
                    ui.label("configs").style("width:70px")
                    ui.label("default fab").style("width:96px")
                    ui.label("best fab").style("width:80px")
                for r in runs:
                    state = ("done" if r["finished"]
                             else "running" if r["running"] else "new")
                    col = {"done": "#4ade80", "running": "#fbbf24",
                           "new": "#64748b"}[state]
                    row = ui.row().classes(
                        "w-full items-center gap-3 text-xs cursor-pointer").style(
                        "border-top:1px solid var(--kc-border);padding:5px 6px")
                    tok = _register_project_dir(Path(r["path"]))
                    row.on("click", lambda _e=None, t=tok:
                           ui.navigate.to(f"/admin/tuning/run?run={quote(t)}"))
                    with row:
                        ui.label(r["name"]).classes("font-mono").style(
                            "width:210px;color:#e2e8f0")
                        ui.label(state).style(f"width:90px;color:{col}")
                        ui.label(str(r["gen"])).style("width:54px;color:#cbd5e1")
                        ui.label(str(r["n_configs"])).style("width:70px;color:#cbd5e1")
                        ui.label("—" if r["baseline_fab"] is None
                                 else f"{r['baseline_fab']:.2f}").style(
                            "width:96px;color:#cbd5e1")
                        ui.label("—" if r["best_fab"] is None
                                 else f"{r['best_fab']:.2f}").style(
                            "width:80px;color:#cbd5e1")

        ui.timer(10.0, render_list)
        render_list()


@ui.page("/admin/tuning/run")
def admin_tuning_run_page(run: str = ""):
    """Admin: charts for one tuning run; auto-refreshes for a live run."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect
    ui.dark_mode().enable()
    _admin_header("tuning")
    from kicraft.tuning import report_data

    run_dir = _resolve_project_token(run) if run else None
    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1300px"):
        ui.button("← All runs", icon="arrow_back",
                  on_click=lambda: ui.navigate.to("/admin/tuning")) \
            .props("flat dense no-caps color=white").classes("text-xs")
        if run_dir is None or not run_dir.is_dir():
            ui.label("Run not found (the link may be stale).").classes(
                "text-sm").style("color:#f87171")
            return
        ui.label(run_dir.name).classes("text-xl font-bold").style("color:#e2e8f0")
        body = ui.column().classes("w-full gap-3")

        def render():
            body.clear()
            try:
                d = report_data.load_run(run_dir)
            except Exception as exc:  # noqa: BLE001 — surface, don't crash the page
                with body:
                    ui.label(f"(could not load run: {exc})").classes(
                        "text-xs").style("color:#f87171")
                return
            with body:
                _tuning_detail_ui(d)

        ui.timer(15.0, render)
        render()


# --------------------------------------------------------------------------- #
# Admin: soft-tidiness A/B galleries (scripts/soft_tidiness_ab.py). Each run is a
# dir holding a self-contained ``index.html`` (classic|soft leaf renders + tidiness
# and routing metrics); the viewer just iframes it. Discovery mirrors self-eval /
# tuning: a durable root (~/.kicraft/tidiness_ab) plus the repo's logs/tidiness_ab.
# --------------------------------------------------------------------------- #
def _tidiness_ab_out_roots() -> list[Path]:
    """Every root an A/B gallery can live under -- the durable sibling of the
    projects dir (where the harness defaults now) and the repo ``logs/`` dir.
    Existing dirs only, de-duplicated by resolved path."""
    base = Path(getattr(Settings.from_env(), "projects_dir",
                        Path.home() / ".kicraft" / "projects"))
    roots = [base.parent / "tidiness_ab",
             Path(__file__).resolve().parents[2] / "logs" / "tidiness_ab"]
    out: list[Path] = []
    seen: set = set()
    for r in roots:
        try:
            rp = r.resolve()
        except OSError:
            continue
        if rp not in seen and r.is_dir():
            seen.add(rp)
            out.append(r)
    return out


def _tidiness_ab_run_dirs() -> list[Path]:
    """Every gallery dir (one holding an ``index.html``) across all roots, newest
    first by that file's mtime. Capped so a long-lived box never lists unboundedly."""
    cands: list[Path] = []
    for root in _tidiness_ab_out_roots():
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for d in entries:
            if d.is_dir() and (d / "index.html").is_file():
                cands.append(d)
    cands.sort(key=lambda d: (d / "index.html").stat().st_mtime, reverse=True)
    return cands[:50]


def _tidiness_ab_summary(d: Path) -> str:
    """A one-line summary from ``summary.json`` (design count + how many regressed
    on routing under soft), best-effort -- '' when the file is missing/unreadable."""
    try:
        data = json.loads((d / "summary.json").read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(data, list):
        return ""
    reg = 0
    for r in data:
        if not isinstance(r, dict):
            continue
        unc = r.get("unconnected")
        # Variant pair is [baseline, candidate]; older runs are classic/soft.
        base, cand = (r.get("variants") or ["classic", "soft"])[:2]
        if (isinstance(unc, dict) and unc.get(base) is not None
                and unc.get(cand) is not None and unc[cand] > unc[base]):
            reg += 1
    tail = f" · {reg} routing-regressed" if reg else ""
    return f"{len(data)} designs{tail}"


@ui.page("/admin/tidiness-ab")
def admin_tidiness_ab_page():
    """Admin: list of soft-tidiness A/B galleries, newest first. Click one to open
    its self-contained classic|soft render page in the viewer."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect
    ui.dark_mode().enable()
    _admin_header("tidiness A/B")

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1300px"):
        ui.label("Soft-tidiness A/B").classes("text-2xl font-bold text-white")
        ui.label("Classic SA vs the soft-tidiness placement term, per leaf: "
                 "side-by-side renders plus orientation / row-residual / routing "
                 "metrics. Generated by scripts/soft_tidiness_ab.py ($0, no LLM); "
                 "the rigorous routing verdict is the N-of-3 median.") \
            .classes("text-sm").style("color:#94a3b8")
        ui.separator().style("background:var(--kc-border);margin-top:4px")
        runs_box = ui.column().classes("w-full gap-0")

        def render_list():
            runs = _tidiness_ab_run_dirs()
            runs_box.clear()
            with runs_box:
                if not runs:
                    ui.label("No A/B galleries under ~/.kicraft/tidiness_ab or "
                             "logs/tidiness_ab yet. Generate one with "
                             "scripts/soft_tidiness_ab.py.") \
                        .classes("text-xs").style("color:#64748b")
                    return
                with ui.row().classes("w-full items-center gap-3 text-xs").style(
                        "padding:4px 6px;color:#64748b"):
                    ui.label("gallery").style("width:300px")
                    ui.label("summary").classes("flex-1")
                for d in runs:
                    tok = _register_project_dir(d)
                    row = ui.row().classes(
                        "w-full items-center gap-3 text-xs cursor-pointer").style(
                        "border-top:1px solid var(--kc-border);padding:6px 6px")
                    row.on("click", lambda _e=None, t=tok:
                           ui.navigate.to(f"/admin/tidiness-ab/view?run={quote(t)}"))
                    with row:
                        ui.icon("grid_view").style("color:#60a5fa")
                        ui.label(d.name).classes("font-mono").style(
                            "width:280px;color:#e2e8f0")
                        ui.label(_tidiness_ab_summary(d)).classes("flex-1").style(
                            "color:#94a3b8")

        ui.timer(15.0, render_list)
        render_list()


@ui.page("/admin/tidiness-ab/view")
def admin_tidiness_ab_view_page(run: str = ""):
    """Admin: render one A/B gallery in an <iframe>. The page is a self-contained
    HTML doc (light theme, inline SVG), so it is isolated from the dark admin chrome
    and served token-gated via /tidiness-ab/<token>/index.html."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect
    ui.dark_mode().enable()
    _admin_header("tidiness A/B")

    run_dir = _resolve_project_token(run) if run else None
    with ui.column().classes("w-full mx-auto p-2 gap-2").style("max-width:1300px"):
        ui.button("← All galleries", icon="arrow_back",
                  on_click=lambda: ui.navigate.to("/admin/tidiness-ab")) \
            .props("flat dense no-caps color=white").classes("text-xs")
        if (run_dir is None or not run_dir.is_dir()
                or not (run_dir / "index.html").is_file()):
            ui.label("Gallery not found (the link may be stale).").classes(
                "text-sm").style("color:#f87171")
            return
        ui.label(run_dir.name).classes("text-lg font-bold font-mono").style(
            "color:#e2e8f0")
        # `run` is the (URL-safe base64 + '.') token verbatim -- no slashes, so it
        # slots straight into the path without re-quoting.
        # sanitize=False is REQUIRED: ui.html defaults to DOMPurify, which strips
        # <iframe> outright (the gallery would render as a blank page). Same escape
        # hatch KiCanvas embeds use; the framed src is our own token-gated HTML.
        ui.html(
            f'<iframe src="/tidiness-ab/{run}/index.html" title="A/B gallery" '
            'style="width:100%;height:calc(100vh - 150px);border:1px solid '
            'var(--kc-border);border-radius:8px;background:#fff"></iframe>',
            sanitize=False) \
            .classes("w-full")


# --------------------------------------------------------------------------- #
# Admin: support -- failed boards, user-reported highlighting, headless
# /kicraft-investigate. Every failed run auto-files a support_reports row; the
# ones a user actually reported (Support button, or feedback on a failure) are
# highlighted. Investigations run via kicraft.server.investigate_runner.
# --------------------------------------------------------------------------- #
def _is_user_reported(r) -> bool:
    """A report a human actually filed: the manual Support button (kind='user')
    or freeform feedback typed into the failure dialog (a message attached to an
    otherwise silent auto-filed row)."""
    return r.kind == "user" or bool((r.message or "").strip())


@ui.page("/admin/support")
def admin_support_page():
    """Admin: triage failed boards. Lists every support report newest-first,
    highlights the user-reported ones, and runs /kicraft-investigate headlessly
    (on-demand, or auto on new user reports per the page toggle)."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect
    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    _admin_header("support")

    store = _store()

    def guard() -> bool:
        """Defense in depth: never trust the page-load gate for a mutation."""
        if not is_admin(_current_user()):
            ui.notify("Admin access required.", color="warning")
            return False
        return True

    detail_dialog = ui.dialog()
    report_dialog = ui.dialog()

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1300px"):
        ui.label("Support").classes("text-2xl font-bold text-white")
        ui.label("Every failed board files a report automatically; the ones a "
                 "user actually reported are highlighted. Run the investigate "
                 "skill on any report for a ranked pipeline-gap analysis.") \
            .classes("text-sm").style("color:#94a3b8")

        with ui.row().classes("items-center gap-3"):
            auto = ui.switch(
                "Auto-investigate new user reports",
                value=(store.get_setting("support.auto_investigate", "1") == "1"))

            def on_toggle(e) -> None:
                if not guard():
                    auto.value = (store.get_setting(
                        "support.auto_investigate", "1") == "1")
                    return
                store.set_setting("support.auto_investigate",
                                  "1" if e.value else "0")
                ui.notify("Auto-investigate " + ("on" if e.value else "off"),
                          color="positive")
            auto.on_value_change(on_toggle)
            ui.label("A user report launches a headless /kicraft-investigate "
                     "run — each is a real Claude Code session.") \
                .classes("text-xs").style("color:#64748b")

        with ui.row().classes("items-center gap-3"):
            auto_err = ui.switch(
                "Auto-investigate failed builds (error_auto, capped/day)",
                value=(store.get_setting(
                    "support.auto_investigate_errors", "0") == "1"))

            def on_toggle_err(e) -> None:
                if not guard():
                    auto_err.value = (store.get_setting(
                        "support.auto_investigate_errors", "0") == "1")
                    return
                store.set_setting("support.auto_investigate_errors",
                                  "1" if e.value else "0")
                ui.notify("Auto-investigate failed builds "
                          + ("on" if e.value else "off"), color="positive")
            auto_err.on_value_change(on_toggle_err)
            ui.label("OFF by default: every failed build files an error_auto "
                     "report; this triages them headlessly, capped per day "
                     "(KICRAFT_INVESTIGATE_ERRORS_DAILY_CAP, default 6).") \
                .classes("text-xs").style("color:#64748b")

        ui.separator().style("background:var(--kc-border);margin-top:4px")

        state = {"filter": "all"}   # all | user | new | reviewed
        cards_box = ui.row().classes("w-full flex-wrap gap-3")
        filter_box = ui.row().classes("items-center gap-2")
        list_box = ui.column().classes("w-full gap-0")

        # ---- dialogs ----------------------------------------------------------
        def show_details(r) -> None:
            d = r.diagnostics or {}
            detail_dialog.clear()
            with detail_dialog, ui.card().classes("w-[820px] max-w-[96vw] gap-2") \
                    .style("background:var(--kc-surface);"
                           "border:1px solid var(--kc-border-strong)"):
                title = f"Report #{r.id}" + (f" · {r.board_code}"
                                             if r.board_code else "")
                ui.label(title).classes("text-lg font-bold text-white")
                ui.label(f"{r.kind} · {r.status} · {r.created_at[:19]}") \
                    .classes("text-xs").style("color:#94a3b8")
                if r.message:
                    ui.label("User message").classes("text-xs") \
                        .style("color:#64748b")
                    ui.label(r.message).classes("text-sm whitespace-pre-wrap") \
                        .style("color:#e2e8f0")
                if d.get("brief"):
                    ui.label("Brief").classes("text-xs").style("color:#64748b")
                    ui.label(str(d["brief"])) \
                        .classes("text-sm whitespace-pre-wrap") \
                        .style("color:#cbd5e1;max-height:120px;overflow:auto")
                facts = [f"{k}: {d[k]}" for k in
                         ("run_status", "stages_done", "spend_usd", "app_version")
                         if d.get(k) is not None]
                if facts:
                    ui.label(" · ".join(facts)).classes("text-xs font-mono") \
                        .style("color:#94a3b8")

                def block(label: str, items) -> None:
                    if not items:
                        return
                    with ui.expansion(f"{label} ({len(items)})") \
                            .classes("w-full").style("background:var(--kc-bg);"
                                                     "border:1px solid var(--kc-border)"):
                        ui.label("\n".join(str(x) for x in items)) \
                            .classes("text-xs font-mono whitespace-pre-wrap") \
                            .style("color:#94a3b8;max-height:260px;overflow:auto")
                block("Build log tail", d.get("build_log_tail") or [])
                block("ERC errors", d.get("erc_errors") or [])
                block("Failed checks", d.get("failed_checks") or [])
                with ui.row().classes("justify-end w-full"):
                    ui.button("Close", on_click=detail_dialog.close) \
                        .props("flat color=white")
            detail_dialog.open()

        def show_report(inv) -> None:
            report_dialog.clear()
            with report_dialog, ui.card().classes("w-[900px] max-w-[97vw] gap-2") \
                    .style("background:var(--kc-surface);"
                           "border:1px solid var(--kc-border-strong)"):
                head = f"Investigation #{inv.id} · {inv.status}"
                if inv.rc is not None:
                    head += f" · rc={inv.rc}"
                ui.label(head).classes("text-lg font-bold text-white")
                if inv.report_md:
                    with ui.element("div").classes("w-full") \
                            .style("max-height:70vh;overflow:auto"):
                        ui.markdown(inv.report_md)
                else:
                    ui.label("No report captured yet.").classes("text-sm") \
                        .style("color:#94a3b8")
                with ui.row().classes("justify-end w-full"):
                    ui.button("Close", on_click=report_dialog.close) \
                        .props("flat color=white")
            report_dialog.open()

        # ---- mutations --------------------------------------------------------
        def do_investigate(r) -> None:
            if not guard():
                return
            from . import investigate_runner
            log_dir = store.projects_dir.parent / "support_investigations"
            inv_id = investigate_runner.enqueue_investigation(
                store, r, log_dir=log_dir)
            if inv_id is None:
                ui.notify("Nothing locatable to investigate, or one is already "
                          "running for this report.", color="warning")
            else:
                ui.notify(f"Investigation #{inv_id} started.", color="positive")
            render()

        def mark_reviewed(r) -> None:
            if not guard():
                return
            store.set_support_report_status(r.id, "reviewed")
            ui.notify(f"Report #{r.id} marked reviewed.", color="positive")
            render()

        # ---- render -----------------------------------------------------------
        def stat(label: str, value: int) -> None:
            with ui.card().classes("gap-0 items-start").style(
                    "background:var(--kc-surface);border:1px solid var(--kc-border);"
                    "min-width:150px"):
                ui.label(str(value)).classes("text-2xl font-bold") \
                    .style("color:#e2e8f0")
                ui.label(label).classes("text-xs").style("color:#94a3b8")

        def filter_chip(key: str, label: str) -> None:
            active = state["filter"] == key
            btn = ui.button(label, on_click=lambda _e=None, k=key: set_filter(k)) \
                .props("dense no-caps " + ("unelevated" if active else "flat")) \
                .classes("text-xs")
            btn.style("color:#0b1220;background:#60a5fa" if active
                      else "color:#94a3b8")

        def set_filter(key: str) -> None:
            state["filter"] = key
            render()

        def render_row(r, inv) -> None:
            user_reported = _is_user_reported(r)
            base = ("w-full items-center gap-2 text-xs")
            style = "border-top:1px solid var(--kc-border);padding:6px 4px"
            if user_reported:
                style += ";border-left:3px solid #fbbf24;background:rgba(251,191,36,0.06)"
            with ui.row().classes(base).style(style):
                ui.label(str(r.id)).style("width:44px;color:#64748b")
                ui.label(r.created_at[:19]).classes("font-mono") \
                    .style("width:140px;color:#94a3b8")
                ui.label(r.board_code or "—").classes("font-mono") \
                    .style("width:92px;color:#e2e8f0")
                reporter = "—"
                if r.user_id:
                    u = store.get_user(r.user_id)
                    reporter = (u.email if u else f"#{r.user_id}")
                ui.label(reporter).style("width:170px;color:#cbd5e1;"
                                         "overflow:hidden;text-overflow:ellipsis")
                if user_reported:
                    ui.label("user").classes("font-bold").style(
                        "width:56px;color:#fbbf24")
                else:
                    ui.label(r.kind).style("width:56px;color:#64748b")
                ui.label(r.status).style(
                    "width:64px;color:" + ("#4ade80" if r.status == "reviewed"
                                           else "#60a5fa"))
                ui.label((r.message or "").replace("\n", " ")[:70] or "—") \
                    .classes("flex-1").style("color:#94a3b8;overflow:hidden;"
                                             "text-overflow:ellipsis")
                # investigation status + actions
                if inv and inv.status in ("queued", "running"):
                    ui.spinner(size="sm").style("color:#fbbf24")
                    ui.label("investigating…").style("color:#fbbf24")
                else:
                    lbl = "Re-investigate" if inv else "Investigate"
                    ui.button(lbl, icon="biotech",
                              on_click=lambda _e=None, rr=r: do_investigate(rr)) \
                        .props("flat dense no-caps color=white").classes("text-xs")
                if inv and inv.status == "failed":
                    ui.icon("error").style("color:#f87171")
                if inv and inv.report_md:
                    ui.button("Report", icon="description",
                              on_click=lambda _e=None, ii=inv: show_report(ii)) \
                        .props("flat dense no-caps color=white").classes("text-xs")
                ui.button("Details", icon="info",
                          on_click=lambda _e=None, rr=r: show_details(rr)) \
                    .props("flat dense no-caps color=white").classes("text-xs")
                if r.status != "reviewed":
                    ui.button("Reviewed", icon="done",
                              on_click=lambda _e=None, rr=r: mark_reviewed(rr)) \
                        .props("flat dense no-caps color=white").classes("text-xs")

        def render() -> None:
            reports = store.list_support_reports(status=None, limit=300)
            inv_by = store.latest_investigations_by_report()
            cards_box.clear()
            with cards_box:
                stat("Failed boards", len(reports))
                stat("User-reported", sum(1 for r in reports
                                          if _is_user_reported(r)))
                stat("Untriaged", sum(1 for r in reports if r.status == "new"))
                stat("Investigated", sum(1 for r in reports if r.id in inv_by))
            filter_box.clear()
            with filter_box:
                filter_chip("all", "All")
                filter_chip("user", "User-reported")
                filter_chip("new", "Untriaged")
                filter_chip("reviewed", "Reviewed")

            f = state["filter"]

            def keep(r) -> bool:
                if f == "user":
                    return _is_user_reported(r)
                if f == "new":
                    return r.status == "new"
                if f == "reviewed":
                    return r.status == "reviewed"
                return True
            rows = [r for r in reports if keep(r)]
            list_box.clear()
            with list_box:
                if not rows:
                    ui.label("No reports match this filter.") \
                        .classes("text-xs").style("color:#64748b;padding:8px 4px")
                    return
                with ui.row().classes("w-full items-center gap-2 text-xs") \
                        .style("padding:4px 4px;color:#64748b"):
                    ui.label("id").style("width:44px")
                    ui.label("created").style("width:140px")
                    ui.label("board").style("width:92px")
                    ui.label("reporter").style("width:170px")
                    ui.label("kind").style("width:56px")
                    ui.label("status").style("width:64px")
                    ui.label("message").classes("flex-1")
                    ui.label("actions")
                for r in rows:
                    render_row(r, inv_by.get(r.id))

        ui.timer(10.0, render)   # flip running investigations to done live
        render()


# --------------------------------------------------------------------------- #
# Admin: self-eval batch over the curated example briefs.
#
# Drives kicraft.eval.self_eval (the /self-eval harness) as a subprocess writing
# to a fresh out dir, then polls that dir to show live per-brief progress, the
# A-F scorecard (reusing _render_scorecard), and on-demand kicad-cli renders of
# each leaf board so a failed route can be inspected in-page. One batch at a time
# (it shares the spend guard and is heavy); state lives at module scope so every
# admin client + every timer tick sees the same run.
# --------------------------------------------------------------------------- #
_SELF_EVAL: dict = {"proc": None, "out": None, "started_at": None, "args": {}}


def _self_eval_out_root() -> Path:
    """Where the GUI *launches* new batches (and the harness defaults to): a
    ``self_eval/`` sibling of the configured projects dir."""
    base = Path(getattr(Settings.from_env(), "projects_dir",
                        Path.home() / ".kicraft" / "projects"))
    return base.parent / "self_eval"


def _self_eval_out_roots() -> list[Path]:
    """Every root a self-eval batch can live under, so the page lists *all* runs --
    those started from this page as well as those an agent drove from the command
    line:

      * ``<projects_dir>/../self_eval`` -- the GUI/harness default (where Run writes);
      * ``<repo>/logs/self_eval`` -- where the ``/self-eval`` command writes when the
        batch is launched from the CLI.

    Existing dirs only, de-duplicated by resolved path (the two roots coincide when
    the projects dir is the repo)."""
    roots = [_self_eval_out_root(),
             Path(__file__).resolve().parents[2] / "logs" / "self_eval"]
    out: list[Path] = []
    seen: set = set()
    for r in roots:
        try:
            rp = r.resolve()
        except OSError:
            continue
        if rp not in seen and r.is_dir():
            seen.add(rp)
            out.append(r)
    return out


def _self_eval_root_label(out: Path) -> str:
    """A short tag for which root a batch lives under, so the list distinguishes a
    Run-from-here batch from a CLI/agent one."""
    try:
        local = _self_eval_out_root().resolve()
        return "this page" if Path(out).resolve().parent == local else "command line"
    except OSError:
        return ""


def _self_eval_batch_dirs() -> list[Path]:
    """Every adoptable batch dir across all roots, newest first. A dir qualifies if
    it carries persisted launch args, a finished summary, or any ``run_NN_*``
    subdir (so an in-flight or CLI batch is listed too). Capped so a long-lived box
    never builds an unbounded list."""
    cands: list[Path] = []
    for root in _self_eval_out_roots():
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for d in entries:
            if d.is_dir() and (
                    (d / "_args.json").is_file()
                    or (d / "summary.json").is_file()
                    or any(d.glob("run_[0-9][0-9]_*"))):
                cands.append(d)
    cands.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return cands[:50]


def _self_eval_args_for(out: Path) -> dict:
    """The persisted launch args for a batch (``_args.json``), or {} for a CLI batch
    that has none -- {} makes _self_eval_selected derive the brief set from the run
    dirs on disk."""
    ap = Path(out) / "_args.json"
    if ap.is_file():
        try:
            return json.loads(ap.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _self_eval_batch_overview(out: Path) -> dict:
    """Headline stats for one batch dir, for the runs list. Prefers the finished
    ``summary.json``; else derives counts from the per-brief reports so an in-flight
    or CLI batch still shows useful totals (fab-ready needs the summary, so it is
    None until the batch finishes)."""
    out = Path(out)
    info = {"path": str(out), "name": out.name, "label": _self_eval_root_label(out),
            "mtime": out.stat().st_mtime, "n": 0, "scored": 0, "fab_ready": None,
            "mean": None, "grades": {}, "done": (out / "summary.json").is_file()}
    sj = out / "summary.json"
    if sj.is_file():
        try:
            s = json.loads(sj.read_text())
            info.update(n=s.get("n") or 0, scored=s.get("graded_n") or 0,
                        fab_ready=s.get("fab_ready"), mean=s.get("mean_final"),
                        grades=s.get("grade_counts") or {})
            return info
        except (OSError, json.JSONDecodeError):
            pass
    runs = sorted(out.glob("run_[0-9][0-9]_*"))
    info["n"] = len(runs)
    finals: list = []
    for rd in runs:
        rep = rd / "eval" / "report.json"
        if not rep.is_file():
            continue
        try:
            sc = json.loads(rep.read_text()).get("score") or {}
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(sc.get("final"), (int, float)):
            finals.append(sc["final"])
        if sc.get("grade"):
            info["grades"][sc["grade"]] = info["grades"].get(sc["grade"], 0) + 1
    info["scored"] = len(finals)
    info["mean"] = round(sum(finals) / len(finals), 1) if finals else None
    return info


def _self_eval_selected(out, args: dict) -> list:
    """The (index, entry) brief set a batch covers, where entry is a benchmark
    ``{"slug", "archetype", "brief"}`` dict from ``BENCHMARK_PROMPTS``."""
    from kicraft.eval.self_eval import _select
    if "no_judge" in args:  # args from a launch / _args.json are authoritative
        return _select(list(BENCHMARK_PROMPTS), args.get("limit"), args.get("only"))
    # Legacy run (pre _args.json): reconstruct the brief set from the run dirs on disk
    # (``run_<NN>_<slug>``), matching each dir's slug back to its benchmark entry.
    by_slug = {e["slug"]: e for e in BENCHMARK_PROMPTS}
    found = []
    for p in sorted(Path(out).glob("run_[0-9][0-9]_*")):
        parts = p.name.split("_", 2)
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        entry = by_slug.get(parts[2])
        if entry:
            found.append((int(parts[1]), entry))
    return found or _select(list(BENCHMARK_PROMPTS), None, None)


def _self_eval_running() -> bool:
    p = _SELF_EVAL.get("proc")
    return bool(p is not None and p.poll() is None)


def _self_eval_launch(limit, only, no_judge) -> str:
    """Start the batch harness as a subprocess; return the out dir ('' if busy)."""
    if _self_eval_running():
        return ""
    import datetime as _dt
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = _self_eval_out_root() / ts
    out.mkdir(parents=True, exist_ok=True)
    # Persist the run args so the page can re-adopt this batch after a server
    # restart (the harness keeps running as a detached subprocess).
    (out / "_args.json").write_text(
        json.dumps({"limit": limit, "only": only, "no_judge": no_judge}))
    cmd = [KICRAFT[0], "-m", "kicraft.eval.self_eval", "--out", str(out)]
    if limit:
        cmd += ["--limit", str(int(limit))]
    if only:
        cmd += ["--only", str(only)]
    if no_judge:
        cmd += ["--no-judge"]
    logf = (out / "run.log").open("w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env={**os.environ, "KICRAFT_CALLER": "web"},
                            cwd=str(Path(__file__).resolve().parents[2]))
    _SELF_EVAL.update(proc=proc, out=out, started_at=time.time(),
                      args={"limit": limit, "only": only, "no_judge": no_judge})
    return str(out)


def _self_eval_adopt_latest() -> None:
    """When no run is tracked in this process (e.g. after a server restart), adopt
    the most recent batch on disk so its progress + artifacts stay viewable. Never
    overrides a run we launched this process (proc still alive)."""
    if _self_eval_running():
        return
    cur = _SELF_EVAL.get("out")
    if cur and Path(cur).is_dir():
        return  # already pointing at a real run; keep it
    cands = _self_eval_batch_dirs()  # newest first, across every root
    if not cands:
        return
    latest = cands[0]
    _SELF_EVAL.update(proc=None, out=latest, args=_self_eval_args_for(latest))


_REVIEW_NONBLOCK_RE = re.compile(r"electrical review: (\d+) non-blocking")


def _build_review_outcome(lines) -> dict | None:
    """The Layer-4 electrical-review verdict parsed from a run's build_log events.

    A review block returns rc 7 -- the SAME build_done rc as a real DRC/route
    failure -- so the only durable signal that an otherwise structurally-sound
    board was rejected for an ELECTRICAL defect (vs failing to route) is the
    build_log markers the gate writes. Returns
    ``{status: 'blocked'|'passed'|'skipped', n, blockers[]}`` or None if the
    review never ran (e.g. the build never reached the verify gate).
    """
    status = None
    blockers: list[str] = []
    n = 0
    for line in lines:
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("kind") != "build_log":
            continue
        t = e.get("text", "")
        if "electrical review found a blocker" in t:
            status = "blocked"
        elif "electrical review skipped" in t and status is None:
            status = "skipped"
        if "review BLOCKER:" in t:
            blockers.append(t.split("review BLOCKER:", 1)[1].strip())
        m = _REVIEW_NONBLOCK_RE.search(t)
        if m:
            n = int(m.group(1))
            if status is None:
                status = "passed"
    if status is None:
        return None
    return {"status": status, "n": (len(blockers) if status == "blocked" else n),
            "blockers": blockers}


def _self_eval_brief_status(out: Path, idx: int, entry: dict) -> dict:
    """Parse one brief's live status from its run dir under `out`. ``entry`` is a
    benchmark ``{"slug", "archetype", "brief"}`` dict."""
    hits = sorted(out.glob(f"run_{idx:02d}_*"))
    rd = hits[0] if hits else None
    base = {"index": idx, "slug": entry["slug"], "archetype": entry["archetype"],
            "prompt": entry["brief"], "rundir": (str(rd) if rd else None)}
    if rd is None:
        return {**base, "status": "pending"}
    stage = build_label = None
    review = None
    ev = rd / "events.jsonl"
    if ev.is_file():
        lines = ev.read_text(errors="replace").splitlines()
        for line in lines:
            try:
                e = json.loads(line)
            except Exception:
                continue
            k = e.get("kind")
            if k == "stage_start":
                stage = e.get("stage")
            elif k == "stage_done" and e.get("ok"):
                stage = (e.get("stage") or "") + " ✓"
            elif k == "build_start":
                build_label = "building…"
            elif k == "build_done":
                build_label = "fab-ready" if e.get("ok") else f"build rc={e.get('rc')}"
        review = _build_review_outcome(lines)
        # A review block returns rc 7 -- the same label as a DRC/route fail.
        # Relabel it so the dashboard distinguishes an electrically-flagged board
        # (structurally clean, the gate doing its job) from one that failed to route.
        if review and review["status"] == "blocked" and build_label \
                and build_label.startswith("build rc="):
            build_label = f"review-blocked ({review['n']})"
    rep = rd / "eval" / "report.json"
    if rep.is_file():
        try:
            r = json.loads(rep.read_text())
        except Exception:
            r = None
        if r:
            sc = r.get("score") or {}
            gates = [g.get("id") for g in (r.get("gates") or {}).get("triggered", [])]
            return {**base, "status": "done", "grade": sc.get("grade"),
                    "final": sc.get("final"), "verdict": sc.get("verdict"),
                    "gates": gates, "build": build_label, "review": review}
    return {**base, "status": "running", "stage": stage, "build": build_label,
            "review": review}


def _self_eval_leaf_boards(gen_dir: Path) -> list:
    """Interactive-viewer descriptors for each per-leaf routed board: a per-leaf
    signed token (each leaf lives in a nested ``.experiments/subcircuits/<uuid>/``
    dir, which the flat ``/project/<token>/<file>`` route can't reach under the gen
    token) + the leaf's accept state. KiCanvas renders ``leaf_routed.kicad_pcb``
    directly, so a failed route is zoomable in-page even after the live ``renders/``
    previews have been cleaned up on a finished batch."""
    out = []
    # Layout + filename come from artifact_paths (docs/ARTIFACTS.md: never
    # hand-code these literals), so a canonical-name move can't silently make
    # this viewer render zero leaves.
    for leaf in sorted(artifact_root(gen_dir).glob(f"*/{LEAF_ROUTED}")):
        leafdir = leaf.parent
        tok = _register_project_dir(leafdir)
        out.append({
            # "accepted" == produced a solved_layout.json (routed cleanly enough to be
            # composed into the parent); a ✗ leaf is where a bad route lives.
            "label": f"leaf {leafdir.name.split('__')[0][:8]}",
            "accepted": (leafdir / "solved_layout.json").is_file(),
            "url": f"/project/{tok}/{leaf.name}",
            "filename": leaf.name,
        })
    return out


@ui.page("/admin/self-eval")
def admin_self_eval_page():
    """Admin: start a self-eval batch over the example briefs, browse *every* batch
    (those launched here and those an agent drove from the command line), watch live
    per-brief progress + grades, and open any run's schematic + boards."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect
    ui.dark_mode().enable()
    _admin_header("self-eval")

    n_avail = len(BENCHMARK_PROMPTS)
    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1300px"):
        ui.label("Self-evaluation").classes("text-2xl font-bold text-white")
        ui.label("Drive every curated example brief end to end (auto-answering "
                 "clarifying questions with the model's suggested option) and grade "
                 "each with the kicraft.eval rubric. Runs in the background and "
                 "SPENDS real money via the capped client; the spend guard still caps "
                 "the day.").classes("text-sm").style("color:#94a3b8")
        with ui.row().classes("items-end gap-3 flex-wrap"):
            limit_in = ui.number("Limit (first N)", value=1, min=0, max=n_avail,
                                 format="%d").props("dense outlined dark").classes("w-40")
            only_in = ui.input("Only (slugs, e.g. usb-pd-trigger,buck-3a)") \
                .props("dense outlined dark").classes("w-44")
            judge_sw = ui.switch("LLM judge (A–F)", value=True)
            run_btn = ui.button("Run self-eval", icon="play_arrow").props("color=primary")
            ui.label(f"{n_avail} briefs available").classes("text-xs") \
                .style("color:#64748b")

        # Every batch on disk, across both roots (this page's and the CLI's), so a
        # run an agent launched from the command line is listed here too. Click one
        # to drive the per-brief table below it.
        ui.label("All runs").classes("text-sm font-bold mt-2").style("color:#cbd5e1")
        runs_box = ui.column().classes("w-full gap-0")

        ui.separator().style("background:var(--kc-border);margin-top:6px")
        head = ui.row().classes("items-center gap-4 text-sm font-mono") \
            .style("color:#cbd5e1")
        table = ui.column().classes("w-full gap-0")

    # Per-client element refs, built ONCE (not every timer tick). Rebuilding the
    # table each second would replace the 'view' buttons mid-click, so a click
    # would land on a deleted element and do nothing. Build rows once; update
    # their text in place; only rebuild when the selected set changes.
    rows: dict = {}
    latest: dict = {}
    sig = {"sel": None}
    runs_sig = {"key": None}
    # Which batch the brief table shows. Follows the live/most-recent run until the
    # user clicks a row to pin one (so a finished CLI batch stays put for review).
    view = {"dir": None, "pinned": False}
    _SE_COLORS = {"done": "#4ade80", "running": "#fbbf24",
                  "pending": "#64748b", "error": "#f87171"}

    def select_batch(path: str):
        view["dir"] = path
        view["pinned"] = True
        sig["sel"] = None       # force the brief table to rebuild for the new batch
        runs_sig["key"] = None  # re-highlight the selected row

    def open_run(idx):
        s = latest.get(idx)
        if s and s.get("rundir"):
            tok = _register_project_dir(Path(s["rundir"]))
            ui.navigate.to(f"/admin/self-eval/run?run={quote(tok)}")

    def build_runs(batches):
        runs_box.clear()
        with runs_box:
            if not batches:
                ui.label("No self-eval runs yet — configure above and press Run.") \
                    .classes("text-xs").style("color:#64748b")
                return
            for b in batches:
                selected = (b["path"] == view["dir"])
                bg = "#13233f" if selected else "transparent"
                row = ui.row().classes("w-full items-center gap-3 text-xs cursor-pointer") \
                    .style(f"border-top:1px solid var(--kc-border);padding:4px 6px;background:{bg}")
                row.on("click", lambda _e=None, p=b["path"]: select_batch(p))
                with row:
                    ui.icon("check_circle" if b["done"] else "play_circle") \
                        .style("color:%s" % ("#4ade80" if b["done"] else "#fbbf24"))
                    ui.label(b["name"]).classes("font-mono") \
                        .style("width:188px;color:#e2e8f0")
                    ui.label(b["label"]).style("width:104px;color:#94a3b8")
                    ui.label(f"{b['n']} briefs").style("width:74px;color:#cbd5e1")
                    fr = "—" if b["fab_ready"] is None else f"{b['fab_ready']}/{b['n']}"
                    ui.label(f"fab {fr}").style("width:84px;color:#cbd5e1")
                    ui.label("mean —" if b["mean"] is None else f"mean {b['mean']}") \
                        .style("width:84px;color:#cbd5e1")
                    grds = "  ".join(f"{g}:{n}" for g, n in sorted(b["grades"].items()))
                    ui.label(grds).classes("flex-1 truncate").style("color:#64748b")

    def build_rows(selected):
        table.clear()
        rows.clear()
        with table:
            with ui.row().classes("w-full items-center gap-2 text-xs font-bold") \
                    .style("color:#64748b;padding-bottom:2px"):
                ui.label("#").style("width:24px")
                ui.label("status").style("width:108px")
                ui.label("grade").style("width:60px")
                ui.label("build").style("width:118px")
                ui.label("archetype").style("width:140px")
                ui.label("brief").classes("flex-1")
                ui.label("").style("width:56px")
            for idx, entry in selected:
                with ui.row().classes("w-full items-center gap-2 text-xs") \
                        .style("border-top:1px solid var(--kc-border);padding:3px 0"):
                    ui.label(str(idx)).style("width:24px;color:#cbd5e1")
                    st_l = ui.label("pending").style("width:108px;color:#64748b")
                    gr_l = ui.label("").style("width:60px;color:#e2e8f0")
                    bd_l = ui.label("").style("width:118px;color:#94a3b8")
                    ui.label(entry["archetype"]).style("width:140px;color:#64748b")
                    ui.label(f"{entry['slug']} — {entry['brief']}"[:96]) \
                        .classes("flex-1").style("color:#cbd5e1")
                    btn = ui.button("view", on_click=lambda _e=None, i=idx: open_run(i)) \
                        .props("flat dense no-caps color=primary").classes("text-xs") \
                        .style("width:56px")
                    btn.set_visibility(False)
                    rows[idx] = {"status": st_l, "grade": gr_l, "build": bd_l, "btn": btn}

    def start():
        if _self_eval_running():
            ui.notify("A self-eval batch is already running.", color="warning")
            return
        limit = int(limit_in.value) if limit_in.value else None
        only = (only_in.value or "").strip() or None
        out = _self_eval_launch(limit, only, not judge_sw.value)
        if out:
            view["pinned"] = False   # follow the run we just launched
            runs_sig["key"] = None   # rebuild the list so it appears immediately
        ui.notify(f"Started → {out}" if out else "Could not start.",
                  color=("positive" if out else "warning"))
    run_btn.on_click(start)

    def render():
        _self_eval_adopt_latest()
        running = _self_eval_running()
        run_btn.set_enabled(not running)
        live_out = _SELF_EVAL.get("out")
        batches = [_self_eval_batch_overview(d) for d in _self_eval_batch_dirs()]

        # Default selection follows the live/most-recent run until the user pins one;
        # a pinned dir that was deleted falls back to the default.
        default = str(live_out) if live_out else (batches[0]["path"] if batches else None)
        if not view["pinned"] or (view["dir"] and not Path(view["dir"]).is_dir()):
            view["pinned"] = False
            view["dir"] = default

        rkey = tuple((b["path"], b["done"], b["n"], b["scored"],
                      b["path"] == view["dir"]) for b in batches)
        if rkey != runs_sig["key"]:
            build_runs(batches)
            runs_sig["key"] = rkey

        head.clear()
        if not view["dir"]:
            with head:
                ui.label("No run yet — configure above and press Run.") \
                    .style("color:#64748b")
            if sig["sel"] is not None:
                table.clear()
                rows.clear()
                sig["sel"] = None
            return
        out = Path(view["dir"])
        args = _self_eval_args_for(out)
        selected = _self_eval_selected(out, args)
        sel_key = (str(out),) + tuple(i for i, _ in selected)
        if sel_key != sig["sel"]:
            build_rows(selected)
            sig["sel"] = sel_key
        statuses = [_self_eval_brief_status(out, idx, entry) for idx, entry in selected]
        done = [s for s in statuses if s["status"] == "done"]
        graded = [s for s in done if isinstance(s.get("final"), (int, float))]
        fab = sum(1 for s in done if s.get("build") == "fab-ready")
        rblk = sum(1 for s in statuses if (s.get("review") or {}).get("status") == "blocked")
        summary_done = (out / "summary.json").is_file()
        is_live = bool(live_out and str(live_out) == str(out) and running)
        if is_live:
            state, col = "RUNNING ", "#fbbf24"
        elif summary_done:
            state, col = "DONE ", "#4ade80"
        else:
            state, col = "PARTIAL ", "#94a3b8"
        with head:
            ui.label(state + out.name).style(f"color:{col}")
            ui.label(f"{len(done)}/{len(selected)} scored")
            if graded:
                ui.label(f"mean {round(sum(s['final'] for s in graded) / len(graded), 1)}")
            ui.label(f"fab-ready {fab}/{len(selected)}")
            if rblk:
                # Structurally-clean boards the Layer-4 electrical review rejected
                # -- the gate working, NOT routing failures (which are 'build rc=').
                ui.label(f"⚡ review-blocked {rblk}").style("color:#f59e0b") \
                    .tooltip("structurally clean but the electrical review found a "
                             "blocker (counted separately from route/DRC failures)")
            ui.label(f"judge {'off' if args.get('no_judge') else 'on'}")
        for s in statuses:
            idx = s["index"]
            latest[idx] = s
            r = rows.get(idx)
            if not r:
                continue
            st = s["status"]
            r["status"].set_text((s.get("stage") or st) if st == "running" else st)
            r["status"].style("color:" + _SE_COLORS.get(st, "#94a3b8"))
            g = s.get("grade")
            r["grade"].set_text(f"{g} {s.get('final')}" if g
                                else ("—" if st == "done" else ""))
            bl = s.get("build") or ""
            r["build"].set_text(bl)
            if bl == "fab-ready":
                bcol = "#4ade80"                       # green: shipped
            elif bl.startswith("review-blocked"):
                bcol = "#f59e0b"                       # amber: electrically flagged
            elif bl.startswith("build rc=") or bl == "building…":
                bcol = "#f87171" if bl.startswith("build rc=") else "#94a3b8"
            else:
                bcol = "#94a3b8"
            r["build"].style("color:" + bcol)
            rv = s.get("review")
            if rv and rv.get("blockers"):
                r["build"].tooltip("electrical review blocker(s): "
                                   + " | ".join(rv["blockers"])[:400])
            r["btn"].set_visibility(bool(s.get("rundir")))
    ui.timer(1.0, render)


@ui.page("/admin/self-eval/run")
def admin_self_eval_run_page(run: str = ""):
    """Admin: one self-eval brief's scorecard, its interactive schematic (KiCanvas),
    the composed parent board, and each per-leaf routed board -- so a failed route or
    a bad schematic is inspectable in-page. ``run`` is a signed project token for the
    brief's run dir, minted by the runs table."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect
    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    kicanvas_head()
    _admin_header("self-eval")

    run_dir = _resolve_project_token(run) if run else None
    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1300px"):
        ui.button("← All runs", icon="arrow_back",
                  on_click=lambda: ui.navigate.to("/admin/self-eval")) \
            .props("flat dense no-caps color=white").classes("text-xs")
        if run_dir is None or not run_dir.is_dir():
            ui.label("Run not found (the link may be stale or the run was deleted).") \
                .classes("text-sm").style("color:#f87171")
            return

        brief = ""
        bf = run_dir / "brief.txt"
        if bf.is_file():
            try:
                brief = bf.read_text(errors="replace").strip()
            except OSError:
                brief = ""
        ui.label(brief[:160] or run_dir.name).classes("text-lg font-bold") \
            .style("color:#e2e8f0")

        rep = run_dir / "eval" / "report.json"
        if rep.is_file():
            try:
                _render_scorecard(ui.column().classes("w-full gap-1"),
                                  json.loads(rep.read_text()))
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                ui.label("(could not load report.json)").classes("text-xs") \
                    .style("color:#f87171")
        else:
            ui.label("Not scored yet.").classes("text-sm").style("color:#94a3b8")

        # Layer-4 electrical-review findings, distinct from the DRC/route outcome:
        # a blocked board is structurally sound but electrically flagged.
        _ev = run_dir / "events.jsonl"
        if _ev.is_file():
            findings = []
            for ln in _ev.read_text(errors="replace").splitlines():
                try:
                    e = json.loads(ln)
                except Exception:
                    continue
                if e.get("kind") != "build_log":
                    continue
                t = e.get("text", "")
                for sev in ("BLOCKER", "WARNING", "NOTE"):
                    mark = f"review {sev}:"
                    if mark in t:
                        findings.append((sev, t.split(mark, 1)[1].strip()))
            if findings:
                blocked = any(s == "BLOCKER" for s, _ in findings)
                sevcol = {"BLOCKER": "#f87171", "WARNING": "#f59e0b", "NOTE": "#94a3b8"}
                with ui.card().classes("w-full mt-2").style("background:#111827"):
                    ui.label("⚡ Electrical review — "
                             + ("BLOCKED (kept for inspection, no fab package)"
                                if blocked else "passed (non-blocking findings)")) \
                        .classes("text-sm font-bold") \
                        .style("color:" + ("#f87171" if blocked else "#4ade80"))
                    for sev, txt in findings:
                        ui.label(f"[{sev}] {txt}").classes("text-xs") \
                            .style("color:" + sevcol[sev])

        gen = _discover_generated_dir(run_dir)
        if gen is None:
            ui.label("No synthesized project — the design stages did not produce "
                     "schematic sheets for this brief.").classes("text-sm mt-2") \
                .style("color:#94a3b8")
            ui.label(f"artifacts: {run_dir}").classes("text-xs font-mono mt-2") \
                .style("color:#64748b")
            return
        token = _register_project_dir(gen)
        stem = gen.name

        # Schematic (interactive). Kept on-screen (not in a tab/dialog) because a
        # KiCanvas WebGL canvas built in a hidden / zero-size container never repaints.
        with ui.card().classes("w-full") \
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
            srcs = _schematic_sources(gen, stem, token)
            if srcs:
                _render_synth_view(srcs, stem, gen)
            else:
                ui.label("No schematic sheets found.").classes("text-xs") \
                    .style("color:#94a3b8")

        # Parent board (interactive): the composed <stem>.kicad_pcb. Absent if the
        # build failed before composing a parent.
        parent_pcb = gen / f"{stem}.kicad_pcb"
        with ui.card().classes("w-full") \
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
            ui.label("Parent board").classes("text-xs font-medium").style("color:#94a3b8")
            if parent_pcb.is_file():
                KiCanvasView([KiCanvasSource(f"/project/{token}/{parent_pcb.name}",
                                             parent_pcb.name)], height="h-[520px]")
            else:
                ui.label("No composed parent board (the build did not reach parent "
                         "routing).").classes("text-xs").style("color:#94a3b8")

        # Per-leaf routed boards, each interactive (KiCanvas) so the actual routing is
        # zoomable -- not a flat thumbnail. Every leaf lives in its own nested
        # .experiments/subcircuits/<uuid>/ dir, so it carries its own signed token
        # (the flat /project/<token>/<file> route only serves files sitting directly
        # under the token dir). A ✗ leaf is where a rejected route lives.
        leaves = _self_eval_leaf_boards(gen)
        ui.label("Leaf boards — a ✗ leaf is where a rejected route lives") \
            .classes("text-sm font-bold mt-2").style("color:#cbd5e1")
        if not leaves:
            ui.label("No per-leaf routed boards (single-leaf design, or the leaves "
                     "were composed into the parent).").classes("text-xs") \
                .style("color:#94a3b8")
        for b in leaves[:8]:
            with ui.card().classes("w-full") \
                    .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                lab, col = b["label"], "#cbd5e1"
                if b["accepted"]:
                    lab += "  ✓ accepted"; col = "#4ade80"
                else:
                    lab += "  ✗ rejected"; col = "#f87171"
                ui.label(lab).classes("text-xs font-mono").style(f"color:{col}")
                KiCanvasView([KiCanvasSource(b["url"], b["filename"])],
                             height="h-[460px]")
        if len(leaves) > 8:
            ui.label(f"(+{len(leaves) - 8} more leaves not shown)") \
                .classes("text-xs").style("color:#64748b")

        # Bill of materials -- the same read-only table the project viewer renders,
        # built from the run's committed state.json so an eval run is inspectable to
        # a real board's depth. Lazy import: web.py imports THIS module at its bottom,
        # so a module-level `from .web import ...` would be circular.
        try:
            from .web import _load_persisted_state, _render_bom_table
            bom_state = _load_persisted_state(run_dir)
            if bom_state is not None:
                _render_bom_table(bom_state)
        except Exception:  # noqa: BLE001 - BOM is best-effort; never break the page
            pass

        # Thinking stream -- the model's per-stage reasoning, replayed read-only from
        # events.jsonl (the self-eval full-fidelity sink keeps reasoning/answer
        # deltas). Mirrors a live run's Thinking panes: reasoning per stage, falling
        # back to the answer channel for stages that emit no reasoning. A stage that
        # retried appears more than once, in order -- the real trace, warts and all.
        ev_path = run_dir / "events.jsonl"
        if ev_path.is_file():
            run_stages: list[dict] = []
            cur_stage = None
            for ln in ev_path.read_text(errors="replace").splitlines():
                try:
                    e = json.loads(ln)
                except Exception:  # noqa: BLE001 - skip a torn/partial line
                    continue
                k = e.get("kind")
                if k == "stage_start":
                    cur_stage = {"stage": e.get("stage") or "stage",
                                 "reason": [], "answer": []}
                    run_stages.append(cur_stage)
                elif cur_stage is not None and k == "reasoning_delta":
                    cur_stage["reason"].append(e.get("text", ""))
                elif cur_stage is not None and k == "answer_delta":
                    cur_stage["answer"].append(e.get("text", ""))
            blocks = []
            for s in run_stages:
                txt = "".join(s["reason"]).strip() or "".join(s["answer"]).strip()
                if txt:
                    blocks.append((s["stage"], txt))
            if blocks:
                with ui.card().classes("w-full") \
                        .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                    ui.label("Thinking stream — the model's reasoning per stage") \
                        .classes("text-xs font-medium").style("color:#94a3b8")
                    for i, (stg, txt) in enumerate(blocks):
                        with ui.expansion(stg, icon="psychology",
                                          value=(i == 0)).classes("w-full") \
                                .props("dense"):
                            ui.label(txt).classes("text-xs font-mono w-full").style(
                                "white-space:pre-wrap;color:#cbd5e1;display:block;"
                                "max-height:360px;overflow:auto")

        ui.label(f"artifacts: {run_dir}").classes("text-xs font-mono mt-2") \
            .style("color:#64748b")
        ui.label(f"deep DRC inspection: kicraft inspect-parent {gen}") \
            .classes("text-xs font-mono").style("color:#64748b")


# --------------------------------------------------------------------------- #
# Admin: load / stress testing (kicraft.loadtest). Launch a build-storm or
# full-pipeline scenario as a detached subprocess (so it survives a web restart),
# then chart the live LoadResultStore. Scenarios are $0 (replay / mock LLM); the
# ABORT button writes the harness's abort file and terminates the subprocess.
# --------------------------------------------------------------------------- #
_LOADTEST: dict = {"proc": None, "scenario": None, "started_at": None,
                   "abort_file": None, "log": None}


def _loadtest_running() -> bool:
    p = _LOADTEST.get("proc")
    return bool(p is not None and p.poll() is None)


def _loadtest_launch(scenario, *, n, slots, parallel, build_slots, route, do_build) -> str:
    if _loadtest_running():
        return ""
    import datetime as _dt

    from kicraft.loadtest.store import default_store_path
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = default_store_path().parent
    base.mkdir(parents=True, exist_ok=True)
    log = base / f"run-{ts}.log"
    abort_file = base / f"abort-{ts}"
    cmd = [KICRAFT[0], "-m", "kicraft.loadtest"]
    if scenario == "build-storm":
        cmd += ["build-storm", "--n", str(int(n)), "--slots", str(int(slots)),
                "--abort-file", str(abort_file)]
        if route:
            cmd += ["--route"]
    else:
        cmd += ["pipeline", "--n", str(int(n)), "--parallel", str(int(parallel)),
                "--build-slots", str(int(build_slots))]
        if not do_build:
            cmd += ["--no-build"]
    logf = log.open("w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env={**os.environ, "KICRAFT_CALLER": "web"},
                            cwd=str(Path(__file__).resolve().parents[2]))
    _LOADTEST.update(proc=proc, scenario=scenario, started_at=time.time(),
                     abort_file=abort_file, log=log)
    return str(log)


def _loadtest_abort() -> None:
    """Graceful (build-storm honors the abort file) + hard (terminate the proc)."""
    af = _LOADTEST.get("abort_file")
    if af:
        try:
            Path(af).write_text("abort")
        except OSError:
            pass
    p = _LOADTEST.get("proc")
    if p is not None and p.poll() is None:
        p.terminate()


@ui.page("/admin/loadtest")
def admin_loadtest_page():
    """Admin: launch + watch load/stress scenarios; live host/queue/latency charts."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect
    ui.dark_mode().enable()
    _admin_header("load")
    from kicraft.loadtest import charts as lc
    from kicraft.loadtest.store import LoadResultStore, default_store_path

    cfg = {"scenario": "build-storm", "n": 8, "slots": 2, "parallel": 3,
           "build_slots": 2, "route": False, "do_build": True}

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1300px"):
        ui.label("Load / stress testing").classes("text-2xl font-bold text-white")
        ui.label("Build-storm (replay, $0) and full-pipeline (mock LLM, $0) scenarios. "
                 "Find the build-slot saturation knee and what breaks first.") \
            .classes("text-sm").style("color:#94a3b8")
        ui.separator().style("background:var(--kc-border)")

        with ui.row().classes("items-end gap-3 w-full"):
            scen = ui.select(["build-storm", "pipeline"], value=cfg["scenario"],
                             label="scenario").props("dark dense").style("width:150px")
            n_in = ui.number("designs (n)", value=cfg["n"], min=1, max=200,
                             format="%d").props("dark dense").style("width:120px")
            slots_in = ui.number("build slots", value=cfg["slots"], min=1, max=16,
                                 format="%d").props("dark dense").style("width:110px")
            par_in = ui.number("parallel", value=cfg["parallel"], min=1, max=32,
                               format="%d").props("dark dense").style("width:100px")
            route_cb = ui.checkbox("route (heavy)", value=cfg["route"]).props("dark")
            build_cb = ui.checkbox("build", value=cfg["do_build"]).props("dark")
            launch_btn = ui.button("Launch", icon="play_arrow")
            ui.button("ABORT", icon="stop", color="red") \
                .props("outline").on("click", lambda: (_loadtest_abort(), _refresh()))
        status = ui.label("").classes("text-xs font-mono").style("color:#94a3b8")

        def _launch():
            log = _loadtest_launch(
                scen.value, n=n_in.value, slots=slots_in.value, parallel=par_in.value,
                build_slots=slots_in.value, route=route_cb.value, do_build=build_cb.value)
            ui.notify("launched" if log else "a load run is already in progress",
                      type="positive" if log else "warning")
        launch_btn.on("click", lambda: (_launch(), _refresh()))

        charts_box = ui.column().classes("w-full gap-3")
        runs_box = ui.column().classes("w-full gap-0")

        def _store_ro() -> LoadResultStore:
            return LoadResultStore(default_store_path())

        def _refresh():
            running = _loadtest_running()
            status.text = (f"running: {_LOADTEST.get('scenario')} "
                           f"(log {_LOADTEST.get('log')})" if running
                           else "idle — launch a scenario above")
            store = _store_ro()
            runs = store.list_runs(limit=40)
            # live charts for the newest run
            charts_box.clear()
            if runs:
                rid = runs[0]["run_id"]
                samples = store.samples_for(rid)
                summary = runs[0]["summary"] or {}
                with charts_box:
                    ui.label(f"latest: {rid}").classes("text-sm font-mono") \
                        .style("color:#e2e8f0")
                    with ui.row().classes("w-full gap-3"):
                        ui.echart(lc.host_chart(samples)).classes("flex-1") \
                            .style("height:260px;min-width:380px")
                        ui.echart(lc.queue_chart(samples)).classes("flex-1") \
                            .style("height:260px;min-width:380px")
                    with ui.row().classes("w-full gap-3"):
                        ui.echart(lc.latency_bar(summary)).classes("flex-1") \
                            .style("height:240px;min-width:300px")
                        ui.echart(lc.outcome_pie(summary)).classes("flex-1") \
                            .style("height:240px;min-width:300px")
                        ui.echart(lc.disk_chart(samples)).classes("flex-1") \
                            .style("height:240px;min-width:300px")
            runs_box.clear()
            with runs_box:
                ui.label("Recent runs").classes("text-sm font-bold text-white")
                with ui.row().classes("w-full items-center gap-3 text-xs").style(
                        "padding:4px 6px;color:#64748b"):
                    ui.label("run").style("width:280px")
                    ui.label("scenario").style("width:110px")
                    ui.label("n").style("width:50px")
                    ui.label("ok").style("width:50px")
                    ui.label("max run").style("width:80px")
                    ui.label("wall s").style("width:80px")
                for r in runs:
                    s = r["summary"] or {}
                    with ui.row().classes("w-full items-center gap-3 text-xs").style(
                            "border-top:1px solid var(--kc-border);padding:5px 6px"):
                        ui.label(r["run_id"]).classes("font-mono").style(
                            "width:280px;color:#e2e8f0")
                        ui.label(r["scenario"] or "").style("width:110px;color:#cbd5e1")
                        ui.label(str(s.get("n", "—"))).style("width:50px;color:#cbd5e1")
                        ui.label(str(s.get("ok", s.get("design_ok", "—")))).style(
                            "width:50px;color:#cbd5e1")
                        ui.label(str(s.get("max_running", "—"))).style(
                            "width:80px;color:#cbd5e1")
                        ui.label(str(s.get("wall_total_s", "—"))).style(
                            "width:80px;color:#cbd5e1")

        ui.timer(2.0, _refresh)
        _refresh()


# --------------------------------------------------------------------------- #
# Admin: security scans (kicraft.security). Launch bandit/pip-audit/gitleaks into
# the SecurityResultStore and triage findings (acknowledge persists).
# --------------------------------------------------------------------------- #
_SECURITY: dict = {"proc": None, "started_at": None, "log": None}


def _security_running() -> bool:
    p = _SECURITY.get("proc")
    return bool(p is not None and p.poll() is None)


def _security_launch() -> bool:
    if _security_running():
        return False
    from kicraft.security.store import default_store_path
    base = default_store_path().parent
    base.mkdir(parents=True, exist_ok=True)
    log = base / "scan.log"
    proc = subprocess.Popen(
        [KICRAFT[0], "-m", "kicraft.security.scans"],
        stdout=log.open("w"), stderr=subprocess.STDOUT,
        env={**os.environ, "KICRAFT_CALLER": "web"},
        cwd=str(Path(__file__).resolve().parents[2]))
    _SECURITY.update(proc=proc, started_at=time.time(), log=log)
    return True


@ui.page("/admin/security")
def admin_security_page():
    """Admin: run static security scans + triage findings (open/acknowledged)."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect
    ui.dark_mode().enable()
    _admin_header("security")
    from kicraft.security import charts as sc
    from kicraft.security.store import SecurityResultStore

    store = SecurityResultStore()

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1300px"):
        ui.label("Security scans").classes("text-2xl font-bold text-white")
        ui.label("Static analysis (bandit), dependency CVEs (pip-audit), and secret "
                 "scanning (gitleaks). Acknowledge a triaged finding to hide it from "
                 "the open list (it survives a re-scan).").classes("text-sm") \
            .style("color:#94a3b8")
        ui.separator().style("background:var(--kc-border)")

        with ui.row().classes("items-center gap-3"):
            run_btn = ui.button("Run scans", icon="play_arrow")
            ftab = ui.toggle(["open", "acknowledged", "all"], value="open").props("dark")
            status = ui.label("").classes("text-xs font-mono").style("color:#94a3b8")

        run_btn.on("click", lambda: (
            ui.notify("scanning…" if _security_launch() else "a scan is already running",
                      type="positive" if not _security_running() else "info"),
            _refresh()))

        charts_box = ui.row().classes("w-full gap-3")
        table_box = ui.column().classes("w-full gap-0")

        def _refresh():
            status.text = ("scanning…" if _security_running()
                           else f"{sum(store.severity_counts().values())} open finding(s)")
            charts_box.clear()
            with charts_box:
                ui.echart(sc.severity_bar(store.severity_counts())).classes("flex-1") \
                    .style("height:240px;min-width:380px")
                ui.echart(sc.status_pie(store.status_counts())).classes("flex-1") \
                    .style("height:240px;min-width:380px")
            want = ftab.value
            status_filter = None if want == "all" else want
            findings = store.list_findings(status=status_filter)
            table_box.clear()
            with table_box:
                with ui.row().classes("w-full items-center gap-2 text-xs").style(
                        "padding:4px 6px;color:#64748b"):
                    ui.label("sev").style("width:70px")
                    ui.label("tool").style("width:80px")
                    ui.label("rule").style("width:90px")
                    ui.label("location").style("width:260px")
                    ui.label("message").style("flex:1")
                    ui.label("").style("width:110px")
                for f in findings[:400]:
                    col = {"critical": "#ef4444", "high": "#f97316",
                           "medium": "#f59e0b", "low": "#60a5fa"}.get(
                        f["severity"], "#94a3b8")
                    with ui.row().classes("w-full items-center gap-2 text-xs").style(
                            "border-top:1px solid var(--kc-border);padding:4px 6px"):
                        ui.label(f["severity"]).style(f"width:70px;color:{col}")
                        ui.label(f["tool"]).style("width:80px;color:#cbd5e1")
                        ui.label(f["rule"]).classes("font-mono").style(
                            "width:90px;color:#cbd5e1")
                        ui.label(f["location"]).classes("font-mono").style(
                            "width:260px;color:#94a3b8")
                        ui.label(f["message"]).style("flex:1;color:#cbd5e1")
                        if f["status"] == "open":
                            ui.button("ack", on_click=lambda _e=None, fid=f["id"]:
                                      (store.set_status(fid, "acknowledged"), _refresh())) \
                                .props("flat dense no-caps").classes("text-xs") \
                                .style("width:110px")
                        else:
                            ui.button("reopen", on_click=lambda _e=None, fid=f["id"]:
                                      (store.set_status(fid, "open"), _refresh())) \
                                .props("flat dense no-caps").classes("text-xs") \
                                .style("width:110px")

        ftab.on("update:model-value", lambda _e=None: _refresh())
        ui.timer(2.0, lambda: status.text.startswith("scanning") and _refresh())
        _refresh()


@ui.page("/admin")
def admin_overview_page():
    """Admin overview: headline stat cards + trend/distribution charts + top users.
    Read-only snapshot per load; the header's Overview button re-navigates to refresh."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect

    store = _store()
    stats = store.overview_stats()
    # Headline spend (Total spend card + Spend/day chart) comes from the SpendGuard
    # ledger, so it matches the OpenRouter dashboard exactly -- it counts every model
    # call, including non-project ones (eval/judge/smoketest). The per-user / per-
    # project / avg figures below stay project-attributed. Fall back to the project
    # numbers if the ledger can't be read.
    try:
        _guard = SpendGuard(Settings.from_env())
        ledger_total = _guard.spent_total()
        ledger_by_day = _guard.spent_by_day(30)
    except Exception:
        ledger_total = None
        ledger_by_day = store.spend_per_day(30)
    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    _admin_header("overview")

    def money(x):
        return "—" if x is None else f"${x:,.2f}"

    def latency(x):
        if x is None:
            return "—"
        return f"{x / 60:.1f} min" if x >= 60 else f"{x:.0f} s"

    with ui.column().classes("w-full mx-auto p-4 gap-4").style("max-width:1400px"):
        ui.label("Admin dashboard").classes("text-2xl font-bold text-white")

        def card(label: str, value: str, hint: str = "") -> None:
            with ui.card().classes("gap-0 items-start") \
                    .style("background:var(--kc-surface);border:1px solid var(--kc-border);min-width:150px"):
                ui.label(value).classes("text-2xl font-bold").style("color:#e2e8f0")
                ui.label(label).classes("text-xs").style("color:#94a3b8")
                if hint:
                    ui.label(hint).classes("text-xs").style("color:#64748b")

        w = stats["window_days"]
        with ui.row().classes("w-full flex-wrap gap-3"):
            card("Total users", str(stats["users_total"]), f"+{stats['users_new']} in {w}d")
            card("Admins", str(stats["admins"]))
            card("Total projects", str(stats["projects_total"]),
                 f"+{stats['projects_new']} in {w}d")
            spend_total = ledger_total if ledger_total is not None \
                else stats["spend_total_usd"]
            card("Total spend", money(spend_total),
                 f"${stats['spend_total_usd']:,.2f} on user projects")
            card("Avg / design", money(stats["spend_avg_usd"]))
            card("Avg latency", latency(stats["avg_latency_s"]))

        pp = store.projects_per_day(30)
        su = store.signups_per_day(30)
        sp = ledger_by_day  # ledger (all calls) -> matches the OpenRouter daily chart
        with ui.row().classes("w-full flex-wrap gap-4"):
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_line([d for d, _ in pp], [v for _, v in pp],
                                       title="Projects / day (30d)")) \
                    .classes("w-full").style("height:260px")
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_line([d for d, _ in su], [v for _, v in su],
                                       title="Signups / day (30d)", color="#60a5fa")) \
                    .classes("w-full").style("height:260px")
        with ui.row().classes("w-full flex-wrap gap-4"):
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_bar([d for d, _ in sp], [round(v, 2) for _, v in sp],
                                      title="Spend / day (30d)", color="#fbbf24")) \
                    .classes("w-full").style("height:260px")
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_pie(store.status_distribution(),
                                      title="Project status")) \
                    .classes("w-full").style("height:260px")
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_pie(store.tier_distribution(), title="User tiers")) \
                    .classes("w-full").style("height:260px")

        # Host-resource trends: drive / RAM / CPU over time. A background
        # sampler (started by the web process, kicraft/server/host_metrics.py)
        # appends a row every ~30 s; the admin page only reads. The timescale
        # selector rebuilds the three charts in place via ui.refreshable.
        ui.label("Host usage (drive · RAM · CPU)").classes(
            "text-base font-semibold text-white mt-2")
        with ui.row().classes("items-center gap-2 flex-wrap"):
            ui.label("Window").classes("text-xs").style("color:#94a3b8")

            @ui.refreshable
            def host_charts(selection: str) -> None:
                _render_host_charts(selection)

            def _on_window(e, hc=host_charts) -> None:
                hc.refresh(e.value)

            ui.select(_HOST_TIMESCALES, value="7d", on_change=_on_window) \
                .props("dark dense options-dense no-caps").style("width:110px")
            host_charts("7d")
        ui.label("Top users by projects") \
            .classes("text-base font-semibold text-white mt-2")
        top = sorted(store.users_with_project_counts(),
                     key=lambda r: r["project_count"], reverse=True)[:10]
        with ui.column().classes("w-full gap-1"):
            for r in top:
                with ui.row().classes("w-full items-center gap-3 text-xs") \
                        .style("border-top:1px solid var(--kc-border);padding:3px 0"):
                    ui.label(r["email"]).style("width:260px;color:#e2e8f0")
                    ui.badge(r["tier"], color="primary")
                    if r["role"] == "admin":
                        ui.badge("admin", color="purple")
                    ui.label(f"{r['project_count']} projects").style("color:#94a3b8")
                    ui.label(f"${r['spend_usd']:.2f}").style("color:#64748b")


@ui.page("/admin/users")
def admin_users_page():
    """User management: one row per user (tier, role, project_count, spend) with
    actions -- change tier, grant/revoke admin, issue a reset link, export JSON,
    delete. Every mutating handler re-checks is_admin() (defense in depth), and the
    self-demotion / last-admin guards keep the system from losing all its admins."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect

    store = _store()
    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    _admin_header("users")

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1400px"):
        ui.label("User management").classes("text-2xl font-bold text-white")
        search = ui.input(placeholder="Filter by email…").props("dense clearable") \
            .classes("w-72").style("color:#e2e8f0")
        container = ui.column().classes("w-full gap-0")

        def guard() -> bool:
            """Defense in depth: never trust the page-load gate for a mutation."""
            if not is_admin(_current_user()):
                ui.notify("Admin access required.", color="warning")
                return False
            return True

        def do_set_tier(email: str, value: str) -> None:
            if not guard():
                return
            try:
                store.set_tier(email, value)
                ui.notify(f"{email}: tier set to {value}.", color="positive")
            except ValueError as e:
                ui.notify(str(e), color="negative")
            build_users()

        def do_toggle_admin(row: dict) -> None:
            if not guard():
                return
            me = _current_user()
            making = row["role"] != "admin"
            if not making:
                if me is not None and row["id"] == me.id:
                    ui.notify("You can't revoke your own admin access.", color="warning")
                    return
                if store.count_role("admin") <= 1:
                    ui.notify("Refusing to remove the last admin.", color="warning")
                    return
            store.set_role(row["id"], "admin" if making else "user")
            ui.notify(f"{row['email']} is now {'an admin' if making else 'a user'}.",
                      color="positive")
            build_users()

        def do_reset_link(email: str) -> None:
            if not guard():
                return
            token = store.create_reset_token(email)
            if token is None:
                ui.notify("A reset link was issued moments ago; wait a minute and retry.",
                          color="warning")
                return
            url = f"{Settings.from_env().public_url}/reset?token={token}"
            with ui.dialog() as dlg, ui.card() \
                    .style("background:var(--kc-surface);border:1px solid var(--kc-border);min-width:520px"):
                ui.label(f"Password-reset link for {email}") \
                    .classes("text-sm font-bold").style("color:#e2e8f0")
                ui.label(f"Valid ~{_RESET_TTL_SECONDS // 60} min, single use. "
                         "Relay it to the user out-of-band.") \
                    .classes("text-xs").style("color:#94a3b8")
                ui.input(value=url).props("readonly outlined dense") \
                    .classes("w-full").style("color:#e2e8f0")
                with ui.row().classes("w-full justify-end"):
                    ui.button("Close", on_click=dlg.close).props("flat dense")
            dlg.open()

        def do_export(uid: int) -> None:
            if not guard():
                return
            data = store.export_user(uid)
            if data is None:
                ui.notify("No such user.", color="negative")
                return
            payload = json.dumps(data, indent=2, default=str).encode("utf-8")
            ui.download(payload, f"kicraft_export_{uid}.json", "application/json")

        def do_delete(row: dict) -> None:
            if not guard():
                return
            me = _current_user()
            if me is not None and row["id"] == me.id:
                ui.notify("You can't delete your own account here.", color="warning")
                return
            if row["role"] == "admin" and store.count_role("admin") <= 1:
                ui.notify("Refusing to delete the last admin.", color="warning")
                return

            def confirm() -> None:
                if not guard():
                    dlg.close()
                    return
                # A deleted account must never be charged again: cancel any
                # live Stripe subscription first (best-effort; deletion
                # proceeds either way and the orphaned sub stays visible in
                # the Stripe dashboard).
                target = store.get_user(row["id"])
                if target is not None:
                    billing.cancel_subscription_for_user(Settings.from_env(), target)
                store.delete_user(row["id"])
                dlg.close()
                ui.notify(f"Deleted {row['email']}.", color="positive")
                build_users()

            with ui.dialog() as dlg, ui.card() \
                    .style("background:var(--kc-surface);border:1px solid var(--kc-border);min-width:420px"):
                ui.label(f"Delete {row['email']}?") \
                    .classes("text-base font-bold").style("color:#e2e8f0")
                ui.label("Removes their account, project rows, and stored files. "
                         "This is irreversible.").classes("text-xs").style("color:#f87171")
                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("Cancel", on_click=dlg.close).props("flat dense")
                    ui.button("Delete", color="negative", on_click=confirm).props("dense")
            dlg.open()

        def build_users() -> None:
            container.clear()
            me = _current_user()
            flt = (search.value or "").strip().lower()
            rows = store.users_with_project_counts()
            if flt:
                rows = [r for r in rows if flt in r["email"].lower()]
            with container:
                with ui.row().classes("w-full items-center gap-2 text-xs font-bold") \
                        .style("color:#64748b;padding:2px 0"):
                    ui.label("email").style("width:230px")
                    ui.label("tier").style("width:96px")
                    ui.label("billing").style("width:76px")
                    ui.label("role").style("width:70px")
                    ui.label("proj").style("width:48px")
                    ui.label("spend").style("width:64px")
                    ui.label("joined").style("width:84px")
                    ui.label("actions").classes("flex-1")
                if not rows:
                    ui.label("No users match.").classes("text-sm").style("color:#94a3b8")
                for r in rows:
                    is_admin_row = r["role"] == "admin"
                    is_self = me is not None and r["id"] == me.id
                    with ui.row().classes("w-full items-center gap-2 text-xs") \
                            .style("border-top:1px solid var(--kc-border);padding:4px 0"):
                        ui.label(r["email"] + ("  (you)" if is_self else "")) \
                            .style("width:230px;color:#e2e8f0")
                        ui.select({"free": "Free", "pro": "Pro", "max": "Max"},
                                  value=r["tier"],
                                  on_change=lambda e, em=r["email"]: do_set_tier(em, e.value)) \
                            .props("dense options-dense").style("width:96px")
                        sub_status = r.get("subscription_status") or "-"
                        ui.label(sub_status).style(
                            "width:76px;color:"
                            + ("#34d399" if sub_status in ("active", "trialing")
                               else "#64748b")) \
                            .tooltip("Stripe subscription status (manual tier "
                                     "changes hold until the next webhook sync)")
                        ui.label(r["role"]).style(
                            f"width:70px;color:{'#a78bfa' if is_admin_row else '#64748b'}")
                        ui.label(str(r["project_count"])).style("width:48px;color:#cbd5e1")
                        ui.label(f"${r['spend_usd']:.2f}").style("width:64px;color:#cbd5e1")
                        ui.label((r["created_at"] or "")[:10]) \
                            .style("width:84px;color:#64748b")
                        with ui.row().classes("flex-1 gap-1 items-center"):
                            ui.button("Revoke" if is_admin_row else "Make admin",
                                      icon="remove_moderator" if is_admin_row
                                      else "admin_panel_settings",
                                      on_click=lambda row=r: do_toggle_admin(row)) \
                                .props("flat dense no-caps").classes("text-xs")
                            ui.button("Reset link", icon="link",
                                      on_click=lambda em=r["email"]: do_reset_link(em)) \
                                .props("flat dense no-caps").classes("text-xs")
                            ui.button("Export", icon="download",
                                      on_click=lambda uid=r["id"]: do_export(uid)) \
                                .props("flat dense no-caps").classes("text-xs") \
                                .tooltip("Account + project metadata as JSON "
                                         "(on-disk files via the CLI)")
                            ui.button("Delete", icon="delete", color="negative",
                                      on_click=lambda row=r: do_delete(row)) \
                                .props("flat dense no-caps").classes("text-xs")

        search.on_value_change(lambda: build_users())
        build_users()


@ui.page("/admin/invites")
def admin_invites_page():
    """Invite-code management: mint codes that sign a user up at a chosen tier
    for a set number of days (blank = forever), disable leaked or retired codes,
    and flip the public-launch switch that lets the Free tier register with no
    code at all. Every mutating handler re-checks is_admin() (defense in depth,
    same as /admin/users)."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect

    store = _store()
    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    _admin_header("invites")

    tier_options = {t: TIERS[t]["label"] for t in TIERS}

    def guard() -> bool:
        """Defense in depth: never trust the page-load gate for a mutation."""
        if not is_admin(_current_user()):
            ui.notify("Admin access required.", color="warning")
            return False
        return True

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1100px"):
        ui.label("Invite codes").classes("text-2xl font-bold text-white")

        # -- public-launch switch -------------------------------------------
        with ui.card().classes("w-full gap-1") \
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
            ui.label("Public signup").classes("text-base font-semibold text-white")

            def on_toggle(e) -> None:
                if not guard():
                    open_sw.value = store.signup_open()  # revert the flip
                    return
                store.set_signup_open(bool(e.value))
                ui.notify("Public signup is now "
                          f"{'OPEN: anyone can register on the Free tier' if e.value else 'closed: an invite code is required'}.",
                          color="positive")

            open_sw = ui.switch("Allow Free-tier signup without an invite code",
                                value=store.signup_open())
            open_sw.on_value_change(on_toggle)
            ui.label("Off = invite-only beta (every signup needs a code below). "
                     "On = public launch: the code field becomes optional, and a "
                     "code still upgrades the signup to its tier.") \
                .classes("text-xs").style("color:#94a3b8")

        # -- mint a new code --------------------------------------------------
        with ui.card().classes("w-full gap-1") \
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
            ui.label("New invite code").classes("text-base font-semibold text-white")
            with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                code_in = ui.input("Code", placeholder="FREEMAX") \
                    .props("dense").classes("w-44")
                tier_sel = ui.select(tier_options, value=DEFAULT_TIER, label="Tier") \
                    .props("dense options-dense").classes("w-28")
                days_in = ui.number("Days", min=1, precision=0) \
                    .props("dense clearable").classes("w-28") \
                    .tooltip("How long the signup keeps the tier; blank = forever")
                uses_in = ui.number("Max uses", min=1, precision=0) \
                    .props("dense clearable").classes("w-28") \
                    .tooltip("How many signups may redeem it; blank = unlimited")

                def do_create() -> None:
                    if not guard():
                        return
                    try:
                        c = store.create_invite_code(
                            code_in.value or "", tier_sel.value or DEFAULT_TIER,
                            duration_days=int(days_in.value) if days_in.value else None,
                            max_uses=int(uses_in.value) if uses_in.value else None)
                    except ValueError as e:
                        ui.notify(str(e), color="negative")
                        return
                    ui.notify(f"Created {c['code']}.", color="positive")
                    code_in.value = ""
                    build_codes()

                ui.button("Create", icon="add", on_click=do_create).props("dense")
            ui.label("Example: code FREEMAX, tier Max, days blank gives the Max "
                     "tier free forever; days 30 gives it for 30 days, then the "
                     "account drops back to Free.") \
                .classes("text-xs").style("color:#94a3b8")

        # -- existing codes ----------------------------------------------------
        container = ui.column().classes("w-full gap-0")

        def do_set_enabled(row: dict, enabled: bool) -> None:
            if not guard():
                return
            try:
                store.set_invite_code_enabled(row["id"], enabled)
            except ValueError as e:
                ui.notify(str(e), color="negative")
                return
            ui.notify(f"{row['code']} {'re-enabled' if enabled else 'disabled'}.",
                      color="positive")
            build_codes()

        def build_codes() -> None:
            container.clear()
            rows = store.list_invite_codes()
            with container:
                with ui.row().classes("w-full items-center gap-2 text-xs font-bold") \
                        .style("color:#64748b;padding:2px 0"):
                    ui.label("code").style("width:170px")
                    ui.label("tier").style("width:60px")
                    ui.label("grants").style("width:110px")
                    ui.label("uses").style("width:70px")
                    ui.label("status").style("width:70px")
                    ui.label("created").style("width:84px")
                    ui.label("last used").style("width:84px")
                    ui.label("actions").classes("flex-1")
                if not rows:
                    ui.label("No invite codes yet. Mint one above; the legacy "
                             "env code (if set) also still works.") \
                        .classes("text-sm").style("color:#94a3b8")
                for r in rows:
                    with ui.row().classes("w-full items-center gap-2 text-xs") \
                            .style("border-top:1px solid var(--kc-border);padding:4px 0"):
                        ui.label(r["code"]).style(
                            "width:170px;color:#e2e8f0;font-family:monospace")
                        ui.badge(TIERS[r["tier"]]["label"]
                                 if r["tier"] in TIERS else r["tier"],
                                 color="primary").style("width:60px")
                        ui.label("forever" if r["duration_days"] is None
                                 else f"{r['duration_days']} days") \
                            .style("width:110px;color:#cbd5e1")
                        ui.label(f"{r['use_count']} / "
                                 f"{r['max_uses'] if r['max_uses'] is not None else '∞'}") \
                            .style("width:70px;color:#cbd5e1")
                        if not r["enabled"]:
                            ui.label("disabled").style("width:70px;color:#f87171")
                        elif r["max_uses"] is not None \
                                and r["use_count"] >= r["max_uses"]:
                            ui.label("used up").style("width:70px;color:#f59e0b")
                        else:
                            ui.label("active").style("width:70px;color:#34d399")
                        ui.label((r["created_at"] or "")[:10]) \
                            .style("width:84px;color:#64748b")
                        ui.label((r["last_used_at"] or "")[:10] or "never") \
                            .style("width:84px;color:#64748b")
                        with ui.row().classes("flex-1 gap-1 items-center"):
                            if r["enabled"]:
                                ui.button("Disable", icon="block",
                                          on_click=lambda row=r:
                                          do_set_enabled(row, False)) \
                                    .props("flat dense no-caps").classes("text-xs")
                            else:
                                ui.button("Enable", icon="check_circle",
                                          on_click=lambda row=r:
                                          do_set_enabled(row, True)) \
                                    .props("flat dense no-caps").classes("text-xs")

        build_codes()

        if _signup_code():
            ui.label("Note: the legacy KICRAFT_SIGNUP_CODE env code is also still "
                     "accepted at signup (it grants the Free tier). Remove it from "
                     ".env to retire it.").classes("text-xs").style("color:#64748b")


@ui.page("/admin/core-components")
def admin_core_components_page():
    """Core-components registry viewer: one curated default part per common
    functional block (LDO tiers, buck/boost tiers, sensors, passive series).
    The repo catalog (kicraft/parts_library/core_blocks.json) is the source
    of truth and re-syncs into the DB on every restart; the architecture/BOM
    prompts consume the table per run. This page owns only the runtime state
    the sync preserves: the enabled flag and jlcparts price/stock snapshots.
    Part/block edits happen via git. Every mutating handler re-checks
    is_admin(), same as /admin/invites."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect

    store = _store()
    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    _admin_header("core components")

    def guard() -> bool:
        """Defense in depth: never trust the page-load gate for a mutation."""
        if not is_admin(_current_user()):
            ui.notify("Admin access required.", color="warning")
            return False
        return True

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1400px"):
        ui.label("Core components").classes("text-2xl font-bold text-white")

        with ui.card().classes("w-full gap-1") \
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("Default part per functional block") \
                    .classes("text-base font-semibold text-white")
            ui.label("Synced from the repo catalog "
                     "(kicraft/parts_library/core_blocks.json) on every "
                     "restart; the architecture/BOM prompts consume it per "
                     "run. Part and block edits happen via git; this page "
                     "owns only enable/disable and price/stock snapshots.") \
                .classes("text-xs").style("color:#94a3b8")
            if not jlcparts.available():
                ui.label("The jlcparts offline catalog is not installed on this "
                         "host, so the per-row price/stock refresh is hidden "
                         "(it works on the production box).") \
                    .classes("text-xs").style("color:#64748b")

        container = ui.column().classes("w-full gap-0")

        def do_set_enabled(row: dict, enabled: bool) -> None:
            if not guard():
                return
            try:
                store.update_core_component(row["id"], enabled=enabled)
            except ValueError as e:
                ui.notify(str(e), color="negative")
                return
            ui.notify(f"{row['function_key']} "
                      f"{'re-enabled' if enabled else 'disabled'}.",
                      color="positive")
            build_table()

        def do_refresh(row: dict) -> None:
            """Re-read price/stock for the row's LCSC id from the offline
            jlcparts catalog and stamp today's snapshot date."""
            if not guard():
                return
            info = jlcparts.lookup(row["default_lcsc"])
            if not info:
                ui.notify(f"{row['default_lcsc']} not found in the offline "
                          "catalog.", color="negative")
                return
            upd = store.record_core_component_snapshot(
                row["id"], price_usd=info.get("price"), stock=info.get("stock"))
            price_s = (f"${upd['price_usd']:.4f}"
                       if upd["price_usd"] is not None else "price unknown")
            ui.notify(f"{row['function_key']}: stock {upd['stock']}, {price_s} "
                      "(qty 1).", color="positive")
            build_table()

        def build_table() -> None:
            container.clear()
            rows = store.list_core_components()
            by_cat: dict[str, list[dict]] = {}
            for r in rows:
                by_cat.setdefault(r["category"], []).append(r)
            with container:
                if not rows:
                    ui.label("The registry is empty: the catalog sync found "
                             "no blocks (check the server log).") \
                        .classes("text-sm").style("color:#94a3b8")
                for cat in CORE_COMPONENT_CATEGORIES:
                    cat_rows = by_cat.get(cat, [])
                    if not cat_rows:
                        continue
                    ui.label(cat.capitalize()) \
                        .classes("text-sm font-bold mt-3").style("color:#e2e8f0")
                    with ui.row().classes(
                            "w-full items-center gap-2 text-xs font-bold") \
                            .style("color:#64748b;padding:2px 0"):
                        ui.label("function key").style("width:140px")
                        ui.label("block / tier").style("width:220px")
                        ui.label("default part").style("width:170px")
                        ui.label("lcsc").style("width:80px")
                        ui.label("bundle").style("width:120px")
                        ui.label("package").style("width:130px")
                        ui.label("price").style("width:70px")
                        ui.label("stock").style("width:80px")
                        ui.label("snapshot").style("width:84px")
                        ui.label("status").style("width:64px")
                        ui.label("actions").classes("flex-1")
                    for r in cat_rows:
                        dim = "" if r["enabled"] else "opacity:0.45;"
                        with ui.row().classes(
                                "w-full items-center gap-2 text-xs") \
                                .style("border-top:1px solid var(--kc-border);"
                                       f"padding:4px 0;{dim}"):
                            ui.label(r["function_key"]).style(
                                "width:140px;color:#e2e8f0;font-family:monospace")
                            with ui.column().classes("gap-0").style("width:220px"):
                                name = ui.label(r["display_name"]) \
                                    .style("color:#e2e8f0")
                                if r["selection_notes"]:
                                    name.tooltip(r["selection_notes"])
                                if r["qualifier"]:
                                    ui.label(r["qualifier"]) \
                                        .style("color:#64748b")
                            ui.label(r["default_mpn"]).style(
                                "width:170px;color:#cbd5e1;font-family:monospace")
                            if r["default_lcsc"]:
                                ui.link(r["default_lcsc"],
                                        "https://www.lcsc.com/product-detail/"
                                        f"{r['default_lcsc']}.html",
                                        new_tab=True) \
                                    .style("width:80px;color:#60a5fa;"
                                           "font-family:monospace")
                            else:
                                ui.label("series").style(
                                    "width:80px;color:#64748b")
                            ui.label(r.get("bundle") or "").style(
                                "width:120px;color:#34d399;"
                                "font-family:monospace")
                            ui.label(r["package"] or "").style(
                                "width:130px;color:#94a3b8")
                            ui.label(f"${r['price_usd']:.4f}"
                                     if r["price_usd"] is not None else "") \
                                .style("width:70px;color:#cbd5e1")
                            ui.label(f"{r['stock']:,}"
                                     if r["stock"] is not None else "") \
                                .style("width:80px;color:#cbd5e1")
                            ui.label(r["snapshot_date"] or "").style(
                                "width:84px;color:#64748b")
                            if r["enabled"]:
                                ui.label("active").style(
                                    "width:64px;color:#34d399")
                            else:
                                ui.label("disabled").style(
                                    "width:64px;color:#f87171")
                            with ui.row().classes("flex-1 gap-1 items-center"):
                                if r["enabled"]:
                                    ui.button("Disable", icon="block",
                                              on_click=lambda row=r:
                                              do_set_enabled(row, False)) \
                                        .props("flat dense no-caps") \
                                        .classes("text-xs")
                                else:
                                    ui.button("Enable", icon="check_circle",
                                              on_click=lambda row=r:
                                              do_set_enabled(row, True)) \
                                        .props("flat dense no-caps") \
                                        .classes("text-xs")
                                if jlcparts.available() and r["default_lcsc"]:
                                    ui.button("Refresh", icon="sync",
                                              on_click=lambda row=r:
                                              do_refresh(row)) \
                                        .props("flat dense no-caps") \
                                        .classes("text-xs") \
                                        .tooltip("Re-read price/stock from the "
                                                 "offline jlcparts catalog")

        build_table()
