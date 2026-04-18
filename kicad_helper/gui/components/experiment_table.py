"""AG Grid experiment table component for hierarchical experiment rounds."""

from __future__ import annotations

from nicegui import ui


def create_experiment_table(
    rounds: list[dict],
    on_select=None,
) -> ui.aggrid:
    """Build an AG Grid table for hierarchical experiment rounds."""

    columns = [
        {
            "field": "round_num",
            "headerName": "#",
            "width": 70,
            "sortable": True,
            "filter": "agNumberColumnFilter",
        },
        {
            "field": "score",
            "headerName": "Score",
            "width": 95,
            "sortable": True,
            "filter": "agNumberColumnFilter",
            "valueFormatter": "x.value == null ? '—' : x.value.toFixed(2)",
            "cellClass": "text-green-300 font-medium",
        },
        {
            "field": "mode",
            "headerName": "Mode",
            "width": 120,
            "sortable": True,
            "filter": True,
            "cellClassRules": {
                "text-blue-400": "x.value === 'hierarchical'",
                "text-purple-400": "x.value === 'baseline'",
                "text-gray-300": "x.value == null || x.value === ''",
            },
        },
        {
            "field": "kept",
            "headerName": "Kept",
            "width": 85,
            "sortable": True,
            "filter": True,
            "cellRenderer": "params.value ? '✓' : '✗'",
            "cellClassRules": {
                "text-green-400 font-bold": "x.value === true",
                "text-gray-500": "x.value === false",
            },
        },
        {
            "field": "leaf_accepted",
            "headerName": "Leafs",
            "width": 110,
            "sortable": True,
            "filter": "agNumberColumnFilter",
            "valueGetter": """
                (() => {
                    const accepted = x.data?.leaf_accepted ?? x.data?.hierarchy?.leaf_accepted;
                    const total = x.data?.leaf_total ?? x.data?.hierarchy?.leaf_total;
                    if (accepted == null && total == null) return null;
                    return `${accepted ?? 0}/${total ?? 0}`;
                })()
            """,
            "cellClass": "font-mono text-cyan-300",
        },
        {
            "field": "accepted_trace_count",
            "headerName": "Traces",
            "width": 95,
            "sortable": True,
            "filter": "agNumberColumnFilter",
            "valueGetter": "x.data?.accepted_trace_count ?? x.data?.hierarchy?.accepted_trace_count ?? null",
            "cellClass": "text-cyan-200",
        },
        {
            "field": "accepted_via_count",
            "headerName": "Vias",
            "width": 85,
            "sortable": True,
            "filter": "agNumberColumnFilter",
            "valueGetter": "x.data?.accepted_via_count ?? x.data?.hierarchy?.accepted_via_count ?? null",
            "cellClass": "text-amber-200",
        },
        {
            "field": "parent_composed",
            "headerName": "Parent",
            "width": 95,
            "sortable": True,
            "filter": True,
            "valueGetter": "x.data?.parent_composed ?? x.data?.hierarchy?.parent_composed ?? null",
            "cellRenderer": """
                params => {
                    if (params.value === true) return 'OK';
                    if (params.value === false) return '—';
                    return '—';
                }
            """,
            "cellClassRules": {
                "text-green-400 font-bold": "x.value === true",
                "text-gray-500": "x.value !== true",
            },
        },
        {
            "field": "parent_routed",
            "headerName": "Parent Route",
            "width": 120,
            "sortable": True,
            "filter": True,
            "valueGetter": "x.data?.parent_routed ?? x.data?.hierarchy?.parent_routed ?? null",
            "cellRenderer": """
                params => {
                    if (params.value === true) return 'ROUTED';
                    if (params.value === false) return '—';
                    return '—';
                }
            """,
            "cellClassRules": {
                "text-green-400 font-bold": "x.value === true",
                "text-gray-500": "x.value !== true",
            },
        },
        {
            "field": "latest_stage",
            "headerName": "Stage",
            "width": 140,
            "sortable": True,
            "filter": True,
            "valueGetter": "x.data?.latest_stage ?? x.data?.stage ?? ''",
            "cellClassRules": {
                "text-blue-300": "x.value === 'solve_leafs'",
                "text-amber-300": "x.value === 'compose_parent'",
                "text-green-300": "x.value === 'route_parent' || x.value === 'done' || x.value === 'complete'",
            },
        },
        {
            "field": "duration_s",
            "headerName": "Time(s)",
            "width": 95,
            "sortable": True,
            "filter": "agNumberColumnFilter",
            "valueFormatter": "x.value == null ? '—' : x.value.toFixed(1)",
        },
        {
            "field": "details",
            "headerName": "Details",
            "flex": 1,
            "minWidth": 260,
            "sortable": False,
            "filter": True,
            "tooltipField": "details",
            "cellClass": "text-gray-300",
        },
    ]

    grid = ui.aggrid(
        {
            "columnDefs": columns,
            "rowData": rounds,
            "defaultColDef": {
                "resizable": True,
                "sortable": True,
                "filter": True,
            },
            "rowSelection": "single",
            "domLayout": "autoHeight",
            "pagination": True,
            "paginationPageSize": 25,
            ":getRowStyle": """
                params => {
                    if (params.data?.kept) {
                        return {background: 'rgba(81, 207, 102, 0.08)'};
                    }
                    if (params.data?.parent_routed) {
                        return {background: 'rgba(34, 197, 94, 0.05)'};
                    }
                    return {};
                }
            """,
        }
    ).classes("w-full")

    if on_select:
        grid.on("cellClicked", on_select)

    return grid
