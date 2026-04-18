"""Parameter sensitivity analysis components."""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def build_sensitivity_figure(rounds: list[dict],
                             param_keys: list[str] | None = None,
                             ) -> go.Figure | None:
    """Scatter plots: parameter value vs score for each search dimension.

    Returns None if insufficient data.
    """
    if not rounds or len(rounds) < 5:
        return None

    # Extract parameter values from config_delta
    all_params = set()
    for r in rounds:
        delta = r.get("config_delta", {})
        if isinstance(delta, str):
            import json
            try:
                delta = json.loads(delta)
            except (json.JSONDecodeError, TypeError):
                delta = {}
        all_params.update(delta.keys())

    # Filter to requested keys or auto-detect
    if param_keys:
        params = [k for k in param_keys if k in all_params]
    else:
        # Skip internal keys
        params = [k for k in sorted(all_params)
                  if not k.startswith("_") and k not in (
                      "randomize_group_layout", "scatter_mode",
                      "reheat_strength",
                  )]

    if not params:
        return None

    n_cols = min(3, len(params))
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=params,
                        vertical_spacing=0.12,
                        horizontal_spacing=0.08)

    for idx, key in enumerate(params):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        x_vals, y_vals = [], []
        for r in rounds:
            delta = r.get("config_delta", {})
            if isinstance(delta, str):
                import json
                try:
                    delta = json.loads(delta)
                except (json.JSONDecodeError, TypeError):
                    continue
            if key in delta:
                x_vals.append(delta[key])
                y_vals.append(r.get("score", 0))

        if len(x_vals) < 3:
            continue

        # Correlation
        corr_text = ""
        if HAS_SCIPY and len(x_vals) >= 5:
            rho, pval = scipy_stats.spearmanr(x_vals, y_vals)
            corr_text = f" (ρ={rho:.2f}, p={pval:.3f})"

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            marker=dict(size=5, color="#4dabf7", opacity=0.6),
            name=key,
            showlegend=False,
            hovertemplate=f"{key}=%{{x:.4f}}<br>score=%{{y:.2f}}<extra></extra>",
        ), row=row, col=col)

        # Update subplot title with correlation
        fig.layout.annotations[idx].text = f"{key}{corr_text}"

    fig.update_layout(
        template="plotly_dark",
        height=300 * n_rows,
        margin=dict(l=50, r=20, t=40, b=30),
        showlegend=False,
    )
    fig.update_xaxes(title_font_size=10)
    fig.update_yaxes(title_text="Score", title_font_size=10)

    return fig


def build_correlation_matrix(rounds: list[dict],
                             param_keys: list[str] | None = None,
                             ) -> go.Figure | None:
    """Heatmap of Spearman correlations between parameters and scores."""
    if not HAS_SCIPY or len(rounds) < 10:
        return None

    # Collect parameter columns
    all_params = set()
    for r in rounds:
        delta = r.get("config_delta", {})
        if isinstance(delta, str):
            import json
            try:
                delta = json.loads(delta)
            except (json.JSONDecodeError, TypeError):
                delta = {}
        all_params.update(k for k in delta.keys()
                          if not k.startswith("_") and
                          isinstance(delta.get(k), (int, float)))

    if param_keys:
        params = [k for k in param_keys if k in all_params]
    else:
        params = sorted(all_params)

    score_keys = ["score", "placement_score", "route_completion",
                  "drc_total", "drc_shorts"]

    if len(params) < 2:
        return None

    # Build correlation matrix
    import json as json_mod
    labels = params + score_keys
    n = len(labels)
    z = [[0.0] * n for _ in range(n)]

    for i, ki in enumerate(labels):
        for j, kj in enumerate(labels):
            if i == j:
                z[i][j] = 1.0
                continue
            xi, xj = [], []
            for r in rounds:
                delta = r.get("config_delta", {})
                if isinstance(delta, str):
                    try:
                        delta = json_mod.loads(delta)
                    except (json_mod.JSONDecodeError, TypeError):
                        delta = {}
                vi = delta.get(ki, r.get(ki))
                vj = delta.get(kj, r.get(kj))
                if vi is not None and vj is not None:
                    xi.append(float(vi))
                    xj.append(float(vj))
            if len(xi) >= 5:
                rho, _ = scipy_stats.spearmanr(xi, xj)
                z[i][j] = rho if not (rho != rho) else 0  # NaN check
            else:
                z[i][j] = 0

    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}: ρ=%{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="Spearman Correlation Matrix",
        template="plotly_dark",
        height=400 + len(labels) * 20,
        margin=dict(l=120, r=20, t=50, b=80),
        xaxis=dict(tickangle=45),
    )
    return fig
