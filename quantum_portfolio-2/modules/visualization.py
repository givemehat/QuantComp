"""
modules/visualization.py
─────────────────────────
Publication-quality Plotly charts — premium dark theme.

Bug fixes
─────────
- `titlefont` removed from all ColorBar objects (deprecated in Plotly ≥ 5.x).
  Replaced with `title=dict(text=..., font=dict(...))`.
- `title_font` in axis dicts also replaced with correct nested form.
- `opacity` on Scatterpolar moved to `fillcolor` rgba so border lines stay sharp.
- MC scatter color bug: `best_bitstring` (decoded) vs raw key mismatch fixed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS  — premium dark palette
# ══════════════════════════════════════════════════════════════════════════════
_BG        = "#070810"      # near-black base
_SURFACE   = "#0E101A"      # card / chart bg
_SURFACE2  = "#141728"      # elevated surface
_BORDER    = "#1C1F33"      # subtle divider
_BORDER2   = "#252944"      # hover / active border
_TEXT      = "#E8EAFF"      # primary text — slightly blue-tinted white
_MUTED     = "#6B7280"      # secondary / axis labels
_DIM       = "#3D4263"      # very muted elements

# Accent palette — vivid but harmonious
_INDIGO    = "#818CF8"      # primary accent (was #6366F1 — lighter for dark bg)
_CYAN      = "#22D3EE"      # quantum / QAOA
_VIOLET    = "#A78BFA"      # AI / LSTM
_AMBER     = "#FCD34D"      # max-Sharpe star
_EMERALD   = "#34D399"      # min-variance
_ROSE      = "#FB7185"      # negative / risk / warning
_ORANGE    = "#FB923C"      # tertiary accent
_TEAL      = "#2DD4BF"      # extra series

FONT_MONO   = "IBM Plex Mono, JetBrains Mono, Fira Code, monospace"
FONT_SANS   = "IBM Plex Sans, Inter, system-ui, sans-serif"

PALETTE = [_INDIGO, _CYAN, _VIOLET, _AMBER, _EMERALD, _ROSE, _ORANGE, _TEAL]


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _colorbar(title: str = "", x: float = 1.02) -> dict:
    """
    Canonical ColorBar dict — compatible with Plotly ≥ 5.x.
    `titlefont` is removed; replaced with `title=dict(font=dict(...))`.
    """
    cb: dict = {
        "tickfont":   dict(color=_MUTED, size=10, family=FONT_MONO),
        "bgcolor":    _SURFACE,
        "bordercolor": _BORDER,
        "borderwidth": 1,
        "outlinecolor": _BORDER,
        "x":           x,
    }
    if title:
        cb["title"] = dict(
            text=title,
            font=dict(color=_MUTED, size=10, family=FONT_MONO),
            side="right",
        )
    return cb


def _axis(title: str = "", show_grid: bool = True) -> dict:
    """Consistent axis styling."""
    return dict(
        title=dict(text=title, font=dict(color=_MUTED, size=11, family=FONT_MONO)),
        tickfont=dict(color=_MUTED, size=10, family=FONT_MONO),
        gridcolor=_BORDER if show_grid else "rgba(0,0,0,0)",
        gridwidth=1,
        linecolor=_BORDER2,
        tickcolor=_BORDER2,
        zeroline=False,
    )


def _base_layout(title: str = "", height: int = 420) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(color=_TEXT, size=14, family=FONT_MONO, weight=500),
            x=0.0,
            xanchor="left",
            pad=dict(l=4),
        ),
        height        = height,
        paper_bgcolor = _BG,
        plot_bgcolor  = _SURFACE,
        font          = dict(color=_MUTED, family=FONT_MONO, size=11),
        legend        = dict(
            bgcolor     = _SURFACE2,
            bordercolor = _BORDER2,
            borderwidth = 1,
            font        = dict(color=_TEXT, size=11, family=FONT_MONO),
            itemsizing  = "constant",
        ),
        margin  = dict(l=52, r=24, t=52, b=44),
        xaxis   = _axis(),
        yaxis   = _axis(),
        hoverlabel=dict(
            bgcolor   = _SURFACE2,
            bordercolor = _BORDER2,
            font      = dict(color=_TEXT, size=11, family=FONT_MONO),
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  1. PRICE HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def price_history(prices: pd.DataFrame) -> go.Figure:
    """Normalised (rebased to 100) price performance for all tickers."""
    rebased = prices / prices.iloc[0] * 100
    fig     = go.Figure()
    for i, col in enumerate(rebased.columns):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x    = rebased.index,
            y    = rebased[col],
            name = col,
            mode = "lines",
            line = dict(color=color, width=2),
            fill = "none",
            hovertemplate = f"<b>{col}</b><br>%{{x|%d %b %Y}}<br>Index: <b>%{{y:.1f}}</b><extra></extra>",
        ))
    layout = _base_layout("Rebased Price Performance  (Base = 100)", height=400)
    layout["xaxis"] = _axis(show_grid=False)
    layout["yaxis"] = _axis("Index Value")
    fig.update_layout(**layout)
    # Subtle range-selector
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        rangeselector=dict(
            buttons=[
                dict(count=3,  label="3M", step="month", stepmode="backward"),
                dict(count=6,  label="6M", step="month", stepmode="backward"),
                dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor    = _SURFACE2,
            activecolor= _INDIGO,
            bordercolor= _BORDER2,
            font       = dict(color=_TEXT, size=10, family=FONT_MONO),
            x=0, y=1.04,
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  2. LSTM TRAINING LOSS
# ══════════════════════════════════════════════════════════════════════════════

def lstm_loss_curve(history: "tf.keras.callbacks.History") -> go.Figure:
    """Train vs validation MSE loss over epochs with gradient fill."""
    epochs    = list(range(1, len(history.history["loss"]) + 1))
    train_loss = history.history["loss"]
    val_loss   = history.history["val_loss"]

    fig = go.Figure()
    # Train loss with fill
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss,
        name="Train",
        mode="lines",
        line=dict(color=_VIOLET, width=2.5),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(_VIOLET, 0.06),
        hovertemplate="Epoch %{x}<br>Train MSE: <b>%{y:.6f}</b><extra></extra>",
    ))
    # Val loss
    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss,
        name="Validation",
        mode="lines+markers",
        line=dict(color=_CYAN, width=2, dash="dot"),
        marker=dict(size=5, color=_CYAN, symbol="circle-open"),
        hovertemplate="Epoch %{x}<br>Val MSE: <b>%{y:.6f}</b><extra></extra>",
    ))
    # Best val epoch marker
    best_epoch = int(np.argmin(val_loss)) + 1
    fig.add_vline(
        x=best_epoch,
        line_color=_AMBER, line_dash="dot", line_width=1,
        annotation=dict(
            text=f"Best: epoch {best_epoch}",
            font=dict(color=_AMBER, size=10, family=FONT_MONO),
            yref="paper", y=1.0, yanchor="bottom",
        ),
    )
    layout = _base_layout("LSTM — Training Loss (MSE)", height=320)
    layout["xaxis"] = _axis("Epoch", show_grid=False)
    layout["yaxis"] = _axis("MSE")
    fig.update_xaxes(dtick=1)
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  3. PREDICTED RETURNS BAR
# ══════════════════════════════════════════════════════════════════════════════

def predicted_returns_bar(
    tickers: list[str],
    predicted_returns: np.ndarray,
    selected: Optional[list[str]] = None,
) -> go.Figure:
    """Bar chart of predicted expected returns with QAOA-selected highlight."""
    selected = selected or []
    colors, borders = [], []
    for t, r in zip(tickers, predicted_returns):
        if t in selected:
            colors.append(_AMBER)
            borders.append(_AMBER)
        elif r >= 0:
            colors.append(_hex_to_rgba(_EMERALD, 0.8))
            borders.append(_EMERALD)
        else:
            colors.append(_hex_to_rgba(_ROSE, 0.8))
            borders.append(_ROSE)

    fig = go.Figure(go.Bar(
        x=tickers,
        y=predicted_returns * 100,
        marker=dict(
            color      = colors,
            line       = dict(color=borders, width=1.5),
            cornerradius = 4,
        ),
        text=[f"{v*100:+.3f}%" for v in predicted_returns],
        textposition="outside",
        textfont=dict(color=_MUTED, size=10, family=FONT_MONO),
        hovertemplate="<b>%{x}</b><br>Predicted: <b>%{y:.4f}%</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_color=_BORDER2, line_width=1)
    layout = _base_layout("AI-Predicted Expected Returns", height=360)
    layout["xaxis"] = _axis()
    layout["yaxis"] = _axis("Predicted Return (%)")
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  4. COVARIANCE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def covariance_heatmap(cov_matrix: np.ndarray, tickers: list[str]) -> go.Figure:
    """
    Annotated covariance heatmap.
    FIX: `titlefont` replaced with `title=dict(font=dict(...))`.
    """
    fig = go.Figure(go.Heatmap(
        z         = cov_matrix,
        x         = tickers,
        y         = tickers,
        colorscale = [
            [0.00, "#050A14"],
            [0.35, "#0A2040"],
            [0.65, "#0E4080"],
            [1.00, _CYAN],
        ],
        text         = [[f"{v:.4f}" for v in row] for row in cov_matrix],
        texttemplate = "%{text}",
        textfont     = dict(size=10, family=FONT_MONO, color=_TEXT),
        showscale    = True,
        colorbar     = _colorbar("Cov"),          # ← FIX: no titlefont
        hovertemplate= "<b>%{x} / %{y}</b><br>Covariance: <b>%{z:.6f}</b><extra></extra>",
    ))
    layout = _base_layout("Annualised Covariance Matrix", height=400)
    layout["xaxis"] = dict(tickfont=dict(color=_TEXT, size=11, family=FONT_MONO))
    layout["yaxis"] = dict(tickfont=dict(color=_TEXT, size=11, family=FONT_MONO))
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  5. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def correlation_heatmap(log_returns: pd.DataFrame) -> go.Figure:
    """Pearson correlation heatmap with diverging RdBu palette."""
    corr    = log_returns.corr().values
    tickers = list(log_returns.columns)

    # Custom diverging scale: rose → surface → cyan
    scale = [
        [0.00, _ROSE],
        [0.35, _hex_to_rgba(_ROSE, 0.3)],
        [0.50, _SURFACE2],
        [0.65, _hex_to_rgba(_CYAN, 0.3)],
        [1.00, _CYAN],
    ]

    fig = go.Figure(go.Heatmap(
        z            = corr,
        x            = tickers,
        y            = tickers,
        colorscale   = scale,
        zmid         = 0,
        zmin         = -1,
        zmax         = 1,
        text         = [[f"{v:.2f}" for v in row] for row in corr],
        texttemplate = "%{text}",
        textfont     = dict(size=10, family=FONT_MONO, color=_TEXT),
        showscale    = True,
        colorbar     = _colorbar("Corr"),         # ← FIX: no titlefont
        hovertemplate= "<b>%{x} / %{y}</b><br>Correlation: <b>%{z:.4f}</b><extra></extra>",
    ))
    layout = _base_layout("Return Correlation Matrix", height=400)
    layout["xaxis"] = dict(tickfont=dict(color=_TEXT, size=11, family=FONT_MONO))
    layout["yaxis"] = dict(tickfont=dict(color=_TEXT, size=11, family=FONT_MONO))
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  6. EFFICIENT FRONTIER
# ══════════════════════════════════════════════════════════════════════════════

def efficient_frontier_chart(
    frontier_vols: np.ndarray,
    frontier_rets: np.ndarray,
    ms_stats: tuple[float, float, float],
    mv_stats: tuple[float, float, float],
    tickers: list[str],
    expected_returns: "pd.Series | np.ndarray",
    cov_matrix: np.ndarray,
    q_stats: Optional[tuple[float, float]] = None,
    mc_vols: Optional[np.ndarray] = None,
    mc_rets: Optional[np.ndarray] = None,
    mc_sharpes: Optional[np.ndarray] = None,
) -> go.Figure:
    """Full efficient frontier — MC scatter, frontier line, portfolio markers."""
    fig = go.Figure()

    # ── Monte Carlo feasibility scatter ──────────────────────────────────────
    if mc_vols is not None and len(mc_vols) > 0:
        colorbar_mc = _colorbar("Sharpe", x=1.02)   # ← FIX: no titlefont
        fig.add_trace(go.Scatter(
            x    = mc_vols * 100,
            y    = mc_rets * 100,
            mode = "markers",
            marker=dict(
                color     = mc_sharpes,
                colorscale= [
                    [0.0, _hex_to_rgba(_VIOLET, 0.9)],
                    [0.5, _hex_to_rgba(_INDIGO, 0.9)],
                    [1.0, _hex_to_rgba(_CYAN,   0.9)],
                ],
                size      = 3,
                opacity   = 0.30,
                showscale = True,
                colorbar  = colorbar_mc,
            ),
            name      = "Random Portfolios",
            hoverinfo = "skip",
        ))

    # ── Efficient frontier line ───────────────────────────────────────────────
    if len(frontier_vols) > 0:
        fig.add_trace(go.Scatter(
            x    = frontier_vols * 100,
            y    = frontier_rets * 100,
            mode = "lines",
            name = "Efficient Frontier",
            line = dict(color=_INDIGO, width=3),
            hovertemplate="Vol: <b>%{x:.2f}%</b><br>Return: <b>%{y:.2f}%</b><extra></extra>",
        ))

    # ── Max Sharpe ────────────────────────────────────────────────────────────
    ms_ret, ms_vol, ms_sharpe = ms_stats
    fig.add_trace(go.Scatter(
        x    = [ms_vol * 100],
        y    = [ms_ret * 100],
        mode = "markers+text",
        marker=dict(symbol="star", size=22, color=_AMBER,
                    line=dict(color=_TEXT, width=1.5)),
        text         = ["Max Sharpe"],
        textposition = "top right",
        textfont     = dict(color=_AMBER, size=11, family=FONT_MONO),
        name         = f"Max Sharpe  ({ms_sharpe:.2f})",
        hovertemplate= f"Max Sharpe: <b>{ms_sharpe:.2f}</b><br>Return: {ms_ret*100:.2f}%<br>Vol: {ms_vol*100:.2f}%<extra></extra>",
    ))

    # ── Min Variance ──────────────────────────────────────────────────────────
    mv_ret, mv_vol, _ = mv_stats
    fig.add_trace(go.Scatter(
        x    = [mv_vol * 100],
        y    = [mv_ret * 100],
        mode = "markers+text",
        marker=dict(symbol="diamond", size=14, color=_EMERALD,
                    line=dict(color=_TEXT, width=1.5)),
        text         = ["Min Var"],
        textposition = "top right",
        textfont     = dict(color=_EMERALD, size=11, family=FONT_MONO),
        name         = "Min Variance",
        hovertemplate= f"Min Variance<br>Return: {mv_ret*100:.2f}%<br>Vol: {mv_vol*100:.2f}%<extra></extra>",
    ))

    # ── QAOA ──────────────────────────────────────────────────────────────────
    if q_stats is not None:
        q_ret, q_vol = q_stats
        fig.add_trace(go.Scatter(
            x    = [q_vol * 100],
            y    = [q_ret * 100],
            mode = "markers+text",
            marker=dict(symbol="square", size=16, color=_CYAN,
                        line=dict(color=_TEXT, width=1.5)),
            text         = ["QAOA"],
            textposition = "top right",
            textfont     = dict(color=_CYAN, size=11, family=FONT_MONO),
            name         = "Quantum (QAOA)",
            hovertemplate= f"Quantum QAOA<br>Return: {q_ret*100:.2f}%<br>Vol: {q_vol*100:.2f}%<extra></extra>",
        ))

    # ── Individual assets ─────────────────────────────────────────────────────
    asset_vols = np.sqrt(np.diag(cov_matrix))
    mu_arr     = np.asarray(expected_returns)
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x    = [asset_vols[i] * 100],
            y    = [mu_arr[i] * 100],
            mode = "markers+text",
            marker=dict(symbol="circle", size=8, color=_ROSE, opacity=0.9,
                        line=dict(color=_BORDER2, width=1)),
            text         = [ticker],
            textposition = "top center",
            textfont     = dict(color=_MUTED, size=10, family=FONT_MONO),
            name         = ticker,
            showlegend   = False,
            hovertemplate= f"<b>{ticker}</b><br>Return: {mu_arr[i]*100:.2f}%<br>Vol: {asset_vols[i]*100:.2f}%<extra></extra>",
        ))

    layout = _base_layout("Efficient Frontier", height=520)
    layout["xaxis"] = _axis("Volatility (% p.a.)")
    layout["yaxis"] = _axis("Return (% p.a.)")
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  7. QAOA BITSTRING PROBABILITIES
# ══════════════════════════════════════════════════════════════════════════════

def bitstring_probability_chart(
    probabilities: dict[str, float],
    best_raw_bitstring: str,
    tickers: list[str],
    top_k: int = 12,
) -> go.Figure:
    """
    Horizontal bar chart of top-k bitstring probabilities.
    Gold bar = the best (most probable / selected) bitstring.
    """
    sorted_items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    labels, vals, colors = [], [], []

    for bits, prob in sorted_items:
        n   = len(tickers)
        sel = [tickers[i] for i, b in enumerate(bits[:n]) if b == "1"]
        label = f"{bits[:n]}  [{', '.join(sel) if sel else 'none'}]"
        labels.append(label)
        vals.append(prob)
        colors.append(_AMBER if bits == best_raw_bitstring else _hex_to_rgba(_INDIGO, 0.75))

    if not vals:
        vals, labels, colors = [0], ["no data"], [_MUTED]

    fig = go.Figure(go.Bar(
        y            = labels,
        x            = vals,
        orientation  = "h",
        marker       = dict(
            color        = colors,
            line         = dict(color=_BORDER2, width=0.5),
            cornerradius = 4,
        ),
        text         = [f"{v*100:.1f}%" for v in vals],
        textposition = "outside",
        textfont     = dict(color=_MUTED, size=10, family=FONT_MONO),
        hovertemplate= "Bitstring: <b>%{y}</b><br>Probability: <b>%{x:.4f}</b><extra></extra>",
    ))
    layout = _base_layout("QAOA Measurement Distribution  (Top Solutions)", height=max(340, top_k * 34))
    layout["xaxis"] = _axis("Probability", show_grid=True)
    layout["yaxis"] = dict(
        tickfont   = dict(color=_TEXT, size=10, family=FONT_MONO),
        autorange  = "reversed",
        linecolor  = _BORDER,
        tickcolor  = _BORDER,
    )
    layout["xaxis"]["range"] = [0, max(vals) * 1.30]
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  8. PORTFOLIO ALLOCATION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def allocation_comparison(
    tickers: list[str],
    ms_weights: np.ndarray,
    mv_weights: np.ndarray,
    q_selection: np.ndarray,
) -> go.Figure:
    """Grouped bar chart comparing three portfolio weight allocations."""
    q_weights = np.zeros(len(tickers))
    sel_idx   = np.where(q_selection == 1)[0]
    if len(sel_idx) > 0:
        q_weights[sel_idx] = 1.0 / len(sel_idx)

    fig = go.Figure()
    for name, weights, color in [
        ("Max Sharpe",     ms_weights, _AMBER),
        ("Min Variance",   mv_weights, _EMERALD),
        ("Quantum (QAOA)", q_weights,  _CYAN),
    ]:
        fig.add_trace(go.Bar(
            name         = name,
            x            = tickers,
            y            = weights * 100,
            marker       = dict(
                color        = color,
                line         = dict(color=_hex_to_rgba(color, 0.5), width=1),
                cornerradius = 4,
            ),
            hovertemplate= f"<b>{name}</b><br>%{{x}}: <b>%{{y:.2f}}%</b><extra></extra>",
        ))

    layout = _base_layout("Portfolio Allocation Comparison", height=400)
    layout["barmode"]        = "group"
    layout["bargap"]         = 0.25
    layout["bargroupgap"]    = 0.08
    layout["xaxis"]          = _axis()
    layout["yaxis"]          = _axis("Weight (%)")
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  9. ROLLING VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

def rolling_volatility_chart(rolling_vol: pd.DataFrame) -> go.Figure:
    """21-day rolling annualised volatility for all tickers."""
    fig = go.Figure()
    for i, col in enumerate(rolling_vol.columns):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x    = rolling_vol.index,
            y    = rolling_vol[col] * 100,
            name = col,
            mode = "lines",
            line = dict(color=color, width=1.8),
            fill = "tozeroy",
            fillcolor = _hex_to_rgba(color, 0.04),
            hovertemplate = f"<b>{col}</b><br>%{{x|%d %b}}<br>Vol: <b>%{{y:.1f}}%</b><extra></extra>",
        ))
    layout = _base_layout("21-Day Rolling Volatility  (Annualised)", height=360)
    layout["xaxis"] = _axis(show_grid=False)
    layout["yaxis"] = _axis("Annualised Vol (%)")
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  10. RISK RADAR
# ══════════════════════════════════════════════════════════════════════════════

def risk_radar(
    labels: list[str],
    ms_vals: list[float],
    mv_vals: list[float],
    q_vals: list[float],
) -> go.Figure:
    """
    Spider / radar chart comparing normalised risk metrics.
    FIX: opacity on fill moved to rgba so lines stay crisp.
    """
    fig = go.Figure()
    for name, vals, color in [
        ("Max Sharpe",   ms_vals, _AMBER),
        ("Min Variance", mv_vals, _EMERALD),
        ("QAOA",         q_vals,  _CYAN),
    ]:
        fig.add_trace(go.Scatterpolar(
            r         = vals + [vals[0]],
            theta     = labels + [labels[0]],
            name      = name,
            line      = dict(color=color, width=2.5),
            fill      = "toself",
            fillcolor = _hex_to_rgba(color, 0.10),  # ← FIX: alpha only on fill
            hovertemplate="<b>%{theta}</b><br>Score: <b>%{r:.3f}</b><extra></extra>",
        ))

    layout = _base_layout("Risk Profile Comparison  (Normalised)", height=420)
    layout["polar"] = dict(
        bgcolor     = _SURFACE,
        radialaxis  = dict(
            visible    = True,
            color      = _DIM,
            gridcolor  = _BORDER,
            tickfont   = dict(color=_MUTED, size=9, family=FONT_MONO),
            range      = [0, 1],
        ),
        angularaxis = dict(
            color     = _MUTED,
            gridcolor = _BORDER,
            tickfont  = dict(color=_TEXT, size=11, family=FONT_MONO),
        ),
    )
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  11. CUMULATIVE RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def cumulative_returns_chart(
    prices: pd.DataFrame,
    ms_weights: np.ndarray,
    mv_weights: np.ndarray,
    q_weights: np.ndarray,
) -> go.Figure:
    """Hypothetical growth of $1 invested in each strategy."""
    rets   = np.log(prices / prices.shift(1)).dropna()
    ms_cum = (rets @ ms_weights).cumsum().apply(np.exp)
    mv_cum = (rets @ mv_weights).cumsum().apply(np.exp)
    q_cum  = (rets @ q_weights).cumsum().apply(np.exp)

    fig = go.Figure()
    for label, series, color in [
        ("Max Sharpe",     ms_cum, _AMBER),
        ("Min Variance",   mv_cum, _EMERALD),
        ("Quantum (QAOA)", q_cum,  _CYAN),
    ]:
        fig.add_trace(go.Scatter(
            x    = series.index,
            y    = series.values,
            name = label,
            mode = "lines",
            line = dict(color=color, width=2.5),
            fill = "tozeroy",
            fillcolor = _hex_to_rgba(color, 0.05),
            hovertemplate = f"<b>{label}</b><br>%{{x|%b %Y}}<br>$<b>%{{y:.3f}}</b><extra></extra>",
        ))

    fig.add_hline(y=1.0, line_color=_BORDER2, line_dash="dot", line_width=1,
                  annotation=dict(text="Initial $1", font=dict(color=_MUTED, size=9, family=FONT_MONO),
                                  xref="paper", x=0.01))
    layout = _base_layout("Cumulative Portfolio Growth  ($1 Invested)", height=400)
    layout["xaxis"] = _axis(show_grid=False)
    layout["yaxis"] = _axis("Portfolio Value ($)")
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  12. DRAWDOWN
# ══════════════════════════════════════════════════════════════════════════════

def drawdown_chart(
    prices: pd.DataFrame,
    ms_weights: np.ndarray,
    mv_weights: np.ndarray,
    q_weights: np.ndarray,
) -> go.Figure:
    """Portfolio drawdown time-series for all three strategies."""
    rets = np.log(prices / prices.shift(1)).dropna()

    def _dd(w: np.ndarray) -> pd.Series:
        port  = (rets @ w).cumsum().apply(np.exp)
        peak  = port.cummax()
        return ((port - peak) / peak) * 100

    fig = go.Figure()
    for label, w, color in [
        ("Max Sharpe",     ms_weights, _AMBER),
        ("Min Variance",   mv_weights, _EMERALD),
        ("Quantum (QAOA)", q_weights,  _CYAN),
    ]:
        dd = _dd(w)
        fig.add_trace(go.Scatter(
            x    = dd.index,
            y    = dd.values,
            name = label,
            mode = "lines",
            line = dict(color=color, width=1.8),
            fill = "tozeroy",
            fillcolor = _hex_to_rgba(color, 0.12),
            hovertemplate = f"<b>{label}</b><br>%{{x|%b %Y}}<br>DD: <b>%{{y:.2f}}%</b><extra></extra>",
        ))

    layout = _base_layout("Portfolio Drawdown", height=360)
    layout["xaxis"] = _axis(show_grid=False)
    layout["yaxis"] = _axis("Drawdown (%)")
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  13. RETURN DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def returns_distribution_chart(log_returns: pd.DataFrame) -> go.Figure:
    """Daily log-return histogram overlay for all tickers."""
    fig = go.Figure()
    for i, col in enumerate(log_returns.columns):
        color = PALETTE[i % len(PALETTE)]
        vals  = log_returns[col].dropna().values * 100
        fig.add_trace(go.Histogram(
            x             = vals,
            name          = col,
            opacity       = 0.65,
            nbinsx        = 60,
            marker        = dict(
                color = _hex_to_rgba(color, 0.7),
                line  = dict(color=color, width=0.5),
            ),
            hovertemplate = f"<b>{col}</b><br>Return bin: %{{x:.2f}}%<br>Count: <b>%{{y}}</b><extra></extra>",
        ))

    layout = _base_layout("Daily Return Distribution", height=380)
    layout["barmode"]        = "overlay"
    layout["xaxis"]          = _axis("Daily Log-Return (%)")
    layout["yaxis"]          = _axis("Frequency")
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  14. ROLLING SHARPE
# ══════════════════════════════════════════════════════════════════════════════

def rolling_sharpe_chart(rolling_sharpe: pd.DataFrame) -> go.Figure:
    """63-day rolling annualised Sharpe ratio per ticker."""
    fig = go.Figure()
    for i, col in enumerate(rolling_sharpe.columns):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x    = rolling_sharpe.index,
            y    = rolling_sharpe[col],
            name = col,
            mode = "lines",
            line = dict(color=color, width=1.8),
            hovertemplate = f"<b>{col}</b><br>%{{x|%d %b}}<br>Sharpe: <b>%{{y:.2f}}</b><extra></extra>",
        ))

    fig.add_hline(y=0,   line_color=_BORDER2, line_width=1)
    fig.add_hrect(y0=0, y1=1, fillcolor=_hex_to_rgba(_EMERALD, 0.03), line_width=0)

    layout = _base_layout("63-Day Rolling Sharpe Ratio  (Annualised)", height=360)
    layout["xaxis"] = _axis(show_grid=False)
    layout["yaxis"] = _axis("Sharpe Ratio")
    fig.update_layout(**layout)
    return fig
