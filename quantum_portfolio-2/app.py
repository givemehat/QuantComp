"""
app.py — Hybrid AI + Quantum Portfolio Optimizer
═══════════════════════════════════════════════════
Premium Streamlit dashboard integrating:
  • LSTM return forecasting (AI layer)
  • QAOA portfolio selection (quantum layer)
  • Markowitz mean-variance (classical benchmark)

Bug fixes vs original
─────────────────────
- approx_ratio : was q_score / bf_score which is incorrect when both scores
                 are negative (common when risk_aversion > 0).  Now uses
                 quantum_optimizer.approximate_ratio() which handles all cases.
- bitstring chart : passed best_bitstring (decoded) to the chart; now passes
                    best_raw_bitstring so the gold highlight actually works.
- ticker input  : was limited to a hardcoded list of 10 symbols; users can now
                  type any valid Yahoo Finance ticker.
- cache control : added a clear-cache button so stale data can be refreshed
                  without restarting the server.
- session_state : results are cached in st.session_state so they survive
                  sidebar interaction without a full re-run.

New features
────────────
- Sortino, Calmar, Max Drawdown metrics displayed
- Cumulative returns tab with growth-of-$1 chart
- Drawdown chart
- Returns distribution chart
- Rolling Sharpe chart
- Walk-forward prediction mode for LSTM
- Min weight constraint for the classical optimiser
- More granular progress tracking
- Export enriched with all new metrics
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── Module path setup ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.data_pipeline import (
    fetch_prices,
    compute_annualised_stats,
    normalise_prices,
    prepare_lstm_sequences,
    compute_rolling_volatility,
    compute_rolling_sharpe,
    brute_force_optimal,
    get_cache_info,
    clear_cache,
    clear_all_cache,
)
from modules.lstm_model import (
    build_model, train, predict_expected_returns,
    walk_forward_predict, get_model_summary_dict,
)
from modules.classical_optimizer import (
    optimize_portfolio,
    portfolio_stats,
    efficient_frontier,
    monte_carlo_portfolios,
    value_at_risk,
    conditional_var,
    sortino_ratio,
    calmar_ratio,
    max_drawdown_from_weights,
)
from modules.quantum_optimizer import (
    run_qaoa,
    portfolio_metrics_from_selection,
    approximate_ratio,
)
from modules.visualization import (
    price_history,
    lstm_loss_curve,
    predicted_returns_bar,
    covariance_heatmap,
    correlation_heatmap,
    efficient_frontier_chart,
    bitstring_probability_chart,
    allocation_comparison,
    rolling_volatility_chart,
    risk_radar,
    cumulative_returns_chart,
    drawdown_chart,
    returns_distribution_chart,
    rolling_sharpe_chart,
)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Quantum Portfolio Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:      #0A0B0F; --surface: #12141A; --surface2: #181A22;
    --border:  #1E2028; --border2: #2A2D3A;
    --text:    #F0F2F7; --muted:   #8B91A5;
    --accent:  #6366F1; --cyan:    #00C9E0;
    --purple:  #8B5CF6; --gold:    #F59E0B;
    --green:   #10B981; --red:     #EF4444;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--muted) !important; font-size: 0.65rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    font-family: var(--mono) !important; font-weight: 500 !important;
    padding-top: 1.2rem !important; padding-bottom: 0.3rem !important;
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.8rem !important; color: var(--muted) !important;
}

.main-header {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.main-header h1 {
    font-family: var(--mono) !important; font-size: 1.6rem !important;
    font-weight: 600 !important; color: var(--text) !important;
    letter-spacing: -0.01em !important; margin: 0 !important;
}
.main-header p {
    font-size: 0.85rem !important; color: var(--muted) !important;
    margin: 0.4rem 0 0 0 !important; font-family: var(--mono) !important;
}
.badge {
    display: inline-block; background: var(--surface2);
    border: 1px solid var(--border2); color: var(--cyan);
    font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.08em;
    padding: 0.15rem 0.5rem; border-radius: 3px; margin-right: 0.4rem;
    text-transform: uppercase;
}

.section-label {
    font-family: var(--mono) !important; font-size: 0.65rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    color: var(--muted) !important; font-weight: 500 !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 1rem !important;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem; margin: 1rem 0;
}
.metric-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.2rem 1rem;
    transition: border-color 0.2s ease;
}
.metric-card:hover { border-color: var(--border2); }
.metric-card .label {
    font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem;
}
.metric-card .value {
    font-family: var(--mono); font-size: 1.45rem; font-weight: 600;
    color: var(--text); line-height: 1;
}
.metric-card .sub {
    font-family: var(--mono); font-size: 0.72rem;
    color: var(--muted); margin-top: 0.3rem;
}
.metric-card.pos .value  { color: var(--green); }
.metric-card.neg .value  { color: var(--red); }
.metric-card.cyan .value { color: var(--cyan); }
.metric-card.gold .value { color: var(--gold); }
.metric-card.purple .value { color: var(--purple); }

.result-banner {
    background: linear-gradient(135deg,rgba(0,201,224,0.08),rgba(99,102,241,0.08));
    border: 1px solid var(--cyan); border-radius: 8px;
    padding: 1.5rem; margin: 1rem 0; font-family: var(--mono);
}
.result-banner .headline {
    font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--cyan); margin-bottom: 0.8rem;
}
.result-banner .portfolio {
    font-size: 1.3rem; font-weight: 600; color: var(--text);
}
.result-banner .bits {
    font-size: 0.9rem; color: var(--muted);
    margin-top: 0.4rem; letter-spacing: 0.05em;
}

.compare-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.2rem; margin-bottom: 0.8rem;
}
.compare-card .name {
    font-family: var(--mono); font-size: 0.75rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem;
}
.compare-card .stats { display: flex; gap: 2rem; flex-wrap: wrap; }
.compare-card .stat-label {
    font-family: var(--mono); font-size: 0.62rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em;
}
.compare-card .stat-value {
    font-family: var(--mono); font-size: 1rem; font-weight: 600; color: var(--text);
}

.info-box {
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 6px; padding: 1rem 1.2rem; font-size: 0.82rem;
    color: var(--muted); font-family: var(--mono); line-height: 1.6; margin: 0.8rem 0;
}
.warn-box {
    background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.3);
    border-radius: 6px; padding: 1rem 1.2rem; font-size: 0.82rem;
    color: var(--gold); font-family: var(--mono); line-height: 1.6; margin: 0.8rem 0;
}
.error-box {
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3);
    border-radius: 6px; padding: 1rem 1.2rem; font-size: 0.82rem;
    color: var(--red); font-family: var(--mono); line-height: 1.6; margin: 0.8rem 0;
}

.stButton > button {
    background: var(--accent) !important; color: white !important;
    border: none !important; border-radius: 5px !important;
    font-family: var(--mono) !important; font-size: 0.8rem !important;
    letter-spacing: 0.05em !important; padding: 0.5rem 1.4rem !important;
    font-weight: 500 !important; transition: opacity 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--muted) !important;
    font-family: var(--mono) !important; font-size: 0.75rem !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important; border-bottom: 2px solid var(--accent) !important;
}

.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; font-family: var(--mono) !important;
    font-size: 0.82rem !important;
}
.stSlider > div { padding: 0 !important; }
[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 1rem;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important; color: var(--text) !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important; font-size: 0.65rem !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: var(--muted) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
.streamlit-expanderHeader {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 5px !important; font-family: var(--mono) !important;
    font-size: 0.78rem !important; color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, sub: str = "", variant: str = "") -> str:
    cls      = f"metric-card {variant}".strip()
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
    <div class="{cls}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
    </div>"""

def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)

def info(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def warn(text: str) -> None:
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def error_msg(text: str) -> None:
    st.markdown(f'<div class="error-box">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 1rem 0;border-bottom:1px solid var(--border);">
      <div style="font-family:var(--mono);font-size:1rem;font-weight:600;color:var(--text);">
        ◈ Quantum Portfolio Lab
      </div>
      <div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);
                  margin-top:0.25rem;letter-spacing:0.08em;">
        LSTM + QAOA · Research Prototype
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Universe ──────────────────────────────────────────────────────────────
    st.markdown("### Universe")

    PRESET_TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "NVDA", "META", "JPM", "BRK-B", "V",
        "XOM", "JNJ", "UNH", "WMT", "PG",
        "GLD", "TLT", "QQQ", "SPY", "BTC-USD",
    ]
    tickers_preset = st.multiselect(
        "Select from presets",
        options=PRESET_TICKERS,
        default=["AAPL", "MSFT", "GOOGL", "AMZN"],
        help="Choose 2–8 stocks from the preset list.",
    )

    # BUG FIX / NEW: allow custom tickers
    custom_input = st.text_input(
        "Add custom tickers (comma-separated)",
        value="",
        placeholder="e.g.  NFLX, SHOP, ASML",
        help="Type any valid Yahoo Finance symbol.",
    )
    custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers_preset + custom_tickers))  # deduplicate, preserve order

    period = st.selectbox(
        "Historical period",
        options=["1y", "2y", "3y", "5y"],
        index=2,
        help="Data window for training and covariance estimation.",
    )

    # Cache controls ──────────────────────────────────────────────────────────
    cache_info = get_cache_info(tickers, period) if tickers else None
    use_cache  = st.checkbox("Use disk cache", value=True,
                             help="Uncheck to force a fresh download.")
    if cache_info:
        st.caption(
            f"Cache: {cache_info['age_hours']}h old · {cache_info['size_kb']} KB"
        )
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("Clear cache", use_container_width=True):
            n = clear_all_cache()
            st.success(f"Cleared {n} cached file(s).")
    with col_c2:
        if st.button("Refresh data", use_container_width=True):
            clear_cache(tickers, period)
            st.rerun()

    # ── AI Model ──────────────────────────────────────────────────────────────
    st.markdown("### AI Model")
    model_type = st.radio(
        "Forecasting model",
        options=["LSTM (neural network)", "Historical mean (baseline)"],
        index=0,
    )
    use_lstm = model_type.startswith("LSTM")

    with st.expander("LSTM hyperparameters", expanded=False):
        n_steps    = st.slider("Look-back window (days)",  20, 120,  60, step=10)
        epochs     = st.slider("Training epochs",           5,  50,  15, step=5)
        patience   = st.slider("Early stopping patience",  2,  10,   5)
        test_win   = st.slider("Prediction window (days)", 10,  60,  30, step=5)
        wf_mode    = st.checkbox("Walk-forward prediction",  False,
                                 help="Average predictions over multiple windows "
                                      "for more robust estimates.")

    # ── Classical Optimizer ───────────────────────────────────────────────────
    st.markdown("### Classical Optimizer")
    risk_free  = st.slider(
        "Risk-free rate (%)", 0.0, 8.0, 4.0, step=0.25,
        help="Annualised risk-free rate for Sharpe computation.",
    ) / 100
    max_weight = st.slider(
        "Max single-asset weight (%)", 10, 100, 100, step=5,
        help="Diversification cap. 100 = unconstrained.",
    ) / 100
    min_weight = st.slider(
        "Min single-asset weight (%)", 0, 20, 0, step=1,
        help="Floor per asset (0 = standard long-only).",
    ) / 100

    # ── QAOA ─────────────────────────────────────────────────────────────────
    st.markdown("### Quantum Optimizer")
    reps           = st.slider("Circuit depth p", 1, 4, 1)
    optimizer_name = st.selectbox("Classical optimizer", ["COBYLA", "SPSA", "ADAM"])
    risk_aversion  = st.slider("Risk aversion λ", 0.0, 3.0, 0.5, step=0.1)
    max_k          = max(1, len(tickers)) if tickers else 4
    cardinality_raw = st.slider(
        "Cardinality k (0 = none)", 0, max_k, 0,
        help="Force QAOA to select exactly k stocks.",
    )
    cardinality_k: Optional[int] = cardinality_raw if cardinality_raw > 0 else None

    # ── Run ───────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    run = st.button("Run Analysis", use_container_width=True)
    if st.button("Clear results", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("qpl_"):
                del st.session_state[key]
        st.rerun()

    st.markdown("""
    <div style="padding-top:2rem;font-family:var(--mono);font-size:0.6rem;
                color:var(--muted);letter-spacing:0.06em;line-height:1.7;
                border-top:1px solid var(--border);margin-top:1rem;">
      Research prototype — not financial advice.<br>
      Quantum simulation: local statevector.<br>
      No IBM Quantum account required.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>Hybrid AI + Quantum Portfolio Optimizer</h1>
  <p>
    <span class="badge">LSTM</span>
    <span class="badge">QAOA</span>
    <span class="badge">Markowitz</span>
    Return forecasting via deep learning · Combinatorial selection via quantum optimization
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  IDLE STATE
# ══════════════════════════════════════════════════════════════════════════════
if not run and "qpl_results" not in st.session_state:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="compare-card">
          <div class="name">Data Layer</div>
          <div style="font-family:var(--mono);font-size:0.82rem;color:var(--muted);line-height:1.7;">
            Yahoo Finance via yfinance<br>Adjusted close · Forward-filled<br>
            Annualised returns + PSD covariance
          </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="compare-card">
          <div class="name">AI Forecasting</div>
          <div style="font-family:var(--mono);font-size:0.82rem;color:var(--muted);line-height:1.7;">
            Stacked LSTM (64→32 units)<br>BatchNorm · Early stopping<br>
            Walk-forward or rolling window
          </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="compare-card">
          <div class="name">Quantum Optimizer</div>
          <div style="font-family:var(--mono);font-size:0.82rem;color:var(--muted);line-height:1.7;">
            QUBO → Ising Hamiltonian<br>QAOA with StatevectorSampler<br>
            COBYLA / SPSA / ADAM tuning
          </div>
        </div>""", unsafe_allow_html=True)

    info("Configure the sidebar and click <strong>Run Analysis</strong> to start the pipeline.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
if not tickers:
    error_msg("No tickers selected. Add at least 2 stocks in the sidebar.")
    st.stop()
if len(tickers) < 2:
    error_msg("Please select at least 2 tickers for a meaningful portfolio.")
    st.stop()
if len(tickers) > 8:
    warn("More than 8 tickers increases QAOA circuit complexity significantly. "
         "Consider limiting to 4–6 for reasonable runtime.")

n_stocks = len(tickers)

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE EXECUTION (only when Run is clicked or session_state is empty)
# ══════════════════════════════════════════════════════════════════════════════
if run:
    progress = st.progress(0, text="Initialising pipeline...")

    # ── Step 1: Market Data ───────────────────────────────────────────────────
    with st.spinner("Downloading market data..."):
        try:
            prices = fetch_prices(tickers, period=period, use_cache=use_cache)
            tickers  = [t for t in tickers if t in prices.columns]
            n_stocks = len(tickers)
            expected_returns_series, cov_matrix, log_returns = compute_annualised_stats(prices)
            rolling_vol   = compute_rolling_volatility(prices)
            rolling_sharpe_df = compute_rolling_sharpe(prices, risk_free_rate=risk_free)
            progress.progress(15, text="Market data loaded.")
        except Exception as exc:
            error_msg(f"Data download failed: {exc}")
            progress.empty()
            st.stop()

    # ── Step 2: AI Forecasting ────────────────────────────────────────────────
    with st.spinner("Training forecasting model..."):
        try:
            if use_lstm:
                scaled, scaler = normalise_prices(prices)
                X_tr, y_tr, X_val, y_val = prepare_lstm_sequences(scaled, n_steps=n_steps)
                if len(X_tr) < 32:
                    warn("Training set is very small (<32 samples). "
                         "Try a longer period or a smaller look-back window.")
                lstm    = build_model(input_shape=(n_steps, n_stocks), n_outputs=n_stocks)
                history = train(lstm, X_tr, y_tr, X_val, y_val,
                                epochs=epochs, patience=patience)
                if wf_mode:
                    predicted_returns = walk_forward_predict(
                        lstm, scaled, n_steps=n_steps)
                else:
                    predicted_returns = predict_expected_returns(
                        lstm, scaled, n_steps=n_steps, test_window=test_win)
                model_info = get_model_summary_dict(lstm)
            else:
                history           = None
                model_info        = {}
                predicted_returns = (log_returns.mean() * 252).values
            progress.progress(45, text="Forecasting complete.")
        except Exception as exc:
            error_msg(f"AI model failed: {exc}")
            if st.checkbox("Show AI traceback"):
                st.code(traceback.format_exc())
            progress.empty()
            st.stop()

    # ── Step 3: Classical Optimization ───────────────────────────────────────
    with st.spinner("Running classical optimization..."):
        try:
            mu_series  = pd.Series(predicted_returns, index=tickers)

            ms_res     = optimize_portfolio(mu_series, cov_matrix, risk_free,
                                            max_weight=max_weight, min_weight=min_weight)
            ms_weights = ms_res.x
            ms_ret, ms_vol, ms_sharpe = portfolio_stats(ms_weights, mu_series, cov_matrix, risk_free)
            ms_sortino = sortino_ratio(ms_weights, log_returns, risk_free)
            ms_calmar  = calmar_ratio(ms_weights, log_returns, prices, risk_free)
            ms_mdd     = max_drawdown_from_weights(ms_weights, prices)

            mv_res     = optimize_portfolio(mu_series, cov_matrix, risk_free,
                                            minimize_variance=True, max_weight=max_weight,
                                            min_weight=min_weight)
            mv_weights = mv_res.x
            mv_ret, mv_vol, mv_sharpe = portfolio_stats(mv_weights, mu_series, cov_matrix, risk_free)
            mv_sortino = sortino_ratio(mv_weights, log_returns, risk_free)
            mv_calmar  = calmar_ratio(mv_weights, log_returns, prices, risk_free)
            mv_mdd     = max_drawdown_from_weights(mv_weights, prices)

            fvols, frets = efficient_frontier(mu_series, cov_matrix, risk_free,
                                              n_points=60, max_weight=max_weight)
            mc_vols, mc_rets, mc_sharpes = monte_carlo_portfolios(
                mu_series, cov_matrix, n_portfolios=3000,
                risk_free_rate=risk_free, max_weight=max_weight)  # BUG FIX: max_weight passed

            ms_var  = value_at_risk(ms_weights, log_returns)
            ms_cvar = conditional_var(ms_weights, log_returns)
            mv_var  = value_at_risk(mv_weights, log_returns)
            mv_cvar = conditional_var(mv_weights, log_returns)

            progress.progress(65, text="Classical optimization complete.")
        except Exception as exc:
            error_msg(f"Classical optimization failed: {exc}")
            progress.empty()
            st.stop()

    # ── Step 4: QAOA ─────────────────────────────────────────────────────────
    with st.spinner(f"Running QAOA (p={reps}, {optimizer_name})… 30–120 s"):
        try:
            qaoa_result = run_qaoa(
                tickers=tickers,
                predicted_returns=predicted_returns,
                cov_matrix=cov_matrix,
                risk_aversion=risk_aversion,
                cardinality_k=cardinality_k,
                reps=reps,
                maxiter=200,
                optimizer_name=optimizer_name,
            )
            q_ret, q_var, q_vol = portfolio_metrics_from_selection(
                qaoa_result.selection, predicted_returns, cov_matrix)
            q_sharpe = (q_ret - risk_free) / q_vol if q_vol > 1e-10 else 0.0
            q_score  = q_ret - risk_aversion * q_var

            q_weights = np.zeros(n_stocks)
            sel_idx   = np.where(qaoa_result.selection == 1)[0]
            if len(sel_idx) > 0:
                q_weights[sel_idx] = 1.0 / len(sel_idx)

            q_var_m  = value_at_risk(q_weights, log_returns)  if q_weights.sum() > 0 else 0.0
            q_cvar_m = conditional_var(q_weights, log_returns) if q_weights.sum() > 0 else 0.0
            q_sortino = sortino_ratio(q_weights, log_returns, risk_free) if q_weights.sum() > 0 else 0.0
            q_calmar  = calmar_ratio(q_weights, log_returns, prices, risk_free) if q_weights.sum() > 0 else 0.0
            q_mdd     = max_drawdown_from_weights(q_weights, prices) if q_weights.sum() > 0 else 0.0

            # Brute-force reference (feasible n ≤ 8)
            bf_sel, bf_score = brute_force_optimal(predicted_returns, cov_matrix, risk_aversion)
            # BUG FIX: use correct ratio formula for signed scores
            approx_ratio_val = approximate_ratio(q_score, bf_score)
            bf_tickers_sel   = [tickers[i] for i in bf_sel]

            progress.progress(100, text="Pipeline complete.")
            time.sleep(0.3)
            progress.empty()

            # Persist results in session_state so sidebar interactions don't re-run
            st.session_state["qpl_results"] = {
                "tickers": tickers, "period": period, "n_stocks": n_stocks,
                "prices": prices, "log_returns": log_returns,
                "rolling_vol": rolling_vol, "rolling_sharpe_df": rolling_sharpe_df,
                "cov_matrix": cov_matrix, "mu_series": mu_series,
                "predicted_returns": predicted_returns,
                "history": history, "model_info": model_info, "wf_mode": wf_mode,
                "ms_weights": ms_weights, "ms_ret": ms_ret, "ms_vol": ms_vol,
                "ms_sharpe": ms_sharpe, "ms_sortino": ms_sortino,
                "ms_calmar": ms_calmar, "ms_mdd": ms_mdd,
                "ms_var": ms_var, "ms_cvar": ms_cvar,
                "mv_weights": mv_weights, "mv_ret": mv_ret, "mv_vol": mv_vol,
                "mv_sharpe": mv_sharpe, "mv_sortino": mv_sortino,
                "mv_calmar": mv_calmar, "mv_mdd": mv_mdd,
                "mv_var": mv_var, "mv_cvar": mv_cvar,
                "fvols": fvols, "frets": frets,
                "mc_vols": mc_vols, "mc_rets": mc_rets, "mc_sharpes": mc_sharpes,
                "qaoa_result": qaoa_result,
                "q_weights": q_weights, "q_ret": q_ret, "q_vol": q_vol,
                "q_sharpe": q_sharpe, "q_score": q_score, "q_var": q_var,
                "q_var_m": q_var_m, "q_cvar_m": q_cvar_m,
                "q_sortino": q_sortino, "q_calmar": q_calmar, "q_mdd": q_mdd,
                "bf_sel": bf_sel, "bf_score": bf_score,
                "bf_tickers_sel": bf_tickers_sel,
                "approx_ratio_val": approx_ratio_val,
                "risk_free": risk_free, "risk_aversion": risk_aversion,
                "reps": reps, "optimizer_name": optimizer_name,
                "use_lstm": use_lstm, "n_steps": n_steps, "epochs": epochs,
            }

        except Exception as exc:
            progress.empty()
            error_msg(f"QAOA failed: {exc}")
            if st.checkbox("Show QAOA traceback"):
                st.code(traceback.format_exc())
            st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD FROM SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "qpl_results" not in st.session_state:
    st.stop()

R = st.session_state["qpl_results"]

# Unpack everything
(tickers, period, n_stocks)         = R["tickers"], R["period"], R["n_stocks"]
prices                               = R["prices"]
log_returns                          = R["log_returns"]
rolling_vol                          = R["rolling_vol"]
rolling_sharpe_df                    = R["rolling_sharpe_df"]
cov_matrix                           = R["cov_matrix"]
mu_series                            = R["mu_series"]
predicted_returns                    = R["predicted_returns"]
history, model_info, wf_mode         = R["history"], R["model_info"], R["wf_mode"]
ms_weights, ms_ret, ms_vol           = R["ms_weights"], R["ms_ret"], R["ms_vol"]
ms_sharpe, ms_sortino, ms_calmar     = R["ms_sharpe"], R["ms_sortino"], R["ms_calmar"]
ms_mdd, ms_var, ms_cvar              = R["ms_mdd"], R["ms_var"], R["ms_cvar"]
mv_weights, mv_ret, mv_vol           = R["mv_weights"], R["mv_ret"], R["mv_vol"]
mv_sharpe, mv_sortino, mv_calmar     = R["mv_sharpe"], R["mv_sortino"], R["mv_calmar"]
mv_mdd, mv_var, mv_cvar              = R["mv_mdd"], R["mv_var"], R["mv_cvar"]
fvols, frets                         = R["fvols"], R["frets"]
mc_vols, mc_rets, mc_sharpes         = R["mc_vols"], R["mc_rets"], R["mc_sharpes"]
qaoa_result                          = R["qaoa_result"]
q_weights, q_ret, q_vol              = R["q_weights"], R["q_ret"], R["q_vol"]
q_sharpe, q_score, q_var             = R["q_sharpe"], R["q_score"], R["q_var"]
q_var_m, q_cvar_m                    = R["q_var_m"], R["q_cvar_m"]
q_sortino, q_calmar, q_mdd           = R["q_sortino"], R["q_calmar"], R["q_mdd"]
bf_sel, bf_score                     = R["bf_sel"], R["bf_score"]
bf_tickers_sel, approx_ratio_val     = R["bf_tickers_sel"], R["approx_ratio_val"]
risk_free, risk_aversion             = R["risk_free"], R["risk_aversion"]
reps, optimizer_name                 = R["reps"], R["optimizer_name"]
use_lstm, n_steps, epochs            = R["use_lstm"], R["n_steps"], R["epochs"]


# ══════════════════════════════════════════════════════════════════════════════
#  TOP SUMMARY BANNER
# ══════════════════════════════════════════════════════════════════════════════
selected_display = " · ".join(qaoa_result.selected_tickers) if qaoa_result.selected_tickers else "None"
st.markdown(f"""
<div class="result-banner">
  <div class="headline">Quantum Selection Result (QAOA)</div>
  <div class="portfolio">{selected_display}</div>
  <div class="bits">
    Bitstring: {qaoa_result.best_bitstring}
    &nbsp;|&nbsp; {qaoa_result.n_qubits} qubits
    &nbsp;|&nbsp; Approx ratio: {approx_ratio_val:.3f}
    &nbsp;|&nbsp; Eigenvalue: {qaoa_result.eigenvalue:.6f}
  </div>
</div>
""", unsafe_allow_html=True)

# Key metrics row
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1: st.metric("QAOA Return",  f"{q_ret*100:+.2f}%",
                   delta=f"vs MS {(q_ret-ms_ret)*100:+.2f}%")
with m2: st.metric("QAOA Vol",     f"{q_vol*100:.2f}%")
with m3: st.metric("QAOA Sharpe",  f"{q_sharpe:.3f}",
                   delta=f"vs MS {(q_sharpe-ms_sharpe):+.3f}")
with m4: st.metric("Max Sharpe",   f"{ms_sharpe:.3f}")
with m5: st.metric("QAOA VaR 95%", f"{q_var_m*100:.2f}%")
with m6: st.metric("Approx Ratio", f"{approx_ratio_val:.3f}",
                   delta="optimal" if approx_ratio_val >= 0.999 else None)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABBED RESULTS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "Market Data", "AI Forecasting", "Risk Analysis",
    "Efficient Frontier", "Quantum Optimizer",
    "Comparison", "Performance", "Export",
])
(tab_data, tab_ai, tab_risk, tab_frontier,
 tab_quantum, tab_compare, tab_perf, tab_export) = tabs


# ── Tab: Market Data ──────────────────────────────────────────────────────────
with tab_data:
    section_label("Price performance")
    st.plotly_chart(price_history(prices), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        section_label("Rolling volatility (21-day)")
        st.plotly_chart(rolling_volatility_chart(rolling_vol), use_container_width=True)
    with c2:
        section_label("Rolling Sharpe (63-day)")
        st.plotly_chart(rolling_sharpe_chart(rolling_sharpe_df), use_container_width=True)

    section_label("Return distribution")
    st.plotly_chart(returns_distribution_chart(log_returns), use_container_width=True)

    section_label("Summary statistics")
    stats_df = prices.describe().T
    stats_df.index.name = "Ticker"
    st.dataframe(
        stats_df.style.format("{:.2f}").background_gradient(cmap="Blues", subset=["std"]),
        use_container_width=True,
    )

    section_label("Raw price data (last 15 rows)")
    st.dataframe(prices.tail(15).style.format("{:.2f}"), use_container_width=True)


# ── Tab: AI Forecasting ───────────────────────────────────────────────────────
with tab_ai:
    if use_lstm and history is not None:
        c1, c2 = st.columns(2)
        with c1:
            section_label("Training loss curve")
            st.plotly_chart(lstm_loss_curve(history), use_container_width=True)
        with c2:
            section_label("Model summary")
            final_train   = history.history["loss"][-1]
            final_val     = history.history["val_loss"][-1]
            best_val      = min(history.history["val_loss"])
            actual_epochs = len(history.history["loss"])
            total_params  = model_info.get("total_params", "—")
            cards_html = f"""
            <div class="metric-grid">
                {metric_card("Final Train MSE", f"{final_train:.6f}")}
                {metric_card("Final Val MSE",   f"{final_val:.6f}")}
                {metric_card("Best Val MSE",    f"{best_val:.6f}", variant="cyan")}
                {metric_card("Epochs Run",      str(actual_epochs), sub=f"of {epochs} max")}
                {metric_card("Total Params",    f"{total_params:,}" if isinstance(total_params, int) else total_params)}
                {metric_card("Pred Mode", "Walk-fwd" if wf_mode else "Rolling")}
            </div>"""
            st.markdown(cards_html, unsafe_allow_html=True)
    else:
        info("Historical mean model used — no LSTM training curve available.")

    section_label("Predicted expected returns")
    st.plotly_chart(
        predicted_returns_bar(tickers, predicted_returns, qaoa_result.selected_tickers),
        use_container_width=True,
    )

    section_label("Return estimates")
    ret_df = pd.DataFrame({
        "Ticker":           tickers,
        "Predicted Return": [f"{r:+.6f}" for r in predicted_returns],
        "Historical Mean":  [f"{v:+.6f}" for v in (log_returns.mean() * 252).values],
        "Selected (QAOA)":  ["Yes" if t in qaoa_result.selected_tickers else "No" for t in tickers],
    }).set_index("Ticker")
    st.dataframe(ret_df, use_container_width=True)


# ── Tab: Risk Analysis ────────────────────────────────────────────────────────
with tab_risk:
    c1, c2 = st.columns(2)
    with c1:
        section_label("Covariance matrix")
        st.plotly_chart(covariance_heatmap(cov_matrix, tickers), use_container_width=True)
    with c2:
        section_label("Correlation matrix")
        st.plotly_chart(correlation_heatmap(log_returns), use_container_width=True)

    section_label("Risk metrics comparison")
    radar_labels = ["Return", "Volatility", "Sharpe", "VaR 95%", "CVaR 95%"]
    def _norm(v, mn, mx):
        return float((v - mn) / (mx - mn + 1e-10))
    all_rets    = [ms_ret, mv_ret, q_ret]
    all_vols    = [ms_vol, mv_vol, q_vol]
    all_sharpes = [ms_sharpe, mv_sharpe, q_sharpe]
    all_var     = [ms_var, mv_var, q_var_m]
    all_cvar    = [ms_cvar, mv_cvar, q_cvar_m]
    def norm_vals(vals): return [_norm(v, min(vals), max(vals)) for v in vals]
    ms_radar = [norm_vals(all_rets)[0], 1-norm_vals(all_vols)[0], norm_vals(all_sharpes)[0],
                1-norm_vals(all_var)[0], 1-norm_vals(all_cvar)[0]]
    mv_radar = [norm_vals(all_rets)[1], 1-norm_vals(all_vols)[1], norm_vals(all_sharpes)[1],
                1-norm_vals(all_var)[1], 1-norm_vals(all_cvar)[1]]
    q_radar  = [norm_vals(all_rets)[2], 1-norm_vals(all_vols)[2], norm_vals(all_sharpes)[2],
                1-norm_vals(all_var)[2], 1-norm_vals(all_cvar)[2]]
    st.plotly_chart(risk_radar(radar_labels, ms_radar, mv_radar, q_radar),
                    use_container_width=True)

    section_label("Detailed risk metrics")
    risk_df = pd.DataFrame({
        "Portfolio":    ["Max Sharpe", "Min Variance", "Quantum (QAOA)"],
        "Return (%)":   [ms_ret*100, mv_ret*100, q_ret*100],
        "Vol (%)":      [ms_vol*100, mv_vol*100, q_vol*100],
        "Sharpe":       [ms_sharpe, mv_sharpe, q_sharpe],
        "Sortino":      [ms_sortino, mv_sortino, q_sortino],
        "Calmar":       [ms_calmar, mv_calmar, q_calmar],
        "Max DD (%)":   [ms_mdd*100, mv_mdd*100, q_mdd*100],
        "VaR 95% (%)":  [ms_var*100, mv_var*100, q_var_m*100],
        "CVaR 95% (%)": [ms_cvar*100, mv_cvar*100, q_cvar_m*100],
    })
    st.dataframe(
        risk_df.set_index("Portfolio").style.format("{:.3f}"),
        use_container_width=True,
    )


# ── Tab: Efficient Frontier ────────────────────────────────────────────────────
with tab_frontier:
    section_label("Efficient frontier")
    ef_fig = efficient_frontier_chart(
        frontier_vols   = fvols,
        frontier_rets   = frets,
        ms_stats        = (ms_ret, ms_vol, ms_sharpe),
        mv_stats        = (mv_ret, mv_vol, mv_sharpe),
        tickers         = tickers,
        expected_returns= mu_series,
        cov_matrix      = cov_matrix,
        q_stats         = (q_ret, q_vol) if qaoa_result.selected_tickers else None,
        mc_vols         = mc_vols,
        mc_rets         = mc_rets,
        mc_sharpes      = mc_sharpes,
    )
    st.plotly_chart(ef_fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col_w, name, ret, vol, shr, color in [
        (c1, "Max Sharpe Portfolio",   ms_ret, ms_vol, ms_sharpe, "var(--gold)"),
        (c2, "Min Variance Portfolio", mv_ret, mv_vol, mv_sharpe, "var(--green)"),
        (c3, "Quantum QAOA Portfolio", q_ret,  q_vol,  q_sharpe,  "var(--cyan)"),
    ]:
        with col_w:
            st.markdown(f"""
            <div class="compare-card">
              <div class="name" style="color:{color};">{name}</div>
              <div class="stats">
                <div class="stat-item">
                  <div class="stat-label">Return</div>
                  <div class="stat-value" style="color:{color}">{ret*100:+.2f}%</div>
                </div>
                <div class="stat-item">
                  <div class="stat-label">Vol</div>
                  <div class="stat-value">{vol*100:.2f}%</div>
                </div>
                <div class="stat-item">
                  <div class="stat-label">Sharpe</div>
                  <div class="stat-value">{shr:.3f}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)


# ── Tab: Quantum Optimizer ─────────────────────────────────────────────────────
with tab_quantum:
    section_label("QAOA configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Circuit depth p", reps)
    with c2: st.metric("Qubits", qaoa_result.n_qubits)
    with c3: st.metric("Optimizer", optimizer_name)
    with c4: st.metric("Risk aversion λ", risk_aversion)

    section_label("Measurement distribution")
    # BUG FIX: pass best_raw_bitstring (not best_bitstring) so gold highlight works
    st.plotly_chart(
        bitstring_probability_chart(
            qaoa_result.probabilities,
            qaoa_result.best_raw_bitstring,  # ← FIXED
            tickers,
        ),
        use_container_width=True,
    )

    section_label("Selection result")
    sel_df = pd.DataFrame({
        "Ticker":          tickers,
        "Bit":             list(qaoa_result.best_bitstring[:n_stocks]),
        "Selected":        ["Yes" if b == "1" else "No"
                            for b in qaoa_result.best_bitstring[:n_stocks]],
        "Pred Return":     [f"{r:+.6f}" for r in predicted_returns],
        "Weight (QAOA)":   [f"{w:.4f}" for w in q_weights],
    })
    st.dataframe(sel_df.set_index("Ticker"), use_container_width=True)

    section_label("Brute-force comparison")
    cols_bf = st.columns(3)
    with cols_bf[0]: st.metric("Brute-force Optimal", " · ".join(bf_tickers_sel))
    with cols_bf[1]: st.metric("QAOA Score", f"{q_score:.6f}")
    with cols_bf[2]: st.metric("Optimal Score", f"{bf_score:.6f}")

    if approx_ratio_val >= 0.999:
        info("QAOA recovered the globally optimal binary selection.")
    else:
        warn(
            f"QAOA approximation ratio: {approx_ratio_val:.3f}. "
            "Consider increasing circuit depth p or the iteration budget."
        )


# ── Tab: Comparison ────────────────────────────────────────────────────────────
with tab_compare:
    section_label("Portfolio allocation comparison")
    st.plotly_chart(
        allocation_comparison(tickers, ms_weights, mv_weights, qaoa_result.selection),
        use_container_width=True,
    )

    section_label("Full weight table")
    alloc_df = pd.DataFrame({
        "Ticker":           tickers,
        "Max Sharpe (%)":   [f"{w*100:.2f}" for w in ms_weights],
        "Min Variance (%)": [f"{w*100:.2f}" for w in mv_weights],
        "Quantum QAOA (%)": [f"{w*100:.2f}" for w in q_weights],
    }).set_index("Ticker")
    st.dataframe(alloc_df, use_container_width=True)

    section_label("Performance summary")
    summary_df = pd.DataFrame({
        "Method":       ["Max Sharpe", "Min Variance", "Quantum (QAOA)", "Brute-force Optimal"],
        "Return (%)":   [ms_ret*100, mv_ret*100, q_ret*100, None],
        "Vol (%)":      [ms_vol*100, mv_vol*100, q_vol*100, None],
        "Sharpe":       [ms_sharpe, mv_sharpe, q_sharpe, None],
        "Sortino":      [ms_sortino, mv_sortino, q_sortino, None],
        "Max DD (%)":   [ms_mdd*100, mv_mdd*100, q_mdd*100, None],
        "QUBO Score":   [None, None, q_score, bf_score],
        "Selection":    [
            ", ".join(tickers),
            ", ".join(tickers),
            ", ".join(qaoa_result.selected_tickers),
            ", ".join(bf_tickers_sel),
        ],
    }).set_index("Method")
    st.dataframe(
        summary_df.style.format("{:.3f}", na_rep="—"),
        use_container_width=True,
    )


# ── Tab: Performance ──────────────────────────────────────────────────────────
with tab_perf:
    section_label("Cumulative portfolio growth")
    st.plotly_chart(
        cumulative_returns_chart(prices, ms_weights, mv_weights, q_weights),
        use_container_width=True,
    )

    section_label("Portfolio drawdown")
    st.plotly_chart(
        drawdown_chart(prices, ms_weights, mv_weights, q_weights),
        use_container_width=True,
    )

    section_label("Extended risk metrics")
    ext_df = pd.DataFrame({
        "Portfolio":    ["Max Sharpe", "Min Variance", "Quantum (QAOA)"],
        "Sortino":      [ms_sortino, mv_sortino, q_sortino],
        "Calmar":       [ms_calmar, mv_calmar, q_calmar],
        "Max DD (%)":   [ms_mdd*100, mv_mdd*100, q_mdd*100],
        "VaR 95% (%)":  [ms_var*100, mv_var*100, q_var_m*100],
        "CVaR 95% (%)": [ms_cvar*100, mv_cvar*100, q_cvar_m*100],
    })
    st.dataframe(
        ext_df.set_index("Portfolio").style.format("{:.3f}"),
        use_container_width=True,
    )


# ── Tab: Export ────────────────────────────────────────────────────────────────
with tab_export:
    section_label("Download results")

    export_rows = []
    for i, t in enumerate(tickers):
        export_rows.append({
            "Ticker":           t,
            "Predicted Return": predicted_returns[i],
            "Hist Mean Return": (log_returns.mean() * 252).values[i],
            "Selected (QAOA)":  int(qaoa_result.selection[i]),
            "Weight MaxSharpe": ms_weights[i],
            "Weight MinVar":    mv_weights[i],
            "Weight QAOA":      q_weights[i],
        })
    export_df = pd.DataFrame(export_rows)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download portfolio weights (CSV)",
        data=csv,
        file_name="portfolio_results.csv",
        mime="text/csv",
    )

    # Risk metrics CSV
    risk_export = pd.DataFrame({
        "Portfolio":    ["Max Sharpe", "Min Variance", "Quantum (QAOA)"],
        "Return_pct":   [ms_ret*100, mv_ret*100, q_ret*100],
        "Vol_pct":      [ms_vol*100, mv_vol*100, q_vol*100],
        "Sharpe":       [ms_sharpe, mv_sharpe, q_sharpe],
        "Sortino":      [ms_sortino, mv_sortino, q_sortino],
        "Calmar":       [ms_calmar, mv_calmar, q_calmar],
        "MaxDD_pct":    [ms_mdd*100, mv_mdd*100, q_mdd*100],
        "VaR95_pct":    [ms_var*100, mv_var*100, q_var_m*100],
        "CVaR95_pct":   [ms_cvar*100, mv_cvar*100, q_cvar_m*100],
    })
    st.download_button(
        label="Download risk metrics (CSV)",
        data=risk_export.to_csv(index=False).encode("utf-8"),
        file_name="risk_metrics.csv",
        mime="text/csv",
    )

    # Summary JSON
    summary_json = json.dumps({
        "tickers":              tickers,
        "period":               period,
        "qaoa_selection":       qaoa_result.selected_tickers,
        "qaoa_bitstring":       qaoa_result.best_bitstring,
        "qaoa_eigenvalue":      qaoa_result.eigenvalue,
        "qaoa_return_pct":      round(q_ret * 100, 4),
        "qaoa_vol_pct":         round(q_vol * 100, 4),
        "qaoa_sharpe":          round(q_sharpe, 4),
        "qaoa_sortino":         round(q_sortino, 4),
        "qaoa_calmar":          round(q_calmar, 4),
        "qaoa_max_dd_pct":      round(q_mdd * 100, 4),
        "max_sharpe":           round(ms_sharpe, 4),
        "approx_ratio":         round(approx_ratio_val, 4),
        "risk_aversion_lambda": risk_aversion,
        "qaoa_reps":            reps,
        "optimizer":            optimizer_name,
        "model":                "LSTM" if use_lstm else "HistoricalMean",
    }, indent=2)
    st.download_button(
        label="Download configuration & results (JSON)",
        data=summary_json.encode("utf-8"),
        file_name="qaoa_results.json",
        mime="application/json",
    )

    section_label("Pipeline log")
    with st.expander("Execution log", expanded=False):
        st.code(f"""
Pipeline: Hybrid AI + Quantum Portfolio Optimizer
─────────────────────────────────────────────────
Tickers      : {tickers}
Period       : {period}
Data points  : {len(prices)} trading days

AI Model     : {"LSTM (" + str(n_steps) + "-day window, " + ("walk-fwd)" if wf_mode else "rolling)") if use_lstm else "Historical mean"}
Predicted μ  : {dict(zip(tickers, [f'{r:+.6f}' for r in predicted_returns]))}

QAOA Setup   : depth={reps}, optimizer={optimizer_name}, λ={risk_aversion}
Qubits       : {qaoa_result.n_qubits}
Eigenvalue   : {qaoa_result.eigenvalue:.6f}
Best bits    : {qaoa_result.best_bitstring}
Selection    : {qaoa_result.selected_tickers}
QUBO score   : {q_score:.6f}  (brute-force: {bf_score:.6f})
Approx ratio : {approx_ratio_val:.4f}

Portfolios
  Max Sharpe : ret={ms_ret*100:.2f}%  vol={ms_vol*100:.2f}%  sharpe={ms_sharpe:.3f}  sortino={ms_sortino:.3f}  calmar={ms_calmar:.3f}  mdd={ms_mdd*100:.2f}%
  Min Var    : ret={mv_ret*100:.2f}%  vol={mv_vol*100:.2f}%  sharpe={mv_sharpe:.3f}  sortino={mv_sortino:.3f}  calmar={mv_calmar:.3f}  mdd={mv_mdd*100:.2f}%
  QAOA       : ret={q_ret*100:.2f}%  vol={q_vol*100:.2f}%  sharpe={q_sharpe:.3f}  sortino={q_sortino:.3f}  calmar={q_calmar:.3f}  mdd={q_mdd*100:.2f}%
""")
