# Quantum Portfolio Lab

**Hybrid AI + Quantum Portfolio Optimizer — v2.0 (patched & enhanced)**  
LSTM return forecasting · QAOA combinatorial selection · Markowitz classical benchmark

---

## Overview

A research-grade Streamlit dashboard that chains three layers:

1. **Data Layer** — Yahoo Finance prices via `yfinance`, annualised returns and PSD-clamped covariance
2. **AI Layer** — Stacked LSTM neural network forecasts expected returns per stock
3. **Quantum Layer** — QAOA (via Qiskit) solves the binary portfolio selection QUBO

The classical Markowitz mean-variance solution runs in parallel as a benchmark.

---

## Installation

```bash
# 1. Clone / unzip the project
cd quantum_portfolio

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

---

## Project Structure

```
quantum_portfolio/
├── app.py                          # Main Streamlit dashboard
├── modules/
│   ├── __init__.py
│   ├── data_pipeline.py            # Data fetch, normalisation, sequence prep
│   ├── lstm_model.py               # LSTM build, train, predict
│   ├── classical_optimizer.py      # Markowitz, efficient frontier, risk metrics
│   ├── quantum_optimizer.py        # QUBO formulation, QAOA execution
│   └── visualization.py            # Plotly dark-theme chart library
├── .cache/                         # Auto-created price cache (gitignore)
└── README.md
```

---

## Bug Fixes (v2.0)

### Critical

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `data_pipeline.py` | `brute_force_optimal` initialised `best_score = 0`, causing a wrong (empty) selection whenever every feasible QUBO portfolio had a negative score — common when `risk_aversion > 0`. | Initialised to `−∞`. |
| 2 | `quantum_optimizer.py` | `_extract_probabilities` only handled two eigenstate formats; Qiskit 1.x `StatevectorSampler` returns a `DataBin`/`BitArray` V2 result that was silently falling through to an empty dict, making all charts blank and the selection random. | Added handlers for `DataBin`, `BitArray`, `quasi_dists`, and amplitude dicts; added normalisation pass. |
| 3 | `quantum_optimizer.py` | `_get_sampler` hardcoded `StatevectorSampler` (Qiskit 1.x only) — older installs raised `ImportError` at runtime. | Try V2 `StatevectorSampler` first, fall back to V1 `Sampler`. |
| 4 | `app.py` | Approximation ratio computed as `q_score / bf_score`, which is **wrong** when both scores are negative (produces ratio > 1, implying QAOA beat the optimum). | Replaced with `approximate_ratio()` helper that handles all sign combinations correctly. |
| 5 | `visualization.py` | `bitstring_probability_chart` compared `best_bitstring` (decoded, reversed, `n_stocks` chars) against raw probability-dict keys (`n_qubits` chars, unreversed). The gold highlight **never** fired. | `QAOAResult` now stores `best_raw_bitstring`; chart receives the raw key for a direct comparison. |
| 6 | `lstm_model.py` | `predict_expected_returns` returned `np.mean([], axis=0)` → `NaN` when `test_window` was too large relative to the dataset length. | Guard added; falls back to the widest possible window before returning zeros. |
| 7 | `classical_optimizer.py` | `monte_carlo_portfolios` never accepted `max_weight`, so the random-portfolio scatter cloud was unconstrained while the frontier was capped — inconsistent visual. | Added `max_weight` parameter; weights are clipped and re-normalised. |

### Minor

| # | File | Bug | Fix |
|---|------|-----|-----|
| 8 | `lstm_model.py` | `BatchNormalization` was imported but never inserted into the model. | Added between the two LSTM layers for more stable training. |
| 9 | `data_pipeline.py` | `compute_annualised_stats` returned a raw covariance matrix that was not guaranteed positive-semi-definite due to floating-point noise; downstream SLSQP could fail silently. | Eigenvalue-clipping applied (`max(λ, 0)`). |
| 10 | `visualization.py` | `risk_radar` set `opacity=0.15` on the trace, which also faded the **border line** to near-invisible. | Moved alpha into `fillcolor` as `rgba(r,g,b,0.15)`; lines stay at full opacity. |
| 11 | `data_pipeline.py` | `prepare_lstm_sequences` used `1e-8` as a scalar guard, relying on Python broadcasting which can silently misfire for rank-1 arrays. | Replaced with `np.maximum(scaled[i], 1e-8)` (vectorised). |
| 12 | `app.py` | Single-ticker downloads returned a `pd.Series`; accessing `.columns` raised `AttributeError`. | `fetch_prices` converts Series → DataFrame before processing. |

---

## New Features (v2.0)

### Sidebar
- **Custom ticker input** — type any valid Yahoo Finance symbol, not just the preset list
- **Min weight constraint** — add a floor per asset alongside the existing cap
- **Cache controls** — "Clear cache" and "Refresh data" buttons with age/size display
- **ADAM optimizer** option for QAOA (alongside COBYLA and SPSA)
- **Walk-forward prediction** mode for LSTM (averages across multiple windows)
- **Session state persistence** — results survive sidebar interaction without re-running the pipeline

### Metrics
- **Sortino ratio** — downside-deviation-adjusted return
- **Calmar ratio** — annualised return / max-drawdown magnitude
- **Maximum drawdown** — per-portfolio time-series and scalar
- All new metrics visible in the Risk Analysis tab, Comparison tab, and exports

### Charts
- `cumulative_returns_chart` — growth-of-$1 for all three strategies
- `drawdown_chart` — portfolio drawdown time-series
- `returns_distribution_chart` — daily return histogram overlay
- `rolling_sharpe_chart` — 63-day rolling Sharpe per ticker

### New tab
- **Performance** tab — dedicated to cumulative growth, drawdown, and extended risk metrics

### New utility functions
| Function | Module | Purpose |
|---|---|---|
| `validate_tickers()` | `data_pipeline` | Pre-flight symbol check before download |
| `get_cache_info()` | `data_pipeline` | Returns age and size of cached file |
| `clear_cache()` / `clear_all_cache()` | `data_pipeline` | Programmatic cache invalidation |
| `compute_rolling_sharpe()` | `data_pipeline` | 63-day rolling Sharpe per ticker |
| `compute_max_drawdown()` | `data_pipeline` | Per-asset max drawdown |
| `compute_portfolio_drawdown_series()` | `data_pipeline` | Portfolio drawdown time-series |
| `compute_sortino_ratio()` | `data_pipeline` | Downside Sharpe |
| `compute_calmar_ratio()` | `data_pipeline` | Return / drawdown ratio |
| `sortino_ratio()` | `classical_optimizer` | Sortino for optimised weight vectors |
| `calmar_ratio()` | `classical_optimizer` | Calmar for optimised weight vectors |
| `max_drawdown_from_weights()` | `classical_optimizer` | MDD from fixed-weight portfolio |
| `information_ratio()` | `classical_optimizer` | Active return / tracking error |
| `approximate_ratio()` | `quantum_optimizer` | Correct QAOA approximation ratio |
| `walk_forward_predict()` | `lstm_model` | Multi-fold walk-forward return estimate |
| `get_model_summary_dict()` | `lstm_model` | Architecture metadata for UI |
| `cumulative_returns_chart()` | `visualization` | Growth-of-$1 chart |
| `drawdown_chart()` | `visualization` | Portfolio drawdown time-series |
| `returns_distribution_chart()` | `visualization` | Return histogram overlay |
| `rolling_sharpe_chart()` | `visualization` | Rolling Sharpe per ticker |

---

## Configuration

All parameters are exposed in the sidebar:

| Parameter | Description |
|---|---|
| Tickers (preset) | Select from 20 presets |
| Custom tickers | Type any Yahoo Finance symbol |
| Period | Historical data window |
| Use disk cache | Toggle caching; shows age and file size |
| Forecasting model | LSTM neural network or historical mean |
| Look-back window | LSTM input sequence length |
| Training epochs | LSTM training budget |
| Early stopping patience | Epochs without improvement before stopping |
| Walk-forward prediction | Average across multiple LSTM windows |
| Risk-free rate | Annualised rate for Sharpe |
| Max weight | Diversification cap per asset |
| Min weight | Floor per asset (long-only with minimum exposure) |
| Circuit depth p | QAOA approximation quality |
| Classical optimizer | COBYLA / SPSA / ADAM |
| Risk aversion λ | QUBO risk penalty |
| Cardinality k | Force selection of exactly k stocks |

---

## Limitations

- LSTM may overfit on small datasets; walk-forward mode mitigates this
- QAOA at depth p=1 on 4 stocks does not demonstrate quantum advantage (2^4 = 16 states)
- Noiseless simulator — real hardware introduces gate errors and decoherence
- Pipeline is a research proof-of-concept, not a trading system

---

## References

1. Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*
2. Farhi, E., Goldstone, J., Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm. *arXiv:1411.4028*
3. Hochreiter, S., Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*
4. Egger, D.J. et al. (2020). Quantum Computing for Finance. *IEEE Trans. Quantum Engineering*
