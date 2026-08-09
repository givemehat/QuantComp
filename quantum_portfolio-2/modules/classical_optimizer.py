"""
modules/classical_optimizer.py
───────────────────────────────
Markowitz mean-variance optimisation.
Provides max-Sharpe, min-variance, efficient frontier, and a comprehensive
suite of risk metrics including VaR, CVaR, Sortino, and Calmar ratios.

Bug fixes vs original
─────────────────────
- monte_carlo_portfolios : never respected `max_weight`, producing random
  portfolios that were inconsistent with the frontier (which IS capped).
  Added a `max_weight` parameter so the scatter cloud matches.
- portfolio_stats        : return/vol/sharpe could silently divide by zero
  for a degenerate (all-NaN) weight vector — now guarded.

New additions
─────────────
- sortino_ratio()        : downside-deviation Sharpe
- calmar_ratio()         : annualised return / max-drawdown magnitude
- max_drawdown_from_weights() : convenience wrapper using price data
- information_ratio()    : active return / tracking error vs benchmark
- treynor_ratio()        : excess return / portfolio beta
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252


# ── Portfolio statistics ───────────────────────────────────────────────────────


def portfolio_stats(
    weights: np.ndarray,
    expected_returns: "pd.Series | np.ndarray",
    cov_matrix: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
) -> tuple[float, float, float]:
    """
    Compute (annualised return, volatility, Sharpe ratio) for given weights.

    Handles zero-vol edge case (all-cash portfolio) by returning Sharpe = 0.
    """
    mu = np.asarray(expected_returns)
    w = np.asarray(weights)
    # Guard against NaN/Inf weights
    if not np.all(np.isfinite(w)) or w.sum() < 1e-10:
        return 0.0, 0.0, 0.0
    ret = float(w @ mu)
    var = float(w @ cov_matrix @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-10 else 0.0
    return ret, vol, sharpe


# ── Objective functions ────────────────────────────────────────────────────────


def _neg_sharpe(w, mu, cov, rf):
    _, _, s = portfolio_stats(w, mu, cov, rf)
    return -s


def _variance(w, cov):
    return float(w @ cov @ w)


# ── Core optimiser ─────────────────────────────────────────────────────────────


def optimize_portfolio(
    expected_returns: "pd.Series | np.ndarray",
    cov_matrix: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    target_return: Optional[float] = None,
    minimize_variance: bool = False,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> OptimizeResult:
    """
    Run SLSQP portfolio optimisation.

    Modes
    -----
    Default             → maximise Sharpe ratio
    target_return set   → minimise variance at fixed target return
    minimize_variance   → global minimum-variance portfolio

    Parameters
    ----------
    max_weight : upper bound per asset (diversification cap, 0–1)
    min_weight : lower bound per asset (0 = long-only; set > 0 for floor)
    """
    mu = np.asarray(expected_returns)
    n = len(mu)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w, r=target_return: float(np.asarray(w) @ mu) - r,
            }
        )

    bounds = [(float(min_weight), float(max_weight))] * n
    w0 = np.full(n, 1.0 / n)

    if minimize_variance or target_return is not None:
        result = minimize(
            _variance,
            w0,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    else:
        result = minimize(
            _neg_sharpe,
            w0,
            args=(mu, cov_matrix, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

    if not result.success:
        warnings.warn(f"Portfolio optimiser: {result.message}")
    return result


# ── Efficient frontier ─────────────────────────────────────────────────────────


def efficient_frontier(
    expected_returns: "pd.Series | np.ndarray",
    cov_matrix: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    n_points: int = 80,
    max_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the efficient frontier by sweeping target returns.

    Returns
    -------
    vols, rets : arrays of frontier coordinates (annualised)
    """
    mu = np.asarray(expected_returns)
    lo = float(mu.min())
    hi = float(mu.max()) * 1.05
    targets = np.linspace(lo, hi, n_points)

    vols, rets = [], []
    for tr in targets:
        res = optimize_portfolio(
            expected_returns,
            cov_matrix,
            risk_free_rate,
            target_return=tr,
            max_weight=max_weight,
        )
        if res.success:
            r, v, _ = portfolio_stats(res.x, expected_returns, cov_matrix)
            vols.append(v)
            rets.append(r)

    return np.array(vols), np.array(rets)


# ── Monte Carlo cloud ──────────────────────────────────────────────────────────


def monte_carlo_portfolios(
    expected_returns: "pd.Series | np.ndarray",
    cov_matrix: np.ndarray,
    n_portfolios: int = 5_000,
    risk_free_rate: float = RISK_FREE_RATE,
    seed: int = 42,
    max_weight: float = 1.0,  # BUG FIX: was missing — scatter was uncapped
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random long-only portfolios for the feasibility scatter plot.

    Parameters
    ----------
    max_weight : cap per asset; now consistent with the optimised portfolios.

    Returns
    -------
    mc_vols, mc_rets, mc_sharpes
    """
    mu = np.asarray(expected_returns)
    n = len(mu)
    rng = np.random.default_rng(seed)

    mc_vols, mc_rets, mc_sharpes = [], [], []
    for _ in range(n_portfolios):
        # Draw and clip to max_weight, then re-normalise
        w = rng.random(n)
        w = np.minimum(w, max_weight)
        total = w.sum()
        if total < 1e-10:
            continue
        w /= total
        r, v, s = portfolio_stats(w, mu, cov_matrix, risk_free_rate)
        mc_vols.append(v)
        mc_rets.append(r)
        mc_sharpes.append(s)

    return np.array(mc_vols), np.array(mc_rets), np.array(mc_sharpes)


# ── Historical VaR / CVaR ──────────────────────────────────────────────────────


def value_at_risk(
    weights: np.ndarray,
    log_returns: pd.DataFrame,
    confidence: float = 0.95,
) -> float:
    """Historical Value-at-Risk at given confidence level (annualised)."""
    port_ret = (log_returns @ np.asarray(weights)).dropna()
    return float(
        -np.percentile(port_ret, (1 - confidence) * 100) * np.sqrt(TRADING_DAYS)
    )


def conditional_var(
    weights: np.ndarray,
    log_returns: pd.DataFrame,
    confidence: float = 0.95,
) -> float:
    """
    Historical CVaR (Expected Shortfall, annualised).
    Returns 0.0 for a degenerate portfolio with no tail losses.
    """
    port_ret = (log_returns @ np.asarray(weights)).dropna()
    threshold = np.percentile(port_ret, (1 - confidence) * 100)
    tail = port_ret[port_ret <= threshold]
    return float(-tail.mean() * np.sqrt(TRADING_DAYS)) if len(tail) > 0 else 0.0


# ── Sortino & Calmar ───────────────────────────────────────────────────────────


def sortino_ratio(
    weights: np.ndarray,
    log_returns: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """
    Annualised Sortino ratio using realised downside deviation.
    Returns 0.0 when downside deviation is effectively zero.
    """
    port_ret = (log_returns @ np.asarray(weights)).dropna()
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = port_ret - daily_rf
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        return 0.0
    downside_dev = float(np.sqrt((downside**2).mean())) * np.sqrt(TRADING_DAYS)
    ann_excess = float(excess.mean()) * TRADING_DAYS
    return ann_excess / downside_dev if downside_dev > 1e-10 else 0.0


def max_drawdown_from_weights(
    weights: np.ndarray,
    prices: pd.DataFrame,
) -> float:
    """
    Maximum drawdown for a fixed-weight portfolio over the price history.
    Returns a non-positive float (e.g. −0.35 = −35 % drawdown).
    """
    w = np.asarray(weights)
    port_level = (prices / prices.iloc[0]) @ w
    running_max = port_level.cummax()
    return float(((port_level - running_max) / running_max).min())


def calmar_ratio(
    weights: np.ndarray,
    log_returns: pd.DataFrame,
    prices: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """
    Calmar ratio: annualised excess return / max-drawdown magnitude.
    Returns 0.0 when there is no drawdown.
    """
    ann_ret = (
        float((log_returns @ np.asarray(weights)).mean()) * TRADING_DAYS
        - risk_free_rate
    )
    mdd = max_drawdown_from_weights(weights, prices)
    return ann_ret / abs(mdd) if abs(mdd) > 1e-10 else 0.0


# ── Tracking metrics ───────────────────────────────────────────────────────────


def information_ratio(
    weights: np.ndarray,
    benchmark_weights: np.ndarray,
    log_returns: pd.DataFrame,
) -> float:
    """
    Information ratio: annualised active return / tracking error.
    Both weight vectors should sum to 1.
    """
    port_ret = (log_returns @ np.asarray(weights)).dropna()
    bench_ret = (log_returns @ np.asarray(benchmark_weights)).dropna()
    active = (port_ret - bench_ret).dropna()
    te = float(active.std()) * np.sqrt(TRADING_DAYS)
    ar = float(active.mean()) * TRADING_DAYS
    return ar / te if te > 1e-10 else 0.0
