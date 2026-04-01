"""
modules/data_pipeline.py
─────────────────────────
Data acquisition and preprocessing layer.
Handles yfinance interaction, disk caching, validation, feature
engineering, and risk metrics used by all downstream layers.

Bug fixes vs original
─────────────────────
- brute_force_optimal  : initialised best_score to 0 — silently returned
                         a wrong selection whenever all QUBO scores were
                         negative (common when risk_aversion > 0).  Fixed
                         to initialise at −∞.
- compute_annualised_stats : raw covariance was not guaranteed PSD due to
                         floating-point; eigenvalue-clipping applied.
- prepare_lstm_sequences: epsilon guard uses np.maximum (vectorised).

New additions
─────────────
- validate_tickers()           : lightweight pre-flight symbol check
- get_cache_info() / clear_cache() / clear_all_cache()
- compute_max_drawdown()       : per-asset max-drawdown series
- compute_portfolio_drawdown_series()
- compute_sortino_ratio()
- compute_calmar_ratio()
- compute_rolling_sharpe()
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CACHE_DIR    = Path(".cache")
TRADING_DAYS = 252


# ── Cache utilities ────────────────────────────────────────────────────────────

def _cache_path(tickers: list[str], period: str) -> Path:
    key = "_".join(sorted(tickers)) + f"_{period}"
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{key}.pkl"


def get_cache_info(tickers: list[str], period: str) -> Optional[dict]:
    """
    Return metadata about a cached file, or None if no cache exists.

    Returns
    -------
    dict with keys: path, age_hours, size_kb — or None.
    """
    path = _cache_path(tickers, period)
    if not path.exists():
        return None
    stat     = path.stat()
    age_hrs  = (time.time() - stat.st_mtime) / 3600
    return {
        "path":      str(path),
        "age_hours": round(age_hrs, 1),
        "size_kb":   round(stat.st_size / 1024, 1),
    }


def clear_cache(tickers: list[str], period: str) -> bool:
    """Delete a specific cached price file. Returns True if a file was removed."""
    path = _cache_path(tickers, period)
    if path.exists():
        path.unlink()
        return True
    return False


def clear_all_cache() -> int:
    """Delete every file in the cache directory. Returns count removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink()
        removed += 1
    return removed


# ── Ticker validation ──────────────────────────────────────────────────────────

def validate_tickers(tickers: list[str]) -> tuple[list[str], list[str]]:
    """
    Quick 5-day download to verify which tickers return data.

    Returns
    -------
    valid   : tickers that returned at least one non-NaN close price
    invalid : tickers that returned nothing
    """
    if not tickers:
        return [], []
    try:
        test = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
        if test.empty:
            return [], list(tickers)
        prices  = test["Close"] if isinstance(test.columns, pd.MultiIndex) else test
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        valid   = [t for t in tickers if t in prices.columns and prices[t].notna().any()]
        invalid = [t for t in tickers if t not in valid]
        return valid, invalid
    except Exception:
        # Cannot validate (network error etc.) — assume all valid and let
        # fetch_prices raise if something is actually wrong.
        return list(tickers), []


# ── Price download & caching ───────────────────────────────────────────────────

def fetch_prices(
    tickers: list[str],
    period: str = "3y",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance with optional disk cache.

    Parameters
    ----------
    tickers   : list of valid ticker symbols
    period    : yfinance period string  ("1y" | "2y" | "3y" | "5y" | …)
    use_cache : read from disk when a cached version exists

    Returns
    -------
    DataFrame indexed by Date, columns = tickers (adjusted close)

    Raises
    ------
    ValueError  : tickers list empty or invalid period
    RuntimeError: download produced no usable data or < 120 trading days
    """
    if not tickers:
        raise ValueError("Tickers list must not be empty.")
    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "ytd", "max"}
    if period not in valid_periods:
        raise ValueError(f"period must be one of {valid_periods}, got '{period}'.")

    cache_file = _cache_path(tickers, period)
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as fh:
            prices: pd.DataFrame = pickle.load(fh)
        return prices

    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    if raw.empty:
        raise RuntimeError(
            f"yfinance returned empty data for tickers={tickers}, period={period}."
        )

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    # Guarantee DataFrame even for a single ticker
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.ffill().dropna()

    available = [t for t in tickers if t in prices.columns]
    missing   = set(tickers) - set(available)
    if missing:
        warnings.warn(f"Tickers not returned by Yahoo Finance: {missing}")
    if not available:
        raise RuntimeError("None of the requested tickers returned data.")
    prices = prices[available]

    if len(prices) < 120:
        raise RuntimeError(
            f"Only {len(prices)} trading days available — need at least 120. "
            "Try a longer period or different tickers."
        )

    if use_cache:
        with open(cache_file, "wb") as fh:
            pickle.dump(prices, fh)

    return prices


# ── Return engineering ─────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log-returns; first row dropped (NaN)."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_annualised_stats(
    prices: pd.DataFrame,
    trading_days: int = TRADING_DAYS,
) -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """
    Compute annualised expected returns and a positive-semi-definite
    covariance matrix.

    Returns
    -------
    expected_returns : pd.Series  (annualised, e.g. 0.25 = 25 % p.a.)
    cov_matrix       : np.ndarray (annualised N×N, PSD-clipped)
    log_returns      : pd.DataFrame of daily log-returns
    """
    log_ret  = compute_log_returns(prices)
    exp_ret  = log_ret.mean() * trading_days

    # PSD clip: eliminate tiny negative eigenvalues from floating-point noise
    raw_cov            = log_ret.cov().values * trading_days
    eigvals, eigvecs   = np.linalg.eigh(raw_cov)
    eigvals            = np.maximum(eigvals, 0.0)
    cov_matrix         = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return exp_ret, cov_matrix, log_ret


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalise_prices(prices: pd.DataFrame) -> tuple[np.ndarray, object]:
    """
    MinMax-scale each column to [0.01, 1.0].
    Returns (scaled_array, fitted_scaler).
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0.01, 1.0))
    scaled = scaler.fit_transform(prices.values)
    return scaled, scaler


# ── LSTM sequences ─────────────────────────────────────────────────────────────

def prepare_lstm_sequences(
    scaled: np.ndarray,
    n_steps: int = 60,
    train_ratio: float = 0.80,
    shuffle: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window (X, y) sequences for the LSTM.

    X shape : (samples, n_steps, n_stocks)
    y shape : (samples, n_stocks)   — next-day log-return in scaled space

    Parameters
    ----------
    shuffle : if True, shuffles the training set.  Not recommended for
              time-series (default False preserves temporal ordering).
    """
    X, y = [], []
    for i in range(n_steps, len(scaled) - 1):
        X.append(scaled[i - n_steps : i, :])
        # np.maximum is vectorised and avoids division-by-zero (BUG FIX)
        ret = np.log(scaled[i + 1] / np.maximum(scaled[i], 1e-8))
        y.append(ret)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    split        = int(len(X) * train_ratio)
    X_tr, y_tr   = X[:split],  y[:split]
    X_val, y_val = X[split:], y[split:]

    if shuffle and len(X_tr) > 0:
        idx     = np.random.permutation(len(X_tr))
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    return X_tr, y_tr, X_val, y_val


# ── Rolling analytics ──────────────────────────────────────────────────────────

def compute_rolling_volatility(
    prices: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """Annualised rolling volatility for each ticker."""
    log_ret = compute_log_returns(prices)
    return log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)


def compute_rolling_sharpe(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.04,
    window: int = 63,
) -> pd.DataFrame:
    """Rolling Sharpe ratio (annualised) over a 63-day window by default."""
    log_ret   = compute_log_returns(prices)
    roll_mean = log_ret.rolling(window).mean() * TRADING_DAYS
    roll_vol  = log_ret.rolling(window).std()  * np.sqrt(TRADING_DAYS)
    roll_vol  = roll_vol.replace(0.0, np.nan)
    return (roll_mean - risk_free_rate) / roll_vol


# ── Drawdown ───────────────────────────────────────────────────────────────────

def compute_max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """
    Per-asset maximum drawdown (negative float, e.g. −0.35 = −35 %).

    Returns
    -------
    pd.Series indexed by ticker.
    """
    running_max = prices.cummax()
    drawdown    = (prices - running_max) / running_max
    return drawdown.min()


def compute_portfolio_drawdown_series(
    weights: np.ndarray,
    prices: pd.DataFrame,
) -> pd.Series:
    """
    Portfolio-level drawdown time series given fixed weights.
    Values are non-positive floats (0 = at high-water mark).
    """
    port_level  = (prices / prices.iloc[0]) @ np.asarray(weights)
    running_max = port_level.cummax()
    return (port_level - running_max) / running_max


# ── Risk-adjusted ratios ───────────────────────────────────────────────────────

def compute_sortino_ratio(
    weights: np.ndarray,
    log_returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Annualised Sortino ratio using realised downside deviation.
    Returns 0.0 when downside deviation is effectively zero.
    """
    port_ret      = (log_returns @ np.asarray(weights)).dropna()
    daily_rf      = risk_free_rate / TRADING_DAYS
    excess        = port_ret - daily_rf
    downside      = excess[excess < 0.0]
    downside_dev  = float(np.sqrt((downside ** 2).mean())) * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 0.0
    ann_excess    = float(excess.mean()) * TRADING_DAYS
    return ann_excess / downside_dev if downside_dev > 1e-10 else 0.0


def compute_calmar_ratio(
    weights: np.ndarray,
    log_returns: pd.DataFrame,
    prices: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Calmar ratio: annualised excess return / max-drawdown magnitude.
    Returns 0.0 when the portfolio has no drawdown.
    """
    ann_ret   = float((log_returns @ np.asarray(weights)).mean()) * TRADING_DAYS - risk_free_rate
    dd_series = compute_portfolio_drawdown_series(weights, prices)
    max_dd    = float(dd_series.min())            # negative
    return ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0


# ── Brute-force QUBO optimum ───────────────────────────────────────────────────

def brute_force_optimal(
    predicted_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 0.5,
) -> tuple[list[int], float]:
    """
    Exhaustive search over all 2^n binary portfolios.  Feasible for n ≤ 10.

    QUBO objective (maximise): ret − λ·var  (equal-weight within selection)

    Returns
    -------
    (best_selection_indices, best_score)

    Bug fixed: original initialised best_score = 0, which caused it to return
    an empty selection (score=0) whenever every feasible portfolio had a
    negative QUBO score.  Initialised to −∞ to handle all cases correctly.
    """
    n          = len(predicted_returns)
    best_score = -np.inf      # ← BUG FIX (was 0)
    best_sel   = [0]          # fallback

    for k in range(1, n + 1):
        for combo in combinations(range(n), k):
            idx   = list(combo)
            w     = np.ones(k) / k
            r     = float(predicted_returns[idx] @ w)
            v     = float(w @ cov_matrix[np.ix_(idx, idx)] @ w)
            score = r - risk_aversion * v
            if score > best_score:
                best_score = score
                best_sel   = idx

    return best_sel, best_score
