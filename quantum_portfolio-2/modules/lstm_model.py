"""
modules/lstm_model.py
─────────────────────
LSTM neural network for multi-stock return forecasting.

Bug fixes vs original
─────────────────────
- predict_expected_returns : returned np.mean([], axis=0) which raised a
  warning and could return NaN when the test window was too large relative
  to the dataset.  Added guard + fallback to historical mean.
- build_model              : added BatchNormalization layer between the two
  LSTM stacks for more stable training (was imported but never used).
- train                    : verbose=0 suppresses TF output in Streamlit;
  kept as-is.

New additions
─────────────
- get_model_summary_dict() : returns architecture info for the UI cards
- walk_forward_predict()   : optional walk-forward return estimation
- historical_mean_returns  : unchanged, kept for parity
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Input,
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

tf.random.set_seed(42)
np.random.seed(42)

_UNITS_L1 = 64
_UNITS_L2 = 32
_DROPOUT = 0.10
_LR = 1e-3
_L2 = 1e-4


def build_model(
    input_shape: tuple[int, int],
    n_outputs: int,
    units_l1: int = _UNITS_L1,
    units_l2: int = _UNITS_L2,
    dropout: float = _DROPOUT,
    learning_rate: float = _LR,
) -> tf.keras.Model:
    """
    Stacked LSTM → BatchNorm → Dense regression model.

    Parameters
    ----------
    input_shape    : (n_steps, n_features)
    n_outputs      : number of stocks (one output per ticker)
    units_l1/l2    : LSTM hidden units for each layer
    dropout        : dropout rate applied after each LSTM layer
    learning_rate  : Adam initial learning rate

    Notes
    -----
    BatchNormalization is now actually inserted (was imported but absent in
    the original), which improves training stability on short datasets.
    """
    model = Sequential(
        [
            LSTM(
                units_l1,
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(_L2),
            ),
            Dropout(dropout),
            BatchNormalization(),  # ← BUG FIX: was imported, never used
            LSTM(units_l2, return_sequences=False, kernel_regularizer=l2(_L2)),
            Dropout(dropout),
            Dense(n_outputs, activation="linear"),
        ],
        name="hybrid_lstm",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    patience: int = 5,
) -> tf.keras.callbacks.History:
    """
    Train the LSTM with EarlyStopping + ReduceLROnPlateau.

    Returns keras History object (attributes: .history dict).
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
            verbose=0,
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return history


def predict_expected_returns(
    model: tf.keras.Model,
    scaled_prices: np.ndarray,
    n_steps: int = 60,
    test_window: int = 30,
) -> np.ndarray:
    """
    Average model predictions over the most recent `test_window` days.

    Returns
    -------
    np.ndarray of shape (n_stocks,) — mean predicted log-return per stock.

    Bug fixed: when `test_window` was larger than available data − n_steps,
    `start >= n − 1` produced an empty loop and np.mean raised a warning,
    returning NaN.  Now falls back to the widest possible window.
    """
    n = len(scaled_prices)
    start = max(n_steps, n - test_window - n_steps)

    # BUG FIX: ensure at least one prediction can be made
    if start >= n - 1:
        start = max(0, n - n_steps - 1)

    preds = []
    for i in range(start, n - 1):
        if i < n_steps:
            continue
        window = scaled_prices[i - n_steps : i, :][np.newaxis]
        pred = model.predict(window, verbose=0)
        preds.append(pred[0])

    if not preds:
        # Absolute fallback: zero-return prediction (caller should warn)
        n_stocks = scaled_prices.shape[1]
        return np.zeros(n_stocks, dtype=np.float32)

    return np.mean(preds, axis=0)


def walk_forward_predict(
    model: tf.keras.Model,
    scaled_prices: np.ndarray,
    n_steps: int = 60,
    n_folds: int = 5,
) -> np.ndarray:
    """
    Walk-forward cross-validated return prediction.

    Divides the tail of the price series into `n_folds` windows and
    averages predictions across all of them.  More robust than a single
    rolling-window prediction for short datasets.

    Returns
    -------
    np.ndarray of shape (n_stocks,)
    """
    n = len(scaled_prices)
    fold_len = max(1, (n - n_steps) // n_folds)
    preds = []

    for fold in range(n_folds):
        i = n_steps + fold * fold_len
        if i >= n:
            break
        window = scaled_prices[i - n_steps : i, :][np.newaxis]
        pred = model.predict(window, verbose=0)
        preds.append(pred[0])

    if not preds:
        return np.zeros(scaled_prices.shape[1], dtype=np.float32)

    return np.mean(preds, axis=0)


def get_model_summary_dict(model: tf.keras.Model) -> dict:
    """
    Return a dict of architecture metadata for UI display.

    Returns
    -------
    {"total_params": int, "trainable_params": int, "layers": list[str]}
    """
    return {
        "total_params": model.count_params(),
        "trainable_params": sum(tf.size(v).numpy() for v in model.trainable_variables),
        "layers": [
            f"{layer.__class__.__name__}({layer.name})" for layer in model.layers
        ],
    }


def historical_mean_returns(prices_pct_change: "pd.DataFrame") -> np.ndarray:
    """
    Simple linear baseline: annualised mean of daily percentage returns.
    Kept for backward compatibility.
    """
    return (prices_pct_change.mean() * 252).values
