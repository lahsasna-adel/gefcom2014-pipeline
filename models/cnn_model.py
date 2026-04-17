"""
models/cnn_model.py
-------------------
1D CNN forecaster with dilated causal convolutions for time-series forecasting.

Architecture:
  Input (lookback=48, 1)
  → Conv1D(32, kernel=3, dilation=1, causal padding, relu)
  → Conv1D(32, kernel=3, dilation=2, causal padding, relu)
  → Conv1D(64, kernel=3, dilation=4, causal padding, relu)
  → GlobalAveragePooling1D
  → Dense(32, relu)
  → Dropout(0.2)
  → Dense(1)

Design rationale
----------------
Causal padding:
  Output at time t uses ONLY inputs at times ≤ t — no future leakage
  into the convolutional receptive field.

Dilated convolutions:
  Exponentially growing receptive fields without proportional parameter cost.
    dilation=1 → covers  3h of context
    dilation=2 → covers  5h of context
    dilation=4 → covers  9h of context
  Combined with lookback=48h, this captures intra-day patterns (dilation=1),
  inter-peak patterns (dilation=2), and near-daily cycles (dilation=4).

GlobalAveragePooling:
  Collapses the time dimension robustly — more stable than Flatten for
  variable-length inputs and less prone to overfitting than GlobalMaxPooling.

No recurrence:
  Fully parallelisable across time → trains 3–5× faster than LSTM/GRU on GPU,
  and 2–3× faster on CPU for the batch sizes used here.

References:
  Borovykh et al. (2017) — Conditional Time Series Forecasting with CNN
  Bai et al. (2018)      — An Empirical Evaluation of Generic Convolutional
                           and Recurrent Networks for Sequence Modeling
"""

import numpy as np
import time


class CNNForecaster:
    """
    Univariate 1D-CNN forecaster with dilated causal convolutions.

    Parameters
    ----------
    lookback : int   — input window (hours), default 48
    horizon  : int   — steps ahead, default 24
    epochs   : int   — max epochs
    patience : int   — early stopping patience
    filters  : int   — base number of conv filters (default 32)
    """

    def __init__(self, lookback=48, horizon=24, epochs=30,
                 patience=5, filters=32):
        self.lookback    = lookback
        self.horizon     = horizon
        self.epochs      = epochs
        self.patience    = patience
        self.filters     = filters
        self.model_      = None
        self.train_time_ = 0.0
        self._mean       = 0.0
        self._std        = 1.0

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_sequences(self, series):
        """Slide a window over series → (X, y) arrays for training."""
        X, y = [], []
        for i in range(self.lookback, len(series) - self.horizon + 1):
            X.append(series[i - self.lookback: i])
            y.append(series[i + self.horizon - 1])
        return np.array(X)[..., np.newaxis], np.array(y)

    def _build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Conv1D, GlobalAveragePooling1D,
                                              Dense, Dropout)

        f = self.filters
        model = Sequential([
            # Layer 1 — fine-grained local patterns (receptive field: 3h)
            Conv1D(f, kernel_size=3, dilation_rate=1,
                   padding="causal", activation="relu",
                   input_shape=(self.lookback, 1)),

            # Layer 2 — medium-range dependencies (receptive field: 5h)
            Conv1D(f, kernel_size=3, dilation_rate=2,
                   padding="causal", activation="relu"),

            # Layer 3 — longer-range dependencies (receptive field: 9h)
            Conv1D(f * 2, kernel_size=3, dilation_rate=4,
                   padding="causal", activation="relu"),

            # Aggregate temporal features into a fixed-size vector
            GlobalAveragePooling1D(),

            # Non-linear projection head
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    # ── Public API (mirrors LSTMForecaster) ───────────────────────────────

    def fit(self, raw_train):
        from tensorflow.keras.callbacks import EarlyStopping

        self._mean = raw_train.mean()
        self._std  = raw_train.std() + 1e-8
        s = (raw_train - self._mean) / self._std

        X, y = self._build_sequences(s)
        if len(X) == 0:
            raise ValueError("Training series too short for CNN lookback.")

        self.model_ = self._build_model()
        cb = EarlyStopping(monitor="val_loss", patience=self.patience,
                           restore_best_weights=True)
        t0 = time.time()
        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=128,        # CNN is parallelisable → larger batch safe
            validation_split=0.1,
            callbacks=[cb],
            verbose=0,
        )
        self.train_time_ = time.time() - t0
        print(f"        [CNN] Trained {self.model_.count_params():,} params "
              f"in {self.train_time_:.1f}s")

    def _predict_one(self, window):
        """Predict one step given a raw (un-normalised) window array."""
        s = (np.array(window) - self._mean) / self._std
        x = s[-self.lookback:][np.newaxis, :, np.newaxis]
        pred_norm = self.model_.predict(x, verbose=0)[0, 0]
        return float(pred_norm * self._std + self._mean)
