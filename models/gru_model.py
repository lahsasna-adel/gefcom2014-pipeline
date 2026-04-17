"""
models/gru_model.py
-------------------
GRU (Gated Recurrent Unit) forecaster — mirrors LSTMForecaster interface.

Architecture: Input → GRU(64) → Dropout(0.2) → Dense(1)

GRU vs LSTM:
  - GRU uses 2 gates (reset, update) vs LSTM's 3 (input, forget, output)
  - GRU has no separate cell state — combines it into hidden state
  - ~25% fewer parameters than LSTM → faster training
  - Often matches LSTM on medium-length sequences (Chung et al., 2014)
  - Strong choice for hourly electricity data (Marino et al., 2016)

GRU update equations:
  z_t = σ(W_z · [h_{t-1}, x_t])           ← update gate
  r_t = σ(W_r · [h_{t-1}, x_t])           ← reset gate
  h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])   ← candidate hidden state
  h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  ← final hidden state
"""

import numpy as np
import time


class GRUForecaster:
    """
    Univariate GRU forecaster.

    Parameters
    ----------
    lookback : int   — input window size in hours (default 48)
    horizon  : int   — prediction step (default 24)
    epochs   : int   — max training epochs
    patience : int   — early stopping patience
    units    : int   — GRU units (default 64)
    """

    def __init__(self, lookback=168, horizon=24, epochs=50,
                 patience=8, units=64):
        self.lookback    = lookback
        self.horizon     = horizon
        self.epochs      = epochs
        self.patience    = patience
        self.units       = units
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
        from tensorflow.keras.layers import GRU, Dense, Dropout

        model = Sequential([
            GRU(self.units, input_shape=(self.lookback, 1),
                return_sequences=False),
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
            raise ValueError("Training series too short for GRU lookback.")

        self.model_ = self._build_model()
        cb = EarlyStopping(monitor="val_loss", patience=self.patience,
                           restore_best_weights=True)
        t0 = time.time()
        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=64,
            validation_split=0.1,
            callbacks=[cb],
            verbose=0,
        )
        self.train_time_ = time.time() - t0
        print(f"        [GRU] Trained {self.model_.count_params():,} params "
              f"in {self.train_time_:.1f}s")

    def _predict_one(self, window):
        """Predict one step given a raw (un-normalised) window array."""
        s = (np.array(window) - self._mean) / self._std
        x = s[-self.lookback:][np.newaxis, :, np.newaxis]
        pred_norm = self.model_.predict(x, verbose=0)[0, 0]
        return float(pred_norm * self._std + self._mean)
