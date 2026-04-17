"""
models/rnn_model.py
-------------------
Vanilla RNN forecaster — mirrors LSTMForecaster interface exactly.

Architecture: Input → SimpleRNN(64) → Dropout(0.2) → Dense(1)

Vanilla RNN is intentionally simpler than LSTM:
  - No cell state, only hidden state h_t
  - Update: h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
  - Faster to train but suffers from vanishing gradients on long sequences
  - Useful as a baseline to measure the value of gating (LSTM/GRU)

Lookback=48h captures two full daily cycles, which is sufficient for
the vanilla RNN given its limited long-range memory.
"""

import numpy as np
import time


class RNNForecaster:
    """
    Univariate RNN forecaster.

    Parameters
    ----------
    lookback : int   — input window size in hours (default 48)
    horizon  : int   — prediction step size (default 24, one day ahead)
    epochs   : int   — max training epochs
    patience : int   — early stopping patience
    units    : int   — number of SimpleRNN units
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
        from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

        model = Sequential([
            SimpleRNN(self.units, input_shape=(self.lookback, 1),
                      return_sequences=False),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    # ── Public API (mirrors LSTMForecaster) ───────────────────────────────

    def fit(self, raw_train):
        from tensorflow.keras.callbacks import EarlyStopping

        # Normalise
        self._mean = raw_train.mean()
        self._std  = raw_train.std() + 1e-8
        s = (raw_train - self._mean) / self._std

        X, y = self._build_sequences(s)
        if len(X) == 0:
            raise ValueError("Training series too short for RNN lookback.")

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
        print(f"        [RNN] Trained {self.model_.count_params():,} params "
              f"in {self.train_time_:.1f}s")

    def _predict_one(self, window):
        """Predict one step given a raw (un-normalised) window array."""
        s = (np.array(window) - self._mean) / self._std
        x = s[-self.lookback:][np.newaxis, :, np.newaxis]
        pred_norm = self.model_.predict(x, verbose=0)[0, 0]
        return float(pred_norm * self._std + self._mean)
