"""
models/lstm_model.py
--------------------
Stacked bidirectional LSTM for electricity demand forecasting.

Architecture
------------
  Input  →  [LSTM(128, return_seq=True) → Dropout(0.2)]
         →  [LSTM(64, return_seq=False) → Dropout(0.2)]
         →  Dense(horizon)

Loss     : Huber (robust to outliers)
Optimizer: Adam with ReduceLROnPlateau callback
Training : Early stopping on validation loss (patience=5)

Dependencies:
    pip install tensorflow  (or torch — see LSTMPyTorch below)
"""

import time
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple

warnings.filterwarnings("ignore")


class LSTMForecaster:
    """
    Stacked bidirectional LSTM for electricity demand forecasting (TensorFlow/Keras).

    Call fit() with the full demand array; the class handles sliding-window
    creation, scaling, training, and inverse-scaling internally.
    """

    def __init__(self,
                 lookback: int = 48,
                 horizon: int = 24,
                 lstm_units: Tuple[int, ...] = (128, 64),
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 dense_units: Tuple[int, ...] = (),
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 epochs: int = 50,
                 patience: int = 5,
                 loss: str = "huber"):
        """
        Parameters
        ----------
        lookback       : input sequence length (past hours)
        horizon        : forecast horizon (future hours)
        lstm_units     : number of units in each LSTM layer
        dropout        : dropout rate after each LSTM layer
        bidirectional  : wrap LSTM layers with Bidirectional()
        dense_units    : extra Dense layers before output (optional)
        learning_rate  : initial Adam learning rate
        batch_size     : mini-batch size
        epochs         : maximum training epochs
        patience       : early-stopping patience (epochs)
        loss           : Keras loss ('huber', 'mse', 'mae')
        """
        self.lookback       = lookback
        self.horizon        = horizon
        self.lstm_units     = lstm_units
        self.dropout        = dropout
        self.bidirectional  = bidirectional
        self.dense_units    = dense_units
        self.learning_rate  = learning_rate
        self.batch_size     = batch_size
        self.epochs         = epochs
        self.patience       = patience
        self.loss           = loss

        self.model_      = None
        self.scaler_     = None
        self.history_    = None
        self.train_time_ = None

    # ── Build architecture ────────────────────────────────────────────────────

    def _build_model(self):
        """Construct and compile the Keras model."""
        import tensorflow as tf
        from tensorflow import keras

        inputs = keras.Input(shape=(self.lookback, 1))
        x = inputs

        for i, units in enumerate(self.lstm_units):
            return_seq = (i < len(self.lstm_units) - 1)
            lstm_layer = keras.layers.LSTM(units, return_sequences=return_seq)
            if self.bidirectional:
                lstm_layer = keras.layers.Bidirectional(lstm_layer)
            x = lstm_layer(x)
            x = keras.layers.Dropout(self.dropout)(x)

        for units in self.dense_units:
            x = keras.layers.Dense(units, activation="relu")(x)
            x = keras.layers.Dropout(self.dropout / 2)(x)

        outputs = keras.layers.Dense(self.horizon)(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss=self.loss,
        )
        return model

    # ── Data preparation ─────────────────────────────────────────────────────

    @staticmethod
    def _make_sequences(series: np.ndarray,
                        lookback: int, horizon: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(series) - lookback - horizon + 1):
            X.append(series[i: i + lookback, np.newaxis])
            y.append(series[i + lookback: i + lookback + horizon])
        return np.array(X), np.array(y)

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.scaler_["mean"]) / self.scaler_["std"]

    def _unscale(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.scaler_["std"] + self.scaler_["mean"]

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, y_train: np.ndarray,
            y_val: Optional[np.ndarray] = None,
            val_split: float = 0.1) -> "LSTMForecaster":
        """
        Fit the LSTM on training demand values.

        Parameters
        ----------
        y_train   : 1-D array of training demand (chronological)
        y_val     : optional explicit validation array
        val_split : fraction of y_train to use as validation (if y_val is None)
        """
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
        except ImportError:
            raise ImportError("Install TensorFlow:  pip install tensorflow")

        from tensorflow import keras

        # Scaling
        self.scaler_ = {"mean": float(np.mean(y_train)),
                        "std":  float(np.std(y_train)) + 1e-8}
        scaled = self._scale(y_train)

        X, y = self._make_sequences(scaled, self.lookback, self.horizon)
        print(f"[LSTM] Sequences: X={X.shape}, y={y.shape} …", end=" ", flush=True)

        # Validation
        if y_val is not None:
            scaled_val = self._scale(y_val)
            X_val, y_val_seq = self._make_sequences(scaled_val, self.lookback, self.horizon)
            validation_data = (X_val, y_val_seq)
        else:
            validation_data = val_split

        # Build & train
        self.model_ = self._build_model()
        t0 = time.time()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience,
                restore_best_weights=True, verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3,
                min_lr=1e-6, verbose=0
            ),
        ]

        self.history_ = self.model_.fit(
            X, y,
            validation_split=val_split if isinstance(validation_data, float) else 0.0,
            validation_data=validation_data if not isinstance(validation_data, float) else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
            shuffle=False,          # preserve time order
        )

        self.train_time_ = time.time() - t0
        best_ep = np.argmin(self.history_.history["val_loss"]) + 1
        best_loss = min(self.history_.history["val_loss"])
        print(f"done in {self.train_time_:.1f}s | "
              f"best epoch: {best_ep} | val_loss: {best_loss:.4f}")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, y_context: np.ndarray) -> np.ndarray:
        """
        Forecast the next `horizon` steps using the last `lookback` values.

        Parameters
        ----------
        y_context : array of at least `lookback` recent demand values

        Returns
        -------
        1-D array of length `horizon`
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        if len(y_context) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} context values.")

        window = self._scale(np.array(y_context[-self.lookback:]))
        X = window.reshape(1, self.lookback, 1)
        pred_scaled = self.model_.predict(X, verbose=0)[0]
        return np.maximum(self._unscale(pred_scaled), 0)

    def predict_rolling(self, y_full: np.ndarray,
                        test_start_idx: int) -> np.ndarray:
        """
        Walk-forward (rolling) prediction over the test period.
        At each step, uses actual values as context (teacher forcing).

        Returns array of length (len(y_full) - test_start_idx).
        """
        preds = []
        for i in range(test_start_idx, len(y_full)):
            context = y_full[max(0, i - self.lookback): i]
            if len(context) < self.lookback:
                context = np.pad(context, (self.lookback - len(context), 0),
                                 mode="edge")
            preds.append(self._predict_one(context))
        return np.array(preds)

    def _predict_one(self, context: np.ndarray) -> float:
        """Predict one step ahead from a context window."""
        window = self._scale(context[-self.lookback:]).reshape(1, self.lookback, 1)
        pred   = self.model_.predict(window, verbose=0)[0, 0]
        return max(float(self._unscale(np.array([pred]))[0]), 0)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def plot_loss(self):
        """Plot training and validation loss curves."""
        import matplotlib.pyplot as plt
        if self.history_ is None:
            print("No training history.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.history_.history["loss"],     label="Train loss")
        plt.plot(self.history_.history["val_loss"], label="Val loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("LSTM — Training curve")
        plt.legend(); plt.tight_layout(); plt.show()

    def summary(self):
        if self.model_:
            self.model_.summary()
