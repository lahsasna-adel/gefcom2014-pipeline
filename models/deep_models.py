"""
models/deep_models.py
---------------------
State-of-the-art deep learning forecasters:

1. TFT  — Temporal Fusion Transformer (Lim et al., 2021)
          Multi-horizon, interpretable, quantile forecasting.
          Uses pytorch-forecasting library.

2. N-BEATS — Neural Basis Expansion Analysis (Oreshkin et al., 2020)
             Pure MLP stack; decomposes into trend + seasonality.
             Uses pytorch-forecasting library.

Dependencies:
    pip install torch pytorch-forecasting pytorch-lightning
"""

import time
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")


# ─── Helper: build TimeSeriesDataSet ─────────────────────────────────────────

def _build_timeseries_dataset(df: pd.DataFrame,
                               target: str,
                               max_encoder_length: int,
                               max_prediction_length: int,
                               time_varying_known: List[str],
                               time_varying_unknown: List[str],
                               static_categoricals: List[str],
                               group_id: str = "group"):
    """Build a pytorch-forecasting TimeSeriesDataSet."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import NaNLabelEncoder

    # pytorch-forecasting requires an integer time index
    df = df.copy()
    df[group_id] = "electricity"
    df["time_idx"] = np.arange(len(df))

    dataset = TimeSeriesDataSet(
        df,
        time_idx=group_id + "_time_idx" if group_id + "_time_idx" in df.columns else "time_idx",
        target=target,
        group_ids=[group_id],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=time_varying_unknown,
        static_categoricals=static_categoricals if static_categoricals else [],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


# ─── Temporal Fusion Transformer ─────────────────────────────────────────────

class TFTForecaster:
    """
    Temporal Fusion Transformer for electricity demand forecasting.

    Provides:
    - Multi-horizon point forecasts
    - Quantile predictions (10th, 50th, 90th percentile)
    - Interpretable attention weights
    - Variable importance scores

    Reference: Lim et al. (2021) "Temporal Fusion Transformers for
               Interpretable Multi-horizon Time Series Forecasting"
               https://arxiv.org/abs/1912.09363
    """

    def __init__(self,
                 max_encoder_length: int = 168,   # 1 week of hourly data
                 max_prediction_length: int = 24,  # 24-hour horizon
                 hidden_size: int = 64,
                 attention_head_size: int = 4,
                 dropout: float = 0.1,
                 hidden_continuous_size: int = 32,
                 quantiles: List[float] = [0.1, 0.5, 0.9],
                 learning_rate: float = 1e-3,
                 batch_size: int = 64,
                 max_epochs: int = 50,
                 patience: int = 5,
                 gradient_clip_val: float = 0.1):
        self.max_encoder_length    = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size           = hidden_size
        self.attention_head_size   = attention_head_size
        self.dropout               = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.quantiles             = quantiles
        self.learning_rate         = learning_rate
        self.batch_size            = batch_size
        self.max_epochs            = max_epochs
        self.patience              = patience
        self.gradient_clip_val     = gradient_clip_val
        self.model_                = None
        self.trainer_              = None
        self.train_time_           = None

    def fit(self, df_train: pd.DataFrame,
            df_val: pd.DataFrame,
            target: str = "demand",
            time_varying_known: Optional[List[str]] = None,
            time_varying_unknown: Optional[List[str]] = None) -> "TFTForecaster":
        """
        Train TFT on a DataFrame with DatetimeIndex.

        Parameters
        ----------
        df_train/df_val         : DataFrames with target + feature columns
        target                  : name of the target column
        time_varying_known      : features known in advance (hour, weekday, …)
        time_varying_unknown    : features only known historically (demand lags)
        """
        try:
            import torch
            import pytorch_lightning as pl
            from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
            from pytorch_forecasting.metrics import QuantileLoss
        except ImportError:
            raise ImportError(
                "Install required packages:\n"
                "  pip install torch pytorch-forecasting pytorch-lightning"
            )

        known   = time_varying_known   or ["hour", "dayofweek", "month", "is_weekend"]
        unknown = time_varying_unknown or [target]

        # Prepare DataFrames
        def _prep(df):
            df = df.copy()
            df["group"]    = "electricity"
            df["time_idx"] = np.arange(len(df))
            df["hour"]       = df.index.hour
            df["dayofweek"]  = df.index.dayofweek
            df["month"]      = df.index.month
            df["is_weekend"] = (df.index.dayofweek >= 5).astype(float)
            return df

        df_train = _prep(df_train)
        df_val   = _prep(df_val)

        train_ds = TimeSeriesDataSet(
            df_train, time_idx="time_idx", target=target, group_ids=["group"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=known,
            time_varying_unknown_reals=unknown,
            add_relative_time_idx=True, add_target_scales=True,
        )
        val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val, predict=True, stop_randomization=True)

        train_dl = train_ds.to_dataloader(train=True,  batch_size=self.batch_size, num_workers=0)
        val_dl   = val_ds.to_dataloader(  train=False, batch_size=self.batch_size, num_workers=0)

        self.model_ = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=QuantileLoss(self.quantiles),
            log_interval=10,
            reduce_on_plateau_patience=3,
        )

        self.trainer_ = pl.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=self.gradient_clip_val,
            callbacks=[pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, mode="min"
            )],
            enable_progress_bar=True,
            logger=False,
        )

        print(f"[TFT] Fitting ({sum(p.numel() for p in self.model_.parameters()):,} params) …")
        t0 = time.time()
        self.trainer_.fit(self.model_, train_dl, val_dl)
        self.train_time_ = time.time() - t0
        print(f"[TFT] Done in {self.train_time_:.1f}s")
        return self

    def predict(self, df_test: pd.DataFrame,
                target: str = "demand") -> dict:
        """
        Generate quantile forecasts.

        Returns
        -------
        dict with keys: 'p10', 'p50', 'p90' (numpy arrays)
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        raw_preds = self.model_.predict(df_test, return_x=False)
        return {
            "p10": raw_preds[:, 0].numpy(),
            "p50": raw_preds[:, 1].numpy(),
            "p90": raw_preds[:, 2].numpy(),
        }

    def interpret(self):
        """Return variable importance dict (attention weights)."""
        return self.model_.interpret_output(
            self.trainer_.predict(self.model_),
            reduction="sum"
        )


# ─── N-BEATS ─────────────────────────────────────────────────────────────────

class NBEATSForecaster:
    """
    N-BEATS: Neural Basis Expansion Analysis for Time Series.

    Decomposed architecture:
    - Trend stack   : polynomial basis expansion
    - Seasonality stack : Fourier basis expansion
    - Generic stack  : learned basis (optional)

    No feature engineering required — learns directly from raw series.

    Reference: Oreshkin et al. (2020) "N-BEATS: Neural Basis Expansion
               Analysis for Interpretable Time Series Forecasting"
               https://arxiv.org/abs/1905.10437
    """

    def __init__(self,
                 max_encoder_length: int = 168,
                 max_prediction_length: int = 24,
                 num_blocks: List[int] = [3, 3],
                 num_block_layers: List[int] = [4, 4],
                 widths: List[int] = [256, 2048],
                 sharing: List[bool] = [True, True],
                 expansion_coefficient_lengths: List[int] = [3, 7],
                 stack_types: List[str] = ["trend", "seasonality"],
                 learning_rate: float = 1e-3,
                 batch_size: int = 64,
                 max_epochs: int = 50,
                 patience: int = 5):
        self.max_encoder_length            = max_encoder_length
        self.max_prediction_length         = max_prediction_length
        self.num_blocks                    = num_blocks
        self.num_block_layers              = num_block_layers
        self.widths                        = widths
        self.sharing                       = sharing
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.stack_types                   = stack_types
        self.learning_rate                 = learning_rate
        self.batch_size                    = batch_size
        self.max_epochs                    = max_epochs
        self.patience                      = patience
        self.model_                        = None
        self.trainer_                      = None
        self.train_time_                   = None

    def fit(self, df_train: pd.DataFrame,
            df_val: pd.DataFrame,
            target: str = "demand") -> "NBEATSForecaster":
        """
        Train N-BEATS on raw demand series (no external features needed).
        """
        try:
            import pytorch_lightning as pl
            from pytorch_forecasting import NBeats, TimeSeriesDataSet
            from pytorch_forecasting.metrics import SMAPE
        except ImportError:
            raise ImportError("pip install torch pytorch-forecasting pytorch-lightning")

        def _prep(df, start_idx=0):
            df = df.copy()
            df["group"]    = "electricity"
            df["time_idx"] = np.arange(start_idx, start_idx + len(df))
            return df

        df_train = _prep(df_train)
        df_val   = _prep(df_val, start_idx=len(df_train))

        train_ds = TimeSeriesDataSet(
            df_train, time_idx="time_idx", target=target, group_ids=["group"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_unknown_reals=[target],
        )
        val_ds = TimeSeriesDataSet.from_dataset(
            train_ds, df_val, predict=True, stop_randomization=True
        )

        train_dl = train_ds.to_dataloader(train=True,  batch_size=self.batch_size, num_workers=0)
        val_dl   = val_ds.to_dataloader(  train=False, batch_size=self.batch_size, num_workers=0)

        self.model_ = NBeats.from_dataset(
            train_ds,
            learning_rate=self.learning_rate,
            weight_decay=1e-2,
            backcast_loss_ratio=0.1,
            num_blocks=self.num_blocks,
            num_block_layers=self.num_block_layers,
            widths=self.widths,
            sharing=self.sharing,
            expansion_coefficient_lengths=self.expansion_coefficient_lengths,
            stack_types=self.stack_types,
            loss=SMAPE(),
        )

        self.trainer_ = pl.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=0.1,
            callbacks=[pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, mode="min"
            )],
            enable_progress_bar=True,
            logger=False,
        )

        print(f"[N-BEATS] Fitting ({sum(p.numel() for p in self.model_.parameters()):,} params) …")
        t0 = time.time()
        self.trainer_.fit(self.model_, train_dl, val_dl)
        self.train_time_ = time.time() - t0
        print(f"[N-BEATS] Done in {self.train_time_:.1f}s")
        return self

    def predict(self, df_test: pd.DataFrame,
                target: str = "demand") -> np.ndarray:
        """Return point forecast as a numpy array."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        raw = self.model_.predict(df_test)
        return np.maximum(raw.numpy().flatten(), 0)
