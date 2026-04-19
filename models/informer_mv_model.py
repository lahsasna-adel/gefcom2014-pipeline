"""
models/informer_mv_model.py
---------------------------
Multivariate Informer — Phase 4b extension.

Extends the univariate InformerModel (informer_model.py) to accept an
engineered feature matrix X alongside the raw demand series. This creates
a fair comparison with gradient boosting methods (LightGBM, XGBoost) that
also operate on the full feature matrix.

Key difference from univariate Informer
----------------------------------------
Univariate  : encoder input = (demand_t-L, ..., demand_t-1)  shape (B, L, 1)
Multivariate: encoder input = (demand + F features) at each step shape (B, L, 1+F)

The F engineered features (lags, rolling stats, calendar, Fourier, etc.) are
aligned with the demand series and concatenated channel-wise before the linear
input projection. The projection maps (1 + F) → d_model regardless of F,
so the architecture is identical beyond the first linear layer.

Decoder input is kept univariate (zeros) — generative decoding is unchanged.

Integration notes
-----------------
- Feature-sensitive model: listed in MULTIVARIATE_MV_MODELS, NOT UNIVARIATE_MODELS
- Runs across all 9 feature subsets (subset loop is not skipped)
- fit()         : accepts (X_train_arr, y_train_arr, raw_train)
- predict_batch(): accepts (X_test_arr, raw, raw_split, n)
- train_time_  : set after fit()
"""

import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────────────────────────────────────
# Re-use building blocks from univariate Informer
# ──────────────────────────────────────────────────────────────────────────────

from models.informer_model import (
    PositionalEncoding,
    ProbSparseSelfAttention,
    InformerEncoderLayer,
    DistillingConv,
    InformerDecoderLayer,
)


# ──────────────────────────────────────────────────────────────────────────────
# Multivariate Informer architecture
# ──────────────────────────────────────────────────────────────────────────────

class InformerMVModel(nn.Module):
    """
    Informer with multivariate encoder input.

    enc_in  : number of input channels = 1 (demand) + n_features
    d_model : embedding dimension (same as univariate version)
    """

    def __init__(self, enc_in: int, lookback: int = 168, horizon: int = 24,
                 d_model: int = 64, n_heads: int = 4,
                 d_ff: int = 256, n_enc_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon  = horizon

        # Encoder projects enc_in channels → d_model
        self.enc_proj = nn.Linear(enc_in, d_model)
        # Decoder stays univariate (generative zeros)
        self.dec_proj = nn.Linear(1, d_model)

        self.enc_pe = PositionalEncoding(d_model, dropout=dropout)
        self.dec_pe = PositionalEncoding(d_model, dropout=dropout)

        self.enc_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_enc_layers)
        ])
        self.distil = nn.ModuleList([
            DistillingConv(d_model) for _ in range(n_enc_layers - 1)
        ])

        self.dec_layer = InformerDecoderLayer(d_model, n_heads, d_ff, dropout)
        self.out_proj  = nn.Linear(d_model, 1)

    def forward(self, enc_x, dec_x):
        """
        enc_x : (B, lookback, enc_in)   — demand + features at each step
        dec_x : (B, horizon,  1)         — zeros (generative)
        """
        e = self.enc_pe(self.enc_proj(enc_x))
        for i, layer in enumerate(self.enc_layers):
            e = layer(e)
            if i < len(self.distil):
                e = self.distil[i](e)

        d = self.dec_pe(self.dec_proj(dec_x))
        d = self.dec_layer(d, e)
        return self.out_proj(d).squeeze(-1)           # (B, horizon)


# ──────────────────────────────────────────────────────────────────────────────
# Sliding-window dataset builder (multivariate)
# ──────────────────────────────────────────────────────────────────────────────

def _make_mv_sequences(demand: np.ndarray, features: np.ndarray,
                       lookback: int, horizon: int):
    """
    Build (X, y) sliding-window sequences.

    demand   : (N,)      scaled demand series
    features : (N, F)    scaled feature matrix (aligned with demand)
    Returns
    -------
    X : (n_samples, lookback, 1+F)  encoder inputs
    y : (n_samples, horizon)        target demand values
    """
    N = len(demand)
    n_samples = N - lookback - horizon + 1
    F = features.shape[1]

    X = np.empty((n_samples, lookback, 1 + F), dtype=np.float32)
    y = np.empty((n_samples, horizon),          dtype=np.float32)

    for i in range(n_samples):
        d_window = demand[i: i + lookback].reshape(-1, 1)       # (L, 1)
        f_window = features[i: i + lookback]                     # (L, F)
        X[i] = np.concatenate([d_window, f_window], axis=1)
        y[i] = demand[i + lookback: i + lookback + horizon]

    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Forecaster wrapper
# ──────────────────────────────────────────────────────────────────────────────

class InformerMVForecaster:
    """
    Multivariate Informer forecaster — pipeline-compatible wrapper.

    fit(X_train, y_train, raw_train)
        X_train : pd.DataFrame or np.ndarray  (N_train, F)  feature matrix
        y_train : pd.Series   or np.ndarray  (N_train,)    demand target
        raw_train : np.ndarray (N_train,) raw demand (for scaling reference)

    predict_batch(X_test, raw, raw_split, n)
        X_test    : (N_test, F)  feature matrix for the test window
        raw       : full raw demand array
        raw_split : index of first test row in raw
        n         : number of steps to predict
    """

    def __init__(self, lookback: int = 168, horizon: int = 24,
                 d_model: int = 64, n_heads: int = 4,
                 d_ff: int = 256, n_enc_layers: int = 2,
                 epochs: int = 30, patience: int = 5,
                 batch_size: int = 64, lr: float = 1e-3,
                 dropout: float = 0.1):
        self.lookback     = lookback
        self.horizon      = horizon
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.d_ff         = d_ff
        self.n_enc_layers = n_enc_layers
        self.epochs       = epochs
        self.patience     = patience
        self.batch_size   = batch_size
        self.lr           = lr
        self.dropout      = dropout

        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_      = None
        self.d_mu_       = np.float32(0.0)   # demand scale
        self.d_sigma_    = np.float32(1.0)
        self.f_mu_       = None              # feature scale (per-column)
        self.f_sigma_    = None
        self.train_time_ = 0.0

    # ── scaling ───────────────────────────────────────────────────────────────
    def _scale_demand(self, x):
        return (x - self.d_mu_) / (self.d_sigma_ + 1e-8)

    def _unscale_demand(self, x):
        return x * self.d_sigma_ + self.d_mu_

    def _scale_features(self, X):
        return (X - self.f_mu_) / (self.f_sigma_ + 1e-8)

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, X_train, y_train, raw_train: np.ndarray):
        t0 = time.time()

        # Convert to numpy float32
        X_arr = np.array(X_train, dtype=np.float32)
        y_arr = np.array(y_train, dtype=np.float32)

        # Fit scalers on training data only (no leakage)
        self.d_mu_    = np.float32(y_arr.mean())
        self.d_sigma_ = np.float32(y_arr.std())
        self.f_mu_    = X_arr.mean(axis=0).astype(np.float32)
        self.f_sigma_ = X_arr.std(axis=0).astype(np.float32)
        # Avoid division by zero for constant features
        self.f_sigma_ = np.where(self.f_sigma_ < 1e-8,
                                  np.float32(1.0), self.f_sigma_)

        demand_s  = self._scale_demand(y_arr)
        features_s = self._scale_features(X_arr)

        X_seq, y_seq = _make_mv_sequences(
            demand_s, features_s, self.lookback, self.horizon
        )

        n_val   = max(1, int(len(X_seq) * 0.10))
        X_tr, X_va = X_seq[:-n_val], X_seq[-n_val:]
        y_tr, y_va = y_seq[:-n_val], y_seq[-n_val:]

        dl_tr = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=self.batch_size, shuffle=True
        )
        dl_va = DataLoader(
            TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
            batch_size=self.batch_size
        )

        enc_in = 1 + X_arr.shape[1]
        self.model_ = InformerMVModel(
            enc_in=enc_in, lookback=self.lookback, horizon=self.horizon,
            d_model=self.d_model, n_heads=self.n_heads,
            d_ff=self.d_ff, n_enc_layers=self.n_enc_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                                     weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=3, factor=0.5, min_lr=1e-5
        )
        criterion = nn.HuberLoss(delta=1.0)

        best_val   = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in dl_tr:
                xb    = xb.to(self.device)                          # (B, L, enc_in)
                yb    = yb.to(self.device)
                dec_x = torch.zeros(xb.size(0), self.horizon, 1,
                                    device=self.device)
                optimiser.zero_grad()
                loss = criterion(self.model_(xb, dec_x), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimiser.step()

            self.model_.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb    = xb.to(self.device)
                    yb    = yb.to(self.device)
                    dec_x = torch.zeros(xb.size(0), self.horizon, 1,
                                        device=self.device)
                    val_losses.append(
                        criterion(self.model_(xb, dec_x), yb).item()
                    )
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)

            if val_loss < best_val - 1e-6:
                best_val   = val_loss
                no_improve = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model_.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"        [Informer-MV] Early stop epoch {epoch+1} "
                          f"(val_loss={best_val:.5f})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
        self.train_time_ = time.time() - t0
        print(f"        [Informer-MV] {X_arr.shape[1]} features | "
              f"Trained {epoch+1} epochs in {self.train_time_:.1f}s on {self.device}")

    # ── batch prediction ──────────────────────────────────────────────────────
    def predict_batch(self, X_test, raw: np.ndarray,
                      raw_split: int, n: int,
                      chunk: int = 256) -> np.ndarray:
        """
        Batched inference over the test window.

        X_test : (N_test, F) feature matrix for the test window — aligned
                 row-by-row with raw[raw_split : raw_split + n]
        """
        X_arr = np.array(X_test, dtype=np.float32)
        X_s   = self._scale_features(X_arr)           # (N_test, F)

        # Build encoder contexts: for step i, use rows [i-lookback .. i-1]
        # of the combined (demand + features) sequence.
        # We need demand values from raw for the lookback window — these come
        # from BEFORE raw_split (training history) as well as within X_test.
        demand_history = self._scale_demand(
            raw[:raw_split + n].astype(np.float32)
        )

        contexts = np.empty((n, self.lookback, 1 + X_arr.shape[1]),
                             dtype=np.float32)

        for i in range(n):
            # Demand context window (lookback steps ending just before step i)
            d_start = max(0, raw_split + i - self.lookback)
            d_ctx   = demand_history[d_start: raw_split + i]
            if len(d_ctx) < self.lookback:
                pad   = np.full(self.lookback - len(d_ctx),
                                d_ctx[0], dtype=np.float32)
                d_ctx = np.concatenate([pad, d_ctx])
            d_ctx = d_ctx[-self.lookback:].reshape(-1, 1)         # (L, 1)

            # Feature context window (same lookback, from X_s)
            # For steps before X_test starts (i < lookback), pad with zeros
            f_start_in_test = i - self.lookback
            F = X_arr.shape[1]
            if f_start_in_test >= 0:
                f_ctx = X_s[f_start_in_test: i]                   # (L, F)
            else:
                available = X_s[:i] if i > 0 else np.zeros((0, F),
                                                             dtype=np.float32)
                pad_rows  = self.lookback - len(available)
                f_ctx     = np.concatenate([
                    np.zeros((pad_rows, F), dtype=np.float32),
                    available
                ], axis=0)

            contexts[i] = np.concatenate([d_ctx, f_ctx], axis=1)  # (L, 1+F)

        self.model_.eval()
        preds = np.empty(n, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, n, chunk):
                end   = min(start + chunk, n)
                xb    = torch.from_numpy(contexts[start:end]).to(self.device)
                dec_x = torch.zeros(end - start, self.horizon, 1,
                                    device=self.device)
                out   = self.model_(xb, dec_x)[:, 0].cpu().numpy()
                preds[start:end] = self._unscale_demand(out)

        return preds
