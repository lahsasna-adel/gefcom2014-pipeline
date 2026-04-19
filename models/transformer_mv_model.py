"""
models/transformer_mv_model.py
-------------------------------
Multivariate Transformer — Phase 4b extension.

Extends the univariate TransformerModel (transformer_model.py) to accept an
engineered feature matrix X alongside the raw demand series.

Architecture change: enc_proj maps (1 + F) → d_model instead of 1 → d_model.
Everything else (encoder stack, decoder, output projection) is identical to
the univariate version, making this a clean ablation.

Integration notes
-----------------
- Feature-sensitive model — runs across all 9 feature subsets
- fit(X_train, y_train, raw_train)
- predict_batch(X_test, raw, raw_split, n)
- train_time_ set after fit()
"""

import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.transformer_model import (
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from models.informer_mv_model import _make_mv_sequences


# ──────────────────────────────────────────────────────────────────────────────
# Multivariate Transformer architecture
# ──────────────────────────────────────────────────────────────────────────────

class TransformerMVModel(nn.Module):
    """Vanilla Transformer with multivariate encoder input."""

    def __init__(self, enc_in: int, lookback: int = 168, horizon: int = 24,
                 d_model: int = 64, n_heads: int = 4,
                 d_ff: int = 256, n_enc_layers: int = 2,
                 n_dec_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon  = horizon

        self.enc_proj = nn.Linear(enc_in, d_model)
        self.dec_proj = nn.Linear(1, d_model)
        self.enc_pe   = PositionalEncoding(d_model, dropout=dropout)
        self.dec_pe   = PositionalEncoding(d_model, dropout=dropout)

        self.enc_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_enc_layers)
        ])
        self.dec_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_dec_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, enc_x, dec_x):
        """
        enc_x : (B, lookback, enc_in)
        dec_x : (B, horizon,  1)
        """
        e = self.enc_pe(self.enc_proj(enc_x))
        for layer in self.enc_layers:
            e = layer(e)
        e = self.enc_norm(e)

        d = self.dec_pe(self.dec_proj(dec_x))
        for layer in self.dec_layers:
            d = layer(d, e)
        d = self.dec_norm(d)

        return self.out_proj(d).squeeze(-1)           # (B, horizon)


# ──────────────────────────────────────────────────────────────────────────────
# Forecaster wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TransformerMVForecaster:
    """
    Multivariate Transformer forecaster — pipeline-compatible wrapper.
    API identical to InformerMVForecaster.
    """

    def __init__(self, lookback: int = 168, horizon: int = 24,
                 d_model: int = 64, n_heads: int = 4,
                 d_ff: int = 256, n_enc_layers: int = 2,
                 n_dec_layers: int = 1,
                 epochs: int = 30, patience: int = 5,
                 batch_size: int = 64, lr: float = 1e-3,
                 dropout: float = 0.1):
        self.lookback     = lookback
        self.horizon      = horizon
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.d_ff         = d_ff
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.epochs       = epochs
        self.patience     = patience
        self.batch_size   = batch_size
        self.lr           = lr
        self.dropout      = dropout

        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_      = None
        self.d_mu_       = np.float32(0.0)
        self.d_sigma_    = np.float32(1.0)
        self.f_mu_       = None
        self.f_sigma_    = None
        self.train_time_ = 0.0

    def _scale_demand(self, x):   return (x - self.d_mu_)  / (self.d_sigma_ + 1e-8)
    def _unscale_demand(self, x): return x * self.d_sigma_ + self.d_mu_
    def _scale_features(self, X): return (X - self.f_mu_)  / (self.f_sigma_ + 1e-8)

    def fit(self, X_train, y_train, raw_train: np.ndarray):
        t0 = time.time()

        X_arr = np.array(X_train, dtype=np.float32)
        y_arr = np.array(y_train, dtype=np.float32)

        self.d_mu_    = np.float32(y_arr.mean())
        self.d_sigma_ = np.float32(y_arr.std())
        self.f_mu_    = X_arr.mean(axis=0).astype(np.float32)
        self.f_sigma_ = X_arr.std(axis=0).astype(np.float32)
        self.f_sigma_ = np.where(self.f_sigma_ < 1e-8,
                                  np.float32(1.0), self.f_sigma_)

        demand_s   = self._scale_demand(y_arr)
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
        self.model_ = TransformerMVModel(
            enc_in=enc_in, lookback=self.lookback, horizon=self.horizon,
            d_model=self.d_model, n_heads=self.n_heads,
            d_ff=self.d_ff, n_enc_layers=self.n_enc_layers,
            n_dec_layers=self.n_dec_layers, dropout=self.dropout,
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
                xb    = xb.to(self.device)
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
                    print(f"        [Transformer-MV] Early stop epoch {epoch+1} "
                          f"(val_loss={best_val:.5f})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
        self.train_time_ = time.time() - t0
        print(f"        [Transformer-MV] {X_arr.shape[1]} features | "
              f"Trained {epoch+1} epochs in {self.train_time_:.1f}s on {self.device}")

    def predict_batch(self, X_test, raw: np.ndarray,
                      raw_split: int, n: int,
                      chunk: int = 256) -> np.ndarray:
        X_arr = np.array(X_test, dtype=np.float32)
        X_s   = self._scale_features(X_arr)
        F     = X_arr.shape[1]

        demand_history = self._scale_demand(
            raw[:raw_split + n].astype(np.float32)
        )

        contexts = np.empty((n, self.lookback, 1 + F), dtype=np.float32)

        for i in range(n):
            d_start = max(0, raw_split + i - self.lookback)
            d_ctx   = demand_history[d_start: raw_split + i]
            if len(d_ctx) < self.lookback:
                pad   = np.full(self.lookback - len(d_ctx),
                                d_ctx[0], dtype=np.float32)
                d_ctx = np.concatenate([pad, d_ctx])
            d_ctx = d_ctx[-self.lookback:].reshape(-1, 1)

            f_start_in_test = i - self.lookback
            if f_start_in_test >= 0:
                f_ctx = X_s[f_start_in_test: i]
            else:
                available = X_s[:i] if i > 0 else np.zeros((0, F),
                                                             dtype=np.float32)
                pad_rows  = self.lookback - len(available)
                f_ctx     = np.concatenate([
                    np.zeros((pad_rows, F), dtype=np.float32), available
                ], axis=0)

            contexts[i] = np.concatenate([d_ctx, f_ctx], axis=1)

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
