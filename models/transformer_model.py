"""
models/transformer_model.py
---------------------------
Vanilla Transformer for univariate electricity demand forecasting.
Vaswani et al. (2017) — "Attention Is All You Need", NeurIPS 2017.

Architecture
------------
Encoder : standard multi-head self-attention + FFN × n_enc_layers
Decoder : cross-attention generative decoder (same as Informer decoder,
          but with full O(L²) attention instead of ProbSparse)

Compared with Informer:
  - Full quadratic self-attention (O(L²)) — no sparsification
  - No distilling between encoder layers — full sequence length kept
  - Serves as ablation baseline: does ProbSparse + distilling help?

Integration notes:
  - Univariate model (raw series only) — listed in UNIVARIATE_MODELS
  - lookback  : 168 h (1 week)  — same as Informer for fair comparison
  - horizon   : 24  h (1 day)
  - API       : fit(raw_train), _predict_one(context), train_time_
"""

import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding (shared with Informer)
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


# ──────────────────────────────────────────────────────────────────────────────
# Encoder layer (full self-attention)
# ──────────────────────────────────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer.
    Pre-LN variant (Ba et al., 2016) — more stable for time-series.
    """

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, src_key_padding_mask=None):
        # Pre-LN: normalise before sub-layer
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                key_padding_mask=src_key_padding_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Decoder layer (cross-attention + causal self-attention)
# ──────────────────────────────────────────────────────────────────────────────

class TransformerDecoderLayer(nn.Module):
    """
    Generative decoder layer: causal self-attention + cross-attention.
    Pre-LN variant for training stability.
    """

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1      = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.norm2      = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    @staticmethod
    def _causal_mask(sz: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, tgt, memory):
        L = tgt.size(1)
        mask = self._causal_mask(L, tgt.device)
        # Self-attention (causal)
        normed     = self.norm1(tgt)
        sa_out, _  = self.self_attn(normed, normed, normed, attn_mask=mask)
        tgt        = tgt + sa_out
        # Cross-attention with encoder memory
        normed     = self.norm2(tgt)
        ca_out, _  = self.cross_attn(normed, memory, memory)
        tgt        = tgt + ca_out
        tgt        = tgt + self.ff(self.norm3(tgt))
        return tgt


# ──────────────────────────────────────────────────────────────────────────────
# Full Transformer model
# ──────────────────────────────────────────────────────────────────────────────

class TransformerModel(nn.Module):
    """
    Vanilla Transformer for univariate time-series forecasting.

    Encoder: n_enc_layers × standard MHSA + FFN (full O(L²) attention)
    Decoder: n_dec_layers × causal MHSA + cross-attention (generative)

    Default config (matches Informer for ablation fairness):
      lookback=168, horizon=24, d_model=64, n_heads=4, d_ff=256
    """

    def __init__(self, lookback: int = 168, horizon: int = 24,
                 d_model: int = 64, n_heads: int = 4,
                 d_ff: int = 256, n_enc_layers: int = 2,
                 n_dec_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon  = horizon

        self.enc_proj = nn.Linear(1, d_model)
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
        enc_x : (B, lookback, 1)
        dec_x : (B, horizon,  1) — zeros at inference (generative mode)
        """
        e = self.enc_pe(self.enc_proj(enc_x))
        for layer in self.enc_layers:
            e = layer(e)
        e = self.enc_norm(e)

        d = self.dec_pe(self.dec_proj(dec_x))
        for layer in self.dec_layers:
            d = layer(d, e)
        d = self.dec_norm(d)

        return self.out_proj(d).squeeze(-1)          # (B, horizon)


# ──────────────────────────────────────────────────────────────────────────────
# Sequence dataset helper (shared pattern with Informer)
# ──────────────────────────────────────────────────────────────────────────────

def _make_sequences(series: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i: i + lookback])
        y.append(series[i + lookback: i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Forecaster wrapper (matches pipeline API)
# ──────────────────────────────────────────────────────────────────────────────

class TransformerForecaster:
    """
    Scikit-learn style wrapper around TransformerModel.

    Hyperparameters
    ---------------
    lookback      : 168 h — same as Informer for ablation fairness
    horizon       : 24  h
    d_model       : 64
    n_heads       : 4
    d_ff          : 256
    n_enc_layers  : 2
    n_dec_layers  : 1
    epochs        : 30
    patience      : 5
    batch_size    : 64
    lr            : 1e-3
    """

    def __init__(self, lookback: int = 168, horizon: int = 24,
                 d_model: int = 64, n_heads: int = 4,
                 d_ff: int = 256, n_enc_layers: int = 2,
                 n_dec_layers: int = 1,
                 epochs: int = 30, patience: int = 5,
                 batch_size: int = 64, lr: float = 1e-3,
                 dropout: float = 0.1,
                 use_log_norm: bool = True):
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
        self.use_log_norm = use_log_norm   # Smyl & Hua (2019): log(x/level)

        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_      = None
        self.mu_         = 0.0
        self.sigma_      = 1.0
        self.level_      = 1.0
        self.train_time_ = 0.0

    def _scale(self, x):
        if self.use_log_norm:
            return np.log(np.maximum(x, 1e-6) / (self.level_ + 1e-8)).astype(np.float32)
        return ((x - self.mu_) / (self.sigma_ + 1e-8)).astype(np.float32)

    def _unscale(self, x):
        if self.use_log_norm:
            return (np.exp(x) * self.level_).astype(np.float32)
        return (x * self.sigma_ + self.mu_).astype(np.float32)

    def fit(self, raw_train: np.ndarray):
        t0 = time.time()

        if self.use_log_norm:
            self.level_ = np.float32(raw_train.mean())
            print(f"        [Transformer] Log-norm: level={self.level_:.2f} "
                  f"(Smyl & Hua 2019 §3.1.3)")
        else:
            self.mu_    = np.float32(raw_train.mean())
            self.sigma_ = np.float32(raw_train.std())
        series = self._scale(raw_train).astype(np.float32)

        X, y = _make_sequences(series, self.lookback, self.horizon)
        n_val   = max(1, int(len(X) * 0.10))
        X_tr, X_va = X[:-n_val], X[-n_val:]
        y_tr, y_va = y[:-n_val], y[-n_val:]

        dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr),
                                         torch.from_numpy(y_tr)),
                           batch_size=self.batch_size, shuffle=True)
        dl_va = DataLoader(TensorDataset(torch.from_numpy(X_va),
                                         torch.from_numpy(y_va)),
                           batch_size=self.batch_size)

        self.model_ = TransformerModel(
            lookback=self.lookback, horizon=self.horizon,
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
                xb    = xb.unsqueeze(-1).to(self.device)
                yb    = yb.to(self.device)
                dec_x = torch.zeros(xb.size(0), self.horizon, 1,
                                    device=self.device)
                optimiser.zero_grad()
                loss  = criterion(self.model_(xb, dec_x), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimiser.step()

            self.model_.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb    = xb.unsqueeze(-1).to(self.device)
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
                    print(f"        [Transformer] Early stop epoch {epoch+1} "
                          f"(val_loss={best_val:.5f})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
        self.train_time_ = time.time() - t0
        print(f"        [Transformer] Trained {epoch+1} epochs in "
              f"{self.train_time_:.1f}s on {self.device}")

    def _predict_one(self, context: np.ndarray) -> float:
        if len(context) < self.lookback:
            pad     = np.full(self.lookback - len(context), context[0],
                              dtype=np.float32)
            context = np.concatenate([pad, context])
        ctx   = self._scale(context[-self.lookback:].astype(np.float32))
        xb    = torch.tensor(ctx).unsqueeze(0).unsqueeze(-1).to(self.device)
        dec_x = torch.zeros(1, self.horizon, 1, device=self.device)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(xb, dec_x)[0, 0].item()
        return float(self._unscale(pred))

    def predict_batch(self, raw: np.ndarray,
                      raw_split: int, n: int,
                      chunk: int = 512) -> np.ndarray:
        """
        Batched GPU inference — replaces the per-step loop in the runner.
        See InformerForecaster.predict_batch for full documentation.
        """
        contexts = np.empty((n, self.lookback), dtype=np.float32)
        for i in range(n):
            start = max(0, raw_split + i - self.lookback)
            ctx   = raw[start: raw_split + i].astype(np.float32)
            if len(ctx) < self.lookback:
                pad = np.full(self.lookback - len(ctx), ctx[0], dtype=np.float32)
                ctx = np.concatenate([pad, ctx])
            contexts[i] = self._scale(ctx[-self.lookback:])

        self.model_.eval()
        preds = np.empty(n, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, n, chunk):
                end   = min(start + chunk, n)
                xb    = torch.from_numpy(contexts[start:end]
                                         ).unsqueeze(-1).to(self.device)
                dec_x = torch.zeros(end - start, self.horizon, 1,
                                    device=self.device)
                out   = self.model_(xb, dec_x)[:, 0].cpu().numpy()
                preds[start:end] = self._unscale(out)

        return preds
