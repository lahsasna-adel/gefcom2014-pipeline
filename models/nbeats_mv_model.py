"""
models/nbeats_mv_model.py
--------------------------
Multivariate N-BEATS — Phase 4b extension.

N-BEATS was originally designed as a pure univariate model with no external
inputs. This extension adds a feature-conditioning mechanism: the engineered
features are compressed by a small MLP into a context vector that is added
to the input of every block (FiLM-style feature injection).

Design choice — FiLM conditioning (Feature-wise Linear Modulation)
-------------------------------------------------------------------
Rather than concatenating F features to each lookback timestep (which would
change the basis expansion dimensions and break the elegant N-BEATS structure),
we use a lightweight affine conditioning:

    h_block_input = γ(z) ⊙ x + β(z)

where z = MLP(mean(X_features)) is a global feature summary vector and
γ, β are per-block learned affine parameters. This preserves the doubly-
residual stack structure while allowing the model to be informed by the
feature matrix.

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

from models.nbeats_model import (
    NBeatsBlock,
    _trend_basis, _trend_basis_back,
    _seasonality_basis, _seasonality_basis_back,
    _make_sequences,
)


# ──────────────────────────────────────────────────────────────────────────────
# FiLM conditioning module
# ──────────────────────────────────────────────────────────────────────────────

class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation conditioner.

    Maps a feature vector z (global mean of X_features) → (γ, β) pair
    that scales and shifts the N-BEATS block input.

    z     : (B, n_features)
    γ, β  : (B, lookback)
    """

    def __init__(self, n_features: int, lookback: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * lookback),
        )

    def forward(self, z: torch.Tensor):
        out = self.net(z)                              # (B, 2*lookback)
        gamma, beta = out.chunk(2, dim=-1)             # each (B, lookback)
        return gamma, beta


# ──────────────────────────────────────────────────────────────────────────────
# FiLM-conditioned N-BEATS block
# ──────────────────────────────────────────────────────────────────────────────

class NBeatsMVBlock(nn.Module):
    """
    N-BEATS block with FiLM conditioning.

    Applies γ ⊙ x + β to the residual input before the FC stack,
    allowing the feature context to modulate what each block attends to.
    """

    def __init__(self, lookback: int, horizon: int,
                 hidden_size: int, n_layers: int,
                 n_features: int, film_hidden: int = 64,
                 block_type: str = "generic",
                 trend_degree: int = 3,
                 n_harmonics: int = 12,
                 n_basis: int = 16):
        super().__init__()
        self.lookback   = lookback
        self.horizon    = horizon
        self.block_type = block_type

        self.film = FiLMConditioner(n_features, lookback, film_hidden)

        layers = [nn.Linear(lookback, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.fc_stack = nn.Sequential(*layers)

        if block_type == "trend":
            self.n_theta = trend_degree + 1
        elif block_type == "seasonality":
            self.n_theta = 2 * n_harmonics
        else:
            self.n_theta = n_basis

        self.theta_back = nn.Linear(hidden_size, self.n_theta, bias=False)
        self.theta_fore = nn.Linear(hidden_size, self.n_theta, bias=False)

        self.trend_degree = trend_degree
        self.n_harmonics  = n_harmonics

        if block_type == "generic":
            self.basis_back = nn.Linear(self.n_theta, lookback, bias=False)
            self.basis_fore = nn.Linear(self.n_theta, horizon,  bias=False)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        x : (B, lookback)   residual demand series
        z : (B, n_features) global feature context (mean of X_features)
        """
        gamma, beta = self.film(z)
        x_cond = gamma * x + beta                     # FiLM modulation

        h    = self.fc_stack(x_cond)
        th_b = self.theta_back(h)
        th_f = self.theta_fore(h)
        device = x.device

        if self.block_type == "generic":
            backcast = self.basis_back(th_b)
            forecast = self.basis_fore(th_f)
        elif self.block_type == "trend":
            T_fore = _trend_basis(self.trend_degree, self.horizon, device)
            T_back = _trend_basis_back(self.trend_degree, self.lookback, device)
            backcast = th_b @ T_back.T
            forecast = th_f @ T_fore.T
        elif self.block_type == "seasonality":
            S_fore = _seasonality_basis(self.n_harmonics, self.horizon, device)
            S_back = _seasonality_basis_back(self.n_harmonics, self.lookback, device)
            backcast = th_b @ S_back.T
            forecast = th_f @ S_fore.T

        return backcast, forecast


# ──────────────────────────────────────────────────────────────────────────────
# Full MV N-BEATS model
# ──────────────────────────────────────────────────────────────────────────────

class NBeatsMVModel(nn.Module):
    """
    N-BEATS with FiLM feature conditioning.
    3 stacks: Trend → Seasonality → Generic (same as univariate version).
    Each block receives the global feature summary z = mean(X_features).
    """

    def __init__(self, n_features: int,
                 lookback: int = 168, horizon: int = 24,
                 hidden_size: int = 256, n_layers: int = 4,
                 n_blocks_per_stack: int = 3,
                 trend_degree: int = 3, n_harmonics: int = 12,
                 n_generic_basis: int = 16, film_hidden: int = 64):
        super().__init__()
        common = dict(lookback=lookback, horizon=horizon,
                      hidden_size=hidden_size, n_layers=n_layers,
                      n_features=n_features, film_hidden=film_hidden)

        def _make_stack(block_type, **kwargs):
            return nn.ModuleList([
                NBeatsMVBlock(block_type=block_type, **common, **kwargs)
                for _ in range(n_blocks_per_stack)
            ])

        self.trend_blocks   = _make_stack("trend",       trend_degree=trend_degree)
        self.season_blocks  = _make_stack("seasonality", n_harmonics=n_harmonics)
        self.generic_blocks = _make_stack("generic",     n_basis=n_generic_basis)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x : (B, lookback)    scaled demand series
        z : (B, n_features)  scaled feature mean
        Returns forecast : (B, horizon)
        """
        total_fc = 0
        for block in [*self.trend_blocks,
                       *self.season_blocks,
                       *self.generic_blocks]:
            backcast, forecast = block(x, z)
            x        = x - backcast
            total_fc = total_fc + forecast
        return total_fc


# ──────────────────────────────────────────────────────────────────────────────
# Forecaster wrapper
# ──────────────────────────────────────────────────────────────────────────────

class NBeatsMVForecaster:
    """
    Multivariate N-BEATS forecaster — pipeline-compatible wrapper.

    The feature matrix is summarised as its row-wise mean before being passed
    to each block's FiLM conditioner. This gives the model a global sense of
    the calendar, lag, and meteorological context without changing the
    lookback-length backbone architecture.
    """

    def __init__(self, lookback: int = 168, horizon: int = 24,
                 hidden_size: int = 256, n_layers: int = 4,
                 n_blocks_per_stack: int = 3,
                 trend_degree: int = 3, n_harmonics: int = 12,
                 n_generic_basis: int = 16, film_hidden: int = 64,
                 epochs: int = 50, patience: int = 8,
                 batch_size: int = 128, lr: float = 1e-3):
        self.lookback           = lookback
        self.horizon            = horizon
        self.hidden_size        = hidden_size
        self.n_layers           = n_layers
        self.n_blocks_per_stack = n_blocks_per_stack
        self.trend_degree       = trend_degree
        self.n_harmonics        = n_harmonics
        self.n_generic_basis    = n_generic_basis
        self.film_hidden        = film_hidden
        self.epochs             = epochs
        self.patience           = patience
        self.batch_size         = batch_size
        self.lr                 = lr

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

        # Build sliding-window sequences
        # x_seq: (n, lookback) demand windows
        # z_seq: (n, F)        mean features over the lookback window
        # y_seq: (n, horizon)
        N = len(demand_s)
        n_samples = N - self.lookback - self.horizon + 1
        F = X_arr.shape[1]

        x_seq = np.empty((n_samples, self.lookback), dtype=np.float32)
        z_seq = np.empty((n_samples, F),             dtype=np.float32)
        y_seq = np.empty((n_samples, self.horizon),  dtype=np.float32)

        for i in range(n_samples):
            x_seq[i] = demand_s[i: i + self.lookback]
            z_seq[i] = features_s[i: i + self.lookback].mean(axis=0)
            y_seq[i] = demand_s[i + self.lookback: i + self.lookback + self.horizon]

        n_val   = max(1, int(n_samples * 0.10))
        x_tr, x_va = x_seq[:-n_val], x_seq[-n_val:]
        z_tr, z_va = z_seq[:-n_val], z_seq[-n_val:]
        y_tr, y_va = y_seq[:-n_val], y_seq[-n_val:]

        dl_tr = DataLoader(
            TensorDataset(torch.from_numpy(x_tr),
                          torch.from_numpy(z_tr),
                          torch.from_numpy(y_tr)),
            batch_size=self.batch_size, shuffle=True
        )
        dl_va = DataLoader(
            TensorDataset(torch.from_numpy(x_va),
                          torch.from_numpy(z_va),
                          torch.from_numpy(y_va)),
            batch_size=self.batch_size
        )

        self.model_ = NBeatsMVModel(
            n_features=F,
            lookback=self.lookback, horizon=self.horizon,
            hidden_size=self.hidden_size, n_layers=self.n_layers,
            n_blocks_per_stack=self.n_blocks_per_stack,
            trend_degree=self.trend_degree, n_harmonics=self.n_harmonics,
            n_generic_basis=self.n_generic_basis,
            film_hidden=self.film_hidden,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                                     weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.epochs, eta_min=1e-5
        )
        criterion = nn.HuberLoss(delta=1.0)

        best_val   = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, zb, yb in dl_tr:
                xb = xb.to(self.device)
                zb = zb.to(self.device)
                yb = yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.model_(xb, zb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimiser.step()
            scheduler.step()

            self.model_.eval()
            val_losses = []
            with torch.no_grad():
                for xb, zb, yb in dl_va:
                    xb = xb.to(self.device)
                    zb = zb.to(self.device)
                    yb = yb.to(self.device)
                    val_losses.append(criterion(self.model_(xb, zb), yb).item())
            val_loss = np.mean(val_losses)

            if val_loss < best_val - 1e-6:
                best_val   = val_loss
                no_improve = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model_.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"        [N-BEATS-MV] Early stop epoch {epoch+1} "
                          f"(val_loss={best_val:.5f})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
        self.train_time_ = time.time() - t0
        print(f"        [N-BEATS-MV] {F} features | "
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

        x_ctx = np.empty((n, self.lookback), dtype=np.float32)
        z_ctx = np.empty((n, F),             dtype=np.float32)

        for i in range(n):
            # Demand window
            d_start = max(0, raw_split + i - self.lookback)
            d_ctx   = demand_history[d_start: raw_split + i]
            if len(d_ctx) < self.lookback:
                pad   = np.full(self.lookback - len(d_ctx),
                                d_ctx[0], dtype=np.float32)
                d_ctx = np.concatenate([pad, d_ctx])
            x_ctx[i] = d_ctx[-self.lookback:]

            # Feature window mean
            f_start = i - self.lookback
            if f_start >= 0:
                f_win = X_s[f_start: i]
            else:
                available = X_s[:i] if i > 0 else np.zeros((0, F),
                                                             dtype=np.float32)
                pad_rows  = self.lookback - len(available)
                f_win     = np.concatenate([
                    np.zeros((pad_rows, F), dtype=np.float32), available
                ], axis=0)
            z_ctx[i] = f_win.mean(axis=0) if len(f_win) > 0 \
                       else np.zeros(F, dtype=np.float32)

        self.model_.eval()
        preds = np.empty(n, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, n, chunk):
                end  = min(start + chunk, n)
                xb   = torch.from_numpy(x_ctx[start:end]).to(self.device)
                zb   = torch.from_numpy(z_ctx[start:end]).to(self.device)
                out  = self.model_(xb, zb)[:, 0].cpu().numpy()
                preds[start:end] = self._unscale_demand(out)

        return preds
