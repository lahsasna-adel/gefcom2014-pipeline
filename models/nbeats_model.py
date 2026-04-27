"""
models/nbeats_model.py
----------------------
N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
Oreshkin et al. (2020) — ICLR 2020

Architecture
------------
N-BEATS is a pure deep learning architecture with NO recurrence, NO convolution,
and NO attention. It achieves state-of-the-art results through:

  1. Doubly Residual Stacking
     The network is organised as a stack of blocks. Each block produces:
       - backcast : reconstruction of the lookback window (subtracted from input)
       - forecast : contribution to the horizon prediction (summed at output)

  2. Basis Expansion
     Each block projects its hidden representation onto a set of basis functions:
       - Generic   : learnable basis (data-driven)
       - Trend     : polynomial basis (t, t², t³, ...) — interpretable
       - Seasonality: Fourier basis (sin/cos at harmonic frequencies) — interpretable

  3. No exogenous inputs needed
     N-BEATS operates purely on the lookback series (univariate).

Stack design used here (matches GEFCom2014 hourly demand):
  - Stack 0: Trend     (degree 3 polynomial basis)
  - Stack 1: Seasonality (Fourier basis with H=12 harmonics)
  - Stack 2: Generic   (fully learnable basis)

Integration notes:
  - Univariate model (raw series only) — listed in UNIVARIATE_MODELS
  - lookback  : 168 h (1 week) — standard for N-BEATS (Oreshkin et al. use 2-7×H)
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
# Basis generators
# ──────────────────────────────────────────────────────────────────────────────

def _trend_basis(degree: int, horizon: int,
                 device: torch.device) -> torch.Tensor:
    """
    Polynomial basis matrix  T ∈ R^{horizon × (degree+1)}.
    Columns: [t^0, t^1, ..., t^degree]  where t ∈ [0, 1].
    """
    t = torch.linspace(0, 1, horizon, device=device)          # (H,)
    T = torch.stack([t ** i for i in range(degree + 1)], dim=1)  # (H, deg+1)
    return T


def _trend_basis_back(degree: int, lookback: int,
                      device: torch.device) -> torch.Tensor:
    """Same polynomial basis but for backcast window."""
    t = torch.linspace(0, 1, lookback, device=device)
    return torch.stack([t ** i for i in range(degree + 1)], dim=1)


def _seasonality_basis(n_harmonics: int, horizon: int,
                       device: torch.device) -> torch.Tensor:
    """
    Fourier basis matrix  S ∈ R^{horizon × 2H}.
    Columns: [cos(2π·1·t/H), sin(2π·1·t/H), ..., cos(2π·H·t/H), sin(2π·H·t/H)]
    where t ∈ {0, ..., horizon-1}.
    """
    t     = torch.arange(horizon, device=device, dtype=torch.float32)
    freqs = torch.arange(1, n_harmonics + 1, device=device, dtype=torch.float32)
    # outer product: (H, n_harmonics)
    angles = 2 * math.pi * freqs.unsqueeze(0) * t.unsqueeze(1) / horizon
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)  # (H, 2*n_harm)


def _seasonality_basis_back(n_harmonics: int, lookback: int,
                             device: torch.device) -> torch.Tensor:
    t      = torch.arange(lookback, device=device, dtype=torch.float32)
    freqs  = torch.arange(1, n_harmonics + 1, device=device, dtype=torch.float32)
    angles = 2 * math.pi * freqs.unsqueeze(0) * t.unsqueeze(1) / lookback
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# N-BEATS Block
# ──────────────────────────────────────────────────────────────────────────────

class NBeatsBlock(nn.Module):
    """
    Single N-BEATS block (Oreshkin et al., 2020 §3.3).

    Input  : residual backcast of length `lookback`
    Output : (backcast, forecast) — both expressed in the block's basis

    block_type ∈ {'generic', 'trend', 'seasonality'}
    """

    def __init__(self, lookback: int, horizon: int,
                 hidden_size: int, n_layers: int,
                 block_type: str = "generic",
                 trend_degree: int = 3,
                 n_harmonics: int = 12,
                 n_basis: int = 16):
        super().__init__()
        self.lookback   = lookback
        self.horizon    = horizon
        self.block_type = block_type

        # Shared fully-connected stack
        layers = [nn.Linear(lookback, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.fc_stack = nn.Sequential(*layers)

        # Basis-specific output heads
        if block_type == "trend":
            self.n_theta = trend_degree + 1
        elif block_type == "seasonality":
            self.n_theta = 2 * n_harmonics
        else:
            self.n_theta = n_basis               # generic: learnable basis

        # θ for backcast and forecast (separate heads)
        self.theta_back = nn.Linear(hidden_size, self.n_theta, bias=False)
        self.theta_fore = nn.Linear(hidden_size, self.n_theta, bias=False)

        # Static parameters for structured blocks
        self.trend_degree = trend_degree
        self.n_harmonics  = n_harmonics

        if block_type == "generic":
            # Generic: learnable basis matrices
            self.basis_back = nn.Linear(self.n_theta, lookback, bias=False)
            self.basis_fore = nn.Linear(self.n_theta, horizon,  bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x : (B, lookback)
        Returns (backcast, forecast), each (B, lookback) and (B, horizon)
        """
        h      = self.fc_stack(x)
        th_b   = self.theta_back(h)   # (B, n_theta)
        th_f   = self.theta_fore(h)   # (B, n_theta)
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
# N-BEATS Stack
# ──────────────────────────────────────────────────────────────────────────────

class NBeatsStack(nn.Module):
    """A stack is a sequence of blocks of the same type."""

    def __init__(self, n_blocks: int, **block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(**block_kwargs) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor):
        """
        x : (B, lookback) — residual input
        Returns (residual_x, stack_forecast)
        """
        stack_fc = 0
        for block in self.blocks:
            backcast, forecast = block(x)
            x        = x - backcast
            stack_fc = stack_fc + forecast
        return x, stack_fc


# ──────────────────────────────────────────────────────────────────────────────
# Full N-BEATS model
# ──────────────────────────────────────────────────────────────────────────────

class NBeatsModel(nn.Module):
    """
    N-BEATS with 3 stacks: Trend → Seasonality → Generic.

    Default config for GEFCom2014 hourly electricity demand:
      lookback=168 (1 week), horizon=24 (1 day)
      Each stack: 3 blocks, hidden=256, 4 FC layers per block
      Trend: degree-3 polynomial
      Seasonality: 12 Fourier harmonics (period = horizon = 24h)
      Generic: 16 learnable basis functions
    """

    def __init__(self, lookback: int = 168, horizon: int = 24,
                 hidden_size: int = 256, n_layers: int = 4,
                 n_blocks_per_stack: int = 3,
                 trend_degree: int = 3, n_harmonics: int = 12,
                 n_generic_basis: int = 16):
        super().__init__()
        common = dict(lookback=lookback, horizon=horizon,
                      hidden_size=hidden_size, n_layers=n_layers)

        self.trend_stack = NBeatsStack(
            n_blocks_per_stack, block_type="trend",
            trend_degree=trend_degree, **common
        )
        self.season_stack = NBeatsStack(
            n_blocks_per_stack, block_type="seasonality",
            n_harmonics=n_harmonics, **common
        )
        self.generic_stack = NBeatsStack(
            n_blocks_per_stack, block_type="generic",
            n_basis=n_generic_basis, **common
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, lookback)
        Returns forecast : (B, horizon)
        """
        x, fc_trend  = self.trend_stack(x)
        x, fc_season = self.season_stack(x)
        _, fc_generic= self.generic_stack(x)
        return fc_trend + fc_season + fc_generic


# ──────────────────────────────────────────────────────────────────────────────
# Sequence dataset helper
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

class NBeatsForecaster:
    """
    Scikit-learn style wrapper around NBeatsModel.

    Hyperparameters
    ---------------
    lookback           : 168 h — 7× horizon, matches Oreshkin et al. recommendation
    horizon            : 24  h
    hidden_size        : 256 — wider than Informer/Transformer (no attention overhead)
    n_layers           : 4   — FC layers per block
    n_blocks_per_stack : 3   — blocks per stack (trend / seasonality / generic)
    trend_degree       : 3   — cubic polynomial basis
    n_harmonics        : 12  — Fourier harmonics (captures up to 2h period in 24h)
    n_generic_basis    : 16  — learnable generic basis size
    epochs             : 50  — more epochs: N-BEATS is simpler per step than Transformer
    patience           : 8
    batch_size         : 128 — larger batch: no sequence of hidden states needed
    lr                 : 1e-3

    Design rationale
    ----------------
    N-BEATS uses a doubly-residual architecture: each block subtracts its
    backcast from the input, so earlier stacks (trend, seasonality) handle
    structured components and the generic stack handles residual patterns.
    This decomposition makes the model interpretable AND accurate.
    """

    def __init__(self, lookback: int = 168, horizon: int = 24,
                 hidden_size: int = 256, n_layers: int = 4,
                 n_blocks_per_stack: int = 3,
                 trend_degree: int = 3, n_harmonics: int = 12,
                 n_generic_basis: int = 16,
                 epochs: int = 50, patience: int = 8,
                 batch_size: int = 128, lr: float = 1e-3,
                 use_log_norm: bool = True):
        self.lookback           = lookback
        self.horizon            = horizon
        self.hidden_size        = hidden_size
        self.n_layers           = n_layers
        self.n_blocks_per_stack = n_blocks_per_stack
        self.trend_degree       = trend_degree
        self.n_harmonics        = n_harmonics
        self.n_generic_basis    = n_generic_basis
        self.epochs             = epochs
        self.patience           = patience
        self.batch_size         = batch_size
        self.lr                 = lr
        self.use_log_norm       = use_log_norm   # Smyl & Hua (2019): log(x/level)

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
            print(f"        [N-BEATS] Log-norm: level={self.level_:.2f} "
                  f"(Smyl & Hua 2019 §3.1.3)")
        else:
            self.mu_    = np.float32(raw_train.mean())
            self.sigma_ = np.float32(raw_train.std())
        series = self._scale(raw_train).astype(np.float32)

        X, y  = _make_sequences(series, self.lookback, self.horizon)
        n_val = max(1, int(len(X) * 0.10))
        X_tr, X_va = X[:-n_val], X[-n_val:]
        y_tr, y_va = y[:-n_val], y[-n_val:]

        dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr),
                                         torch.from_numpy(y_tr)),
                           batch_size=self.batch_size, shuffle=True)
        dl_va = DataLoader(TensorDataset(torch.from_numpy(X_va),
                                         torch.from_numpy(y_va)),
                           batch_size=self.batch_size)

        self.model_ = NBeatsModel(
            lookback=self.lookback, horizon=self.horizon,
            hidden_size=self.hidden_size, n_layers=self.n_layers,
            n_blocks_per_stack=self.n_blocks_per_stack,
            trend_degree=self.trend_degree, n_harmonics=self.n_harmonics,
            n_generic_basis=self.n_generic_basis,
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
            for xb, yb in dl_tr:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimiser.step()
            scheduler.step()

            self.model_.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    val_losses.append(criterion(self.model_(xb), yb).item())
            val_loss = np.mean(val_losses)

            if val_loss < best_val - 1e-6:
                best_val   = val_loss
                no_improve = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model_.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"        [N-BEATS] Early stop epoch {epoch+1} "
                          f"(val_loss={best_val:.5f})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
        self.train_time_ = time.time() - t0
        print(f"        [N-BEATS] Trained {epoch+1} epochs in "
              f"{self.train_time_:.1f}s on {self.device}")

    def _predict_one(self, context: np.ndarray) -> float:
        if len(context) < self.lookback:
            pad     = np.full(self.lookback - len(context), context[0],
                              dtype=np.float32)
            context = np.concatenate([pad, context])
        ctx = self._scale(context[-self.lookback:].astype(np.float32))
        xb  = torch.tensor(ctx).unsqueeze(0).to(self.device)   # (1, lookback)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(xb)[0, 0].item()
        return float(self._unscale(pred))

    def predict_batch(self, raw: np.ndarray,
                      raw_split: int, n: int,
                      chunk: int = 512) -> np.ndarray:
        """
        Batched GPU inference — replaces the per-step loop in the runner.

        N-BEATS has no encoder/decoder split so the forward pass is simply
        model(xb) → (B, horizon). We take [:, 0] (first horizon step).
        chunk=512 is safe for lookback=168, hidden=256 on T4 VRAM.
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
                end  = min(start + chunk, n)
                xb   = torch.from_numpy(contexts[start:end]).to(self.device)
                out  = self.model_(xb)[:, 0].cpu().numpy()
                preds[start:end] = self._unscale(out)

        return preds
