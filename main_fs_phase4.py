"""
main_fs.py
----------
Feature Selection Experiment Orchestrator — Extended Study.

Runs the complete experiment study:
  13 models × 9 feature subsets × 10 CV folds

Original models (Phase 1):
  lgb   — LightGBM
  xgb   — XGBoost
  rf    — Random Forest
  lstm  — LSTM (raw series, feature-insensitive)
  arima — SARIMA(1,1,2)(1,1,1)[24]

New models added (Phase 2):
  gam      — Generalised Additive Model (pyGAM, LinearGAM)
  ridge    — Ridge Regression (L2)
  lasso    — Lasso Regression (L1)
  enet     — Elastic Net (L1 + L2)
  svr      — Support Vector Regression (RBF kernel)
  mlp      — Multi-Layer Perceptron (sklearn)
  ets      — Error–Trend–Seasonality (statsmodels ETSModel)
  naive    — Naïve baseline (lag-1: ŷ_t = y_{t-1})
  rnn      — Vanilla RNN (SimpleRNN, lookback=48h)
  gru      — Gated Recurrent Unit (lookback=48h)
  cnn      — 1D CNN with dilated causal convolutions (lookback=48h)

New models added (Phase 4 — Advanced Deep Learning):
  informer    — Informer (Zhou et al., 2021): ProbSparse attention + distilling
                O(L log L) encoder, generative decoder. lookback=168h.
  transformer — Vanilla Transformer (Vaswani et al., 2017): full O(L²) attention.
                Ablation baseline for Informer. lookback=168h.
  nbeats      — N-BEATS (Oreshkin et al., 2020): doubly-residual stacking with
                trend / seasonality / generic basis expansion. lookback=168h.

Notes on univariate models (arima / ets / naive / lstm):
  These models ignore the feature matrix — they operate on the raw demand
  series only. They are therefore subset-independent: results are identical
  across all 9 feature subsets. The guard clause in run_cv_for_subset
  skips the redundant 8 extra runs and copies results automatically.

CV splitting strategy (custom fixed-horizon expanding window):
  - Total period    : ~7 years (GEFCom2014 2005-2011)
  - Horizon         : 4 months = 2,920 rows (fixed, identical across all folds)
  - Number of folds : 10
  - Fold 1 training : 3.55 years (31,064 rows) — satisfies >3 year minimum
  - Last fold end   : last row of dataset (no data wasted)
  - All folds valid : no fold skipped (unlike sklearn TimeSeriesSplit)

Usage
-----
    # Run only new models (skip feature selection, keep existing results)
    python main_fs.py --csv GEFCom2014_clean.csv --skip-fs \\
                      --models gam ridge lasso enet svr mlp ets naive

    # Run only the Phase 4 DL models (skip feature selection)
    python main_fs.py --csv GEFCom2014_clean.csv --skip-fs \\
                      --models informer transformer nbeats

    # Run all 13 models from scratch
    python main_fs.py --csv GEFCom2014_clean.csv \\
                      --models xgb lgb rf lstm arima gam ridge lasso enet svr mlp ets naive

    # Run only the linear models (fast, ~minutes)
    python main_fs.py --csv GEFCom2014_clean.csv --skip-fs \\
                      --models ridge lasso enet

Outputs (saved to results/fs/)
-------------------------------
    fs_scores.csv          — raw filter scores per method
    fs_ranks.csv           — ranks per method
    fs_summary.csv         — aggregated rankings (primary result)
    fs_experiments.csv     — all experiment results (one row per fold)
    fs_cv_summary.csv      — mean ± std per model × subset
    fs_performance_curve.png
    fs_heatmap.png
    fs_importance_plot.png
"""

import argparse
import warnings
import time
import os
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from utils.data_loader               import load_csv
from utils.feature_engineering_full  import build_full_feature_matrix
from utils.feature_selection         import (run_filter_selection,
                                              define_feature_subsets,
                                              save_selection_results)
from utils.metrics                   import evaluate


# ─── Model category registry ──────────────────────────────────────────────────
# Maps model_key → whether it is univariate (uses raw series, not feature matrix)
# Univariate models are subset-independent — run once per fold, copy across subsets.
UNIVARIATE_MODELS = {"arima", "ets", "naive", "lstm", "rnn", "gru", "cnn",
                     "informer", "transformer", "nbeats"}   # Phase 4

# Representative subset to actually run for univariate models
UNIVARIATE_ANCHOR_SUBSET = "all"


# ─── Custom CV splitter ───────────────────────────────────────────────────────

def make_custom_folds(N: int,
                      n_folds: int = 10,
                      horizon: int = 2920) -> list:
    """
    Build custom fixed-horizon expanding-window CV folds.

    Design
    ------
    - Test windows are non-overlapping, each exactly `horizon` rows
    - Test windows are placed from the END of the dataset backwards
    - Fold 10 test ends at the last row (no data wasted)
    - Fold 1 trains on everything before its test window

    Parameters
    ----------
    N       : total number of rows
    n_folds : number of folds
    horizon : test window size in rows (2920 = 4 months of hourly data)

    Returns
    -------
    list of (train_indices, test_indices) tuples, fold 1 first
    """
    folds = []
    for i in range(n_folds - 1, -1, -1):
        test_end   = N - i * horizon
        test_start = test_end - horizon
        if test_start <= 0:
            continue
        train_idx = np.arange(0, test_start)
        test_idx  = np.arange(test_start, test_end)
        folds.append((train_idx, test_idx))
    return folds


# ─── Individual model runners ─────────────────────────────────────────────────

def _run_xgb(X_tr, y_train, X_te, y_test):
    from models.tree_models import XGBoostForecaster
    m = XGBoostForecaster()
    m.fit(X_tr, y_train, X_te, y_test)
    return evaluate(y_test.values, m.predict(X_te), "XGBoost",
                    train_time_s=m.train_time_)


def _run_lgb(X_tr, y_train, X_te, y_test):
    from models.tree_models import LightGBMForecaster
    m = LightGBMForecaster()
    m.fit(X_tr, y_train, X_te, y_test)
    return evaluate(y_test.values, m.predict(X_te), "LightGBM",
                    train_time_s=m.train_time_)


def _run_rf(X_tr, y_train, X_te, y_test):
    from models.tree_models import RandomForestForecaster
    m = RandomForestForecaster()
    m.fit(X_tr, y_train)
    return evaluate(y_test.values, m.predict(X_te), "RandomForest",
                    train_time_s=m.train_time_)


def _run_lstm(raw_train, raw_test, raw_split, raw):
    from models.lstm_model import LSTMForecaster
    m = LSTMForecaster(lookback=48, horizon=24, epochs=30, patience=5)
    m.fit(raw_train)
    preds = np.array([
        m._predict_one(raw[max(0, raw_split + i - 48): raw_split + i])
        for i in range(len(raw_test))
    ])
    return evaluate(raw_test, preds, "LSTM", train_time_s=m.train_time_)


def _run_rnn(raw_train, raw_test, raw_split, raw):
    from models.rnn_model import RNNForecaster
    lookback = 168
    m = RNNForecaster(lookback=lookback, horizon=24, epochs=50, patience=8)
    m.fit(raw_train)
    preds = np.array([
        m._predict_one(raw[max(0, raw_split + i - lookback): raw_split + i])
        for i in range(len(raw_test))
    ])
    return evaluate(raw_test, preds, "RNN", train_time_s=m.train_time_)


def _run_gru(raw_train, raw_test, raw_split, raw):
    from models.gru_model import GRUForecaster
    lookback=168
    m = GRUForecaster(lookback=lookback, horizon=24, epochs=50, patience=8)
    m.fit(raw_train)
    preds = np.array([
        m._predict_one(raw[max(0, raw_split + i - lookback): raw_split + i])
        for i in range(len(raw_test))
    ])
    return evaluate(raw_test, preds, "GRU", train_time_s=m.train_time_)


def _run_cnn(raw_train, raw_test, raw_split, raw):
    from models.cnn_model import CNNForecaster
    m = CNNForecaster(lookback=48, horizon=24, epochs=30, patience=5)
    m.fit(raw_train)
    preds = np.array([
        m._predict_one(raw[max(0, raw_split + i - 48): raw_split + i])
        for i in range(len(raw_test))
    ])
    return evaluate(raw_test, preds, "CNN", train_time_s=m.train_time_)


# ─── Phase 4: Advanced Deep Learning runners ──────────────────────────────────

def _run_informer(raw_train, raw_test, raw_split, raw):
    """
    Informer (Zhou et al., 2021) — ProbSparse self-attention.

    Uses lookback=168h (1 week) to give ProbSparse attention enough context
    to identify dominant query positions. The generative decoder predicts
    the full 24h horizon in one forward pass (no autoregressive rollout).

    Hyperparameters are conservative for stability on GEFCom2014 scale:
      d_model=64, n_heads=4, d_ff=256, n_enc_layers=2, epochs=30.
    """
    from models.informer_model import InformerForecaster
    lookback = 168
    m = InformerForecaster(
        lookback=lookback, horizon=24,
        d_model=64, n_heads=4, d_ff=256,
        n_enc_layers=2, epochs=30, patience=5,
        batch_size=64, lr=1e-3,
    )
    m.fit(raw_train)
    preds = np.array([
        m._predict_one(raw[max(0, raw_split + i - lookback): raw_split + i])
        for i in range(len(raw_test))
    ])
    return evaluate(raw_test, preds, "Informer", train_time_s=m.train_time_)


def _run_transformer(raw_train, raw_test, raw_split, raw):
    """
    Vanilla Transformer (Vaswani et al., 2017) — full O(L²) attention.

    Identical hyperparameters to Informer (d_model=64, lookback=168h)
    to serve as a clean ablation baseline: isolates the effect of
    ProbSparse attention + distilling vs standard self-attention.
    """
    from models.transformer_model import TransformerForecaster
    lookback = 168
    m = TransformerForecaster(
        lookback=lookback, horizon=24,
        d_model=64, n_heads=4, d_ff=256,
        n_enc_layers=2, n_dec_layers=1,
        epochs=30, patience=5,
        batch_size=64, lr=1e-3,
    )
    m.fit(raw_train)
    preds = np.array([
        m._predict_one(raw[max(0, raw_split + i - lookback): raw_split + i])
        for i in range(len(raw_test))
    ])
    return evaluate(raw_test, preds, "Transformer", train_time_s=m.train_time_)


def _run_nbeats(raw_train, raw_test, raw_split, raw):
    """
    N-BEATS (Oreshkin et al., 2020) — doubly-residual basis expansion.

    Stack design: Trend (degree-3 poly) → Seasonality (12 Fourier harmonics)
    → Generic (16 learnable basis functions).

    lookback=168h (7× horizon) follows Oreshkin et al.'s recommendation of
    2–7× horizon. No attention, no recurrence: faster per epoch than
    Informer/Transformer. Uses CosineAnnealingLR scheduler and 50 epochs
    to allow the residual stack to converge gradually.
    """
    from models.nbeats_model import NBeatsForecaster
    lookback = 168
    m = NBeatsForecaster(
        lookback=lookback, horizon=24,
        hidden_size=256, n_layers=4,
        n_blocks_per_stack=3,
        trend_degree=3, n_harmonics=12,
        n_generic_basis=16,
        epochs=50, patience=8,
        batch_size=128, lr=1e-3,
    )
    m.fit(raw_train)
    preds = np.array([
        m._predict_one(raw[max(0, raw_split + i - lookback): raw_split + i])
        for i in range(len(raw_test))
    ])
    return evaluate(raw_test, preds, "N-BEATS", train_time_s=m.train_time_)


def _run_arima(raw_train, raw_test):
    from models.sarima_model import SARIMAForecaster
    m = SARIMAForecaster(seasonal_period=24,
                          manual_order=(1, 1, 2),
                          manual_seasonal_order=(1, 1, 1, 24))
    m.fit(raw_train)
    preds, _ = m.predict(len(raw_test))
    return evaluate(raw_test, preds, "SARIMA", train_time_s=m.train_time_)


def _run_gam(X_tr, y_train, X_te, y_test):
    """
    Generalised Additive Model — pygam LinearGAM.

    LinearGAM fits one smooth spline term per feature:
        g(E[y]) = β₀ + f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ)
    where each fⱼ is a B-spline.

    Hyperparameters
    ---------------
    lam=0.6 : smoothing penalty (lambda). Selected via cross-validated
               grid search on a representative fold. Higher = smoother
               (less prone to overfitting on correlated lag features).
    n_splines=20 : knots per feature — suitable for hourly electricity data
                   where intra-day patterns have clear non-linearities.
    """
    from pygam import LinearGAM, s
    import time

    # Build spline terms for all features: s(0) + s(1) + ... + s(p-1)
    Xtr_arr = X_tr.values
    Xte_arr = X_te.values
    p       = Xtr_arr.shape[1]

    # Compose term: sum of splines for each feature
    terms = s(0, n_splines=20)
    for j in range(1, p):
        terms = terms + s(j, n_splines=20)

    t0 = time.time()
    m  = LinearGAM(terms, max_iter=100, lam=0.6)
    m.fit(Xtr_arr, y_train.values)
    elapsed = time.time() - t0

    preds = np.maximum(m.predict(Xte_arr), 0)
    return evaluate(y_test.values, preds, "GAM", train_time_s=elapsed)


def _run_ridge(X_tr, y_train, X_te, y_test):
    """
    Ridge Regression (L2 regularisation).

    Objective: min ‖y − Xβ‖² + α‖β‖²
    Closed-form solution: β* = (XᵀX + αI)⁻¹Xᵀy

    Features are StandardScaled before fitting (mandatory for Ridge —
    without scaling, features with large magnitudes dominate the penalty).

    alpha=10.0 was selected via time-series cross-validation on a
    representative fold. The relatively large α reflects the high
    multicollinearity among lag and rolling features.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import time

    t0  = time.time()
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    m = Ridge(alpha=10.0, fit_intercept=True, max_iter=5000)
    m.fit(Xtr, y_train)
    elapsed = time.time() - t0

    preds = np.maximum(m.predict(Xte), 0)
    return evaluate(y_test.values, preds, "Ridge", train_time_s=elapsed)


def _run_lasso(X_tr, y_train, X_te, y_test):
    """
    Lasso Regression (L1 regularisation).

    Objective: min ‖y − Xβ‖² + α‖β‖₁
    L1 penalty drives irrelevant feature coefficients exactly to zero,
    performing implicit feature selection.

    alpha=0.1 is intentionally smaller than Ridge's alpha=10 because L1
    is a stronger sparsifying penalty per unit of alpha — the two scales
    are not directly comparable.
    """
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    import time

    t0  = time.time()
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    m = Lasso(alpha=0.1, fit_intercept=True, max_iter=10000,
              warm_start=False, selection="cyclic")
    m.fit(Xtr, y_train)
    elapsed = time.time() - t0

    preds = np.maximum(m.predict(Xte), 0)
    n_nonzero = int((m.coef_ != 0).sum())
    print(f"        [Lasso] Non-zero features: {n_nonzero}/{len(m.coef_)}")
    return evaluate(y_test.values, preds, "Lasso", train_time_s=elapsed)


def _run_enet(X_tr, y_train, X_te, y_test):
    """
    Elastic Net (combined L1 + L2 regularisation).

    Objective: min ‖y − Xβ‖² + α[ρ‖β‖₁ + (1−ρ)‖β‖²]
    where ρ = l1_ratio balances L1 (sparsity) vs L2 (stability).

    l1_ratio=0.5 applies equal weight to both penalties, which handles
    groups of correlated features better than pure Lasso (which selects
    one feature from each correlated group arbitrarily).

    alpha=0.1, l1_ratio=0.5 are standard starting values for electricity
    load data with correlated lag features (Ziel & Liu, 2016).
    """
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler
    import time

    t0  = time.time()
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    m = ElasticNet(alpha=0.1, l1_ratio=0.5,
                   fit_intercept=True, max_iter=10000,
                   selection="cyclic")
    m.fit(Xtr, y_train)
    elapsed = time.time() - t0

    preds = np.maximum(m.predict(Xte), 0)
    n_nonzero = int((m.coef_ != 0).sum())
    print(f"        [ElasticNet] Non-zero features: {n_nonzero}/{len(m.coef_)}")
    return evaluate(y_test.values, preds, "ElasticNet", train_time_s=elapsed)


def _run_svr(X_tr, y_train, X_te, y_test):
    """
    Support Vector Regression (RBF kernel).

    SVR minimises:
        min  ½‖w‖² + C Σ max(0, |yᵢ − ŷᵢ| − ε)
    where ε = insensitive tube width and C controls margin softness.

    Kernel: RBF — k(x,z) = exp(−γ‖x−z‖²)
    Hyperparameters: C=100, epsilon=0.1, gamma='scale'
      - C=100 allows flexible margin for electricity demand volatility
      - epsilon=0.1 is appropriate for normalised demand values
      - gamma='scale' = 1/(n_features × var(X)), auto-adapts to feature count

    WARNING: SVR runtime is O(n²) to O(n³) in training set size. On large
    folds (>30k rows), subsampling to the most recent MAX_SVR_ROWS rows is
    applied — demand patterns are dominated by recent seasonality.
    """
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    import time

    MAX_SVR_ROWS = 8000   # subsampling limit to keep runtime < ~10 min/fold

    Xtr_arr = X_tr.values
    ytr_arr = y_train.values

    if len(Xtr_arr) > MAX_SVR_ROWS:
        print(f"        [SVR] Subsampling training set: "
              f"{len(Xtr_arr):,} → {MAX_SVR_ROWS:,} rows (most recent)")
        Xtr_arr = Xtr_arr[-MAX_SVR_ROWS:]
        ytr_arr = ytr_arr[-MAX_SVR_ROWS:]

    t0 = time.time()
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr_arr)
    Xte_sc = sc.transform(X_te.values)

    m = SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale",
            cache_size=500)
    m.fit(Xtr_sc, ytr_arr)
    elapsed = time.time() - t0

    preds = np.maximum(m.predict(Xte_sc), 0)
    return evaluate(y_test.values, preds, "SVR", train_time_s=elapsed)


def _run_mlp(X_tr, y_train, X_te, y_test):
    """
    Multi-Layer Perceptron Regressor (sklearn).

    Architecture: Input → 128 → 64 → 32 → Output
    Activation  : ReLU (hidden layers), linear (output)
    Optimiser   : Adam (adaptive learning rate)
    Regulariser : L2 penalty α=1e-4 on all weights

    This is a feedforward (non-recurrent) MLP that operates on the
    engineered feature matrix — it is therefore sensitive to feature
    engineering, unlike LSTM which processes raw sequences.

    Early stopping (n_iter_no_change=15) prevents overfitting on
    the large training folds. StandardScaler is mandatory for MLP
    because gradient descent converges poorly with unnormalised inputs.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import time

    t0 = time.time()
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    m = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,                  # L2 regularisation
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    m.fit(Xtr, y_train)
    elapsed = time.time() - t0

    preds = np.maximum(m.predict(Xte), 0)
    best_loss_str = f"{m.best_loss_:.4f}" if m.best_loss_ is not None else "N/A"
    print(f"        [MLP] Iterations: {m.n_iter_}  Best val loss: {best_loss_str}")
    return evaluate(y_test.values, preds, "MLP", train_time_s=elapsed)


def _run_gbm(X_tr, y_train, X_te, y_test):
    """
    Histogram Gradient Boosting (sklearn HistGradientBoostingRegressor).

    Histogram-based splitting is O(n_bins) vs O(n_samples) for classic GBM,
    making it 10-100x faster on large folds. Uses ordered early stopping
    on a 10% validation split to prevent overfitting.
    """
    from models.gbm_model import GBMForecaster
    import time

    m = GBMForecaster(
        max_iter=500,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        l2_reg=0.1,
    )
    m.fit(X_tr, y_train)
    preds = m.predict(X_te)
    return evaluate(y_test.values, preds, "GBM", train_time_s=m.train_time_)


def _run_catboost(X_tr, y_train, X_te, y_test):
    """
    CatBoost gradient boosting with ordered boosting and symmetric trees.

    CatBoost's ordered boosting prevents target leakage during training —
    particularly relevant for lag and rolling features that are correlated
    with the target across nearby time steps.
    """
    from models.catboost_model import CatBoostForecaster

    m = CatBoostForecaster(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        early_stopping_rounds=50,
    )
    m.fit(X_tr.values, y_train.values)
    preds = m.predict(X_te.values)
    return evaluate(y_test.values, preds, "CatBoost",
                    train_time_s=m.train_time_)


def _run_cart(X_tr, y_train, X_te, y_test):
    """
    CART — single Decision Tree Regressor (sklearn).

    Serves as an interpretable baseline and as a reference point for the
    ensemble models (RF, XGB, LGB, GBM) built on top of CART trees.
    Expected to underperform ensembles — variance is high for a single tree.
    """
    from models.cart_model import CARTForecaster

    m = CARTForecaster(
        max_depth=10,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features="sqrt",
    )
    m.fit(X_tr, y_train)
    preds = m.predict(X_te)
    return evaluate(y_test.values, preds, "CART", train_time_s=m.train_time_)


def _run_ets(raw_train, raw_test):
    """
    Error–Trend–Seasonality (ETS) model — statsmodels ETSModel.

    ETS decomposes the time series into:
        - Error  (E): additive or multiplicative
        - Trend  (T): none / additive / multiplicative (+ damped variant)
        - Season (S): additive with period=24h

    Configuration: ETS(A, N, A) — Additive errors, No trend,
    Additive seasonality, period=24.
      - No trend: electricity demand does not have a strong deterministic
        trend within a 4-month horizon.
      - Additive seasonality avoids multiplicative instability at low
        demand values (night hours).
      - Period=24 captures the daily consumption cycle.

    Training is subsampled to the most recent MAX_ETS_ROWS rows for
    computational feasibility — ETS parameter estimation is O(n) but
    seasonal initialisation benefits from complete seasonal cycles.
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    import time

    MAX_ETS_ROWS = 8760   # 1 year — captures all seasonal patterns

    train = raw_train[-MAX_ETS_ROWS:] if len(raw_train) > MAX_ETS_ROWS else raw_train
    if len(raw_train) > MAX_ETS_ROWS:
        print(f"        [ETS] Subsampled to last {MAX_ETS_ROWS} rows (1 year)")

    t0 = time.time()
    m  = ETSModel(
        train,
        error="add",
        trend=None,
        seasonal="add",
        seasonal_periods=24,
        initialization_method="estimated",
    )
    fitted = m.fit(disp=False, maxiter=100)
    elapsed = time.time() - t0

    fc    = fitted.forecast(steps=len(raw_test))
    preds = np.maximum(np.array(fc), 0)
    return evaluate(raw_test, preds, "ETS", train_time_s=elapsed)


def _run_naive(raw_train, raw_test):
    """
    Naïve baseline model — lag-1 (persistence forecast).

    ŷ_t = y_{t-1}

    The simplest possible forecast: predict that tomorrow's demand equals
    today's demand. This is the standard seasonal naïve baseline for
    electricity forecasting (Hyndman & Athanasopoulos, 2021 §5.2).

    MASE = 1.0 exactly when a model equals this baseline in MAE.
    Any model with MASE < 1.0 outperforms the naïve forecast.

    No training is needed — the last observed value from raw_train
    seeds the first forecast, then each test observation becomes the
    next seed (true lag-1 persistence across the test horizon).
    """
    import time
    t0 = time.time()

    # Seed with the last training value, then walk forward
    preds = np.empty(len(raw_test))
    prev  = raw_train[-1]
    for i, true_val in enumerate(raw_test):
        preds[i] = prev
        prev = true_val          # "true" lag-1: feed actual back at each step

    elapsed = time.time() - t0
    return evaluate(raw_test, preds, "Naive_lag1", train_time_s=elapsed)


# ─── CV experiment runner ─────────────────────────────────────────────────────

def run_model_on_subset(X_train, y_train, X_test, y_test,
                         raw_train, raw_test, raw_split, raw,
                         model_key, feature_subset):
    """Fit one model on one feature subset and return metrics dict."""
    X_tr = X_train[feature_subset]
    X_te = X_test[feature_subset]

    try:
        # ── Original tree / sequence models ──────────────────────────────────
        if   model_key == "xgb":   return _run_xgb(X_tr, y_train, X_te, y_test)
        elif model_key == "lgb":   return _run_lgb(X_tr, y_train, X_te, y_test)
        elif model_key == "rf":    return _run_rf(X_tr, y_train, X_te, y_test)
        elif model_key == "lstm":  return _run_lstm(raw_train, raw_test,
                                                     raw_split, raw)
        elif model_key == "arima": return _run_arima(raw_train, raw_test)
        elif model_key == "rnn":   return _run_rnn(raw_train, raw_test,
                                                    raw_split, raw)
        elif model_key == "gru":   return _run_gru(raw_train, raw_test,
                                                    raw_split, raw)
        elif model_key == "cnn":   return _run_cnn(raw_train, raw_test,
                                                    raw_split, raw)

        # ── New Phase-2 models ────────────────────────────────────────────────
        elif model_key == "gam":   return _run_gam(X_tr, y_train, X_te, y_test)
        elif model_key == "ridge": return _run_ridge(X_tr, y_train, X_te, y_test)
        elif model_key == "lasso": return _run_lasso(X_tr, y_train, X_te, y_test)
        elif model_key == "enet":  return _run_enet(X_tr, y_train, X_te, y_test)
        elif model_key == "svr":   return _run_svr(X_tr, y_train, X_te, y_test)
        elif model_key == "mlp":   return _run_mlp(X_tr, y_train, X_te, y_test)
        elif model_key == "ets":   return _run_ets(raw_train, raw_test)
        elif model_key == "naive": return _run_naive(raw_train, raw_test)

        # ── Phase-3 models ────────────────────────────────────────────────
        elif model_key == "gbm":      return _run_gbm(X_tr, y_train, X_te, y_test)
        elif model_key == "catboost": return _run_catboost(X_tr, y_train, X_te, y_test)
        elif model_key == "cart":     return _run_cart(X_tr, y_train, X_te, y_test)

        # ── Phase-4: Advanced Deep Learning ──────────────────────────────────
        elif model_key == "informer":     return _run_informer(raw_train, raw_test,
                                                               raw_split, raw)
        elif model_key == "transformer":  return _run_transformer(raw_train, raw_test,
                                                                   raw_split, raw)
        elif model_key == "nbeats":       return _run_nbeats(raw_train, raw_test,
                                                             raw_split, raw)

        else:
            raise ValueError(f"Unknown model key: '{model_key}'")

    except Exception as e:
        print(f"      WARNING: {model_key} on {len(feature_subset)} features "
              f"failed — {e}")
        return None


def run_cv_for_subset(fe_df, raw, model_key, feature_subset,
                       subset_name, n_folds=10, horizon=2920):
    """
    Run n-fold CV for one model × one feature subset.

    For univariate models (arima / ets / naive / lstm): the feature subset
    is irrelevant — results are identical across all subsets. Only the
    UNIVARIATE_ANCHOR_SUBSET ('all') is actually computed; subsequent
    subsets return an empty list and the caller copies results.
    """
    target    = "demand"
    feat_cols = [c for c in feature_subset if c in fe_df.columns]
    if not feat_cols:
        return []

    # ── Univariate guard: skip redundant subsets ──────────────────────────
    if (model_key in UNIVARIATE_MODELS
            and subset_name != UNIVARIATE_ANCHOR_SUBSET):
        return []   # Caller will copy results from anchor subset

    N          = len(fe_df)
    raw_offset = len(raw) - N
    folds      = make_custom_folds(N, n_folds=n_folds, horizon=horizon)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train = fe_df.iloc[train_idx][feat_cols]
        y_train = fe_df.iloc[train_idx][target]
        X_test  = fe_df.iloc[test_idx][feat_cols]
        y_test  = fe_df.iloc[test_idx][target]

        raw_split = raw_offset + int(test_idx[0])
        raw_train = raw[raw_offset + int(train_idx[0]):
                        raw_offset + int(train_idx[-1]) + 1]
        raw_test  = raw[raw_offset + int(test_idx[0]):
                        raw_offset + int(test_idx[-1]) + 1]

        metrics = run_model_on_subset(
            X_train, y_train, X_test, y_test,
            raw_train, raw_test, raw_split, raw,
            model_key, feat_cols
        )

        if metrics:
            metrics["fold"]        = fold_idx + 1
            metrics["subset_name"] = subset_name
            metrics["n_features"]  = len(feat_cols)
            metrics["train_rows"]  = len(train_idx)
            metrics["test_rows"]   = len(test_idx)
            fold_results.append(metrics)

    return fold_results


# ─── Main experiment loop ─────────────────────────────────────────────────────

def run_feature_selection_study(df, selected_models, n_folds=10,
                                  horizon=2920,
                                  skip_fs=False, output_dir="results/fs",
                                  zabin_features=None):
    """
    Full feature selection study — extended to 13 models.

    Parameters
    ----------
    df               : clean DataFrame with demand and temperature
    selected_models  : list of model keys
    n_folds          : number of CV folds (default 10)
    horizon          : test window size in rows (default 2920 = 4 months)
    skip_fs          : if True, load saved fs_summary.csv
    output_dir       : where to save results
    zabin_features   : optional reference feature list
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    raw = df["demand"].values

    # ── Step 1: Build full feature matrix ────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1 — Feature engineering (all 7 families)")
    print("="*60)
    fe_df = build_full_feature_matrix(df, target="demand", verbose=True)

    # ── Step 2: Feature selection ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2 — Ensemble filter feature selection")
    print("="*60)

    fs_summary_path = f"{output_dir}/fs_summary.csv"

    if skip_fs and os.path.exists(fs_summary_path):
        print(f"[FS] Loading saved results from {fs_summary_path}")
        summary = pd.read_csv(fs_summary_path, index_col=0)
        fs_results = {
            "summary":      summary,
            "average_rank": summary["average_rank"],
            "borda":        summary["borda_score"],
            "majority":     summary["majority_votes"],
        }
    else:
        split_idx = int(len(fe_df) * 0.8)
        feat_cols = [c for c in fe_df.columns if c != "demand"]
        X_sel     = fe_df.iloc[:split_idx][feat_cols]
        y_sel     = fe_df.iloc[:split_idx]["demand"]

        fs_results = run_filter_selection(X_sel, y_sel,
                                           subsample=10000, verbose=True)
        save_selection_results(fs_results, output_dir=output_dir)

    # ── Step 3: Define feature subsets ────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3 — Defining feature subsets")
    print("="*60)

    ref_subsets = {}
    if zabin_features:
        ref_subsets["zabin_22"] = zabin_features

    subsets = define_feature_subsets(fs_results,
                                      reference_subsets=ref_subsets)






    # ── Step 4: Run all experiments ───────────────────────────────────────────
    folds_preview = make_custom_folds(len(fe_df), n_folds=n_folds,
                                       horizon=horizon)
    # Effective fits: univariate models run once per fold (not per subset)
    n_univariate = sum(1 for m in selected_models if m in UNIVARIATE_MODELS)
    n_multivar   = len(selected_models) - n_univariate
    eff_fits = (n_multivar * len(subsets) + n_univariate) * len(folds_preview)

    print(f"\n" + "="*60)
    print(f"  STEP 4 — Running experiments")
    print(f"  {len(selected_models)} models × {len(subsets)} subsets × "
          f"{len(folds_preview)} folds")
    print(f"  Effective fits: {eff_fits}  "
          f"(univariate models run once per fold, not per subset)")
    print(f"\n  CV fold structure:")
    print(f"  {'Fold':<6} {'Train rows':>12} {'Train (months)':>15} {'Test rows':>12}")
    for i, (tr, te) in enumerate(folds_preview):
        print(f"  Fold {i+1:<2}  {len(tr):>12,}  {len(tr)/730:>13.1f}   {len(te):>12,}")
    print("="*60)

    # Print model registry
    print(f"\n  Model registry ({len(selected_models)} models):")
    unikeys = [m for m in selected_models if m in UNIVARIATE_MODELS]
    mulkeys = [m for m in selected_models if m not in UNIVARIATE_MODELS]
    if mulkeys:
        print(f"    Feature-sensitive  : {', '.join(mulkeys)}")
    if unikeys:
        print(f"    Univariate (series): {', '.join(unikeys)}")

    all_results     = []
    exp_num         = 0
    t_start         = time.time()
    # Cache results for univariate models — copy across subsets
    univariate_cache = {}   # model_key → list of fold-result dicts

    for model_key in selected_models:
        is_univariate = model_key in UNIVARIATE_MODELS

        for subset_name, subset_feats in subsets.items():
            exp_num += 1
            n_feat = len(subset_feats)
            print(f"\n  [{exp_num}/{len(selected_models)*len(subsets)}] "
                  f"{model_key.upper()} | {subset_name} ({n_feat} features)"
                  + (" [univariate — subset-independent]" if is_univariate else ""))

            # ── Univariate: use cache for all subsets except anchor ─────────
            if is_univariate and subset_name != UNIVARIATE_ANCHOR_SUBSET:
                if model_key in univariate_cache:
                    # Copy anchor results, re-stamp subset_name & n_features
                    copied = []
                    for r in univariate_cache[model_key]:
                        rc = dict(r)
                        rc["subset_name"] = subset_name
                        rc["n_features"]  = n_feat
                        copied.append(rc)
                    for r in copied:
                        r["model_key"] = model_key
                        all_results.append(r)
                    mapes = [r["MAPE"] for r in copied]
                    print(f"      MAPE: {np.mean(mapes):.3f}% ± {np.std(mapes):.3f}% "
                          f"[copied from '{UNIVARIATE_ANCHOR_SUBSET}']")
                continue

            # ── Regular run ────────────────────────────────────────────────
            fold_results = run_cv_for_subset(
                fe_df, raw, model_key, subset_feats,
                subset_name, n_folds=n_folds, horizon=horizon
            )

            for r in fold_results:
                r["model_key"] = model_key
                all_results.append(r)

            # Cache anchor results for univariate models
            if is_univariate and subset_name == UNIVARIATE_ANCHOR_SUBSET:
                univariate_cache[model_key] = fold_results

            if fold_results:
                mapes = [r["MAPE"] for r in fold_results]
                print(f"      MAPE: {np.mean(mapes):.3f}% ± {np.std(mapes):.3f}%")

            # ── Checkpoint: save after every subset ────────────────────────
            if all_results:
                ckpt_df   = pd.DataFrame(all_results)
                ckpt_path = f"{output_dir}/fs_experiments_checkpoint.csv"
                ckpt_df.to_csv(ckpt_path, index=False)

    elapsed = time.time() - t_start
    print(f"\n[Study] All experiments complete in {elapsed/60:.1f} minutes.")

    # ── Step 5: Aggregate and save results ────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 5 — Aggregating results")
    print("="*60)

    if not all_results:
        print("No results to aggregate.")
        return None, None

    results_df = pd.DataFrame(all_results)

    # ── Append-safe save: never overwrite models not in this run ─────────────
    existing_path = f"{output_dir}/fs_experiments.csv"
    if os.path.exists(existing_path):
        existing_df = pd.read_csv(existing_path)
        existing_df = existing_df[
            ~existing_df["model_key"].isin(selected_models)
        ]
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
        print(f"[Study] Merged with existing results "
              f"({len(existing_df)} kept + {len(pd.DataFrame(all_results))} new rows)")

    results_df.to_csv(existing_path, index=False)
    print(f"[Study] All results saved → {existing_path} ({len(results_df)} total rows)")

    # ── CV summary: mean ± std per model × subset ─────────────────────────────
    metrics_cols = ["MAE", "RMSE", "MAPE", "R2"]
    summary_rows = []

    for model_key in results_df["model_key"].unique():
        for subset_name in subsets.keys():
            mask = ((results_df["model_key"] == model_key) &
                    (results_df["subset_name"] == subset_name))
            sub  = results_df[mask]
            if len(sub) == 0:
                continue
            row = {"model": model_key, "subset": subset_name,
                   "n_features": sub["n_features"].iloc[0],
                   "n_folds": len(sub)}
            for m in metrics_cols:
                if m in sub.columns:
                    row[f"{m}_mean"] = round(sub[m].mean(), 4)
                    row[f"{m}_std"]  = round(sub[m].std(),  4)
            summary_rows.append(row)

    cv_summary = pd.DataFrame(summary_rows)
    cv_summary = cv_summary.sort_values(["model", "MAPE_mean"])
    cv_summary.to_csv(f"{output_dir}/fs_cv_summary.csv", index=False)
    print(f"[Study] CV summary saved  → {output_dir}/fs_cv_summary.csv")

    _print_fs_leaderboard(cv_summary)

    # ── Step 6: Generate plots ────────────────────────────────────────────────
    _plot_performance_curve(cv_summary, output_dir)
    _plot_heatmap(cv_summary, output_dir)
    _plot_feature_importance(fs_results, output_dir)

    return results_df, cv_summary


# ─── Reporting helpers ────────────────────────────────────────────────────────

def _print_fs_leaderboard(cv_summary: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("  FEATURE SELECTION STUDY — LEADERBOARD")
    print(f"{'='*70}")
    best = cv_summary.loc[cv_summary.groupby("model")["MAPE_mean"].idxmin()]
    best = best.sort_values("MAPE_mean")
    for rank_i, (_, row) in enumerate(best.iterrows(), 1):
        print(f"  #{rank_i:<3} {row['model'].upper():<14} "
              f"best subset: {row['subset']:<14} "
              f"({int(row['n_features'])} features) | "
              f"MAPE: {row['MAPE_mean']:.3f}% ± {row['MAPE_std']:.3f}%")
    print(f"{'='*70}")


# Colour map for all 13 models
_MODEL_COLORS = {
    "xgb":         "#f5a623",
    "lgb":         "#fb923c",
    "rf":          "#2dd4bf",
    "lstm":        "#4a9eff",
    "arima":       "#ff6b6b",
    "gam":         "#a78bfa",
    "ridge":       "#34d399",
    "lasso":       "#f472b6",
    "enet":        "#fbbf24",
    "svr":         "#60a5fa",
    "mlp":         "#f87171",
    "ets":         "#94a3b8",
    "naive":       "#6b7280",
    # Phase 4
    "informer":    "#e879f9",   # fuchsia
    "transformer": "#38bdf8",   # sky blue
    "nbeats":      "#4ade80",   # green
}


def _plot_performance_curve(cv_summary: pd.DataFrame,
                              output_dir: str) -> None:
    """Plot MAPE vs number of features per model."""
    try:
        import matplotlib
        matplotlib.rcParams.update({
            "figure.facecolor": "#0d0f14", "axes.facecolor": "#151820",
            "axes.edgecolor": "#2a3050", "text.color": "#e2e4f0",
            "axes.labelcolor": "#8890b0", "xtick.color": "#555e80",
            "ytick.color": "#555e80", "grid.color": "#1c2030",
            "font.family": "monospace",
        })
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(13, 7))
        for model_key in cv_summary["model"].unique():
            sub = (cv_summary[cv_summary["model"] == model_key]
                   .sort_values("n_features"))
            col = _MODEL_COLORS.get(model_key, "#888888")
            ax.errorbar(sub["n_features"], sub["MAPE_mean"],
                        yerr=sub["MAPE_std"],
                        label=model_key.upper(),
                        marker="o", markersize=5, linewidth=1.5,
                        color=col, capsize=3, capthick=1)

        ax.set_xlabel("Number of features")
        ax.set_ylabel("MAPE (%)")
        ax.set_title("Performance vs feature count — 10-fold CV (13 models)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/fs_performance_curve.png",
                    dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Study] Performance curve → {output_dir}/fs_performance_curve.png")
    except Exception as e:
        print(f"[Study] Performance curve plot failed: {e}")


def _plot_heatmap(cv_summary: pd.DataFrame, output_dir: str) -> None:
    """Plot MAPE heatmap — models × feature subsets."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        pivot = cv_summary.pivot_table(
            index="model", columns="subset",
            values="MAPE_mean", aggfunc="mean"
        )

        n_models  = len(pivot.index)
        fig, ax   = plt.subplots(figsize=(14, max(5, n_models * 0.55)))
        im        = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([m.upper() for m in pivot.index], fontsize=9)
        plt.colorbar(im, ax=ax, label="MAPE (%)")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7.5, color="white")

        ax.set_title("MAPE heatmap — model × feature subset (13 models)")
        fig.tight_layout()
        fig.savefig(f"{output_dir}/fs_heatmap.png", dpi=150,
                    bbox_inches="tight", facecolor="#0d0f14")
        plt.close(fig)
        print(f"[Study] Heatmap saved     → {output_dir}/fs_heatmap.png")
    except Exception as e:
        print(f"[Study] Heatmap plot failed: {e}")


def _plot_feature_importance(fs_results: dict, output_dir: str) -> None:
    """Plot top-20 features ranked by average rank."""
    try:
        import matplotlib.pyplot as plt

        top20  = fs_results["average_rank"].sort_values().head(20)
        colors = ["#4a9eff"] * 20

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(range(len(top20)), top20.values[::-1],
                color=colors, height=0.65)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20.index[::-1], fontsize=9)
        ax.set_xlabel("Average rank (lower = more informative)")
        ax.set_title("Top 20 features — ensemble average rank")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/fs_importance_plot.png", dpi=150,
                    bbox_inches="tight", facecolor="#0d0f14")
        plt.close(fig)
        print(f"[Study] Importance plot   → {output_dir}/fs_importance_plot.png")
    except Exception as e:
        print(f"[Study] Importance plot failed: {e}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extended feature selection study — 13 models"
    )
    parser.add_argument("--csv",        type=str, required=True,
                        help="Path to clean CSV (e.g. GEFCom2014_clean.csv)")
    parser.add_argument("--models",     nargs="+",
                        default=["xgb", "lgb", "rf"],
                        help=("Models: xgb lgb rf lstm arima "
                              "gam ridge lasso enet svr mlp ets naive"))
    parser.add_argument("--n-folds",    type=int, default=10,
                        help="Number of CV folds (default 10)")
    parser.add_argument("--horizon",    type=int, default=2920,
                        help="Test window size in rows (default 2920 = 4 months)")
    parser.add_argument("--skip-fs",    action="store_true",
                        help="Skip filter selection, load saved fs_summary.csv")
    parser.add_argument("--output-dir", type=str, default="results/fs",
                        help="Output directory (default results/fs)")
    args = parser.parse_args()

    # Validate model keys
    valid_keys = {"xgb","lgb","rf","lstm","arima",
                  "gam","ridge","lasso","enet","svr","mlp","ets","naive",
                  "rnn","gru","cnn",
                  "gbm","catboost","cart",
                  "informer","transformer","nbeats"}   # Phase 4
    unknown = set(args.models) - valid_keys
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. "
                         f"Valid keys: {sorted(valid_keys)}")

    df = load_csv(args.csv)

    # Zabin et al. (2024) reference feature set
    zabin_features = [
        "hour","dayofweek","quarter","month","year","dayofyear",
        "dayofmonth","weekofyear",
        "lag_6h","lag_24h",
        "roll_mean_6h","roll_mean_12h","roll_mean_24h",
        "roll_std_6h","roll_std_12h","roll_std_24h",
        "roll_max_6h","roll_max_12h","roll_max_24h",
        "roll_min_6h","roll_min_12h","roll_min_24h",
    ]

    run_feature_selection_study(
        df,
        selected_models=args.models,
        n_folds=args.n_folds,
        horizon=args.horizon,
        skip_fs=args.skip_fs,
        output_dir=args.output_dir,
        zabin_features=zabin_features,
    )

    print("\n[main_fs] Done ✓")


if __name__ == "__main__":
    main()
