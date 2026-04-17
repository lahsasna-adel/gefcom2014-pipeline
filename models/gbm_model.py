"""
models/gbm_model.py
-------------------
Histogram-based Gradient Boosting forecaster (sklearn HistGradientBoostingRegressor).

Algorithm overview
------------------
Gradient Boosting builds an additive ensemble of weak learners (decision trees)
by fitting each new tree to the *residuals* (negative gradient of the loss)
of the current ensemble:

    F_0(x) = argmin_γ Σ L(yᵢ, γ)           ← initialise with constant
    For m = 1 to M:
        rᵢₘ = −[∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)]   ← pseudo-residuals
        hₘ  = fit tree to {(xᵢ, rᵢₘ)}       ← weak learner
        F_m(x) = F_{m-1}(x) + η · hₘ(x)    ← update with learning rate η

HistGradientBoosting vs classic GradientBoosting
-------------------------------------------------
- Bins continuous features into 256 integer bins before splitting
- Split finding is O(n_bins) instead of O(n_samples) → much faster on large data
- Supports native NaN handling (no imputation needed)
- Equivalent to LightGBM's histogram algorithm (Ke et al., 2017)
- Typically 10–100× faster than sklearn GradientBoostingRegressor on >10k rows

Hyperparameters (selected for GEFCom2014 hourly demand data)
-------------------------------------------------------------
max_iter=500         : number of boosting rounds (trees)
learning_rate=0.05   : shrinkage — small rate + more trees = better generalisation
max_depth=6          : controls individual tree complexity
min_samples_leaf=20  : minimum samples per leaf — prevents overfitting on large folds
l2_regularization=0.1: L2 penalty on leaf values
early_stopping=True  : stops when validation score stops improving
validation_fraction=0.1
n_iter_no_change=20  : patience for early stopping
"""

import numpy as np
import time


class GBMForecaster:
    """
    Histogram Gradient Boosting forecaster — feature matrix input.

    Parameters
    ----------
    max_iter         : int   — max boosting rounds (default 500)
    learning_rate    : float — shrinkage rate (default 0.05)
    max_depth        : int   — max tree depth (default 6)
    min_samples_leaf : int   — min samples per leaf (default 20)
    l2_reg           : float — L2 regularisation (default 0.1)
    """

    def __init__(self, max_iter=500, learning_rate=0.05,
                 max_depth=6, min_samples_leaf=20, l2_reg=0.1):
        self.max_iter         = max_iter
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_reg           = l2_reg
        self.model_           = None
        self.train_time_      = 0.0

    def fit(self, X_train, y_train):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.model_ = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_reg,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=0,
        )
        t0 = time.time()
        self.model_.fit(X_train, y_train)
        self.train_time_ = time.time() - t0
        n_iter = self.model_.n_iter_
        print(f"        [GBM] Rounds: {n_iter}  "
              f"Train time: {self.train_time_:.1f}s")

    def predict(self, X_test):
        return np.maximum(self.model_.predict(X_test), 0)
