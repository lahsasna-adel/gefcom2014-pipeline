"""
models/catboost_model.py
------------------------
CatBoost Gradient Boosting forecaster (Yandex CatBoost).

Algorithm overview
------------------
CatBoost is a gradient boosting algorithm with two key innovations over
XGBoost and LightGBM:

1. Ordered boosting (oblivious trees):
   Uses a permutation-based approach to compute leaf values, avoiding
   target leakage during training — each tree is built on a random
   permutation of the training data, and the gradient for sample i is
   computed using only the samples processed before i in that permutation.

2. Symmetric (oblivious) trees:
   All nodes at the same depth use the same split condition:
       if feature_j < threshold → go left (for ALL nodes at depth d)
   This produces balanced trees that are fast to evaluate and less prone
   to overfitting than asymmetric trees used by XGBoost/LightGBM.

CatBoost update rule (same as standard gradient boosting):
    F_m(x) = F_{m-1}(x) + η · hₘ(x)
where hₘ is an oblivious tree fitted to pseudo-residuals, and η is the
learning rate (shrinkage).

Advantages for electricity forecasting
---------------------------------------
- Ordered boosting reduces overfitting on correlated lag/rolling features
- Symmetric trees are fast at prediction time (important for walk-forward CV)
- Built-in handling of categorical features (unused here but available)
- Competitive with XGBoost/LightGBM on tabular data (Prokhorenkova et al., 2018)

Hyperparameters
---------------
iterations=500       : number of trees
learning_rate=0.05   : shrinkage
depth=6              : tree depth (symmetric trees — equivalent to ~depth 8-9
                       in asymmetric trees due to balanced structure)
l2_leaf_reg=3.0      : L2 regularisation on leaf values (CatBoost default)
early_stopping_rounds=50 : patience

Reference
---------
Prokhorenkova et al. (2018) — CatBoost: unbiased boosting with categorical
features. NeurIPS 2018.
"""

import numpy as np
import time


class CatBoostForecaster:
    """
    CatBoost gradient boosting forecaster — feature matrix input.

    Parameters
    ----------
    iterations    : int   — number of boosting trees (default 500)
    learning_rate : float — shrinkage (default 0.05)
    depth         : int   — symmetric tree depth (default 6)
    l2_leaf_reg   : float — L2 regularisation (default 3.0)
    early_stopping_rounds : int — patience (default 50)
    """

    def __init__(self, iterations=500, learning_rate=0.05,
                 depth=6, l2_leaf_reg=3.0, early_stopping_rounds=50):
        self.iterations           = iterations
        self.learning_rate        = learning_rate
        self.depth                = depth
        self.l2_leaf_reg          = l2_leaf_reg
        self.early_stopping_rounds= early_stopping_rounds
        self.model_               = None
        self.train_time_          = 0.0

    def fit(self, X_train, y_train):
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError(
                "CatBoost is not installed. Run: pip install catboost"
            )

        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            early_stopping_rounds=self.early_stopping_rounds,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=0,              # suppress per-iteration output
        )
        t0 = time.time()
        # Use last 10% of training data as internal validation for early stopping
        n_val = max(1, int(len(X_train) * 0.1))
        X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
        y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

        self.model_.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )
        self.train_time_ = time.time() - t0
        n_iter = self.model_.tree_count_
        print(f"        [CatBoost] Trees: {n_iter}  "
              f"Train time: {self.train_time_:.1f}s")

    def predict(self, X_test):
        return np.maximum(self.model_.predict(X_test), 0)
