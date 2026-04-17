"""
models/cart_model.py
--------------------
CART (Classification and Regression Trees) forecaster — sklearn DecisionTreeRegressor.

Algorithm overview
------------------
CART builds a binary decision tree by recursively partitioning the feature
space using the following greedy procedure:

    At each node, find the split (feature j, threshold t) that minimises
    the weighted sum of squared errors in the two child nodes:

        min_{j,t}  [ Σ_{x∈R_left}(yᵢ − ȳ_left)²
                   + Σ_{x∈R_right}(yᵢ − ȳ_right)² ]

    Leaf prediction: ŷ = mean(y) of all training samples in the leaf

    Splitting stops when:
        - max_depth is reached, OR
        - min_samples_split samples remain in a node, OR
        - min_samples_leaf would be violated

Key properties vs ensemble methods
------------------------------------
- Single tree → interpretable, fast to train and predict
- High variance: a single tree is sensitive to training data perturbations
  (this is why bagging → Random Forest and boosting → GBM were invented)
- No ensemble averaging → typically lower accuracy than RF/XGB/LGB
- Useful as a baseline and for understanding feature importance patterns
- CART is the base learner inside Random Forest, XGBoost, LightGBM, and GBM

Hyperparameters (tuned for GEFCom2014 hourly demand)
------------------------------------------------------
max_depth=10         : limits tree complexity; deeper = more variance
min_samples_leaf=20  : at least 20 samples per leaf — prevents overfitting
                       on the large training folds (30k–60k rows)
min_samples_split=40 : minimum samples required to split a node
max_features="sqrt"  : random feature subsampling at each split (same as RF)
                       reduces variance slightly vs using all features

Note on max_features
--------------------
Setting max_features="sqrt" makes CART slightly closer to a single tree
from a Random Forest. Set to None (all features) for a pure deterministic
CART. Both are valid experimental choices; "sqrt" generally performs better
on high-dimensional feature matrices.
"""

import numpy as np
import time


class CARTForecaster:
    """
    CART (Decision Tree) forecaster — feature matrix input.

    Parameters
    ----------
    max_depth         : int or None — max tree depth (default 10)
    min_samples_leaf  : int         — min samples per leaf (default 20)
    min_samples_split : int         — min samples to split a node (default 40)
    max_features      : str or None — feature subsampling per split
                                      ("sqrt", "log2", None = all)
    """

    def __init__(self, max_depth=10, min_samples_leaf=20,
                 min_samples_split=40, max_features="sqrt"):
        self.max_depth         = max_depth
        self.min_samples_leaf  = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.model_            = None
        self.train_time_       = 0.0

    def fit(self, X_train, y_train):
        from sklearn.tree import DecisionTreeRegressor

        self.model_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=42,
        )
        t0 = time.time()
        self.model_.fit(X_train, y_train)
        self.train_time_ = time.time() - t0

        n_leaves = self.model_.get_n_leaves()
        depth    = self.model_.get_depth()
        print(f"        [CART] Depth: {depth}  Leaves: {n_leaves}  "
              f"Train time: {self.train_time_:.1f}s")

    def predict(self, X_test):
        return np.maximum(self.model_.predict(X_test), 0)
