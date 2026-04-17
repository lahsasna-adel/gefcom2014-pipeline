"""
main_fs_patch_phase3.py
-----------------------
Four edits required in main_fs.py to integrate GBM, CatBoost, and CART.
Apply them in order using Find & Replace in your editor.

All three models are MULTIVARIATE (use the engineered feature matrix),
so they are NOT added to UNIVARIATE_MODELS — no change needed there.
"""

# ══════════════════════════════════════════════════════════════════════════════
# EDIT 1 — Add runner functions
# Location: after _run_mlp function (~line 408), before _run_ets
# ══════════════════════════════════════════════════════════════════════════════

EDIT_1_FIND = '''def _run_ets(raw_train, raw_test):'''

EDIT_1_REPLACE = '''def _run_gbm(X_tr, y_train, X_te, y_test):
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


def _run_ets(raw_train, raw_test):'''


# ══════════════════════════════════════════════════════════════════════════════
# EDIT 2 — Add GBM, CatBoost, CART to the model dispatcher
# Location: inside run_model_on_subset, after the "enet" branch (~line 510)
# ══════════════════════════════════════════════════════════════════════════════

EDIT_2_FIND = '''        elif model_key == "svr":    return _run_svr(X_tr, y_train, X_te, y_test)'''

EDIT_2_REPLACE = '''        elif model_key == "svr":      return _run_svr(X_tr, y_train, X_te, y_test)
        elif model_key == "gbm":      return _run_gbm(X_tr, y_train, X_te, y_test)
        elif model_key == "catboost": return _run_catboost(X_tr, y_train, X_te, y_test)
        elif model_key == "cart":     return _run_cart(X_tr, y_train, X_te, y_test)'''


# ══════════════════════════════════════════════════════════════════════════════
# EDIT 3 — Register gbm/catboost/cart as valid CLI keys
# Location: inside main(), valid_keys set (~line 954)
# ══════════════════════════════════════════════════════════════════════════════

EDIT_3_FIND = '''    valid_keys = {"xgb","lgb","rf","lstm","arima",
                  "gam","ridge","lasso","enet","svr","mlp","ets","naive",
                  "rnn","gru","cnn"}'''

EDIT_3_REPLACE = '''    valid_keys = {"xgb","lgb","rf","lstm","arima",
                  "gam","ridge","lasso","enet","svr","mlp","ets","naive",
                  "rnn","gru","cnn",
                  "gbm","catboost","cart"}'''


# ══════════════════════════════════════════════════════════════════════════════
# EDIT 4 — Update the module docstring (optional but keeps docs accurate)
# Location: top of file, inside the docstring, after the "cnn" line
# ══════════════════════════════════════════════════════════════════════════════

EDIT_4_FIND = '''  cnn      — 1D CNN with dilated causal convolutions (lookback=48h)'''

EDIT_4_REPLACE = '''  cnn      — 1D CNN with dilated causal convolutions (lookback=48h)

Phase 3 models:
  gbm      — Histogram Gradient Boosting (sklearn HistGradientBoostingRegressor)
  catboost — CatBoost with ordered boosting and symmetric trees
  cart     — Single CART Decision Tree (interpretable baseline)'''


# ══════════════════════════════════════════════════════════════════════════════
# INSTALLATION CHECK
# ══════════════════════════════════════════════════════════════════════════════
# CatBoost requires a separate install:
#
#   pip install catboost
#
# GBM and CART use sklearn which is already installed.
# Verify CatBoost is available before running:
#
#   python -c "import catboost; print(catboost.__version__)"
#
# ══════════════════════════════════════════════════════════════════════════════
# RUN COMMANDS
# ══════════════════════════════════════════════════════════════════════════════
#
# Run Phase 3 models only (fastest, uses saved feature selection):
#   python main_fs.py --csv GEFCom2014_clean.csv --skip-fs \
#                     --models gbm catboost cart
#
# Run all Phase 3 models + selected Phase 1/2 for comparison:
#   python main_fs.py --csv GEFCom2014_clean.csv --skip-fs \
#                     --models xgb lgb rf gbm catboost cart
#
# ══════════════════════════════════════════════════════════════════════════════
# EXPECTED RESULTS (indicative, based on GEFCom2014 literature)
# ══════════════════════════════════════════════════════════════════════════════
#
#   GBM      : MAPE ~2-4%   — competitive with XGBoost/LightGBM
#   CatBoost : MAPE ~2-4%   — often slightly better than XGBoost on tabular data
#   CART     : MAPE ~6-10%  — higher variance, useful as ensemble baseline
#
# ══════════════════════════════════════════════════════════════════════════════
