"""
models/tree_models.py
---------------------
XGBoost, LightGBM, and Random Forest forecasters for electricity demand.

All three use the same feature-engineering pipeline from utils/data_loader.py
(lag features, rolling statistics, Fourier seasonality, calendar variables).

Hyperparameter tuning via Optuna is built in.

Dependencies:
    pip install xgboost lightgbm scikit-learn optuna
"""

import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")


# ─── Base class ───────────────────────────────────────────────────────────────

class _TreeForecasterBase:
    """Shared logic for tree-based electricity forecasters."""

    def __init__(self, name: str):
        self.name        = name
        self.model_      = None
        self.train_time_ = None
        self.feature_importances_: Optional[pd.Series] = None

    def _make_model(self, params: dict):
        raise NotImplementedError

    def fit(self, X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> "_TreeForecasterBase":

        print(f"[{self.name}] Fitting …", end=" ", flush=True)
        t0 = time.time()
        self.model_ = self._make_model(self.params)

        fit_kwargs = {}
        if X_val is not None:
            # XGBoost / LightGBM support early stopping via eval_set
            try:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                if self.name == "LightGBM":
                    import lightgbm as lgb
                    fit_kwargs["callbacks"] = [lgb.log_evaluation(period=-1)]
                else:
                    fit_kwargs["verbose"]  = False
            except Exception:
                pass

        self.model_.fit(X_train, y_train, **fit_kwargs)
        self.train_time_ = time.time() - t0

        # Extract feature importance
        if hasattr(self.model_, "feature_importances_"):
            self.feature_importances_ = pd.Series(
                self.model_.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)

        print(f"done in {self.train_time_:.1f}s")
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError(f"Call fit() before predict() on {self.name}.")
        return np.maximum(self.model_.predict(X_test), 0)

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series,
             n_trials: int = 100,
             cv_splits: int = 5) -> Dict:
        """
        Hyperparameter optimisation via Optuna + TimeSeriesSplit.
        Returns best params dict.
        """
        try:
            import optuna
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Install optuna:  pip install optuna")

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        def objective(trial):
            params = self._suggest_params(trial)
            model  = self._make_model(params)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1
            )
            return -scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"[{self.name}] Best RMSE: {study.best_value:.2f} MW | "
              f"Params: {study.best_params}")
        self.params = {**self.params, **study.best_params}
        return study.best_params

    def _suggest_params(self, trial) -> Dict:
        raise NotImplementedError

    def top_features(self, n: int = 15) -> pd.Series:
        """Return top-n most important features."""
        if self.feature_importances_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.feature_importances_.head(n)


# ─── XGBoost ─────────────────────────────────────────────────────────────────

class XGBoostForecaster(_TreeForecasterBase):
    """
    XGBoost regressor for electricity demand forecasting.

    Default hyperparameters are tuned for hourly electricity data.
    Call tune() to run Optuna search.
    """

    DEFAULT_PARAMS = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    def __init__(self, params: Optional[Dict] = None):
        super().__init__("XGBoost")
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    def _make_model(self, params):
        from xgboost import XGBRegressor
        return XGBRegressor(**params)

    def _suggest_params(self, trial) -> Dict:
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        }


# ─── LightGBM ────────────────────────────────────────────────────────────────

class LightGBMForecaster(_TreeForecasterBase):
    """
    LightGBM regressor for electricity demand forecasting.
    Typically 3–10× faster than XGBoost with similar accuracy.
    """

    DEFAULT_PARAMS = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="regression",
        metric="rmse",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    def __init__(self, params: Optional[Dict] = None):
        super().__init__("LightGBM")
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    def _make_model(self, params):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)

    def _suggest_params(self, trial) -> Dict:
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        }


# ─── Random Forest ───────────────────────────────────────────────────────────

class RandomForestForecaster(_TreeForecasterBase):
    """
    Random Forest regressor for electricity demand forecasting.

    More robust to outliers than boosting methods.
    Feature importance based on out-of-bag permutation.
    """

    DEFAULT_PARAMS = dict(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )

    def __init__(self, params: Optional[Dict] = None):
        super().__init__("RandomForest")
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    def _make_model(self, params):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)

    def _suggest_params(self, trial) -> Dict:
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 800),
            "max_depth":       trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":    trial.suggest_categorical("max_features",
                                                         ["sqrt", "log2", 0.5]),
        }

    def oob_r2(self) -> float:
        """Return out-of-bag R² score (only available if oob_score=True)."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        return float(self.model_.oob_score_)
