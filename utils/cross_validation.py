"""
utils/cross_validation.py
--------------------------
Time-series cross-validation (walk-forward validation) for all models.

Strategy
--------
Unlike random k-fold CV, time-series CV always trains on the PAST
and tests on the FUTURE — no data leakage.

Illustration (5 folds, expanding window):

  Fold 1: [====TRAIN====] [TEST]
  Fold 2: [======TRAIN======] [TEST]
  Fold 3: [========TRAIN========] [TEST]
  Fold 4: [==========TRAIN==========] [TEST]
  Fold 5: [============TRAIN============] [TEST]

Each TEST window is the same size (test_size fraction of total data).
The TRAIN window expands with each fold (expanding window strategy).

Results reported as mean ± std across folds for each metric.
"""

import time
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

from utils.metrics import evaluate


# ─── Core CV engine ───────────────────────────────────────────────────────────

def run_cv(df: pd.DataFrame,
           selected: List[str],
           n_splits: int = 5,
           test_size: float = 0.2,
          # test_size_months: int = 4, # New parameter
           horizon: int = 24,
           tune: List[str] = None) -> Dict:
    """
    Run walk-forward cross-validation for all selected models.

    Parameters
    ----------
    df        : full DataFrame with DatetimeIndex and 'demand' column
    selected  : list of model keys e.g. ['xgb', 'lstm', 'prophet']
    n_splits  : number of CV folds
    test_size : fraction of data used per test fold
    horizon   : forecast horizon in steps (hours)
    tune      : models to tune with Optuna on first fold only

    Returns
    -------
    cv_results : {
        model_name: {
            "fold_metrics": [dict, dict, ...],   # one dict per fold
            "mean": dict,                         # mean across folds
            "std":  dict,                         # std across folds
        }
    }
    """
    tune = tune or []

    from utils.data_loader import engineer_features

    print(f"\n{'='*60}")
    print(f"  TIME-SERIES CROSS-VALIDATION  |  {n_splits} folds")
    print(f"{'='*60}")

    # ── Feature engineering (once, on full dataset) ──────────────────────────
    fe_df = engineer_features(df, lags=[1,2,3,24,48,168], rolling_windows=[6,24,168])
    raw   = df["demand"].values
    N     = len(fe_df)

    # ── Build fold indices using sklearn TimeSeriesSplit ─────────────────────
    fold_test_size = max(horizon, int(N * test_size))
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=fold_test_size)

    # Collect fold index pairs
    folds = []
    for train_idx, test_idx in tscv.split(fe_df):
        folds.append((train_idx, test_idx))

    print(f"  Total rows (after feature eng): {N:,}")
    print(f"  Test window per fold          : {fold_test_size:,} rows")
    print(f"  Models                        : {selected}\n")

    # ── Initialise results storage ───────────────────────────────────────────
    cv_results = {m: {"fold_metrics": []} for m in _model_display_names(selected)}

    # ── Loop over folds ───────────────────────────────────────────────────────
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n── Fold {fold_idx + 1}/{n_splits} "
              f"| Train: {len(train_idx):,} | Test: {len(test_idx):,} ──")

        # Slice feature-engineered data
        X_train = fe_df.iloc[train_idx][[c for c in fe_df.columns if c != "demand"]]
        y_train = fe_df.iloc[train_idx]["demand"]
        X_test  = fe_df.iloc[test_idx][[c for c in fe_df.columns if c != "demand"]]
        y_test  = fe_df.iloc[test_idx]["demand"]

        # Map fe_df indices back to raw array positions
        # fe_df starts at row 168 of raw (due to lagging), so offset accordingly
        raw_offset = len(raw) - N
        raw_train  = raw[raw_offset + train_idx[0]  : raw_offset + train_idx[-1] + 1]
        raw_test   = raw[raw_offset + test_idx[0]   : raw_offset + test_idx[-1] + 1]
        raw_split  = raw_offset + test_idx[0]

        # ── SARIMA ────────────────────────────────────────────────────────────
        if "arima" in selected:
            fold_metrics = _cv_sarima(raw_train, raw_test, fold_idx)
            cv_results["SARIMA"]["fold_metrics"].append(fold_metrics)

        # ── Prophet ───────────────────────────────────────────────────────────
        if "prophet" in selected:
            train_series = df["demand"].iloc[raw_offset + train_idx[0]:
                                              raw_offset + train_idx[-1] + 1]
            test_index   = df.index[raw_offset + test_idx[0]:
                                     raw_offset + test_idx[-1] + 1]
            extra = [c for c in ["temperature","holiday"] if c in df.columns]
            fold_metrics = _cv_prophet(train_series, raw_test, test_index, df, extra, fold_idx)
            cv_results["Prophet"]["fold_metrics"].append(fold_metrics)

        # ── XGBoost ───────────────────────────────────────────────────────────
        if "xgb" in selected:
            fold_metrics = _cv_tree("XGBoost", X_train, y_train, X_test, y_test,
                                    tune="xgb" in tune and fold_idx == 0)
            cv_results["XGBoost"]["fold_metrics"].append(fold_metrics)

        # ── LightGBM ──────────────────────────────────────────────────────────
        if "lgb" in selected:
            fold_metrics = _cv_tree("LightGBM", X_train, y_train, X_test, y_test,
                                    tune="lgb" in tune and fold_idx == 0)
            cv_results["LightGBM"]["fold_metrics"].append(fold_metrics)

        # ── Random Forest ─────────────────────────────────────────────────────
        if "rf" in selected:
            fold_metrics = _cv_tree("RandomForest", X_train, y_train, X_test, y_test,
                                    tune=False)
            cv_results["RandomForest"]["fold_metrics"].append(fold_metrics)

        # ── LSTM ──────────────────────────────────────────────────────────────
        if "lstm" in selected:
            fold_metrics = _cv_lstm(raw, raw_split, raw_test, fold_idx)
            cv_results["LSTM"]["fold_metrics"].append(fold_metrics)

    # ── Aggregate across folds ────────────────────────────────────────────────
    metrics_to_agg = ["MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE"]
    for model_name, res in cv_results.items():
        if not res["fold_metrics"]:
            continue
        agg_mean, agg_std = {}, {}
        agg_mean["model"] = model_name
        agg_std["model"]  = model_name
        for m in metrics_to_agg:
            vals = [f[m] for f in res["fold_metrics"] if m in f]
            if vals:
                agg_mean[m] = round(float(np.mean(vals)), 3)
                agg_std[m]  = round(float(np.std(vals)),  3)
        res["mean"] = agg_mean
        res["std"]  = agg_std

    return cv_results


# ─── Per-model CV helpers ─────────────────────────────────────────────────────

def _cv_sarima(raw_train, raw_test, fold_idx):
    from models.sarima_model import SARIMAForecaster
    try:
        m = SARIMAForecaster(seasonal_period=24,
                             manual_order=(1, 1, 2),
                             manual_seasonal_order=(1, 1, 1, 24))
        m.fit(raw_train)
        preds, _ = m.predict(len(raw_test))
        return evaluate(raw_test, preds, "SARIMA")
    except Exception as e:
        print(f"  [SARIMA] Fold {fold_idx+1} failed: {e}")
        return {}


def _cv_prophet(train_series, raw_test, test_index, df, extra_regs, fold_idx):
    from models.prophet_model import ProphetForecaster
    try:
        m = ProphetForecaster(country_holidays="DZ",
                              extra_regressors=extra_regs)
        m.fit(train_series,
              regressors_train=df[extra_regs].loc[train_series.index] if extra_regs else None)
        fc = m.predict(test_index,
                       regressors_future=df[extra_regs].loc[test_index] if extra_regs else None)
        preds = fc["yhat"].values
        return evaluate(raw_test, preds[:len(raw_test)], "Prophet")
    except Exception as e:
        print(f"  [Prophet] Fold {fold_idx+1} failed: {e}")
        return {}


def _cv_tree(name, X_train, y_train, X_test, y_test, tune=False):
    from models.tree_models import XGBoostForecaster, LightGBMForecaster, RandomForestForecaster
    cls = {"XGBoost": XGBoostForecaster,
           "LightGBM": LightGBMForecaster,
           "RandomForest": RandomForestForecaster}[name]
    try:
        m = cls()
        if tune:
            m.tune(X_train, y_train, n_trials=30)
        if name == "RandomForest":
            m.fit(X_train, y_train)
        else:
            m.fit(X_train, y_train, X_test, y_test)
        preds = m.predict(X_test)
        return evaluate(y_test.values, preds, name)
    except Exception as e:
        print(f"  [{name}] Fold failed: {e}")
        return {}


def _cv_lstm(raw, raw_split, raw_test, fold_idx):
    from models.lstm_model import LSTMForecaster
    try:
        m = LSTMForecaster(lookback=48, horizon=24, epochs=30, patience=5)
        m.fit(raw[:raw_split])
        preds = []
        for i in range(len(raw_test)):
            ctx = raw[max(0, raw_split + i - 48): raw_split + i]
            preds.append(m._predict_one(ctx))
        preds = np.array(preds)
        return evaluate(raw_test, preds, "LSTM")
    except Exception as e:
        print(f"  [LSTM] Fold {fold_idx+1} failed: {e}")
        return {}


# ─── Display helpers ──────────────────────────────────────────────────────────

def _model_display_names(selected):
    mapping = {"arima":"SARIMA","prophet":"Prophet","xgb":"XGBoost",
               "lgb":"LightGBM","rf":"RandomForest","lstm":"LSTM"}
    return [mapping[k] for k in selected if k in mapping]


def print_cv_leaderboard(cv_results: Dict) -> None:
    """Pretty-print the CV leaderboard with mean ± std for each metric."""
    print(f"\n{'='*72}")
    print("  CROSS-VALIDATION LEADERBOARD  (mean ± std across folds)")
    print(f"{'='*72}")
    print(f"  {'Model':<16} {'MAE':>10} {'RMSE':>10} {'MAPE%':>10} {'R²':>8}")
    print(f"  {'-'*56}")

    # Sort by mean MAPE
    rows = [(k, v) for k, v in cv_results.items() if "mean" in v]
    rows.sort(key=lambda x: x[1]["mean"].get("MAPE", 999))

    for model_name, res in rows:
        mn, sd = res["mean"], res["std"]
        mae_s  = f"{mn.get('MAE',0):.1f}±{sd.get('MAE',0):.1f}"
        rmse_s = f"{mn.get('RMSE',0):.1f}±{sd.get('RMSE',0):.1f}"
        mape_s = f"{mn.get('MAPE',0):.2f}±{sd.get('MAPE',0):.2f}"
        r2_s   = f"{mn.get('R2',0):.3f}±{sd.get('R2',0):.3f}"
        print(f"  {model_name:<16} {mae_s:>10} {rmse_s:>10} {mape_s:>10} {r2_s:>8}")
    print(f"{'='*72}\n")


def cv_results_to_dataframe(cv_results: Dict) -> pd.DataFrame:
    """
    Convert CV results to a tidy DataFrame with columns:
    model, fold, MAE, RMSE, MAPE, sMAPE, R2, MASE
    """
    rows = []
    for model_name, res in cv_results.items():
        for fold_idx, metrics in enumerate(res["fold_metrics"]):
            row = {"model": model_name, "fold": fold_idx + 1}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def cv_summary_dataframe(cv_results: Dict) -> pd.DataFrame:
    """
    Build a summary DataFrame with mean and std columns for each metric.
    One row per model, sorted by MAPE mean.
    """
    records = []
    for model_name, res in cv_results.items():
        if "mean" not in res:
            continue
        record = {"model": model_name}
        for metric in ["MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE"]:
            record[f"{metric}_mean"] = res["mean"].get(metric, np.nan)
            record[f"{metric}_std"]  = res["std"].get(metric, np.nan)
        records.append(record)
    df = pd.DataFrame(records).sort_values("MAPE_mean").reset_index(drop=True)
    df.index += 1
    return df
