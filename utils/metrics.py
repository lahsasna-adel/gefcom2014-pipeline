"""
utils/metrics.py
----------------
Evaluation metrics for time-series forecasting:
  MAE, RMSE, MAPE, SMAPE, R², MASE, Pinball loss (for quantile forecasts)
plus a unified evaluate() function that returns a tidy dict.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# ─── Individual metrics ───────────────────────────────────────────────────────

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error (MW)."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error (MW)."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray,
         epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray,
          epsilon: float = 1e-8) -> float:
    """Symmetric MAPE (%) — bounded [0, 200]."""
    denom = (np.abs(actual) + np.abs(predicted)) / 2 + epsilon
    return float(np.mean(np.abs(actual - predicted) / denom) * 100)


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def mase(actual: np.ndarray, predicted: np.ndarray,
         seasonal_period: int = 24) -> float:
    """
    Mean Absolute Scaled Error.
    Scales MAE by the in-sample MAE of a naive seasonal forecast.
    """
    naive_errors = np.abs(actual[seasonal_period:] - actual[:-seasonal_period])
    scale = np.mean(naive_errors) + 1e-8
    return float(np.mean(np.abs(actual - predicted)) / scale)


def pinball_loss(actual: np.ndarray,
                 quantile_pred: np.ndarray,
                 quantile: float = 0.5) -> float:
    """Pinball (quantile) loss for probabilistic forecasts."""
    errors = actual - quantile_pred
    return float(np.mean(np.where(errors >= 0,
                                  quantile * errors,
                                  (quantile - 1) * errors)))


# ─── Unified evaluation ───────────────────────────────────────────────────────

def evaluate(actual: np.ndarray,
             predicted: np.ndarray,
             model_name: str = "model",
             seasonal_period: int = 24,
             train_time_s: Optional[float] = None) -> Dict:
    """
    Compute all metrics and return a tidy dictionary.

    Parameters
    ----------
    actual          : ground-truth demand values
    predicted       : model predictions
    model_name      : label for the model
    seasonal_period : period for MASE (24 for hourly data)
    train_time_s    : training time in seconds (optional)

    Returns
    -------
    dict with keys: model, MAE, RMSE, MAPE, sMAPE, R2, MASE, [train_time_s]
    """
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    result = {
        "model":    model_name,
        "MAE":      round(mae(actual, predicted), 3),
        "RMSE":     round(rmse(actual, predicted), 3),
        "MAPE":     round(mape(actual, predicted), 3),
        "sMAPE":    round(smape(actual, predicted), 3),
        "R2":       round(r2_score(actual, predicted), 4),
        "MASE":     round(mase(actual, predicted, seasonal_period), 3),
    }
    if train_time_s is not None:
        result["train_time_s"] = round(train_time_s, 2)
    return result


def compare_models(results: Dict[str, Dict],
                   sort_by: str = "MAPE") -> pd.DataFrame:
    """
    Build a leaderboard DataFrame from a dict of evaluate() outputs.

    Parameters
    ----------
    results : {model_name: evaluate_dict, ...}
    sort_by : metric column to sort by (ascending)

    Returns
    -------
    pd.DataFrame ranked by sort_by
    """
    rows = list(results.values())
    df = pd.DataFrame(rows)
    df["Rank"] = df[sort_by].rank().astype(int)
    df = df.sort_values(sort_by).reset_index(drop=True)
    df.index += 1
    return df


def print_leaderboard(results: Dict[str, Dict]) -> None:
    """Pretty-print the comparison table."""
    df = compare_models(results)
    print("\n" + "=" * 72)
    print("  ELECTRICITY FORECASTING — MODEL LEADERBOARD")
    print("=" * 72)
    print(df.to_string(index=True))
    print("=" * 72)
    best = df.iloc[0]
    print(f"\n  Best model: {best['model']}  |  "
          f"MAPE: {best['MAPE']:.2f}%  |  RMSE: {best['RMSE']:.2f} MW\n")
