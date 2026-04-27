"""
utils/metrics.py
----------------
Evaluation metrics for electricity demand forecasting.

Metrics returned by evaluate()
--------------------------------
MAE       : Mean Absolute Error (MW)
RMSE      : Root Mean Squared Error (MW)
MAPE      : Mean Absolute Percentage Error (%)
R2        : Coefficient of Determination
pcBias    : Percentage Bias (signed, %) — NEW (Smyl & Hua, 2019)

pcBias definition (Smyl & Hua, 2019, Table 1)
----------------------------------------------
    pcBias = 100 × mean(ŷ - y) / mean(y)

Interpretation:
  pcBias > 0  → model systematically OVER-forecasts (positive bias)
  pcBias < 0  → model systematically UNDER-forecasts (negative bias)
  pcBias = 0  → perfectly unbiased on average

This metric is operationally important for electricity grid operators:
  - Consistent over-forecasting wastes reserve capacity
  - Consistent under-forecasting risks grid instability

In Smyl & Hua (2019):
  - NN:  pcBias = -1.28  (under-forecasts)
  - GBM: pcBias = +0.66  (over-forecasts)
  - QRF: pcBias = +0.12 to +0.51 (slight over-forecast)
  - Ensemble average cancels opposite biases → pcBias ≈ +0.03

Reference
---------
Smyl, S., & Hua, N. G. (2019). Machine learning methods for GEFCom2017
probabilistic load forecasting. International Journal of Forecasting,
35(4), 1424–1431.
"""

import numpy as np


def evaluate(y_true: np.ndarray,
             y_pred: np.ndarray,
             model_name: str = "",
             train_time_s: float = 0.0) -> dict:
    """
    Compute regression metrics for electricity demand forecasting.

    Parameters
    ----------
    y_true       : array of true demand values (MW)
    y_pred       : array of predicted demand values (MW)
    model_name   : label string stored in the returned dict
    train_time_s : training time in seconds (passed through, not computed here)

    Returns
    -------
    dict with keys:
        model_key, MAE, RMSE, MAPE, R2, pcBias, train_time_s
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # ── Guard against degenerate inputs ──────────────────────────────────────
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "model_key":    model_name,
            "MAE":          np.nan,
            "RMSE":         np.nan,
            "MAPE":         np.nan,
            "R2":           np.nan,
            "pcBias":       np.nan,
            "train_time_s": train_time_s,
        }

    # ── Core metrics ──────────────────────────────────────────────────────────
    residuals = y_pred - y_true                          # ŷ - y (signed)
    abs_res   = np.abs(residuals)

    mae  = float(np.mean(abs_res))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # MAPE: avoid division by zero on near-zero demand values
    nonzero_mask = np.abs(y_true) > 1e-8
    if nonzero_mask.sum() == 0:
        mape = np.nan
    else:
        mape = float(
            100.0 * np.mean(abs_res[nonzero_mask] / np.abs(y_true[nonzero_mask]))
        )

    # R² = 1 - SS_res / SS_tot
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan

    # ── pcBias (Smyl & Hua, 2019) ─────────────────────────────────────────────
    # pcBias = 100 × mean(ŷ - y) / mean(y)
    # Measures systematic over/under-forecasting as a percentage of mean demand.
    mean_y = float(np.mean(y_true))
    if abs(mean_y) > 1e-8:
        pc_bias = float(100.0 * np.mean(residuals) / mean_y)
    else:
        pc_bias = np.nan

    return {
        "model_key":    model_name,
        "MAE":          round(mae,     4),
        "RMSE":         round(rmse,    4),
        "MAPE":         round(mape,    4),
        "R2":           round(r2,      4),
        "pcBias":       round(pc_bias, 4),   # NEW — Smyl & Hua (2019)
        "train_time_s": round(train_time_s, 2),
    }
