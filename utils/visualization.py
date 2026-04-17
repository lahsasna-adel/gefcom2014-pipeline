"""
utils/visualization.py
-----------------------
Plotting utilities for electricity forecasting results.

All functions return matplotlib Figure objects so they can be saved,
displayed, or embedded in reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d0f14",
    "axes.facecolor":    "#151820",
    "axes.edgecolor":    "#2a3050",
    "axes.labelcolor":   "#8890b0",
    "xtick.color":       "#555e80",
    "ytick.color":       "#555e80",
    "text.color":        "#e2e4f0",
    "grid.color":        "#1c2030",
    "grid.linewidth":    0.5,
    "legend.framealpha": 0.3,
    "legend.edgecolor":  "#2a3050",
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})

PALETTE = {
    "actual":  "#888890",
    "arima":   "#ff6b6b",
    "prophet": "#3dd68c",
    "xgb":     "#f5a623",
    "lgb":     "#fb923c",
    "rf":      "#2dd4bf",
    "lstm":    "#4a9eff",
    "tft":     "#a78bfa",
    "nbeats":  "#f472b6",
}


def _color(key: str) -> str:
    key = key.lower().replace(" ", "_").replace("-", "")
    for k, v in PALETTE.items():
        if k in key:
            return v
    return "#e2e4f0"


# ─── 1. Forecast vs Actual ────────────────────────────────────────────────────

def plot_forecast_vs_actual(actual: pd.Series,
                             predictions: Dict[str, np.ndarray],
                             title: str = "Forecast vs Actual — Electricity Demand",
                             n_display: int = 168) -> plt.Figure:
    """
    Plot actual demand alongside predictions from multiple models.

    Parameters
    ----------
    actual      : ground-truth demand Series with DatetimeIndex
    predictions : {model_name: array_of_predictions}
    n_display   : number of time steps to show (default 168 = 7 days)
    """
    idx = actual.index[:n_display]
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(idx, actual.values[:n_display],
            color=PALETTE["actual"], linewidth=1.5, label="Actual", zorder=5)

    for name, preds in predictions.items():
        ax.plot(idx, preds[:n_display],
                color=_color(name), linewidth=1, alpha=0.85,
                linestyle="--" if "arima" in name.lower() else "-",
                label=name)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    return fig


# ─── 2. Confidence Interval ───────────────────────────────────────────────────

def plot_confidence_interval(actual: pd.Series,
                              point_forecast: np.ndarray,
                              lower: np.ndarray,
                              upper: np.ndarray,
                              model_name: str = "LSTM",
                              n_display: int = 48) -> plt.Figure:
    """Plot forecast with 95% confidence interval shading."""
    idx = actual.index[:n_display]
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(idx, lower[:n_display], upper[:n_display],
                    alpha=0.2, color=_color(model_name), label="95% CI")
    ax.plot(idx, actual.values[:n_display],
            color=PALETTE["actual"], linewidth=1.5, label="Actual")
    ax.plot(idx, point_forecast[:n_display],
            color=_color(model_name), linewidth=1.5, label=model_name)

    ax.set_title(f"{model_name} — 95% Prediction Interval")
    ax.set_xlabel("Time")
    ax.set_ylabel("Demand (MW)")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    return fig


# ─── 3. Metrics Comparison Bar Chart ─────────────────────────────────────────

def plot_metrics_comparison(metrics_df: pd.DataFrame,
                             metric: str = "MAPE") -> plt.Figure:
    """
    Horizontal bar chart comparing models on a single metric.

    Parameters
    ----------
    metrics_df : output of utils.metrics.compare_models()
    metric     : column to plot ('MAE', 'RMSE', 'MAPE', 'R2')
    """
    df = metrics_df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.6)))

    colors = [_color(m) for m in df["model"]]
    bars = ax.barh(df["model"], df[metric], color=colors, height=0.55)

    for bar, val in zip(bars, df[metric]):
        ax.text(bar.get_width() + bar.get_width() * 0.01, bar.get_y() + bar.get_height() / 2,
                f" {val:.2f}", va="center", fontsize=8, color="#e2e4f0")

    ax.set_xlabel(metric)
    ax.set_title(f"Model comparison — {metric}")
    ax.grid(axis="x")
    fig.tight_layout()
    return fig


# ─── 4. Residual Diagnostics ──────────────────────────────────────────────────

def plot_residuals(actual: np.ndarray,
                   predicted: np.ndarray,
                   model_name: str = "Model") -> plt.Figure:
    """
    2×2 diagnostic panel:
      [0,0] Residuals over time
      [0,1] Residual histogram + normal curve
      [1,0] Actual vs Predicted scatter
      [1,1] MAE by hour of day
    """
    resids = actual - predicted
    col = _color(model_name)

    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Residuals over time
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(resids, color=col, linewidth=0.7, alpha=0.8)
    ax0.axhline(0, color="#555e80", linewidth=0.8, linestyle="--")
    ax0.fill_between(range(len(resids)), resids, 0, alpha=0.15, color=col)
    ax0.set_title("Residuals over time")
    ax0.set_xlabel("Step"); ax0.set_ylabel("Error (MW)")

    # Histogram
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(resids, bins=40, color=col, alpha=0.7, edgecolor="none")
    mu, sigma = np.mean(resids), np.std(resids)
    x = np.linspace(resids.min(), resids.max(), 200)
    from scipy.stats import norm
    ax1.plot(x, norm.pdf(x, mu, sigma) * len(resids) * (resids.max()-resids.min()) / 40,
             color="#e2e4f0", linewidth=1.2, linestyle="--", label="Normal fit")
    ax1.set_title(f"Residual distribution  (μ={mu:.1f}, σ={sigma:.1f})")
    ax1.set_xlabel("Residual (MW)"); ax1.legend(fontsize=7)

    # Actual vs Predicted
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(actual, predicted, color=col, s=5, alpha=0.5)
    mn, mx = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax2.plot([mn, mx], [mn, mx], color="#555e80", linewidth=1, linestyle="--")
    ax2.set_title("Actual vs Predicted")
    ax2.set_xlabel("Actual (MW)"); ax2.set_ylabel("Predicted (MW)")

    # MAE by hour of day
    ax3 = fig.add_subplot(gs[1, 1])
    n   = len(actual)
    hour_mae = [np.mean(np.abs(resids[h::24])) for h in range(24)]
    ax3.bar(range(24), hour_mae, color=col, alpha=0.8, width=0.7)
    ax3.set_xticks(range(0, 24, 3))
    ax3.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 3)])
    ax3.set_title("MAE by hour of day")
    ax3.set_xlabel("Hour"); ax3.set_ylabel("MAE (MW)")

    fig.suptitle(f"Residual diagnostics — {model_name}", fontsize=12, y=1.01)
    return fig


# ─── 5. Feature Importance ───────────────────────────────────────────────────

def plot_feature_importance(importance_series: pd.Series,
                             model_name: str = "XGBoost",
                             top_n: int = 15) -> plt.Figure:
    """
    Horizontal bar chart of top-N feature importances.

    Parameters
    ----------
    importance_series : pd.Series with feature names as index, importance as values
    """
    top = importance_series.head(top_n).sort_values()
    col = _color(model_name)

    fig, ax = plt.subplots(figsize=(9, max(3, top_n * 0.45)))
    ax.barh(top.index, top.values, color=col, height=0.6, alpha=0.85)
    ax.set_title(f"{model_name} — Top {top_n} feature importances")
    ax.set_xlabel("Importance score")
    ax.grid(axis="x")
    fig.tight_layout()
    return fig


# ─── 6. Training Curve ────────────────────────────────────────────────────────

def plot_training_curve(train_loss: List[float],
                         val_loss: List[float],
                         model_name: str = "LSTM") -> plt.Figure:
    """Plot train vs validation loss per epoch."""
    col = _color(model_name)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_loss, color=col,        linewidth=1.5, label="Train loss")
    ax.plot(val_loss,   color="#e2e4f0",  linewidth=1.5, linestyle="--", label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} — Training curve")
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout()
    return fig


# ─── 7. Full Dashboard (save to PNG) ─────────────────────────────────────────

def save_dashboard(actual: pd.Series,
                   predictions: Dict[str, np.ndarray],
                   metrics_df: pd.DataFrame,
                   output_path: str = "results/dashboard.png") -> None:
    """
    Save a 3-panel summary dashboard as a PNG file.
    Panels: Forecast vs Actual | MAPE bar chart | RMSE bar chart
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    # Top: forecast
    ax_fc = fig.add_subplot(gs[0, :])
    n = min(len(actual), 168)
    idx = actual.index[:n]
    ax_fc.plot(idx, actual.values[:n], color=PALETTE["actual"], lw=1.5, label="Actual")
    for name, preds in predictions.items():
        ax_fc.plot(idx, preds[:n], color=_color(name), lw=1, alpha=0.85, label=name)
    ax_fc.set_title("Forecast vs Actual"); ax_fc.legend(fontsize=7); ax_fc.grid(True)

    # Bottom-left: MAPE
    ax_mape = fig.add_subplot(gs[1, 0])
    df_m = metrics_df.sort_values("MAPE")
    ax_mape.barh(df_m["model"], df_m["MAPE"],
                 color=[_color(m) for m in df_m["model"]], height=0.55)
    ax_mape.set_title("MAPE (%) — lower is better"); ax_mape.grid(axis="x")

    # Bottom-right: RMSE
    ax_rmse = fig.add_subplot(gs[1, 1])
    df_r = metrics_df.sort_values("RMSE")
    ax_rmse.barh(df_r["model"], df_r["RMSE"],
                 color=[_color(m) for m in df_r["model"]], height=0.55)
    ax_rmse.set_title("RMSE (MW) — lower is better"); ax_rmse.grid(axis="x")

    fig.suptitle("Electricity Forecasting — Model Dashboard", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[viz] Dashboard saved → {output_path}")
    plt.close(fig)
