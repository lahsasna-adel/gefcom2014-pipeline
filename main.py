"""
main.py
-------
Electricity Forecasting Pipeline — End-to-end orchestrator.

Usage
-----
    # Run with demo data (auto-generated):
    python main.py --demo

    # Run with your CSV file:
    python main.py --csv path/to/your/data.csv

    # Run specific models only:
    python main.py --csv data.csv --models arima prophet xgb lstm

    # Tune XGBoost hyperparameters with Optuna before fitting:
    python main.py --csv data.csv --tune xgb --tune-trials 100

    # Save all plots to results/ directory:
    python main.py --csv data.csv --save-plots

CSV format expected
-------------------
    timestamp,demand[,temperature,humidity,holiday,...]
    2024-01-01 00:00,523.4[,15.2,60,0,...]
    ...
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Local imports ──────────────────────────────────────────────────────────────
from utils.data_loader      import load_csv, engineer_features, time_series_split
from utils.metrics          import evaluate, compare_models, print_leaderboard
from utils.visualization    import (plot_forecast_vs_actual, plot_residuals,
                                     plot_feature_importance, save_dashboard)
from utils.cross_validation import (run_cv, print_cv_leaderboard,
                                     cv_results_to_dataframe, cv_summary_dataframe)


# ─── Demo data generator ──────────────────────────────────────────────────────

def generate_demo_data(n_days: int = 60, freq: str = "h") -> pd.DataFrame:
    """Generate realistic synthetic electricity demand data."""
    N = n_days * 24
    idx = pd.date_range("2024-01-01", periods=N, freq=freq)
    t   = np.arange(N)

    daily   = np.sin((t % 24 - 6) * np.pi / 12) * 130
    weekly  = np.sin(t * 2 * np.pi / 168) * 55
    yearly  = np.sin(t * 2 * np.pi / 8760) * 80
    trend   = t * 0.04
    noise   = np.random.default_rng(42).normal(0, 40, N)
    peak    = np.where((t % 24 == 8) | (t % 24 == 18), 70, 0)
    weekend = np.where(pd.DatetimeIndex(idx).dayofweek >= 5, -80, 0)

    demand = np.maximum(180, 520 + trend + daily + weekly + yearly + noise + peak + weekend)
    temp   = 15 + 8 * np.sin((t % 24 - 14) * np.pi / 12) + np.random.default_rng(7).normal(0, 2, N)

    df = pd.DataFrame({
        "demand":      demand.round(1),
        "temperature": temp.round(1),
        "holiday":     (pd.DatetimeIndex(idx).dayofweek == 6).astype(int),
    }, index=idx)
    print(f"[demo] Generated {N:,} hourly observations "
          f"({df['demand'].min():.0f}–{df['demand'].max():.0f} MW).")
    return df


# ─── Model registry ───────────────────────────────────────────────────────────

def run_all_models(df: pd.DataFrame,
                   selected: list,
                   test_size: float = 0.2,
                   tune: list = None,
                   tune_trials: int = 50) -> dict:
    """
    Fit and evaluate all selected models. Returns a results dict.

    results = {
        model_name: {
            "preds":      np.ndarray,
            "metrics":    dict,
            "importance": pd.Series or None,
        }
    }
    """
    tune = tune or []

    # ── Feature engineering & split ─────────────────────────────────────────
    print("\n[pipeline] Engineering features …")
    fe_df = engineer_features(df, lags=[1,2,3,24,48,168], rolling_windows=[6,24,168])

    X_train, y_train, X_test, y_test = time_series_split(
        fe_df, target="demand", test_size=test_size
    )

    # Raw series (no feature engineering) for sequence models
    raw   = df["demand"].values
    split = int(len(raw) * (1 - test_size))
    y_train_raw = raw[:split]
    y_test_raw  = raw[split:]

    results = {}

    # ── SARIMA ───────────────────────────────────────────────────────────────
    if "arima" in selected:
        from models.sarima_model import SARIMAForecaster
        model = SARIMAForecaster(seasonal_period=24)
        model.fit(y_train_raw)
        preds, _ = model.predict(len(y_test_raw))
        results["SARIMA"] = {
            "preds":      preds,
            "metrics":    evaluate(y_test_raw, preds, "SARIMA", train_time_s=model.train_time_),
            "importance": None,
        }

    # ── Prophet ──────────────────────────────────────────────────────────────
    if "prophet" in selected:
        from models.prophet_model import ProphetForecaster
        extra_regs = [c for c in ["temperature", "holiday"] if c in df.columns]
        model = ProphetForecaster(
            country_holidays="DZ",
            extra_regressors=extra_regs if extra_regs else [],
        )
        model.fit(
            df["demand"].iloc[:split],
            regressors_train=df[extra_regs].iloc[:split] if extra_regs else None,
        )
        fc = model.predict(
            df.index[split:],
            regressors_future=df[extra_regs].iloc[split:] if extra_regs else None,
        )
        preds = fc["yhat"].values
        results["Prophet"] = {
            "preds":      preds,
            "metrics":    evaluate(y_test_raw, preds, "Prophet", train_time_s=model.train_time_),
            "importance": None,
        }

    # ── XGBoost ──────────────────────────────────────────────────────────────
    if "xgb" in selected:
        from models.tree_models import XGBoostForecaster
        model = XGBoostForecaster()
        if "xgb" in tune:
            print("[XGBoost] Running Optuna hyperparameter search …")
            model.tune(X_train, y_train, n_trials=tune_trials)
        model.fit(X_train, y_train, X_test, y_test)
        preds = model.predict(X_test)
        results["XGBoost"] = {
            "preds":      preds,
            "metrics":    evaluate(y_test.values, preds, "XGBoost", train_time_s=model.train_time_),
            "importance": model.feature_importances_,
        }

    # ── LightGBM ─────────────────────────────────────────────────────────────
    if "lgb" in selected:
        from models.tree_models import LightGBMForecaster
        model = LightGBMForecaster()
        if "lgb" in tune:
            model.tune(X_train, y_train, n_trials=tune_trials)
        model.fit(X_train, y_train, X_test, y_test)
        preds = model.predict(X_test)
        results["LightGBM"] = {
            "preds":      preds,
            "metrics":    evaluate(y_test.values, preds, "LightGBM", train_time_s=model.train_time_),
            "importance": model.feature_importances_,
        }

    # ── Random Forest ─────────────────────────────────────────────────────────
    if "rf" in selected:
        from models.tree_models import RandomForestForecaster
        model = RandomForestForecaster()
        if "rf" in tune:
            model.tune(X_train, y_train, n_trials=tune_trials)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results["RandomForest"] = {
            "preds":      preds,
            "metrics":    evaluate(y_test.values, preds, "RandomForest", train_time_s=model.train_time_),
            "importance": model.feature_importances_,
        }

    # ── LSTM ─────────────────────────────────────────────────────────────────
    if "lstm" in selected:
        from models.lstm_model import LSTMForecaster
        model = LSTMForecaster(lookback=48, horizon=24, epochs=50)
        model.fit(y_train_raw)
        # Rolling prediction over the test set (step-by-step)
        preds = []
        for i in range(len(y_test_raw)):
            ctx = raw[max(0, split + i - 48): split + i]
            p   = model._predict_one(ctx)
            preds.append(p)
        preds = np.array(preds)
        results["LSTM"] = {
            "preds":      preds,
            "metrics":    evaluate(y_test_raw, preds, "LSTM", train_time_s=model.train_time_),
            "importance": None,
        }

    # ── TFT (optional — requires pytorch-forecasting) ─────────────────────
    if "tft" in selected:
        try:
            from models.deep_models import TFTForecaster
            split_dt  = df.index[split]
            df_train  = df.iloc[:split]
            df_val    = df.iloc[split:]
            model = TFTForecaster(max_encoder_length=168, max_prediction_length=24)
            model.fit(df_train, df_val, target="demand")
            fc = model.predict(df_val)
            preds = fc["p50"][:len(y_test_raw)]
            results["TFT"] = {
                "preds":      preds,
                "metrics":    evaluate(y_test_raw[:len(preds)], preds, "TFT", train_time_s=model.train_time_),
                "importance": None,
            }
        except Exception as e:
            print(f"[TFT] Skipped — {e}")

    # ── N-BEATS (optional) ────────────────────────────────────────────────
    if "nbeats" in selected:
        try:
            from models.deep_models import NBEATSForecaster
            df_train = df[["demand"]].iloc[:split]
            df_val   = df[["demand"]].iloc[split:]
            model = NBEATSForecaster(max_encoder_length=168, max_prediction_length=24)
            model.fit(df_train, df_val, target="demand")
            preds = model.predict(df_val)[:len(y_test_raw)]
            results["N-BEATS"] = {
                "preds":      preds,
                "metrics":    evaluate(y_test_raw[:len(preds)], preds, "N-BEATS", train_time_s=model.train_time_),
                "importance": None,
            }
        except Exception as e:
            print(f"[N-BEATS] Skipped — {e}")

    return results, y_test, y_test_raw, df.index[split:]


# ─── CV plot helper ───────────────────────────────────────────────────────────

def _plot_cv_results(cv_results: dict) -> None:
    """
    Save three plots for the CV run:
      1. MAPE mean ± std bar chart (all models)
      2. RMSE mean ± std bar chart (all models)
      3. Per-fold MAPE line chart (shows stability across folds)
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({
        "figure.facecolor": "#0d0f14", "axes.facecolor": "#151820",
        "axes.edgecolor": "#2a3050", "text.color": "#e2e4f0",
        "axes.labelcolor": "#8890b0", "xtick.color": "#555e80",
        "ytick.color": "#555e80", "grid.color": "#1c2030",
        "grid.linewidth": 0.5, "font.family": "monospace",
    })

    from utils.visualization import _color

    rows = [(k, v) for k, v in cv_results.items() if "mean" in v]
    rows.sort(key=lambda x: x[1]["mean"].get("MAPE", 999))
    names  = [r[0] for r in rows]
    colors = [_color(n) for n in names]

    # ── 1 & 2: MAPE and RMSE bar charts with error bars ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["MAPE", "RMSE"]):
        means = [r[1]["mean"].get(metric, 0) for r in rows]
        stds  = [r[1]["std"].get(metric, 0)  for r in rows]
        bars  = ax.barh(names, means, xerr=stds, color=colors,
                        height=0.55, capsize=4,
                        error_kw={"ecolor": "#8890b0", "elinewidth": 1})
        unit  = "%" if metric == "MAPE" else " MW"
        for bar, val, std in zip(bars, means, stds):
            ax.text(bar.get_width() + std + 0.2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}±{std:.2f}{unit}",
                    va="center", fontsize=8, color="#e2e4f0")
        n_folds = len(rows[0][1]["fold_metrics"]) if rows else 0
        ax.set_title(f"{metric} — mean ± std across {n_folds} folds")
        ax.set_xlabel(f"{metric} ({unit.strip()})")
        ax.grid(axis="x", alpha=0.4)

    fig.suptitle("Cross-Validation Results", fontsize=13)
    fig.tight_layout()
    fig.savefig("results/cv_metrics.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("[CV] Metrics plot saved   → results/cv_metrics.png")

    # ── 3: Per-fold MAPE line chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, res), col in zip(rows, colors):
        fold_mapes = [m.get("MAPE", np.nan) for m in res["fold_metrics"]]
        folds      = list(range(1, len(fold_mapes) + 1))
        ax.plot(folds, fold_mapes, marker="o", color=col,
                linewidth=1.5, markersize=5, label=name)

    n_folds = len(rows[0][1]["fold_metrics"]) if rows else 1
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE per fold — model stability across time windows")
    ax.set_xticks(range(1, n_folds + 1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/cv_fold_stability.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("[CV] Stability plot saved → results/cv_fold_stability.png")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Electricity demand forecasting pipeline")
    parser.add_argument("--csv",         type=str, default=None, help="Path to CSV file")
    parser.add_argument("--demo",        action="store_true",    help="Use generated demo data")
    parser.add_argument("--models",      nargs="+", default=["arima","prophet","xgb","lgb","rf","lstm"],
                        help="Models to run: arima prophet xgb lgb rf lstm tft nbeats")
    parser.add_argument("--test-size",   type=float, default=0.2, help="Test fraction (default 0.2)")
    parser.add_argument("--tune",        nargs="*", default=[],  help="Models to tune with Optuna")
    parser.add_argument("--tune-trials", type=int,  default=50,  help="Optuna trials (default 50)")
    parser.add_argument("--save-plots",  action="store_true",    help="Save plots to results/")
    parser.add_argument("--cv",          action="store_true",    help="Run time-series cross-validation")
    parser.add_argument("--cv-splits",   type=int, default=5,    help="Number of CV folds (default 5)")
    args = parser.parse_args()

    if not args.csv and not args.demo:
        print("Provide --csv <path> or --demo. Run with --help for usage.")
        return

    Path("results").mkdir(exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    if args.demo:
        df = generate_demo_data(n_days=60)
    else:
        df = load_csv(args.csv)

    # ── Cross-validation mode ─────────────────────────────────────────────
    if args.cv:
        cv_results = run_cv(
            df,
            selected=args.models,
            n_splits=args.cv_splits,
            test_size=args.test_size,
            horizon=24,
            tune=args.tune,
        )

        # Print leaderboard
        print_cv_leaderboard(cv_results)

        # Save per-fold details
        fold_df = cv_results_to_dataframe(cv_results)
        fold_df.to_csv("results/cv_fold_details.csv", index=False)
        print("[CV] Per-fold details saved → results/cv_fold_details.csv")

        # Save summary (mean ± std)
        summary_df = cv_summary_dataframe(cv_results)
        summary_df.to_csv("results/cv_summary.csv", index=False)
        print("[CV] Summary saved         → results/cv_summary.csv")

        # Plot CV metrics if requested
        if args.save_plots:
            _plot_cv_results(cv_results)

        print("\n[pipeline] Cross-validation done ✓")
        return

    # ── Single train/test split mode ──────────────────────────────────────
    print(f"\n[pipeline] Running: {args.models}")
    results, y_test, y_test_raw, test_index = run_all_models(
        df, selected=args.models,
        test_size=args.test_size,
        tune=args.tune, tune_trials=args.tune_trials,
    )

    # ── Leaderboard ───────────────────────────────────────────────────────
    metrics_dict = {k: v["metrics"] for k, v in results.items()}
    print_leaderboard(metrics_dict)
    metrics_df = compare_models(metrics_dict)
    metrics_df.to_csv("results/leaderboard.csv", index=False)
    print("[pipeline] Leaderboard saved → results/leaderboard.csv")

    # ── Plots ─────────────────────────────────────────────────────────────
    if args.save_plots:
        import matplotlib.pyplot as plt

        # Actual series for plotting
        actual_series = pd.Series(y_test_raw, index=test_index[:len(y_test_raw)])
        preds_dict    = {k: v["preds"] for k, v in results.items()}

        # Forecast vs Actual
        fig = plot_forecast_vs_actual(actual_series, preds_dict)
        fig.savefig("results/forecast_vs_actual.png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        # Residuals for each model
        for name, res in results.items():
            n = min(len(y_test_raw), len(res["preds"]))
            fig = plot_residuals(y_test_raw[:n], res["preds"][:n], model_name=name)
            fig.savefig(f"results/residuals_{name.lower()}.png",
                        dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

        # Feature importance
        for name, res in results.items():
            if res["importance"] is not None:
                fig = plot_feature_importance(res["importance"], model_name=name)
                fig.savefig(f"results/importance_{name.lower()}.png",
                            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
                plt.close(fig)

        # Dashboard summary
        save_dashboard(actual_series, preds_dict, metrics_df,
                       output_path="results/dashboard.png")

        print("[pipeline] All plots saved to results/")

    print("\n[pipeline] Done ✓")


if __name__ == "__main__":
    main()
