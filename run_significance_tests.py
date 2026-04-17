"""
run_significance_tests.py
--------------------------
Loads fs_experiments.csv and runs all Phase 1 statistical tests:

  1. Wilcoxon pairwise tests — subset comparison per model
  2. Wilcoxon pairwise tests — model comparison per subset
  3. Friedman test — all subsets simultaneously per model
  4. Nemenyi post-hoc — pairwise significance after Friedman
  5. Effect sizes (Cohen's d) for key comparisons

Outputs saved to results/fs/statistical/
  wilcoxon_subsets_<model>.csv
  wilcoxon_models_<subset>.csv
  friedman_results.csv
  nemenyi_<model>.csv
  effect_sizes.csv
  statistical_summary.csv

Usage
-----
    python run_significance_tests.py --input results/fs/fs_experiments.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
from utils.statistical_tests import (
    wilcoxon_test, wilcoxon_matrix,
    friedman_test, nemenyi_test,
    cohens_d, interpret_cohens_d,
    significance_symbol,
    print_wilcoxon_results, print_friedman_results
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_mape_series(df, model_key, subset_name):
    """Extract sorted fold-level MAPE values for one model-subset pair."""
    mask = (df["model_key"] == model_key) & (df["subset_name"] == subset_name)
    sub  = df[mask].sort_values("fold")
    return sub["MAPE"].tolist()


def build_mape_matrix(df, model_key, subsets):
    """
    Build a folds x subsets MAPE matrix for Friedman/Nemenyi tests.
    Rows = folds, columns = subsets.
    """
    rows = {}
    for s in subsets:
        vals = get_mape_series(df, model_key, s)
        if vals:
            rows[s] = vals

    if not rows:
        return pd.DataFrame()

    min_len = min(len(v) for v in rows.values())
    data    = {k: v[:min_len] for k, v in rows.items()}
    return pd.DataFrame(data,
                        index=[f"fold_{i+1}" for i in range(min_len)])


def build_model_mape_matrix(df, subset_name, models):
    """
    Build a folds x models MAPE matrix for a fixed subset.
    """
    rows = {}
    for m in models:
        vals = get_mape_series(df, m, subset_name)
        if vals:
            rows[m] = vals

    if not rows:
        return pd.DataFrame()

    min_len = min(len(v) for v in rows.values())
    data    = {k: v[:min_len] for k, v in rows.items()}
    return pd.DataFrame(data,
                        index=[f"fold_{i+1}" for i in range(min_len)])


# ─── Main analysis ────────────────────────────────────────────────────────────

def run_all_tests(experiments_path: str,
                   output_dir: str = "results/fs/statistical") -> None:

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(experiments_path)
    print(f"[stats] Loaded {len(df)} rows from {experiments_path}")
    print(f"[stats] Columns: {list(df.columns)}")

    # Detect model and subset names
    models  = sorted(df["model_key"].unique())
    subsets = sorted(df["subset_name"].unique())
    model_display = {"xgb": "XGBoost", "lgb": "LightGBM", "rf": "RandomForest"}

    print(f"[stats] Models  : {models}")
    print(f"[stats] Subsets : {subsets}")

    # ── SECTION 1: Friedman test per model ───────────────────────────────────
    print("\n" + "="*65)
    print("  SECTION 1: FRIEDMAN TEST — all subsets per model")
    print("="*65)

    friedman_rows = []
    for mk in models:
        mat = build_mape_matrix(df, mk, subsets)
        if mat.empty or len(mat.columns) < 3:
            continue
        result = friedman_test(mat)
        result["model"] = model_display.get(mk, mk)
        friedman_rows.append(result)
        print_friedman_results(result)

    friedman_df = pd.DataFrame(friedman_rows)
    friedman_df.to_csv(f"{output_dir}/friedman_results.csv", index=False)
    print(f"[stats] Friedman results → {output_dir}/friedman_results.csv")

    # ── SECTION 2: Nemenyi post-hoc per model ────────────────────────────────
    print("\n" + "="*65)
    print("  SECTION 2: NEMENYI POST-HOC — pairwise subset comparison")
    print("="*65)

    # Exclude calendar_only for readability (MAPE ~16% skews ranks)
    core_subsets = [s for s in subsets if s != "calendar_only"]

    for mk in models:
        mat = build_mape_matrix(df, mk, core_subsets)
        if mat.empty:
            continue
        p_mat, sig_mat, avg_ranks, CD = nemenyi_test(mat)
        mname = model_display.get(mk, mk)

        print(f"\n  {mname} — Critical Difference: {CD:.3f}")
        print(f"  Average ranks (lower rank = better MAPE):")
        for name, rank in avg_ranks.sort_values().items():
            print(f"    {name:<16} : {rank:.3f}")
        print(f"\n  Significance matrix (* p<0.05, ** p<0.01, *** p<0.001):")
        print(sig_mat.to_string())

        sig_mat.to_csv(f"{output_dir}/nemenyi_{mk}.csv")
        p_mat.to_csv(f"{output_dir}/nemenyi_{mk}_pvalues.csv")
        avg_ranks.to_csv(f"{output_dir}/nemenyi_{mk}_ranks.csv")
        print(f"[stats] Nemenyi ({mname}) → {output_dir}/nemenyi_{mk}.csv")

    # ── SECTION 3: Wilcoxon — key subset comparisons per model ───────────────
    print("\n" + "="*65)
    print("  SECTION 3: WILCOXON — key subset comparisons per model")
    print("="*65)

    # Focus on the most informative comparisons
    key_pairs = [
        ("top_30",       "all",          "Top-30 vs All-118"),
        ("top_30",       "top_20",       "Top-30 vs Top-20"),
        ("top_30",       "zabin_22",     "Top-30 vs Zabin-22"),
        ("top_20",       "zabin_22",     "Top-20 vs Zabin-22"),
        ("top_50",       "all",          "Top-50 vs All-118"),
        ("top_10",       "lag_only",     "Top-10 vs Lag-only"),
        ("lag_only",     "zabin_22",     "Lag-only vs Zabin-22"),
        ("calendar_only","lag_only",     "Calendar-only vs Lag-only"),
    ]

    all_wilcoxon_subset = []
    for mk in models:
        mname = model_display.get(mk, mk)
        results = []
        for s1, s2, label in key_pairs:
            m1 = get_mape_series(df, mk, s1)
            m2 = get_mape_series(df, mk, s2)
            if not m1 or not m2:
                continue
            r = wilcoxon_test(m1, m2, s1, s2)
            r["comparison_label"] = label
            r["model"]            = mname
            r["cohen_d"]          = round(cohens_d(m1, m2), 4)
            r["effect_interp"]    = interpret_cohens_d(cohens_d(m1, m2))
            results.append(r)
            all_wilcoxon_subset.append(r)

        print_wilcoxon_results(results)
        pd.DataFrame(results).to_csv(
            f"{output_dir}/wilcoxon_subsets_{mk}.csv", index=False)

    # ── SECTION 4: Wilcoxon — model comparison on key subsets ────────────────
    print("\n" + "="*65)
    print("  SECTION 4: WILCOXON — model comparison on key subsets")
    print("="*65)

    focus_subsets = ["top_30", "all", "zabin_22"]
    model_pairs   = list(combinations(models, 2))
    all_wilcoxon_models = []

    for subset in focus_subsets:
        results = []
        for m1, m2 in model_pairs:
            mape1 = get_mape_series(df, m1, subset)
            mape2 = get_mape_series(df, m2, subset)
            if not mape1 or not mape2:
                continue
            mn1 = model_display.get(m1, m1)
            mn2 = model_display.get(m2, m2)
            r   = wilcoxon_test(mape1, mape2, mn1, mn2)
            r["subset"]       = subset
            r["cohen_d"]      = round(cohens_d(mape1, mape2), 4)
            r["effect_interp"]= interpret_cohens_d(cohens_d(mape1, mape2))
            results.append(r)
            all_wilcoxon_models.append(r)

        print(f"\n  Subset: {subset}")
        print_wilcoxon_results(results)

    pd.DataFrame(all_wilcoxon_models).to_csv(
        f"{output_dir}/wilcoxon_models.csv", index=False)
    print(f"[stats] Model comparisons → {output_dir}/wilcoxon_models.csv")

    # ── SECTION 5: Comprehensive summary table ────────────────────────────────
    print("\n" + "="*65)
    print("  SECTION 5: COMPREHENSIVE SUMMARY TABLE")
    print("="*65)

    summary_rows = []
    for mk in models:
        mname = model_display.get(mk, mk)
        for s in core_subsets:
            mapes = get_mape_series(df, mk, s)
            if not mapes:
                continue
            summary_rows.append({
                "model":        mname,
                "subset":       s,
                "n_folds":      len(mapes),
                "MAPE_mean":    round(np.mean(mapes), 4),
                "MAPE_std":     round(np.std(mapes, ddof=1), 4),
                "MAPE_min":     round(np.min(mapes), 4),
                "MAPE_max":     round(np.max(mapes), 4),
                "MAPE_median":  round(np.median(mapes), 4),
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(["model","MAPE_mean"])
    summary_df.to_csv(f"{output_dir}/statistical_summary.csv", index=False)
    print(f"[stats] Summary table → {output_dir}/statistical_summary.csv")
    print(summary_df.to_string(index=False))

    # ── SECTION 6: Publication-ready p-value table ────────────────────────────
    print("\n" + "="*65)
    print("  SECTION 6: PUBLICATION-READY P-VALUE TABLE")
    print("="*65)

    pub_rows = []
    for r in all_wilcoxon_subset:
        pub_rows.append({
            "Model":        r.get("model",""),
            "Comparison":   r.get("comparison_label",""),
            "MAPE_A":       r.get("mean_mape1",""),
            "MAPE_B":       r.get("mean_mape2",""),
            "W_statistic":  r.get("statistic",""),
            "p_value":      r.get("p_value",""),
            "Significance": significance_symbol(r.get("p_value",1.0)),
            "Cohen_d":      r.get("cohen_d",""),
            "Effect_size":  r.get("effect_interp",""),
            "Winner":       r.get("winner",""),
        })

    pub_df = pd.DataFrame(pub_rows)
    pub_df.to_csv(f"{output_dir}/publication_table.csv", index=False)
    print(f"[stats] Publication table → {output_dir}/publication_table.csv")
    print(pub_df[["Model","Comparison","p_value",
                   "Significance","Effect_size","Winner"]].to_string(index=False))

    print(f"\n[stats] All results saved to {output_dir}/")
    print("[stats] Done ✓")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run statistical significance tests on feature selection results"
    )
    parser.add_argument("--input",      type=str,
                        default="results/fs/fs_experiments.csv",
                        help="Path to fs_experiments.csv")
    parser.add_argument("--output-dir", type=str,
                        default="results/fs/statistical",
                        help="Output directory for test results")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        print("Make sure you have run main_fs.py first.")
        return

    run_all_tests(args.input, args.output_dir)


if __name__ == "__main__":
    main()
