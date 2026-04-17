"""
utils/feature_selection.py
---------------------------
Ensemble filter-based feature selection for electricity demand forecasting.

Eight independent filter methods are applied and their rankings aggregated
using three strategies:
  A — Average rank       (primary method)
  B — Borda count
  C — Majority vote at threshold

Methods
-------
  1. Pearson correlation      (linear)
  2. Spearman correlation     (monotonic / rank-based)
  3. Mutual Information       (nonlinear, information-theoretic)
  4. ANOVA F-score            (variance ratio)
  5. LASSO (L1)               (regularisation-based shrinkage)
  6. Random Forest importance (nonlinear, ensemble-based)
  7. XGBoost importance       (nonlinear, gradient boosting)
  8. mRMR                     (min redundancy max relevance)

Usage
-----
    from utils.feature_selection import run_filter_selection, get_top_features
    results = run_filter_selection(X_train, y_train)
    top20   = get_top_features(results, n=20, strategy="average_rank")
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")


# ─── Individual filter methods ────────────────────────────────────────────────

def pearson_ranking(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Absolute Pearson correlation coefficient with target."""
    corr = X.corrwith(y, method="pearson").abs()
    return corr.fillna(0)


def spearman_ranking(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Absolute Spearman rank correlation coefficient with target."""
    corr = X.corrwith(y, method="spearman").abs()
    return corr.fillna(0)


def mutual_info_ranking(X: pd.DataFrame, y: pd.Series,
                         random_state: int = 42) -> pd.Series:
    """Mutual information between each feature and the target."""
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(X.fillna(0), y, random_state=random_state)
    return pd.Series(mi, index=X.columns)


def anova_f_ranking(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    ANOVA F-score between each feature and the target.
    Discretises the continuous target into quintiles for F-test.
    """
    from sklearn.feature_selection import f_classif
    # Bin target into 5 classes for ANOVA
    y_binned = pd.qcut(y, q=5, labels=False, duplicates="drop")
    f_scores, _ = f_classif(X.fillna(0), y_binned)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    return pd.Series(f_scores, index=X.columns)


def lasso_ranking(X: pd.DataFrame, y: pd.Series,
                   cv: int = 5, max_iter: int = 5000) -> pd.Series:
    """
    LASSO L1 regularisation coefficient magnitudes.
    Uses LassoCV to select the optimal alpha via cross-validation.
    Features with zero coefficients receive a score of 0.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    lasso   = LassoCV(cv=cv, max_iter=max_iter, random_state=42, n_jobs=-1)
    lasso.fit(X_scaled, y)
    coef = np.abs(lasso.coef_)
    return pd.Series(coef, index=X.columns)


def rf_importance_ranking(X: pd.DataFrame, y: pd.Series,
                           n_estimators: int = 200,
                           random_state: int = 42) -> pd.Series:
    """
    Random Forest feature importance (mean decrease in impurity).
    Uses a lightweight forest for speed.
    """
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=8,
        min_samples_leaf=5, random_state=random_state, n_jobs=-1
    )
    rf.fit(X.fillna(0), y)
    return pd.Series(rf.feature_importances_, index=X.columns)


def xgb_importance_ranking(X: pd.DataFrame, y: pd.Series,
                             random_state: int = 42) -> pd.Series:
    """
    XGBoost feature importance (gain-based).
    Uses a lightweight model for speed.
    """
    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, n_jobs=-1, verbosity=0,
        eval_metric="rmse"
    )
    xgb.fit(X.fillna(0), y)
    imp = xgb.get_booster().get_score(importance_type="gain")
    scores = pd.Series({col: imp.get(col, 0.0) for col in X.columns})
    return scores


def mrmr_ranking(X: pd.DataFrame, y: pd.Series,
                  n_features: Optional[int] = None) -> pd.Series:
    """
    Minimum Redundancy Maximum Relevance (mRMR) ranking.
    Returns features ordered by mRMR relevance score.
    """
    from mrmr import mrmr_regression

    if n_features is None:
        n_features = len(X.columns)

    try:
        selected = mrmr_regression(
            X=X.fillna(0).reset_index(drop=True),
            y=y.reset_index(drop=True),
            K=n_features
        )
        # Build score from rank position (top rank = highest score)
        scores = pd.Series(0.0, index=X.columns)
        for rank, feat in enumerate(selected):
            scores[feat] = n_features - rank
        return scores
    except Exception as e:
        print(f"  [mRMR] Warning: {e} — falling back to MI ranking")
        return mutual_info_ranking(X, y)


# ─── Rank aggregation ─────────────────────────────────────────────────────────

def scores_to_ranks(scores: pd.Series) -> pd.Series:
    """
    Convert a score series to ranks.
    Higher scores → lower rank numbers (rank 1 = best).
    Ties are broken by average ranking.
    """
    return scores.rank(ascending=False, method="average")


def average_rank_aggregation(rank_df: pd.DataFrame) -> pd.Series:
    """
    Strategy A — Average rank across all methods.
    Lower average rank = more informative feature.
    """
    return rank_df.mean(axis=1).rename("average_rank")


def borda_count_aggregation(rank_df: pd.DataFrame) -> pd.Series:
    """
    Strategy B — Borda count.
    Each feature receives (N - rank) points per method.
    Higher total points = more informative feature.
    """
    N = len(rank_df)
    borda = (N - rank_df).sum(axis=1)
    return borda.rename("borda_score")


def majority_vote_aggregation(rank_df: pd.DataFrame,
                               top_k: int = 20) -> pd.Series:
    """
    Strategy C — Majority vote at top-K threshold.
    Count how many methods include each feature in their top-K.
    Higher count = more consensus agreement.
    """
    votes = (rank_df <= top_k).sum(axis=1)
    return votes.rename("majority_votes")


# ─── Master function ──────────────────────────────────────────────────────────

def run_filter_selection(X_train: pd.DataFrame,
                          y_train: pd.Series,
                          methods: Optional[List[str]] = None,
                          subsample: int = 10000,
                          verbose: bool = True) -> Dict:
    """
    Apply all filter methods and aggregate rankings.

    Parameters
    ----------
    X_train   : feature matrix (train split only — no leakage)
    y_train   : target series
    methods   : list of method names to run — runs all if None
    subsample : max rows to use for heavy methods (MI, RF, XGB, mRMR)
               Set to None to use all rows (slower)
    verbose   : print progress

    Returns
    -------
    dict with keys:
        "scores"       : DataFrame — raw scores per method
        "ranks"        : DataFrame — ranks per method (1=best)
        "average_rank" : Series — Strategy A aggregation
        "borda"        : Series — Strategy B aggregation
        "majority"     : Series — Strategy C aggregation
        "summary"      : DataFrame — all aggregations + final recommendation
    """
    all_methods = ["pearson", "spearman", "mutual_info", "anova_f",
                   "lasso", "rf_importance", "xgb_importance", "mrmr"]
    if methods is None:
        methods = all_methods

    # Subsample for heavy methods to control compute time
    if subsample and len(X_train) > subsample:
        idx_sub = np.random.default_rng(42).choice(len(X_train), subsample,
                                                    replace=False)
        idx_sub = np.sort(idx_sub)
        X_sub = X_train.iloc[idx_sub]
        y_sub = y_train.iloc[idx_sub]
    else:
        X_sub, y_sub = X_train, y_train

    method_funcs = {
        "pearson":        (pearson_ranking,        X_sub,   y_sub),
        "spearman":       (spearman_ranking,        X_sub,   y_sub),
        "mutual_info":    (mutual_info_ranking,     X_sub,   y_sub),
        "anova_f":        (anova_f_ranking,         X_sub,   y_sub),
        "lasso":          (lasso_ranking,           X_train, y_train),
        "rf_importance":  (rf_importance_ranking,   X_sub,   y_sub),
        "xgb_importance": (xgb_importance_ranking,  X_sub,   y_sub),
        "mrmr":           (mrmr_ranking,            X_sub,   y_sub),
    }

    scores_dict = {}
    for name in methods:
        if name not in method_funcs:
            print(f"  [FS] Unknown method '{name}' — skipping")
            continue
        if verbose:
            print(f"  [FS] Running {name:<20} ...", end=" ", flush=True)
        try:
            func, X_arg, y_arg = method_funcs[name]
            score = func(X_arg, y_arg)
            score = score.reindex(X_train.columns).fillna(0)
            scores_dict[name] = score
            if verbose:
                print("done")
        except Exception as e:
            if verbose:
                print(f"FAILED — {e}")

    if not scores_dict:
        raise RuntimeError("All feature selection methods failed.")

    # Build scores and ranks DataFrames
    scores_df = pd.DataFrame(scores_dict)
    ranks_df  = scores_df.apply(scores_to_ranks, axis=0)

    # Aggregation strategies
    avg_rank = average_rank_aggregation(ranks_df)
    borda    = borda_count_aggregation(ranks_df)
    majority = majority_vote_aggregation(ranks_df, top_k=20)

    # Build summary table
    summary = pd.DataFrame({
        "average_rank":   avg_rank,
        "borda_score":    borda,
        "majority_votes": majority,
    })
    summary["final_rank"] = avg_rank.rank().astype(int)
    summary = summary.sort_values("average_rank")

    if verbose:
        print(f"\n[FS] Feature selection complete.")
        print(f"     Methods run   : {list(scores_dict.keys())}")
        print(f"     Features ranked: {len(summary)}")
        print(f"\n     Top 15 features (average rank):")
        print(summary.head(15)[["average_rank","borda_score",
                                  "majority_votes"]].to_string())

    return {
        "scores":       scores_df,
        "ranks":        ranks_df,
        "average_rank": avg_rank,
        "borda":        borda,
        "majority":     majority,
        "summary":      summary,
    }


# ─── Feature subset helpers ───────────────────────────────────────────────────

def get_top_features(fs_results: Dict,
                     n: int = 20,
                     strategy: str = "average_rank") -> List[str]:
    """
    Return the top-N feature names from filter selection results.

    Parameters
    ----------
    fs_results : output of run_filter_selection()
    n          : number of features to return
    strategy   : "average_rank" (A), "borda" (B), or "majority" (C)

    Returns
    -------
    list of feature names, best first
    """
    if strategy == "average_rank":
        ranked = fs_results["average_rank"].sort_values(ascending=True)
    elif strategy == "borda":
        ranked = fs_results["borda"].sort_values(ascending=False)
    elif strategy == "majority":
        ranked = fs_results["majority"].sort_values(ascending=False)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                         f"Choose from: average_rank, borda, majority")
    return ranked.head(n).index.tolist()


def define_feature_subsets(fs_results: Dict,
                             reference_subsets: Dict = None) -> Dict[str, List[str]]:
    """
    Define all 9 experimental feature subsets.

    Parameters
    ----------
    fs_results         : output of run_filter_selection()
    reference_subsets  : optional dict of manually defined subsets
                         e.g. {"zabin_22": [...list of feature names...]}

    Returns
    -------
    dict mapping subset name → list of feature names
    """
    subsets = {
        "top_5":  get_top_features(fs_results, n=5),
        "top_10": get_top_features(fs_results, n=10),
        "top_20": get_top_features(fs_results, n=20),
        "top_30": get_top_features(fs_results, n=30),
        "top_50": get_top_features(fs_results, n=50),
        "all":    fs_results["summary"].index.tolist(),
    }

    # Calendar-only baseline (8 features)
    calendar_feats = ["hour","dayofweek","month","dayofyear",
                      "weekofyear","quarter","is_weekend","is_holiday"]
    subsets["calendar_only"] = [f for f in calendar_feats
                                 if f in fs_results["summary"].index]

    # Lag-only baseline
    lag_feats = ["lag_1h","lag_2h","lag_3h","lag_6h",
                 "lag_24h","lag_48h","lag_168h"]
    subsets["lag_only"] = [f for f in lag_feats
                           if f in fs_results["summary"].index]

    # Add any reference subsets from the literature
    if reference_subsets:
        for name, feats in reference_subsets.items():
            valid = [f for f in feats if f in fs_results["summary"].index]
            subsets[name] = valid

    # Print summary
    print("\n[FS] Feature subsets defined:")
    for name, feats in subsets.items():
        print(f"     {name:<16} : {len(feats):>3} features")

    return subsets


def save_selection_results(fs_results: Dict,
                            output_dir: str = "results") -> None:
    """Save all feature selection results to CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fs_results["scores"].to_csv(f"{output_dir}/fs_scores.csv")
    fs_results["ranks"].to_csv(f"{output_dir}/fs_ranks.csv")
    fs_results["summary"].to_csv(f"{output_dir}/fs_summary.csv")

    print(f"[FS] Results saved to {output_dir}/")
    print(f"     fs_scores.csv  — raw scores per method")
    print(f"     fs_ranks.csv   — ranks per method")
    print(f"     fs_summary.csv — aggregated rankings")
