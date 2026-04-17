"""
utils/statistical_tests.py
---------------------------
Statistical significance tests for forecasting model comparison.

Tests implemented (all using fold-level MAPE values):

  1. Wilcoxon Signed-Rank Test
     Pairwise non-parametric test comparing two configurations.
     Null hypothesis: no difference in median MAPE between two models/subsets.
     Suitable for small samples (4 folds), no normality assumption.

  2. Friedman Test
     Non-parametric equivalent of repeated-measures ANOVA.
     Tests whether at least one configuration differs from the others.
     Applied across all models x subsets simultaneously.

  3. Nemenyi Post-hoc Test (implemented via critical difference)
     Applied after a significant Friedman test to identify which
     specific pairs differ significantly.

References
----------
  - Demsar, J. (2006). Statistical comparisons of classifiers over multiple
    data sets. JMLR, 7, 1-30.
  - Wilcoxon, F. (1945). Individual comparisons by ranking methods.
    Biometrics Bulletin, 1(6), 80-83.
  - Friedman, M. (1940). A comparison of alternative tests of significance
    for the problem of m rankings. Annals of Mathematical Statistics, 11(1), 86-92.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")


# ─── 1. Wilcoxon Signed-Rank Test ─────────────────────────────────────────────

def wilcoxon_test(mape1: list, mape2: list,
                   name1: str = "A", name2: str = "B",
                   alternative: str = "two-sided") -> dict:
    """
    Wilcoxon signed-rank test comparing fold-level MAPE of two configurations.

    Parameters
    ----------
    mape1, mape2 : lists of per-fold MAPE values (must be same length)
    name1, name2 : labels for the two configurations
    alternative  : 'two-sided', 'less', or 'greater'

    Returns
    -------
    dict with: statistic, p_value, significant, effect_size, winner
    """
    x = np.array(mape1)
    y = np.array(mape2)

    if len(x) != len(y):
        raise ValueError("Both lists must have the same number of folds.")

    if len(x) < 3:
        return {"error": "Need at least 3 folds for Wilcoxon test.",
                "name1": name1, "name2": name2}

    # Handle zero differences (all folds equal)
    if np.all(x == y):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "effect_size": 0.0, "winner": "tie",
                "name1": name1, "name2": name2, "n_folds": len(x)}

    try:
        stat, p_val = wilcoxon(x, y, alternative=alternative,
                                zero_method="wilcox", correction=False)
    except ValueError:
        # All differences are zero
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "effect_size": 0.0, "winner": "tie",
                "name1": name1, "name2": name2, "n_folds": len(x)}

    # Effect size: rank-biserial correlation r = 1 - 2W / (n(n+1)/2)
    n = len(x)
    max_w = n * (n + 1) / 2
    effect_size = 1 - (2 * stat) / max_w if max_w > 0 else 0.0

    # Determine winner
    mean_diff = np.mean(x) - np.mean(y)
    if p_val < 0.05:
        winner = name2 if mean_diff > 0 else name1
    else:
        winner = "no significant difference"

    return {
        "name1":       name1,
        "name2":       name2,
        "mean_mape1":  round(float(np.mean(x)), 4),
        "mean_mape2":  round(float(np.mean(y)), 4),
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p_val), 4),
        "significant": bool(p_val < 0.05),
        "effect_size": round(float(effect_size), 4),
        "winner":      winner,
        "n_folds":     len(x),
    }


def wilcoxon_matrix(mape_dict: dict,
                     alpha: float = 0.05) -> pd.DataFrame:
    """
    Compute pairwise Wilcoxon tests for all pairs in mape_dict.

    Parameters
    ----------
    mape_dict : {config_name: [mape_fold1, mape_fold2, ...]}
    alpha     : significance threshold

    Returns
    -------
    DataFrame with p-values (upper triangle) and significance markers
    """
    names = list(mape_dict.keys())
    n     = len(names)
    p_matrix  = pd.DataFrame(np.ones((n, n)), index=names, columns=names)
    sig_matrix = pd.DataFrame("—", index=names, columns=names)

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i >= j:
                continue
            res = wilcoxon_test(mape_dict[n1], mape_dict[n2], n1, n2)
            p   = res.get("p_value", 1.0)
            p_matrix.loc[n1, n2]  = p
            p_matrix.loc[n2, n1]  = p

            # Significance markers
            if p < 0.001:   sig = "***"
            elif p < 0.01:  sig = "**"
            elif p < 0.05:  sig = "*"
            else:            sig = "ns"
            sig_matrix.loc[n1, n2] = sig
            sig_matrix.loc[n2, n1] = sig

    return p_matrix, sig_matrix


# ─── 2. Friedman Test ─────────────────────────────────────────────────────────

def friedman_test(mape_matrix: pd.DataFrame) -> dict:
    """
    Friedman test across multiple configurations.

    Parameters
    ----------
    mape_matrix : DataFrame where rows = folds, columns = configurations.
                  Values are MAPE for each fold x configuration.

    Returns
    -------
    dict with: statistic, p_value, significant, df
    """
    # Each column is one configuration, rows are folds (blocks)
    groups = [mape_matrix[col].values for col in mape_matrix.columns]

    if len(groups) < 3:
        return {"error": "Friedman test requires at least 3 groups."}

    stat, p_val = friedmanchisquare(*groups)

    return {
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p_val), 6),
        "significant": bool(p_val < 0.05),
        "n_groups":    len(groups),
        "n_folds":     len(mape_matrix),
        "df":          len(groups) - 1,
    }


# ─── 3. Nemenyi Post-hoc Test ─────────────────────────────────────────────────

def nemenyi_test(mape_matrix: pd.DataFrame,
                  alpha: float = 0.05) -> tuple:
    """
    Nemenyi post-hoc test for pairwise comparisons after Friedman.

    Uses the critical difference (CD) approach from Demsar (2006).

    Parameters
    ----------
    mape_matrix : DataFrame where rows = folds, columns = configurations
    alpha       : significance threshold (0.05 or 0.10)

    Returns
    -------
    (p_matrix, sig_matrix, avg_ranks, CD)
    - p_matrix  : pairwise p-values (approximate)
    - sig_matrix: significance markers
    - avg_ranks : average rank per configuration
    - CD        : critical difference value
    """
    n_folds  = len(mape_matrix)
    n_groups = len(mape_matrix.columns)
    names    = list(mape_matrix.columns)

    # Compute average ranks across folds
    # For each fold, rank configurations (lower MAPE = lower rank = better)
    ranks_per_fold = mape_matrix.rank(axis=1)
    avg_ranks      = ranks_per_fold.mean(axis=0)

    # Critical difference (Demsar 2006, Table 5)
    # q_alpha values for alpha=0.05 (two-tailed):
    q_alpha_table = {
        2:  1.960, 3:  2.343, 4:  2.569, 5:  2.728,
        6:  2.850, 7:  2.949, 8:  3.031, 9:  3.102, 10: 3.164
    }
    q_alpha = q_alpha_table.get(n_groups, 3.164)
    CD = q_alpha * np.sqrt(n_groups * (n_groups + 1) / (6 * n_folds))

    # Build pairwise significance matrix based on rank differences
    p_matrix   = pd.DataFrame(1.0,  index=names, columns=names)
    sig_matrix = pd.DataFrame("—",  index=names, columns=names)

    for n1, n2 in combinations(names, 2):
        rank_diff = abs(avg_ranks[n1] - avg_ranks[n2])
        # Approximate p-value using normal approximation
        se = np.sqrt(n_groups * (n_groups + 1) / (6 * n_folds))
        z  = rank_diff / se if se > 0 else 0
        p  = float(2 * (1 - stats.norm.cdf(abs(z))))
        p  = max(min(p, 1.0), 0.0)

        p_matrix.loc[n1, n2]  = round(p, 4)
        p_matrix.loc[n2, n1]  = round(p, 4)

        if p < 0.001:   sig = "***"
        elif p < 0.01:  sig = "**"
        elif p < 0.05:  sig = "*"
        else:            sig = "ns"
        sig_matrix.loc[n1, n2] = sig
        sig_matrix.loc[n2, n1] = sig

    return p_matrix, sig_matrix, avg_ranks.round(3), round(CD, 3)


# ─── 4. Effect size — Cohen's d ───────────────────────────────────────────────

def cohens_d(mape1: list, mape2: list) -> float:
    """
    Cohen's d effect size for the difference between two MAPE distributions.
    |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, >0.8 = large
    """
    x, y = np.array(mape1), np.array(mape2)
    pooled_std = np.sqrt((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return float(abs(np.mean(x) - np.mean(y)) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:   return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else:          return "large"


# ─── 5. Summary helpers ───────────────────────────────────────────────────────

def significance_symbol(p: float) -> str:
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else:           return "ns"


def print_wilcoxon_results(results: list) -> None:
    """Pretty-print a list of wilcoxon_test() result dicts."""
    print(f"\n{'='*70}")
    print("  WILCOXON SIGNED-RANK TEST RESULTS")
    print(f"  (* p<0.05,  ** p<0.01,  *** p<0.001,  ns = not significant)")
    print(f"{'='*70}")
    print(f"  {'Comparison':<40} {'p-value':>8} {'Sig':>5} {'Effect':>8} {'Winner'}")
    print(f"  {'-'*65}")
    for r in results:
        if "error" in r:
            continue
        comparison = f"{r['name1']} vs {r['name2']}"
        sig = significance_symbol(r['p_value'])
        effect = f"{r['effect_size']:.3f}"
        print(f"  {comparison:<40} {r['p_value']:>8.4f} {sig:>5} {effect:>8}   {r['winner']}")
    print(f"{'='*70}\n")


def print_friedman_results(result: dict,
                            avg_ranks: pd.Series = None) -> None:
    """Pretty-print Friedman test result."""
    print(f"\n{'='*60}")
    print("  FRIEDMAN TEST")
    print(f"{'='*60}")
    print(f"  Chi-squared : {result['statistic']:.4f}")
    print(f"  df          : {result['df']}")
    print(f"  p-value     : {result['p_value']:.6f}")
    print(f"  Significant : {'YES — at least one group differs' if result['significant'] else 'NO'}")
    if avg_ranks is not None:
        print(f"\n  Average ranks (lower = better):")
        for name, rank in avg_ranks.sort_values().items():
            print(f"    {name:<20} : {rank:.3f}")
    print(f"{'='*60}\n")
