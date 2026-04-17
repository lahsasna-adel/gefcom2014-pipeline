"""
utils/feature_engineering_full.py
----------------------------------
Comprehensive feature engineering for electricity demand forecasting.
Generates ~90 candidate features across 7 families:

  Family 1 — Lag features          (autoregressive)
  Family 2 — Rolling statistics     (local dynamics)
  Family 3 — Calendar features      (temporal structure)
  Family 4 — Fourier seasonality    (cyclic encoding)
  Family 5 — Decomposition features (trend/seasonal/residual)
  Family 6 — Interaction features   (non-additive relationships)
  Family 7 — Meteorological features(temperature-based, GEFCom2014)

Usage
-----
    from utils.feature_engineering_full import build_full_feature_matrix
    fe_df = build_full_feature_matrix(df, target="load")
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ─── Family 1 — Lag features ──────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame,
                     target: str = "load",
                     lags: list = None) -> pd.DataFrame:
    """
    Add autoregressive lag features for the target variable.
    Default lags cover short-term (1-6h), daily (24h), weekly (168h),
    bi-weekly (336h), and yearly (8760h) dependencies.
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]
    for lag in lags:
        df[f"lag_{lag}h"] = df[target].shift(lag)
    return df


# ─── Family 2 — Rolling statistics ───────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame,
                         target: str = "load",
                         windows: list = None) -> pd.DataFrame:
    """
    Add rolling mean, std, max, min, and median over multiple windows.
    All windows are shifted by 1 to avoid data leakage.
    """
    if windows is None:
        windows = [6, 12, 24, 48, 168]
    series = df[target].shift(1)
    for w in windows:
        rolled = series.rolling(window=w, min_periods=max(1, w//2))
        df[f"roll_mean_{w}h"] = rolled.mean()
        df[f"roll_std_{w}h"]  = rolled.std()
        df[f"roll_max_{w}h"]  = rolled.max()
        df[f"roll_min_{w}h"]  = rolled.min()
    # Median only for larger windows (computationally heavier)
    for w in [24, 168]:
        df[f"roll_median_{w}h"] = series.rolling(window=w, min_periods=w//2).median()
    return df


# ─── Family 3 — Calendar features ────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame,
                          country_holidays: str = "US") -> pd.DataFrame:
    """
    Add calendar-based features from the DatetimeIndex.
    Includes hour, day, month, week, weekend flag, holiday flag,
    peak hour flag, and business hour flag.
    """
    idx = df.index

    df["hour"]        = idx.hour
    df["dayofweek"]   = idx.dayofweek       # 0=Mon ... 6=Sun
    df["month"]       = idx.month
    df["dayofyear"]   = idx.dayofyear
    df["weekofyear"]  = idx.isocalendar().week.astype(int)
    df["quarter"]     = idx.quarter
    df["dayofmonth"]  = idx.day
    df["year"]        = idx.year
    df["is_weekend"]  = (idx.dayofweek >= 5).astype(int)

    # Peak hour: morning (7-9h) and evening (17-20h) demand peaks
    df["is_peak_hour"] = idx.hour.isin([7, 8, 9, 17, 18, 19, 20]).astype(int)

    # Business hour: standard working hours
    df["is_business_hour"] = (
        (idx.dayofweek < 5) & (idx.hour >= 8) & (idx.hour < 18)
    ).astype(int)

    # Public holidays
    try:
        import holidays as hol_lib
        years = list(range(idx.year.min(), idx.year.max() + 1))
        hols  = hol_lib.country_holidays(country_holidays, years=years)
        hol_dates = set(pd.Timestamp(d).date() for d in hols.keys())
        df["is_holiday"] = pd.Series(
            [int(ts.date() in hol_dates) for ts in idx], index=idx
        )
    except Exception:
        df["is_holiday"] = 0

    return df


# ─── Family 4 — Fourier seasonality terms ────────────────────────────────────

def add_fourier_features(df: pd.DataFrame,
                         periods: dict = None,
                         orders: dict = None) -> pd.DataFrame:
    """
    Add Fourier sin/cos pairs for multiple seasonal periods.
    Default: daily (24h, order 4), weekly (168h, order 4),
             yearly (8760h, order 3).
    """
    if periods is None:
        periods = {"daily": 24, "weekly": 168, "yearly": 8760}
    if orders is None:
        orders  = {"daily": 4, "weekly": 4, "yearly": 3}

    t = np.arange(len(df))
    for name, period in periods.items():
        order = orders.get(name, 2)
        for k in range(1, order + 1):
            df[f"sin_{name}_{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"cos_{name}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


# ─── Family 5 — Decomposition features ───────────────────────────────────────

def add_decomposition_features(df: pd.DataFrame,
                                target: str = "load",
                                period: int = 24) -> pd.DataFrame:
    """
    Add decomposition and EWM features — all strictly lag-based (no leakage).
    STL on the full series is removed because stl_residual[t] = demand[t]
    − trend[t] − seasonal[t], which directly encodes the target at time t
    and causes perfect R²=1.0 for linear models on large feature subsets.

    Replacements:
      - stl_trend    → ewm_alpha01 (slow EWM approximates trend)
      - stl_seasonal → Fourier terms already capture seasonality (Family 4)
      - stl_residual → dropped entirely (no leak-free substitute exists)
    """
    
    # Exponential weighted mean — captures recent trend (strictly causal)
    df["ewm_alpha01"] = df[target].shift(1).ewm(alpha=0.1, adjust=False).mean()
    df["ewm_alpha03"] = df[target].shift(1).ewm(alpha=0.3, adjust=False).mean()

    # Linear trend (normalised row index — no leakage, purely positional)
    df["linear_trend"] = np.arange(len(df)) / len(df)
    return df





# ─── Family 6 — Interaction features ─────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame,
                              target: str = "load") -> pd.DataFrame:
    """
    Add non-additive interaction features combining calendar
    variables and demand dynamics.
    """
    # Hour × weekend interaction (weekend hourly profile differs)
    df["hour_x_weekend"]   = df["hour"] * df["is_weekend"]

    # Hour × month interaction (seasonal modulation of daily profile)
    df["hour_x_month"]     = df["hour"] * df["month"]

    # Demand relative to weekly rolling mean (relative level)
    if "roll_mean_168h" in df.columns:
        df["demand_rel_week"] = df[target].shift(1) / (
            df["roll_mean_168h"].replace(0, np.nan) + 1e-8
        )

    # Demand change rate (rate of change from lag_1h to lag_2h)
    if "lag_1h" in df.columns and "lag_2h" in df.columns:
        df["demand_change_rate"] = (
            (df["lag_1h"] - df["lag_2h"]) / (df["lag_2h"].abs() + 1e-8)
        )

    # Day × hour combined index (encodes 168 unique hour-day combinations)
    df["dayhour_idx"] = df["dayofweek"] * 24 + df["hour"]

    return df


# ─── Family 7 — Meteorological features ──────────────────────────────────────

def add_meteorological_features(df: pd.DataFrame,
                                  temp_col: str = "temp_avg",
                                  heating_threshold: float = 18.0,
                                  cooling_threshold: float = 18.0) -> pd.DataFrame:
    """
    Add temperature-derived features for electricity demand forecasting.

    Features include:
    - Raw temperature lags (1h, 24h, 168h)
    - Rolling temperature statistics
    - Heating Degree Hours (HDH) and Cooling Degree Hours (CDH)
    - Squared temperature (captures V-shape load-temperature curve)
    - Temperature × hour interaction (temp effect varies by hour)
    - Temperature × weekend interaction
    - Temperature change rate

    Parameters
    ----------
    temp_col            : name of the average temperature column
    heating_threshold   : base temperature for HDH (°C) — typically 18°C
    cooling_threshold   : base temperature for CDH (°C) — typically 18°C
    """
    if temp_col not in df.columns:
        return df

    T = df[temp_col]

    # ── Raw temperature lags ──────────────────────────────────────────────────
    df["temp_lag_1h"]   = T.shift(1)
    df["temp_lag_24h"]  = T.shift(24)
    df["temp_lag_168h"] = T.shift(168)

    # ── Rolling temperature statistics ────────────────────────────────────────
    df["temp_roll_mean_24h"]  = T.shift(1).rolling(24,  min_periods=12).mean()
    df["temp_roll_mean_168h"] = T.shift(1).rolling(168, min_periods=84).mean()
    df["temp_roll_std_24h"]   = T.shift(1).rolling(24,  min_periods=12).std()

    # ── Heating and Cooling Degree Hours ──────────────────────────────────────
    # HDH: demand for heating when temperature is below threshold
    # CDH: demand for cooling when temperature is above threshold
    df["HDH"] = np.maximum(heating_threshold - T, 0)
    df["CDH"] = np.maximum(T - cooling_threshold, 0)

    # ── Nonlinear temperature terms ───────────────────────────────────────────
    # Squared temperature captures the V-shape load-temperature curve
    df["temp_squared"]    = T ** 2
    df["temp_cubed"]      = T ** 3   # For asymmetric curves

    # ── Temperature interaction features ──────────────────────────────────────
    if "hour" in df.columns:
        # Temperature effect on demand varies by hour of day
        df["temp_x_hour"]    = T * df["hour"]

        # Cooling demand concentrated in afternoon hours (12-18h)
        df["CDH_x_afternoon"] = df["CDH"] * (
            df["hour"].isin(range(12, 19)).astype(int)
        )

    if "is_weekend" in df.columns:
        # Temperature sensitivity differs on weekends (less industrial load)
        df["temp_x_weekend"] = T * df["is_weekend"]

    if "month" in df.columns:
        # Temperature × month captures seasonal modulation
        df["temp_x_month"] = T * df["month"]

    # ── Temperature change rate ───────────────────────────────────────────────
    df["temp_change_rate"] = T.diff(1)   # Hour-to-hour temperature change

    return df


# ─── Master function ──────────────────────────────────────────────────────────

def build_full_feature_matrix(df: pd.DataFrame,
                               target: str = "load",
                               temp_col: str = "temp_avg",
                               lags: list = None,
                               rolling_windows: list = None,
                               fourier_periods: dict = None,
                               fourier_orders: dict = None,
                               stl_period: int = 24,
                               country_holidays: str = "US",
                               drop_na: bool = True,
                               verbose: bool = True) -> pd.DataFrame:
    """
    Build the complete feature matrix by applying all 7 feature families.

    Parameters
    ----------
    df               : DataFrame with DatetimeIndex, target column, and
                       optionally temperature columns
    target           : name of the demand column
    temp_col         : name of the average temperature column
    lags             : list of lag periods (hours) — uses defaults if None
    rolling_windows  : list of rolling window sizes — uses defaults if None
    fourier_periods  : dict of {name: period} for Fourier terms
    fourier_orders   : dict of {name: order} for Fourier terms
    stl_period       : period for STL decomposition (24 for hourly)
    country_holidays : ISO country code for public holidays
    drop_na          : drop rows with NaN values after feature generation
    verbose          : print progress messages

    Returns
    -------
    DataFrame with all engineered features plus the target column.
    NaN rows from lagging are dropped if drop_na=True.
    """
    fe = df.copy()
    n_start = len(fe)

    if verbose:
        print(f"[feature_eng] Input shape: {fe.shape}")
        has_temp = temp_col in fe.columns
        print(f"[feature_eng] Temperature column: "
              f"{'found — Family 7 enabled' if has_temp else 'not found — Family 7 skipped'}")

    # Family 1 — Lags
    fe = add_lag_features(fe, target=target, lags=lags)
    if verbose: print(f"[feature_eng] Family 1 (lags)          done")

    # Family 2 — Rolling statistics
    fe = add_rolling_features(fe, target=target, windows=rolling_windows)
    if verbose: print(f"[feature_eng] Family 2 (rolling stats)  done")

    # Family 3 — Calendar
    fe = add_calendar_features(fe, country_holidays=country_holidays)
    if verbose: print(f"[feature_eng] Family 3 (calendar)       done")

    # Family 4 — Fourier
    fe = add_fourier_features(fe, periods=fourier_periods, orders=fourier_orders)
    if verbose: print(f"[feature_eng] Family 4 (Fourier)        done")

    # Family 5 — Decomposition
    if verbose: print(f"[feature_eng] Family 5 (decomposition)  fitting STL ...")
    fe = add_decomposition_features(fe, target=target, period=stl_period)
    if verbose: print(f"[feature_eng] Family 5 (decomposition)  done")

    # Family 6 — Interactions
    fe = add_interaction_features(fe, target=target)
    if verbose: print(f"[feature_eng] Family 6 (interactions)   done")

    # Family 7 — Meteorological (only if temperature column exists)
    if temp_col in fe.columns:
        fe = add_meteorological_features(fe, temp_col=temp_col)
        if verbose: print(f"[feature_eng] Family 7 (meteorological) done")

    # Drop NaN rows created by lagging
    if drop_na:
        fe = fe.dropna()
        n_dropped = n_start - len(fe)
        if verbose:
            print(f"[feature_eng] Dropped {n_dropped} NaN rows from lagging.")

    # Report feature count by family
    feature_cols = [c for c in fe.columns if c != target]
    if verbose:
        print(f"[feature_eng] Final shape: {fe.shape} "
              f"| Features: {len(feature_cols)} | Target: {target}")

    return fe


# ─── Feature catalogue ────────────────────────────────────────────────────────

def get_feature_families(fe_df: pd.DataFrame,
                          target: str = "load") -> dict:
    """
    Return a dict mapping each feature family name to its list of columns.
    Useful for subsetting and reporting.
    """
    cols = [c for c in fe_df.columns if c != target]
    families = {
        "lag":          [c for c in cols if c.startswith("lag_")],
        "rolling":      [c for c in cols if c.startswith("roll_")],
        "calendar":     [c for c in cols if c in [
                            "hour","dayofweek","month","dayofyear","weekofyear",
                            "quarter","dayofmonth","year","is_weekend",
                            "is_peak_hour","is_business_hour","is_holiday"]],
        "fourier":      [c for c in cols if c.startswith("sin_") or
                                            c.startswith("cos_")],
        "decomposition":[c for c in cols if c.startswith("stl_") or
                                            c.startswith("ewm_") or
                                            c == "linear_trend"],
        "interaction":  [c for c in cols if c in [
                            "hour_x_weekend","hour_x_month","demand_rel_week",
                            "demand_change_rate","dayhour_idx"]],
        "meteorological":[c for c in cols if c.startswith("temp_") or
                                             c.startswith("HDH") or
                                             c.startswith("CDH") or
                                             c.startswith("temp_x") or
                                             c.startswith("CDH_x")],
    }
    return families


if __name__ == "__main__":
    # Quick test on synthetic data
    import pandas as pd
    import numpy as np

    idx = pd.date_range("2005-01-01", periods=5000, freq="h")
    df_test = pd.DataFrame({
        "load":     np.random.normal(150, 30, 5000),
        "temp_avg": np.random.normal(15, 10, 5000),
    }, index=idx)

    fe = build_full_feature_matrix(df_test, verbose=True)
    families = get_feature_families(fe)

    print("\nFeatures per family:")
    total = 0
    for fam, cols in families.items():
        print(f"  {fam:<16} : {len(cols):>3} features")
        total += len(cols)
    print(f"  {'TOTAL':<16} : {total:>3} features")
