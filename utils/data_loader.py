"""
utils/data_loader.py
--------------------
CSV loading, validation, feature engineering, and train/test splitting
for electricity demand time series.

Expected CSV columns:
    Required : timestamp  (parseable datetime), demand (MW)
    Optional : temperature (°C), humidity (%), holiday (0/1),
               wind_speed, cloud_cover, ...
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")


# ─── Column auto-detection ────────────────────────────────────────────────────

TIMESTAMP_ALIASES = ["timestamp", "datetime", "date", "time", "ds", "index"]
DEMAND_ALIASES    = ["demand", "load", "power", "energy", "consumption",
                     "mw", "kwh", "value", "y"]


def _find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    """Return the first column whose name (lower-cased) contains an alias."""
    cols_lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        for lower, original in cols_lower.items():
            if alias in lower:
                return original
    return None


# ─── Loader ───────────────────────────────────────────────────────────────────

def load_csv(path: str,
             timestamp_col: Optional[str] = None,
             demand_col: Optional[str] = None,
             freq: str = "h") -> pd.DataFrame:
    """
    Load electricity demand data from a CSV file.

    Parameters
    ----------
    path          : path to the CSV file
    timestamp_col : column name for timestamps (auto-detected if None)
    demand_col    : column name for demand values (auto-detected if None)
    freq          : resampling frequency ('h' = hourly, 'D' = daily, etc.)

    Returns
    -------
    pd.DataFrame with DatetimeIndex, 'demand' column, and any extra features.
    """
    df = pd.read_csv(path)
    print(f"[loader] Loaded {len(df):,} rows, {len(df.columns)} columns.")

    # Detect columns
    ts_col  = timestamp_col or _find_column(df, TIMESTAMP_ALIASES)
    dem_col = demand_col    or _find_column(df, DEMAND_ALIASES)

    if dem_col is None:
        raise ValueError(
            f"Could not detect a demand column. Found: {list(df.columns)}. "
            "Please pass demand_col='your_column_name'."
        )

    # Parse timestamps
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], infer_datetime_format=True)
        df = df.set_index(ts_col).sort_index()
    else:
        df.index = pd.date_range(start="2020-01-01", periods=len(df), freq=freq)
        print("[loader] No timestamp column found — generating synthetic index.")

    # Rename demand
    df = df.rename(columns={dem_col: "demand"})
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

    # Resample to target frequency
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols].resample(freq).mean()

    # Handle missing values
    n_missing = df["demand"].isna().sum()
    if n_missing:
        print(f"[loader] Interpolating {n_missing} missing demand values.")
        df["demand"] = df["demand"].interpolate(method="time").ffill().bfill()

    print(f"[loader] Final shape: {df.shape} | Range: "
          f"{df.index[0]} → {df.index[-1]} | "
          f"Demand: {df['demand'].min():.1f}–{df['demand'].max():.1f} MW")
    return df


# ─── Feature Engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame,
                      lags: List[int] = [1, 2, 3, 24, 48, 168],
                      rolling_windows: List[int] = [6, 24, 168],
                      add_fourier: bool = True,
                      fourier_order: int = 4) -> pd.DataFrame:
    """
    Add lag features, rolling statistics, calendar variables, and Fourier terms.

    Parameters
    ----------
    df              : DataFrame with DatetimeIndex and 'demand' column
    lags            : list of lag periods (in rows)
    rolling_windows : window sizes for rolling mean/std
    add_fourier     : add sine/cosine seasonality terms
    fourier_order   : number of Fourier pairs per period

    Returns
    -------
    DataFrame with all original columns plus engineered features.
    (Rows with NaN from lagging are dropped.)
    """
    fe = df.copy()

    # ── Lag features
    for lag in lags:
        fe[f"lag_{lag}h"] = fe["demand"].shift(lag)

    # ── Rolling statistics
    for w in rolling_windows:
        fe[f"rolling_mean_{w}h"] = fe["demand"].shift(1).rolling(w).mean()
        fe[f"rolling_std_{w}h"]  = fe["demand"].shift(1).rolling(w).std()
        fe[f"rolling_max_{w}h"]  = fe["demand"].shift(1).rolling(w).max()

    # ── Calendar features
    fe["hour"]       = fe.index.hour
    fe["dayofweek"]  = fe.index.dayofweek          # 0=Mon … 6=Sun
    fe["month"]      = fe.index.month
    fe["dayofyear"]  = fe.index.dayofyear
    fe["weekofyear"] = fe.index.isocalendar().week.astype(int)
    fe["is_weekend"] = (fe.index.dayofweek >= 5).astype(int)
    fe["quarter"]    = fe.index.quarter

    # ── Fourier seasonality (daily: period=24, weekly: period=168)
    if add_fourier:
        for period, name in [(24, "daily"), (168, "weekly"), (8760, "yearly")]:
            t = np.arange(len(fe))
            for k in range(1, fourier_order + 1):
                fe[f"sin_{name}_{k}"] = np.sin(2 * np.pi * k * t / period)
                fe[f"cos_{name}_{k}"] = np.cos(2 * np.pi * k * t / period)

    # Drop NaN rows created by lagging
    fe = fe.dropna()
    print(f"[features] Shape after engineering: {fe.shape} "
          f"({len(df)-len(fe)} rows dropped due to lagging).")
    return fe


# ─── Train / Test Split ───────────────────────────────────────────────────────

def time_series_split(df: pd.DataFrame,
                      target: str = "demand",
                      test_size: float = 0.2
                      ) -> Tuple[pd.DataFrame, pd.Series,
                                 pd.DataFrame, pd.Series]:
    """
    Chronological train/test split (no shuffling).

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    feature_cols = [c for c in df.columns if c != target]
    X_train, y_train = train[feature_cols], train[target]
    X_test,  y_test  = test[feature_cols],  test[target]

    print(f"[split] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    return X_train, y_train, X_test, y_test


def prepare_sequences(series: np.ndarray,
                      lookback: int = 48,
                      horizon: int = 24
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sliding-window arrays for LSTM / sequence models.

    Parameters
    ----------
    series   : 1-D array of demand values
    lookback : input window length
    horizon  : forecast horizon (steps ahead)

    Returns
    -------
    X : shape (n_samples, lookback, 1)
    y : shape (n_samples, horizon)
    """
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i: i + lookback])
        y.append(series[i + lookback: i + lookback + horizon])
    X = np.array(X)[..., np.newaxis]   # add feature dim
    y = np.array(y)
    return X, y
