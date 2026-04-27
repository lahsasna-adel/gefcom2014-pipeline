"""
utils/family8_sdwh.py
---------------------
Family 8 — Same-Day-of-Week-and-Hour (SDWH) Aggregation Features.

Motivation (Smyl & Hua, 2019)
------------------------------
In GEFCom2017, a mandatory 1.5–2 month gap between the last training
observation and the first forecast date made recent lag features (lag-1,
lag-24, lag-168) unavailable. Team Orbuculum solved this by computing
averages of historical load values that share the same:
  - day of the week  (e.g. Tuesday)
  - hour of the day  (e.g. 17:00)

...over 8 rolling 4-week sub-windows, each anchored at a fixed distance
from the forecast origin.

Adaptation for GEFCom2014
--------------------------
In our dataset there is no temporal gap, so recent lags are already
available in Family 1. We include SDWH features as a complementary
long-horizon signal that captures:

  (a) Stable weekly periodicity patterns beyond the 168h lag window
  (b) Seasonal drift — how the same weekday-hour slot has evolved
      over the past 4–32 weeks
  (c) A signal that is more robust to single-day anomalies than lag-168
      because it averages over an entire 4-week window

Feature naming convention
--------------------------
  sdwh_w{k}_mean  — mean demand for same DoW×Hour, k-th 4-week window back
  sdwh_w{k}_std   — standard deviation of that window
  sdwh_w{k}_q10   — 10th percentile
  sdwh_w{k}_q90   — 90th percentile

Window definitions (k = 1 … N_WINDOWS, anchored from current row)
-------------------------------------------------------------------
  Window k spans rows with the SAME (dayofweek, hour) as the current row,
  falling in the time range:
      [k*4 weeks ago − 2 weeks,  k*4 weeks ago + 2 weeks]
  i.e. a ±2 week band centred at k×28 days before the current timestep.

  With N_WINDOWS = 4 (default):
    Window 1: same DoW×Hour from  2–6  weeks ago
    Window 2: same DoW×Hour from  6–10 weeks ago
    Window 3: same DoW×Hour from 10–14 weeks ago
    Window 4: same DoW×Hour from 14–18 weeks ago

Implementation
--------------
For each row t with (dow_t, h_t), we look back in the raw series for all
rows t' < t such that:
  - dow(t') == dow_t
  - hour(t') == h_t
  - centre_k - 2 weeks ≤ t - t' ≤ centre_k + 2 weeks

where centre_k = k * 28 * 24 hours.

This is equivalent to Smyl & Hua's "8 × 4-week sub-windows" strategy,
scaled to 4 windows with ±2 week bands for the GEFCom2014 scale.

Complexity note
---------------
Naïve O(N²) computation is too slow for 60k rows. We use a vectorised
groupby approach: for each (dow, hour) pair, extract all historical
values, then for each row compute window statistics using pre-sorted
arrays and searchsorted — O(N log N) overall.

Reference
---------
Smyl, S., & Hua, N. G. (2019). Machine learning methods for GEFCom2017
probabilistic load forecasting. International Journal of Forecasting,
35(4), 1424–1431. Section 3.1.1.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Configuration ──────────────────────────────────────────────────────────
N_WINDOWS    = 4       # number of 4-week sub-windows
WINDOW_WEEKS = 4       # each window spans ±2 weeks around centre_k
BAND_HOURS   = 14 * 24  # ±2 weeks = ±336 hours half-band
STATS        = ["mean", "std", "q10", "q90"]


def _window_stats(vals: np.ndarray) -> dict:
    """Compute summary statistics for a window of demand values."""
    if len(vals) == 0:
        return {"mean": np.nan, "std": np.nan, "q10": np.nan, "q90": np.nan}
    return {
        "mean": float(np.mean(vals)),
        "std":  float(np.std(vals)),
        "q10":  float(np.percentile(vals, 10)),
        "q90":  float(np.percentile(vals, 90)),
    }


def add_sdwh_features(df: pd.DataFrame,
                       demand_col: str = "demand",
                       n_windows: int = N_WINDOWS,
                       band_hours: int = BAND_HOURS,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Add Family 8 — Same-Day-of-Week-and-Hour aggregation features.

    Parameters
    ----------
    df          : DataFrame with a DatetimeIndex or a datetime column,
                  must contain `demand_col`.
    demand_col  : name of the demand column.
    n_windows   : number of 4-week lookback windows (default 4).
    band_hours  : half-band in hours around each window centre (default 336 = ±2 weeks).
    verbose     : print progress.

    Returns
    -------
    df with new columns  sdwh_w{k}_{stat}  for k in 1..n_windows, stat in STATS.
    Rows where historical data is insufficient are filled with NaN (not dropped).
    """
    if verbose:
        print("[feature_eng] Family 8 (SDWH aggregation)  computing ...")

    # ── Ensure integer position index for fast lookup ────────────────────────
    df = df.copy()
    demand = df[demand_col].values.astype(np.float64)
    N = len(df)

    # ── Extract time components ──────────────────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        dow  = df.index.dayofweek.values   # 0=Mon … 6=Sun
        hour = df.index.hour.values
    else:
        # Try to find a datetime column
        dt_col = None
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                dt_col = c
                break
        if dt_col is None:
            raise ValueError("DataFrame must have a DatetimeIndex or a datetime column.")
        dow  = df[dt_col].dt.dayofweek.values
        hour = df[dt_col].dt.hour.values

    # ── Window centres in hours ──────────────────────────────────────────────
    # centre_k = k * 28 * 24  for k = 1 … n_windows
    centres = [(k * 28 * 24) for k in range(1, n_windows + 1)]

    # ── Feature arrays (pre-filled with NaN) ────────────────────────────────
    feat_arrays = {}
    col_names   = []
    for k in range(1, n_windows + 1):
        for stat in STATS:
            col = f"sdwh_w{k}_{stat}"
            feat_arrays[col] = np.full(N, np.nan, dtype=np.float64)
            col_names.append(col)

    # ── Group rows by (dayofweek, hour) ─────────────────────────────────────
    # For each group, build a sorted array of (position_index, demand_value)
    # so we can use searchsorted to find values within each time window.
    group_positions = {}   # (dow, hour) → sorted array of integer positions
    group_demands   = {}   # (dow, hour) → demand values at those positions

    for dw in range(7):
        for hr in range(24):
            mask = (dow == dw) & (hour == hr)
            pos  = np.where(mask)[0]          # positions in df (sorted by construction)
            if len(pos) == 0:
                continue
            group_positions[(dw, hr)] = pos
            group_demands[(dw, hr)]   = demand[pos]

    # ── Main computation loop ────────────────────────────────────────────────
    # For each row t, look up its (dow, hour) group and extract window values.
    # Each hourly row in GEFCom2014 corresponds to exactly 1 hour gap,
    # so position difference = hour difference.

    for t in range(N):
        key = (int(dow[t]), int(hour[t]))
        if key not in group_positions:
            continue

        pos_arr  = group_positions[key]    # sorted positions of same DoW×Hour rows
        dem_arr  = group_demands[key]

        for k_idx, (k, centre) in enumerate(zip(range(1, n_windows + 1), centres)):
            # We want positions t' < t such that:
            #   centre - band_hours ≤ (t - t') ≤ centre + band_hours
            # ⟺  t - centre - band_hours ≤ t' ≤ t - centre + band_hours
            # ⟺  t' ∈ [lo, hi)

            lo = t - centre - band_hours
            hi = t - centre + band_hours

            if hi < 0:
                continue

            lo = max(lo, 0)

            # searchsorted: find indices in pos_arr within [lo, hi)
            idx_lo = int(np.searchsorted(pos_arr, lo, side="left"))
            idx_hi = int(np.searchsorted(pos_arr, hi, side="right"))

            # Exclude t itself
            window_pos  = pos_arr[idx_lo:idx_hi]
            window_mask = window_pos < t
            window_dem  = dem_arr[idx_lo:idx_hi][window_mask]

            if len(window_dem) == 0:
                continue

            stats = _window_stats(window_dem)
            base  = f"sdwh_w{k}"
            feat_arrays[f"{base}_mean"][t] = stats["mean"]
            feat_arrays[f"{base}_std"][t]  = stats["std"]
            feat_arrays[f"{base}_q10"][t]  = stats["q10"]
            feat_arrays[f"{base}_q90"][t]  = stats["q90"]

    # ── Append to df ─────────────────────────────────────────────────────────
    for col in col_names:
        df[col] = feat_arrays[col]

    n_new = len(col_names)
    n_nan = int(pd.DataFrame(feat_arrays).isna().all(axis=1).sum())

    if verbose:
        print(f"[feature_eng] Family 8 (SDWH aggregation)  done "
              f"({n_new} features, {n_nan} rows with all-NaN windows dropped downstream)")

    return df


def get_sdwh_feature_names(n_windows: int = N_WINDOWS) -> list:
    """Return list of all SDWH feature column names."""
    names = []
    for k in range(1, n_windows + 1):
        for stat in STATS:
            names.append(f"sdwh_w{k}_{stat}")
    return names
