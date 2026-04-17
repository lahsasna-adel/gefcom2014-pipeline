"""
models/sarima_model.py
----------------------
Seasonal ARIMA forecaster using statsmodels SARIMAX with fixed
manual order specification. Auto-ARIMA removed due to memory
infeasibility on large training sets (>30,000 rows).

Uses a 1-year (8,760 rows) subsample of the most recent training
data — sufficient for estimating SARIMA(1,1,2)(1,1,1)[24] parameters
and consistent with the GEFCom2014 competition benchmark approach.
"""

import time
import warnings
import numpy as np
from typing import Optional, Tuple

warnings.filterwarnings("ignore")


class SARIMAForecaster:
    """
    Wrapper around statsmodels SARIMAX for electricity demand forecasting.

    Fixed order: SARIMA(1,1,2)(1,1,1)[24]
    Training data subsampled to the most recent 8,760 rows (1 year)
    to avoid memory allocation failures on large training sets.
    """

    def __init__(self,
                 seasonal_period: int = 24,
                 manual_order: Tuple = (1, 1, 2),
                 manual_seasonal_order: Tuple = (1, 1, 1, 24),
                 max_train_rows: int = 8760):
        """
        Parameters
        ----------
        seasonal_period        : seasonality length (24 for hourly data)
        manual_order           : (p, d, q) ARIMA order
        manual_seasonal_order  : (P, D, Q, m) seasonal order
        max_train_rows         : maximum training rows (default 8760 = 1 year)
        """
        self.seasonal_period       = seasonal_period
        self.manual_order          = manual_order
        self.manual_seasonal_order = manual_seasonal_order
        self.max_train_rows        = max_train_rows
        self.result_               = None   # stores the fitted SARIMAX result
        self.train_time_           = None

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, train_series: np.ndarray) -> None:
        """
        Fit SARIMA on the most recent max_train_rows of train_series.

        Parameters
        ----------
        train_series : 1-D array of historical demand values
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        t0 = time.time()

        # Subsample to most recent 1 year to avoid memory issues
        if len(train_series) > self.max_train_rows:
            train_series = train_series[-self.max_train_rows:]
            print(f"[SARIMA] Subsampled to last {self.max_train_rows} rows "
                  f"({self.max_train_rows // 8760:.0f} year)")

        print(f"[SARIMA] Fitting SARIMA{self.manual_order}"
              f"{self.manual_seasonal_order} on {len(train_series):,} rows ...",
              end=" ", flush=True)

        model = SARIMAX(
            train_series,
            order=self.manual_order,
            seasonal_order=self.manual_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.result_     = model.fit(disp=False, maxiter=50)
        self.train_time_ = time.time() - t0
        print(f"done in {self.train_time_:.1f}s")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self,
                n_periods: int,
                return_conf_int: bool = False,
                alpha: float = 0.05) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forecast n_periods steps ahead from the end of the training series.

        Parameters
        ----------
        n_periods       : number of steps to forecast
        return_conf_int : also return confidence intervals
        alpha           : significance level for confidence intervals

        Returns
        -------
        predictions : 1-D array of shape (n_periods,)
        conf_int    : array of shape (n_periods, 2) or None
        """
        if self.result_ is None:
            raise RuntimeError("Call fit() before predict().")

        fc    = self.result_.get_forecast(steps=n_periods)
        preds = np.array(fc.predicted_mean)
        preds = np.maximum(preds, 0)   # demand cannot be negative

        if return_conf_int:
            conf = np.array(fc.conf_int(alpha=alpha))
            return preds, conf

        return preds, None

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print fitted model summary."""
        if self.result_ is None:
            print("[SARIMA] Model not fitted yet.")
            return
        print(self.result_.summary())

    def aic(self) -> float:
        """Return fitted model AIC."""
        if self.result_ is None:
            raise RuntimeError("Model not fitted yet.")
        return float(self.result_.aic)