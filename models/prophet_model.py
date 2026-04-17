"""
models/prophet_model.py
-----------------------
Facebook / Meta Prophet forecaster for electricity demand.

Features
--------
- Automatic daily, weekly, yearly Fourier seasonality
- Holiday effects (uses Python 'holidays' library)
- Uncertainty estimation via Monte Carlo sampling
- Regressors support (temperature, humidity, …)

Dependencies:
    pip install prophet holidays
"""

import time
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional

warnings.filterwarnings("ignore")


class ProphetForecaster:
    """
    Wrapper around Facebook Prophet for electricity demand forecasting.

    Prophet expects a DataFrame with columns ['ds', 'y'] (datetime, value).
    This class handles the conversion from your standard DataFrame.
    """

    def __init__(self,
                 country_holidays: Optional[str] = "DZ",   # Algeria
                 daily_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 yearly_seasonality: bool = True,
                 seasonality_mode: str = "multiplicative",  # or 'additive'
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 uncertainty_samples: int = 1000,
                 interval_width: float = 0.95,
                 extra_regressors: Optional[List[str]] = None):
        """
        Parameters
        ----------
        country_holidays        : ISO country code for public holidays
        daily/weekly/yearly_*   : toggle built-in seasonality components
        seasonality_mode        : 'additive' or 'multiplicative'
        changepoint_prior_scale : flexibility of trend changepoints (higher = more flexible)
        seasonality_prior_scale : strength of seasonality regularisation
        holidays_prior_scale    : strength of holiday effect regularisation
        uncertainty_samples     : Monte Carlo draws for prediction intervals
        interval_width          : confidence interval width (0.95 = 95%)
        extra_regressors        : list of additional regressor column names
        """
        self.country_holidays           = country_holidays
        self.daily_seasonality          = daily_seasonality
        self.weekly_seasonality         = weekly_seasonality
        self.yearly_seasonality         = yearly_seasonality
        self.seasonality_mode           = seasonality_mode
        self.changepoint_prior_scale    = changepoint_prior_scale
        self.seasonality_prior_scale    = seasonality_prior_scale
        self.holidays_prior_scale       = holidays_prior_scale
        self.uncertainty_samples        = uncertainty_samples
        self.interval_width             = interval_width
        self.extra_regressors           = extra_regressors or []
        self.model_                     = None
        self.train_time_                = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
        """Convert a pandas Series with DatetimeIndex to Prophet format."""
        return pd.DataFrame({"ds": series.index, "y": series.values})

    def _build_holidays(self, years: List[int]) -> Optional[pd.DataFrame]:
        """Build a Prophet-compatible holidays DataFrame."""
        if not self.country_holidays:
            return None
        try:
            import holidays as hol_lib
            country_hols = hol_lib.country_holidays(self.country_holidays, years=years)
            hol_df = pd.DataFrame([
                {"ds": pd.Timestamp(date), "holiday": name}
                for date, name in country_hols.items()
            ])
            return hol_df if not hol_df.empty else None
        except Exception:
            print("[Prophet] Could not load holidays — continuing without them.")
            return None

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, y_train: pd.Series,
            regressors_train: Optional[pd.DataFrame] = None) -> "ProphetForecaster":
        """
        Fit Prophet on training series.

        Parameters
        ----------
        y_train           : demand Series with DatetimeIndex
        regressors_train  : DataFrame with extra regressor columns (same index)
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Install Prophet:  pip install prophet")

        print("[Prophet] Fitting …", end=" ", flush=True)
        t0 = time.time()

        # Build holidays
        years = list(range(y_train.index.year.min(), y_train.index.year.max() + 2))
        hol_df = self._build_holidays(years)

        self.model_ = Prophet(
            holidays=hol_df,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            uncertainty_samples=self.uncertainty_samples,
            interval_width=self.interval_width,
        )

        # Add extra regressors
        for reg in self.extra_regressors:
            self.model_.add_regressor(reg)

        # Build training DataFrame
        train_df = self._to_prophet_df(y_train)
        if regressors_train is not None:
            for col in self.extra_regressors:
                if col in regressors_train.columns:
                    train_df[col] = regressors_train[col].values

        self.model_.fit(train_df)
        self.train_time_ = time.time() - t0
        print(f"done in {self.train_time_:.1f}s")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, future_index: pd.DatetimeIndex,
                regressors_future: Optional[pd.DataFrame] = None
                ) -> pd.DataFrame:
        """
        Generate forecasts for a future DatetimeIndex.

        Parameters
        ----------
        future_index       : timestamps to forecast
        regressors_future  : extra regressors for the forecast period

        Returns
        -------
        DataFrame with columns: ds, yhat, yhat_lower, yhat_upper,
                                trend, weekly, yearly, daily (components)
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        future_df = pd.DataFrame({"ds": future_index})
        if regressors_future is not None:
            for col in self.extra_regressors:
                if col in regressors_future.columns:
                    future_df[col] = regressors_future[col].values

        forecast = self.model_.predict(future_df)
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        return forecast

    def predict_array(self, future_index: pd.DatetimeIndex,
                      regressors_future: Optional[pd.DataFrame] = None
                      ) -> np.ndarray:
        """Convenience: return just the point forecast as a numpy array."""
        return self.predict(future_index, regressors_future)["yhat"].values

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def plot_components(self, forecast: pd.DataFrame):
        """Plot trend, seasonality, and holiday components."""
        self.model_.plot_components(forecast)

    def cross_validate(self, y_train: pd.Series,
                       initial: str = "180 days",
                       period: str = "30 days",
                       horizon: str = "24 hours") -> pd.DataFrame:
        """
        Run Prophet's built-in time-series cross-validation.

        Returns
        -------
        DataFrame with columns: ds, y, yhat, yhat_lower, yhat_upper, cutoff
        """
        from prophet.diagnostics import cross_validation
        train_df = self._to_prophet_df(y_train)
        cv_df = cross_validation(
            self.model_, initial=initial, period=period, horizon=horizon,
            parallel="processes"
        )
        return cv_df
