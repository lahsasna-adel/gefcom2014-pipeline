# Electricity Forecasting Studio — Python Backend

Complete implementation of 7 time-series forecasting algorithms
for electricity demand prediction, with CSV input, feature engineering,
evaluation metrics, and diagnostic visualizations.

---

## Project structure

```
electricity_forecasting/
│
├── main.py                     ← End-to-end pipeline (CLI entry point)
├── notebook.ipynb              ← Interactive Jupyter notebook
├── requirements.txt            ← All Python dependencies
│
├── models/
│   ├── sarima_model.py         ← SARIMA via pmdarima (auto order selection)
│   ├── prophet_model.py        ← Facebook Prophet (holidays, regressors)
│   ├── tree_models.py          ← XGBoost, LightGBM, Random Forest + Optuna tuning
│   ├── lstm_model.py           ← Stacked bidirectional LSTM (TensorFlow/Keras)
│   └── deep_models.py          ← TFT (Temporal Fusion Transformer) + N-BEATS
│
├── utils/
│   ├── data_loader.py          ← CSV loader, feature engineering, train/test split
│   ├── metrics.py              ← MAE, RMSE, MAPE, sMAPE, R², MASE, leaderboard
│   └── visualization.py        ← Forecast plots, residuals, feature importance
│
└── results/                    ← Auto-created; leaderboard CSV + PNG plots
```

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 2. Install core dependencies
pip install -r requirements.txt

# 3. (Optional) GPU support for LSTM / TFT / N-BEATS
pip install tensorflow[and-cuda]   # TensorFlow GPU
pip install torch --index-url https://download.pytorch.org/whl/cu118  # PyTorch GPU
```

---

## CSV format

Your CSV must contain at least two columns:

| Column | Required | Description |
|--------|----------|-------------|
| `timestamp` | Yes | Any parseable datetime string |
| `demand` | Yes | Electricity consumption in MW (or kWh, load, power, energy) |
| `temperature` | No | Ambient temperature — used as Prophet regressor |
| `humidity` | No | Relative humidity |
| `holiday` | No | 1 = public holiday, 0 = regular day |

Column names are auto-detected (case-insensitive, partial match).

Example:
```csv
timestamp,demand,temperature,holiday
2024-01-01 00:00,523.4,12.1,1
2024-01-01 01:00,498.7,11.8,1
2024-01-01 02:00,476.2,11.5,1
...
```

---

## Usage

### Command line

```bash
# Run all models on demo data (auto-generated 60 days):
python main.py --demo --save-plots

# Run with your CSV:
python main.py --csv data/electricity.csv --save-plots

# Choose specific models:
python main.py --csv data/electricity.csv --models arima prophet xgb lstm

# Tune XGBoost with Optuna (100 trials):
python main.py --csv data/electricity.csv --tune xgb --tune-trials 100

# Change test split and save results:
python main.py --csv data/electricity.csv --test-size 0.15 --save-plots
```

### Jupyter notebook

```bash
jupyter notebook notebook.ipynb
```

Set `CSV_PATH` and `MODELS` in the CONFIG cell, then run all cells.

### Individual models (Python API)

```python
from utils.data_loader import load_csv, engineer_features, time_series_split

df = load_csv("data/electricity.csv")
fe = engineer_features(df)
X_train, y_train, X_test, y_test = time_series_split(fe)

# XGBoost
from models.tree_models import XGBoostForecaster
m = XGBoostForecaster()
m.fit(X_train, y_train)
preds = m.predict(X_test)

# Evaluate
from utils.metrics import evaluate, print_leaderboard
result = evaluate(y_test.values, preds, "XGBoost")
print(result)
# {'model': 'XGBoost', 'MAE': 12.4, 'RMSE': 17.8, 'MAPE': 2.9, ...}
```

---

## Algorithm guide

| Model | File | Best for | Speed | Interpretability |
|-------|------|----------|-------|-----------------|
| **SARIMA** | `sarima_model.py` | Short series, strong seasonality | Fast | High |
| **Prophet** | `prophet_model.py` | Business series with holidays | Fast | High |
| **XGBoost** | `tree_models.py` | Large datasets, tabular features | Medium | Medium |
| **LightGBM** | `tree_models.py` | Same as XGBoost, 3–10× faster | Fast | Medium |
| **Random Forest** | `tree_models.py` | Robustness to outliers | Medium | Medium |
| **LSTM** | `lstm_model.py` | Long-range dependencies, sequences | Slow | Low |
| **TFT** | `deep_models.py` | Multi-horizon, interpretable DL | Very slow | Medium |
| **N-BEATS** | `deep_models.py` | No feature engineering, pure series | Slow | Low |

---

## Evaluation metrics

| Metric | Formula | Unit | Notes |
|--------|---------|------|-------|
| **MAE** | mean(|actual - pred|) | MW | Easy to interpret |
| **RMSE** | sqrt(mean((actual - pred)²)) | MW | Penalises large errors |
| **MAPE** | mean(|actual - pred| / actual) × 100 | % | Scale-independent |
| **sMAPE** | symmetric version of MAPE | % | Bounded 0–200% |
| **R²** | 1 - SS_res / SS_tot | — | 1.0 = perfect |
| **MASE** | MAE / naive seasonal MAE | — | < 1 beats naive forecast |

---

## Feature engineering (auto-applied)

- **Lag features**: t-1h, t-2h, t-3h, t-24h, t-48h, t-168h
- **Rolling statistics**: mean, std, max over 6h, 24h, 7-day windows
- **Calendar**: hour, day-of-week, month, day-of-year, week-of-year, quarter
- **Fourier seasonality**: sin/cos pairs for daily (24h), weekly (168h), yearly (8760h) cycles
- **Binary flags**: is_weekend, holiday (if column present)
