<div align="center">

# 🛒 Supermarket Demand Forecasting

**Turning five years of daily sales into practical, data-driven demand forecasts.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Analysis-Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Forecasting](https://img.shields.io/badge/Focus-Time%20Series%20%7C%20Machine%20Learning-16A085)](#-project-highlights)

</div>

---

## Project overview

This project analyzes historical supermarket sales and forecasts **chain-wide daily demand**. It aggregates store and item transactions into a single daily sales series, explores its trend and weekly seasonality, and compares classical time-series methods with a feature-based machine-learning model.

The repository contains two complementary Jupyter notebooks:

- **Time-series analysis:** stationarity and autocorrelation analysis, ARIMA/SARIMAX model selection, residual diagnostics, and a seven-day forecast with 95% confidence intervals.
- **Machine-learning analysis:** lag, rolling-window, and calendar features; Random Forest regression; baseline comparison; feature importance; and a recursive seven-day forecast.

Both notebooks use a chronological **90-day holdout** rather than a random split. The `test.csv` competition data has no sales target and is not used to calculate the reported validation scores.

## Project highlights

- **913,000** store–item–date observations across **10 stores** and **50 items**.
- Daily history from **January 1, 2013 through December 31, 2017** (**1,826 days** after aggregation).
- A shared holdout period: **October 3 through December 31, 2017**.
- Two forecasting approaches evaluated against naive baselines.
- Exported predictions, comparisons, residuals, and forecasts in `output/`.

## Validation results

The table below summarizes the best reported result for each model family on the 90-day holdout. Lower scores indicate lower error.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| SARIMAX (weekly seasonality) | 619.38 | **1,222.85** | 2.57% |
| Random Forest | 671.54 | 1,387.48 | 2.71% |
| Seasonal naive (7 days) | 1,154.64 | 2,674.65 | 4.75% |
| Naive (previous day) | 3,108.08 | 4,500.75 | 12.66% |

The selected SARIMAX specification is `SARIMAX(1, 1, 1) × (1, 1, 1, 7)`. The Random Forest uses 300 trees and lag, rolling-average, and calendar features. Both improve on the reported baselines. These are **rolling one-step holdout** scores: each forecast uses actual observations available up to that day. They should not be read as seven-day-ahead recursive accuracy.

> **Forecast scope:** the SARIMAX model includes a weekly seasonal cycle. Its exported seven-day forecast covers January 1–7, 2018; longer-horizon forecasts are not established by these validation results.

## Repository structure

```text
.
├── Person1_Supermarket_TimeSeries_ARIMA.ipynb
├── Person2_Supermarket_Machine_Learning.ipynb
├── demand-forecasting-kernels-only/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── output/
    ├── arima_7_day_forecast.csv
    ├── arima_model_comparison.csv
    ├── arima_residuals.csv
    ├── arima_validation_predictions.csv
    ├── final_model_comparison.csv
    ├── person2_7_day_forecast.csv
    ├── person2_model_comparison.csv
    ├── person2_residuals.csv
    └── person2_validation_predictions.csv
```

### Output files

| File | Contents |
|---|---|
| `arima_model_comparison.csv` | ARIMA, SARIMAX, and baseline metrics for the time-series notebook. |
| `arima_validation_predictions.csv` | Actual holdout values and rolling one-step predictions from candidate time-series models. |
| `arima_residuals.csv` | Selected time-series model's training residuals after differencing burn-in removal. |
| `arima_7_day_forecast.csv` | SARIMAX daily forecast and 95% lower/upper bounds for January 1–7, 2018. |
| `person2_model_comparison.csv` | Random Forest and baseline validation metrics. |
| `person2_validation_predictions.csv` | Actual holdout values and baseline/Random Forest predictions. |
| `person2_residuals.csv` | Random Forest holdout residuals (`actual − predicted`). |
| `person2_7_day_forecast.csv` | Recursive Random Forest forecast for January 1–7, 2018. |
| `final_model_comparison.csv` | Combined model comparison exported by the machine-learning notebook. |

## Getting started

### Requirements

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `statsmodels`

Create an environment and install the required libraries:

```bash
python -m venv .venv
```

Activate it, then install dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter
```

Launch Jupyter from the project root:

```bash
jupyter notebook
```

Open either notebook and run its cells from top to bottom. The machine-learning notebook reads the included training file at `demand-forecasting-kernels-only/train.csv`. The ARIMA notebook currently looks for the data folder one level above the project (or under a sibling `Downloads` folder); to run it with the repository's included data, update its `DATA_CANDIDATES` setting to include `PROJECT_DIR / 'demand-forecasting-kernels-only'`, or place the data folder in one of the locations it searches.

The notebooks write their CSV exports to `output/`. Existing CSV files are provided as the results already generated by the analyses.

## Methodology

### Shared data preparation

1. Parse the transaction date and inspect the raw records.
2. Sum `sales` across stores and items for each date.
3. Build a continuous daily time series from 2013-01-01 to 2017-12-31.
4. Use the first 1,736 days for training and the final 90 days for chronological validation.

### Time-series notebook

The analysis examines moving averages, weekday and monthly patterns, classical decomposition and STL, ACF/PACF, and ADF/KPSS stationarity tests. It compares seven ARIMA candidates and four period-7 SARIMAX candidates using rolling one-step evaluation, then checks residual autocorrelation and normality for the selected specification.

### Machine-learning notebook

The Random Forest model uses 11 predictors: sales lags (1, 7, 14, and 28 days), prior-only rolling means (7, 14, and 28 days), day of week, day of month, month, and a weekend flag. Its holdout predictions and naive baseline predictions are evaluated on the same 90 dates. A final model recursively forecasts seven future dates.

## Data dictionary

`train.csv` contains one row per date, store, and item:

| Column | Description |
|---|---|
| `date` | Observation date. |
| `store` | Store identifier. |
| `item` | Item identifier. |
| `sales` | Units sold for that store, item, and date. |

`test.csv` contains `id`, `date`, `store`, and `item`, but no `sales` target. `sample_submission.csv` provides the expected submission format for the source forecasting dataset.

## Notes and limitations

- The target is **total sales across all stores and items**, not a separate forecast for every store or item.
- The holdout protocol is rolling one-step: the actual from each holdout date is available before forecasting the next one.
- The two seven-day forecasts are produced by different models and should not be treated as a single ensemble.
- The stored forecast dates reflect the supplied dataset's final date (2017-12-31); rerunning the notebooks on the same data recreates forecasts for January 1–7, 2018.
- The ARIMA notebook's current data search path differs from the repository layout; see the run instructions above.

## Built with

Python · pandas · NumPy · Matplotlib · Seaborn · statsmodels · scikit-learn · Jupyter
