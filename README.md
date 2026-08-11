# Freight Rate Prediction — Spotter ML Engineer Assessment

Predicts `posted_rate` for freight loads and produces:
1. `validation_predictions.csv` — predictions for the 12,000 loads in `data/validation.csv`
2. A filled `data/december_chart_inputs.csv` + `scorer_results/candidate_december.png` — a fixed-lane December 2025 rate forecast

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Run

```bash
# 1. Train both models (full model w/ market signals, reduced model for December)
python src/train.py --data data/train_test.csv --out models --reports reports

# 2. Generate validation_predictions.csv and fill data/december_chart_inputs.csv
python src/predict.py

# 3. Validate outputs and generate the December chart (provided scorer)
python -m pip install -r requirements.txt   # matplotlib/numpy/pandas pins from Spotter
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Outputs land in `models/`, `reports/`, and `scorer_results/candidate_december.png`.

## Repo layout

```
data/                 train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv
src/data_utils.py      shared cleaning + feature engineering (single source of truth for train/serve)
src/train.py           time-based train/holdout split, trains linear/RF/XGBoost baselines, saves best models
src/predict.py         loads saved models, produces both required prediction outputs
score.py                provided scorer (validates outputs, renders the December chart)
reports/                metrics.json, feature importances, EDA notes
report/                 PDF/DOCX write-up with approach + December chart (per assessment instructions)
validation_predictions.csv
```

## Approach summary

See `report/Freight_Rate_Assessment_Report.docx` (or the PDF export) for the full write-up. Short version:

- **Data quality issues found:** `weight` had ~0.6% sign-flipped negative values (fixed with `abs()`) and ~0.6% missing values (imputed by equipment-type median); `market_index` had ~0.8% missing values (imputed by month median); a handful of lat/lon pairs for a couple of city names look mis-geocoded relative to their `distance` value (flagged, not corrected — no ground truth to correct against, and it's <0.05% of rows).
- **Split:** time-based (sorted by date, last 20% held out) rather than random, since the real task is forecasting rates for dates the model hasn't seen (validation.csv and December both sit later in time than most of `train_test.csv`). A random split would understate real error.
- **Model:** XGBoost regressor, chosen over linear regression and random forest on held-out MAE/MAPE (see `reports/metrics.json`).
- **Two feature sets, one pipeline:** `validation.csv` includes `market_index`/`quote_signal`, so the main model uses them. `december_chart_inputs.csv` does not include those two market-condition columns (they aren't knowable in advance for a future date), so a second model is trained on the same data with those two features excluded, and is used only for the December forecast. Both models share the same cleaning/feature code in `src/data_utils.py`.
