# Loom Walkthrough — Talking Points (aim for 2:30–3:00)

Record your screen going through the repo + report; use this as a script, in your own words.

## 1. Key findings from exploring the data (~30s)
- 48,000 labeled loads, Jan–Oct 2025. Target `posted_rate` is heavily driven by `distance`
  (r ≈ 0.91) — show the correlation quickly if you have a notebook/cell open.
- Equipment type adds a modest premium: Reefer > Flatbed > Dry Van.
- `market_index`/`quote_signal` don't correlate much with raw rate, but do help once you
  control for distance (rate-per-mile) — mention this is why they're still in the model.

## 2. Data-quality issues and how you addressed them (~40s)
- `weight`: ~0.6% of rows had sign-flipped negative values — fixed with `abs()` (magnitudes
  matched the real distribution exactly, so this reads as a data-entry sign bug, not garbage).
- `weight` and `market_index` had small amounts of missing data — imputed with group medians
  (by equipment type, and by month, respectively) rather than dropping rows.
- Flag (not "fixed"): ~22 rows for the Shreveport↔New Orleans lane have a distance value that
  doesn't match their lat/lon via haversine — looks like a geocoding issue for those two
  cities. Called it out in the report rather than silently altering it without ground truth.

## 3. Reasoning behind the chosen model (~30s)
- Compared linear regression, random forest, and XGBoost on the same held-out data.
- XGBoost won clearly on MAE/MAPE (~$125 MAE, 5.5% MAPE vs ~10.7% for linear regression).
- Two XGBoost models, same pipeline: one with `market_index`/`quote_signal` (used for the main
  validation predictions, since that file has those columns), one without them (used only for
  the December forecast, since `december_chart_inputs.csv` doesn't include those two columns —
  they're live market-quote signals you can't know in advance for a future date).

## 4. Training and validation approach, including how the data was split (~30s)
- Time-based split, not random: sorted by date, trained on the earliest 80% (through Aug 2025),
  held out the most recent 20% (Sep–Oct 2025) purely for evaluation.
- Rationale: the real task is forecasting forward (validation.csv and December both sit later
  in time), so a random split would leak "future" rows into training and overstate accuracy.

## 5. Walkthrough of the most important code (~40s)
- `src/data_utils.py` — one shared `clean()` function used by both training and prediction, so
  train-time and serve-time cleaning can't drift apart. Two feature lists (`FULL_FEATURES`,
  `REDUCED_FEATURES`) built from it.
- `src/train.py` — time split, trains/evaluates 3 baselines + saves the two XGBoost models.
- `src/predict.py` — loads saved models, produces `validation_predictions.csv`, and fills
  `data/december_chart_inputs.csv` (with a Lexington/Fort Wayne lat-lon lookup since that file
  has no lat/lon columns of its own).
- Close by running `score.py` on screen and showing it passes validation + the December chart.
