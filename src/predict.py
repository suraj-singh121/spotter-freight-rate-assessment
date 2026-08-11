"""
Generate the two required prediction outputs.

Usage:
    python src/predict.py

Produces:
    validation_predictions.csv               (load_id, predicted_rate) x 12,000
    data/december_chart_inputs.csv           (filled in place, predicted_rate column)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from data_utils import FULL_FEATURES, REDUCED_FEATURES, build_matrix, clean

MODELS_DIR = Path("models")
DATA_DIR = Path("data")


def build_city_coord_lookup(train_df: pd.DataFrame) -> dict:
    """City name -> (lat, lon), built from train_test.csv.

    december_chart_inputs.csv only gives city names (no lat/lon columns),
    so we look up the same coordinates the model was trained on for
    Lexington / Fort Wayne rather than dropping the lat/lon features
    (which would change the feature schema between training and this
    prediction pass).
    """
    pickup_coords = train_df[["pickup", "pickup_lat", "pickup_lon"]].drop_duplicates()
    pickup_coords.columns = ["city", "lat", "lon"]
    delivery_coords = train_df[["delivery", "delivery_lat", "delivery_lon"]].drop_duplicates()
    delivery_coords.columns = ["city", "lat", "lon"]
    combined = pd.concat([pickup_coords, delivery_coords]).groupby("city").mean()
    return combined.to_dict(orient="index")


def main() -> None:
    fit_stats = joblib.load(MODELS_DIR / "fit_stats.joblib")
    full_model = joblib.load(MODELS_DIR / "full_model.joblib")
    reduced_model = joblib.load(MODELS_DIR / "reduced_model.joblib")

    # ---------------- validation.csv -> validation_predictions.csv ----------------
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")

    cleaned_val, _ = clean(validation, fit_stats=fit_stats)
    X_val = build_matrix(cleaned_val, FULL_FEATURES)
    cleaned_val["predicted_rate"] = full_model.predict(X_val)

    preds = template[["load_id"]].merge(
        cleaned_val[["load_id", "predicted_rate"]], on="load_id", how="left"
    )
    assert preds["predicted_rate"].isna().sum() == 0, "missing predictions for some load_id"
    assert (preds["predicted_rate"] > 0).all(), "non-positive predictions produced"
    preds.to_csv("validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv with {len(preds):,} rows")

    # ---------------- december_chart_inputs.csv (fixed lane, date-only) -----------
    # december_chart_inputs.csv has no lat/lon columns, so look up the same
    # Lexington / Fort Wayne coordinates the reduced model was trained on.
    train_raw = pd.read_csv(DATA_DIR / "train_test.csv")
    coord_lookup = build_city_coord_lookup(train_raw)

    december = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")
    cleaned_dec, _ = clean(december, fit_stats=fit_stats)
    cleaned_dec["pickup_lat"] = cleaned_dec["pickup"].map(lambda c: coord_lookup[c]["lat"])
    cleaned_dec["pickup_lon"] = cleaned_dec["pickup"].map(lambda c: coord_lookup[c]["lon"])
    cleaned_dec["delivery_lat"] = cleaned_dec["delivery"].map(lambda c: coord_lookup[c]["lat"])
    cleaned_dec["delivery_lon"] = cleaned_dec["delivery"].map(lambda c: coord_lookup[c]["lon"])

    X_dec = build_matrix(cleaned_dec, REDUCED_FEATURES)
    december["predicted_rate"] = reduced_model.predict(X_dec)
    december.to_csv(DATA_DIR / "december_chart_inputs.csv", index=False)
    print(f"Filled data/december_chart_inputs.csv with {len(december)} rows")


if __name__ == "__main__":
    main()
