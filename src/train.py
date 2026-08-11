"""
Train the freight-rate models.

Usage:
    python src/train.py --data data/train_test.csv --out models

Produces:
    models/full_model.joblib       (uses market_index + quote_signal)
    models/reduced_model.joblib    (no market signals -- used for December)
    models/fit_stats.joblib        (imputation medians learned on train)
    reports/metrics.json
    reports/eda_notes.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb

from data_utils import FULL_FEATURES, REDUCED_FEATURES, TARGET, build_matrix, clean


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def time_split(df: pd.DataFrame, holdout_frac: float = 0.2):
    """Sort by date and hold out the most recent `holdout_frac` of rows.

    This mimics the real task: the model is trained on the past and has to
    forecast rates for dates it has not seen (validation.csv dates extend
    past the train_test range, and December 2025 is further out still), so
    a random split would understate real-world error and let information
    from "the future" leak into training.
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - holdout_frac))
    return df_sorted.iloc[:cutoff].copy(), df_sorted.iloc[cutoff:].copy()


def fit_xgb(X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=4,
        eval_metric="mae",
        early_stopping_rounds=40,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate(name: str, y_true, y_pred) -> dict:
    return {
        "model": name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAPE_%": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train_test.csv")
    parser.add_argument("--out", default="models")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data)
    cleaned, fit_stats = clean(raw, fit_stats=None)
    joblib.dump(fit_stats, out_dir / "fit_stats.joblib")

    train_df, val_df = time_split(cleaned, holdout_frac=0.2)
    print(f"Train rows: {len(train_df):,}  |  Time-holdout rows: {len(val_df):,}")
    print(f"Train dates: {train_df['date'].min().date()} -> {train_df['date'].max().date()}")
    print(f"Holdout dates: {val_df['date'].min().date()} -> {val_df['date'].max().date()}")

    metrics = []

    # ---------------- FULL model (with market_index / quote_signal) ----------------
    X_train_full = build_matrix(train_df, FULL_FEATURES)
    X_val_full = build_matrix(val_df, FULL_FEATURES)
    y_train = train_df[TARGET].values
    y_val = val_df[TARGET].values

    lr_full = LinearRegression().fit(X_train_full, y_train)
    metrics.append(evaluate("linear_full", y_val, lr_full.predict(X_val_full)))

    rf_full = RandomForestRegressor(
        n_estimators=400, max_depth=14, min_samples_leaf=3, n_jobs=4, random_state=42
    ).fit(X_train_full, y_train)
    metrics.append(evaluate("random_forest_full", y_val, rf_full.predict(X_val_full)))

    xgb_full = fit_xgb(X_train_full, y_train, X_val_full, y_val)
    metrics.append(evaluate("xgboost_full", y_val, xgb_full.predict(X_val_full)))

    joblib.dump(xgb_full, out_dir / "full_model.joblib")

    # ---------------- REDUCED model (no market signals -- for December) -----------
    X_train_red = build_matrix(train_df, REDUCED_FEATURES)
    X_val_red = build_matrix(val_df, REDUCED_FEATURES)

    xgb_red = fit_xgb(X_train_red, y_train, X_val_red, y_val)
    metrics.append(evaluate("xgboost_reduced_no_market_signals", y_val, xgb_red.predict(X_val_red)))

    joblib.dump(xgb_red, out_dir / "reduced_model.joblib")

    # ---------------- persist metrics + feature importance --------------------
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fi_full = pd.Series(xgb_full.feature_importances_, index=FULL_FEATURES).sort_values(ascending=False)
    fi_red = pd.Series(xgb_red.feature_importances_, index=REDUCED_FEATURES).sort_values(ascending=False)
    fi_full.to_csv(reports_dir / "feature_importance_full.csv", header=["importance"])
    fi_red.to_csv(reports_dir / "feature_importance_reduced.csv", header=["importance"])

    print("\n=== Time-holdout metrics ===")
    for m in metrics:
        print(m)
    print("\nTop features (full model):")
    print(fi_full.head(8))
    print("\nTop features (reduced model, used for December):")
    print(fi_red.head(8))

    print("\nSaved models to", out_dir.resolve())


if __name__ == "__main__":
    main()
