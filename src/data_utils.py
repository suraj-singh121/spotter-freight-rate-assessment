"""
Shared data-cleaning and feature-engineering utilities for the freight
rate prediction assessment.

Two feature sets are produced from the same cleaning pipeline:

- FULL_FEATURES   : includes market_index / quote_signal. Used for the
                     main validation.csv predictions, where those columns
                     are provided.
- REDUCED_FEATURES: excludes market_index / quote_signal. Used for the
                     December chart-input predictions, where those two
                     market-condition columns are NOT provided (they are
                     effectively unknowable for a future, hypothetical
                     lane/date combination). Training a second model on
                     the same lane/date/equipment/weight features that
                     ARE available for December keeps train/serve
                     features consistent instead of guessing values for
                     columns we don't actually have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT_CATEGORIES = ["Dry Van", "Reefer", "Flatbed"]

FULL_FEATURES = [
    "distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month",
    "day_of_week",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
    "market_index",
    "quote_signal",
    "equipment_Dry Van",
    "equipment_Reefer",
    "equipment_Flatbed",
]

REDUCED_FEATURES = [f for f in FULL_FEATURES if f not in ("market_index", "quote_signal")]

TARGET = "posted_rate"


def clean(df: pd.DataFrame, fit_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean raw columns and engineer date features.

    If `fit_stats` is None, imputation statistics (medians) are computed
    from `df` and returned so the *exact same* values can be reused when
    cleaning validation / December data (avoids leakage and keeps
    train/serve consistent).
    """
    out = df.copy()

    # --- weight: fix sign-flip data-entry errors (weight should be > 0) ---
    out["weight"] = out["weight"].abs()

    # --- date features ---
    out["date"] = pd.to_datetime(out["date"])
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    doy = out["date"].dt.dayofyear
    out["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    # --- equipment one-hot ---
    for cat in EQUIPMENT_CATEGORIES:
        out[f"equipment_{cat}"] = (out["equipment"] == cat).astype(int)

    learned = {} if fit_stats is None else dict(fit_stats)

    # --- weight imputation: median weight within the same equipment type ---
    if "weight_median_by_equipment" not in learned:
        learned["weight_median_by_equipment"] = out.groupby("equipment")["weight"].median().to_dict()
    global_weight_median = out["weight"].median()
    weight_fill = out["equipment"].map(learned["weight_median_by_equipment"])
    out["weight"] = out["weight"].fillna(weight_fill).fillna(global_weight_median)

    # --- market_index imputation (full-feature path only): median by month ---
    if "market_index" in out.columns:
        if "market_index_median_by_month" not in learned:
            learned["market_index_median_by_month"] = out.groupby("month")["market_index"].median().to_dict()
        global_mi_median = out["market_index"].median()
        mi_fill = out["month"].map(learned["market_index_median_by_month"])
        out["market_index"] = out["market_index"].fillna(mi_fill).fillna(global_mi_median)

    return out, learned


def build_matrix(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    return df[feature_list].astype(float)
