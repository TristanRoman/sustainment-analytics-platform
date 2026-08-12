"""
Train the "event exceeds 48 hours" classifier.

Time-aware split: the first 2 years of the observation window are train,
the final year is held out for evaluation -- no random shuffling, since a
random split would leak future maintenance patterns into training and
overstate performance versus how the model will actually be used (predict
forward on events not yet seen).

Only Closed events have a known outcome, so training and evaluation both
exclude Open events (they are scored later, by score.py, not evaluated
here -- there is no ground truth for them yet). A reopened event's clock
still starts at its original creation time, so it sits wherever that
creation date falls in the split; we don't create a second observation for
the reopen itself.

Two models are trained on identical inputs (via features.py) so the
comparison is apples-to-apples:
  - Logistic regression baseline (one-hot + scaling via ColumnTransformer)
  - HistGradientBoostingClassifier (native pandas categorical support)

The better model by PR-AUC on the held-out year is saved as models/model.pkl
along with models/metadata.json describing training window, features, and
metrics for both candidates.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FEATURE_COLUMNS, build_feature_frame

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "warehouse.db"

TOP_DECILE = 0.10  # planner capacity: flag the top 10% of scored events


def time_aware_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    start = df["opened_at"].min()
    cutoff = start + pd.DateOffset(years=2)
    train = df[df["opened_at"] < cutoff]
    test = df[df["opened_at"] >= cutoff]
    return train, test, cutoff


def build_lr_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def build_gbm_pipeline() -> Pipeline:
    # HistGradientBoostingClassifier reads pandas "category" dtype columns
    # natively -- no manual encoding needed, so the raw feature frame from
    # features.py can be passed straight through.
    clf = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline([("clf", clf)])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)

    threshold = float(np.quantile(proba, 1 - TOP_DECILE))
    flagged = proba >= threshold
    precision = precision_score(y_test, flagged, zero_division=0)
    recall = recall_score(y_test, flagged, zero_division=0)

    return {
        "pr_auc": pr_auc,
        "base_rate": float(y_test.mean()),
        "operating_threshold": threshold,
        "flagged_fraction": float(flagged.mean()),
        "precision_at_threshold": float(precision),
        "recall_at_threshold": float(recall),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df = build_feature_frame(conn)
    conn.close()

    closed = df[df["status"] == "Closed"].copy()
    closed["exceeds_48h"] = closed["exceeds_48h"].astype(int)
    print(f"Closed events available for train/eval: {len(closed)} "
          f"(excluded {len(df) - len(closed)} still-open events)")

    train_df, test_df, cutoff = time_aware_split(closed)
    print(f"Train window: {train_df['opened_at'].min()} -> {train_df['opened_at'].max()} "
          f"({len(train_df)} events)")
    print(f"Test window (held out year 3): {cutoff} -> {test_df['opened_at'].max()} "
          f"({len(test_df)} events)")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["exceeds_48h"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["exceeds_48h"]

    print(f"\nBase rate -- train: {y_train.mean():.3%}, test: {y_test.mean():.3%}")

    results = {}

    print("\nTraining logistic regression baseline...")
    lr = build_lr_pipeline()
    lr.fit(X_train, y_train)
    results["logistic_regression"] = evaluate(lr, X_test, y_test)
    print(f"  PR-AUC: {results['logistic_regression']['pr_auc']:.4f} "
          f"(base rate {results['logistic_regression']['base_rate']:.3%})")
    print(f"  precision@top10%: {results['logistic_regression']['precision_at_threshold']:.3f}  "
          f"recall@top10%: {results['logistic_regression']['recall_at_threshold']:.3f}")

    print("\nTraining gradient boosting...")
    gbm = build_gbm_pipeline()
    gbm.fit(X_train, y_train)
    results["gradient_boosting"] = evaluate(gbm, X_test, y_test)
    print(f"  PR-AUC: {results['gradient_boosting']['pr_auc']:.4f} "
          f"(base rate {results['gradient_boosting']['base_rate']:.3%})")
    print(f"  precision@top10%: {results['gradient_boosting']['precision_at_threshold']:.3f}  "
          f"recall@top10%: {results['gradient_boosting']['recall_at_threshold']:.3f}")

    winner = max(results, key=lambda k: results[k]["pr_auc"])
    winner_model = {"logistic_regression": lr, "gradient_boosting": gbm}[winner]
    print(f"\nSelected model: {winner} (higher PR-AUC on held-out year)")

    model_path = MODELS_DIR / "model.pkl"
    joblib.dump(winner_model, model_path)

    metadata = {
        "version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": winner,
        "data_window": {
            "train_start": str(train_df["opened_at"].min()),
            "train_end": str(train_df["opened_at"].max()),
            "test_start": str(cutoff),
            "test_end": str(test_df["opened_at"].max()),
        },
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
        },
        "target": "exceeds_48h (opened_at to closed_at > 48 hours; still-open events excluded from training)",
        "top_decile_capacity": TOP_DECILE,
        "metrics": results,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved model to {model_path}")
    print(f"Saved metadata to {MODELS_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
