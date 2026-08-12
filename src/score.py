"""
Batch-score all currently Open maintenance events with the trained model.

Writes results to a `scored_events` table in the warehouse (idempotent:
full delete-then-insert in one transaction) so the Streamlit app can read
scores directly without needing to load the model at request time -- and so
scoring has an auditable, reproducible artifact independent of the app.

For each event we also compute "top drivers": for the linear (logistic
regression) model this is an exact decomposition of the predicted logit into
per-feature contributions; for a tree ensemble we fall back to the model's
global feature importances, since per-instance attribution would need a
library like SHAP that's out of scope for this thin slice.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_COLUMNS, build_feature_frame

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "warehouse.db"

TOP_K_DRIVERS = 4


def _linear_top_drivers(pipeline, X: pd.DataFrame) -> list[list[dict]]:
    """Exact per-row logit decomposition for a (preprocess, linear clf) pipeline."""
    preprocess = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]
    feature_names = preprocess.get_feature_names_out()
    coefs = clf.coef_[0]

    transformed = preprocess.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    contributions = transformed * coefs  # row-wise contribution per output feature

    drivers = []
    for row in contributions:
        idx = np.argsort(-np.abs(row))[:TOP_K_DRIVERS]
        drivers.append([
            {"feature": feature_names[i].split("__", 1)[-1], "contribution": round(float(row[i]), 4)}
            for i in idx if abs(row[i]) > 1e-9
        ])
    return drivers


def _global_importance_drivers(pipeline, X: pd.DataFrame) -> list[list[dict]]:
    """Fallback for non-linear models: same global ranking repeated per row,
    labeled as global (not per-instance) importance."""
    clf = pipeline.named_steps["clf"]
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        return [[] for _ in range(len(X))]
    idx = np.argsort(-importances)[:TOP_K_DRIVERS]
    top = [{"feature": FEATURE_COLUMNS[i], "contribution": round(float(importances[i]), 4)}
           for i in idx]
    return [top for _ in range(len(X))]


def compute_top_drivers(pipeline, X: pd.DataFrame) -> list[list[dict]]:
    if "preprocess" in pipeline.named_steps and hasattr(pipeline.named_steps["clf"], "coef_"):
        return _linear_top_drivers(pipeline, X)
    return _global_importance_drivers(pipeline, X)


SCORED_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS scored_events (
    event_id TEXT PRIMARY KEY,
    component_id TEXT,
    aircraft_tail TEXT,
    base_code TEXT,
    component_class TEXT,
    opened_at TEXT,
    priority_code TEXT,
    risk_score REAL NOT NULL,
    percentile_rank REAL NOT NULL,
    top_drivers_json TEXT,
    model_version INTEGER,
    scored_at TEXT NOT NULL
);
"""


def main():
    model_path = MODELS_DIR / "model.pkl"
    metadata_path = MODELS_DIR / "metadata.json"
    if not model_path.exists():
        raise SystemExit("No trained model found at models/model.pkl -- run src/train.py first.")

    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())

    conn = sqlite3.connect(DB_PATH)
    df = build_feature_frame(conn)

    open_df = df[df["status"] == "Open"].copy()
    print(f"Scoring {len(open_df)} open events...")

    if open_df.empty:
        print("No open events to score.")
        conn.close()
        return

    X = open_df[FEATURE_COLUMNS]
    scores = pipeline.predict_proba(X)[:, 1]
    open_df["risk_score"] = scores
    open_df["percentile_rank"] = open_df["risk_score"].rank(pct=True)
    open_df = open_df.sort_values("risk_score", ascending=False)

    drivers = compute_top_drivers(pipeline, open_df[FEATURE_COLUMNS])
    open_df["top_drivers_json"] = [json.dumps(d) for d in drivers]

    scored_at = datetime.now(timezone.utc).isoformat()
    out = open_df[[
        "event_id", "component_id", "aircraft_tail", "base_code", "component_class",
        "opened_at", "priority_code", "risk_score", "percentile_rank", "top_drivers_json",
    ]].copy()
    out["opened_at"] = out["opened_at"].astype(str)
    out["model_version"] = metadata.get("version")
    out["scored_at"] = scored_at

    conn.executescript(SCORED_EVENTS_DDL)
    with conn:
        conn.execute("DELETE FROM scored_events")
        conn.executemany(
            "INSERT INTO scored_events (event_id, component_id, aircraft_tail, base_code, "
            "component_class, opened_at, priority_code, risk_score, percentile_rank, "
            "top_drivers_json, model_version, scored_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            out[["event_id", "component_id", "aircraft_tail", "base_code", "component_class",
                 "opened_at", "priority_code", "risk_score", "percentile_rank",
                 "top_drivers_json", "model_version", "scored_at"]].itertuples(index=False, name=None),
        )
    conn.close()

    print(f"Wrote {len(out)} scored events to scored_events table.")
    print("\nTop 10 highest-risk open events:")
    print(out[["event_id", "aircraft_tail", "component_class", "risk_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
