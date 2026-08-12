"""
Shared feature engineering for train.py and score.py.

This module is the single source of truth for which columns become model
inputs and how they're computed, because that's what stops scoring from
silently drifting away from what the model was trained on -- if train.py
and score.py each computed features independently, a change to one could
go unnoticed by the other. Only fields knowable at event-creation time are
included here, since duration_hours, closed_at, and exceeds_48h are
targets/outcomes that wouldn't exist yet at prediction time and would leak
the answer if used as features.

The actual numeric encoding (one-hot, scaling, etc.) lives inside the
sklearn Pipeline saved by train.py, not here, because that pipeline is the
artifact reused unchanged by score.py -- a stronger anti-drift guarantee
than re-implementing encoding logic in two places.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "warehouse.db"

CATEGORICAL_FEATURES = ["component_class", "base_code", "aircraft_tail", "priority_code", "month"]
NUMERIC_FEATURES = ["component_age_hours", "hours_since_overhaul", "prior_event_count"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# These columns are kept separate from FEATURE_COLUMNS because they're
# needed for splitting/labeling/display (e.g. opened_at for the time-aware
# split, exceeds_48h as the label) but would leak the outcome or add noise
# if fed to the model as inputs.
CONTEXT_COLUMNS = ["event_id", "component_id", "opened_at", "status", "exceeds_48h", "event_type"]


FEATURE_QUERY = """
SELECT
    e.event_id,
    e.component_id,
    e.aircraft_tail,
    e.base_code,
    e.opened_at,
    e.status,
    e.exceeds_48h,
    e.event_type,
    e.priority_code,
    e.component_age_hours,
    e.hours_since_overhaul,
    c.component_class,
    CAST(strftime('%m', e.opened_at) AS INTEGER) AS month,
    COUNT(*) OVER (
        PARTITION BY e.component_id
        ORDER BY e.opened_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prior_event_count
FROM fact_maintenance_events e
JOIN dim_component c ON c.component_id = e.component_id
"""


def build_feature_frame(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    """This returns both Closed and Open events, rather than filtering
    here, because train.py and score.py need different subsets (train.py
    excludes Open due to having no ground truth yet; score.py scores
    exactly the Open ones) -- pushing the filter into this shared function
    would force one caller's needs onto the other."""
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(FEATURE_QUERY, conn, parse_dates=["opened_at"])
    finally:
        if own_conn:
            conn.close()

    df["priority_code"] = df["priority_code"].fillna("Unknown")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    return df[CONTEXT_COLUMNS + FEATURE_COLUMNS]
