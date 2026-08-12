"""
Shared feature engineering for train.py and score.py.

Single source of truth for which columns become model inputs and how they're
computed, so scoring can never silently drift from what the model was
trained on. Only fields knowable at event-creation time are included here --
nothing populated after the event resolves (duration_hours, closed_at,
exceeds_48h are targets/outcomes, never features).

The actual numeric encoding (one-hot, scaling, etc.) lives inside the sklearn
Pipeline saved by train.py, not here -- that pipeline is the artifact reused
by score.py, which is a stronger anti-drift guarantee than re-implementing
encoding logic in two places.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "warehouse.db"

CATEGORICAL_FEATURES = ["component_class", "base_code", "aircraft_tail", "priority_code", "month"]
NUMERIC_FEATURES = ["component_age_hours", "hours_since_overhaul", "prior_event_count"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columns carried alongside the features for splitting/labeling/display but
# that are never fed to the model.
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
    """Return one row per event with all model features plus context
    columns. Includes both Closed and Open events -- callers decide how to
    filter (train.py excludes Open from training; score.py scores exactly
    the Open ones)."""
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
