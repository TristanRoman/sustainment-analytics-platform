"""
Ingest raw source files into the SQLite star schema (data/warehouse.db).

Pipeline per source: read -> coerce types (log failures) -> canonicalize
categories -> validate business rules (quarantine violations) -> dedupe ->
referential integrity check -> idempotent load.

Idempotency: every load uses a fixed BATCH_ID and loads inside a single
transaction as delete-then-insert, so re-running ingest on unchanged raw
files reproduces exactly the same warehouse state.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "warehouse.db"

BATCH_ID = "full_history_load"

# Canonical base reference data (crosswalk from messy source aliases).
BASE_ALIASES = {
    "sd": "SD", "san diego": "SD",
    "norva": "NORVA", "norfolk": "NORVA", "norfolk va": "NORVA",
    "jax": "JAX", "jacksonville": "JAX",
    "lemoore": "LEMOORE", "nas lemoore": "LEMOORE",
    "whidbey": "WHIDBEY", "whidbey island": "WHIDBEY", "whidbey isl": "WHIDBEY",
    "oceana": "OCEANA", "nas oceana": "OCEANA",
}
BASE_NAMES = {
    "SD": "San Diego", "NORVA": "Norfolk", "JAX": "Jacksonville",
    "LEMOORE": "Lemoore", "WHIDBEY": "Whidbey Island", "OCEANA": "Oceana",
}

VALID_COMPONENT_CLASSES = {
    "hydraulic_pump", "actuator", "avionics_module", "landing_gear",
    "environmental_control",
}
VALID_EVENT_TYPES = {"Unscheduled", "Scheduled", "Routine"}
VALID_PRIORITY_CODES = {"Routine", "Priority", "AOG"}
VALID_STATUS = {"Open", "Closed"}

DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d-%b-%Y %H:%M:%S",
]


# --------------------------------------------------------------------------
# Logging / metrics collection
# --------------------------------------------------------------------------

class RunLog:
    """Collects quarantine rows and stage row-counts for one ingest run."""

    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.quarantine_rows: list[dict] = []
        self.metrics_rows: list[dict] = []

    def quarantine(self, source_table: str, key: str, rule: str, detail: str, raw_row: dict):
        self.quarantine_rows.append({
            "source_table": source_table,
            "record_key": key,
            "rule_broken": rule,
            "detail": detail,
            "raw_row_json": json.dumps(raw_row, default=str),
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
            "batch_id": self.batch_id,
        })

    def stage(self, table_name: str, stage: str, rows_in: int, rows_out: int):
        rows_q = rows_in - rows_out
        self.metrics_rows.append({
            "batch_id": self.batch_id,
            "stage": stage,
            "table_name": table_name,
            "rows_in": rows_in,
            "rows_out": rows_out,
            "rows_quarantined": rows_q,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"  [{table_name}:{stage}] in={rows_in} out={rows_out} quarantined={rows_q}")


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

def parse_mixed_date(value) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        return value
    s = str(value).strip()
    if not s:
        return None
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    return None  # unparseable; caller logs the failure


def canonicalize_base(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    key = str(value).strip().lower()
    return BASE_ALIASES.get(key)


# --------------------------------------------------------------------------
# Schema DDL
# --------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS dim_base (
    base_code TEXT PRIMARY KEY,
    base_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_aircraft (
    aircraft_tail TEXT PRIMARY KEY,
    base_code TEXT NOT NULL REFERENCES dim_base(base_code),
    model TEXT,
    entered_service_date TEXT,
    monthly_flight_hours REAL,
    batch_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_component (
    component_id TEXT PRIMARY KEY,
    aircraft_tail TEXT NOT NULL REFERENCES dim_aircraft(aircraft_tail),
    component_class TEXT NOT NULL,
    install_date TEXT,
    batch_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_id INTEGER PRIMARY KEY,
    date TEXT UNIQUE NOT NULL,
    year INTEGER, month INTEGER, day INTEGER,
    day_of_week INTEGER, month_name TEXT, quarter INTEGER, is_weekend INTEGER
);

CREATE TABLE IF NOT EXISTS fact_maintenance_events (
    event_id TEXT PRIMARY KEY,
    component_id TEXT NOT NULL REFERENCES dim_component(component_id),
    aircraft_tail TEXT NOT NULL REFERENCES dim_aircraft(aircraft_tail),
    base_code TEXT NOT NULL REFERENCES dim_base(base_code),
    date_id INTEGER NOT NULL REFERENCES dim_date(date_id),
    event_type TEXT NOT NULL,
    priority_code TEXT,
    status TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    hours_since_overhaul REAL,
    component_age_hours REAL,
    duration_hours REAL,
    staffing_level REAL,
    parts_backorder INTEGER,
    technician_id TEXT,
    notes TEXT,
    record_loaded_at TEXT,
    exceeds_48h INTEGER,
    batch_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_flight_hours (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    aircraft_tail TEXT NOT NULL REFERENCES dim_aircraft(aircraft_tail),
    month_start TEXT NOT NULL,
    flight_hours REAL,
    batch_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    base_code TEXT NOT NULL REFERENCES dim_base(base_code),
    snapshot_date TEXT NOT NULL,
    part_category TEXT,
    qty_available INTEGER,
    qty_backorder INTEGER,
    batch_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quarantine (
    quarantine_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_table TEXT NOT NULL,
    record_key TEXT,
    rule_broken TEXT NOT NULL,
    detail TEXT,
    raw_row_json TEXT,
    quarantined_at TEXT NOT NULL,
    batch_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_quality_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    table_name TEXT NOT NULL,
    rows_in INTEGER,
    rows_out INTEGER,
    rows_quarantined INTEGER,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fact_events_component ON fact_maintenance_events(component_id);
CREATE INDEX IF NOT EXISTS idx_fact_events_aircraft ON fact_maintenance_events(aircraft_tail);
CREATE INDEX IF NOT EXISTS idx_fact_events_date ON fact_maintenance_events(date_id);
CREATE INDEX IF NOT EXISTS idx_quarantine_batch ON quarantine(batch_id);
"""


def init_schema(conn: sqlite3.Connection):
    conn.executescript(DDL)


# --------------------------------------------------------------------------
# Idempotent load helper: delete-then-insert by batch, in a transaction
# --------------------------------------------------------------------------

def idempotent_load(conn: sqlite3.Connection, table: str, df: pd.DataFrame, batch_id: str,
                     delete_key: str = "batch_id"):
    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)
    with conn:  # transaction
        conn.execute(f"DELETE FROM {table} WHERE {delete_key} = ?", (batch_id,))
        conn.executemany(
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
            df[cols].itertuples(index=False, name=None),
        )


def build_dim_date(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(min_date.normalize(), max_date.normalize(), freq="D")
    return pd.DataFrame({
        "date_id": dates.strftime("%Y%m%d").astype(int),
        "date": dates.strftime("%Y-%m-%d"),
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "day_of_week": dates.dayofweek,
        "month_name": dates.strftime("%B"),
        "quarter": dates.quarter,
        "is_weekend": (dates.dayofweek >= 5).astype(int),
    })


# --------------------------------------------------------------------------
# Dimension ingest
# --------------------------------------------------------------------------

def ingest_aircraft(conn: sqlite3.Connection, log: RunLog) -> set[str]:
    raw = pd.read_csv(RAW_DIR / "aircraft.csv")
    n_in = len(raw)
    good_rows, valid_tails = [], set()

    for row in raw.to_dict("records"):
        tail = row.get("aircraft_tail")
        if not tail or pd.isna(tail):
            log.quarantine("aircraft", str(row), "missing_aircraft_tail", "aircraft_tail is null", row)
            continue
        entered = parse_mixed_date(row.get("entered_service_date")) or \
            pd.to_datetime(row.get("entered_service_date"), errors="coerce")
        if pd.isna(entered):
            log.quarantine("aircraft", tail, "unparseable_date", "entered_service_date", row)
            continue
        base = row.get("base_code")
        if base not in BASE_NAMES:  # already canonical in this source
            log.quarantine("aircraft", tail, "unknown_base_code", str(base), row)
            continue
        good_rows.append({
            "aircraft_tail": tail,
            "base_code": base,
            "model": row.get("model"),
            "entered_service_date": entered.strftime("%Y-%m-%d"),
            "monthly_flight_hours": row.get("monthly_flight_hours"),
            "batch_id": BATCH_ID,
        })
        valid_tails.add(tail)

    df = pd.DataFrame(good_rows)
    log.stage("dim_aircraft", "validate", n_in, len(df))

    dim_base_df = pd.DataFrame(
        [{"base_code": c, "base_name": n} for c, n in BASE_NAMES.items()]
    )
    with conn:
        conn.execute("DELETE FROM dim_base")
        conn.executemany("INSERT INTO dim_base VALUES (?, ?)",
                          dim_base_df.itertuples(index=False, name=None))
    idempotent_load(conn, "dim_aircraft", df, BATCH_ID)
    return valid_tails


def ingest_components(conn: sqlite3.Connection, log: RunLog, valid_tails: set[str]) -> set[str]:
    raw = pd.read_csv(RAW_DIR / "components.csv")
    n_in = len(raw)
    good_rows, valid_ids = [], set()

    for row in raw.to_dict("records"):
        cid = row.get("component_id")
        tail = row.get("aircraft_tail")
        cls = row.get("component_class")
        install = parse_mixed_date(row.get("install_date")) or \
            pd.to_datetime(row.get("install_date"), errors="coerce")

        if not cid or pd.isna(cid):
            log.quarantine("components", str(row), "missing_component_id", "component_id is null", row)
            continue
        if cls not in VALID_COMPONENT_CLASSES:
            log.quarantine("components", cid, "invalid_component_class", str(cls), row)
            continue
        if tail not in valid_tails:
            log.quarantine("components", cid, "orphan_aircraft_reference", str(tail), row)
            continue
        if pd.isna(install):
            log.quarantine("components", cid, "unparseable_date", "install_date", row)
            continue

        good_rows.append({
            "component_id": cid, "aircraft_tail": tail, "component_class": cls,
            "install_date": install.strftime("%Y-%m-%d"), "batch_id": BATCH_ID,
        })
        valid_ids.add(cid)

    df = pd.DataFrame(good_rows)
    log.stage("dim_component", "validate", n_in, len(df))
    idempotent_load(conn, "dim_component", df, BATCH_ID)
    return valid_ids


# --------------------------------------------------------------------------
# Fact ingest: events
# --------------------------------------------------------------------------

def ingest_events(conn: sqlite3.Connection, log: RunLog, valid_tails: set[str], valid_components: set[str]):
    raw = pd.read_csv(RAW_DIR / "events.csv")
    n_in = len(raw)

    records = raw.to_dict("records")
    validated = []
    for row in records:
        eid = row.get("event_id")
        cid = row.get("component_id")
        tail = row.get("aircraft_tail")
        cls = row.get("component_class")

        if not eid or pd.isna(eid):
            log.quarantine("events", str(row), "missing_event_id", "event_id is null", row)
            continue

        opened = parse_mixed_date(row.get("opened_at"))
        if opened is None:
            log.quarantine("events", eid, "unparseable_date", f"opened_at={row.get('opened_at')!r}", row)
            continue
        closed_raw = row.get("closed_at")
        closed = None
        if isinstance(closed_raw, str) and closed_raw.strip():
            closed = parse_mixed_date(closed_raw)
            if closed is None:
                log.quarantine("events", eid, "unparseable_date", f"closed_at={closed_raw!r}", row)
                continue

        if tail not in valid_tails:
            log.quarantine("events", eid, "orphan_aircraft_reference", str(tail), row)
            continue
        if cid not in valid_components:
            log.quarantine("events", eid, "orphan_component_reference", str(cid), row)
            continue

        base_canon = canonicalize_base(row.get("base_code"))
        if base_canon is None:
            log.quarantine("events", eid, "unknown_base_code", str(row.get("base_code")), row)
            continue

        hours_since_overhaul = row.get("hours_since_overhaul")
        if pd.isna(hours_since_overhaul) or hours_since_overhaul < 0:
            log.quarantine("events", eid, "negative_hours_since_overhaul", str(hours_since_overhaul), row)
            continue

        if closed is not None and closed < opened:
            log.quarantine("events", eid, "closed_before_opened",
                            f"opened={opened} closed={closed}", row)
            continue

        event_type = row.get("event_type")
        if event_type not in VALID_EVENT_TYPES:
            log.quarantine("events", eid, "invalid_event_type", str(event_type), row)
            continue

        status = row.get("status")
        if status not in VALID_STATUS:
            log.quarantine("events", eid, "invalid_status", str(status), row)
            continue

        priority = row.get("priority_code")
        priority = priority if priority in VALID_PRIORITY_CODES else None

        duration_hours = (closed - opened).total_seconds() / 3600 if closed is not None else None
        exceeds_48h = None if closed is None else int(duration_hours > 48)

        record_loaded_at = parse_mixed_date(row.get("record_loaded_at"))

        validated.append({
            "event_id": eid,
            "component_id": cid,
            "aircraft_tail": tail,
            "base_code": base_canon,
            "date_id": int(opened.strftime("%Y%m%d")),
            "event_type": event_type,
            "priority_code": priority,
            "status": status,
            "opened_at": opened.isoformat(),
            "closed_at": closed.isoformat() if closed is not None else None,
            "hours_since_overhaul": float(hours_since_overhaul),
            "component_age_hours": row.get("component_age_hours"),
            "duration_hours": duration_hours,
            "staffing_level": row.get("staffing_level"),
            "parts_backorder": int(bool(row.get("parts_backorder"))) if not pd.isna(row.get("parts_backorder")) else None,
            "technician_id": row.get("technician_id") if not pd.isna(row.get("technician_id")) else None,
            "notes": row.get("notes") if not pd.isna(row.get("notes")) else None,
            "record_loaded_at": record_loaded_at.isoformat() if record_loaded_at is not None else None,
            "exceeds_48h": exceeds_48h,
            "batch_id": BATCH_ID,
            "component_class": cls,  # used for date-range calc only, dropped before load
        })

    n_after_validate = len(validated)
    log.stage("fact_maintenance_events", "validate", n_in, n_after_validate)

    df = pd.DataFrame(validated)
    if df.empty:
        return

    # dedupe on event_id, keep the row with the latest record_loaded_at
    n_before_dedupe = len(df)
    df = df.sort_values("record_loaded_at").drop_duplicates(subset="event_id", keep="last")
    log.stage("fact_maintenance_events", "dedupe", n_before_dedupe, len(df))

    min_date = pd.to_datetime(df["opened_at"]).min()
    max_date = pd.to_datetime(df["opened_at"]).max()
    dim_date_df = build_dim_date(min_date, max_date)
    with conn:
        conn.execute("DELETE FROM dim_date")
        conn.executemany(
            "INSERT INTO dim_date VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            dim_date_df.itertuples(index=False, name=None),
        )

    df = df.drop(columns=["component_class"])
    idempotent_load(conn, "fact_maintenance_events", df, BATCH_ID)


# --------------------------------------------------------------------------
# Auxiliary sources: flights.json, inventory.dat
# --------------------------------------------------------------------------

def ingest_flight_hours(conn: sqlite3.Connection, log: RunLog, valid_tails: set[str]):
    records = json.loads((RAW_DIR / "flights.json").read_text())
    n_in = len(records)
    good = []
    for r in records:
        tail = r.get("aircraft_tail")
        if tail not in valid_tails:
            log.quarantine("flights", tail, "orphan_aircraft_reference", str(tail), r)
            continue
        if r.get("flight_hours") is None or r["flight_hours"] < 0:
            log.quarantine("flights", tail, "negative_flight_hours", str(r.get("flight_hours")), r)
            continue
        month_start = pd.to_datetime(r["month_start"], unit="ms").strftime("%Y-%m-%d")
        good.append({
            "aircraft_tail": tail, "month_start": month_start,
            "flight_hours": r["flight_hours"], "batch_id": BATCH_ID,
        })
    df = pd.DataFrame(good)
    log.stage("fact_flight_hours", "validate", n_in, len(df))
    idempotent_load(conn, "fact_flight_hours", df, BATCH_ID)


def parse_inventory_fixed_width(path: Path) -> list[dict]:
    # base_code(10) snapshot_date(10, MM/DD/YYYY) part_category(24) qty_available(8) qty_backorder(8)
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append({
                "base_code": line[0:10].strip(),
                "snapshot_date": line[10:20].strip(),
                "part_category": line[20:44].strip(),
                "qty_available": line[44:52].strip(),
                "qty_backorder": line[52:60].strip(),
            })
    return rows


def ingest_inventory(conn: sqlite3.Connection, log: RunLog):
    records = parse_inventory_fixed_width(RAW_DIR / "inventory.dat")
    n_in = len(records)
    good = []
    for r in records:
        base_canon = canonicalize_base(r["base_code"]) or (
            r["base_code"] if r["base_code"] in BASE_NAMES else None
        )
        if base_canon is None:
            log.quarantine("inventory", r["base_code"], "unknown_base_code", r["base_code"], r)
            continue
        snap_date = pd.to_datetime(r["snapshot_date"], format="%m/%d/%Y", errors="coerce")
        if pd.isna(snap_date):
            log.quarantine("inventory", r["base_code"], "unparseable_date", r["snapshot_date"], r)
            continue
        try:
            qty_avail = int(r["qty_available"])
            qty_back = int(r["qty_backorder"])
        except ValueError:
            log.quarantine("inventory", r["base_code"], "non_numeric_quantity",
                            f"{r['qty_available']}/{r['qty_backorder']}", r)
            continue
        good.append({
            "base_code": base_canon, "snapshot_date": snap_date.strftime("%Y-%m-%d"),
            "part_category": r["part_category"], "qty_available": qty_avail,
            "qty_backorder": qty_back, "batch_id": BATCH_ID,
        })
    df = pd.DataFrame(good)
    log.stage("fact_inventory_snapshot", "validate", n_in, len(df))
    idempotent_load(conn, "fact_inventory_snapshot", df, BATCH_ID)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    log = RunLog(BATCH_ID)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = OFF")  # SQLite FK checks off during load order; enforced by app logic
    init_schema(conn)

    print("Ingesting dim_aircraft...")
    valid_tails = ingest_aircraft(conn, log)

    print("Ingesting dim_component...")
    valid_components = ingest_components(conn, log, valid_tails)

    print("Ingesting fact_maintenance_events...")
    ingest_events(conn, log, valid_tails, valid_components)

    print("Ingesting fact_flight_hours...")
    ingest_flight_hours(conn, log, valid_tails)

    print("Ingesting fact_inventory_snapshot...")
    ingest_inventory(conn, log)

    # persist quarantine + metrics
    with conn:
        conn.execute("DELETE FROM quarantine WHERE batch_id = ?", (BATCH_ID,))
        if log.quarantine_rows:
            q_df = pd.DataFrame(log.quarantine_rows)
            conn.executemany(
                "INSERT INTO quarantine (source_table, record_key, rule_broken, detail, "
                "raw_row_json, quarantined_at, batch_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                q_df[["source_table", "record_key", "rule_broken", "detail",
                      "raw_row_json", "quarantined_at", "batch_id"]].itertuples(index=False, name=None),
            )
        conn.execute("DELETE FROM data_quality_metrics WHERE batch_id = ?", (BATCH_ID,))
        m_df = pd.DataFrame(log.metrics_rows)
        conn.executemany(
            "INSERT INTO data_quality_metrics (batch_id, stage, table_name, rows_in, rows_out, "
            "rows_quarantined, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            m_df.itertuples(index=False, name=None),
        )

    print(f"\nTotal quarantined rows: {len(log.quarantine_rows)}")
    if log.quarantine_rows:
        q_df = pd.DataFrame(log.quarantine_rows)
        print(q_df["rule_broken"].value_counts())

    conn.close()
    print(f"\nWarehouse written to {DB_PATH}")


if __name__ == "__main__":
    main()
