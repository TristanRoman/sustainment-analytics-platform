"""
Synthetic data generator for the Aircraft Sustainment Analytics Platform.

Simulates 3 years of fleet maintenance activity for 200 aircraft / ~2,000
components across 5 component classes, and emits deliberately messy raw
source files (events.csv, components.csv, aircraft.csv, flights.json,
inventory.dat) that downstream ingest logic must clean up.

Design notes:
- Component wear is modeled as a Weibull renewal process in flight-hours.
  Each renewal ("overhaul") resets the hours-since-overhaul clock, which is
  what makes hours_since_overhaul a genuine leading indicator of failure risk
  rather than a random label.
- Repair duration is lognormal; the location parameter (mu) is shifted by
  component degradation, base staffing, and parts availability so the >48h
  tail has real structure instead of being pure noise.
- All "dirtiness" (missing values, mixed date formats, inconsistent base
  codes, impossible values, duplicate rows) is applied only to the raw
  output representation, never to the internal clean simulation, so we can
  verify the simulation is sound before corrupting it.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 42
N_AIRCRAFT = 200
YEARS = 3
WINDOW_END = pd.Timestamp("2026-08-11")
WINDOW_START = WINDOW_END - pd.DateOffset(years=YEARS)
N_EVENTS_TARGET = 50_000

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Bases: canonical code -> (display name, staffing level 0-1, messy aliases)
BASES = {
    "SD": {"name": "San Diego", "staffing": 0.90,
           "aliases": ["SD", "San Diego", "sd ", "SAN DIEGO"]},
    "NORVA": {"name": "Norfolk", "staffing": 0.75,
              "aliases": ["NORVA", "Norfolk", "norva", "NORFOLK VA"]},
    "JAX": {"name": "Jacksonville", "staffing": 0.85,
            "aliases": ["JAX", "Jacksonville", "jax", "Jax "]},
    "LEMOORE": {"name": "Lemoore", "staffing": 0.70,
                "aliases": ["LEMOORE", "Lemoore", "NAS Lemoore", "lemoore"]},
    "WHIDBEY": {"name": "Whidbey Island", "staffing": 0.80,
                "aliases": ["WHIDBEY", "Whidbey Island", "whidbey", "WHIDBEY ISL"]},
    "OCEANA": {"name": "Oceana", "staffing": 0.65,
               "aliases": ["OCEANA", "Oceana", "oceana ", "NAS OCEANA"]},
}
BASE_CODES = list(BASES.keys())

AIRCRAFT_MODELS = ["F/A-18E", "P-8A", "MH-60R", "E-2D"]

# Component class specs: Weibull shape/scale (flight hours), count per
# aircraft, and lognormal duration base params.
COMPONENT_CLASSES = {
    "hydraulic_pump": {"k": 2.5, "scale": 6000, "per_aircraft": 2,
                        "duration_sigma": 0.55, "inspection_interval": 600,
                        "routine_rate_per_hour": 0.012},
    "actuator": {"k": 1.8, "scale": 7500, "per_aircraft": 4,
                 "duration_sigma": 0.60, "inspection_interval": 500,
                 "routine_rate_per_hour": 0.018},
    "avionics_module": {"k": 1.1, "scale": 8000, "per_aircraft": 2,
                         "duration_sigma": 0.65, "inspection_interval": 800,
                         "routine_rate_per_hour": 0.020},
    "landing_gear": {"k": 2.5, "scale": 9000, "per_aircraft": 1,
                      "duration_sigma": 0.50, "inspection_interval": 700,
                      "routine_rate_per_hour": 0.008},
    "environmental_control": {"k": 1.6, "scale": 5500, "per_aircraft": 1,
                               "duration_sigma": 0.60, "inspection_interval": 550,
                               "routine_rate_per_hour": 0.014},
}

PRIORITY_CODES = ["Routine", "Priority", "AOG"]

MONTH_SEASONAL_WEIGHT = {
    1: 0.85, 2: 0.85, 3: 0.95, 4: 1.05, 5: 1.25, 6: 1.35,
    7: 1.40, 8: 1.35, 9: 1.15, 10: 0.95, 11: 0.85, 12: 0.80,
}

rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# Dimension generation
# --------------------------------------------------------------------------

def generate_aircraft() -> pd.DataFrame:
    tails = [f"AC{1000 + i}" for i in range(N_AIRCRAFT)]
    base_codes = rng.choice(BASE_CODES, size=N_AIRCRAFT)
    models = rng.choice(AIRCRAFT_MODELS, size=N_AIRCRAFT)
    # entered service between 5 years before window start and window start,
    # so components can have accumulated meaningful hours before year 1
    entered_service = WINDOW_START - pd.to_timedelta(
        rng.integers(200, 1800, size=N_AIRCRAFT), unit="D"
    )
    monthly_rate = np.clip(rng.normal(60, 14, size=N_AIRCRAFT), 20, 100)

    return pd.DataFrame({
        "aircraft_tail": tails,
        "base_code": base_codes,
        "model": models,
        "entered_service_date": entered_service,
        "monthly_flight_hours": monthly_rate.round(1),
    })


def generate_components(aircraft_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comp_seq = 0
    for _, ac in aircraft_df.iterrows():
        for cls, spec in COMPONENT_CLASSES.items():
            for _ in range(spec["per_aircraft"]):
                comp_seq += 1
                # install date: same as aircraft entry, with small jitter
                # for components that were swapped in pre-window
                install_date = ac["entered_service_date"] + pd.to_timedelta(
                    int(rng.integers(0, 60)), unit="D"
                )
                rows.append({
                    "component_id": f"CMP{comp_seq:06d}",
                    "aircraft_tail": ac["aircraft_tail"],
                    "component_class": cls,
                    "install_date": install_date,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Event simulation (clean, internal representation)
# --------------------------------------------------------------------------

def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (end - start).days / 30.44


def _component_renewal_points(comp, rate: float, spec: dict) -> list[dict]:
    """Weibull renewal process: each point is a real overhaul-triggering
    failure ("Unscheduled" event) that resets the hours-since-overhaul
    clock. Returns points within the observation window."""
    points = []
    t = comp.install_date
    cumulative_hours = 0.0
    while True:
        hours_to_failure = rng.weibull(spec["k"]) * spec["scale"]
        months_needed = hours_to_failure / rate
        event_date = t + pd.to_timedelta(months_needed * 30.44, unit="D")
        cumulative_hours += hours_to_failure
        if event_date > WINDOW_END:
            break
        points.append({
            "opened_at": event_date,
            "hours_since_overhaul": hours_to_failure,
            "component_age_hours": cumulative_hours,
        })
        t = event_date
    return points


def _periodic_events_in_segment(comp, rate: float, seg_start_date, seg_start_age: float,
                                 seg_length_hours: float, interval_hours: float) -> list[dict]:
    """Fixed-interval events (formal scheduled inspections) within one
    inter-overhaul segment."""
    events = []
    t_hours = interval_hours * rng.uniform(0.5, 1.0)
    while t_hours < seg_length_hours:
        event_date = seg_start_date + pd.to_timedelta(t_hours / rate * 30.44, unit="D")
        events.append({
            "opened_at": event_date,
            "hours_since_overhaul": t_hours,
            "component_age_hours": seg_start_age + t_hours,
        })
        t_hours += interval_hours * rng.uniform(0.85, 1.15)
    return events


def _routine_events_in_segment(comp, rate: float, seg_start_date, seg_start_age: float,
                                seg_length_hours: float, hourly_rate: float) -> list[dict]:
    """Poisson process of minor write-ups/discrepancies -- the volume driver
    of the log, in contrast to the rare, high-signal Unscheduled overhauls."""
    events = []
    t_hours = rng.exponential(1.0 / hourly_rate)
    while t_hours < seg_length_hours:
        event_date = seg_start_date + pd.to_timedelta(t_hours / rate * 30.44, unit="D")
        events.append({
            "opened_at": event_date,
            "hours_since_overhaul": t_hours,
            "component_age_hours": seg_start_age + t_hours,
        })
        t_hours += rng.exponential(1.0 / hourly_rate)
    return events


def build_event_pool(components_df: pd.DataFrame,
                      aircraft_df: pd.DataFrame) -> pd.DataFrame:
    """Per-component simulation built around overhaul segments, so that
    hours_since_overhaul is consistent across all three event types:
    Unscheduled (segment-ending overhaul), Scheduled (fixed-interval
    inspection), and Routine (Poisson-process minor write-ups)."""
    ac_rate = aircraft_df.set_index("aircraft_tail")["monthly_flight_hours"]
    events = []

    for comp in components_df.itertuples():
        spec = COMPONENT_CLASSES[comp.component_class]
        rate = ac_rate[comp.aircraft_tail]

        renewals = _component_renewal_points(comp, rate, spec)
        for r in renewals:
            events.append({
                "component_id": comp.component_id, "aircraft_tail": comp.aircraft_tail,
                "component_class": comp.component_class, "event_type": "Unscheduled",
                **r,
            })

        seg_start_date, seg_start_age = comp.install_date, 0.0
        boundaries = renewals + [{"opened_at": WINDOW_END, "component_age_hours": None}]
        for b in boundaries:
            seg_end_date = b["opened_at"]
            seg_length_hours = _months_between(seg_start_date, seg_end_date) * rate
            if seg_length_hours > 0:
                for ev in _periodic_events_in_segment(
                    comp, rate, seg_start_date, seg_start_age,
                    seg_length_hours, spec["inspection_interval"],
                ):
                    events.append({
                        "component_id": comp.component_id, "aircraft_tail": comp.aircraft_tail,
                        "component_class": comp.component_class, "event_type": "Scheduled",
                        **ev,
                    })
                for ev in _routine_events_in_segment(
                    comp, rate, seg_start_date, seg_start_age,
                    seg_length_hours, spec["routine_rate_per_hour"],
                ):
                    events.append({
                        "component_id": comp.component_id, "aircraft_tail": comp.aircraft_tail,
                        "component_class": comp.component_class, "event_type": "Routine",
                        **ev,
                    })
            seg_start_date = seg_end_date
            seg_start_age = b["component_age_hours"] if b["component_age_hours"] is not None \
                else seg_start_age + seg_length_hours

    df = pd.DataFrame(events)
    df["opened_at"] = pd.to_datetime(df["opened_at"])
    df["hours_since_overhaul"] = df["hours_since_overhaul"].round(1)
    df["component_age_hours"] = df["component_age_hours"].round(1)
    df = df[(df["opened_at"] >= WINDOW_START) & (df["opened_at"] <= WINDOW_END)]
    return df.reset_index(drop=True)


def apply_seasonal_thinning(pool: pd.DataFrame, target_n: int) -> pd.DataFrame:
    weights = pool["opened_at"].dt.month.map(MONTH_SEASONAL_WEIGHT).to_numpy()
    weights = weights / weights.sum()
    n = min(target_n, len(pool))
    idx = rng.choice(pool.index.to_numpy(), size=n, replace=False, p=weights)
    return pool.loc[idx].sort_values("opened_at").reset_index(drop=True)


# --------------------------------------------------------------------------
# Duration model
# --------------------------------------------------------------------------

def _calibrate_mu(sigma: float, target_p_over_48: float = 0.085) -> float:
    """Solve mu for lognormal(mu, sigma) s.t. P(X > 48) = target_p_over_48."""
    from scipy.stats import norm
    z = norm.ppf(1 - target_p_over_48)
    return np.log(48) - z * sigma


# Covariate coefficients and base >48h tail rate by event_type. Unscheduled
# repairs (genuine component failures) are the population most exposed to
# degradation/staffing/parts effects; Routine write-ups are usually quick but
# still occasionally escalate; Scheduled inspections use a separate short
# fixed distribution below and don't go through this covariate model at all.
# hours/staffing coefficients are deliberately strong -- these map directly to
# allowed model features (hours_since_overhaul, base) -- while parts_backorder
# stays a real but *unobservable* confounder (not an allowed feature), so the
# trained model has genuine signal without being perfectly separable.
DURATION_MODEL = {
    "Unscheduled": {"hours": 1.1, "staffing": 0.7, "backorder": 0.5, "target": 0.30},
    "Routine": {"hours": 0.85, "staffing": 0.55, "backorder": 0.3, "target": 0.0885},
}

# Per-class multiplier on the Routine/Unscheduled base tail rate, giving
# component_class its own genuine (not just sigma-driven) predictive signal.
CLASS_TAIL_MULTIPLIER = {
    "hydraulic_pump": 0.9, "actuator": 1.0, "avionics_module": 0.7,
    "landing_gear": 1.3, "environmental_control": 1.1,
}


def assign_duration_and_context(events_df: pd.DataFrame,
                                 aircraft_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.copy()
    n = len(df)

    staffing = aircraft_df.set_index("aircraft_tail")["base_code"].map(
        lambda b: BASES[b]["staffing"]
    )
    ac_base = aircraft_df.set_index("aircraft_tail")["base_code"]
    df["base_code"] = df["aircraft_tail"].map(ac_base)
    df["staffing_level"] = df["aircraft_tail"].map(staffing)
    df["parts_backorder"] = rng.random(n) < 0.15

    scale_map = {cls: spec["scale"] for cls, spec in COMPONENT_CLASSES.items()}
    norm_hours = (df["hours_since_overhaul"] / df["component_class"].map(scale_map)).to_numpy()
    staffing_gap = (1 - df["staffing_level"]).to_numpy()
    backorder = df["parts_backorder"].to_numpy().astype(float)

    duration_hours = np.empty(n)
    sigma_arr = df["component_class"].map(
        {cls: spec["duration_sigma"] for cls, spec in COMPONENT_CLASSES.items()}
    ).to_numpy()

    for event_type, coefs in DURATION_MODEL.items():
        type_mask = (df["event_type"] == event_type).to_numpy()
        s = coefs["hours"] * norm_hours + coefs["staffing"] * staffing_gap + coefs["backorder"] * backorder
        for cls, spec in COMPONENT_CLASSES.items():
            m = type_mask & (df["component_class"] == cls).to_numpy()
            if not m.any():
                continue
            class_target = coefs["target"] * CLASS_TAIL_MULTIPLIER[cls]
            mu0 = _calibrate_mu(spec["duration_sigma"], class_target)
            # recenter so the group's *average* covariate shift lands the
            # group at its target tail rate, while per-row shifts still vary
            mu_row = mu0 - s[m].mean() + s[m]
            duration_hours[m] = rng.lognormal(mean=mu_row, sigma=sigma_arr[m])

    # formal scheduled inspections: short, predictable, minimal tail
    scheduled_mask = (df["event_type"] == "Scheduled").to_numpy()
    duration_hours[scheduled_mask] = rng.lognormal(
        mean=np.log(4.0), sigma=0.5, size=scheduled_mask.sum()
    )

    df["duration_hours"] = np.round(duration_hours, 2)
    return df


def assign_priority(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # priority correlates loosely with event_type and class criticality
    p_aog = df["event_type"].map({"Unscheduled": 0.18, "Routine": 0.06, "Scheduled": 0.01}).to_numpy()
    p_priority = df["event_type"].map({"Unscheduled": 0.35, "Routine": 0.22, "Scheduled": 0.10}).to_numpy()
    r = rng.random(len(df))
    priority = np.select(
        [r < p_aog, r < p_aog + p_priority],
        ["AOG", "Priority"],
        default="Routine",
    )
    df["priority_code"] = priority
    return df


def finalize_close_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["closed_at"] = df["opened_at"] + pd.to_timedelta(df["duration_hours"], unit="h")
    # events whose close would fall after the observation window are still open
    still_open = df["closed_at"] > WINDOW_END
    df.loc[still_open, "closed_at"] = pd.NaT
    df["status"] = np.where(still_open, "Open", "Closed")
    return df


def assign_event_ids_and_misc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("opened_at").reset_index(drop=True)
    df["event_id"] = [f"EVT{100000 + i}" for i in range(len(df))]
    technicians = [f"TECH{i:03d}" for i in range(1, 61)]
    df["technician_id"] = rng.choice(technicians, size=len(df))
    df["notes"] = rng.choice(
        ["", "Recurring discrepancy", "Parts on hand", "Follow-up required",
         "No fault found", "AD compliance"],
        size=len(df),
        p=[0.55, 0.10, 0.10, 0.10, 0.10, 0.05],
    )
    df["record_loaded_at"] = df["opened_at"] + pd.to_timedelta(
        rng.integers(1, 72, size=len(df)), unit="h"
    )
    return df


# --------------------------------------------------------------------------
# flights.json and inventory.dat (auxiliary sources)
# --------------------------------------------------------------------------

def generate_flight_hours(aircraft_df: pd.DataFrame) -> list[dict]:
    records = []
    months = pd.date_range(WINDOW_START, WINDOW_END, freq="MS")
    for ac in aircraft_df.itertuples():
        for m in months:
            noise = rng.normal(1.0, 0.12)
            hours = max(0.0, ac.monthly_flight_hours * noise)
            records.append({
                "aircraft_tail": ac.aircraft_tail,
                "month_start": int(m.timestamp() * 1000),  # epoch millis
                "flight_hours": round(hours, 1),
            })
    return records


def generate_inventory_snapshots() -> pd.DataFrame:
    rows = []
    weeks = pd.date_range(WINDOW_START, WINDOW_END, freq="W-MON")
    for base_code, spec in BASES.items():
        for wk in weeks:
            for cls in COMPONENT_CLASSES:
                avail_pct = np.clip(rng.normal(spec["staffing"], 0.1), 0.3, 1.0)
                qty_available = int(rng.integers(5, 40) * avail_pct)
                qty_backorder = int(rng.integers(0, 6) * (1 - avail_pct) * 3)
                rows.append({
                    "base_code": base_code,
                    "snapshot_date": wk,
                    "part_category": cls,
                    "qty_available": qty_available,
                    "qty_backorder": qty_backorder,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Dirtiness injection (applied only to output representation)
# --------------------------------------------------------------------------

NON_CRITICAL_FIELDS = ["technician_id", "notes", "priority_code"]


def dirty_events_for_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    # duplicate rows (~1%): a late-arriving re-extract with a status/notes
    # update and a later record_loaded_at, so "dedupe keeping latest" has a
    # genuine timestamp to key off. Done first, on real Timestamps, so the
    # bumped record_loaded_at flows through the date formatting below.
    dup_idx = rng.choice(out.index, size=int(n * 0.01), replace=False)
    dups = out.loc[dup_idx].copy()
    dups["notes"] = "Status updated on re-extract"
    dups["record_loaded_at"] = dups["record_loaded_at"] + pd.to_timedelta(
        rng.integers(6, 96, size=len(dups)), unit="h"
    )
    out = pd.concat([out, dups], ignore_index=True)
    n = len(out)

    # ~2% missing values in non-critical fields
    for col in NON_CRITICAL_FIELDS:
        mask = rng.random(n) < 0.02
        out.loc[mask, col] = np.nan

    # a few impossible values, injected on real Timestamps before formatting
    n_bad = max(1, int(n * 0.003))
    neg_idx = rng.choice(out.index, size=n_bad // 2, replace=False)
    out.loc[neg_idx, "hours_since_overhaul"] = -out.loc[neg_idx, "hours_since_overhaul"].abs()

    closed_pool = out[out["closed_at"].notna()].index
    bad_close_idx = rng.choice(closed_pool, size=min(n_bad // 2, len(closed_pool)), replace=False)
    out.loc[bad_close_idx, "closed_at"] = (
        out.loc[bad_close_idx, "opened_at"]
        - pd.to_timedelta(rng.integers(1, 6, size=len(bad_close_idx)), unit="h")
    )

    # mixed date formats: split opened_at/closed_at/record_loaded_at across
    # a few plausible source formats
    def mixed_format(ts_series: pd.Series) -> pd.Series:
        fmt_choice = rng.integers(0, 3, size=len(ts_series))
        out_vals = []
        for ts, fmt in zip(ts_series, fmt_choice):
            if pd.isna(ts):
                out_vals.append("")
            elif fmt == 0:
                out_vals.append(ts.strftime("%Y-%m-%d %H:%M:%S"))
            elif fmt == 1:
                out_vals.append(ts.strftime("%m/%d/%Y %H:%M"))
            else:
                out_vals.append(ts.strftime("%d-%b-%Y %H:%M:%S"))
        return pd.Series(out_vals, index=ts_series.index)

    out["opened_at"] = mixed_format(out["opened_at"])
    out["closed_at"] = mixed_format(out["closed_at"])
    out["record_loaded_at"] = mixed_format(out["record_loaded_at"])

    # inconsistent base codes: swap canonical code for a random alias
    def messy_base(code: str) -> str:
        return rng.choice(BASES[code]["aliases"])

    out["base_code"] = out["base_code"].map(messy_base)

    return out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def write_inventory_fixed_width(inv_df: pd.DataFrame, path: Path) -> None:
    # base_code(10) snapshot_date(10, MM/DD/YYYY) part_category(24)
    # qty_available(8) qty_backorder(8)
    with open(path, "w") as f:
        for row in inv_df.itertuples():
            date_str = row.snapshot_date.strftime("%m/%d/%Y")
            line = (
                f"{row.base_code:<10}"
                f"{date_str:<10}"
                f"{row.part_category:<24}"
                f"{row.qty_available:<8}"
                f"{row.qty_backorder:<8}"
            )
            f.write(line + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating aircraft...")
    aircraft_df = generate_aircraft()

    print("Generating components...")
    components_df = generate_components(aircraft_df)
    print(f"  {len(components_df)} components")

    print("Simulating maintenance event pool (Weibull renewal + scheduled inspections)...")
    pool = build_event_pool(components_df, aircraft_df)
    print(f"  raw pool: {len(pool)} events before seasonal thinning")

    print("Applying seasonal thinning to target volume...")
    events = apply_seasonal_thinning(pool, N_EVENTS_TARGET)

    print("Assigning duration, priority, and context...")
    events = assign_duration_and_context(events, aircraft_df)
    events = assign_priority(events)
    events = finalize_close_dates(events)
    events = assign_event_ids_and_misc(events)

    closed = events[events["status"] == "Closed"]
    pct_over_48 = (closed["duration_hours"] > 48).mean()
    print(f"  closed events: {len(closed)}, open events: {(events['status']=='Open').sum()}")
    print(f"  pct duration > 48h (closed only): {pct_over_48:.3%}")

    print("Generating flights.json...")
    flight_records = generate_flight_hours(aircraft_df)

    print("Generating inventory.dat...")
    inventory_df = generate_inventory_snapshots()

    print("Injecting dirtiness into events output...")
    dirty_events = dirty_events_for_output(events)

    # --- write outputs ---
    aircraft_out = aircraft_df.copy()
    aircraft_out["entered_service_date"] = aircraft_out["entered_service_date"].dt.strftime("%Y-%m-%d")
    aircraft_out.to_csv(RAW_DIR / "aircraft.csv", index=False)

    components_out = components_df.copy()
    components_out["install_date"] = components_out["install_date"].dt.strftime("%Y-%m-%d")
    components_out.to_csv(RAW_DIR / "components.csv", index=False)

    event_cols = [
        "event_id", "component_id", "aircraft_tail", "base_code", "component_class",
        "event_type", "priority_code", "status", "opened_at", "closed_at",
        "hours_since_overhaul", "component_age_hours", "duration_hours",
        "staffing_level", "parts_backorder", "technician_id", "notes",
        "record_loaded_at",
    ]
    dirty_events[event_cols].to_csv(RAW_DIR / "events.csv", index=False)

    with open(RAW_DIR / "flights.json", "w") as f:
        json.dump(flight_records, f)

    write_inventory_fixed_width(inventory_df, RAW_DIR / "inventory.dat")

    print("\nWrote raw files to", RAW_DIR)
    for p in ["aircraft.csv", "components.csv", "events.csv", "flights.json", "inventory.dat"]:
        fp = RAW_DIR / p
        print(f"  {p}: {fp.stat().st_size / 1024:.1f} KB")

    print("\nSanity checks:")
    print(f"  total event rows written (incl. duplicates): {len(dirty_events)}")
    print(f"  unique event_ids: {dirty_events['event_id'].nunique()}")
    print(f"  base_code raw value counts:\n{dirty_events['base_code'].value_counts()}")


if __name__ == "__main__":
    main()
