# Aircraft Sustainment Analytics Platform (thin slice)

An end-to-end analytics pipeline for fleet maintenance: synthetic data
generation → validated SQLite warehouse → shared feature engineering →
a model that predicts which open maintenance events are at risk of running
past 48 hours → a Streamlit app for planners.

## Pipeline

```
src/generator.py  -->  data/raw/*.csv,*.json,*.dat
src/ingest.py     -->  data/warehouse.db   (star schema + quarantine + DQ metrics)
src/train.py      -->  models/model.pkl, models/metadata.json
src/score.py      -->  data/warehouse.db (scored_events table)
streamlit_app.py  -->  reads warehouse.db + model.pkl
```

Run in order from the repo root (with the venv activated):

```bash
python src/generator.py
python src/ingest.py
python src/train.py
python src/score.py
streamlit run streamlit_app.py
```

Each step is idempotent: re-running `ingest.py` reproduces the same
warehouse state (delete-then-insert by a fixed batch id, in a transaction),
and `score.py` fully replaces the `scored_events` table each run.

## Data generation

3 years, 200 aircraft, 2,000 components across 5 classes, 50,000 maintenance
events. Component wear is a **Weibull renewal process** in flight-hours
(k=2.5 for wear-out items like hydraulic pumps/landing gear, k≈1.1–1.8 for
the more random-failure classes), so `hours_since_overhaul` is a genuine
leading indicator rather than a random label. Repair duration is lognormal,
shifted by degradation (`hours_since_overhaul`), base staffing, and parts
backorder status — the last of which is a real driver of long repairs that
is **deliberately not exposed as a model feature**, since it isn't known at
ticket creation. That keeps the prediction problem honest: real signal,
genuinely incomplete.

Event volume is dominated by frequent minor write-ups (`Routine`, Poisson
process per component) rather than rare full overhauls (`Unscheduled`,
the Weibull renewal points) — this matches real maintenance logs, where
most tickets are short discrepancies, not major repairs.

Raw files carry deliberate messiness that `ingest.py` has to clean up:
~2% missing values in non-critical fields, three different date formats
mixed across rows, inconsistent base codes (`"SD"` / `"San Diego"` / `"sd"`),
a small number of impossible values (negative hours, closure before
opening), and ~1% duplicate event rows (simulating a late re-extract, with a
later `record_loaded_at` so "keep latest" has a real signal to act on).

## Ingest

Per source file: parse → coerce types (logging unparseable dates rather than
silently dropping them) → canonicalize categories (base code crosswalk) →
validate business rules → dedupe on `event_id` keeping the latest
`record_loaded_at` → referential integrity check against `dim_aircraft` /
`dim_component` → idempotent load. Rows that fail any rule go to
`quarantine` with the specific rule they broke; `data_quality_metrics`
logs rows-in/rows-out per stage so the funnel is auditable.

## Star schema

`fact_maintenance_events` with foreign keys to `dim_component`,
`dim_aircraft`, `dim_base`, `dim_date`, plus `quarantine` and
`data_quality_metrics`. `flights.json` and `inventory.dat` are validated
and loaded into `fact_flight_hours` / `fact_inventory_snapshot` the same
way, though the model itself doesn't need them — `component_age_hours` and
`hours_since_overhaul` are already computed at event-generation time.

## Features (`src/features.py`)

One module, imported by both `train.py` and `score.py`, so the two can
never drift apart. Only fields knowable **at event creation**:
`component_class`, `component_age_hours`, `hours_since_overhaul`,
`base_code`, `aircraft_tail`, `priority_code`, `month`, and
`prior_event_count` (a window-function count of that component's prior
events). The actual encoding (one-hot, scaling) lives inside the saved
sklearn `Pipeline`, not duplicated in a second module — that's the stronger
anti-drift guarantee.

## Model (`src/train.py`)

Target: **event exceeds 48 hours** from creation to closure. Still-open
events are excluded from training/evaluation (no ground truth yet) but are
exactly what `score.py` scores. Time-aware split — first 2 years train,
year 3 held out — no random shuffling, since that would leak future
maintenance patterns backward.

Two candidates trained on identical inputs, best PR-AUC on the held-out
year wins:

| model | PR-AUC | base rate | precision@top10% | recall@top10% |
|---|---|---|---|---|
| Logistic regression | **0.179** | 0.102 | 0.208 | 0.204 |
| Gradient boosting (HistGB) | 0.160 | 0.102 | 0.186 | 0.183 |

PR-AUC is the headline because the positive class is rare (~10%) — plain
accuracy would be meaningless here. ~1.75x lift over the base rate is
modest but real; the biggest omitted driver (parts backorder) genuinely
isn't observable at intake, so this is close to the ceiling for this
feature set. The operating threshold is set to flag the **top 10% of open
events** by predicted risk (planner capacity), adjustable in the app.

## Streamlit app

- **At-Risk Queue** — open events ranked by risk score, a capacity slider
  that resizes the flagged set, and top per-event drivers (an exact logit
  decomposition for the linear model).
- **Reliability Explorer** — failure rate by component class, repair
  duration median/P90 (not mean — the tail is the point), and a per-class
  Weibull fit against the observed failure-hours distribution.
- **Data Quality** — quarantine counts by rule, the ingest funnel by stage,
  and field completeness.

## Deploying to Streamlit Community Cloud

The warehouse (`data/warehouse.db`) and model artifact (`models/model.pkl`,
`models/metadata.json`) must be committed — Cloud's filesystem is
ephemeral, so nothing generated at runtime persists between deploys. Run
the full pipeline locally first, commit the artifacts, then point Cloud at
`streamlit_app.py`. All paths in the app resolve relative to the file's own
location, not the working directory, so this works regardless of where
Streamlit is launched from.
