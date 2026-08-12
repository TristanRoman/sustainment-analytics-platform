"""
Aircraft Sustainment Analytics Platform -- Streamlit app.

Three pages:
  1. At-Risk Queue    -- events ranked by predicted risk of exceeding 48h,
                          with a planner-capacity slider and top drivers.
  2. Reliability Explorer -- failure rates, duration distributions
                          (median/P90), and per-class Weibull fits.
  3. Data Quality      -- quarantine counts by rule and field completeness.

All paths are resolved relative to this file, rather than the working
directory, because Streamlit Community Cloud doesn't guarantee the launch
directory matches the repo root -- a cwd-relative path would break there
even though it works locally. The SQLite warehouse and model artifact are
loaded as committed files, not regenerated at runtime, due to Cloud's
filesystem being ephemeral.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data" / "warehouse.db"
METADATA_PATH = APP_DIR / "models" / "metadata.json"

# These hex values come from the dataviz skill's reference palette, because
# it was already validated colorblind-safe in a fixed categorical order --
# picking colors ad hoc here would risk an unvalidated, non-CVD-safe result
SERIES = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948",
}
CLASS_COLORS = {
    "hydraulic_pump": SERIES["blue"], "actuator": SERIES["orange"],
    "avionics_module": SERIES["aqua"], "landing_gear": SERIES["yellow"],
    "environmental_control": SERIES["magenta"],
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
MUTED = "#898781"
GRID = "#e1e0d9"

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color="#0b0b0b"),
    plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    margin=dict(l=40, r=20, t=40, b=40),
)


@st.cache_resource
def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text())


@st.cache_data
def load_scored_events() -> pd.DataFrame:
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM scored_events ORDER BY risk_score DESC", conn)
    except pd.errors.DatabaseError:
        return pd.DataFrame()
    df["top_drivers"] = df["top_drivers_json"].apply(json.loads)
    return df


@st.cache_data
def load_events() -> pd.DataFrame:
    conn = get_conn()
    query = """
    SELECT e.*, c.component_class AS component_class_dim
    FROM fact_maintenance_events e
    JOIN dim_component c ON c.component_id = e.component_id
    """
    df = pd.read_sql_query(query, conn, parse_dates=["opened_at", "closed_at"])
    df["component_class"] = df["component_class_dim"]
    return df


@st.cache_data
def load_quarantine() -> pd.DataFrame:
    conn = get_conn()
    try:
        return pd.read_sql_query("SELECT * FROM quarantine", conn)
    except pd.errors.DatabaseError:
        return pd.DataFrame()


@st.cache_data
def load_dq_metrics() -> pd.DataFrame:
    conn = get_conn()
    try:
        return pd.read_sql_query("SELECT * FROM data_quality_metrics", conn)
    except pd.errors.DatabaseError:
        return pd.DataFrame()


def format_drivers(drivers: list[dict]) -> str:
    if not drivers:
        return "—"
    parts = []
    for d in drivers:
        sign = "+" if d["contribution"] >= 0 else "−"
        parts.append(f"{d['feature']} ({sign}{abs(d['contribution']):.2f})")
    return " · ".join(parts)


# --------------------------------------------------------------------------
# Page 1: At-Risk Queue
# --------------------------------------------------------------------------

def page_at_risk_queue():
    st.title("At-Risk Queue")
    st.caption("Open maintenance events ranked by predicted risk of exceeding 48 hours to close.")

    scored = load_scored_events()
    if scored.empty:
        st.warning("No scored events found. Run `src/score.py` to populate the scored_events table.")
        return

    metadata = load_metadata()
    n_open = len(scored)

    col1, col2, col3 = st.columns(3)
    col1.metric("Open events", n_open)
    default_pct = int(metadata.get("top_decile_capacity", 0.10) * 100)
    pct = st.slider("Planner capacity (% of open queue to flag)", min_value=1, max_value=100,
                     value=default_pct, step=1)
    n_flag = max(1, math.ceil(n_open * pct / 100))
    col2.metric("Flagged this run", n_flag)
    col3.metric("Model", metadata.get("selected_model", "n/a").replace("_", " ").title())

    queue = scored.head(n_flag).copy()
    queue.insert(0, "rank", range(1, len(queue) + 1))
    queue["top_drivers_str"] = queue["top_drivers"].apply(format_drivers)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scored["risk_score"], y=list(range(len(scored), 0, -1)),
        orientation="h", marker_color=MUTED, opacity=0.35, showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Bar(
        x=queue["risk_score"], y=list(range(len(scored), len(scored) - len(queue), -1)),
        orientation="h", marker_color=SERIES["blue"], name="Flagged",
        hovertemplate="%{customdata[0]}<br>risk score %{x:.3f}<extra></extra>",
        customdata=queue[["event_id"]].values,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=260, showlegend=False,
        xaxis_title="Risk score", yaxis=dict(showticklabels=False, title=None),
        title="Risk score distribution (flagged events in blue)",
    )
    fig.update_xaxes(gridcolor=GRID)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Top {n_flag} events")
    display_cols = {
        "rank": "Rank", "event_id": "Event", "aircraft_tail": "Tail",
        "component_class": "Component", "base_code": "Base",
        "priority_code": "Priority", "risk_score": "Risk score",
        "top_drivers_str": "Top drivers",
    }
    st.dataframe(
        queue[list(display_cols)].rename(columns=display_cols),
        use_container_width=True, hide_index=True,
        column_config={"Risk score": st.column_config.ProgressColumn(
            "Risk score", min_value=0, max_value=1, format="%.3f")},
    )


# --------------------------------------------------------------------------
# Page 2: Reliability Explorer
# --------------------------------------------------------------------------

def page_reliability_explorer():
    st.title("Reliability Explorer")
    st.caption("Failure behavior and repair-time distributions by component class.")

    events = load_events()
    n_components = pd.read_sql_query(
        "SELECT component_class, COUNT(*) AS n FROM dim_component GROUP BY component_class",
        get_conn(),
    ).set_index("component_class")["n"]

    obs_years = (events["opened_at"].max() - events["opened_at"].min()).days / 365.25

    st.subheader("Failure rate by component class")
    st.caption("Unscheduled (failure-driven) events per component per year.")
    unsched = events[events["event_type"] == "Unscheduled"]
    rate = (unsched.groupby("component_class").size() / n_components / obs_years).fillna(0)
    rate = rate.reindex(CLASS_COLORS.keys())

    fig = go.Figure(go.Bar(
        x=rate.index, y=rate.values,
        marker_color=[CLASS_COLORS[c] for c in rate.index],
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title="Failures / component / year", xaxis_title=None)
    fig.update_yaxes(gridcolor=GRID)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Repair duration: median and P90 (not mean)")
    st.caption("Mean is skewed by the long tail of backordered/degraded repairs; median and P90 tell the operational story.")
    closed = events[events["status"] == "Closed"]
    dur_stats = closed.groupby("component_class")["duration_hours"].agg(
        median="median", p90=lambda s: s.quantile(0.90)
    ).reindex(list(CLASS_COLORS.keys()))

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Median", x=dur_stats.index, y=dur_stats["median"], marker_color=SERIES["blue"]))
    fig.add_trace(go.Bar(name="P90", x=dur_stats.index, y=dur_stats["p90"], marker_color=SERIES["orange"]))
    fig.update_layout(**PLOTLY_LAYOUT, height=340, barmode="group",
                       yaxis_title="Duration (hours)", xaxis_title=None,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.add_hline(y=48, line_dash="dot", line_color=STATUS["critical"],
                  annotation_text="48h threshold", annotation_position="top left")
    fig.update_yaxes(gridcolor=GRID)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weibull fit: time between failures")
    st.caption("Fit to hours_since_overhaul at each Unscheduled (failure) event, by component class.")
    cls = st.selectbox("Component class", list(CLASS_COLORS.keys()))
    sample = unsched.loc[unsched["component_class"] == cls, "hours_since_overhaul"].dropna()
    sample = sample[sample > 0]

    if len(sample) < 10:
        st.info("Not enough failure events for this class to fit a distribution.")
    else:
        k, loc, scale = stats.weibull_min.fit(sample, floc=0)
        x = np.linspace(0, sample.quantile(0.995), 200)
        pdf = stats.weibull_min.pdf(x, k, loc=0, scale=scale)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=sample, histnorm="probability density", name="Observed",
            marker_color=CLASS_COLORS[cls], opacity=0.55, nbinsx=30,
        ))
        fig.add_trace(go.Scatter(
            x=x, y=pdf, name=f"Weibull fit (k={k:.2f}, λ={scale:.0f}h)",
            line=dict(color="#0b0b0b", width=2),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=360,
            xaxis_title="Hours since overhaul at failure", yaxis_title="Density",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_xaxes(gridcolor=GRID)
        fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig, use_container_width=True)
        shape_word = "wear-out (k>1)" if k > 1.05 else ("random-failure (k≈1)" if k > 0.95 else "infant-mortality (k<1)")
        st.caption(f"Fitted shape k={k:.2f} → {shape_word}; scale λ={scale:.0f} flight hours (characteristic life), n={len(sample)} failures.")


# --------------------------------------------------------------------------
# Page 3: Data Quality
# --------------------------------------------------------------------------

def page_data_quality():
    st.title("Data Quality")
    st.caption("What the ingest pipeline caught, and how complete the loaded data is.")

    quarantine = load_quarantine()
    dq = load_dq_metrics()
    events = load_events()

    col1, col2, col3 = st.columns(3)
    col1.metric("Quarantined rows", len(quarantine))
    col2.metric("Loaded events", len(events))
    reject_rate = len(quarantine) / (len(quarantine) + len(events)) if (len(quarantine) + len(events)) else 0
    col3.metric("Reject rate", f"{reject_rate:.2%}")

    st.subheader("Quarantine counts by rule")
    if quarantine.empty:
        st.info("No quarantined rows.")
    else:
        counts = quarantine["rule_broken"].value_counts().sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=counts.values, y=counts.index, orientation="h", marker_color=SERIES["blue"],
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=280, xaxis_title="Rows quarantined", yaxis_title=None)
        fig.update_xaxes(gridcolor=GRID)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pipeline funnel by stage")
    if dq.empty:
        st.info("No pipeline metrics recorded.")
    else:
        dq = dq.sort_values(["table_name", "stage"])
        st.dataframe(
            dq[["table_name", "stage", "rows_in", "rows_out", "rows_quarantined"]]
            .rename(columns={"table_name": "Table", "stage": "Stage", "rows_in": "In",
                              "rows_out": "Out", "rows_quarantined": "Quarantined"}),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Field completeness (loaded events)")
    non_critical = ["priority_code", "technician_id", "notes", "closed_at"]
    completeness = (1 - events[non_critical].isna().mean()).sort_values()
    fig = go.Figure(go.Bar(
        x=completeness.values, y=completeness.index, orientation="h", marker_color=SERIES["aqua"],
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=240, xaxis_title="% complete", xaxis_tickformat=".0%",
                       yaxis_title=None)
    fig.update_xaxes(gridcolor=GRID, range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("closed_at is expected to be null for still-open events, not a data quality defect.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Sustainment Analytics", layout="wide")
    st.sidebar.title("Aircraft Sustainment Analytics")
    if not DB_PATH.exists():
        st.error(f"Warehouse not found at {DB_PATH}. Run src/generator.py then src/ingest.py first.")
        return

    page = st.sidebar.radio(
        "Page", ["At-Risk Queue", "Reliability Explorer", "Data Quality"], label_visibility="collapsed"
    )
    metadata = load_metadata()
    if metadata:
        st.sidebar.caption(
            f"Model: {metadata.get('selected_model', 'n/a')}\n\n"
            f"Trained: {metadata.get('trained_at', 'n/a')[:10]}\n\n"
            f"Test PR-AUC: {metadata.get('metrics', {}).get(metadata.get('selected_model', ''), {}).get('pr_auc', 0):.3f}"
        )

    if page == "At-Risk Queue":
        page_at_risk_queue()
    elif page == "Reliability Explorer":
        page_reliability_explorer()
    else:
        page_data_quality()


if __name__ == "__main__":
    main()
