"""
Streamlit dashboard for the ERCOT anomaly detector.

Lets you import ERCOT demand data at any time - by uploading a CSV or pulling
fresh rows straight from the EIA API - and always shows the current archive
scored for anomalies. Data is persisted to data/ercot_demand.csv so it
accumulates across visits instead of resetting every reload.

Run locally:
    streamlit run dashboard.py

Deploy (Streamlit Community Cloud):
    1. Push this repo to GitHub (data/, outputs/, and .env are gitignored).
    2. On share.streamlit.io, point at dashboard.py.
    3. In the app's Secrets, optionally set EIA_API_KEY = "..." so visitors
       don't have to paste their own key to fetch fresh data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ercot_anomaly_detector as det

DATA_PATH = det.DATA_PATH
SAMPLE_PATH = Path("sample_data/ercot_demand_sample.csv")

st.set_page_config(page_title="ERCOT Anomaly Detector", layout="wide", page_icon="⚡")


# ==================================================================== data =
def normalize_uploaded(raw: pd.DataFrame) -> pd.DataFrame:
    """Accept either raw EIA export columns (period, value) or the loaded
    shape (timestamp, demand); return a standard period/value frame."""
    cols = {c.lower().strip(): c for c in raw.columns}
    if "period" in cols and "value" in cols:
        period_col, value_col = cols["period"], cols["value"]
    elif "timestamp" in cols and "demand" in cols:
        period_col, value_col = cols["timestamp"], cols["demand"]
    else:
        raise ValueError(
            "Unrecognized columns. Expected 'period' + 'value' (EIA raw export) "
            f"or 'timestamp' + 'demand'. Found: {list(raw.columns)}"
        )
    period = pd.to_datetime(raw[period_col], format="mixed", utc=True, errors="coerce")
    value = pd.to_numeric(raw[value_col], errors="coerce")
    out = pd.DataFrame({"period": period, "value": value}).dropna(subset=["period"])
    if out.empty:
        raise ValueError("No valid rows after parsing - check the date and value columns.")
    return out


def merge_into_archive(new_rows: pd.DataFrame) -> dict:
    """Merge new_rows into the on-disk archive, dedupe by hour, cap to
    ARCHIVE_DAYS, and write atomically so a crash mid-write can't corrupt it."""
    parts = [new_rows[["period", "value"]]]
    if DATA_PATH.exists():
        prior = pd.read_csv(DATA_PATH)
        prior["period"] = pd.to_datetime(prior["period"], utc=True)
        prior["value"] = pd.to_numeric(prior["value"], errors="coerce")
        parts.insert(0, prior[["period", "value"]])

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset="period", keep="last").sort_values("period")

    cap_days = getattr(det, "ARCHIVE_DAYS", 120)
    cutoff = merged["period"].max() - pd.Timedelta(days=cap_days)
    merged = merged[merged["period"] >= cutoff]

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_PATH.with_suffix(".tmp")
    merged.to_csv(tmp, index=False)
    tmp.replace(DATA_PATH)  # atomic on POSIX and Windows

    return {"imported": len(new_rows), "total": len(merged),
            "start": merged["period"].min(), "end": merged["period"].max()}


@st.cache_data(show_spinner="Scoring the archive...")
def score_archive(path_str: str, mtime: float) -> pd.DataFrame:
    loaded = det.load_demand(Path(path_str))
    return det.detect_anomalies(loaded)


# ============================================================== sidebar ===
st.sidebar.header("Import data")

with st.sidebar.expander("Upload a CSV", expanded=not DATA_PATH.exists()):
    st.caption("Columns: 'period' + 'value' (EIA raw export) or 'timestamp' + 'demand'.")
    uploaded = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    if uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        if st.session_state.get("_last_upload_sig") != sig:
            try:
                norm = normalize_uploaded(pd.read_csv(uploaded))
                stats = merge_into_archive(norm)
                st.session_state["_last_upload_sig"] = sig
                st.success(
                    f"Imported {stats['imported']:,} rows. Archive: {stats['total']:,} rows "
                    f"({stats['start']:%Y-%m-%d} to {stats['end']:%Y-%m-%d})."
                )
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Import failed: {e}")

with st.sidebar.expander("Fetch from EIA API"):
    try:
        default_key = st.secrets.get("EIA_API_KEY", "")
    except Exception:
        default_key = ""
    api_key = st.text_input(
        "EIA API key", value=default_key, type="password",
        help="Free key at eia.gov/opendata. Kept in this session only, never written to disk.",
    )
    today = pd.Timestamp.now("UTC")
    default_start = (DATA_PATH.exists() and pd.read_csv(DATA_PATH)["period"].max()) or (today - pd.Timedelta(days=30))
    c1, c2 = st.columns(2)
    fetch_start = c1.date_input("Start", value=pd.Timestamp(default_start).date())
    fetch_end = c2.date_input("End", value=today.date())
    if st.button("Fetch & merge"):
        if not api_key:
            st.error("Enter an EIA API key first.")
        else:
            try:
                with st.spinner("Fetching from EIA..."):
                    rows = det._fetch_rows(
                        api_key,
                        pd.Timestamp(fetch_start).strftime("%Y-%m-%dT%H"),
                        pd.Timestamp(fetch_end).strftime("%Y-%m-%dT%H"),
                    )
                if not rows:
                    st.warning("EIA returned no rows for that range.")
                else:
                    stats = merge_into_archive(det._normalise(rows))
                    st.success(
                        f"Fetched {stats['imported']:,} rows. Archive: {stats['total']:,} rows "
                        f"({stats['start']:%Y-%m-%d} to {stats['end']:%Y-%m-%d})."
                    )
                    st.cache_data.clear()
                    st.rerun()
            except SystemExit as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Fetch failed: {e}")

col1, col2 = st.sidebar.columns(2)
if col1.button("Load sample", help="Seed the archive with bundled ERCOT demo data (summer 2023)."):
    if SAMPLE_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SAMPLE_PATH, DATA_PATH)
        st.cache_data.clear()
        st.rerun()
    else:
        st.sidebar.error("No bundled sample data in this deployment.")
if col2.button("Clear archive"):
    DATA_PATH.unlink(missing_ok=True)
    st.cache_data.clear()
    st.rerun()

if DATA_PATH.exists():
    n_rows = sum(1 for _ in open(DATA_PATH)) - 1
    st.sidebar.caption(f"Archive on disk: {n_rows:,} rows")

# ================================================================ guard ===
if not DATA_PATH.exists():
    st.title("ERCOT Demand Anomaly Detector")
    st.info(
        "No data loaded yet. Use the sidebar to upload a CSV, fetch from the EIA API, "
        "or load the bundled sample data to explore the dashboard."
    )
    st.stop()

try:
    scored_full = score_archive(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as e:
    st.error(f"Couldn't score the current archive: {e}")
    st.stop()

df = scored_full

# ============================================================== filters ===
st.title("ERCOT Demand Anomaly Detector")
st.caption(f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d} · ERCOT hourly demand, Central time")

st.sidebar.header("Filters")
min_d, max_d = df.index.min().date(), df.index.max().date()
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
start_d, end_d = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_d, max_d)

threshold = st.sidebar.slider("Flag threshold (|z|)", 1.0, 6.0, 3.0, 0.1)
types = st.sidebar.multiselect(
    "Anomaly type", options=sorted(df["anomaly_type"].dropna().unique()),
    default=sorted(df["anomaly_type"].dropna().unique()),
)

view = df.loc[str(start_d):str(end_d)].copy()
view["severity"] = view[["level_z", "change_z"]].abs().max(axis=1)
view["is_anomaly"] = (view["severity"] >= threshold).fillna(False)
view = view[view["anomaly_type"].isin(types)]

st.sidebar.download_button(
    "Download scored CSV", data=view.to_csv().encode(),
    file_name="scored_hours.csv", mime="text/csv",
)

# =============================================================== metrics ==
c1, c2, c3, c4 = st.columns(4)
c1.metric("Hours in view", f"{len(view):,}")
c2.metric("Missing readings", f"{int(view['demand'].isna().sum()):,}")
flagged = view["is_anomaly"].sum()
scored_n = view["severity"].notna().sum()
c3.metric("Flagged hours", f"{flagged:,}", f"{(flagged/scored_n*100 if scored_n else 0):.1f}% of scored")
c4.metric("Peak demand", f"{view['demand'].max():,.0f} MW" if view["demand"].notna().any() else "-")

# ========================================================== demand chart ==
st.subheader("Hourly demand")
fig = go.Figure()
fig.add_trace(go.Scatter(x=view.index, y=view["demand"], mode="lines", name="Demand (MW)",
                          line=dict(color="#4C78A8", width=1.2)))
flagged_pts = view[view["is_anomaly"]]
fig.add_trace(go.Scatter(x=flagged_pts.index, y=flagged_pts["demand"], mode="markers", name="Flagged anomaly",
                          marker=dict(color="#E45756", size=7, symbol="circle-open", line=dict(width=2))))
fig.update_layout(height=420, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
                   xaxis_title=None, yaxis_title="MW", legend=dict(orientation="h", y=1.05))
st.plotly_chart(fig, use_container_width=True)

# ========================================================== z-scores ======
st.subheader("Anomaly severity over time")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=view.index, y=view["level_z"], mode="lines", name="Level z",
                           line=dict(color="#F58518", width=1)))
fig2.add_trace(go.Scatter(x=view.index, y=view["change_z"], mode="lines", name="Change z",
                           line=dict(color="#54A24B", width=1)))
fig2.add_hline(y=threshold, line_dash="dash", line_color="#E45756", annotation_text=f"+{threshold}")
fig2.add_hline(y=-threshold, line_dash="dash", line_color="#E45756", annotation_text=f"-{threshold}")
fig2.update_layout(height=320, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), yaxis_title="z-score")
st.plotly_chart(fig2, use_container_width=True)

# ======================================================= distributions ====
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Anomaly type breakdown")
    counts = view["anomaly_type"].value_counts()
    fig3 = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color="#4C78A8"))
    fig3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="hours")
    st.plotly_chart(fig3, use_container_width=True)

with col_b:
    st.subheader("Demand by hour of day")
    hod = view.groupby(view.index.hour)["demand"].mean()
    fig4 = go.Figure(go.Bar(x=hod.index, y=hod.values, marker_color="#72B7B2"))
    fig4.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="hour", yaxis_title="avg MW")
    st.plotly_chart(fig4, use_container_width=True)

# =============================================================== table ====
st.subheader(f"Flagged hours ({flagged:,})")
if flagged:
    show = flagged_pts.sort_values("severity", ascending=False)[
        ["demand", "level_z", "change_z", "severity", "anomaly_type", "explanation"]
    ]
    st.dataframe(show, use_container_width=True, height=400)
else:
    st.info("No hours flagged at this threshold in the selected range.")

with st.expander("Data health"):
    st.text(det.coverage_report(view))
    st.text(det.diagnose_timezone(view))

with st.expander("Full scored data"):
    st.dataframe(view, use_container_width=True, height=400)
