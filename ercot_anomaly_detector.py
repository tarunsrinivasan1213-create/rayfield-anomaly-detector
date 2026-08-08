"""
====================================================================
 OPERATIONS ANOMALY DETECTOR - ERCOT Hourly Electricity Demand
 Rayfield Systems Technical Project | Option A
====================================================================

Finds unusual hours in ERCOT (Texas) electricity demand, scores how unusual each
one is, and explains why in plain language.

  Input : hourly demand readings, U.S. EIA Open Data API v2 (respondent ERCO, type D)
  Output: one row per hour with a severity score, an anomaly type, and a written reason

--------------------------------------------------------------------
 QUICK START
--------------------------------------------------------------------
    pip install pandas numpy requests python-dotenv

    echo "EIA_API_KEY=your_key_here" > .env
    echo ".env" >> .gitignore

  Historical mode (the Demo Day run, fixed June-August 2023 window):
    python ercot_anomaly_detector.py              # fetch (cached) + score + report
    python ercot_anomaly_detector.py --refresh    # force a fresh API pull

  Live mode (continuous operation on new data as EIA publishes it):
    python ercot_anomaly_detector.py --live       # one update-and-alert cycle
    python ercot_anomaly_detector.py --watch 60   # a cycle every 60 minutes

  Always available:
    python ercot_anomaly_detector.py --selftest   # prove it works, no API needed
    python ercot_anomaly_detector.py --diagnose   # check the timezone assumption

--------------------------------------------------------------------
 HOW LIVE MODE WORKS
--------------------------------------------------------------------
Each cycle re-pulls the last 7 days (EIA revises recently published hours during
settlement), merges them over the stored values, prunes the archive to 120 days,
re-scores the WHOLE archive so the baselines stay warm, then reports only hours
that were not flagged on a previous run. Already-reported hours are tracked in
data/state.json.

Nothing about the detection logic changes between the two modes. Every baseline
is trailing-only - it never uses the current reading or anything after it - so
the tool produces identical output on historical data and on a live feed. There
is no retraining step.

Cron is a better scheduler than --watch for anything real:
    0 * * * * cd /path/to/repo && python ercot_anomaly_detector.py --live

--------------------------------------------------------------------
 GROUND TRUTH IN THIS WINDOW
--------------------------------------------------------------------
Summer 2023 was ERCOT's record-breaking summer: 10 new all-time peak demand
records, ending with 85,464 MW on August 10, 2023. Those dates are listed in
KNOWN_EVENTS below and the script reports, unprompted, whether the detector
caught each one. That is the honest evaluation, not a claim of success.

Reported peak values differ slightly between sources (85,435 unofficial ->
85,464 -> 85,508 after settlement), so treat them as approximate. The detector
never uses these numbers; they are for validation only.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ====================================================================
# CONFIG - everything tunable lives here
# ====================================================================

# --- data source ----------------------------------------------------
EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
RESPONDENT = "ERCO"          # ERCOT (Texas)
DATA_TYPE = "D"              # D = demand

# The analysis window is June-August 2023. The pull starts a month EARLIER on
# purpose: the level baseline needs 3 prior same-weekday-same-hour readings plus
# 72 hours of residuals before it can score anything. Starting the pull on June 1
# leaves June 1-24 unscoreable, which is 26% of the target window.
FETCH_START = "2025-12-01T00"     # a month early, for warm-up
FETCH_END = "2026-08-03T23"
ANALYSIS_START = "2026-01-01"
ANALYSIS_END = "2026-06-30"

PAGE_SIZE = 5000             # EIA's per-request maximum
DATA_PATH = Path("data/ercot_demand.csv")
OUT_PATH = Path("outputs/scored_hours.csv")

# --- live mode ------------------------------------------------------
# The archive must always hold enough trailing history to warm the baselines up:
# 4 weeks for the level slots plus 14 days of residuals for the spread. 120 days
# is comfortably past that and keeps the file small.
ARCHIVE_DAYS = 120
BACKFILL_DAYS = 120          # how far back to reach on a cold start
REVISION_DAYS = 7            # re-pull this much trailing data every run
PUBLICATION_LAG_H = 24       # EIA lags roughly a day; not an error
STATE_PATH = Path("data/state.json")
ALERTS_PATH = Path("outputs/new_anomalies.csv")

# --- timezone -------------------------------------------------------
SOURCE_TZ = "UTC"                 # EIA's hourly endpoint returns UTC
LOCAL_TZ = "America/Chicago"      # ERCOT operates on Central Time

# --- detection ------------------------------------------------------
LEVEL_WEEKS = 4          # prior same-slot readings, e.g. the last 4 Tuesdays at 17:00
LEVEL_MIN_HISTORY = 3

CHANGE_DAYS = 14         # prior same-hour ramps, e.g. the last 14 ramps into 17:00
CHANGE_MIN_HISTORY = 7

SCALE_WINDOW = 336       # hours of residual history used for the spread (14 days)
SCALE_MIN_HISTORY = 72

# MEDIAN, not mean. Summer 2023 set ten records, so a previous record day often
# sits inside the baseline window and a mean baseline inflates the expectation,
# hiding the next record. Measured on synthetic data with all ten records
# injected: median lifts the Aug 10 level score from z=4.5 to z=5.2 at an
# identical 2.3% flag rate. Strictly better, so median it is.
CENTER = "median"

FLAG_THRESHOLD = 3.0     # |z| at or above this is flagged
TOP_N = 10               # how many to print
MIN_SCALE = 1e-6         # guard against divide-by-zero on a perfectly flat series
MAD_TO_SIGMA = 1.4826    # scales a median-absolute-deviation to a normal sigma

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# --- validation -----------------------------------------------------
# ERCOT all-time peak demand records set inside the analysis window.
# (date, approximate reported peak MW, note)
KNOWN_EVENTS = [
    ("2023-06-26", None,  "record broken daily Jun 26-29, each above 80,000 MW"),
    ("2023-06-27", 80787, "all-time record"),
    ("2023-06-28", None,  "record broken again"),
    ("2023-06-29", None,  "record broken again"),
    ("2023-07-12", 81351, "all-time record"),
    ("2023-07-13", 81406, "all-time record"),
    ("2023-07-17", 81911, "all-time record"),
    ("2023-07-18", 82579, "record; EIA reports the peak in the 6 p.m. CT hour"),
    ("2023-07-31", 83047, "all-time record"),
    ("2023-08-01", 83593, "all-time record"),
    ("2023-08-10", 85464, "ALL-TIME RECORD for summer 2023"),
]


# ====================================================================
# 1. FETCH
# ====================================================================

def _get_api_key() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise SystemExit(
            "EIA_API_KEY not found.\n"
            "  1. Create a file named .env next to this script\n"
            "  2. Put this line in it:  EIA_API_KEY=your_key_here\n"
            "  3. Add .env to .gitignore so the key never reaches GitHub"
        )
    return api_key


def _fetch_rows(api_key: str, start: str, end: str) -> list[dict]:
    """Page through one date range and return the raw rows."""
    import requests

    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": RESPONDENT,
            "facets[type][]": DATA_TYPE,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": PAGE_SIZE,
        }
        resp = requests.get(EIA_URL, params=params, timeout=60)
        if resp.status_code != 200:
            raise SystemExit(
                f"EIA returned HTTP {resp.status_code}.\n{resp.text[:400]}\n"
                "Most common cause: the API key is missing, wrong, or not loaded from .env."
            )
        payload = resp.json().get("response", {})
        page, total = payload.get("data", []), int(payload.get("total", 0))
        if not page:
            break
        rows.extend(page)
        print(f"  fetched {len(rows):,} / {total:,} rows")
        offset += len(page)
        if offset >= total:
            break
        time.sleep(0.3)     # EIA caps a single request at 5,000 rows; page politely
    return rows


def _normalise(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
    return df[["period", "value"]]


def fetch_ercot_demand(path: Path | None = None, refresh: bool = False) -> Path:
    """
    Historical mode: pull the fixed 2023 window once and cache it.

    If the cache exists and refresh is False, no network call is made -
    re-pulling on every run is slow and rude to a free public API.
    """
    path = path or DATA_PATH
    if path.exists() and not refresh:
        print(f"Using cached data at {path} (pass --refresh to re-pull)")
        return path

    print(f"Pulling {RESPONDENT} demand, {FETCH_START} to {FETCH_END} ...")
    rows = _fetch_rows(_get_api_key(), FETCH_START, FETCH_END)
    if not rows:
        raise SystemExit(
            "EIA returned zero rows. Check the respondent code, the type, and that "
            "the date window is valid and not in the future."
        )
    df = _normalise(rows).drop_duplicates(subset="period", keep="last").sort_values("period")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df):,} rows)")
    return path


# ====================================================================
# 1b. LIVE MODE - incremental archive
# ====================================================================

def update_archive(path: Path | None = None, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Bring the local archive up to date and return (archive, stats).

    Two things make this more than "append the new rows":

    1. EIA REVISES recently published hours during settlement, which is why the
       Aug 10 2023 record is quoted as 85,435 then 85,464 then 85,508. So every
       run re-pulls the last REVISION_DAYS and overwrites, rather than trusting
       the first value it ever saw.
    2. The baselines need weeks of trailing history, so the archive is a rolling
       window rather than "whatever arrived since last time".
    """
    path = path or DATA_PATH
    now = now or pd.Timestamp.now("UTC")
    prior = pd.DataFrame(columns=["period", "value"])
    if path.exists():
        prior = pd.read_csv(path)
        prior["period"] = pd.to_datetime(prior["period"], utc=True)
        prior["value"] = pd.to_numeric(prior["value"], errors="coerce")
        prior = prior[["period", "value"]]

    if prior.empty:
        start = now - pd.Timedelta(days=BACKFILL_DAYS)
        print(f"Cold start: backfilling {BACKFILL_DAYS} days ...")
    else:
        start = prior["period"].max() - pd.Timedelta(days=REVISION_DAYS)
        print(f"Updating from {start:%Y-%m-%d %H:%M} UTC (re-checking "
              f"{REVISION_DAYS} days for revisions) ...")

    rows = _fetch_rows(_get_api_key(), start.strftime("%Y-%m-%dT%H"), now.strftime("%Y-%m-%dT%H"))
    fresh = _normalise(rows) if rows else pd.DataFrame(columns=["period", "value"])

    # Count revisions before merging: same period, different value.
    revised = 0
    if not prior.empty and not fresh.empty:
        merged = fresh.merge(prior, on="period", how="inner", suffixes=("_new", "_old"))
        revised = int((merged["value_new"] != merged["value_old"]).sum())

    combined = pd.concat([prior, fresh], ignore_index=True)
    # keep="last" means the freshly fetched value wins over the stored one.
    combined = combined.drop_duplicates(subset="period", keep="last").sort_values("period")

    added = len(combined) - len(prior)
    combined = combined[combined["period"] >= now - pd.Timedelta(days=ARCHIVE_DAYS)]

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)

    newest = combined["period"].max() if not combined.empty else None
    lag_h = (now - newest).total_seconds() / 3600 if newest is not None else float("nan")
    stats = {"added": added, "revised": revised, "total": len(combined),
             "newest": newest, "lag_hours": lag_h}
    return combined, stats


def _state_load(path: Path | None = None) -> dict:
    path = path or STATE_PATH
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"reported": [], "last_run": None}


def _state_save(state: dict, path: Path | None = None) -> None:
    path = path or STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the file bounded; anything older than the archive can never recur.
    state["reported"] = sorted(set(state["reported"]))[-2000:]
    path.write_text(json.dumps(state, indent=2))


# ====================================================================
# 2. LOAD & CLEAN
# ====================================================================

def load_demand(path, period_col: str = "period", value_col: str = "value",
                source_tz: str | None = SOURCE_TZ,
                local_tz: str | None = LOCAL_TZ) -> pd.DataFrame:
    """
    Load the CSV and return an hourly-indexed frame with one 'demand' column.

    Timestamps are converted from source_tz to local_tz and then made naive, so
    hour-of-day grouping and every printed explanation refer to ERCOT local time.
    Pass local_tz=None to stay on the source clock.

    Missing hours are inserted as NaN rather than silently skipped, so a gap stays
    visible as a gap instead of two unrelated hours appearing adjacent.
    """
    raw = pd.read_csv(path)

    if period_col not in raw.columns or value_col not in raw.columns:
        raise KeyError(
            f"Expected columns '{period_col}' and '{value_col}'. Found: {list(raw.columns)}"
        )

    ts = pd.to_datetime(raw[period_col], format="mixed", utc=True)
    if local_tz is not None:
        ts = ts.dt.tz_convert(local_tz)

    df = pd.DataFrame({
        "timestamp": ts,
        # EIA returns value as a string, so cast explicitly.
        "demand": pd.to_numeric(raw[value_col], errors="coerce"),
    }).dropna(subset=["timestamp"])

    if df.empty:
        raise ValueError(f"No usable rows in {path} - every timestamp failed to parse.")

    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.sort_values("timestamp").set_index("timestamp")

    # Reindex to a complete hourly range so missing hours become explicit NaNs.
    # The index stays TIMEZONE-AWARE on purpose. Making it naive local looks
    # tidier but breaks twice a year: at spring-forward a 2 a.m. hour that never
    # existed is invented as a permanent fake gap, and at fall-back 1 a.m. occurs
    # twice, so drop_duplicates silently deletes a real reading. Verified on the
    # 2023 transitions: naive lost 1 hour and invented 1; tz-aware loses neither.
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(),
                                  freq="h", tz=df.index.tz))
    df.index.name = "timestamp"
    return df[["demand"]]


def diagnose_timezone(df: pd.DataFrame) -> str:
    """
    Check which clock the data is on instead of assuming it. ERCOT demand bottoms
    out around 03:00-07:00 local and peaks around 14:00-20:00 local in summer. If
    these land ~5 hours later, the frame is still on UTC.
    """
    d = df["demand"].dropna()
    if d.empty:
        return "No data to diagnose."
    by_day = d.groupby(d.index.date)
    trough = pd.Series([t.hour for t in by_day.idxmin()])
    peak = pd.Series([t.hour for t in by_day.idxmax()])
    ok = 3 <= trough.median() <= 7 and 14 <= peak.median() <= 20
    return (f"Timezone check: median daily trough {trough.median():.0f}:00, "
            f"median daily peak {peak.median():.0f}:00 -> "
            f"{'consistent with ERCOT local time' if ok else 'NOT local time; fix SOURCE_TZ/LOCAL_TZ'}")


# ====================================================================
# 3. SCORE
# ====================================================================

def _trailing_center(values: pd.Series, keys, window: int, min_history: int) -> pd.Series:
    """
    Center of the previous `window` readings sharing the same key, current excluded.

    .shift(1) inside each group steps back one occurrence of that slot: one week
    for hour-of-day x day-of-week, one day for hour-of-day.
    """
    return values.groupby(keys).transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_history).agg(CENTER)
    )


def _trailing_scale(residual: pd.Series, window: int = SCALE_WINDOW,
                    min_history: int = SCALE_MIN_HISTORY) -> pd.Series:
    """
    Robust spread of the residuals over a wide trailing window.

    Median-based rather than std-based so that a genuine anomaly last week does not
    inflate the yardstick and mask a similar one this week.
    """
    scale = (residual.abs().shift(1)
             .rolling(window, min_periods=min_history).median() * MAD_TO_SIGMA)
    return scale.where(scale > MIN_SCALE)


def add_level_scores(df: pd.DataFrame, weeks: int = LEVEL_WEEKS,
                     min_history: int = LEVEL_MIN_HISTORY) -> pd.DataFrame:
    """Compare each reading to the same hour-of-day and day-of-week in prior weeks."""
    df = df.copy()
    center = _trailing_center(
        df["demand"], [df.index.dayofweek, df.index.hour], weeks, min_history
    )
    residual = df["demand"] - center
    df["level_baseline"] = center
    df["level_scale"] = _trailing_scale(residual)
    df["level_z"] = residual / df["level_scale"]
    return df


def add_change_scores(df: pd.DataFrame, days: int = CHANGE_DAYS,
                      min_history: int = CHANGE_MIN_HISTORY) -> pd.DataFrame:
    """
    Compare each hour-over-hour change to the normal change at that hour.

    Grouping by hour matters: a +3,000 MW jump is routine at 17:00 in August and
    very strange at 03:00. A single global ramp baseline would flag every evening.
    """
    df = df.copy()
    delta = df["demand"].diff()
    # diff() across a gap compares two non-adjacent hours; that is not a real ramp.
    delta[df["demand"].shift(1).isna()] = np.nan
    df["delta"] = delta

    center = _trailing_center(delta, df.index.hour, days, min_history)
    residual = delta - center
    df["change_baseline"] = center
    df["change_scale"] = _trailing_scale(residual)
    df["change_z"] = residual / df["change_scale"]
    return df


def _explain(row: pd.Series, threshold: float) -> str:
    if pd.isna(row["demand"]):
        return "No data for this hour (gap in the source series)."

    level_z, change_z = row["level_z"], row["change_z"]
    if pd.isna(level_z) and pd.isna(change_z):
        return "Not enough prior history to score this hour yet."

    parts = []
    if not pd.isna(level_z):
        slot = f"{DAY_NAMES[row.name.dayofweek]} {row.name.hour:02d}:00"
        gap = row["demand"] - row["level_baseline"]
        if abs(level_z) >= threshold:
            parts.append(
                f"Demand {row['demand']:,.0f} MW is {abs(gap):,.0f} MW "
                f"{'above' if gap >= 0 else 'below'} the typical {slot} level of "
                f"{row['level_baseline']:,.0f} MW (z={level_z:+.1f})."
            )
        else:
            parts.append(f"Level is normal for a {slot} (z={level_z:+.1f}).")

    if not pd.isna(change_z):
        if abs(change_z) >= threshold:
            parts.append(
                f"Demand {'rose' if row['delta'] >= 0 else 'fell'} "
                f"{abs(row['delta']):,.0f} MW from the previous hour, against a typical "
                f"{row['change_baseline']:+,.0f} MW change at this hour (z={change_z:+.1f})."
            )
        else:
            parts.append(f"Hour-over-hour change is normal (z={change_z:+.1f}).")

    return " ".join(parts)


def detect_anomalies(df: pd.DataFrame, threshold: float = FLAG_THRESHOLD) -> pd.DataFrame:
    """
    Score every hour. Adds:
      level_z, change_z  signed z-scores (the sign tells you spike vs dip)
      severity           max absolute z across both tests; the ranking number
      anomaly_type       level / change / both / none / unscored
      is_anomaly         severity >= threshold
      explanation        plain-language reason, written for every hour
    """
    df = add_level_scores(df)
    df = add_change_scores(df)

    magnitude = df[["level_z", "change_z"]].abs()
    df["severity"] = magnitude.max(axis=1)

    level_hit = magnitude["level_z"] >= threshold
    change_hit = magnitude["change_z"] >= threshold
    df["anomaly_type"] = np.select(
        [level_hit & change_hit, level_hit, change_hit],
        ["both", "level", "change"], default="none",
    )
    # Hours we could not score are unknown, not normal. Keep that distinction.
    df.loc[df["severity"].isna(), "anomaly_type"] = "unscored"
    df["is_anomaly"] = (df["severity"] >= threshold).fillna(False)
    df["explanation"] = df.apply(_explain, axis=1, threshold=threshold)
    return df


# ====================================================================
# 4. REPORT
# ====================================================================

def coverage_report(scored: pd.DataFrame) -> str:
    total = len(scored)
    missing = int(scored["demand"].isna().sum())
    unscored = int(scored["severity"].isna().sum())
    flagged = int(scored["is_anomaly"].sum())
    usable = total - unscored
    pct = (flagged / usable * 100) if usable else 0.0
    return (f"{total:,} hours | {missing:,} missing readings | "
            f"{unscored:,} unscored (gap or warm-up) | "
            f"{flagged:,} flagged ({pct:.1f}% of scored hours)")


def top_anomalies(scored: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """
    The n most severe hours, ranked.

    Deliberately ranks by severity rather than filtering on the threshold, so this
    always returns something to look at. The threshold is a reporting decision;
    the ranking is the real output.
    """
    return scored.dropna(subset=["severity"]).nlargest(n, "severity")


def top_days(scored: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Rank whole days by their worst hour. A heat wave is a day-shaped event."""
    d = scored.dropna(subset=["severity"]).copy()
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(d.index.date)
    out = pd.DataFrame({
        "peak_MW": g["demand"].max(),
        "worst_z": g["severity"].max(),
        "flagged_hours": g["is_anomaly"].sum(),
    })
    return out.nlargest(n, "worst_z")


def check_known_events(scored: pd.DataFrame, threshold: float = FLAG_THRESHOLD) -> str:
    """
    Did the detector catch the ERCOT records that actually happened? Reported
    honestly, hits and misses alike. This is the evaluation, not a victory lap.
    """
    lines = [f"{'date':<13}{'peak MW':>10}{'worst z':>9}{'caught':>9}  note", "-" * 78]
    hits = 0
    checked = 0
    for date, reported, note in KNOWN_EVENTS:
        try:
            day = scored.loc[date]
        except KeyError:
            lines.append(f"{date:<13}{'--':>10}{'--':>9}{'no data':>9}  outside window")
            continue
        if day["severity"].notna().sum() == 0:
            lines.append(f"{date:<13}{'--':>10}{'--':>9}{'unscored':>9}  warm-up period")
            continue
        checked += 1
        peak = day["demand"].max()
        worst = day["severity"].max()
        caught = bool(day["is_anomaly"].any())
        hits += caught
        lines.append(f"{date:<13}{peak:>10,.0f}{worst:>9.1f}{'YES' if caught else 'no':>9}  {note}")

    rate = f"{hits}/{checked}" if checked else "0/0"
    lines.append("-" * 78)
    lines.append(f"Caught {rate} known record days at threshold |z| >= {threshold}.")
    return "\n".join(lines)


def timezone_spot_check(scored: pd.DataFrame) -> str:
    """
    EIA reports that ERCOT demand peaked at 82,579 MWh in the 6 p.m. CT hour on
    2023-07-18. That is the same dataset this script pulls, so it is an exact
    external check on the UTC -> Central conversion.
    """
    try:
        day = scored.loc["2023-07-18"].dropna(subset=["demand"])
    except KeyError:
        return "Spot check skipped: 2023-07-18 not in the loaded window."
    if day.empty:
        return "Spot check skipped: no data on 2023-07-18."
    hour = day["demand"].idxmax().hour
    peak = day["demand"].max()
    # The real daily peak hour wanders by an hour or two, so a small offset means
    # nothing. A ~5-hour offset is the signature of an unconverted UTC series.
    if 15 <= hour <= 20:
        verdict = "MATCH"
    elif 21 <= hour <= 23 or hour <= 1:
        verdict = "OFF BY ~5 HOURS - the series is still on UTC; check SOURCE_TZ/LOCAL_TZ"
    else:
        verdict = "UNEXPECTED - inspect the series before trusting the baselines"
    return (f"Spot check 2023-07-18: peak {peak:,.0f} MW at {hour:02d}:00 local "
            f"(EIA reports ~82,579 MW in the 6 p.m. CT hour) -> {verdict}")


def run_report(scored: pd.DataFrame, threshold: float = FLAG_THRESHOLD) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n ERCOT DEMAND ANOMALY REPORT  |  {ANALYSIS_START} to {ANALYSIS_END}\n{bar}")
    print(coverage_report(scored))
    print(timezone_spot_check(scored))

    print(f"\n{'-'*78}\n VALIDATION AGAINST KNOWN ERCOT RECORDS\n{'-'*78}")
    print(check_known_events(scored, threshold))

    print(f"\n{'-'*78}\n MOST ANOMALOUS DAYS\n{'-'*78}")
    days = top_days(scored)
    if days.empty:
        print("No scoreable days.")
    else:
        for date, row in days.iterrows():
            print(f"  {date}   peak {row['peak_MW']:>8,.0f} MW   "
                  f"worst z {row['worst_z']:>5.1f}   {int(row['flagged_hours'])} flagged hours")

    print(f"\n{'-'*78}\n TOP {TOP_N} ANOMALOUS HOURS\n{'-'*78}")
    top = top_anomalies(scored)
    if top.empty:
        print("Nothing scoreable. The window is probably shorter than the warm-up period.")
    else:
        for ts, row in top.iterrows():
            mark = "FLAG" if row["is_anomaly"] else "    "
            print(f"  {mark} {ts}  severity {row['severity']:>5.1f}  [{row['anomaly_type']}]")
            print(f"        {row['explanation']}")
    print(bar)


def run_live(now: pd.Timestamp | None = None,
             threshold: float = FLAG_THRESHOLD) -> tuple[pd.DataFrame, dict]:
    """
    One continuous-operation cycle: update the archive, re-score everything,
    and report only anomalies that have not been reported before.

    Scoring runs over the WHOLE archive, not just the new hours, because the
    baselines need their trailing history. Reporting is then filtered to what is
    new. Re-scoring also means a revised reading is judged on its revised value.
    """
    archive, stats = update_archive(now=now)
    if archive.empty:
        print("Archive is empty and EIA returned nothing. Nothing to score.")
        return pd.DataFrame(), stats

    scored = detect_anomalies(load_demand(DATA_PATH), threshold=threshold)

    state = _state_load()
    seen = set(state["reported"])
    flagged = scored[scored["is_anomaly"]]
    new = flagged[~flagged.index.map(lambda t: t.isoformat()).isin(seen)]

    bar = "=" * 78
    print(f"\n{bar}\n LIVE UPDATE  |  {pd.Timestamp.now("UTC"):%Y-%m-%d %H:%M} UTC\n{bar}")
    print(f"archive: {stats['total']:,} hours | +{stats['added']} new | "
          f"{stats['revised']} revised by EIA")
    if stats["newest"] is not None:
        stale = stats["lag_hours"] > PUBLICATION_LAG_H * 2
        print(f"newest reading: {stats['newest']:%Y-%m-%d %H:%M} UTC "
              f"({stats['lag_hours']:.0f}h behind{'  <-- unusually stale' if stale else ', normal for EIA'})")
    print(diagnose_timezone(load_demand(DATA_PATH)))
    print(coverage_report(scored))

    print(f"\n{'-'*78}\n NEW ANOMALIES SINCE LAST RUN ({len(new)})\n{'-'*78}")
    if new.empty:
        print("  None. Every currently flagged hour has already been reported.")
    else:
        for ts, row in new.sort_index().iterrows():
            print(f"  {ts:%Y-%m-%d %H:%M %Z}  severity {row['severity']:>5.1f}  [{row['anomaly_type']}]")
            print(f"        {row['explanation']}")
        ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(ALERTS_PATH)
        print(f"\n  Wrote {ALERTS_PATH}")
    print(bar)

    state["reported"] = list(seen | set(flagged.index.map(lambda t: t.isoformat())))
    state["last_run"] = pd.Timestamp.now("UTC").isoformat()
    _state_save(state)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT_PATH)
    return new, stats


# ====================================================================
# 5. SELF-TEST  (proves the logic without touching the network)
# ====================================================================

def selftest() -> None:
    """Inject known anomalies into a synthetic series and confirm they are found."""
    print("Running self-test on synthetic data (no API needed) ...\n")
    rng = np.random.default_rng(0)
    local = pd.date_range("2023-05-01", "2023-08-31 23:00", freq="h")
    h, doy = local.hour.to_numpy(), local.dayofyear.to_numpy()
    wknd = (local.dayofweek.to_numpy() >= 5).astype(float)
    s = pd.Series(52000 + 14000*np.sin((h - 9.5)/24*2*np.pi) + 85*(doy - 121)
                  - 3000*wknd + rng.normal(0, 900, len(local)), index=local)

    # Reproduce the shape of the real summer: ten escalating record days.
    for date, reported, _ in KNOWN_EVENTS:
        peak = reported or 81000
        day = s.index.date == pd.Timestamp(date).date()
        ramp = np.clip(1 - np.abs(s.index[day].hour - 17) / 7, 0, 1)
        s.loc[day] = s[day].values + (peak - s[day].max()) * ramp

    s.loc["2023-07-20 03:00"] -= 11000                     # sharp overnight drop
    s.loc["2023-07-05 08:00":"2023-07-05 14:00"] = np.nan  # 7-hour gap

    # Write it out the way EIA delivers it: UTC periods, string values, unsorted,
    # missing hours omitted entirely rather than nulled.
    tmp = Path("selftest_data.csv")
    pd.DataFrame({
        "period": (s.index + pd.Timedelta(hours=5)).strftime("%Y-%m-%dT%H"),
        "value": s.values.astype(str),
    }).query("value != 'nan'").sample(frac=1, random_state=1).to_csv(tmp, index=False)

    df = load_demand(tmp)
    scored = detect_anomalies(df)

    # DST regression: neither transition may lose or invent a reading.
    dst_ok = True
    for a, b in [("2023-03-05", "2023-03-19"), ("2023-10-29", "2023-11-12")]:
        u = pd.date_range(f"{a} 00:00", f"{b} 23:00", freq="h", tz="UTC")
        d = Path("dst_tmp.csv")
        pd.DataFrame({"period": u, "value": np.arange(len(u), dtype=float)}).to_csv(d, index=False)
        got = load_demand(d)
        dst_ok &= (int(got["demand"].notna().sum()) == len(u)) and (int(got["demand"].isna().sum()) == 0)
        d.unlink(missing_ok=True)

    checks = [
        ("DST transitions lose nothing", dst_ok),
        ("UTC -> local conversion", df.index.min() == pd.Timestamp("2023-05-01 00:00", tz=LOCAL_TZ)),
        ("complete hourly index", df.index.freqstr.lower() == "h"),
        ("gap preserved as NaN", int(df["demand"].isna().sum()) == 7),
        ("row count preserved", len(scored) == len(df)),
        ("every hour explained", not scored["explanation"].isna().any()),
        ("gaps never flagged", not scored.loc[scored["demand"].isna(), "is_anomaly"].any()),
        ("gaps marked unscored",
         (scored.loc[scored["demand"].isna(), "anomaly_type"] == "unscored").all()),
        ("no ramp score across a gap", pd.isna(scored.loc["2023-07-05 15:00", "change_z"])),
        ("all-time record caught", bool(scored.loc["2023-08-10", "is_anomaly"].any())),
        ("overnight drop caught", bool(scored.loc["2023-07-20 03:00", "is_anomaly"])),
        ("baseline did not absorb the record",
         scored.loc["2023-08-10 17:00", "level_baseline"] < scored.loc["2023-08-10 17:00", "demand"] - 3000),
        ("quiet week stays quiet", int(scored.loc["2023-06-05":"2023-06-11", "is_anomaly"].sum()) <= 5),
        ("flag rate under 5%",
         scored["is_anomaly"].sum() / scored["severity"].notna().sum() < 0.05),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    caught = sum(bool(scored.loc[d, "is_anomaly"].any())
                 for d, _, _ in KNOWN_EVENTS if scored.loc[d, "severity"].notna().any())
    print(f"\n  Record days caught: {caught}/{len(KNOWN_EVENTS)}")

    tmp.unlink(missing_ok=True)

    # --- live mode, against a mock EIA that lags and revises -------------
    from unittest import mock
    import shutil, tempfile
    sandbox = Path(tempfile.mkdtemp())
    real_data, real_state, real_out, real_alerts = DATA_PATH, STATE_PATH, OUT_PATH, ALERTS_PATH
    try:
        globals()["DATA_PATH"] = sandbox / "a.csv"
        globals()["STATE_PATH"] = sandbox / "s.json"
        globals()["OUT_PATH"] = sandbox / "o.csv"
        globals()["ALERTS_PATH"] = sandbox / "n.csv"

        base = pd.date_range("2023-05-01", "2023-08-31 23:00", freq="h", tz="UTC")
        tr = pd.Series(60000 + 9000*np.sin(base.hour.to_numpy()/24*2*np.pi), index=base)

        def fetch_at(now):
            def _f(key, start, end):
                a = pd.Timestamp(start, tz="UTC")
                b = min(pd.Timestamp(end, tz="UTC"), now - pd.Timedelta(hours=24))
                v = tr[(tr.index >= a) & (tr.index <= b)].copy()
                v[v.index >= now - pd.Timedelta(days=3)] *= 1.004   # preliminary
                return [{"period": t.strftime("%Y-%m-%dT%H"), "value": str(x)} for t, x in v.items()]
            return _f

        out = []
        for when in ["2023-08-10 12:00", "2023-08-12 12:00", "2023-08-12 12:00"]:
            n = pd.Timestamp(when, tz="UTC")
            with mock.patch.object(sys.modules[__name__], "_get_api_key", lambda: "k"), \
                 mock.patch.object(sys.modules[__name__], "_fetch_rows", fetch_at(n)):
                import io, contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    alerts, st = run_live(now=n)
            out.append((st, len(alerts)))

        # Exercise the default clock too - injecting `now` in every test once
        # hid a crash in the production path.
        default_clock_ok = True
        try:
            with mock.patch.object(sys.modules[__name__], "_get_api_key", lambda: "k"), \
                 mock.patch.object(sys.modules[__name__], "_fetch_rows",
                                   lambda k, a, b: []), \
                 contextlib.redirect_stdout(io.StringIO()):
                update_archive()
        except Exception:
            default_clock_ok = False

        live_checks = [
            ("live: default clock path works", default_clock_ok),
            ("live: cold start backfills", out[0][0]["added"] > 1000),
            ("live: EIA revisions detected", out[1][0]["revised"] > 0),
            ("live: no duplicate alerts on re-run", out[2][1] == 0),
            ("live: archive stays capped", out[2][0]["total"] <= ARCHIVE_DAYS * 24 + 48),
        ]
    finally:
        globals()["DATA_PATH"], globals()["STATE_PATH"] = real_data, real_state
        globals()["OUT_PATH"], globals()["ALERTS_PATH"] = real_out, real_alerts
        shutil.rmtree(sandbox, ignore_errors=True)

    for name, ok in live_checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    checks += live_checks

    failed = [n for n, ok in checks if not ok]
    if failed:
        raise SystemExit(f"\nSELF-TEST FAILED: {failed}")
    print("\nAll self-test checks passed.")


# ====================================================================
# 6. MAIN
# ====================================================================

def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        selftest()
        return 0

    if "--live" in argv:
        run_live()
        return 0

    if "--watch" in argv:
        i = argv.index("--watch")
        mins = int(argv[i + 1]) if len(argv) > i + 1 and argv[i + 1].isdigit() else 60
        print(f"Watching: one cycle every {mins} min. Ctrl-C to stop.")
        print("(For anything real, use cron or a scheduler instead of this loop.)")
        while True:
            try:
                run_live()
            except SystemExit:
                raise
            except Exception as exc:                    # a transient API blip
                print(f"Cycle failed: {type(exc).__name__}: {exc}. Retrying next cycle.")
            time.sleep(mins * 60)

    csv_path = fetch_ercot_demand(refresh="--refresh" in argv)
    full = load_demand(csv_path)

    print(diagnose_timezone(full))
    if "--diagnose" in argv:
        return 0

    scored = detect_anomalies(full)

    # Score on the full series (May included, so the baselines are warm), then
    # report only on the analysis window.
    window = scored.loc[ANALYSIS_START:ANALYSIS_END]
    run_report(window)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    window.to_csv(OUT_PATH)
    print(f"\nWrote {OUT_PATH}  ({len(window):,} rows, one per hour)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
