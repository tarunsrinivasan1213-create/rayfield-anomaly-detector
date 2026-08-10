# ERCOT Demand Anomaly Detector

Finds unusual hours in U.S. electric-grid hourly demand data, scores how unusual
each one is, and explains why in plain language. Built around ERCOT (the Texas
grid) but the Streamlit dashboard can point at any grid operator EIA tracks.

Everything lives in a single script, `ercot_anomaly_detector.py`, which is both
a CLI tool and a Streamlit dashboard — same fetch/load/score functions, two
front ends, so there is exactly one detection implementation regardless of how
you run it.

## What it does

- Pulls hourly electricity demand from the [U.S. EIA Open Data API v2](https://www.eia.gov/opendata/)
  (`electricity/rto/region-data`, respondent `ERCO` by default = ERCOT/Texas).
- Cleans and reindexes the series to a complete hourly, timezone-local index,
  turning missing hours into explicit gaps instead of silently skipping them.
- Scores every hour two ways, each a trailing (never-look-ahead) baseline:
  - **Level score** — compares a reading to the median of the same
    hour-of-day/day-of-week over the last `LEVEL_WEEKS` (default 4) weeks.
  - **Change score** — compares the hour-over-hour change to the median change
    at that same hour over the last `CHANGE_DAYS` (default 14) days.
- Combines both into a severity (max |z|-score), an anomaly type
  (`level` / `change` / `both` / `none` / `unscored`), and a plain-language
  explanation for every hour.
- Validates itself against ERCOT's real summer-2023 record-breaking demand
  days (`KNOWN_EVENTS` in the script) and reports honestly whether the
  detector caught each one.
- Supports a **live/continuous mode** that re-pulls the last 7 days each cycle
  (EIA revises recent hours during settlement), maintains a rolling 120-day
  local archive, and only reports newly-flagged hours (state tracked in
  `data/state.json`).

## Running it

There are three ways to use the script, all sharing the same detection logic.

### 1. Dashboard (Streamlit)

Interactive UI: pick a grid, upload a CSV or fetch any date range from the
EIA API, filter by date range/threshold/anomaly type, and explore charts of
demand, z-scores, and flagged hours.

```bash
streamlit run ercot_anomaly_detector.py
```

### 2. Historical CLI mode

Reproduces the fixed June–August 2023 "Demo Day" analysis window (pulling a
month of warm-up data starting `2025-12-01`, see note in the script header
about how those constants line up with warm-up needs):

```bash
python ercot_anomaly_detector.py              # fetch (cached) + score + report
python ercot_anomaly_detector.py --refresh    # force a fresh API pull
```

### 3. Live / continuous mode

```bash
python ercot_anomaly_detector.py --live       # one update-and-alert cycle
python ercot_anomaly_detector.py --watch 60   # run a cycle every 60 minutes
```

For anything real, prefer `cron` over `--watch`:

```
0 * * * * cd /path/to/repo && python ercot_anomaly_detector.py --live
```

### Always available

```bash
python ercot_anomaly_detector.py --selftest   # proves the logic on synthetic data, no API needed
python ercot_anomaly_detector.py --diagnose   # sanity-checks the timezone assumption on cached data
```

## Setup

### Requirements

Python with the packages in `requirements.txt`:

```
streamlit>=1.38
plotly>=5.24
pandas>=2.2
numpy>=1.26
requests>=2.31
python-dotenv>=1.0
```

Install with:

```bash
pip install -r requirements.txt
```

### Conda / dev environment

This repo includes a `.conda/` directory (a local conda environment with
Python 3.14, pip, and supporting libraries) for development on this machine.
It is git-ignored and not something you need to recreate manually — use your
own `venv`/`conda` environment and `pip install -r requirements.txt` instead.

### Dev Container / GitHub Codespaces

`.devcontainer/devcontainer.json` defines a Python 3.11 container that
installs anything in `packages.txt` (if present) plus `requirements.txt` and
`streamlit` on creation. Note its `postAttachCommand` and Codespaces
`openFiles` reference `dashboard.py`, which does not exist in this repo —
launch the dashboard manually instead with:

```bash
streamlit run ercot_anomaly_detector.py --server.enableCORS false --server.enableXsrfProtection false
```

### EIA API key

Fetching live data (CLI historical/live modes, or the dashboard's "Fetch from
EIA API" panel) requires a free API key from [eia.gov/opendata](https://www.eia.gov/opendata/).

```bash
echo "EIA_API_KEY=your_key_here" > .env
```

- The CLI (`fetch_ercot_demand`, `--live`, `--watch`) loads this from `.env`
  via `python-dotenv`.
- The dashboard can also read `EIA_API_KEY` from Streamlit secrets
  (`st.secrets`) to pre-fill the key field, or you can just paste a key into
  the sidebar for that session (it is never written to disk from there).
- `.env` is already listed in `.gitignore`, so the key never reaches GitHub.
- You can explore the dashboard without any key at all by loading the
  bundled sample data (see below) or uploading your own CSV.

## Data

- **`sample_data/ercot_demand_sample.csv`** — bundled ERCOT summer-2023 demo
  data. In the dashboard, click "Load sample" to seed the archive with it
  instantly, no API key needed.
- **`data/`** (git-ignored) — where fetched/uploaded archives are cached:
  - `data/ercot_demand.csv` — the ERCOT (`ERCO`) archive; other grids get
    their own file, e.g. `data/ciso_demand.csv`.
  - `data/state.json` — tracks which anomalous hours have already been
    reported by live/watch mode, so re-runs don't re-alert on the same hour.
- CSV input format (for upload or the on-disk cache) is either the raw EIA
  export shape (`period`, `value` columns) or the already-loaded shape
  (`timestamp`, `demand`).

## Outputs

- **`outputs/scored_hours.csv`** (git-ignored) — every hour in the analysis
  window with `level_z`, `change_z`, `severity`, `anomaly_type`, `is_anomaly`,
  and a written `explanation`. Written by historical mode, live mode, and
  downloadable from the dashboard sidebar.
- **`outputs/new_anomalies.csv`** (git-ignored) — written only by live mode,
  containing just the hours newly flagged since the previous run.
- Console/report output (`run_report`) includes a coverage summary, a
  timezone spot-check against a known EIA-reported peak hour, validation
  against `KNOWN_EVENTS` (did the detector catch each real ERCOT record day),
  the most anomalous days, and the top N most anomalous hours.

## Project structure

```
ercot_anomaly_detector.py   # fetch, load/clean, score, report, self-test, CLI, and Streamlit dashboard
requirements.txt            # Python dependencies
sample_data/                # bundled demo dataset (tracked in git)
data/                       # fetched/uploaded archives + live-mode state (git-ignored)
outputs/                    # scored CSVs written by CLI/live runs (git-ignored)
.streamlit/config.toml      # Streamlit theme/server settings
.devcontainer/               # Codespaces / VS Code dev container config
.conda/                     # local conda environment (git-ignored, not required)
```

## Configuration

Most tunables live at the top of `ercot_anomaly_detector.py` under `CONFIG`,
including:

- `RESPONDENT` / `DATA_TYPE` — which EIA respondent/series the CLI modes pull
  (dashboard lets you pick any respondent at runtime).
- `FETCH_START` / `FETCH_END`, `ANALYSIS_START` / `ANALYSIS_END` — the
  historical-mode pull and report window.
- `LEVEL_WEEKS`, `CHANGE_DAYS`, `SCALE_WINDOW` and their `*_MIN_HISTORY`
  counterparts — how much trailing history each baseline uses.
- `FLAG_THRESHOLD` — the |z|-score at or above which an hour is flagged
  (also adjustable live via the dashboard's threshold slider).
- `ARCHIVE_DAYS`, `BACKFILL_DAYS`, `REVISION_DAYS` — live-mode archive sizing
  and how far back each cycle re-checks for EIA revisions.
