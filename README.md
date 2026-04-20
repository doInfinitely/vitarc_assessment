# Clinical Patient Timeline Pipeline

Programmatic pipeline that ingests clinical PDFs, wearable XML, patient profile JSON, and manual health entries to produce a unified patient timeline with health snapshots, anomaly detection, and an interactive viewer.

## Quick Start

```bash
# Full pipeline (stages 1-5)
python3 run_pipeline.py

# Skip document conversion, start from timeline construction
python3 run_pipeline.py --from 3

# Only dump existing outputs to terminal
python3 run_pipeline.py --dump
```

## Requirements

**Python 3.11+** (uses `tomllib`)

**System dependencies** (stages 1-2 only):
```bash
# macOS
brew install poppler tesseract

# Ubuntu/Debian
sudo apt install poppler-utils tesseract-ocr
```

**Python packages:**
```bash
pip install numpy opencv-python pillow sentence-transformers scikit-learn
```

**API server** (optional):
```bash
pip install fastapi uvicorn
```

> Stages 3-5 require only the Python standard library. If you only need timeline construction and anomaly detection, skip stages 1-2 with `--from 3`.

## Pipeline Stages

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| 1 | `pdf_to_markdown.py`, `image_to_markdown.py` | `data/*.pdf`, `data/*.jpeg` | `markdown_programmatic/*.md` |
| 2 | `markdown_to_json.py` | `markdown_programmatic/*.md` | `markdown_jsons/*.json` |
| 3 | `build_timeline.py` | Patient profile, manual entries, wearable XML, document JSONs | `timeline.json`, `snapshot_ground_truth/`, `anomaly_ground_truth/`, `timeline_viewer/index.html` |
| 4 | `detect_programmatic.py` | `timeline.json` | `snapshot_programmatic/`, `anomaly_programmatic/`, `decision_tree_viz/index.html` |
| 5 | Terminal dump | All outputs | Chronological listing of events, snapshots, and anomalies |

## Running Individual Scripts

```bash
# Stage 1: Convert clinical documents to markdown
python3 pdf_to_markdown.py       # PDFs → markdown
python3 image_to_markdown.py     # Chest X-ray JPEG → markdown (requires tesseract)

# Stage 2: Convert markdown to structured JSON
python3 markdown_to_json.py      # Heading clustering + JSON extraction

# Stage 3: Build timeline and viewer
python3 build_timeline.py        # Outputs timeline.json + viewer HTML

# Stage 4: Programmatic anomaly detection
python3 detect_programmatic.py   # 9 decision-tree detectors + grading report
```

## Outputs

### Terminal (via `run_pipeline.py`)
The pipeline prints a complete chronological dump:
- **Timeline events**: All 35 events with category, datetime, and title
- **Health snapshots**: Vitals, medication adherence, symptoms, and care team attention for each event
- **Anomalies**: 21 detected anomalies with severity, description, and clinical context

### Files
- `timeline.json` — Unified timeline with patient demographics, 35 events, and wearable data
- `snapshot_programmatic/*.json` — 35 health snapshots (one per event)
- `anomaly_programmatic/*.json` — 21 detected anomalies
- `timeline_viewer/index.html` — Interactive visual timeline (open in browser)
- `decision_tree_viz/index.html` — Decision tree visualization for all 9 detectors

### Interactive Viewer
```bash
open timeline_viewer/index.html
```
- Scroll to zoom, drag to pan
- Click events to see snapshots and anomalies in sidebar
- Category filter checkboxes in header
- Anomaly source badges (Ground Truth / Programmatic / Both)
- Wearable data tracks appear when zoomed into the April 8-9 monitoring window

## Health Snapshot API

The `api_server.py` exposes a REST API for querying patient health data. Requires `fastapi` and `uvicorn`.

```bash
# Start the API server (default port 8000)
python3 api_server.py

# Custom port
python3 api_server.py --port 9000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/patient` | Patient demographics and summary stats |
| `GET` | `/patient/snapshot` | Current (latest) health snapshot |
| `GET` | `/patient/snapshot/{event_id}` | Snapshot at a specific event |
| `GET` | `/patient/timeline` | Full event timeline (`?category=`, `?limit=`) |
| `GET` | `/patient/anomalies` | All anomalies (`?severity=`, `?category=`) |
| `GET` | `/patient/anomalies/{anomaly_id}` | Single anomaly detail |

### Example Queries

```bash
# Current health snapshot (latest vitals, meds, symptoms, care team alerts)
curl http://localhost:8000/patient/snapshot

# Snapshot at a specific event
curl http://localhost:8000/patient/snapshot/evt-030

# Only critical anomalies
curl http://localhost:8000/patient/anomalies?severity=critical

# Manual entry events, limited to 5
curl "http://localhost:8000/patient/timeline?category=manual_entry&limit=5"
```

Interactive API docs available at `http://localhost:8000/docs` (Swagger UI).

## Anomaly Detection

The `detect_programmatic.py` script implements 9 rule-based detectors:

1. **BP Detector** — Thresholds at 140/160/170 systolic with symptom correlation
2. **Glucose Detector** — Fasting (>130) and post-prandial (>180) thresholds
3. **Symptom Detector** — Severity scoring with concurrent vital correlation
4. **Medication Detector** — Late dose (>120 min) and missed dose detection
5. **Heart Rate Detector** — Sustained elevated resting HR from wearable data
6. **HRV Detector** — Low HRV (<30 ms) indicating sympathetic overdrive
7. **BP Trend Detector** — Rising systolic trend across 3+ readings
8. **Document Detector** — Keyword extraction from clinical reports
9. **Multi-Signal Correlation** — Cross-category convergence alerting

Grading against ground truth: **100% precision, 100% recall, 100% severity match**.

## Project Structure

```
data/                       # Raw clinical data (PDFs, JPEG, JSON, XML)
markdown_ground_truth/      # LLM-produced markdown (reference)
markdown_programmatic/      # Rule-based markdown conversion output
markdown_jsons/             # Structured JSON from markdown
snapshot_ground_truth/      # Manually curated health snapshots
snapshot_programmatic/      # Programmatic health snapshots
anomaly_ground_truth/       # Manually curated anomalies
anomaly_programmatic/       # Programmatic anomaly detection output
timeline_viewer/            # Interactive HTML viewer
decision_tree_viz/          # Decision tree visualization
api_server.py               # FastAPI health snapshot API
config.toml                 # Heading clustering threshold
PLANNING.md                 # Detailed planning log for each stage
```
