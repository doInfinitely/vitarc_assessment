#!/usr/bin/env python3
"""
Lightweight REST API for querying patient health data.

Endpoints:
  GET /patient                          → Patient demographics + summary
  GET /patient/snapshot                 → Current (latest) health snapshot
  GET /patient/snapshot/{event_id}      → Snapshot at a specific event
  GET /patient/timeline                 → Full event timeline
  GET /patient/anomalies                → All detected anomalies (filterable by severity)
  GET /patient/anomalies/{anomaly_id}   → Single anomaly detail

Usage:
  python3 api_server.py                 # starts on port 8000
  python3 api_server.py --port 9000     # custom port

Requires: pip install fastapi uvicorn
"""

import argparse
import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_timeline():
    path = os.path.join(BASE_DIR, "timeline.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "timeline.json not found. Run 'python3 run_pipeline.py --from 3' first."
        )
    with open(path) as f:
        return json.load(f)


def load_snapshots():
    snap_dir = os.path.join(BASE_DIR, "snapshot_programmatic")
    if not os.path.isdir(snap_dir):
        return {}
    snapshots = {}
    for fname in sorted(os.listdir(snap_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(snap_dir, fname)) as f:
                snap = json.load(f)
                snapshots[snap["event_id"]] = snap
    return snapshots


def load_anomalies():
    anom_dir = os.path.join(BASE_DIR, "anomaly_programmatic")
    if not os.path.isdir(anom_dir):
        return {}
    anomalies = {}
    for fname in sorted(os.listdir(anom_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(anom_dir, fname)) as f:
                anom = json.load(f)
                anomalies[anom["anomaly_id"]] = anom
    return anomalies


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

timeline = load_timeline()
snapshots = load_snapshots()
anomalies = load_anomalies()

app = FastAPI(
    title="Clinical Patient Timeline API",
    description="REST API for querying patient health snapshots, timeline events, and anomalies.",
    version="1.0.0",
)


@app.get("/patient")
def get_patient():
    """Patient demographics and summary statistics."""
    patient = timeline["patient"]
    events = timeline["events"]
    wearable = timeline.get("wearable_data", {})
    return {
        "patient_id": patient["patient_id"],
        "demographics": patient["demographics"],
        "allergies": patient.get("allergies", []),
        "chronic_conditions": patient.get("chronic_conditions", []),
        "medications": patient.get("medications", []),
        "summary": {
            "total_events": len(events),
            "total_snapshots": len(snapshots),
            "total_anomalies": len(anomalies),
            "wearable_readings": {
                k: len(v) for k, v in wearable.items()
            },
        },
    }


@app.get("/patient/snapshot")
def get_current_snapshot():
    """Return the health snapshot at the most recent event (current state)."""
    if not snapshots:
        raise HTTPException(status_code=404, detail="No snapshots available. Run the pipeline first.")
    latest_event_id = timeline["events"][-1]["id"]
    snap = snapshots.get(latest_event_id)
    if not snap:
        raise HTTPException(status_code=404, detail=f"No snapshot for latest event {latest_event_id}.")
    return snap


@app.get("/patient/snapshot/{event_id}")
def get_snapshot_at_event(event_id: str):
    """Return the health snapshot at a specific event."""
    snap = snapshots.get(event_id)
    if not snap:
        valid = sorted(snapshots.keys())
        raise HTTPException(
            status_code=404,
            detail=f"No snapshot for event '{event_id}'. Valid event IDs: {valid}",
        )
    return snap


@app.get("/patient/timeline")
def get_timeline(
    category: Optional[str] = Query(None, description="Filter by event category"),
    limit: Optional[int] = Query(None, description="Max events to return"),
):
    """Return the full event timeline, optionally filtered by category."""
    events = timeline["events"]
    if category:
        events = [e for e in events if e["category"] == category]
    if limit and limit > 0:
        events = events[:limit]
    return {
        "patient_id": timeline["patient"]["patient_id"],
        "total_events": len(events),
        "events": events,
    }


@app.get("/patient/anomalies")
def get_anomalies(
    severity: Optional[str] = Query(None, description="Filter by severity: critical, high, moderate, low"),
    category: Optional[str] = Query(None, description="Filter by anomaly category"),
):
    """Return all detected anomalies, optionally filtered."""
    results = list(anomalies.values())
    if severity:
        results = [a for a in results if a["severity"] == severity]
    if category:
        results = [a for a in results if a["category"] == category]
    by_severity = {}
    for a in results:
        by_severity[a["severity"]] = by_severity.get(a["severity"], 0) + 1
    return {
        "total": len(results),
        "by_severity": by_severity,
        "anomalies": results,
    }


@app.get("/patient/anomalies/{anomaly_id}")
def get_anomaly(anomaly_id: str):
    """Return a single anomaly by ID."""
    anom = anomalies.get(anomaly_id)
    if not anom:
        valid = sorted(anomalies.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Anomaly '{anomaly_id}' not found. Valid IDs: {valid}",
        )
    return anom


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the clinical data API server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"\n  Clinical Patient Timeline API")
    print(f"  Patient: {timeline['patient']['demographics']['name']}")
    print(f"  Events: {len(timeline['events'])} | Snapshots: {len(snapshots)} | Anomalies: {len(anomalies)}")
    print(f"  Docs: http://{args.host}:{args.port}/docs\n")

    uvicorn.run(app, host=args.host, port=args.port)
