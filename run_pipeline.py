#!/usr/bin/env python3
"""
End-to-end clinical data pipeline.

Stages:
  1. PDF/JPEG → Markdown   (pdf_to_markdown.py, image_to_markdown.py)
  2. Markdown → Structured JSON   (markdown_to_json.py)
  3. Data → Unified Timeline   (build_timeline.py)
  4. Timeline → Programmatic Snapshots + Anomaly Detection   (detect_programmatic.py)
  5. Terminal dump of timeline, snapshots, and anomalies in chronological order

Usage:
  python3 run_pipeline.py           # run full pipeline
  python3 run_pipeline.py --from 3  # skip stages 1-2, start from timeline construction
  python3 run_pipeline.py --dump    # only dump existing outputs (no processing)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def header(text):
    w = 70
    print()
    print("=" * w)
    print(f"  {text}")
    print("=" * w)


def run_script(name, description):
    """Run a Python script as a subprocess, streaming output."""
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        print(f"  [SKIP] {name} not found")
        return False
    print(f"\n  Running {name}...")
    print(f"  {description}")
    print("  " + "-" * 50)
    result = subprocess.run(
        [sys.executable, path],
        cwd=BASE_DIR,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"\n  [ERROR] {name} exited with code {result.returncode}")
        return False
    return True


def parse_dt_for_sort(dt_str):
    """Parse datetime string for sorting. Handles date-only and full datetime."""
    s = dt_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.min


def fmt_severity(severity):
    """Format severity with visual indicator."""
    icons = {"critical": "!!!", "high": "!! ", "moderate": "!  ", "low": ".  "}
    return icons.get(severity, "   ")


# ---------------------------------------------------------------------------
# Dump functions
# ---------------------------------------------------------------------------

def dump_timeline(timeline):
    """Print all events in chronological order."""
    header("TIMELINE EVENTS")
    events = timeline["events"]
    p = timeline["patient"]["demographics"]
    print(f"\n  Patient: {p['name']} | ID: {timeline['patient']['patient_id']} | "
          f"DOB: {p['date_of_birth']} | BMI: {p.get('bmi', 'N/A')}")
    print(f"  Events: {len(events)}")
    w = timeline.get("wearable_data", {})
    print(f"  Wearable: HR={len(w.get('heart_rate',[]))} SpO2={len(w.get('spo2',[]))} "
          f"Steps={len(w.get('steps',[]))} Sleep={len(w.get('sleep',[]))} HRV={len(w.get('hrv',[]))}")
    print()

    cat_pad = 12
    for e in events:
        cat = e["category"]
        planned = " (planned)" if e.get("planned") else ""
        dt_display = e["datetime"][:19]
        print(f"  {dt_display}  [{cat:<{cat_pad}}]  {e['title']}{planned}")


def dump_snapshots(snapshot_dir):
    """Print all snapshots in chronological order."""
    header("HEALTH SNAPSHOTS")

    files = sorted(f for f in os.listdir(snapshot_dir) if f.endswith(".json"))
    print(f"\n  Total: {len(files)} snapshots from {snapshot_dir}/\n")

    for fname in files:
        with open(os.path.join(snapshot_dir, fname)) as f:
            snap = json.load(f)

        eid = snap["event_id"]
        dt = snap["datetime"][:19]
        title = snap["event_title"]
        vitals = snap.get("most_recent_vitals", {})

        # Compact vitals line
        v_parts = []
        if vitals.get("blood_pressure"):
            bp = vitals["blood_pressure"]
            v_parts.append(f"BP:{bp['value']}")
        if vitals.get("heart_rate"):
            v_parts.append(f"HR:{vitals['heart_rate']['value']}")
        if vitals.get("spo2"):
            v_parts.append(f"SpO2:{vitals['spo2']['value']*100:.0f}%")
        if vitals.get("blood_glucose"):
            v_parts.append(f"Gluc:{vitals['blood_glucose']['value']}")
        vitals_str = " | ".join(v_parts) if v_parts else "no vitals"

        # Adherence
        adh = snap.get("medication_adherence_48h")
        adh_str = adh["summary"] if adh else "N/A"

        # Symptoms
        syms = snap.get("reported_symptoms", [])
        sym_str = ", ".join(f"{s['symptom']}({s['severity']}/10)" for s in syms) if syms else "none"

        # Attention
        attn = snap.get("care_team_attention", [])
        has_attn = attn and attn != ["No immediate attention items."]

        print(f"  {dt}  {eid}  {title}")
        print(f"    Vitals: {vitals_str}")
        print(f"    Meds: {adh_str} | Symptoms: {sym_str}")
        if has_attn:
            for a in attn:
                print(f"    >> {a}")
        print()


def dump_anomalies(anomaly_dir, label=""):
    """Print all anomalies in chronological order."""
    header(f"ANOMALIES{' — ' + label if label else ''}")

    files = sorted(f for f in os.listdir(anomaly_dir) if f.endswith(".json"))
    print(f"\n  Total: {len(files)} anomalies from {anomaly_dir}/\n")

    # Count by severity
    by_sev = {}
    anomalies = []
    for fname in files:
        with open(os.path.join(anomaly_dir, fname)) as f:
            a = json.load(f)
            anomalies.append(a)
            by_sev[a["severity"]] = by_sev.get(a["severity"], 0) + 1

    sev_summary = "  ".join(f"{s}:{c}" for s, c in
                            sorted(by_sev.items(), key=lambda x: ["critical","high","moderate","low"].index(x[0])))
    print(f"  Severity: {sev_summary}\n")

    for a in anomalies:
        sev = a["severity"].upper()
        icon = fmt_severity(a["severity"])
        dt = a["datetime"][:19]
        print(f"  {icon} [{sev:<8}]  {dt}  {a['anomaly_id']}  {a['title']}")
        # Wrap description to 80 chars
        desc = a.get("description", "")
        if len(desc) > 100:
            desc = desc[:97] + "..."
        print(f"      {desc}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the clinical data pipeline end-to-end.")
    parser.add_argument("--from", dest="start_stage", type=int, default=1,
                        help="Start from stage N (1=markdown, 2=json, 3=timeline, 4=detection, 5=dump)")
    parser.add_argument("--dump", action="store_true",
                        help="Only dump existing outputs, skip all processing")
    args = parser.parse_args()

    start = 5 if args.dump else args.start_stage

    # -----------------------------------------------------------------------
    # Stage 1: PDF/JPEG → Markdown
    # -----------------------------------------------------------------------
    if start <= 1:
        header("STAGE 1: Document → Markdown Conversion")
        run_script("pdf_to_markdown.py", "Converting PDFs to markdown via pdftotext heuristics")
        run_script("image_to_markdown.py", "Converting chest X-ray JPEG to markdown via Tesseract OCR")

    # -----------------------------------------------------------------------
    # Stage 2: Markdown → Structured JSON
    # -----------------------------------------------------------------------
    if start <= 2:
        header("STAGE 2: Markdown → Structured JSON")
        run_script("markdown_to_json.py", "Clustering headings and extracting structured data")

    # -----------------------------------------------------------------------
    # Stage 3: Build unified timeline
    # -----------------------------------------------------------------------
    if start <= 3:
        header("STAGE 3: Build Unified Timeline")
        run_script("build_timeline.py", "Ingesting patient profile, manual entries, wearable XML, and documents")

    # -----------------------------------------------------------------------
    # Stage 4: Programmatic detection
    # -----------------------------------------------------------------------
    if start <= 4:
        header("STAGE 4: Programmatic Anomaly Detection")
        run_script("detect_programmatic.py", "Running 9 decision-tree detectors and grading against ground truth")

    # -----------------------------------------------------------------------
    # Stage 5: Dump results
    # -----------------------------------------------------------------------
    header("STAGE 5: Pipeline Output")

    timeline_path = os.path.join(BASE_DIR, "timeline.json")
    snapshot_dir = os.path.join(BASE_DIR, "snapshot_programmatic")
    anomaly_dir = os.path.join(BASE_DIR, "anomaly_programmatic")

    if not os.path.exists(timeline_path):
        print("\n  [ERROR] timeline.json not found. Run the full pipeline first.")
        sys.exit(1)

    with open(timeline_path) as f:
        timeline = json.load(f)

    dump_timeline(timeline)

    if os.path.exists(snapshot_dir) and os.listdir(snapshot_dir):
        dump_snapshots(snapshot_dir)
    else:
        print("\n  [SKIP] No programmatic snapshots found. Run stage 4 first.")

    if os.path.exists(anomaly_dir) and os.listdir(anomaly_dir):
        dump_anomalies(anomaly_dir, "Programmatic")
    else:
        print("\n  [SKIP] No programmatic anomalies found. Run stage 4 first.")

    header("PIPELINE COMPLETE")
    print(f"\n  Timeline:   {timeline_path}")
    print(f"  Snapshots:  {snapshot_dir}/ ({len(os.listdir(snapshot_dir))} files)")
    print(f"  Anomalies:  {anomaly_dir}/ ({len(os.listdir(anomaly_dir))} files)")
    print(f"  Viewer:     {os.path.join(BASE_DIR, 'timeline_viewer', 'index.html')}")
    print(f"  Tree Viz:   {os.path.join(BASE_DIR, 'decision_tree_viz', 'index.html')}")
    print()


if __name__ == "__main__":
    main()
