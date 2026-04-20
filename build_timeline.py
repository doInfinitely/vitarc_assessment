#!/usr/bin/env python3
"""Build unified patient timeline, health snapshots, anomalies, and interactive viewer."""

import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_DIR = os.path.join(BASE_DIR, "markdown_jsons")

PROFILE_PATH = os.path.join(DATA_DIR, "patient_profile (1).json")
MANUAL_PATH = os.path.join(DATA_DIR, "manual_entries (1).json")
WEARABLE_PATH = os.path.join(DATA_DIR, "wearable_export (1).xml")

SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshot_ground_truth")
ANOMALY_DIR = os.path.join(BASE_DIR, "anomaly_ground_truth")
VIEWER_DIR = os.path.join(BASE_DIR, "timeline_viewer")
TIMELINE_PATH = os.path.join(BASE_DIR, "timeline.json")

AST = timezone(timedelta(hours=3))
UTC = timezone.utc


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def parse_xml_datetime(s):
    """Parse '2026-04-08 00:00:20 +0300' → UTC datetime."""
    s = s.strip()
    # Format: YYYY-MM-DD HH:MM:SS +HHMM
    m = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) ([+-]\d{4})", s)
    if not m:
        return None
    dt_str = f"{m.group(1)}T{m.group(2)}"
    off = m.group(3)
    sign = 1 if off[0] == "+" else -1
    off_h, off_m = int(off[1:3]), int(off[3:5])
    tz = timezone(timedelta(hours=sign * off_h, minutes=sign * off_m))
    dt = datetime.fromisoformat(dt_str).replace(tzinfo=tz)
    return dt.astimezone(UTC)


def parse_manual_timestamp(s):
    """Parse manual entry timestamps (Z or +03:00) → UTC datetime."""
    s = s.strip()
    if s.endswith("Z"):
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)
    return datetime.fromisoformat(s).astimezone(UTC)


def parse_doc_date(s):
    """Parse DD/MM/YYYY → date string YYYY-MM-DD."""
    parts = s.strip().split("/")
    if len(parts) == 3:
        return f"{parts[2]}-{parts[1]}-{parts[0]}"
    return s


def date_to_sort_key(d):
    """YYYY-MM-DD string → datetime for sorting."""
    return datetime.fromisoformat(d + "T00:00:00").replace(tzinfo=UTC)


def fmt_dt(dt):
    """Format datetime to ISO 8601 string without tz suffix for display."""
    if dt.tzinfo:
        dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_date(d):
    """Date-only events get T00:00:00 for consistency."""
    return d + "T00:00:00"


# ---------------------------------------------------------------------------
# Parse patient profile
# ---------------------------------------------------------------------------

def load_profile():
    with open(PROFILE_PATH) as f:
        return json.load(f)


def build_patient_context(profile):
    return {
        "patient_id": profile["patient_id"],
        "demographics": profile["demographics"],
        "allergies": profile["allergies"],
    }


def events_from_profile(profile):
    events = []

    # Conditions
    for c in profile["conditions"]:
        events.append({
            "datetime": fmt_date(c["diagnosed"]),
            "sort_key": date_to_sort_key(c["diagnosed"]),
            "category": "condition",
            "subcategory": "diagnosis",
            "title": f"{c['name']} Diagnosed",
            "source": "patient_profile",
            "planned": False,
            "data": c,
        })

    # Medications
    for m in profile["medications"]:
        events.append({
            "datetime": fmt_date(m["start_date"]),
            "sort_key": date_to_sort_key(m["start_date"]),
            "category": "medication",
            "subcategory": "prescription_start",
            "title": f"{m['name']} {m['dose']} Started",
            "source": "patient_profile",
            "planned": False,
            "data": m,
        })

    # Care team last visits
    for ct in profile["care_team"]:
        if ct.get("last_visit"):
            events.append({
                "datetime": fmt_date(ct["last_visit"]),
                "sort_key": date_to_sort_key(ct["last_visit"]),
                "category": "visit",
                "subcategory": "clinic_visit",
                "title": f"Visit: {ct['name']} ({ct['specialty']})",
                "source": "patient_profile",
                "planned": False,
                "data": ct,
            })

    # Care team next appointments
    for ct in profile["care_team"]:
        if ct.get("next_appointment"):
            events.append({
                "datetime": fmt_date(ct["next_appointment"]),
                "sort_key": date_to_sort_key(ct["next_appointment"]),
                "category": "appointment",
                "subcategory": "scheduled",
                "title": f"Appointment: {ct['name']} ({ct['specialty']})",
                "source": "patient_profile",
                "planned": True,
                "data": ct,
            })

    return events


# ---------------------------------------------------------------------------
# Parse document JSONs
# ---------------------------------------------------------------------------

DOC_TITLES = {
    "echocardiogram_fakeeh.json": "Transthoracic Echocardiogram Report",
    "renal_ultrasound_sgh.json": "Renal Ultrasound Report",
    "chest_xray_kauh.json": "Chest X-Ray Report",
    "lab_cbc_kauh.json": "Complete Blood Count (CBC) Report",
}


def events_from_documents():
    events = []
    for fname in sorted(os.listdir(JSON_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(JSON_DIR, fname)
        with open(path) as f:
            doc = json.load(f)
        date_str = parse_doc_date(doc["date"]) if doc.get("date") else None
        if not date_str:
            continue
        title = DOC_TITLES.get(fname, fname.replace(".json", "").replace("_", " ").title())
        events.append({
            "datetime": fmt_date(date_str),
            "sort_key": date_to_sort_key(date_str),
            "category": "document",
            "subcategory": "clinical_report",
            "title": title,
            "source": "document",
            "source_file": fname,
            "planned": False,
            "data": {"structured_document": doc["structured_document"], "original_path": doc.get("original_document_path")},
        })
    return events


# ---------------------------------------------------------------------------
# Parse manual entries
# ---------------------------------------------------------------------------

def normalize_manual_entry(entry):
    """Normalize BP key variants and glucose units."""
    values = dict(entry.get("values", {}))

    # BP: ME-015 uses systolic/diastolic without _mmhg suffix
    if entry["type"] == "blood_pressure":
        if "systolic" in values and "systolic_mmhg" not in values:
            values["systolic_mmhg"] = values.pop("systolic")
        if "diastolic" in values and "diastolic_mmhg" not in values:
            values["diastolic_mmhg"] = values.pop("diastolic")

    # Glucose: ME-012 uses mmol/L
    if entry["type"] == "blood_glucose":
        if "glucose_mmol_l" in values and "glucose_mg_dl" not in values:
            values["glucose_mg_dl"] = round(values["glucose_mmol_l"] * 18.018, 1)
            values["original_unit"] = "mmol/L"
            values["original_value"] = values.pop("glucose_mmol_l")

    return values


def build_manual_title(entry, values):
    t = entry["type"]
    if t == "blood_pressure":
        return f"Blood Pressure: {values.get('systolic_mmhg')}/{values.get('diastolic_mmhg')} mmHg"
    if t == "blood_glucose":
        return f"Blood Glucose: {values.get('glucose_mg_dl')} mg/dL"
    if t == "medication_taken":
        meds = values.get("medications", [])
        return f"Medication Taken: {', '.join(meds)}"
    if t == "symptom":
        return f"Symptom: {values.get('symptom', '').replace('_', ' ').title()} (severity {values.get('severity')}/10)"
    if t == "inhaler_use":
        return f"Inhaler Use: {values.get('medication')} {values.get('puffs')} puffs"
    return t.replace("_", " ").title()


def events_from_manual():
    with open(MANUAL_PATH) as f:
        data = json.load(f)
    events = []
    for entry in data["entries"]:
        dt = parse_manual_timestamp(entry["timestamp"])
        values = normalize_manual_entry(entry)
        events.append({
            "datetime": fmt_dt(dt),
            "sort_key": dt,
            "category": "manual_entry",
            "subcategory": entry["type"],
            "title": build_manual_title(entry, values),
            "source": "manual_entry",
            "planned": False,
            "data": {
                "entry_id": entry["id"],
                "type": entry["type"],
                "values": values,
                "context": entry.get("context"),
                "notes": entry.get("notes"),
            },
        })
    return events


# ---------------------------------------------------------------------------
# Parse wearable XML
# ---------------------------------------------------------------------------

def parse_wearable():
    tree = ET.parse(WEARABLE_PATH)
    root = tree.getroot()

    heart_rate = []
    spo2 = []
    steps = []
    sleep = []
    hrv = []

    SLEEP_STAGE_MAP = {
        "HKCategoryValueSleepAnalysisAsleepCore": "core",
        "HKCategoryValueSleepAnalysisAsleepDeep": "deep",
        "HKCategoryValueSleepAnalysisAsleepREM": "rem",
        "HKCategoryValueSleepAnalysisAwake": "awake",
    }

    for rec in root.findall("Record"):
        rtype = rec.get("type")
        source = rec.get("sourceName", "")

        if rtype == "HKQuantityTypeIdentifierHeartRate":
            dt = parse_xml_datetime(rec.get("startDate"))
            val = rec.get("value")
            if dt and val:
                heart_rate.append({
                    "datetime": fmt_dt(dt),
                    "value": int(float(val)),
                    "unit": "bpm",
                    "source": "Fitbit Sense 2" if "Fitbit" in source else "Apple Watch",
                })

        elif rtype == "HKQuantityTypeIdentifierOxygenSaturation":
            dt = parse_xml_datetime(rec.get("startDate"))
            val = rec.get("value", "").strip()
            if dt and val:  # skip empty value records
                spo2.append({
                    "datetime": fmt_dt(dt),
                    "value": float(val),
                    "unit": "%",
                    "source": "Fitbit Sense 2" if "Fitbit" in source else "Apple Watch",
                })

        elif rtype == "HKQuantityTypeIdentifierStepCount":
            start = parse_xml_datetime(rec.get("startDate"))
            end = parse_xml_datetime(rec.get("endDate"))
            val = rec.get("value")
            if start and end and val:
                steps.append({
                    "start": fmt_dt(start),
                    "end": fmt_dt(end),
                    "value": int(float(val)),
                    "unit": "count",
                    "source": "iPhone" if "iPhone" in source else "Apple Watch",
                })

        elif rtype == "HKCategoryTypeIdentifierSleepAnalysis":
            start = parse_xml_datetime(rec.get("startDate"))
            end = parse_xml_datetime(rec.get("endDate"))
            stage_raw = rec.get("value", "")
            stage = SLEEP_STAGE_MAP.get(stage_raw, stage_raw)
            if start and end:
                sleep.append({
                    "start": fmt_dt(start),
                    "end": fmt_dt(end),
                    "stage": stage,
                })

        elif rtype == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":
            dt = parse_xml_datetime(rec.get("startDate"))
            val = rec.get("value")
            if dt and val:
                hrv.append({
                    "datetime": fmt_dt(dt),
                    "value": int(float(val)),
                    "unit": "ms",
                })

    # Sort each track by datetime
    heart_rate.sort(key=lambda x: x["datetime"])
    spo2.sort(key=lambda x: x["datetime"])
    steps.sort(key=lambda x: x["start"])
    sleep.sort(key=lambda x: x["start"])
    hrv.sort(key=lambda x: x["datetime"])

    return {
        "heart_rate": heart_rate,
        "spo2": spo2,
        "steps": steps,
        "sleep": sleep,
        "hrv": hrv,
    }


# ---------------------------------------------------------------------------
# Build full timeline
# ---------------------------------------------------------------------------

def build_timeline():
    profile = load_profile()
    patient = build_patient_context(profile)

    # Collect all events
    all_events = []
    all_events.extend(events_from_profile(profile))
    all_events.extend(events_from_documents())
    all_events.extend(events_from_manual())

    # Sort chronologically
    all_events.sort(key=lambda e: e["sort_key"])

    # Assign IDs
    for i, evt in enumerate(all_events, 1):
        evt["id"] = f"evt-{i:03d}"
        del evt["sort_key"]

    wearable = parse_wearable()

    timeline = {
        "patient": patient,
        "events": all_events,
        "wearable_data": wearable,
    }
    return timeline, profile


# ---------------------------------------------------------------------------
# Health snapshot generation
# ---------------------------------------------------------------------------

def find_most_recent_bp(events, up_to_idx):
    """Find most recent BP reading up to (inclusive) event index."""
    for i in range(up_to_idx, -1, -1):
        e = events[i]
        if e.get("category") == "manual_entry" and e.get("data", {}).get("type") == "blood_pressure":
            v = e["data"]["values"]
            return {
                "value": f"{v['systolic_mmhg']}/{v['diastolic_mmhg']}",
                "systolic": v["systolic_mmhg"],
                "diastolic": v["diastolic_mmhg"],
                "unit": "mmHg",
                "when": e["datetime"],
                "how": "manual_entry",
            }
    return None


def find_most_recent_glucose(events, up_to_idx):
    for i in range(up_to_idx, -1, -1):
        e = events[i]
        if e.get("category") == "manual_entry" and e.get("data", {}).get("type") == "blood_glucose":
            v = e["data"]["values"]
            return {
                "value": v["glucose_mg_dl"],
                "unit": "mg/dL",
                "when": e["datetime"],
                "how": "manual_entry",
            }
    return None


def find_most_recent_wearable(track, up_to_dt_str, key="datetime"):
    """Find most recent wearable reading ≤ up_to_dt_str."""
    best = None
    for r in track:
        t = r.get(key, r.get("datetime", ""))
        if t <= up_to_dt_str:
            best = r
        else:
            break
    return best


def get_medication_adherence_48h(events, up_to_idx, profile):
    """Check medication adherence in the 48h window before event."""
    evt_dt = events[up_to_idx]["datetime"]
    if "T" not in evt_dt:
        return None

    try:
        evt_time = datetime.fromisoformat(evt_dt.replace("Z", "+00:00"))
    except ValueError:
        return None

    window_start = evt_time - timedelta(hours=48)
    ws_str = fmt_dt(window_start)

    # Collect med-taken events in window
    taken_entries = []
    for i in range(up_to_idx + 1):
        e = events[i]
        if e["datetime"] < ws_str:
            continue
        if e["datetime"] > evt_dt:
            break
        if e.get("category") == "manual_entry" and e.get("data", {}).get("type") in ("medication_taken", "inhaler_use"):
            taken_entries.append(e)

    if not taken_entries:
        return {"summary": "No medication records in 48h window", "details": []}

    # Build details for scheduled meds
    scheduled_meds = [m for m in profile["medications"] if m["frequency"] != "as needed"]
    details = []
    for te in taken_entries:
        d = te["data"]
        if d["type"] == "medication_taken":
            for med_name in d["values"].get("medications", []):
                # Find matching scheduled med
                sched = None
                for sm in scheduled_meds:
                    if sm["name"].lower() in med_name.lower() or med_name.lower().startswith(sm["name"].lower()):
                        sched = sm
                        break

                entry = {"medication": med_name, "taken_at": te["datetime"]}
                if sched and sched.get("scheduled_time"):
                    sched_times = [t.strip() for t in sched["scheduled_time"].split(",")]
                    # Find which scheduled time this is closest to
                    taken_time = datetime.fromisoformat(te["datetime"].replace("Z", "+00:00"))
                    taken_hm = taken_time.hour * 60 + taken_time.minute
                    best_diff = None
                    best_sched = None
                    for st in sched_times:
                        sh, sm_val = int(st.split(":")[0]), int(st.split(":")[1])
                        sched_min = sh * 60 + sm_val
                        diff = taken_hm - sched_min
                        if best_diff is None or abs(diff) < abs(best_diff):
                            best_diff = diff
                            best_sched = st
                    entry["scheduled"] = best_sched
                    if best_diff is not None:
                        entry["delay_minutes"] = best_diff
                        if abs(best_diff) <= 30:
                            entry["status"] = "on_time"
                        elif best_diff > 0:
                            entry["status"] = "late"
                        else:
                            entry["status"] = "early"

                details.append(entry)

        elif d["type"] == "inhaler_use":
            details.append({
                "medication": f"{d['values'].get('medication', 'Inhaler')} {d['values'].get('puffs', '')} puffs",
                "taken_at": te["datetime"],
                "status": "as_needed",
            })

    # Determine summary
    late_count = sum(1 for d in details if d.get("status") == "late")
    on_time_count = sum(1 for d in details if d.get("status") == "on_time")
    if late_count == 0:
        summary = "Fully adherent"
    elif on_time_count > 0:
        summary = "Partially adherent"
    else:
        summary = "Poor adherence"

    return {"summary": summary, "details": details}


def get_reported_symptoms(events, up_to_idx, window_hours=24):
    """Get symptoms reported within window_hours before event."""
    evt_dt = events[up_to_idx]["datetime"]
    if "T" not in evt_dt:
        return []

    try:
        evt_time = datetime.fromisoformat(evt_dt.replace("Z", "+00:00"))
    except ValueError:
        return []
    window_start = evt_time - timedelta(hours=window_hours)
    ws_str = fmt_dt(window_start)

    symptoms = []
    for i in range(up_to_idx + 1):
        e = events[i]
        if e["datetime"] < ws_str:
            continue
        if e["datetime"] > evt_dt:
            break
        if e.get("category") == "manual_entry" and e.get("data", {}).get("type") == "symptom":
            v = e["data"]["values"]
            symptoms.append({
                "symptom": v.get("symptom", "").replace("_", " "),
                "severity": v.get("severity"),
                "when": e["datetime"],
                "notes": e["data"].get("notes"),
            })
    return symptoms


def get_document_findings(events, up_to_idx):
    """Summarize clinical findings from document events up to this point."""
    findings = []
    for i in range(up_to_idx + 1):
        e = events[i]
        if e["category"] == "document":
            findings.append({"title": e["title"], "date": e["datetime"][:10]})
    return findings


def build_clinical_summary(evt, vitals, symptoms, doc_findings, adherence):
    """Generate a clinical findings summary string."""
    parts = []

    if evt["category"] == "condition":
        parts.append(f"Diagnosis of {evt['data'].get('name', '')} ({evt['data'].get('severity', '')}).")
    elif evt["category"] == "medication":
        parts.append(f"Started {evt['data'].get('name', '')} {evt['data'].get('dose', '')} for {evt['data'].get('indication', '')}.")
    elif evt["category"] == "visit":
        parts.append(f"Clinical visit with {evt['data'].get('name', '')} ({evt['data'].get('specialty', '')}).")
    elif evt["category"] == "document":
        parts.append(f"Clinical document: {evt['title']}.")
    elif evt["category"] == "appointment":
        parts.append(f"Scheduled appointment with {evt['data'].get('name', '')}.")
    elif evt["category"] == "manual_entry":
        parts.append(f"Patient-reported: {evt['title']}.")

    if vitals.get("blood_pressure"):
        bp = vitals["blood_pressure"]
        sys_val = bp.get("systolic", 0)
        if sys_val >= 180:
            parts.append(f"CRITICAL: BP {bp['value']} — hypertensive crisis.")
        elif sys_val >= 160:
            parts.append(f"HIGH: BP {bp['value']} — hypertensive urgency.")
        elif sys_val >= 140:
            parts.append(f"ELEVATED: BP {bp['value']} — Stage 2 hypertension range.")

    if vitals.get("blood_glucose"):
        bg = vitals["blood_glucose"]
        val = bg.get("value", 0)
        if val >= 250:
            parts.append(f"CRITICAL: Blood glucose {val} mg/dL — severe hyperglycemia.")
        elif val >= 180:
            parts.append(f"HIGH: Blood glucose {val} mg/dL — hyperglycemia.")
        elif val >= 126:
            parts.append(f"ELEVATED: Blood glucose {val} mg/dL — above target.")

    if symptoms:
        sym_list = ", ".join(f"{s['symptom']} ({s['severity']}/10)" for s in symptoms)
        parts.append(f"Active symptoms: {sym_list}.")

    if adherence and adherence.get("summary") != "Fully adherent":
        parts.append(f"Medication adherence: {adherence['summary']}.")

    return " ".join(parts) if parts else "No significant clinical findings at this timepoint."


def build_care_team_attention(evt, vitals, symptoms):
    """Generate care team attention items."""
    items = []

    if vitals.get("blood_pressure"):
        sys_val = vitals["blood_pressure"].get("systolic", 0)
        if sys_val >= 180:
            items.append("URGENT: Hypertensive crisis — immediate evaluation required, consider ER.")
        elif sys_val >= 160:
            items.append("Hypertensive urgency — contact physician, consider ER evaluation.")
        elif sys_val >= 140:
            items.append("Elevated BP above target — review antihypertensive regimen.")

    if vitals.get("blood_glucose"):
        val = vitals["blood_glucose"].get("value", 0)
        if val >= 250:
            items.append("Severe hyperglycemia — assess for DKA/HHS, adjust insulin.")
        elif val >= 200:
            items.append("Significant hyperglycemia — review diabetes management plan.")
        elif val >= 180:
            items.append("Blood glucose above post-prandial target — consider medication adjustment.")

    high_symptoms = [s for s in symptoms if (s.get("severity") or 0) >= 5]
    if high_symptoms:
        for s in high_symptoms:
            items.append(f"Patient reporting {s['symptom']} (severity {s['severity']}/10) — assess and manage.")

    return items if items else ["No immediate attention items."]


def generate_snapshot(evt, idx, events, wearable, profile):
    """Generate health snapshot for a single event."""
    # Most recent vitals
    vitals = {}
    bp = find_most_recent_bp(events, idx)
    if bp:
        vitals["blood_pressure"] = bp

    bg = find_most_recent_glucose(events, idx)
    if bg:
        vitals["blood_glucose"] = bg

    # Wearable vitals
    hr_reading = find_most_recent_wearable(wearable["heart_rate"], evt["datetime"])
    if hr_reading:
        vitals["heart_rate"] = {
            "value": hr_reading["value"],
            "unit": "bpm",
            "when": hr_reading["datetime"],
            "how": hr_reading["source"],
        }

    spo2_reading = find_most_recent_wearable(wearable["spo2"], evt["datetime"])
    if spo2_reading:
        vitals["spo2"] = {
            "value": spo2_reading["value"],
            "unit": "%",
            "when": spo2_reading["datetime"],
            "how": spo2_reading["source"],
        }

    adherence = get_medication_adherence_48h(events, idx, profile)
    symptoms = get_reported_symptoms(events, idx)
    doc_findings = get_document_findings(events, idx)
    clinical_summary = build_clinical_summary(evt, vitals, symptoms, doc_findings, adherence)
    attention = build_care_team_attention(evt, vitals, symptoms)

    snapshot = {
        "event_id": evt["id"],
        "datetime": evt["datetime"],
        "event_title": evt["title"],
        "most_recent_vitals": vitals,
        "medication_adherence_48h": adherence,
        "reported_symptoms": symptoms,
        "clinical_findings_summary": clinical_summary,
        "care_team_attention": attention,
    }
    return snapshot


# ---------------------------------------------------------------------------
# Anomaly generation
# ---------------------------------------------------------------------------

def generate_anomalies(events, wearable, profile):
    """Generate all anomaly ground truth files based on clinical knowledge."""
    anomalies = []

    # Build a lookup for quick access
    def evt_by_id(eid):
        for e in events:
            if e["id"] == eid:
                return e
        return None

    def find_event_near(dt_str, category=None, subcategory=None):
        """Find event closest to given datetime string."""
        best = None
        for e in events:
            if category and e["category"] != category:
                continue
            if subcategory and e.get("subcategory") != subcategory:
                continue
            if best is None or abs_dt_diff(e["datetime"], dt_str) < abs_dt_diff(best["datetime"], dt_str):
                best = e
        return best

    def abs_dt_diff(a, b):
        try:
            ta = datetime.fromisoformat(a.replace("Z", "+00:00"))
            tb = datetime.fromisoformat(b.replace("Z", "+00:00"))
            return abs((ta - tb).total_seconds())
        except Exception:
            return 1e12

    # Helper: collect BP readings
    bp_readings = []
    for e in events:
        if e.get("category") == "manual_entry" and e.get("data", {}).get("type") == "blood_pressure":
            v = e["data"]["values"]
            bp_readings.append({
                "datetime": e["datetime"],
                "event_id": e["id"],
                "systolic": v["systolic_mmhg"],
                "diastolic": v["diastolic_mmhg"],
            })

    # Helper: collect glucose readings
    glucose_readings = []
    for e in events:
        if e.get("category") == "manual_entry" and e.get("data", {}).get("type") == "blood_glucose":
            v = e["data"]["values"]
            glucose_readings.append({
                "datetime": e["datetime"],
                "event_id": e["id"],
                "value": v["glucose_mg_dl"],
                "context": e["data"].get("context"),
            })

    anom_id = 0

    def make_anomaly(**kwargs):
        nonlocal anom_id
        anom_id += 1
        a = {"anomaly_id": f"anom-{anom_id:03d}"}
        a.update(kwargs)
        return a

    # --- 1. Elevated morning BP 142/88 ---
    e = find_event_near("2026-04-08T07:45:00Z", "manual_entry", "blood_pressure")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-08T07:45:00Z",
        category="vital_sign",
        severity="moderate",
        title="Elevated Morning Blood Pressure 142/88 mmHg",
        description="Morning fasting BP 142/88 mmHg exceeds target of <130/80 for patient with HTN + diabetes. Stage 2 hypertension range despite dual antihypertensive therapy (Amlodipine 10mg + Losartan 100mg).",
        related_data=[{"type": "blood_pressure", "value": "142/88", "context": "morning_fasting"}],
        clinical_context="Patient has Stage 2 Essential Hypertension diagnosed 2016, currently on Amlodipine 10mg + Losartan 100mg daily. Target BP for diabetic patient is <130/80 mmHg per guidelines.",
    ))

    # --- 2. Elevated fasting glucose 168 ---
    e = find_event_near("2026-04-08T07:50:00Z", "manual_entry", "blood_glucose")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-08T07:50:00Z",
        category="vital_sign",
        severity="moderate",
        title="Elevated Fasting Blood Glucose 168 mg/dL",
        description="Fasting glucose 168 mg/dL significantly above target of <130 mg/dL for T2DM management. Indicates inadequate overnight glycemic control.",
        related_data=[{"type": "blood_glucose", "value": 168, "unit": "mg/dL", "context": "fasting"}],
        clinical_context="Patient has T2DM with hyperglycemia (E11.65), on Metformin 1000mg BID. ADA target fasting glucose is 80-130 mg/dL.",
    ))

    # --- 3. Post-prandial glucose 195 ---
    e = find_event_near("2026-04-08T13:30:00Z", "manual_entry", "blood_glucose")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-08T13:30:00Z",
        category="vital_sign",
        severity="moderate",
        title="Elevated Post-Prandial Blood Glucose 195 mg/dL",
        description="Post-prandial glucose 195 mg/dL exceeds target of <180 mg/dL. Patient notes 'Had kabsa for lunch' — high-carbohydrate meal likely contributed.",
        related_data=[{"type": "blood_glucose", "value": 195, "unit": "mg/dL", "context": "post_meal", "notes": "Had kabsa for lunch"}],
        clinical_context="ADA recommends post-prandial glucose <180 mg/dL. Dietary counseling may be beneficial.",
    ))

    # --- 4. Mild headache at bedtime ---
    e = find_event_near("2026-04-08T23:30:00Z", "manual_entry", "symptom")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-08T23:30:00Z",
        category="symptom",
        severity="low",
        title="Mild Headache at Bedtime",
        description="Patient reports mild headache (severity 3/10) at bedtime, took paracetamol. May be tension headache or related to elevated BP earlier in day (142/88). Evening BP had improved to 134/82.",
        related_data=[
            {"type": "symptom", "symptom": "mild_headache", "severity": 3},
            {"type": "blood_pressure", "value": "134/82", "context": "evening", "when": "2026-04-08T18:00:00Z"},
        ],
        clinical_context="Headache in hypertensive patient warrants monitoring. Evening BP 134/82 improved from morning 142/88.",
    ))

    # --- 5. Elevated morning BP 148/92 (rising trend) ---
    e = find_event_near("2026-04-09T07:30:00Z", "manual_entry", "blood_pressure")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-09T07:30:00Z",
        category="vital_sign",
        severity="moderate",
        title="Elevated Morning Blood Pressure 148/92 mmHg (Rising Trend)",
        description="Morning fasting BP 148/92 mmHg, up from 142/88 yesterday. Patient notes 'Didn't sleep well'. Rising BP trend is concerning. Sleep data confirms poor sleep quality (awake period 00:10-00:35 UTC).",
        related_data=[
            {"type": "blood_pressure", "value": "148/92", "context": "morning_fasting"},
            {"type": "blood_pressure", "value": "142/88", "context": "morning_fasting", "when": "2026-04-08T07:45:00Z"},
            {"type": "sleep", "note": "Poor sleep with awake period"},
        ],
        clinical_context="BP rising from 142→148 systolic over 24h despite medication. Poor sleep can elevate morning BP. Patient notes sleep difficulties.",
    ))

    # --- 6. Elevated fasting glucose 182 (rising) ---
    e = find_event_near("2026-04-09T07:35:00Z", "manual_entry", "blood_glucose")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-09T07:35:00Z",
        category="vital_sign",
        severity="moderate",
        title="Elevated Fasting Blood Glucose 182 mg/dL (Rising Trend)",
        description="Fasting glucose 182 mg/dL, rising from 168 mg/dL yesterday. Well above target of <130 mg/dL. Indicates worsening glycemic control.",
        related_data=[
            {"type": "blood_glucose", "value": 182, "context": "fasting"},
            {"type": "blood_glucose", "value": 168, "context": "fasting", "when": "2026-04-08T07:50:00Z"},
        ],
        clinical_context="Rising fasting glucose (168→182) over 24h suggests inadequate overnight control. Metformin 1000mg BID may need augmentation.",
    ))

    # --- 7. Morning medications 3h45m late ---
    e = find_event_near("2026-04-09T11:45:00Z", "manual_entry", "medication_taken")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-09T11:45:00Z",
        category="medication",
        severity="moderate",
        title="Morning Medications Taken 3h45m Late",
        description="Amlodipine 10mg, Losartan 100mg, and Metformin 1000mg taken at 11:45 instead of scheduled 08:00 — 225 minutes late. Patient notes 'Forgot to take at 8am, took late'. Delay in antihypertensives may contribute to elevated BP readings.",
        related_data=[
            {"type": "medication_taken", "medications": ["Amlodipine 10mg", "Losartan 100mg", "Metformin 1000mg"]},
            {"type": "note", "value": "Scheduled time: 08:00, taken: 11:45, delay: 225 minutes"},
        ],
        clinical_context="Late antihypertensive dosing reduces 24h coverage. Given rising BP trend, medication timing is clinically significant.",
    ))

    # --- 8. Post-prandial glucose ~231 mg/dL (12.8 mmol/L) ---
    e = find_event_near("2026-04-09T11:00:00Z", "manual_entry", "blood_glucose")  # ME-012 at 14:00+03:00 = 11:00Z
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-09T11:00:00Z",
        category="vital_sign",
        severity="high",
        title="High Post-Prandial Blood Glucose ~231 mg/dL (12.8 mmol/L)",
        description="Post-prandial glucose 12.8 mmol/L (~231 mg/dL) significantly exceeds target of <180 mg/dL. Highest glucose reading recorded. Rising trend: 168→195→182→231 mg/dL across the two-day monitoring period.",
        related_data=[
            {"type": "blood_glucose", "value": 231, "unit": "mg/dL", "original": "12.8 mmol/L", "context": "post_meal"},
        ],
        clinical_context="Severe post-prandial hyperglycemia in T2DM patient on Metformin monotherapy. Consider adding second-line agent (SGLT2i, GLP-1 RA, or sulfonylurea).",
    ))

    # --- 9. BP 162/98 with dizziness ---
    e = find_event_near("2026-04-09T16:30:00Z", "manual_entry", "blood_pressure")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-09T16:30:00Z",
        category="vital_sign",
        severity="high",
        title="Blood Pressure 162/98 mmHg with Dizziness",
        description="BP 162/98 mmHg with patient reporting dizziness. Continued rising trend: 134→142→148→162 systolic over 36 hours. Symptomatic hypertension requires clinical attention.",
        related_data=[
            {"type": "blood_pressure", "value": "162/98", "context": "afternoon"},
            {"type": "symptom", "symptom": "dizziness", "notes": "Feeling dizzy"},
        ],
        clinical_context="Symptomatic Stage 2 hypertension with end-organ disease risk. Morning antihypertensives taken 3h45m late may be contributing factor.",
    ))

    # --- 10. Dizziness severity 6/10 ---
    e = find_event_near("2026-04-09T16:35:00Z", "manual_entry", "symptom")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-09T16:35:00Z",
        category="symptom",
        severity="moderate",
        title="Dizziness Severity 6/10",
        description="Patient reports dizziness severity 6/10, started around 3pm and getting worse. Concurrent with BP 162/98. Dizziness in context of hypertensive urgency may indicate cerebrovascular compromise.",
        related_data=[
            {"type": "symptom", "symptom": "dizziness", "severity": 6},
            {"type": "blood_pressure", "value": "162/98", "when": "2026-04-09T16:30:00Z"},
        ],
        clinical_context="Dizziness with significantly elevated BP requires assessment for hypertensive encephalopathy, TIA, or other end-organ damage.",
    ))

    # --- 11. BP 171/101 — hypertensive urgency ---
    e = find_event_near("2026-04-09T19:00:00Z", "manual_entry", "blood_pressure")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-09T19:00:00Z",
        category="vital_sign",
        severity="critical",
        title="Hypertensive Urgency — BP 171/101 mmHg",
        description="BP 171/101 mmHg with symptoms (dizziness, chest tightness). Represents a rising trend: 134→142→148→162→171 systolic over 36 hours. Patient asks 'Should I go to ER?' — clinical evaluation strongly recommended.",
        related_data=[
            {"type": "blood_pressure", "value": "171/101", "context": "evening"},
            {"type": "symptom", "symptom": "dizziness", "severity": 6},
            {"type": "trend", "values": [134, 142, 148, 162, 171], "parameter": "systolic_bp"},
        ],
        clinical_context="Hypertensive urgency (SBP ≥160 with symptoms) in patient with known LVH, cardiomegaly, and early nephropathy. ER evaluation recommended for end-organ assessment.",
    ))

    # --- 12. Chest tightness ---
    e = find_event_near("2026-04-09T19:05:00Z", "manual_entry", "symptom")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-09T19:05:00Z",
        category="symptom",
        severity="moderate",
        title="Chest Tightness with Hypertensive Urgency",
        description="Patient reports chest tightness (severity 4/10), notes 'might be anxiety'. In context of BP 171/101 and resting HR 91-104 bpm, chest tightness may indicate cardiac strain. Patient has known LVH and mild cardiomegaly.",
        related_data=[
            {"type": "symptom", "symptom": "chest_tightness", "severity": 4},
            {"type": "blood_pressure", "value": "171/101", "when": "2026-04-09T19:00:00Z"},
            {"type": "heart_rate", "range": "91-104 bpm", "when": "2026-04-09 evening"},
        ],
        clinical_context="Chest tightness in patient with HTN crisis, LVH, and cardiomegaly requires cardiac workup to rule out ACS. NSAIDs contraindicated (allergy: bronchospasm).",
    ))

    # --- 13. Sustained elevated resting HR ---
    anomalies.append(make_anomaly(
        event_id=e["id"],  # associate with chest tightness event
        datetime="2026-04-09T15:00:00Z",
        category="vital_sign",
        severity="high",
        title="Sustained Elevated Resting Heart Rate 91-104 bpm",
        description="Apple Watch records show sustained resting HR 91-104 bpm throughout the evening of April 9 (15:00-23:55 UTC), well above expected resting range of 60-80 bpm. Likely reflects sympathetic activation from pain, anxiety, and hypertensive crisis.",
        related_data=[
            {"type": "heart_rate", "range": "91-104 bpm", "duration": "~9 hours", "source": "Apple Watch"},
            {"type": "blood_pressure", "value": "171/101", "when": "2026-04-09T19:00:00Z"},
        ],
        clinical_context="Resting tachycardia in hypertensive patient with LVH increases myocardial oxygen demand. Combined with chest tightness, warrants cardiac monitoring.",
    ))

    # --- 14. Low HRV 25ms ---
    anomalies.append(make_anomaly(
        event_id=find_event_near("2026-04-09T19:00:00Z", "manual_entry", "blood_pressure")["id"],
        datetime="2026-04-09T19:00:00Z",
        category="vital_sign",
        severity="moderate",
        title="Low Heart Rate Variability 25 ms (Sympathetic Overdrive)",
        description="HRV declined from 72→75→55→63→32→28→25 ms over 48h monitoring period. Final reading of 25 ms indicates significant sympathetic overdrive and reduced parasympathetic tone. Low HRV is associated with increased cardiovascular risk.",
        related_data=[
            {"type": "hrv", "trend": [72, 75, 55, 63, 32, 28, 25], "unit": "ms"},
            {"type": "heart_rate", "range": "91-104 bpm"},
        ],
        clinical_context="Declining HRV with concurrent tachycardia and hypertensive crisis indicates autonomic dysfunction and heightened cardiovascular risk.",
    ))

    # --- 15. Missing Atorvastatin Day 2 ---
    anomalies.append(make_anomaly(
        event_id=find_event_near("2026-04-09T20:30:00Z", "manual_entry", "medication_taken")["id"],
        datetime="2026-04-09T22:00:00Z",
        category="medication",
        severity="low",
        title="Missing Atorvastatin Dose (Day 2)",
        description="No record of Atorvastatin 20mg taken on April 9 (scheduled 22:00). On April 8, Atorvastatin was taken at 22:05. Evening Metformin was taken at 20:30 but statin dose appears missed.",
        related_data=[
            {"type": "medication_schedule", "medication": "Atorvastatin 20mg", "scheduled": "22:00"},
            {"type": "medication_taken", "when": "2026-04-08T22:05:00Z", "note": "Taken on Day 1"},
        ],
        clinical_context="Single missed statin dose has minimal immediate impact but pattern should be monitored. Patient may have been preoccupied with hypertensive symptoms.",
    ))

    # --- 16. Rising BP trend ---
    anomalies.append(make_anomaly(
        event_id=find_event_near("2026-04-09T19:00:00Z", "manual_entry", "blood_pressure")["id"],
        datetime="2026-04-09T19:00:00Z",
        category="trend",
        severity="high",
        title="Rising Blood Pressure Trend: 134→142→148→162→171 Systolic",
        description="Systolic BP has risen from 134 to 171 mmHg over 36 hours across 5 readings. Diastolic also trending up: 82→88→92→98→101. Consistent upward trajectory despite antihypertensive therapy indicates treatment failure or non-compliance (medications taken 3h45m late on Day 2).",
        related_data=[
            {"type": "trend", "parameter": "systolic_bp", "values": [
                {"value": 134, "when": "2026-04-08T18:00:00Z"},
                {"value": 142, "when": "2026-04-08T07:45:00Z"},
                {"value": 148, "when": "2026-04-09T07:30:00Z"},
                {"value": 162, "when": "2026-04-09T16:30:00Z"},
                {"value": 171, "when": "2026-04-09T19:00:00Z"},
            ]},
        ],
        clinical_context="Progressive BP elevation despite dual antihypertensive therapy. Late medication dosing and poor sleep are contributing factors. Regimen reassessment needed.",
    ))

    # --- 17. Multi-signal alert ---
    anomalies.append(make_anomaly(
        event_id=find_event_near("2026-04-09T19:00:00Z", "manual_entry", "blood_pressure")["id"],
        datetime="2026-04-09T19:00:00Z",
        category="correlation",
        severity="critical",
        title="Multi-Signal Alert: BP 171/101 + HR 91-104 + Dizziness + Chest Tightness + Declining HRV",
        description="Convergence of multiple concerning signals: (1) Hypertensive urgency BP 171/101, (2) Sustained resting tachycardia 91-104 bpm, (3) Dizziness severity 6/10, (4) Chest tightness severity 4/10, (5) HRV declined to 25 ms, (6) Post-prandial glucose 231 mg/dL. In patient with known LVH, mild cardiomegaly, and early diabetic nephropathy.",
        related_data=[
            {"type": "blood_pressure", "value": "171/101"},
            {"type": "heart_rate", "range": "91-104 bpm"},
            {"type": "symptom", "symptom": "dizziness", "severity": 6},
            {"type": "symptom", "symptom": "chest_tightness", "severity": 4},
            {"type": "hrv", "value": 25, "unit": "ms"},
            {"type": "blood_glucose", "value": 231, "unit": "mg/dL"},
        ],
        clinical_context="Multi-system decompensation in patient with hypertensive heart disease, T2DM, and asthma. Immediate clinical evaluation recommended. NSAIDs contraindicated. Consider ER for end-organ damage assessment (ECG, troponin, renal function, urinalysis for proteinuria).",
    ))

    # --- 18. Echo: LV hypertrophy ---
    e = find_event_near("2026-02-18T00:00:00", "document")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-02-18T00:00:00",
        category="document",
        severity="moderate",
        title="Echocardiogram: LV Hypertrophy and Grade I Diastolic Dysfunction",
        description="Transthoracic echocardiogram shows concentric left ventricular hypertrophy with Grade I diastolic dysfunction (impaired relaxation). LVEF preserved at 55%. Consistent with hypertensive heart disease in patient with long-standing Stage 2 HTN.",
        related_data=[
            {"type": "echo_finding", "finding": "LV hypertrophy", "LVEF": "55%"},
            {"type": "echo_finding", "finding": "Grade I diastolic dysfunction"},
        ],
        clinical_context="LVH is a major cardiovascular risk factor. Indicates end-organ damage from chronic hypertension. Optimal BP control (<130/80) is essential to prevent progression to heart failure.",
    ))

    # --- 19. Renal US: Early diabetic nephropathy ---
    e = find_event_near("2026-03-26T00:00:00", "document")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-03-26T00:00:00",
        category="document",
        severity="moderate",
        title="Renal Ultrasound: Early Diabetic Nephropathy with Elevated Resistive Index",
        description="Renal ultrasound shows bilateral increased echogenicity suggesting early diabetic nephropathy. Resistive index elevated (0.72-0.74), indicating early renovascular changes. Kidney sizes preserved.",
        related_data=[
            {"type": "imaging_finding", "finding": "Increased renal echogenicity"},
            {"type": "imaging_finding", "finding": "Elevated resistive index 0.72-0.74"},
        ],
        clinical_context="Early diabetic nephropathy in patient with T2DM and HTN. ACEi/ARB (Losartan already prescribed) is renoprotective. Strict BP and glucose control essential to prevent CKD progression.",
    ))

    # --- 20. Chest XR: Mild cardiomegaly progression ---
    e = find_event_near("2026-03-31T00:00:00", "document")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-03-31T00:00:00",
        category="document",
        severity="low",
        title="Chest X-Ray: Mild Cardiomegaly Progression (CTR 0.52→0.54)",
        description="Chest X-ray shows mild cardiomegaly with cardiothoracic ratio 0.54, increased from 0.52 on prior study. Left ventricular configuration. Mild aortic unfolding consistent with age and hypertension. Lungs are clear.",
        related_data=[
            {"type": "xray_finding", "finding": "Cardiomegaly", "CTR": 0.54, "prior_CTR": 0.52},
            {"type": "xray_finding", "finding": "Mild aortic unfolding"},
        ],
        clinical_context="Progressive cardiomegaly correlates with echo findings of LVH. Subtle worsening suggests suboptimal BP control over time.",
    ))

    # --- 21. CBC: Mild anemia ---
    e = find_event_near("2026-04-01T00:00:00", "document")
    anomalies.append(make_anomaly(
        event_id=e["id"],
        datetime="2026-04-01T00:00:00",
        category="document",
        severity="low",
        title="CBC: Mild Anemia (Hb 11.8, HCT 35.2% Below Range)",
        description="Complete blood count shows mild anemia: Hemoglobin 11.8 g/dL (normal 12.0-16.0) and Hematocrit 35.2% (normal 36.0-46.0). May be related to early diabetic nephropathy (decreased EPO production) or chronic disease.",
        related_data=[
            {"type": "lab_result", "test": "Hemoglobin", "value": 11.8, "unit": "g/dL", "normal": "12.0-16.0"},
            {"type": "lab_result", "test": "Hematocrit", "value": 35.2, "unit": "%", "normal": "36.0-46.0"},
        ],
        clinical_context="Mild anemia in diabetic patient with early nephropathy may represent anemia of CKD. Consider checking ferritin, iron studies, and renal function (eGFR, creatinine).",
    ))

    return anomalies


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------

def generate_viewer_html(timeline, snapshots, anomalies):
    """Generate self-contained interactive timeline viewer."""

    # Build anomaly lookup by event_id
    anomaly_by_event = {}
    for a in anomalies:
        eid = a.get("event_id", "")
        anomaly_by_event.setdefault(eid, []).append(a)

    # Build snapshot lookup by event_id
    snapshot_by_event = {}
    for s in snapshots:
        snapshot_by_event[s["event_id"]] = s

    # Prepare data for embedding
    viewer_data = {
        "patient": timeline["patient"],
        "events": timeline["events"],
        "wearable_data": timeline["wearable_data"],
        "snapshots": snapshot_by_event,
        "anomalies": anomaly_by_event,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Patient Timeline — {timeline['patient']['patient_id']}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e0e0e0; overflow: hidden; height: 100vh; }}
#app {{ display: flex; flex-direction: column; height: 100vh; }}

/* Header */
#header {{ background: #1a1d27; padding: 12px 20px; border-bottom: 1px solid #2a2d3a; display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
#header h1 {{ font-size: 16px; color: #fff; white-space: nowrap; }}
#header .patient-info {{ font-size: 13px; color: #8890a0; }}
#filters {{ display: flex; flex-wrap: wrap; gap: 8px; margin-left: auto; }}
#filters label {{ display: flex; align-items: center; gap: 4px; font-size: 12px; cursor: pointer; padding: 4px 8px; border-radius: 4px; background: #22253a; border: 1px solid #333650; user-select: none; }}
#filters label:hover {{ background: #2a2d45; }}
#filters input[type=checkbox] {{ accent-color: var(--cat-color); }}

/* Main area */
#main {{ display: flex; flex: 1; overflow: hidden; }}

/* Timeline panel */
#timeline-panel {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
#timeline-canvas-wrap {{ flex: 1; position: relative; cursor: grab; overflow: hidden; }}
canvas#timeline {{ width: 100%; height: 100%; display: block; }}
#zoom-info {{ position: absolute; bottom: 8px; left: 12px; font-size: 11px; color: #667; background: rgba(15,17,23,0.8); padding: 4px 8px; border-radius: 4px; pointer-events: none; }}

/* Wearable tracks */
#wearable-panel {{ height: 0; overflow: hidden; transition: height 0.3s; background: #12141c; border-top: 1px solid #2a2d3a; }}
#wearable-panel.visible {{ height: 200px; }}
canvas#wearable {{ width: 100%; height: 100%; display: block; }}

/* Sidebar */
#sidebar {{ width: 0; overflow-y: auto; overflow-x: hidden; background: #1a1d27; border-left: 1px solid #2a2d3a; transition: width 0.2s; flex-shrink: 0; }}
#sidebar.open {{ width: 420px; }}
#sidebar-content {{ padding: 16px; width: 420px; }}
#sidebar .close-btn {{ position: sticky; top: 0; float: right; background: #333; border: none; color: #aaa; width: 28px; height: 28px; border-radius: 4px; cursor: pointer; font-size: 16px; z-index: 10; }}
#sidebar .close-btn:hover {{ background: #555; color: #fff; }}
.sb-section {{ margin-bottom: 16px; }}
.sb-section h3 {{ font-size: 13px; color: #8890a0; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; cursor: pointer; }}
.sb-section h3::before {{ content: '\\25B8 '; }}
.sb-section.open h3::before {{ content: '\\25BE '; }}
.sb-section .sb-body {{ display: none; }}
.sb-section.open .sb-body {{ display: block; }}
.sb-title {{ font-size: 18px; font-weight: 600; color: #fff; margin: 8px 0; }}
.sb-time {{ font-size: 13px; color: #8890a0; margin-bottom: 4px; }}
.sb-source {{ font-size: 12px; color: #667; margin-bottom: 12px; }}
.sb-vital {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #22253a; font-size: 13px; }}
.sb-vital .label {{ color: #8890a0; }}
.sb-vital .val {{ color: #fff; font-weight: 500; }}
.sb-vital .val.high {{ color: #ff6b6b; }}
.sb-vital .val.moderate {{ color: #ffa94d; }}
.sb-symptom {{ padding: 6px 0; border-bottom: 1px solid #22253a; font-size: 13px; }}
.sb-med {{ padding: 4px 0; font-size: 13px; }}
.sb-med .status-late {{ color: #ffa94d; }}
.sb-med .status-on_time {{ color: #69db7c; }}
.sb-attention {{ padding: 6px 8px; margin: 4px 0; background: #2a1a1a; border-left: 3px solid #ff6b6b; border-radius: 0 4px 4px 0; font-size: 13px; }}
.anomaly-card {{ padding: 10px; margin: 6px 0; border-radius: 6px; font-size: 13px; }}
.anomaly-card.critical {{ background: #3a1a1a; border: 1px solid #ff4444; }}
.anomaly-card.high {{ background: #3a2a1a; border: 1px solid #ff8800; }}
.anomaly-card.moderate {{ background: #2a2a1a; border: 1px solid #ffaa00; }}
.anomaly-card.low {{ background: #1a2a1a; border: 1px solid #44aa44; }}
.anomaly-card .sev {{ font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 4px; }}
.anomaly-card.critical .sev {{ color: #ff4444; }}
.anomaly-card.high .sev {{ color: #ff8800; }}
.anomaly-card.moderate .sev {{ color: #ffaa00; }}
.anomaly-card.low .sev {{ color: #44aa44; }}
.anomaly-card .a-title {{ font-weight: 600; color: #fff; margin-bottom: 4px; }}
.anomaly-card .a-desc {{ color: #aab; line-height: 1.4; }}
.sb-finding {{ font-size: 13px; color: #aab; padding: 4px 0; }}
.sb-clinical {{ font-size: 13px; color: #aab; line-height: 1.5; padding: 8px; background: #171922; border-radius: 4px; }}
</style>
</head>
<body>
<div id="app">
<div id="header">
  <h1>Patient Timeline</h1>
  <span class="patient-info">{timeline['patient']['demographics']['name']} | {timeline['patient']['patient_id']} | DOB: {timeline['patient']['demographics']['date_of_birth']}</span>
  <div id="filters"></div>
</div>
<div id="main">
  <div id="timeline-panel">
    <div id="timeline-canvas-wrap">
      <canvas id="timeline"></canvas>
      <div id="zoom-info"></div>
    </div>
    <div id="wearable-panel">
      <canvas id="wearable"></canvas>
    </div>
  </div>
  <div id="sidebar">
    <div id="sidebar-content"></div>
  </div>
</div>
</div>
<script>
const DATA = {json.dumps(viewer_data, default=str)};

const CAT_COLORS = {{
  condition: '#9b59b6',
  medication: '#3498db',
  visit: '#2ecc71',
  document: '#e67e22',
  appointment: '#95a5a6',
  manual_entry: '#1abc9c',
}};

const CAT_LABELS = {{
  condition: 'Conditions',
  medication: 'Medications',
  visit: 'Visits',
  document: 'Documents',
  appointment: 'Appointments',
  manual_entry: 'Manual Entries',
}};

const SEVERITY_COLORS = {{ critical: '#ff4444', high: '#ff8800', moderate: '#ffaa00', low: '#44aa44' }};

// Parse event datetimes to timestamps
const events = DATA.events.map(e => ({{
  ...e,
  ts: new Date(e.datetime.endsWith('Z') || e.datetime.includes('+') ? e.datetime : e.datetime + 'Z').getTime(),
}}));

// Category visibility
const catVisible = {{}};
Object.keys(CAT_COLORS).forEach(c => catVisible[c] = true);

// Build filters
const filtersEl = document.getElementById('filters');
Object.entries(CAT_LABELS).forEach(([cat, label]) => {{
  const lab = document.createElement('label');
  lab.style.setProperty('--cat-color', CAT_COLORS[cat]);
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = true;
  cb.addEventListener('change', () => {{ catVisible[cat] = cb.checked; draw(); }});
  lab.appendChild(cb);
  lab.appendChild(document.createTextNode(' ' + label));
  filtersEl.appendChild(lab);
}});

// Canvas setup
const canvas = document.getElementById('timeline');
const ctx = canvas.getContext('2d');
let W, H;
function resize() {{
  const wrap = canvas.parentElement;
  const newW = wrap.clientWidth;
  const newH = wrap.clientHeight;
  if (newW === W && newH === H) return;
  W = newW;
  H = newH;
  canvas.width = W * devicePixelRatio;
  canvas.height = H * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}}

// View state
const minTs = events.length ? events[0].ts : Date.now();
const maxTs = events.length ? events[events.length - 1].ts : Date.now();
const span = maxTs - minTs || 86400000;
let viewStart = minTs - span * 0.05;
let viewEnd = maxTs + span * 0.05;

const DOT_R = 6;
const TIMELINE_Y_RATIO = 0.45;
let selectedEvent = null;
let hoveredEvent = null;

// Anomaly ring radius by severity
function anomalyRing(severity) {{
  if (severity === 'critical') return 5;
  if (severity === 'high') return 4;
  if (severity === 'moderate') return 3;
  return 2;
}}

function tsToX(ts) {{ return (ts - viewStart) / (viewEnd - viewStart) * W; }}
function xToTs(x) {{ return viewStart + x / W * (viewEnd - viewStart); }}

// Convert mouse event to canvas-local coordinates, accounting for CSS scaling
function canvasCoords(e) {{
  const rect = canvas.getBoundingClientRect();
  return {{ x: (e.clientX - rect.left) / rect.width * W, y: (e.clientY - rect.top) / rect.height * H }};
}}

function formatAxisLabel(ts) {{
  const d = new Date(ts);
  const rangeMs = viewEnd - viewStart;
  const hours = rangeMs / 3600000;
  if (hours < 48) return d.toISOString().substring(11, 16);
  if (hours < 720) return d.toISOString().substring(5, 10);
  if (hours < 8760) return d.toISOString().substring(0, 7);
  return d.toISOString().substring(0, 4);
}}

function draw() {{
  resize(); // keep W/H in sync when layout changes (sidebar open/close)
  ctx.clearRect(0, 0, W, H);
  const y = H * TIMELINE_Y_RATIO;

  // Axis line
  ctx.strokeStyle = '#2a2d3a';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(W, y);
  ctx.stroke();

  // Axis labels
  const rangeMs = viewEnd - viewStart;
  let tickInterval;
  if (rangeMs < 3600000 * 2) tickInterval = 300000;
  else if (rangeMs < 86400000) tickInterval = 3600000;
  else if (rangeMs < 86400000 * 30) tickInterval = 86400000;
  else if (rangeMs < 86400000 * 365) tickInterval = 86400000 * 30;
  else tickInterval = 86400000 * 365;

  const firstTick = Math.ceil(viewStart / tickInterval) * tickInterval;
  ctx.fillStyle = '#556';
  ctx.font = '11px system-ui';
  ctx.textAlign = 'center';
  for (let t = firstTick; t <= viewEnd; t += tickInterval) {{
    const x = tsToX(t);
    ctx.fillText(formatAxisLabel(t), x, y + 20);
    ctx.strokeStyle = '#1e2030';
    ctx.beginPath();
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x, y + 5);
    ctx.stroke();
  }}

  // Draw events
  const visibleEvents = events.filter(e => catVisible[e.category]);
  const positions = [];

  visibleEvents.forEach(e => {{
    const x = tsToX(e.ts);
    if (x < -20 || x > W + 20) return;

    const anoms = DATA.anomalies[e.id] || [];
    const maxSev = anoms.reduce((m, a) => {{
      const order = {{ critical: 4, high: 3, moderate: 2, low: 1 }};
      return order[a.severity] > order[m] ? a.severity : m;
    }}, 'low');

    // Anomaly ring
    if (anoms.length > 0) {{
      const ring = anomalyRing(maxSev);
      ctx.beginPath();
      ctx.arc(x, y, DOT_R + ring, 0, Math.PI * 2);
      ctx.strokeStyle = SEVERITY_COLORS[maxSev];
      ctx.lineWidth = ring > 3 ? 3 : 2;
      if (maxSev === 'critical') {{
        const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 300);
        ctx.globalAlpha = 0.4 + pulse * 0.6;
      }}
      ctx.stroke();
      ctx.globalAlpha = 1;
    }}

    // Dot
    ctx.beginPath();
    ctx.arc(x, y, DOT_R, 0, Math.PI * 2);
    ctx.fillStyle = CAT_COLORS[e.category] || '#888';
    if (e.planned) {{
      ctx.strokeStyle = CAT_COLORS[e.category] || '#888';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#0f1117';
      ctx.fill();
    }} else {{
      ctx.fill();
    }}

    // Selection ring
    if (selectedEvent && selectedEvent.id === e.id) {{
      ctx.beginPath();
      ctx.arc(x, y, DOT_R + 3, 0, Math.PI * 2);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }}

    positions.push({{ x, y, event: e }});
  }});

  // Labels for spaced-out events
  ctx.font = '11px system-ui';
  ctx.textAlign = 'center';
  let lastLabelX = -999;
  positions.forEach(p => {{
    if (p.x - lastLabelX < 80) return;
    const label = p.event.title.length > 25 ? p.event.title.substring(0, 23) + '...' : p.event.title;
    ctx.fillStyle = '#8890a0';
    ctx.save();
    ctx.translate(p.x, p.y - DOT_R - 8);
    ctx.rotate(-0.4);
    ctx.fillText(label, 0, 0);
    ctx.restore();
    lastLabelX = p.x;
  }});

  // Tooltip for hovered
  if (hoveredEvent) {{
    const hx = tsToX(hoveredEvent.ts);
    const tw = ctx.measureText(hoveredEvent.title).width + 16;
    const tx = Math.min(Math.max(hx - tw/2, 4), W - tw - 4);
    const ty = y + 28;
    ctx.fillStyle = 'rgba(30,32,48,0.95)';
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(tx, ty, tw, 28, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'left';
    ctx.fillText(hoveredEvent.title, tx + 8, ty + 18);
  }}

  // Zoom info
  const hours = rangeMs / 3600000;
  let rangeLabel;
  if (hours < 1) rangeLabel = Math.round(hours * 60) + ' min';
  else if (hours < 48) rangeLabel = Math.round(hours) + ' hours';
  else if (hours < 720) rangeLabel = Math.round(hours / 24) + ' days';
  else if (hours < 8760) rangeLabel = Math.round(hours / 720) + ' months';
  else rangeLabel = (hours / 8760).toFixed(1) + ' years';
  document.getElementById('zoom-info').textContent = 'View: ' + rangeLabel + '  |  Scroll to zoom, drag to pan';

  // Show wearable panel if zoomed into Apr 8-9 window
  const apr8 = new Date('2026-04-07T21:00:00Z').getTime();
  const apr10 = new Date('2026-04-10T06:00:00Z').getTime();
  const wp = document.getElementById('wearable-panel');
  if (viewStart >= apr8 - span * 0.5 && viewEnd <= apr10 + span * 0.5 && rangeMs < 86400000 * 7) {{
    wp.classList.add('visible');
    drawWearable();
  }} else {{
    wp.classList.remove('visible');
  }}

  window._eventPositions = positions;
}}

// Wearable canvas
function drawWearable() {{
  const wCanvas = document.getElementById('wearable');
  const wp = document.getElementById('wearable-panel');
  const wW = wp.clientWidth;
  const wH = wp.clientHeight;
  wCanvas.width = wW * devicePixelRatio;
  wCanvas.height = wH * devicePixelRatio;
  const wCtx = wCanvas.getContext('2d');
  wCtx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  wCtx.clearRect(0, 0, wW, wH);

  const tracks = [
    {{ key: 'heart_rate', label: 'HR', unit: 'bpm', color: '#ff6b6b', min: 50, max: 110 }},
    {{ key: 'spo2', label: 'SpO2', unit: '%', color: '#4ecdc4', min: 0.93, max: 1.0 }},
    {{ key: 'hrv', label: 'HRV', unit: 'ms', color: '#ffd93d', min: 0, max: 80 }},
  ];
  const trackH = wH / tracks.length;

  tracks.forEach((track, ti) => {{
    const ty = ti * trackH;
    const data = DATA.wearable_data[track.key];
    if (!data || data.length === 0) return;

    // Background
    wCtx.fillStyle = ti % 2 === 0 ? '#12141c' : '#14161f';
    wCtx.fillRect(0, ty, wW, trackH);

    // Label
    wCtx.fillStyle = track.color;
    wCtx.font = 'bold 11px system-ui';
    wCtx.textAlign = 'left';
    wCtx.fillText(track.label + ' (' + track.unit + ')', 8, ty + 14);

    // Plot
    const pad = 20;
    const plotH = trackH - pad * 2;
    wCtx.strokeStyle = track.color;
    wCtx.lineWidth = 1.5;
    wCtx.globalAlpha = 0.8;
    wCtx.beginPath();
    let first = true;
    data.forEach(d => {{
      const dt = d.datetime || d.start;
      const ts = new Date(dt.endsWith('Z') || dt.includes('+') ? dt : dt + 'Z').getTime();
      const x = (ts - viewStart) / (viewEnd - viewStart) * wW;
      if (x < -5 || x > wW + 5) return;
      const yVal = ty + pad + plotH * (1 - (d.value - track.min) / (track.max - track.min));
      if (first) {{ wCtx.moveTo(x, yVal); first = false; }}
      else wCtx.lineTo(x, yVal);
    }});
    wCtx.stroke();
    wCtx.globalAlpha = 1;

    // Range labels
    wCtx.fillStyle = '#556';
    wCtx.font = '10px system-ui';
    wCtx.textAlign = 'right';
    wCtx.fillText(track.max, wW - 4, ty + pad + 4);
    wCtx.fillText(track.min, wW - 4, ty + trackH - pad + 4);
  }});
}}

// Interaction
let isDragging = false;
let didDrag = false;
let dragStartX = 0;
let dragStartY = 0;
let dragStartView = [0, 0];
let mousedownTarget = null;
const DRAG_THRESHOLD = 4; // px of movement before it counts as a drag

const wrap = document.getElementById('timeline-canvas-wrap');

wrap.addEventListener('wheel', e => {{
  e.preventDefault();
  const cc = canvasCoords(e);
  const pivot = xToTs(cc.x);
  const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
  viewStart = pivot - (pivot - viewStart) * factor;
  viewEnd = pivot + (viewEnd - pivot) * factor;
  draw();
}}, {{ passive: false }});

wrap.addEventListener('mousedown', e => {{
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  dragStartView = [viewStart, viewEnd];
  didDrag = false;
  // Check if mousedown is on a dot
  const cc = canvasCoords(e);
  mousedownTarget = null;
  if (window._eventPositions) {{
    for (const p of window._eventPositions) {{
      if (Math.hypot(p.x - cc.x, p.y - cc.y) < DOT_R + 8) {{
        mousedownTarget = p.event;
        break;
      }}
    }}
  }}
  // Only allow dragging if mousedown was NOT on a dot
  isDragging = !mousedownTarget;
  if (isDragging) canvas.style.cursor = 'grabbing';
}});

window.addEventListener('mousemove', e => {{
  if (isDragging) {{
    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;
    if (!didDrag && Math.hypot(dx, dy) >= DRAG_THRESHOLD) didDrag = true;
    if (didDrag) {{
      const tsPerPx = (dragStartView[1] - dragStartView[0]) / W;
      viewStart = dragStartView[0] - dx * tsPerPx;
      viewEnd = dragStartView[1] - dx * tsPerPx;
      draw();
    }}
  }} else if (!mousedownTarget) {{
    // Hover detection (only when not mid-click on a dot)
    const cc = canvasCoords(e);
    hoveredEvent = null;
    if (window._eventPositions) {{
      for (const p of window._eventPositions) {{
        if (Math.hypot(p.x - cc.x, p.y - cc.y) < DOT_R + 4) {{
          hoveredEvent = p.event;
          break;
        }}
      }}
    }}
    canvas.style.cursor = hoveredEvent ? 'pointer' : 'grab';
    draw();
  }}
}});

window.addEventListener('mouseup', e => {{
  if (mousedownTarget) {{
    // Clicked on a dot — always select it
    selectEvent(mousedownTarget);
  }} else if (isDragging && !didDrag) {{
    // Clicked on empty space without dragging — close sidebar
    closeSidebar();
  }}
  isDragging = false;
  mousedownTarget = null;
  canvas.style.cursor = hoveredEvent ? 'pointer' : 'grab';
}});

function closeSidebar() {{
  selectedEvent = null;
  document.getElementById('sidebar').classList.remove('open');
  draw();
}}

function selectEvent(evt) {{
  selectedEvent = evt;
  const sidebar = document.getElementById('sidebar');
  sidebar.classList.add('open');
  const content = document.getElementById('sidebar-content');
  const snapshot = DATA.snapshots[evt.id];
  const anoms = DATA.anomalies[evt.id] || [];

  let html = '<button class="close-btn" onclick="closeSidebar()">&times;</button>';
  html += '<div class="sb-title">' + esc(evt.title) + '</div>';
  html += '<div class="sb-time">' + esc(evt.datetime) + '</div>';
  html += '<div class="sb-source">Source: ' + esc(evt.source) + (evt.planned ? ' (Planned)' : '') + '</div>';

  // Event data
  html += '<div class="sb-section open"><h3 onclick="this.parentElement.classList.toggle(\\'open\\')">Event Details</h3><div class="sb-body">';
  html += '<pre style="font-size:12px;color:#aab;white-space:pre-wrap;word-break:break-word;max-height:200px;overflow:auto;background:#171922;padding:8px;border-radius:4px;">' + esc(JSON.stringify(evt.data, null, 2)) + '</pre>';
  html += '</div></div>';

  // Snapshot
  if (snapshot) {{
    // Vitals
    html += '<div class="sb-section open"><h3 onclick="this.parentElement.classList.toggle(\\'open\\')">Health Snapshot</h3><div class="sb-body">';
    const v = snapshot.most_recent_vitals || {{}};
    if (v.blood_pressure) {{
      const cls = v.blood_pressure.systolic >= 160 ? 'high' : v.blood_pressure.systolic >= 140 ? 'moderate' : '';
      html += '<div class="sb-vital"><span class="label">Blood Pressure</span><span class="val ' + cls + '">' + esc(v.blood_pressure.value) + ' mmHg</span></div>';
    }}
    if (v.heart_rate) {{
      const cls = v.heart_rate.value >= 100 ? 'high' : v.heart_rate.value >= 90 ? 'moderate' : '';
      html += '<div class="sb-vital"><span class="label">Heart Rate</span><span class="val ' + cls + '">' + v.heart_rate.value + ' bpm</span></div>';
    }}
    if (v.spo2) html += '<div class="sb-vital"><span class="label">SpO2</span><span class="val">' + (v.spo2.value * 100).toFixed(0) + '%</span></div>';
    if (v.blood_glucose) {{
      const cls = v.blood_glucose.value >= 200 ? 'high' : v.blood_glucose.value >= 180 ? 'moderate' : '';
      html += '<div class="sb-vital"><span class="label">Blood Glucose</span><span class="val ' + cls + '">' + v.blood_glucose.value + ' mg/dL</span></div>';
    }}

    // Adherence
    if (snapshot.medication_adherence_48h) {{
      html += '<div style="margin-top:8px;font-size:13px;color:#8890a0;">Medication Adherence: <b style="color:#fff">' + esc(snapshot.medication_adherence_48h.summary) + '</b></div>';
      (snapshot.medication_adherence_48h.details || []).forEach(d => {{
        const statusCls = 'status-' + (d.status || '');
        html += '<div class="sb-med"><span>' + esc(d.medication) + '</span> <span class="' + statusCls + '">' + esc(d.status || '') + (d.delay_minutes ? ' (' + d.delay_minutes + 'min)' : '') + '</span></div>';
      }});
    }}

    // Symptoms
    if (snapshot.reported_symptoms && snapshot.reported_symptoms.length > 0) {{
      html += '<div style="margin-top:8px;font-size:13px;color:#8890a0;">Active Symptoms:</div>';
      snapshot.reported_symptoms.forEach(s => {{
        html += '<div class="sb-symptom">' + esc(s.symptom) + ' — severity ' + s.severity + '/10' + (s.notes ? ' <span style="color:#667">(' + esc(s.notes) + ')</span>' : '') + '</div>';
      }});
    }}

    // Clinical summary
    html += '<div class="sb-clinical" style="margin-top:8px;">' + esc(snapshot.clinical_findings_summary) + '</div>';

    // Attention
    if (snapshot.care_team_attention && snapshot.care_team_attention.length > 0) {{
      html += '<div style="margin-top:8px;">';
      snapshot.care_team_attention.forEach(a => {{
        html += '<div class="sb-attention">' + esc(a) + '</div>';
      }});
      html += '</div>';
    }}
    html += '</div></div>';
  }}

  // Anomalies
  if (anoms.length > 0) {{
    html += '<div class="sb-section open"><h3 onclick="this.parentElement.classList.toggle(\\'open\\')">Anomalies (' + anoms.length + ')</h3><div class="sb-body">';
    anoms.forEach(a => {{
      html += '<div class="anomaly-card ' + esc(a.severity) + '">';
      html += '<div class="sev">' + esc(a.severity) + '</div>';
      html += '<div class="a-title">' + esc(a.title) + '</div>';
      html += '<div class="a-desc">' + esc(a.description) + '</div>';
      html += '</div>';
    }});
    html += '</div></div>';
  }}

  content.innerHTML = html;
  draw();
}}

function esc(s) {{ if (s == null) return ''; return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}

window.addEventListener('resize', () => {{ resize(); draw(); }});
resize();
draw();

// Animate critical pulses
function animate() {{ if (DATA.anomalies && Object.values(DATA.anomalies).some(arr => arr.some(a => a.severity === 'critical'))) draw(); requestAnimationFrame(animate); }}
requestAnimationFrame(animate);
</script>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(ANOMALY_DIR, exist_ok=True)
    os.makedirs(VIEWER_DIR, exist_ok=True)

    print("Building timeline...")
    timeline, profile = build_timeline()

    # Write timeline.json
    with open(TIMELINE_PATH, "w") as f:
        json.dump(timeline, f, indent=2, default=str)
    print(f"  Wrote {TIMELINE_PATH} ({len(timeline['events'])} events)")

    # Generate snapshots
    print("Generating health snapshots...")
    snapshots = []
    for idx, evt in enumerate(timeline["events"]):
        snap = generate_snapshot(evt, idx, timeline["events"], timeline["wearable_data"], profile)
        snapshots.append(snap)

        # Write snapshot file
        dt_prefix = evt["datetime"].replace(":", "-").replace("T", "_")[:19]
        fname = f"{dt_prefix}_{evt['id']}.json"
        fpath = os.path.join(SNAPSHOT_DIR, fname)
        with open(fpath, "w") as f:
            json.dump(snap, f, indent=2, default=str)
    print(f"  Wrote {len(snapshots)} snapshot files to {SNAPSHOT_DIR}/")

    # Generate anomalies
    print("Generating anomalies...")
    anomalies = generate_anomalies(timeline["events"], timeline["wearable_data"], profile)

    for anom in anomalies:
        dt_prefix = anom["datetime"].replace(":", "-").replace("T", "_")[:19]
        fname = f"{dt_prefix}_{anom['anomaly_id']}.json"
        fpath = os.path.join(ANOMALY_DIR, fname)
        with open(fpath, "w") as f:
            json.dump(anom, f, indent=2, default=str)
    print(f"  Wrote {len(anomalies)} anomaly files to {ANOMALY_DIR}/")

    # Generate viewer
    print("Generating timeline viewer...")
    html = generate_viewer_html(timeline, snapshots, anomalies)
    viewer_path = os.path.join(VIEWER_DIR, "index.html")
    with open(viewer_path, "w") as f:
        f.write(html)
    print(f"  Wrote {viewer_path}")

    # Summary
    print("\n=== Build Complete ===")
    print(f"Events: {len(timeline['events'])}")
    print(f"Wearable tracks: HR={len(timeline['wearable_data']['heart_rate'])}, "
          f"SpO2={len(timeline['wearable_data']['spo2'])}, "
          f"Steps={len(timeline['wearable_data']['steps'])}, "
          f"Sleep={len(timeline['wearable_data']['sleep'])}, "
          f"HRV={len(timeline['wearable_data']['hrv'])}")
    print(f"Snapshots: {len(snapshots)}")
    print(f"Anomalies: {len(anomalies)}")
    print(f"\nOpen {viewer_path} in a browser to view the timeline.")


if __name__ == "__main__":
    main()
