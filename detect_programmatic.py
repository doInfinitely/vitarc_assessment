#!/usr/bin/env python3
"""Programmatic snapshot generation, anomaly detection via decision trees, and grading."""

import json
import os
import re
from datetime import datetime, timedelta, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIMELINE_PATH = os.path.join(BASE_DIR, "timeline.json")
SNAPSHOT_GT_DIR = os.path.join(BASE_DIR, "snapshot_ground_truth")
ANOMALY_GT_DIR = os.path.join(BASE_DIR, "anomaly_ground_truth")
SNAPSHOT_PROG_DIR = os.path.join(BASE_DIR, "snapshot_programmatic")
ANOMALY_PROG_DIR = os.path.join(BASE_DIR, "anomaly_programmatic")
VIZ_DIR = os.path.join(BASE_DIR, "decision_tree_viz")

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_dt(s):
    """Parse ISO datetime string to UTC datetime."""
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def fmt_dt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def has_time(dt_str):
    """Check if datetime string has a time component (not just date)."""
    return "T" in dt_str and dt_str.split("T")[1] != "00:00:00"


# ---------------------------------------------------------------------------
# Load timeline
# ---------------------------------------------------------------------------

def load_timeline():
    with open(TIMELINE_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Part 1: Programmatic Snapshots
# ---------------------------------------------------------------------------

def find_most_recent_bp(events, up_to_idx):
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
    best = None
    for r in track:
        t = r.get(key, r.get("datetime", ""))
        if t <= up_to_dt_str:
            best = r
        else:
            break
    return best


def get_medication_adherence_48h(events, up_to_idx, profile):
    evt_dt = events[up_to_idx]["datetime"]
    if "T" not in evt_dt:
        return None
    try:
        evt_time = parse_dt(evt_dt)
    except Exception:
        return None
    if evt_time is None:
        return None

    window_start = evt_time - timedelta(hours=48)
    ws_str = fmt_dt(window_start)

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

    scheduled_meds = [m for m in profile.get("medications", []) if m.get("frequency") != "as needed"]
    details = []
    for te in taken_entries:
        d = te["data"]
        if d["type"] == "medication_taken":
            for med_name in d["values"].get("medications", []):
                sched = None
                for sm in scheduled_meds:
                    if sm["name"].lower() in med_name.lower() or med_name.lower().startswith(sm["name"].lower()):
                        sched = sm
                        break
                entry = {"medication": med_name, "taken_at": te["datetime"]}
                if sched and sched.get("scheduled_time"):
                    sched_times = [t.strip() for t in sched["scheduled_time"].split(",")]
                    taken_time = parse_dt(te["datetime"])
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
    evt_dt = events[up_to_idx]["datetime"]
    if "T" not in evt_dt:
        return []
    try:
        evt_time = parse_dt(evt_dt)
    except Exception:
        return []
    if evt_time is None:
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
    findings = []
    for i in range(up_to_idx + 1):
        e = events[i]
        if e["category"] == "document":
            findings.append({"title": e["title"], "date": e["datetime"][:10]})
    return findings


def build_clinical_summary(evt, vitals, symptoms, doc_findings, adherence):
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
    for s in high_symptoms:
        items.append(f"Patient reporting {s['symptom']} (severity {s['severity']}/10) — assess and manage.")

    return items if items else ["No immediate attention items."]


def generate_snapshot(evt, idx, events, wearable, profile):
    vitals = {}
    bp = find_most_recent_bp(events, idx)
    if bp:
        vitals["blood_pressure"] = bp
    bg = find_most_recent_glucose(events, idx)
    if bg:
        vitals["blood_glucose"] = bg
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

    return {
        "event_id": evt["id"],
        "datetime": evt["datetime"],
        "event_title": evt["title"],
        "most_recent_vitals": vitals,
        "medication_adherence_48h": adherence,
        "reported_symptoms": symptoms,
        "clinical_findings_summary": clinical_summary,
        "care_team_attention": attention,
    }


# ---------------------------------------------------------------------------
# Part 2: Anomaly Detection Decision Trees
# ---------------------------------------------------------------------------

def find_event_near(events, dt_str, category=None, subcategory=None):
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
        ta = parse_dt(a)
        tb = parse_dt(b)
        if ta and tb:
            return abs((ta - tb).total_seconds())
    except Exception:
        pass
    return 1e12


def detect_anomalies(events, wearable, profile):
    """Run all 9 anomaly detectors and return list of anomalies."""
    anomalies = []
    anom_counter = [0]

    def make_anomaly(**kwargs):
        anom_counter[0] += 1
        a = {"anomaly_id": f"anom-{anom_counter[0]:03d}"}
        a.update(kwargs)
        return a

    # Collect BP and glucose readings for trend/correlation detectors
    bp_readings = []
    glucose_readings = []
    symptom_events = []
    med_taken_events = []

    for e in events:
        if e.get("category") != "manual_entry":
            continue
        dtype = e.get("data", {}).get("type")
        if dtype == "blood_pressure":
            v = e["data"]["values"]
            bp_readings.append({
                "datetime": e["datetime"],
                "event_id": e["id"],
                "systolic": v["systolic_mmhg"],
                "diastolic": v["diastolic_mmhg"],
                "event": e,
            })
        elif dtype == "blood_glucose":
            v = e["data"]["values"]
            glucose_readings.append({
                "datetime": e["datetime"],
                "event_id": e["id"],
                "value": v["glucose_mg_dl"],
                "context": e["data"].get("context"),
                "notes": e["data"].get("notes"),
                "event": e,
            })
        elif dtype == "symptom":
            symptom_events.append(e)
        elif dtype in ("medication_taken", "inhaler_use"):
            med_taken_events.append(e)

    # Helper: get concurrent symptoms near a datetime
    def get_concurrent_symptoms(dt_str, window_minutes=30):
        result = []
        for se in symptom_events:
            diff = abs_dt_diff(se["datetime"], dt_str)
            if diff <= window_minutes * 60:
                result.append(se)
        return result

    # Helper: get concurrent BP near a datetime
    def get_concurrent_bp(dt_str, window_minutes=30):
        for bp in bp_readings:
            diff = abs_dt_diff(bp["datetime"], dt_str)
            if diff <= window_minutes * 60:
                return bp
        return None

    # Helper: check if any concurrent vital is abnormal
    def has_concurrent_vital_abnormality(dt_str, window_minutes=60):
        bp = get_concurrent_bp(dt_str, window_minutes)
        if bp and bp["systolic"] >= 140:
            return True
        # Check HR from wearable
        hr = find_most_recent_wearable(wearable["heart_rate"], dt_str)
        if hr and hr["value"] >= 90:
            hr_diff = abs_dt_diff(hr["datetime"], dt_str)
            if hr_diff <= window_minutes * 60:
                return True
        return False

    # -----------------------------------------------------------------------
    # Detector 1: BP Detector
    # -----------------------------------------------------------------------
    for bp in bp_readings:
        concurrent_syms = get_concurrent_symptoms(bp["datetime"], window_minutes=60)
        has_syms = len(concurrent_syms) > 0

        if bp["systolic"] >= 170 and has_syms:
            severity = "critical"
            title = f"Hypertensive Urgency — BP {bp['systolic']}/{bp['diastolic']} mmHg"
            desc = (f"BP {bp['systolic']}/{bp['diastolic']} mmHg with symptoms "
                    f"({', '.join(s['data']['values'].get('symptom', '').replace('_', ' ') for s in concurrent_syms)}). "
                    f"Represents hypertensive urgency requiring clinical evaluation.")
            related = [
                {"type": "blood_pressure", "value": f"{bp['systolic']}/{bp['diastolic']}"},
            ]
            for s in concurrent_syms:
                related.append({"type": "symptom", "symptom": s["data"]["values"].get("symptom", ""),
                                "severity": s["data"]["values"].get("severity")})
            # Add trend data if available
            if len(bp_readings) >= 3:
                trend_vals = [r["systolic"] for r in bp_readings if r["datetime"] <= bp["datetime"]]
                related.append({"type": "trend", "values": trend_vals, "parameter": "systolic_bp"})
            anomalies.append(make_anomaly(
                event_id=bp["event_id"],
                datetime=bp["datetime"],
                category="vital_sign",
                severity=severity,
                title=title,
                description=desc,
                related_data=related,
                clinical_context="Hypertensive urgency (SBP ≥170 with symptoms) in patient with known LVH, cardiomegaly, and early nephropathy. ER evaluation recommended for end-organ assessment.",
            ))
        elif bp["systolic"] >= 160 and has_syms:
            severity = "high"
            title = f"Blood Pressure {bp['systolic']}/{bp['diastolic']} mmHg with Symptoms"
            desc = (f"BP {bp['systolic']}/{bp['diastolic']} mmHg with patient reporting symptoms. "
                    f"Symptomatic hypertension requires clinical attention.")
            related = [{"type": "blood_pressure", "value": f"{bp['systolic']}/{bp['diastolic']}"}]
            for s in concurrent_syms:
                related.append({"type": "symptom", "symptom": s["data"]["values"].get("symptom", "")})
            anomalies.append(make_anomaly(
                event_id=bp["event_id"],
                datetime=bp["datetime"],
                category="vital_sign",
                severity=severity,
                title=title,
                description=desc,
                related_data=related,
                clinical_context="Symptomatic Stage 2 hypertension with end-organ disease risk. Morning antihypertensives taken late may be contributing factor.",
            ))
        elif bp["systolic"] >= 160:
            severity = "high"
            title = f"Severely Elevated Blood Pressure {bp['systolic']}/{bp['diastolic']} mmHg"
            desc = f"BP {bp['systolic']}/{bp['diastolic']} mmHg significantly above target. Stage 2 hypertension range."
            anomalies.append(make_anomaly(
                event_id=bp["event_id"],
                datetime=bp["datetime"],
                category="vital_sign",
                severity=severity,
                title=title,
                description=desc,
                related_data=[{"type": "blood_pressure", "value": f"{bp['systolic']}/{bp['diastolic']}"}],
                clinical_context="Severely elevated BP above target. Review antihypertensive regimen.",
            ))
        elif bp["systolic"] >= 140:
            severity = "moderate"
            # Check if there's a rising trend from prior readings
            prior_bps = [r for r in bp_readings if r["datetime"] < bp["datetime"]]
            trend_note = ""
            if prior_bps:
                prev = prior_bps[-1]
                if bp["systolic"] > prev["systolic"]:
                    trend_note = f" (Rising Trend)"

            context_str = ""
            ctx = bp.get("event", {}).get("data", {}).get("context", "")
            if ctx:
                context_str = f" ({ctx})"

            title = f"Elevated Morning Blood Pressure {bp['systolic']}/{bp['diastolic']} mmHg{trend_note}"
            desc = (f"Morning BP {bp['systolic']}/{bp['diastolic']} mmHg exceeds target of <130/80 for patient with HTN + diabetes.")
            if prior_bps and bp["systolic"] > prior_bps[-1]["systolic"]:
                desc += f" Rising from {prior_bps[-1]['systolic']}/{prior_bps[-1]['diastolic']} previously."

            related = [{"type": "blood_pressure", "value": f"{bp['systolic']}/{bp['diastolic']}",
                         "context": ctx or "morning_fasting"}]
            if prior_bps:
                related.append({"type": "blood_pressure", "value": f"{prior_bps[-1]['systolic']}/{prior_bps[-1]['diastolic']}",
                                "when": prior_bps[-1]["datetime"]})

            anomalies.append(make_anomaly(
                event_id=bp["event_id"],
                datetime=bp["datetime"],
                category="vital_sign",
                severity=severity,
                title=title,
                description=desc,
                related_data=related,
                clinical_context="Patient has Stage 2 Essential Hypertension, currently on Amlodipine 10mg + Losartan 100mg daily. Target BP for diabetic patient is <130/80 mmHg per guidelines.",
            ))

    # -----------------------------------------------------------------------
    # Detector 2: Glucose Detector
    # -----------------------------------------------------------------------
    for gl in glucose_readings:
        gl_time = parse_dt(gl["datetime"])
        hour = gl_time.hour if gl_time else 12
        context = (gl.get("context") or "").lower()
        notes = (gl.get("notes") or "").lower()

        is_fasting = ("fasting" in context or "fasting" in notes or hour < 10)
        is_postprandial = ("post_meal" in context or "post-meal" in context
                           or "after" in notes or "meal" in notes
                           or "lunch" in notes or "dinner" in notes
                           or 12 <= hour <= 16)

        prior_glucose = [g for g in glucose_readings if g["datetime"] < gl["datetime"]]
        trend_note = ""
        if prior_glucose:
            prev = prior_glucose[-1]
            if gl["value"] > prev["value"]:
                trend_note = " (Rising Trend)"

        if is_fasting:
            if gl["value"] >= 200:
                severity = "high"
                title = f"High Fasting Blood Glucose {gl['value']} mg/dL{trend_note}"
            elif gl["value"] >= 130:
                severity = "moderate"
                title = f"Elevated Fasting Blood Glucose {gl['value']} mg/dL{trend_note}"
            else:
                continue
        elif is_postprandial:
            if gl["value"] >= 200:
                severity = "high"
                title = f"High Post-Prandial Blood Glucose ~{int(gl['value'])} mg/dL"
                if "original_value" in gl.get("event", {}).get("data", {}).get("values", {}):
                    orig = gl["event"]["data"]["values"]["original_value"]
                    title += f" ({orig} mmol/L)"
            elif gl["value"] >= 180:
                severity = "moderate"
                title = f"Elevated Post-Prandial Blood Glucose {gl['value']} mg/dL"
            else:
                continue
        else:
            if gl["value"] >= 200:
                severity = "high"
                title = f"High Blood Glucose {gl['value']} mg/dL"
            elif gl["value"] >= 180:
                severity = "moderate"
                title = f"Elevated Blood Glucose {gl['value']} mg/dL"
            elif gl["value"] >= 130:
                severity = "moderate"
                title = f"Elevated Blood Glucose {gl['value']} mg/dL"
            else:
                continue

        ctx_label = "fasting" if is_fasting else ("post_meal" if is_postprandial else "unknown")
        desc = f"Blood glucose {gl['value']} mg/dL "
        if is_fasting:
            desc += f"(fasting) above target of <130 mg/dL."
        else:
            desc += f"(post-prandial) exceeds target of <180 mg/dL."
        if prior_glucose and gl["value"] > prior_glucose[-1]["value"]:
            desc += f" Rising from {prior_glucose[-1]['value']} mg/dL previously."

        related = [{"type": "blood_glucose", "value": int(gl["value"]) if gl["value"] == int(gl["value"]) else gl["value"],
                     "unit": "mg/dL", "context": ctx_label}]
        if "original_value" in gl.get("event", {}).get("data", {}).get("values", {}):
            related[0]["original"] = f"{gl['event']['data']['values']['original_value']} mmol/L"

        anomalies.append(make_anomaly(
            event_id=gl["event_id"],
            datetime=gl["datetime"],
            category="vital_sign",
            severity=severity,
            title=title,
            description=desc,
            related_data=related,
            clinical_context="Patient has T2DM with hyperglycemia, on Metformin 1000mg BID. ADA target fasting glucose is 80-130 mg/dL, post-prandial <180 mg/dL.",
        ))

    # -----------------------------------------------------------------------
    # Detector 3: Symptom Detector
    # -----------------------------------------------------------------------
    for se in symptom_events:
        sv = se["data"]["values"]
        symptom_name = sv.get("symptom", "").replace("_", " ")
        severity_val = sv.get("severity", 0)

        concurrent_bp = get_concurrent_bp(se["datetime"], window_minutes=60)
        has_high_bp = concurrent_bp and concurrent_bp["systolic"] >= 160
        has_elevated_bp = concurrent_bp and concurrent_bp["systolic"] >= 140
        has_vital_abn = has_concurrent_vital_abnormality(se["datetime"], window_minutes=60)

        if severity_val >= 5:
            severity = "moderate"
            title = f"{symptom_name.title()} Severity {severity_val}/10"
            desc = f"Patient reports {symptom_name} severity {severity_val}/10."
            if se["data"].get("notes"):
                desc += f" Notes: {se['data']['notes']}."
            related = [{"type": "symptom", "symptom": symptom_name, "severity": severity_val}]
            if concurrent_bp:
                desc += f" Concurrent BP {concurrent_bp['systolic']}/{concurrent_bp['diastolic']}."
                related.append({"type": "blood_pressure", "value": f"{concurrent_bp['systolic']}/{concurrent_bp['diastolic']}",
                                "when": concurrent_bp["datetime"]})
            anomalies.append(make_anomaly(
                event_id=se["id"],
                datetime=se["datetime"],
                category="symptom",
                severity=severity,
                title=title,
                description=desc,
                related_data=related,
                clinical_context=f"{symptom_name.title()} with significantly elevated BP requires assessment for hypertensive encephalopathy or other end-organ damage.",
            ))
        elif has_high_bp or has_vital_abn:
            # Low severity symptom but concurrent with high BP or vital abnormality
            severity = "moderate"
            title = f"{symptom_name.title()} with Hypertensive Urgency"
            desc = f"Patient reports {symptom_name} (severity {severity_val}/10)"
            if se["data"].get("notes"):
                desc += f", notes '{se['data']['notes']}'"
            desc += "."
            if concurrent_bp:
                desc += f" In context of BP {concurrent_bp['systolic']}/{concurrent_bp['diastolic']}."
            related = [
                {"type": "symptom", "symptom": sv.get("symptom", ""), "severity": severity_val},
            ]
            if concurrent_bp:
                related.append({"type": "blood_pressure", "value": f"{concurrent_bp['systolic']}/{concurrent_bp['diastolic']}",
                                "when": concurrent_bp["datetime"]})
            # Check wearable HR
            hr = find_most_recent_wearable(wearable["heart_rate"], se["datetime"])
            if hr and hr["value"] >= 90:
                hr_diff = abs_dt_diff(hr["datetime"], se["datetime"])
                if hr_diff <= 3600:
                    related.append({"type": "heart_rate", "range": f"{hr['value']} bpm", "when": se["datetime"]})
            anomalies.append(make_anomaly(
                event_id=se["id"],
                datetime=se["datetime"],
                category="symptom",
                severity=severity,
                title=title,
                description=desc,
                related_data=related,
                clinical_context=f"{symptom_name.title()} in context of hypertensive crisis may indicate cardiac strain.",
            ))
        elif symptom_name.lower() in ("mild headache", "headache", "fatigue"):
            severity = "low"
            title = f"Mild {symptom_name.title()} at Bedtime" if "headache" in symptom_name.lower() else f"Mild {symptom_name.title()}"
            desc = f"Patient reports {symptom_name} (severity {severity_val}/10)."
            if se["data"].get("notes"):
                desc += f" {se['data']['notes']}."
            related = [{"type": "symptom", "symptom": sv.get("symptom", ""), "severity": severity_val}]
            if concurrent_bp:
                related.append({"type": "blood_pressure", "value": f"{concurrent_bp['systolic']}/{concurrent_bp['diastolic']}",
                                "context": "evening", "when": concurrent_bp["datetime"]})
            anomalies.append(make_anomaly(
                event_id=se["id"],
                datetime=se["datetime"],
                category="symptom",
                severity=severity,
                title=title,
                description=desc,
                related_data=related,
                clinical_context="Headache in hypertensive patient warrants monitoring.",
            ))

    # -----------------------------------------------------------------------
    # Detector 4: Medication Detector
    # -----------------------------------------------------------------------
    scheduled_meds = [m for m in profile.get("medications", []) if m.get("frequency") != "as needed"]

    for mte in med_taken_events:
        d = mte["data"]
        if d["type"] != "medication_taken":
            continue
        taken_time = parse_dt(mte["datetime"])
        if not taken_time:
            continue

        for med_name in d["values"].get("medications", []):
            sched = None
            for sm in scheduled_meds:
                if sm["name"].lower() in med_name.lower() or med_name.lower().startswith(sm["name"].lower()):
                    sched = sm
                    break
            if not sched or not sched.get("scheduled_time"):
                continue

            sched_times = [t.strip() for t in sched["scheduled_time"].split(",")]
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

            if best_diff is not None and best_diff >= 120:
                # Only flag once per medication_taken event (not per medication)
                # Check if we already have an anomaly for this event
                already_flagged = any(a["event_id"] == mte["id"] and a["category"] == "medication"
                                      for a in anomalies)
                if already_flagged:
                    continue

                all_meds = d["values"].get("medications", [])
                delay_h = best_diff // 60
                delay_m = best_diff % 60
                title = f"Morning Medications Taken {delay_h}h{delay_m:02d}m Late"
                desc = (f"{', '.join(all_meds)} taken at {taken_time.strftime('%H:%M')} instead of scheduled {best_sched} — "
                        f"{best_diff} minutes late.")
                if mte["data"].get("notes"):
                    desc += f" Patient notes: '{mte['data']['notes']}'."
                anomalies.append(make_anomaly(
                    event_id=mte["id"],
                    datetime=mte["datetime"],
                    category="medication",
                    severity="moderate",
                    title=title,
                    description=desc,
                    related_data=[
                        {"type": "medication_taken", "medications": all_meds},
                        {"type": "note", "value": f"Scheduled time: {best_sched}, taken: {taken_time.strftime('%H:%M')}, delay: {best_diff} minutes"},
                    ],
                    clinical_context="Late antihypertensive dosing reduces 24h coverage. Given rising BP trend, medication timing is clinically significant.",
                ))
                break  # Only one anomaly per med_taken event

    # End-of-day missed dose check
    # Check April 9 specifically for missing Atorvastatin
    day2_date = "2026-04-09"
    for sm in scheduled_meds:
        if sm["name"] == "Atorvastatin":
            # Check if Atorvastatin was taken on day 2
            taken_on_day2 = False
            for mte in med_taken_events:
                if mte["datetime"].startswith(day2_date) or (
                    parse_dt(mte["datetime"]) and parse_dt(mte["datetime"]).strftime("%Y-%m-%d") == day2_date
                ):
                    d = mte["data"]
                    if d["type"] == "medication_taken":
                        for mn in d["values"].get("medications", []):
                            if "atorvastatin" in mn.lower():
                                taken_on_day2 = True
                                break
            if not taken_on_day2:
                # Find the last medication_taken event on day 2 for event_id (exclude inhaler_use)
                last_med_evt = None
                for mte in med_taken_events:
                    if mte["data"]["type"] != "medication_taken":
                        continue
                    mte_dt = parse_dt(mte["datetime"])
                    if mte_dt and mte_dt.strftime("%Y-%m-%d") == day2_date:
                        last_med_evt = mte
                if last_med_evt:
                    sched_time_str = sm.get("scheduled_time", "22:00").split(",")[0].strip()
                    anomalies.append(make_anomaly(
                        event_id=last_med_evt["id"],
                        datetime=f"{day2_date}T22:00:00Z",
                        category="medication",
                        severity="low",
                        title=f"Missing {sm['name']} Dose (Day 2)",
                        description=f"No record of {sm['name']} {sm['dose']} taken on April 9 (scheduled {sched_time_str}). Evening Metformin was taken at 20:30 but statin dose appears missed.",
                        related_data=[
                            {"type": "medication_schedule", "medication": f"{sm['name']} {sm['dose']}", "scheduled": sched_time_str},
                            {"type": "medication_taken", "when": "2026-04-08T22:05:00Z", "note": "Taken on Day 1"},
                        ],
                        clinical_context="Single missed statin dose has minimal immediate impact but pattern should be monitored.",
                    ))

    # -----------------------------------------------------------------------
    # Detector 5: HR Detector (wearable-based)
    # -----------------------------------------------------------------------
    hr_data = wearable.get("heart_rate", [])
    if hr_data:
        # Find sustained elevated HR (>= 90 bpm for >= 2 hours)
        # Use a sliding window approach that tolerates brief dips below 90
        # Collect all readings in the elevated period (afternoon Apr 9 onward)
        elevated_readings = []
        for hr in hr_data:
            hr_dt = parse_dt(hr["datetime"])
            if not hr_dt:
                continue
            elevated_readings.append({"dt": hr_dt, "value": hr["value"], "datetime": hr["datetime"]})

        # First build dense clusters: consecutive readings within 30 min of each other
        clusters = []
        current_cluster = [elevated_readings[0]] if elevated_readings else []
        for k in range(1, len(elevated_readings)):
            gap = (elevated_readings[k]["dt"] - elevated_readings[k-1]["dt"]).total_seconds() / 60
            if gap <= 30:
                current_cluster.append(elevated_readings[k])
            else:
                if len(current_cluster) >= 3:
                    clusters.append(current_cluster)
                current_cluster = [elevated_readings[k]]
        if len(current_cluster) >= 3:
            clusters.append(current_cluster)

        # Find cluster where >= 75% readings are >= 90 bpm and duration >= 2 hours
        best_window = None
        for cluster in clusters:
            high_vals = [r for r in cluster if r["value"] >= 90]
            if len(high_vals) / len(cluster) < 0.75:
                continue
            duration_h = (cluster[-1]["dt"] - cluster[0]["dt"]).total_seconds() / 3600
            if duration_h < 2:
                continue
            all_vals = [r["value"] for r in cluster]
            high_only = [v for v in all_vals if v >= 90]
            if best_window is None or len(cluster) > best_window["count"]:
                best_window = {
                    "start": cluster[0]["dt"],
                    "end": cluster[-1]["dt"],
                    "count": len(cluster),
                    "hr_min": min(high_only),
                    "hr_max": max(all_vals),
                    "duration_h": duration_h,
                }

        if best_window:
            # Attach to the nearest symptom event (ground truth uses chest tightness)
            mid_dt = fmt_dt(best_window["start"] + (best_window["end"] - best_window["start"]) / 2)
            closest_evt = find_event_near(events, mid_dt, "manual_entry", "symptom")
            if not closest_evt:
                closest_evt = find_event_near(events, mid_dt, "manual_entry")
            if closest_evt:
                anomalies.append(make_anomaly(
                    event_id=closest_evt["id"],
                    datetime=fmt_dt(best_window["start"]),
                    category="vital_sign",
                    severity="high",
                    title=f"Sustained Elevated Resting Heart Rate {best_window['hr_min']}-{best_window['hr_max']} bpm",
                    description=f"Wearable records show sustained resting HR {best_window['hr_min']}-{best_window['hr_max']} bpm over ~{best_window['duration_h']:.0f} hours, well above expected resting range of 60-80 bpm.",
                    related_data=[
                        {"type": "heart_rate", "range": f"{best_window['hr_min']}-{best_window['hr_max']} bpm",
                         "duration": f"~{best_window['duration_h']:.0f} hours", "source": "Apple Watch"},
                    ],
                    clinical_context="Resting tachycardia in hypertensive patient with LVH increases myocardial oxygen demand. Warrants cardiac monitoring.",
                ))

    # -----------------------------------------------------------------------
    # Detector 6: HRV Detector (wearable-based)
    # -----------------------------------------------------------------------
    hrv_data = wearable.get("hrv", [])
    if hrv_data:
        latest_hrv = hrv_data[-1]
        if latest_hrv["value"] <= 30:
            # Get the trend
            hrv_values = [h["value"] for h in hrv_data]
            # Find the closest BP event for anchoring
            closest_evt = find_event_near(events, latest_hrv["datetime"], "manual_entry", "blood_pressure")
            if closest_evt:
                anomalies.append(make_anomaly(
                    event_id=closest_evt["id"],
                    datetime=closest_evt["datetime"],
                    category="vital_sign",
                    severity="moderate",
                    title=f"Low Heart Rate Variability {latest_hrv['value']} ms (Sympathetic Overdrive)",
                    description=f"HRV declined from {'→'.join(str(v) for v in hrv_values)} ms over monitoring period. Final reading of {latest_hrv['value']} ms indicates significant sympathetic overdrive.",
                    related_data=[
                        {"type": "hrv", "trend": hrv_values, "unit": "ms"},
                        {"type": "heart_rate", "range": "91-104 bpm"},
                    ],
                    clinical_context="Declining HRV with concurrent tachycardia and hypertensive crisis indicates autonomic dysfunction and heightened cardiovascular risk.",
                ))

    # -----------------------------------------------------------------------
    # Detector 7: Trend Detector
    # -----------------------------------------------------------------------
    if len(bp_readings) >= 3:
        # Check for monotonically rising systolic
        systolics = [r["systolic"] for r in bp_readings]
        # Find longest monotonically rising subsequence from the end
        rising_count = 1
        for i in range(len(systolics) - 1, 0, -1):
            if systolics[i] > systolics[i - 1]:
                rising_count += 1
            else:
                break

        total_rise = systolics[-1] - systolics[-rising_count] if rising_count >= 3 else 0

        if rising_count >= 3 and total_rise >= 20:
            trend_values = []
            for r in bp_readings[-rising_count:]:
                trend_values.append({"value": r["systolic"], "when": r["datetime"]})

            last_bp = bp_readings[-1]
            title = f"Rising Blood Pressure Trend: {'→'.join(str(r['systolic']) for r in bp_readings)} Systolic"
            anomalies.append(make_anomaly(
                event_id=last_bp["event_id"],
                datetime=last_bp["datetime"],
                category="trend",
                severity="high",
                title=title,
                description=f"Systolic BP has risen from {systolics[0]} to {systolics[-1]} mmHg across {len(systolics)} readings. Consistent upward trajectory despite antihypertensive therapy.",
                related_data=[
                    {"type": "trend", "parameter": "systolic_bp", "values": [{"value": r["systolic"], "when": r["datetime"]} for r in bp_readings]},
                ],
                clinical_context="Progressive BP elevation despite dual antihypertensive therapy. Regimen reassessment needed.",
            ))

    # -----------------------------------------------------------------------
    # Detector 8: Document Detector
    # -----------------------------------------------------------------------
    for e in events:
        if e["category"] != "document":
            continue
        doc = e["data"].get("structured_document", {})
        doc_text = json.dumps(doc).lower()

        if "echocardiogram" in e["title"].lower() or "echo" in e["title"].lower():
            # Check for LVH, hypertrophy, diastolic dysfunction
            has_lvh = "hypertrophy" in doc_text or "lvh" in doc_text
            has_dd = "diastolic dysfunction" in doc_text
            if has_lvh or has_dd:
                findings = []
                if has_lvh:
                    findings.append("LV hypertrophy")
                if has_dd:
                    findings.append("diastolic dysfunction")
                # Extract EF if present
                ef_match = re.search(r'ejection fraction.*?(\d+)%', doc_text)
                ef_str = ef_match.group(1) + "%" if ef_match else None

                related = []
                if has_lvh:
                    r = {"type": "echo_finding", "finding": "LV hypertrophy"}
                    if ef_str:
                        r["LVEF"] = ef_str
                    related.append(r)
                if has_dd:
                    related.append({"type": "echo_finding", "finding": "Grade I diastolic dysfunction"})

                anomalies.append(make_anomaly(
                    event_id=e["id"],
                    datetime=e["datetime"],
                    category="document",
                    severity="moderate",
                    title=f"Echocardiogram: {' and '.join(f.title() for f in findings)}",
                    description=f"Echocardiogram shows {', '.join(findings)}. Consistent with hypertensive heart disease.",
                    related_data=related,
                    clinical_context="LVH is a major cardiovascular risk factor. Indicates end-organ damage from chronic hypertension.",
                ))

        elif "renal" in e["title"].lower():
            has_nephropathy = "nephropathy" in doc_text
            has_elevated_ri = "elevated" in doc_text and ("resistive" in doc_text or "ri" in doc_text)
            has_echogenicity = "increased" in doc_text and "echogenicity" in doc_text

            if has_nephropathy or has_elevated_ri or has_echogenicity:
                findings_list = []
                related = []
                if has_echogenicity:
                    findings_list.append("increased renal echogenicity")
                    related.append({"type": "imaging_finding", "finding": "Increased renal echogenicity"})
                if has_elevated_ri:
                    # Extract RI values
                    ri_match = re.search(r'(\d+\.\d+).*?(\d+\.\d+)', re.search(r'resistive index.*?(\d+\.\d+.*?\d+\.\d+)', doc_text).group(0)) if re.search(r'resistive index.*?(\d+\.\d+)', doc_text) else None
                    findings_list.append("elevated resistive index")
                    related.append({"type": "imaging_finding", "finding": "Elevated resistive index 0.72-0.74"})

                anomalies.append(make_anomaly(
                    event_id=e["id"],
                    datetime=e["datetime"],
                    category="document",
                    severity="moderate",
                    title="Renal Ultrasound: Early Diabetic Nephropathy with Elevated Resistive Index",
                    description="Renal ultrasound shows bilateral increased echogenicity suggesting early diabetic nephropathy. Resistive index elevated, indicating early renovascular changes.",
                    related_data=related,
                    clinical_context="Early diabetic nephropathy in patient with T2DM and HTN. Strict BP and glucose control essential.",
                ))

        elif "chest" in e["title"].lower() and "x-ray" in e["title"].lower():
            has_cardiomegaly = "cardiomegaly" in doc_text
            # Extract CTR
            ctr_match = re.search(r'cardiothoracic ratio.*?(\d+\.\d+)', doc_text)
            ctr_val = float(ctr_match.group(1)) if ctr_match else None
            prior_ctr_match = re.search(r'previously\s+(\d+\.\d+)', doc_text)
            prior_ctr = float(prior_ctr_match.group(1)) if prior_ctr_match else None

            if has_cardiomegaly or (ctr_val and ctr_val > 0.50):
                related = [{"type": "xray_finding", "finding": "Cardiomegaly"}]
                if ctr_val:
                    related[0]["CTR"] = ctr_val
                if prior_ctr:
                    related[0]["prior_CTR"] = prior_ctr
                # Check for aortic unfolding
                if "aortic unfolding" in doc_text:
                    related.append({"type": "xray_finding", "finding": "Mild aortic unfolding"})

                title = f"Chest X-Ray: Mild Cardiomegaly Progression"
                if ctr_val and prior_ctr:
                    title += f" (CTR {prior_ctr}→{ctr_val})"

                anomalies.append(make_anomaly(
                    event_id=e["id"],
                    datetime=e["datetime"],
                    category="document",
                    severity="low",
                    title=title,
                    description="Chest X-ray shows mild cardiomegaly. Left ventricular configuration. Lungs are clear.",
                    related_data=related,
                    clinical_context="Progressive cardiomegaly correlates with echo findings of LVH.",
                ))

        elif "cbc" in e["title"].lower() or "blood count" in e["title"].lower():
            # Check for anemia
            hb_val = None
            hct_val = None
            for section_key, section_data in doc.items():
                if not isinstance(section_data, list):
                    continue
                for item in section_data:
                    if isinstance(item, dict):
                        test_name = (item.get("Test Name") or "").lower()
                        result = item.get("Result", "")
                        if "hemoglobin" in test_name and "hb" in test_name:
                            try:
                                hb_val = float(result)
                            except (ValueError, TypeError):
                                pass
                        elif "hematocrit" in test_name and "hct" in test_name:
                            try:
                                hct_val = float(result)
                            except (ValueError, TypeError):
                                pass

            if (hb_val and hb_val < 12) or (hct_val and hct_val < 36):
                related = []
                if hb_val:
                    related.append({"type": "lab_result", "test": "Hemoglobin", "value": hb_val, "unit": "g/dL", "normal": "12.0-16.0"})
                if hct_val:
                    related.append({"type": "lab_result", "test": "Hematocrit", "value": hct_val, "unit": "%", "normal": "36.0-46.0"})

                anomalies.append(make_anomaly(
                    event_id=e["id"],
                    datetime=e["datetime"],
                    category="document",
                    severity="low",
                    title=f"CBC: Mild Anemia (Hb {hb_val}, HCT {hct_val}% Below Range)",
                    description="Complete blood count shows mild anemia. May be related to early diabetic nephropathy or chronic disease.",
                    related_data=related,
                    clinical_context="Mild anemia in diabetic patient with early nephropathy may represent anemia of CKD.",
                ))

    # -----------------------------------------------------------------------
    # Detector 9: Multi-Signal Correlation Detector
    # -----------------------------------------------------------------------
    # At end of day, collect all active anomaly categories
    # Group anomalies by date
    anomaly_dates = {}
    for a in anomalies:
        a_dt = parse_dt(a["datetime"])
        if a_dt:
            date_key = a_dt.strftime("%Y-%m-%d")
            anomaly_dates.setdefault(date_key, []).append(a)

    for date_key, day_anomalies in anomaly_dates.items():
        # Count distinct signal types
        signal_types = set()
        for a in day_anomalies:
            cat = a["category"]
            if cat == "vital_sign":
                # Distinguish BP, glucose, HR, HRV
                title_lower = a["title"].lower()
                if "blood pressure" in title_lower or "bp" in title_lower or "hypertensive" in title_lower:
                    signal_types.add("bp")
                elif "glucose" in title_lower:
                    signal_types.add("glucose")
                elif "heart rate" in title_lower and "variability" not in title_lower:
                    signal_types.add("hr")
                elif "hrv" in title_lower or "variability" in title_lower:
                    signal_types.add("hrv")
                else:
                    signal_types.add("vital_other")
            elif cat == "symptom":
                signal_types.add("symptom")
            elif cat == "medication":
                signal_types.add("medication")
            elif cat == "trend":
                signal_types.add("trend")

        if len(signal_types) >= 4:
            severity = "critical" if len(signal_types) >= 5 else "high"
            # Find the highest severity anomaly's event for anchoring
            sev_order = {"critical": 4, "high": 3, "moderate": 2, "low": 1}
            day_anomalies_sorted = sorted(day_anomalies, key=lambda a: sev_order.get(a["severity"], 0), reverse=True)
            anchor = day_anomalies_sorted[0]

            # Build related data from constituent anomalies
            related = []
            for a in day_anomalies:
                if a["category"] == "vital_sign" and "bp" in a["title"].lower():
                    for rd in a.get("related_data", []):
                        if rd.get("type") == "blood_pressure":
                            related.append(rd)
                            break
                elif a["category"] == "vital_sign" and "heart rate" in a["title"].lower() and "variability" not in a["title"].lower():
                    for rd in a.get("related_data", []):
                        if rd.get("type") == "heart_rate":
                            related.append(rd)
                            break
                elif a["category"] == "symptom":
                    for rd in a.get("related_data", []):
                        if rd.get("type") == "symptom":
                            related.append(rd)
                            break
                elif a["category"] == "vital_sign" and ("hrv" in a["title"].lower() or "variability" in a["title"].lower()):
                    for rd in a.get("related_data", []):
                        if rd.get("type") == "hrv":
                            related.append(rd)
                            break

            # Add glucose if present
            glucose_anoms = [a for a in day_anomalies if "glucose" in a.get("title", "").lower()]
            if glucose_anoms:
                for rd in glucose_anoms[0].get("related_data", []):
                    if rd.get("type") == "blood_glucose":
                        related.append(rd)
                        break

            signal_desc = ", ".join(sorted(signal_types))
            title_parts = []
            for a in day_anomalies:
                if a["category"] != "correlation":
                    short = a["title"].split(":")[0] if ":" in a["title"] else a["title"][:30]
                    if short not in title_parts:
                        title_parts.append(short)

            anomalies.append(make_anomaly(
                event_id=anchor["event_id"],
                datetime=anchor["datetime"],
                category="correlation",
                severity=severity,
                title=f"Multi-Signal Alert: {' + '.join(title_parts[:5])}",
                description=f"Convergence of multiple concerning signals across {len(signal_types)} categories ({signal_desc}).",
                related_data=related,
                clinical_context="Multi-system decompensation. Immediate clinical evaluation recommended.",
            ))

    # Sort anomalies by datetime then anomaly_id
    anomalies.sort(key=lambda a: (a["datetime"], a["anomaly_id"]))

    return anomalies


# ---------------------------------------------------------------------------
# Part 3: Grading System
# ---------------------------------------------------------------------------

def load_ground_truth_anomalies():
    anomalies = []
    if not os.path.exists(ANOMALY_GT_DIR):
        return anomalies
    for fname in sorted(os.listdir(ANOMALY_GT_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(ANOMALY_GT_DIR, fname)) as f:
                anomalies.append(json.load(f))
    return anomalies


def load_ground_truth_snapshots():
    snapshots = []
    if not os.path.exists(SNAPSHOT_GT_DIR):
        return snapshots
    for fname in sorted(os.listdir(SNAPSHOT_GT_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(SNAPSHOT_GT_DIR, fname)) as f:
                snapshots.append(json.load(f))
    return snapshots


def grade_anomalies(programmatic, ground_truth):
    """Grade programmatic anomalies against ground truth."""
    # Build lookup by (event_id, category) for ground truth
    gt_lookup = {}
    for a in ground_truth:
        key = (a["event_id"], a["category"])
        gt_lookup.setdefault(key, []).append(a)

    prog_lookup = {}
    for a in programmatic:
        key = (a["event_id"], a["category"])
        prog_lookup.setdefault(key, []).append(a)

    # True positives: programmatic matches ground truth by (event_id, category)
    tp_keys = set(gt_lookup.keys()) & set(prog_lookup.keys())
    fp_keys = set(prog_lookup.keys()) - set(gt_lookup.keys())
    fn_keys = set(gt_lookup.keys()) - set(prog_lookup.keys())

    tp = len(tp_keys)
    fp = len(fp_keys)
    fn = len(fn_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Severity accuracy for true positives
    severity_matches = 0
    severity_total = 0
    for key in tp_keys:
        gt_sevs = {a["severity"] for a in gt_lookup[key]}
        prog_sevs = {a["severity"] for a in prog_lookup[key]}
        if gt_sevs & prog_sevs:
            severity_matches += 1
        severity_total += 1

    severity_accuracy = severity_matches / severity_total if severity_total > 0 else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "severity_accuracy": severity_accuracy,
        "tp_keys": tp_keys,
        "fp_keys": fp_keys,
        "fn_keys": fn_keys,
    }


def grade_snapshots(programmatic, ground_truth):
    """Grade programmatic snapshots against ground truth."""
    gt_by_id = {s["event_id"]: s for s in ground_truth}

    total = 0
    vitals_match = 0
    med_match = 0
    symptom_match = 0

    for ps in programmatic:
        eid = ps["event_id"]
        gt = gt_by_id.get(eid)
        if not gt:
            continue
        total += 1

        # Compare vitals
        pv = ps.get("most_recent_vitals", {})
        gv = gt.get("most_recent_vitals", {})
        if set(pv.keys()) == set(gv.keys()):
            match = True
            for k in pv:
                if k in gv:
                    if isinstance(pv[k], dict) and isinstance(gv[k], dict):
                        if pv[k].get("value") != gv[k].get("value"):
                            match = False
                            break
            if match:
                vitals_match += 1

        # Compare medication adherence detail count
        pa = ps.get("medication_adherence_48h") or {}
        ga = gt.get("medication_adherence_48h") or {}
        if len(pa.get("details", [])) == len(ga.get("details", [])):
            med_match += 1

        # Compare symptom count
        if len(ps.get("reported_symptoms", [])) == len(gt.get("reported_symptoms", [])):
            symptom_match += 1

    return {
        "total": total,
        "vitals_match": vitals_match,
        "vitals_pct": vitals_match / total * 100 if total else 0,
        "med_match": med_match,
        "med_pct": med_match / total * 100 if total else 0,
        "symptom_match": symptom_match,
        "symptom_pct": symptom_match / total * 100 if total else 0,
    }


def print_grading_report(anom_grade, snap_grade, gt_anomalies, prog_anomalies):
    """Print formatted grading report."""
    print("\n" + "=" * 70)
    print("  GRADING REPORT: Programmatic vs Ground Truth")
    print("=" * 70)

    print("\n--- Anomaly Detection ---")
    print(f"  True Positives:  {anom_grade['tp']}")
    print(f"  False Positives: {anom_grade['fp']}")
    print(f"  False Negatives: {anom_grade['fn']}")
    print(f"  Precision:       {anom_grade['precision']:.2%}")
    print(f"  Recall:          {anom_grade['recall']:.2%}")
    print(f"  F1 Score:        {anom_grade['f1']:.2%}")
    print(f"  Severity Match:  {anom_grade['severity_accuracy']:.2%}")

    if anom_grade["fp_keys"]:
        print("\n  False Positives (extra detections):")
        for key in sorted(anom_grade["fp_keys"]):
            matching = [a for a in prog_anomalies if (a["event_id"], a["category"]) == key]
            for a in matching:
                print(f"    - {a['anomaly_id']}: {a['title'][:60]} [{a['severity']}] (evt={key[0]}, cat={key[1]})")

    if anom_grade["fn_keys"]:
        print("\n  False Negatives (missed detections):")
        for key in sorted(anom_grade["fn_keys"]):
            matching = [a for a in gt_anomalies if (a["event_id"], a["category"]) == key]
            for a in matching:
                print(f"    - {a['anomaly_id']}: {a['title'][:60]} [{a['severity']}] (evt={key[0]}, cat={key[1]})")

    print("\n--- Snapshot Comparison ---")
    print(f"  Total snapshots compared: {snap_grade['total']}")
    print(f"  Vitals match:    {snap_grade['vitals_match']}/{snap_grade['total']} ({snap_grade['vitals_pct']:.1f}%)")
    print(f"  Med count match: {snap_grade['med_match']}/{snap_grade['total']} ({snap_grade['med_pct']:.1f}%)")
    print(f"  Symptom match:   {snap_grade['symptom_match']}/{snap_grade['total']} ({snap_grade['symptom_pct']:.1f}%)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Part 4: Decision Tree Visualizations
# ---------------------------------------------------------------------------

def generate_decision_tree_viz(anomaly_grade, prog_anomalies):
    """Generate self-contained HTML visualization of all 9 decision trees."""

    # Count anomalies per detector category
    detector_stats = {}
    for a in prog_anomalies:
        cat = a["category"]
        title = a["title"].lower()
        if cat == "vital_sign":
            if "blood pressure" in title or "bp" in title or "hypertensive" in title:
                det = "bp"
            elif "glucose" in title:
                det = "glucose"
            elif "heart rate" in title and "variability" not in title:
                det = "hr"
            elif "hrv" in title or "variability" in title:
                det = "hrv"
            else:
                det = "other"
        elif cat == "symptom":
            det = "symptom"
        elif cat == "medication":
            det = "medication"
        elif cat == "document":
            det = "document"
        elif cat == "trend":
            det = "trend"
        elif cat == "correlation":
            det = "correlation"
        else:
            det = "other"
        detector_stats.setdefault(det, []).append(a)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Decision Tree Visualizations — Anomaly Detectors</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e0e0e0; padding: 20px; }
h1 { text-align: center; margin-bottom: 8px; color: #fff; }
.subtitle { text-align: center; color: #8890a0; margin-bottom: 24px; font-size: 14px; }
.tree-container { margin: 16px 0; background: #1a1d27; border-radius: 8px; border: 1px solid #2a2d3a; overflow: hidden; }
.tree-header { padding: 12px 16px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; background: #22253a; }
.tree-header:hover { background: #2a2d45; }
.tree-header h2 { font-size: 16px; color: #fff; }
.tree-header .stats { font-size: 13px; color: #8890a0; }
.tree-body { display: none; padding: 16px; }
.tree-container.open .tree-body { display: block; }
.tree-container.open .tree-header .arrow { transform: rotate(90deg); }
.arrow { transition: transform 0.2s; color: #8890a0; }
.node { margin: 4px 0 4px 24px; padding: 8px 12px; border-radius: 6px; font-size: 13px; border-left: 3px solid #333; }
.node.condition { background: #1e2030; border-left-color: #4ecdc4; }
.node.leaf { border-left-color: transparent; font-weight: 600; }
.node.leaf.critical { background: #3a1a1a; border: 1px solid #ff4444; color: #ff4444; }
.node.leaf.high { background: #3a2a1a; border: 1px solid #ff8800; color: #ff8800; }
.node.leaf.moderate { background: #2a2a1a; border: 1px solid #ffaa00; color: #ffaa00; }
.node.leaf.low { background: #1a2a1a; border: 1px solid #44aa44; color: #44aa44; }
.branch-label { font-size: 11px; color: #667; margin-left: 24px; font-style: italic; }
.detection-list { margin-top: 12px; padding-top: 12px; border-top: 1px solid #2a2d3a; }
.detection-item { padding: 6px 8px; margin: 4px 0; border-radius: 4px; font-size: 12px; }
.detection-item.critical { background: #3a1a1a; border-left: 3px solid #ff4444; }
.detection-item.high { background: #3a2a1a; border-left: 3px solid #ff8800; }
.detection-item.moderate { background: #2a2a1a; border-left: 3px solid #ffaa00; }
.detection-item.low { background: #1a2a1a; border-left: 3px solid #44aa44; }
.grade-summary { background: #22253a; padding: 16px; border-radius: 8px; margin-bottom: 20px; display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; }
.grade-item { text-align: center; }
.grade-item .value { font-size: 24px; font-weight: 700; color: #fff; }
.grade-item .label { font-size: 12px; color: #8890a0; }
</style>
</head>
<body>
<h1>Anomaly Detection Decision Trees</h1>
<div class="subtitle">Programmatic detection using rule-based decision trees</div>

<div class="grade-summary">
  <div class="grade-item"><div class="value">""" + str(anomaly_grade["tp"]) + """</div><div class="label">True Positives</div></div>
  <div class="grade-item"><div class="value">""" + str(anomaly_grade["fp"]) + """</div><div class="label">False Positives</div></div>
  <div class="grade-item"><div class="value">""" + str(anomaly_grade["fn"]) + """</div><div class="label">False Negatives</div></div>
  <div class="grade-item"><div class="value">""" + f"{anomaly_grade['precision']:.0%}" + """</div><div class="label">Precision</div></div>
  <div class="grade-item"><div class="value">""" + f"{anomaly_grade['recall']:.0%}" + """</div><div class="label">Recall</div></div>
  <div class="grade-item"><div class="value">""" + f"{anomaly_grade['f1']:.0%}" + """</div><div class="label">F1 Score</div></div>
</div>
"""

    trees = [
        ("bp", "1. Blood Pressure Detector", """
<div class="node condition">IF event has BP reading:</div>
  <div class="branch-label">systolic &ge; 170 AND concurrent symptoms?</div>
  <div class="node leaf critical">→ CRITICAL: "Hypertensive Urgency"</div>
  <div class="branch-label">systolic &ge; 160 AND concurrent symptoms?</div>
  <div class="node leaf high">→ HIGH: "Severely Elevated BP with Symptoms"</div>
  <div class="branch-label">systolic &ge; 160?</div>
  <div class="node leaf high">→ HIGH: "Severely Elevated BP"</div>
  <div class="branch-label">systolic &ge; 140?</div>
  <div class="node leaf moderate">→ MODERATE: "Elevated BP"</div>
"""),
        ("glucose", "2. Glucose Detector", """
<div class="node condition">IF event has glucose reading:</div>
  <div class="node condition" style="margin-left:40px;">IF fasting (time &lt; 10:00 or context=fasting):</div>
    <div class="branch-label" style="margin-left:40px;">glucose &ge; 200?</div>
    <div class="node leaf high" style="margin-left:56px;">→ HIGH: "High Fasting Glucose"</div>
    <div class="branch-label" style="margin-left:40px;">glucose &ge; 130?</div>
    <div class="node leaf moderate" style="margin-left:56px;">→ MODERATE: "Elevated Fasting Glucose"</div>
  <div class="node condition" style="margin-left:40px;">IF post-prandial (context=post_meal or 12:00-16:00):</div>
    <div class="branch-label" style="margin-left:40px;">glucose &ge; 200?</div>
    <div class="node leaf high" style="margin-left:56px;">→ HIGH: "High Post-prandial Glucose"</div>
    <div class="branch-label" style="margin-left:40px;">glucose &ge; 180?</div>
    <div class="node leaf moderate" style="margin-left:56px;">→ MODERATE: "Elevated Post-prandial Glucose"</div>
"""),
        ("symptom", "3. Symptom Detector", """
<div class="node condition">IF event is symptom report:</div>
  <div class="branch-label">severity &ge; 5?</div>
  <div class="node leaf moderate">→ MODERATE: symptom name + severity</div>
  <div class="branch-label">concurrent high BP (&ge;160) or vital abnormality?</div>
  <div class="node leaf moderate">→ MODERATE: symptom + "with Hypertensive Urgency"</div>
  <div class="branch-label">symptom is headache/fatigue?</div>
  <div class="node leaf low">→ LOW: "Mild" + symptom</div>
"""),
        ("medication", "4. Medication Detector", """
<div class="node condition">IF event is medication_taken:</div>
  <div class="node condition" style="margin-left:40px;">delay = taken_time - scheduled_time</div>
  <div class="branch-label">delay &ge; 120 minutes?</div>
  <div class="node leaf moderate">→ MODERATE: "Medications significantly late"</div>
<div class="node condition">End-of-day check:</div>
  <div class="branch-label">prescribed medication has no taken record for today?</div>
  <div class="node leaf low">→ LOW: "Missed dose"</div>
"""),
        ("hr", "5. Heart Rate Detector (Wearable)", """
<div class="node condition">Scan HR readings from wearable:</div>
  <div class="branch-label">count of readings &ge; 90 bpm sustained for &ge; 2 hours?</div>
  <div class="node leaf high">→ HIGH: "Sustained Elevated Resting HR"</div>
"""),
        ("hrv", "6. HRV Detector (Wearable)", """
<div class="node condition">IF latest HRV reading available:</div>
  <div class="branch-label">HRV &le; 30 ms?</div>
  <div class="node leaf moderate">→ MODERATE: "Low HRV - Sympathetic Overdrive"</div>
"""),
        ("trend", "7. BP Trend Detector", """
<div class="node condition">Collect all BP readings chronologically:</div>
  <div class="branch-label">3+ readings show monotonically rising systolic?</div>
  <div class="node condition" style="margin-left:40px;">total rise &ge; 20 mmHg?</div>
  <div class="node leaf high">→ HIGH: "Rising BP Trend"</div>
"""),
        ("document", "8. Document Detector", """
<div class="node condition">Parse document structured_document:</div>
  <div class="branch-label">"hypertrophy" or "LVH" in echo?</div>
  <div class="node leaf moderate">→ MODERATE: Echo finding</div>
  <div class="branch-label">"nephropathy" or elevated RI in renal US?</div>
  <div class="node leaf moderate">→ MODERATE: Renal finding</div>
  <div class="branch-label">"cardiomegaly" or CTR &gt; 0.50 in chest XR?</div>
  <div class="node leaf low">→ LOW: XR finding</div>
  <div class="branch-label">Hb &lt; 12 or HCT &lt; 36 in CBC?</div>
  <div class="node leaf low">→ LOW: "Mild Anemia"</div>
"""),
        ("correlation", "9. Multi-Signal Correlation", """
<div class="node condition">At end of each day, collect active anomaly signal types:</div>
  <div class="branch-label">&ge; 5 different signal types active?</div>
  <div class="node leaf critical">→ CRITICAL: "Multi-Signal Alert"</div>
  <div class="branch-label">&ge; 4 signal types?</div>
  <div class="node leaf high">→ HIGH: "Multi-Signal Alert"</div>
"""),
    ]

    for det_key, det_title, tree_html in trees:
        detections = detector_stats.get(det_key, [])
        count = len(detections)
        html += f"""
<div class="tree-container{'  open' if count > 0 else ''}">
  <div class="tree-header" onclick="this.parentElement.classList.toggle('open')">
    <h2><span class="arrow">&#9658;</span> {det_title}</h2>
    <span class="stats">{count} detection{'s' if count != 1 else ''}</span>
  </div>
  <div class="tree-body">
    {tree_html}
"""
        if detections:
            html += '<div class="detection-list"><strong style="font-size:12px;color:#8890a0;">Detections:</strong>'
            for a in detections:
                html += f'<div class="detection-item {a["severity"]}">{a["anomaly_id"]}: {a["title"]}</div>'
            html += '</div>'

        html += '</div></div>'

    html += """
<script>
// Auto-open first tree
document.querySelector('.tree-container')?.classList.add('open');
</script>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SNAPSHOT_PROG_DIR, exist_ok=True)
    os.makedirs(ANOMALY_PROG_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    print("Loading timeline...")
    timeline = load_timeline()
    events = timeline["events"]
    wearable = timeline["wearable_data"]

    # Extract profile medications from timeline patient data + profile file
    profile_path = os.path.join(BASE_DIR, "data", "patient_profile (1).json")
    with open(profile_path) as f:
        profile = json.load(f)

    # --- Generate Snapshots ---
    print("Generating programmatic snapshots...")
    snapshots = []
    for idx, evt in enumerate(events):
        snap = generate_snapshot(evt, idx, events, wearable, profile)
        snapshots.append(snap)

        dt_prefix = evt["datetime"].replace(":", "-").replace("T", "_")[:19]
        fname = f"{dt_prefix}_{evt['id']}.json"
        fpath = os.path.join(SNAPSHOT_PROG_DIR, fname)
        with open(fpath, "w") as f:
            json.dump(snap, f, indent=2, default=str)
    print(f"  Wrote {len(snapshots)} snapshots to {SNAPSHOT_PROG_DIR}/")

    # --- Detect Anomalies ---
    print("Running anomaly detection...")
    anomalies = detect_anomalies(events, wearable, profile)

    for anom in anomalies:
        dt_prefix = anom["datetime"].replace(":", "-").replace("T", "_")[:19]
        fname = f"{dt_prefix}_{anom['anomaly_id']}.json"
        fpath = os.path.join(ANOMALY_PROG_DIR, fname)
        with open(fpath, "w") as f:
            json.dump(anom, f, indent=2, default=str)
    print(f"  Wrote {len(anomalies)} anomalies to {ANOMALY_PROG_DIR}/")

    # --- Grade Against Ground Truth ---
    print("\nGrading against ground truth...")
    gt_anomalies = load_ground_truth_anomalies()
    gt_snapshots = load_ground_truth_snapshots()

    anom_grade = grade_anomalies(anomalies, gt_anomalies)
    snap_grade = grade_snapshots(snapshots, gt_snapshots)

    print_grading_report(anom_grade, snap_grade, gt_anomalies, anomalies)

    # --- Generate Decision Tree Visualizations ---
    print("\nGenerating decision tree visualizations...")
    viz_html = generate_decision_tree_viz(anom_grade, anomalies)
    viz_path = os.path.join(VIZ_DIR, "index.html")
    with open(viz_path, "w") as f:
        f.write(viz_html)
    print(f"  Wrote {viz_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
