# Architecture Write-Up

## Key Architectural Decisions

**Event-sourced timeline as the core data model.** Every piece of clinical data — diagnoses, prescriptions, manual vitals, wearable readings, document findings — is normalized into a chronologically-sorted event list with a shared schema (`datetime`, `category`, `subcategory`, `data`). This lets all downstream processing (snapshots, anomalies, visualization) work against a single unified structure rather than reaching back into heterogeneous source formats. The timeline is serialized as a single `timeline.json` that serves as the contract between stages.

**Separate ground truth from programmatic outputs.** I maintain parallel directories (`snapshot_ground_truth/` vs `snapshot_programmatic/`, `anomaly_ground_truth/` vs `anomaly_programmatic/`) and grade one against the other. This makes the system self-evaluating: every run prints precision/recall/F1, so regressions are immediately visible. The viewer merges both sets and shows provenance badges, giving clinical reviewers transparency into what was human-curated vs algorithmically detected.

**Decision-tree anomaly detection, not ML.** With only 2 days of monitoring data from 1 patient, statistical or learned models would be badly overfit. Rule-based decision trees with clinical thresholds (ADA glucose targets, JNC BP staging) are transparent, auditable, and match clinical guidelines directly. Each of the 9 detectors is independent, making them individually testable and extensible.

**Self-contained HTML viewer with no build step.** The viewer is a single HTML file with all data embedded as a JSON literal and all rendering done in vanilla Canvas/JS. No framework, no bundler, no server. This means a clinician can open it from a file share or email attachment. The tradeoff is file size (~200KB), which is acceptable for a single-patient view.

## Tradeoffs Considered

**Heading clustering for cross-document normalization.** Clinical documents use inconsistent section names ("Clinical Correlation" vs "Clinical Indication", "Conclusion" vs "Findings"). I use sentence-transformer embeddings with agglomerative clustering to merge semantically equivalent headings. The tradeoff: this requires a ~100MB model download and adds ~5 seconds of latency. The alternative was a hand-coded synonym map, which would be faster but brittle to new document types. The embedding approach generalizes to unseen heading variations.

**Wearable data granularity.** The HR track has 509 readings over 2 days (~1 reading per 5 minutes). I keep all readings in the timeline rather than pre-aggregating to hourly summaries. This preserves the ability to detect short-duration anomalies (the sustained tachycardia detector needs individual readings to identify dense clusters), but it inflates `timeline.json` and viewer load. For a production system, I would store raw readings separately and include only summary statistics in the timeline.

**Snapshot generation is intentionally deterministic.** The snapshot for each event is a pure function of the events and wearable data up to that point. No randomness, no model inference. This means snapshots are reproducible across runs, which is essential for clinical audit trails. The downside is that snapshots don't capture contextual nuance that an LLM might provide (e.g., "the elevated BP is concerning given the echo showing LVH").

## What I Chose Not to Build

**A REST API or database layer.** The exercise focuses on the data pipeline, and the natural interface is a CLI that processes files and produces files. Adding Flask/FastAPI would be premature complexity when the pipeline is batch-oriented. The `run_pipeline.py` script with `--from` and `--dump` flags provides the "exercise the system" interface that a test harness or CI job can call.

**Medication interaction checking.** The patient is on Amlodipine + Losartan + Metformin + Atorvastatin + Salbutamol. A production system would flag potential interactions (e.g., NSAID contraindication with the known allergy), but this requires a drug database (RxNorm/DrugBank) that is out of scope.

**Trend forecasting.** The trend detector identifies rising BP and glucose patterns, but doesn't project forward. A production system might use simple linear regression or ARIMA on vitals to predict when thresholds will be breached, enabling proactive intervention.

## Integrating an LLM or Agentic Component

If this were a real system, I would add an LLM layer at two points. First, **document understanding**: the current pipeline uses regex and keyword matching to extract findings from clinical reports (e.g., grepping for "hypertrophy" in the echocardiogram). An LLM could parse free-text clinical narratives into structured findings with confidence scores, handling the long tail of phrasing variations and implicit findings that rule-based parsers miss. Second, **clinical narrative generation**: each anomaly currently has a hand-templated description. An LLM could generate contextually rich narratives that cross-reference the patient's full history — for example, noting that the rising BP trend is especially concerning given the echo showing LVH and the renal ultrasound showing early nephropathy. The key constraint is that the LLM output must be grounded in the structured data (not hallucinated), so I would implement it as a retrieval-augmented generation step where the LLM receives the relevant snapshot, anomaly, and document findings as structured context, with the rule-based detections serving as the source of truth for what anomalies exist and the LLM only providing the natural-language layer.

## Scaling to 10,000 Concurrent Patients

```
                    ┌─────────────┐
                    │  API Gateway │
                    │  (auth/rate) │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴───┐ ┌──────┴──────┐
        │ Ingestion  │ │ Query │ │  Streaming  │
        │  Workers   │ │  API  │ │  (WebSocket)│
        └─────┬─────┘ └───┬───┘ └──────┬──────┘
              │            │            │
        ┌─────┴────────────┴────────────┴─────┐
        │         Event Store (Postgres)       │
        │   patients | events | wearable_agg   │
        └─────────────────┬───────────────────┘
                          │
                 ┌────────┴────────┐
                 │  Anomaly Engine │
                 │  (async workers)│
                 └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 │  Object Storage │
                 │  (S3: PDFs/JSON)│
                 └─────────────────┘
```

The single-patient file-based pipeline would evolve in three ways. **Storage**: replace `timeline.json` files with a Postgres event store partitioned by `patient_id`, with a time-series extension (TimescaleDB) for wearable data — individual HR/SpO2 readings go into hypertables with automatic compression, while events and anomalies stay in regular tables. Raw documents (PDFs, images) move to S3 with structured extractions cached in the database. **Compute**: the anomaly detection loop becomes an async worker pool (Celery/Temporal) where each patient's detection runs independently. When new data arrives (a manual entry, a wearable sync, a new lab result), only that patient's detectors re-run — not a full batch. The 9 detectors are embarrassingly parallel per-patient, so horizontal scaling is straightforward. **Delivery**: the viewer becomes a thin React client backed by a REST API (`GET /patients/{id}/timeline`, `GET /patients/{id}/anomalies?severity=critical`), with WebSocket push for real-time anomaly alerts. At 10K patients, the bottleneck is wearable ingestion throughput (potentially millions of readings/day), which is handled by the time-series partitioning and by aggregating to 5-minute windows for storage while keeping full resolution only for the latest 48 hours.
