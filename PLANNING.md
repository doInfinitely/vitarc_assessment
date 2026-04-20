# Project Planning Log

## Plan 1: Programmatic PDF-to-Markdown Converter

### Context
We produced markdown from 3 clinical PDFs using LLM interpretation. Now we need a deterministic Python script that approximates the same transformations using heuristics on `pdftotext -layout` output, so we can measure how close rule-based conversion gets vs LLM-produced ground truth.

### Approach: Single Python script `pdf_to_markdown.py`

#### Input/Output
- Reads PDFs from `data/` using `pdftotext -layout` (subprocess)
- Writes markdown to `markdown_programmatic/`
- Runs comparison against `markdown_ground_truth/` and prints Levenshtein distance + table count for each file

#### Heuristic Pipeline (applied line-by-line on layout-mode text)

1. **Strip Arabic text** — Remove RTL characters/lines (they're duplicates of English content in the lab report)

2. **Title detection** — First few non-empty lines with heavy leading whitespace (centered) or at the very top become `#` / `##` headings

3. **ALL-CAPS lines** — Lines that are mostly uppercase (e.g. "ULTRASOUND REPORT — RENAL", "TRANSTHORACIC ECHOCARDIOGRAM REPORT", "LABORATORY REPORT") become `##` headings

4. **Key-value pair detection** — Lines matching `Key: Value` with large gaps (2+ KV pairs per line) get collected into a `| Field | Value |` table. Single KV pairs on consecutive lines also get grouped.

5. **Table detection** — Consecutive lines with consistent column alignment (large whitespace gaps at similar positions) get parsed into markdown tables. Column boundaries detected by splitting on runs of 3+ whitespace.

6. **Section headers** — Short lines ending with `:` (e.g. "Findings:", "Impression:") followed by indented content become `###` headings

7. **Sub-labels within sections** — Indented lines starting with `Label:` (e.g. "Right Kidney:", "Left Ventricle:") become `**Label:**` bold inline labels, with continuation lines joined

8. **Numbered lists** — Lines starting with `1.`, `2.` etc. in impression sections become markdown numbered lists. Inline numbered items (e.g., `1. foo 2. bar`) are split into separate list items.

9. **Signature block** — Lines with `_____` become `---`, lines starting with `Dr.` get preserved, "electronically signed" lines become italic

10. **Footer** — Lines with Report ID, Page, P.O. Box etc. preserved as-is at bottom

#### Comparison Script (in same file)
- Compute Levenshtein distance between programmatic and ground truth markdown (using `difflib` SequenceMatcher + manual DP)
- Count markdown tables (`| ... |` blocks) in each output
- Print summary table

#### Files created
- `pdf_to_markdown.py` — the converter + comparison script
- `markdown_programmatic/` — output directory for generated markdown

#### Dependencies
- Python 3 standard library only (`subprocess`, `re`, `os`, `difflib`, `unicodedata`)
- `pdftotext` CLI (already available)

#### Results

| File | Levenshtein Dist | Similarity | Tables(prog) | Tables(truth) |
|---|---|---|---|---|
| echocardiogram_fakeeh.md | 0 | 100.0% | 2 | 2 |
| lab_cbc_kauh.md | 648 | 56.6% | 2 | 2 |
| renal_ultrasound_sgh.md | 0 | 100.0% | 1 | 1 |

**Notes on lab_cbc_kauh.md gap:** The remaining 56.6% similarity is due to semantic differences the LLM introduced that a rule-based converter cannot replicate:
- The LLM reordered tests into logical clinical groups (RBC indices, platelet indices, WBC differential)
- The LLM added unit annotations (e.g., `(K/uL)`, `(g/dL)`, `(fL)`) and reformatted names with em-dash separators (e.g., `Hemoglobin — Hb (g/dL)`)
- These represent genuine value-added by LLM interpretation over deterministic conversion

## Plan 2: Chest X-ray JPEG → Markdown (OCR + Bold Detection)

### Context
We have `data/chest_xray_kauh.jpeg` — a scanned radiology report from KAUH. Unlike the 3 PDFs (which used `pdftotext`), this requires OCR. The task had two parts: (1) hand-write a ground truth markdown, (2) build a programmatic converter using Tesseract OCR + image-based bold detection via skeletonization.

### Approach: `image_to_markdown.py` with dual-pass OCR

#### Pipeline

```
JPEG → Preprocess (2x upscale) → Tesseract HOCR (dual pass: original + 2x)
                                          ↓
                                  Word bbox merging (2x for header, original for body)
                                          ↓
                                  Bold detection (Zhang-Suen thinning → stroke width ratio)
                                          ↓
                                  Line reconstruction (group words by y-position)
                                          ↓
                                  Structural heuristics → Markdown
                                          ↓
                                  Comparison vs ground truth
```

#### Key design decisions

1. **Dual-pass OCR** — Running Tesseract on both the original image and a 2x upscaled version, then merging results. The 2x version reads digits/dates better (fixed "2/03/2026" → "28/03/2026"), while the original captures more content in the lower page (impression items, doctor names). Top 25% of image uses 2x words, rest uses original.

2. **Zhang-Suen thinning for bold detection** — Pure-Python implementation of the classic iterative thinning algorithm. Produces 1px-wide skeleton from binary word crops. Stroke width ratio (foreground/skeleton pixels) indicates bold when >2.5. At original resolution the ratios are ~1.6-2.5 (not enough differentiation), but bold detection data is included in diagnostics.

3. **Y-gap paragraph detection** — Lines in the Findings section are split into paragraphs based on vertical spacing. Typical line gap is ~16-17px; paragraph gap is ~22+ px. This correctly separates the 3 Findings paragraphs.

4. **OCR artifact cleanup** — Tesseract introduces leading `'`, `{`, `}`, `|`, `;` characters and misreads some words (e.g., "Saudl" → "Saudi"). A `clean_ocr_text()` function strips these artifacts from all output text.

5. **Pattern-based structural heuristics** — Section labels are detected by regex patterns (not just bold flags), allowing the converter to work even when bold detection fails. Handles `{Conclusion}` OCR variants of `[Conclusion]`.

#### Files created
- `markdown_ground_truth/chest_xray_kauh.md` — Hand-written ground truth
- `image_to_markdown.py` — OCR converter + bold detection + comparison

#### Dependencies
- Python: `cv2`, `numpy`, `subprocess`, `re`, `xml.etree.ElementTree`, `difflib`
- CLI: `tesseract` 5.5.2

#### Results

| File | Levenshtein Dist | Similarity | Tables(prog) | Tables(truth) |
|---|---|---|---|---|
| chest_xray_kauh.md | 22 | 96.6% | 1 | 1 |

**Notes on the remaining 3.4% gap:** The difference is purely OCR character-level accuracy:
- Impression items have minor punctuation errors ("," vs ".", missing trailing ".")
- Footer text has character-level OCR errors ("elocronic" vs "electronic", wrong date digits)
- Missing "1 / 1" page number (OCR didn't detect it)
- All structural elements are correctly detected and formatted

## Plan 3: Section Heading Embedding & Clustering

### Context
We have 4 clinical documents converted to markdown (3 PDFs + 1 JPEG). Each has `###` section headings. The goal is to embed these headings using an open-source embedding model, compute pairwise cosine distances, and explore agglomerative clustering at multiple distance thresholds to understand which headings group together.

### Heading Inventory (20 headings across 4 documents)

**chest_xray_kauh.md** (9): Clinical Dx., Medical History and Clinical Dx, Test Name, Position/Type, Conclusion, Clinical Indication, Comparison, Findings, Impression

**echocardiogram_fakeeh.md** (4): Chamber Dimensions & Function, Findings, Impression, Clinical Correlation

**lab_cbc_kauh.md** (1): Complete Blood Count (CBC)

**renal_ultrasound_sgh.md** (6): Clinical Indication, Technique, Comparison, Findings, Impression, Recommendation

### Approach: `cluster_headings.py`

#### Pipeline

```
Read all markdown_ground_truth/*.md → Extract ### headings with source doc
                    ↓
        Embed with all-MiniLM-L6-v2 (sentence-transformers, 384-dim)
                    ↓
        Cosine distance pairwise matrix (scikit-learn)
                    ↓
        Find threshold bounds:
          - min intra-doc distance (lower bound)
          - Clinical Correlation ↔ Clinical Indication distance (upper bound)
                    ↓
        AgglomerativeClustering at 7 thresholds (average linkage, cosine)
                    ↓
        PCA → 2D for visualization context
                    ↓
        Print summary tables to console
```

#### Constraint thresholds
- **Lower bound** (0.1970): smallest intra-doc cosine distance — between "Clinical Dx." and "Medical History and Clinical Dx" (both chest_xray_kauh)
- **Upper bound** (0.4489): cosine distance between "Clinical Correlation" (echocardiogram_fakeeh) and "Clinical Indication" (chest_xray_kauh)
- 7 evenly-spaced thresholds explored between these extremes

#### Results

| Threshold | Value  | Clusters | Singletons | Key merges |
|-----------|--------|----------|------------|------------|
| 1 (lower) | 0.1970 | 14       | 10         | Findings×3, Impression×3, Comparison×2, Clinical Indication×2 |
| 2         | 0.2390 | 13       | 8          | + Clinical Dx. merges with Medical History and Clinical Dx |
| 3         | 0.2810 | 13       | 8          | (same as threshold 2) |
| 4         | 0.3229 | 13       | 8          | (same as threshold 2) |
| 5         | 0.3649 | 13       | 8          | (same as threshold 2) |
| 6         | 0.4069 | 12       | 7          | + Conclusion merges into Findings cluster |
| 7 (upper) | 0.4489 | 11       | 6          | + Clinical Correlation merges with Clinical Indication cluster |

**Validation:**
- ✓ At lower bound: no same-document headings merged
- ✓ At upper bound: Clinical Correlation & Clinical Indication in same cluster

**Observations:**
- Identical headings (Findings, Impression, Comparison) merge immediately at the lowest threshold — the embedding model captures exact lexical matches perfectly
- "Clinical Dx." and "Medical History and Clinical Dx" merge at threshold 2 (0.2390) — these are same-doc headings from chest_xray_kauh, confirming the lower bound correctly prevents intra-doc merging
- Large stable plateau from thresholds 2-5 where no new merges occur
- "Conclusion" joins the Findings cluster at threshold 6 (semantically related: conclusions summarize findings)
- "Clinical Correlation" and "Clinical Indication" merge at the upper bound — both refer to clinical context/reasoning
- Persistent singletons even at upper bound: Complete Blood Count (CBC), Chamber Dimensions & Function, Test Name, Position/Type, Technique, Recommendation — these are domain-specific terms with no semantic neighbors

#### Merge sequence above upper bound (0.4489)

Including `##` headings (26 total: 6 from `##`, 20 from `###`), the full merge sequence above the chosen threshold:

| Distance | Merge | Assessment |
|----------|-------|------------|
| 0.539 | Technique → Impression cluster | Bad — technique ≠ impression |
| 0.559 | Clinical Dx cluster → Clinical Indication cluster | Reasonable — both "clinical context" |
| 0.564 | Comparison → Findings/Conclusion cluster | Reasonable — all "results" headings |
| 0.615 | Radiology dept names → Cardiology dept names | Reasonable — all department/title `##` headings |
| 0.622 | Impression+Technique ⊕ Findings+Comparison+Conclusion | Bad — mega-cluster of unrelated sections |
| 0.654 | Clinical cluster ⊕ all department titles | Bad — mixing `##` and `###` semantics |
| 0.675 | Test Name ⊕ Laboratory Report | Weak |
| 0.725 | Ultrasound Report — Renal → clinical/department mega-cluster | Bad |
| 0.744 | Recommendation → Impression/Findings/Comparison mega-cluster | Bad |
| 0.753 | Complete Blood Count (CBC) → clinical/department mega-cluster | Bad |
| 0.764 | Test Name/Lab Report → results mega-cluster | Bad |
| 0.807 | Position/Type → results mega-cluster | Bad |
| 0.844 | Two mega-clusters merge | Bad |
| 0.917 | Chamber Dimensions & Function (last singleton) → everything | Bad |

**Conclusion:** 0.4489 is the right threshold. The first merge above it (0.539, Technique → Impression) is already wrong. The useful range is 0.45–0.56 at best, but nothing above 0.4489 is clean.

#### Chosen threshold

**0.4489** — stored in `config.toml` as `[heading_clustering].threshold`.

At this threshold (15 clusters from 26 headings):
- 7 multi-heading clusters: Findings+Conclusion (4), Clinical Indication+Correlation (3), Impression (3), Radiology dept names (2), Echocardiogram titles (2), Clinical Dx variants (2), Comparison (2)
- 8 singletons: Laboratory Report, Ultrasound Report — Renal, Complete Blood Count (CBC), Test Name, Position/Type, Chamber Dimensions & Function, Technique, Recommendation

#### Dependencies
- `sentence-transformers` 5.2.3, `scikit-learn` 1.8.0, `scipy`, `torch` 2.10.0, `numpy` 2.4.2

#### Files
- `cluster_headings.py` — Embedding + clustering + analysis script (7 thresholds + above-bound merge scan)
- `config.toml` — Stores chosen clustering threshold (0.4489)

## Plan 4: Programmatic Markdown → Structured JSON

### Context
Convert the 4 programmatic markdown files (`markdown_programmatic/*.md`) into structured JSON files (`markdown_jsons/*.json`). Uses the heading clustering threshold (0.4489) from `config.toml` to merge semantically similar section headings, extracts document dates/times from metadata tables, and produces JSON with tables as lists of dicts and paragraphs as lists of sentences.

### Heading Clusters at Threshold 0.4489

These heading groups merge (from cluster_headings.py results):

| Cluster | Unique headings | Merged name |
|---------|----------------|-------------|
| Findings + Conclusion | Conclusion, Findings | `<Conclusion;Findings>` |
| Clinical variants | Clinical Correlation, Clinical Indication | `<Clinical Correlation;Clinical Indication>` |
| Impression (identical) | Impression | `Impression` (no rename) |
| Comparison (identical) | Comparison | `Comparison` (no rename) |
| Echo titles (##) | Cardiology Department — Echocardiography Laboratory, Transthoracic Echocardiogram Report | `<Cardiology Department — Echocardiography Laboratory;Transthoracic Echocardiogram Report>` |
| Radiology titles (##) | Department of Radiology, Radiology Test | `<Department of Radiology;Radiology Test>` |
| Clinical Dx variants | Clinical Dx., Medical History and Clinical Dx | `<Clinical Dx.;Medical History and Clinical Dx>` |

### Same-Document Heading Merges

When two headings from the same document cluster together, concatenate their content at the first occurrence position and remove the second:

| Document | Headings merged | Notes |
|----------|----------------|-------|
| echocardiogram_fakeeh | ## Cardiology Dept + ## Transthoracic Report | Patient table + address (from 1st) then all ### children (from 2nd) |
| chest_xray_kauh | ### Clinical Dx. + ### Medical History and Clinical Dx | Medical History section is empty; only Clinical Dx. content survives |
| chest_xray_kauh | ### Conclusion + ### Findings | Conclusion content prepended before Findings paragraphs |

### Document Date/Time Extraction

Reference date terms: "Exam Date", "Report Date", "Test Date", "Study Date", "Test Time", "Interpretation Time"

For each metadata table key (stripped of `**` bold markers), embed and check cosine distance to reference terms. If min distance ≤ 0.4489 AND value contains a parseable date (DD/MM/YYYY), treat as a date field. When multiple candidates match, pick the latest date.

Results:
| Document | Date field | Date | Time |
|----------|-----------|------|------|
| echocardiogram_fakeeh | Study Date / Report Date | 18/02/2026 | null |
| lab_cbc_kauh | Test Date | 01/04/2026 | null |
| renal_ultrasound_sgh | Report Date | 26/03/2026 | null |
| chest_xray_kauh | Interpretation Time | 31/03/2026 | 11:22:18 |

### Output JSON Structure

```json
{
  "date": "31/03/2026",
  "time": "11:22:18",
  "original_document_path": "data/chest_xray_kauh.jpeg",
  "markdown_document": "# King Abdulaziz University Hospital...",
  "structured_document": {
    "<Department of Radiology;Radiology Test>": [
      {"Field": "MRN", "Value": "2016-044891"}, ...
    ],
    "<Clinical Dx.;Medical History and Clinical Dx>": [
      ["Essential hypertension, Type 2 diabetes mellitus, Mild persistent asthma"]
    ],
    "Test Name": [["Chest X-ray"]],
    "<Conclusion;Findings>": [
      ["CHEST X-RAY PA AND LATERAL"],
      ["The heart is mildly enlarged...", "Left ventricular configuration.", ...],
      ...
    ],
    "Impression": [
      ["Mild cardiomegaly...", "Correlate with echocardiography."],
      ...
    ]
  }
}
```

#### Content type rules
- **Tables** (lines with `|`): list of dicts, one dict per row, keys from header row (stripped of `**`)
- **Paragraphs** (text blocks separated by blank lines): list of paragraphs, each paragraph is a list of sentences
- **Numbered lists** (`1. ...`): each item is its own paragraph (list of sentences)
- **Bold markers** (`**`) stripped from all paragraph text

#### Sentence splitting
Split on `(?<=[a-z0-9)%])\.\s+(?=[A-Z])` — period preceded by lowercase/digit/closing paren/%, followed by space and uppercase letter. Period preserved with preceding sentence.

#### Content excluded from structured_document
- `# H1` title (institution name)
- Text between H1 and first H2 (e.g., "Kingdom of Saudi Arabia...")
- `---` horizontal rules and everything after them within each section
- Signature blocks and footers after last `---`

### Pipeline

```
Load config.toml threshold → Load SentenceTransformer (all-MiniLM-L6-v2)
Extract ## and ### headings from markdown_programmatic/*.md
Embed → AgglomerativeClustering at threshold → build rename map + same-doc merges
    ↓
For each markdown file:
  1. Parse into flat section list (heading, level, content_lines)
  2. Strip content after --- in each section
  3. Extract metadata table from first H2
  4. Detect document date/time via semantic matching (latest date wins)
  5. Apply same-doc merges (concatenate absorbed sections, remove duplicates)
  6. Rename headings per cluster map
  7. Process content (tables → list of dicts, text → paragraphs → sentences)
  8. Build flat structured_document → write to markdown_jsons/<stem>.json
```

### Files

| File | Purpose |
|------|---------|
| `markdown_to_json.py` | Converter script |
| `markdown_jsons/*.json` | Output (4 JSON files) |

### Source → Original Document Path

| Stem | original_document_path |
|------|----------------------|
| echocardiogram_fakeeh | data/echocardiogram_fakeeh.pdf |
| lab_cbc_kauh | data/lab_cbc_kauh.pdf |
| renal_ultrasound_sgh | data/renal_ultrasound_sgh.pdf |
| chest_xray_kauh | data/chest_xray_kauh.jpeg |

### Dependencies
- `sentence-transformers`, `scikit-learn`, `numpy`, `tomllib` (stdlib)

### Results
- All 4 JSONs produced with correct dates, times, original document paths
- Merged heading names correct: `<Conclusion;Findings>` in echo/renal/chest, `<Clinical Correlation;Clinical Indication>` in echo/renal/chest
- Tables parsed as lists of dicts with correct keys (bold stripped)
- Paragraphs properly sentence-split with periods preserved
- No bold markers in paragraph text
- Lab CBC has no Conclusion/Findings (only metadata table + CBC results table)

## Plan 5: Unified Patient Timeline + Viewer + Ground Truth

### Context
Build a unified patient timeline from all 7 data sources, a browser-based interactive viewer, and generate medical snapshot / anomaly ground truth files using clinical knowledge. This is step 1 of the Vitarc take-home "Ingestion" + "Health Snapshot" + "Anomaly Detection" requirements.

### Data Sources → Timeline Events (35 total)

| Category | Source | Count | Events |
|----------|--------|-------|--------|
| `condition` | patient_profile.json conditions[] | 3 | Asthma 2005-06-01, HTN 2016-08-22, T2DM 2019-01-10 |
| `medication` | patient_profile.json medications[] | 5 | Salbutamol 2005-06-15, Metformin 2019-02-20, Losartan 2021-11-03, Atorvastatin 2022-09-01, Amlodipine 2023-05-15 |
| `visit` | patient_profile.json care_team[].last_visit | 3 | Dr. Batarfi 2025-10-30, Dr. Al-Harbi 2026-01-05, Dr. Al-Ghamdi 2026-02-18 |
| `document` | markdown_jsons/*.json | 4 | Echo 2026-02-18, Renal US 2026-03-26, Chest XR 2026-03-31, CBC 2026-04-01 |
| `appointment` | care_team[].next_appointment | 2 | Dr. Al-Harbi 2026-04-20, Dr. Al-Ghamdi 2026-06-10 (marked `planned`) |
| `manual_entry` | manual_entries.json entries[] | 18 | BP/glucose/meds/symptoms/inhaler across 2026-04-08/09 |

### Wearable Data (continuous tracks, not discrete events)

| Metric | Records | Range | Notes |
|--------|---------|-------|-------|
| Heart Rate | 509 | 56–104 bpm | 5-min intervals; elevated cluster 91–104 bpm on 04/09 evening |
| SpO2 | 62 | 0.95–0.97 | 15-min intervals; 1 empty value skipped; 1 Fitbit duplicate kept |
| Steps | 32 | hourly counts | Low activity; 1 iPhone source record |
| Sleep | 10 | 2 nights | 5 stages each: Core, Deep, REM, Awake |
| HRV | 7 | 25–75 ms | Declining trend: 72→75→55→63→32→28→25 ms |

### Data Messiness Handling

| Issue | Source | Fix |
|-------|--------|-----|
| Mixed timestamp formats | manual_entries (Z vs +03:00) | Parse with fromisoformat, normalize to UTC |
| Inconsistent BP keys | ME-015 uses `systolic`/`diastolic` instead of `_mmhg` suffix | Check both key variants, normalize to `_mmhg` |
| Mixed glucose units | ME-012 uses mmol/L instead of mg/dL | Detect key name, convert mmol/L × 18.018 → mg/dL (12.8 → 230.6) |
| Empty SpO2 value | wearable XML record at 21:00:08 +0300 | Skip records with empty value (62 kept of 63) |
| Duplicate HR reading | Fitbit + Apple Watch at same time (19:00 +0300) | Keep both, tag with source |
| Date-only events | conditions, medications, documents | Use T00:00:00 for sorting |
| DD/MM/YYYY dates | document JSONs | Parse explicitly as day/month/year |
| XML +0300 timestamps | wearable data | Subtract 3h → UTC |

### Timeline JSON Schema

```json
{
  "patient": { "patient_id", "demographics", "allergies" },
  "events": [
    { "id": "evt-001", "datetime", "category", "subcategory", "title", "source", "planned", "data" }
  ],
  "wearable_data": { "heart_rate": [...], "spo2": [...], "steps": [...], "sleep": [...], "hrv": [...] }
}
```

### Health Snapshot Generation (35 snapshots)

One file per event in `snapshot_ground_truth/<datetime>_<event_id>.json`. Each snapshot answers 5 clinical questions:

1. **Most recent vitals** — BP, HR, SpO2, blood glucose (from manual entries + wearable data up to event time)
2. **Medication adherence (48h window)** — Which meds taken, on time vs late, delay in minutes
3. **Reported symptoms (24h window)** — Active symptoms with severity scores
4. **Clinical findings summary** — Generated narrative based on vitals, symptoms, adherence, and document findings
5. **Care team attention items** — Actionable alerts based on clinical thresholds (BP ≥140, glucose ≥180, symptoms ≥5/10)

### Anomaly Generation (21 anomalies)

One file per anomaly in `anomaly_ground_truth/<datetime>_<anomaly_id>.json`. Categories and counts:

| Category | Count | Severity Range |
|----------|-------|---------------|
| vital_sign | 9 | moderate → critical |
| symptom | 3 | low → moderate |
| medication | 2 | low → moderate |
| document | 4 | low → moderate |
| trend | 1 | high |
| correlation | 1 | critical |

Key anomalies:
- **anom-011 (CRITICAL):** Hypertensive urgency — BP 171/101 mmHg with symptoms
- **anom-017 (CRITICAL):** Multi-signal alert — BP 171/101 + HR 91-104 + dizziness + chest tightness + declining HRV
- **anom-016 (HIGH):** Rising BP trend 134→142→148→162→171 systolic over 36h
- **anom-008 (HIGH):** Post-prandial glucose ~231 mg/dL
- **anom-013 (HIGH):** Sustained elevated resting HR 91-104 bpm
- **anom-018–021:** Document-based anomalies from echo (LVH), renal US (nephropathy), chest XR (cardiomegaly), CBC (anemia)

### Timeline Viewer

Self-contained HTML file (`timeline_viewer/index.html`, 165 KB) with inline CSS + JS + embedded data. No server required.

**Features:**
- Canvas-based timeline with all 35 events as colored dots
- Category color coding: condition=purple, medication=blue, visit=green, document=orange, appointment=gray (dashed), manual_entry=teal
- Anomaly overlay: red rings (thickness by severity, critical pulsing animation)
- Zoom (mouse wheel) from full 21-year view down to hourly on Apr 8-9
- Pan (click-drag) on timeline area
- Category filter checkboxes
- Click event → sidebar with event details, health snapshot, and anomaly cards
- Wearable data tracks (HR, SpO2, HRV) auto-show when zoomed into Apr 8-9 window
- Dark theme UI

### Pipeline

```
build_timeline.py
    ├── Parse patient_profile.json → patient context + condition/medication/visit/appointment events
    ├── Parse markdown_jsons/*.json → document events
    ├── Parse manual_entries.json → manual_entry events (normalize timestamps, units)
    ├── Parse wearable_export.xml → wearable_data tracks (normalize +0300 → UTC)
    ├── Sort all events chronologically, assign IDs (evt-001 through evt-035)
    ├── Write timeline.json
    ├── For each event: generate health snapshot → write to snapshot_ground_truth/
    ├── Generate all anomalies → write to anomaly_ground_truth/
    └── Generate timeline_viewer/index.html (embed all data)
```

### Files Created

| File | Purpose | Size |
|------|---------|------|
| `build_timeline.py` | Main ingestion + ground truth generation script | ~22 KB |
| `timeline.json` | Unified timeline data (35 events + wearable tracks) | — |
| `snapshot_ground_truth/*.json` | 35 health snapshot files | — |
| `anomaly_ground_truth/*.json` | 21 anomaly files | — |
| `timeline_viewer/index.html` | Self-contained interactive timeline viewer | 165 KB |

### Dependencies
- Python 3 standard library only (`json`, `os`, `re`, `xml.etree.ElementTree`, `datetime`)

### Verification Results
- ✓ 35 events sorted chronologically
- ✓ All 6 categories present (condition, medication, visit, document, appointment, manual_entry)
- ✓ Wearable data: HR=509, SpO2=62, Steps=32, Sleep=10, HRV=7
- ✓ 35 snapshot files with vitals, adherence, symptoms, clinical summary, attention items
- ✓ 21 anomaly files with appropriate severity levels
- ✓ Data messiness handled: BP key normalization, glucose unit conversion, empty SpO2 skipped, Fitbit duplicates kept with source tag
- ✓ Viewer HTML self-contained with embedded data, interactive zoom/pan/filter/sidebar
