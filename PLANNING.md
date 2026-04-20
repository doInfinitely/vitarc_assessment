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
