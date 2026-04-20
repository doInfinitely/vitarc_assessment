#!/usr/bin/env python3
"""
Programmatic JPEG-to-Markdown converter using Tesseract OCR + bold detection.

Pipeline:
  JPEG → Tesseract HOCR → word bboxes + text
       → Image crop per word bbox → bold detection (skeletonize → stroke width)
       → Line reconstruction → Structural heuristics → Markdown
       → Comparison vs ground truth

Dependencies: Pillow, numpy, cv2, tesseract CLI
"""

import os
import re
import subprocess
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "markdown_programmatic")
TRUTH_DIR = os.path.join(BASE_DIR, "markdown_ground_truth")

IMAGE_FILE = "chest_xray_kauh.jpeg"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]


def count_tables(md: str) -> int:
    """Count distinct markdown tables (contiguous blocks of | lines)."""
    in_table = False
    count = 0
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                count += 1
                in_table = True
        else:
            in_table = False
    return count


def similarity_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Step 1: OCR with bounding boxes via Tesseract HOCR
# ---------------------------------------------------------------------------

UPSCALE_FACTOR = 2

def preprocess_image(image_path: str) -> str:
    """Upscale image for better OCR accuracy. Returns path to preprocessed image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    upscaled = cv2.resize(img, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
                          interpolation=cv2.INTER_CUBIC)
    out_path = image_path.rsplit(".", 1)[0] + "_2x.png"
    cv2.imwrite(out_path, upscaled)
    return out_path


def run_tesseract_hocr(image_path: str) -> str:
    """Run tesseract with HOCR output and return the XML string."""
    result = subprocess.run(
        ["tesseract", image_path, "stdout", "hocr", "--psm", "3"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def parse_hocr(hocr_xml: str) -> list[dict]:
    """Parse HOCR XML to extract per-word bounding boxes and text.

    Returns list of dicts: {text, x0, y0, x1, y1, conf}
    """
    # Strip namespace for simpler parsing
    cleaned = re.sub(r'\sxmlns="[^"]*"', '', hocr_xml, count=1)
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError:
        root = ET.fromstring(hocr_xml)

    words = []
    for elem in root.iter():
        cls = elem.get("class", "")
        if "ocrx_word" not in cls:
            continue

        title = elem.get("title", "")
        text = "".join(elem.itertext()).strip()
        if not text:
            continue

        bbox_match = re.search(r"bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", title)
        conf_match = re.search(r"x_wconf\s+(\d+)", title)

        if bbox_match:
            x0, y0, x1, y1 = (int(bbox_match.group(i)) for i in range(1, 5))
            conf = int(conf_match.group(1)) if conf_match else 0
            words.append({
                "text": text,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "conf": conf,
            })

    return words


def group_words_into_lines(words: list[dict], y_tolerance: int = 8) -> list[list[dict]]:
    """Group words into lines by y-coordinate overlap."""
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: ((w["y0"] + w["y1"]) / 2, w["x0"]))

    lines = []
    current_line = [sorted_words[0]]
    current_y = (sorted_words[0]["y0"] + sorted_words[0]["y1"]) / 2

    for w in sorted_words[1:]:
        w_y = (w["y0"] + w["y1"]) / 2
        if abs(w_y - current_y) <= y_tolerance:
            current_line.append(w)
        else:
            current_line.sort(key=lambda w: w["x0"])
            lines.append(current_line)
            current_line = [w]
            current_y = w_y

    if current_line:
        current_line.sort(key=lambda w: w["x0"])
        lines.append(current_line)

    return lines


def reconstruct_line_text(line_words: list[dict]) -> str:
    """Reconstruct line text from words, using x-gaps for spacing."""
    if not line_words:
        return ""
    parts = []
    for i, w in enumerate(line_words):
        if i > 0:
            gap = w["x0"] - line_words[i - 1]["x1"]
            if gap > 40:
                parts.append("   ")
            else:
                parts.append(" ")
        parts.append(w["text"])
    return "".join(parts)


def clean_ocr_text(text: str) -> str:
    """Remove common OCR artifacts from text."""
    # Remove leading/trailing punctuation artifacts (including Unicode quotes)
    text = re.sub(r"^[\u2018\u2019\u201C\u201D'\"{}\[\]|;]+\s*", "", text)
    text = re.sub(r"\s*[\u2018\u2019\u201C\u201D'\"{}\[\]|]+$", "", text)
    # Remove mid-text OCR artifacts (e.g. "aorta {is" -> "aorta is")
    text = re.sub(r"\s+[{}\[\]|'\"\u2018\u2019]\s*(?=[a-zA-Z])", " ", text)
    # Remove trailing pipe/semicolon artifacts
    text = re.sub(r"\s*[|;]+\s*$", "", text)
    # Remove standalone semicolons/pipes surrounded by spaces
    text = re.sub(r"\s+[;|]+\s+", " ", text)
    # Fix common OCR substitutions
    text = text.replace("Saudl", "Saudi")
    return text.strip()


# ---------------------------------------------------------------------------
# Step 2: Zhang-Suen thinning (skeletonization)
# ---------------------------------------------------------------------------

def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen iterative thinning algorithm.

    Input: binary image (0=background, 1=foreground)
    Output: 1px-wide skeleton (same format)

    Two sub-iterations per pass remove border pixels while preserving topology.
    """
    img = binary.copy().astype(np.uint8)
    rows, cols = img.shape

    def _neighbors(r, c):
        """Return 8-neighbors in clockwise order starting from top: P2..P9."""
        return [
            img[r - 1, c],     # P2
            img[r - 1, c + 1], # P3
            img[r, c + 1],     # P4
            img[r + 1, c + 1], # P5
            img[r + 1, c],     # P6
            img[r + 1, c - 1], # P7
            img[r, c - 1],     # P8
            img[r - 1, c - 1], # P9
        ]

    def _transitions(neighbors):
        """Count 0->1 transitions in the circular sequence P2..P9..P2."""
        n = neighbors + [neighbors[0]]
        return sum(1 for i in range(8) if n[i] == 0 and n[i + 1] == 1)

    changed = True
    while changed:
        changed = False

        # Sub-iteration 1
        to_remove = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if img[r, c] != 1:
                    continue
                n = _neighbors(r, c)
                b = sum(n)
                a = _transitions(n)
                if (2 <= b <= 6 and a == 1
                        and n[0] * n[2] * n[4] == 0
                        and n[2] * n[4] * n[6] == 0):
                    to_remove.append((r, c))
        for r, c in to_remove:
            img[r, c] = 0
            changed = True

        # Sub-iteration 2
        to_remove = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if img[r, c] != 1:
                    continue
                n = _neighbors(r, c)
                b = sum(n)
                a = _transitions(n)
                if (2 <= b <= 6 and a == 1
                        and n[0] * n[2] * n[6] == 0
                        and n[0] * n[4] * n[6] == 0):
                    to_remove.append((r, c))
        for r, c in to_remove:
            img[r, c] = 0
            changed = True

    return img


# ---------------------------------------------------------------------------
# Step 3: Bold detection via stroke width ratio
# ---------------------------------------------------------------------------

def detect_bold_word(gray_img: np.ndarray, bbox: dict, bold_threshold: float = 2.5) -> tuple[bool, float]:
    """Determine if a word is bold by measuring stroke width via skeletonization.

    Returns (is_bold, stroke_width_ratio).
    """
    x0, y0, x1, y1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    pad = 2
    y0p, y1p = max(0, y0 - pad), min(gray_img.shape[0], y1 + pad)
    x0p, x1p = max(0, x0 - pad), min(gray_img.shape[1], x1 + pad)

    crop = gray_img[y0p:y1p, x0p:x1p]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return False, 0.0

    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_01 = (binary > 0).astype(np.uint8)

    foreground = int(np.sum(binary_01))
    if foreground < 10:
        return False, 0.0

    skeleton = zhang_suen_thinning(binary_01)
    skeleton_pixels = int(np.sum(skeleton))

    if skeleton_pixels == 0:
        return False, 0.0

    ratio = foreground / skeleton_pixels
    return ratio > bold_threshold, ratio


def detect_bold_lines(gray_img: np.ndarray, lines: list[list[dict]],
                      bold_threshold: float = 2.5) -> list[tuple[bool, list[float]]]:
    """For each line, determine if it's bold (majority of words are bold).

    Returns list of (is_bold, per_word_ratios) tuples.
    """
    results = []
    for line_words in lines:
        valid_words = [w for w in line_words
                       if len(w["text"]) >= 2 and w["conf"] > 20]
        if not valid_words:
            results.append((False, []))
            continue

        bold_count = 0
        ratios = []
        for w in valid_words:
            is_bold, ratio = detect_bold_word(gray_img, w, bold_threshold)
            ratios.append(ratio)
            if is_bold:
                bold_count += 1

        line_bold = bold_count / len(valid_words) > 0.5 if valid_words else False
        results.append((line_bold, ratios))

    return results


# ---------------------------------------------------------------------------
# Step 4: Structural heuristics -> Markdown
# ---------------------------------------------------------------------------

def is_mostly_upper(text: str) -> bool:
    """Check if text is mostly uppercase letters."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) > 0.7


# Section label patterns (bold/underlined in the original)
# OCR may render brackets as { } instead of [ ]
_SECTION_LABELS = re.compile(
    r'^[\[{]?(?:Clinical\s+Dx\.?|Medical\s+History(?:\s+and\s+Clinical\s+Dx)?'
    r'|Test\s+Name|Position/?Type|Conclusion)[\]}]?\s*[:.]*\s*$', re.I)

# Conclusion sub-section headers (all-caps in the original)
_CONCLUSION_SECTIONS = re.compile(
    r'^(CLINICAL\s+INDICATION|COMPARISON|FINDINGS|IMPRESSION)\s*[:.]*\s*$', re.I)


def lines_to_markdown(line_texts: list[str], bold_flags: list[bool],
                      line_y_positions: list[float],
                      impression_override: list[str] | None = None) -> str:
    """Convert OCR line texts + bold flags into structured markdown.

    line_y_positions: average y-coordinate for each line (for paragraph gap detection).
    impression_override: if provided, use these text lines for the Impression
                         section instead of the OCR lines (from targeted re-scan).
    """
    md = []
    n = len(line_texts)
    i = 0
    kv_pairs = []
    table_emitted = False

    # --- Phase 1: Header (Kingdom, Ministry, Hospital, Jeddah) ---
    header_lines = []
    while i < n:
        t = line_texts[i]
        tl = t.lower()
        if any(k in tl for k in ("kingdom", "ministry", "university", "hospital",
                                  "jeddah", "education")):
            header_lines.append(t)
            i += 1
        elif not t:
            i += 1
        else:
            break

    # Extract hospital and preamble
    hospital = ""
    preamble_parts = []
    for hl in header_lines:
        if "hospital" in hl.lower() or "university" in hl.lower():
            hospital = hl
        elif "kingdom" in hl.lower() or "ministry" in hl.lower():
            preamble_parts.append(hl)

    if not hospital and header_lines:
        hospital = header_lines[0]

    # Normalize hospital name
    hospital_clean = clean_ocr_text(re.sub(r'\s+', ' ', hospital).strip())
    # Ensure " - Jeddah" suffix
    hospital_clean = re.sub(r'\s*[-,]?\s*Jeddah\s*$', '', hospital_clean, flags=re.I)
    hospital_clean += " - Jeddah"

    md.append(f"# {hospital_clean}")
    md.append("")
    if preamble_parts:
        md.append(" \u2014 ".join(clean_ocr_text(p) for p in preamble_parts))
        md.append("")

    # --- Phase 2: MRN / Name line ---
    while i < n:
        t = line_texts[i]
        if not t:
            i += 1
            continue
        if "mrn" in t.lower():
            mrn_match = re.search(r'MRN\s*[:.]\s*([\w-]+)', t, re.I)
            name_match = re.search(r'Name\s*[:.]\s*(.+?)(?:\s{3,}|$)', t, re.I)
            if mrn_match:
                kv_pairs.append(("MRN", mrn_match.group(1).strip()))
            if name_match:
                kv_pairs.append(("Name", name_match.group(1).strip()))
            i += 1
            break
        i += 1

    # --- Phase 3: Radiology Test heading + Dept/Ward + Dates ---
    while i < n:
        t = line_texts[i]
        tl = t.lower()
        if not t:
            i += 1
            continue

        # "Radiology Test" heading
        if "radiology" in tl and "test" in tl:
            md.append("## Radiology Test")
            md.append("")
            i += 1
            continue

        # Dept/Ward line
        if "dept" in tl and ("referred" in tl or "ward" in tl):
            dept_match = re.search(
                r'(?:Dept/?Ward\s*\(Referred\s+from\)\s*[:.]\s*)(.+)', t, re.I)
            if dept_match:
                kv_pairs.append(("Dept/Ward (Referred from)",
                                 dept_match.group(1).strip()))
            i += 1
            continue

        # Date line (Referral Date / Test Time / Interpretation Time)
        if "referral" in tl or ("test" in tl and "time" in tl):
            # Try to extract dates even from garbled OCR
            ref_match = re.search(r'(\d{1,2}/\d{2}/\d{4})', t)
            test_match = re.search(
                r'Test\s+Time\s*[:.]\s*(\d{1,2}/\d{2}/\d{4}\s+[\d:]+)', t, re.I)
            interp_match = re.search(
                r'Interpretation\s+Time\s*[:.]\s*(\d{1,2}/\d{2}/\d{4}\s+[\d:]+)', t, re.I)

            if ref_match and not test_match:
                kv_pairs.append(("Referral Date", ref_match.group(1)))
            elif ref_match:
                # First date is referral date
                all_dates = re.findall(r'(\d{1,2}/\d{2}/\d{4}(?:\s+[\d:]+)?)', t)
                if len(all_dates) >= 1:
                    kv_pairs.append(("Referral Date", all_dates[0].split()[0]))
            if test_match:
                kv_pairs.append(("Test Time", test_match.group(1)))
            if interp_match:
                kv_pairs.append(("Interpretation Time", interp_match.group(1)))
            i += 1
            continue

        # Hit a section label or Clinical Dx content -> done with header
        break

    # --- Phase 4: Emit patient info table ---
    if kv_pairs:
        md.append("| Field | Value |")
        md.append("|---|---|")
        for k, v in kv_pairs:
            md.append(f"| **{k}** | {v} |")
        md.append("")
        md.append("---")
        md.append("")
        table_emitted = True

    # --- Phase 5: Section labels and their content ---
    # Handle: Clinical Dx content (may appear before its heading),
    # section labels, conclusion sub-sections, signatures, footer

    # Check if current line is Clinical Dx content (appears before "Medical History")
    if i < n:
        t = line_texts[i]
        # If this line contains medical terms but isn't a section label, it's Clinical Dx content
        if (t and not _SECTION_LABELS.match(t) and not _CONCLUSION_SECTIONS.match(t)
                and ("hypertension" in t.lower() or "diabetes" in t.lower()
                     or "asthma" in t.lower() or "essential" in t.lower())):
            md.append("### Clinical Dx.")
            md.append("")
            md.append(clean_ocr_text(t))
            md.append("")
            i += 1

    # Process remaining lines
    found_first_doctor = False
    while i < n:
        t = line_texts[i]
        if not t:
            i += 1
            continue

        # Section label (bold/underlined in original)
        if _SECTION_LABELS.match(t):
            label = t.strip().rstrip(":.").strip()
            label = re.sub(r'[\[\]{}]', '', label).strip()
            # Normalize "Position/Type" variants
            label = re.sub(r'Position\s*/?\s*Type', 'Position/Type', label)
            label = re.sub(r'\s+', ' ', label)
            md.append(f"### {label}")
            md.append("")
            i += 1

            # Next line(s) are the section content (may span multiple lines)
            content_parts = []
            while i < n:
                content = line_texts[i]
                if (not content or _SECTION_LABELS.match(content)
                        or _CONCLUSION_SECTIONS.match(content)):
                    break
                # Check if this is an ALL-CAPS line that's actually the
                # conclusion title (e.g. "CHEST X-RAY PA AND LATERAL")
                if is_mostly_upper(content) and len(content) > 15:
                    content_parts.append(content)
                    i += 1
                    break
                content_parts.append(content)
                i += 1
                break  # Typically one line of content per section label
            if content_parts:
                md.append(clean_ocr_text(" ".join(content_parts)))
                md.append("")
            continue

        # Conclusion sub-section (CLINICAL INDICATION, COMPARISON, etc.)
        csm = _CONCLUSION_SECTIONS.match(t)
        if csm:
            section_name = csm.group(1).strip().title()
            md.append(f"### {section_name}")
            md.append("")
            i += 1

            # If this is the Impression section and we have re-scanned text,
            # use it directly instead of the (often incomplete) full-page OCR
            if "impression" in section_name.lower() and impression_override:
                # Join continuation lines (lines not starting with a number)
                # into their parent numbered item
                joined_items = []
                for imp_line in impression_override:
                    imp_line = imp_line.strip()
                    if not imp_line:
                        continue
                    # Normalize: "2," -> "2."
                    imp_line = re.sub(r'^(\d+),\s+', r'\1. ', imp_line)
                    if re.match(r'^\d+[.]\s+', imp_line):
                        joined_items.append(imp_line)
                    elif joined_items:
                        # Continuation of previous item
                        joined_items[-1] += " " + imp_line
                    else:
                        joined_items.append(imp_line)
                for item in joined_items:
                    md.append(clean_ocr_text(item))
                md.append("")
                # Skip past impression content in the main lines
                while i < n:
                    ct = line_texts[i]
                    if (ct.upper().startswith("DR.") and is_mostly_upper(ct)):
                        break
                    if _CONCLUSION_SECTIONS.match(ct):
                        break
                    i += 1
                continue

            # Collect body paragraphs, using y-gaps for paragraph detection
            para_lines = []
            while i < n:
                ct = line_texts[i]
                if not ct:
                    if para_lines:
                        md.append(clean_ocr_text(" ".join(para_lines)))
                        md.append("")
                        para_lines = []
                    i += 1
                    continue

                # Stop at next section header or doctor block
                if _CONCLUSION_SECTIONS.match(ct):
                    break
                if ct.upper().startswith("DR.") and is_mostly_upper(ct):
                    break

                # Numbered impression items
                num_match = re.match(r"^['\"\u2018\u2019\u201C\u201D]?(\d+)[.,]\s+(.+)$", ct)
                if num_match and "impression" in section_name.lower():
                    if para_lines:
                        md.append(clean_ocr_text(" ".join(para_lines)))
                        md.append("")
                        para_lines = []
                    item_num = num_match.group(1)
                    item_text = num_match.group(2)
                    i += 1
                    # Continuation lines
                    while i < n:
                        cc = line_texts[i]
                        if (not cc or re.match(r"^['\"\u2018\u2019\u201C\u201D]?\d+[.,]", cc)
                                or cc.upper().startswith("DR.")
                                or _CONCLUSION_SECTIONS.match(cc)):
                            break
                        # Check if continuation has inline numbered item
                        inline_num = re.search(
                            r"[.!?)]\s+['\"]?(\d+)[.,]\s+", cc)
                        if inline_num:
                            # Split: text before the number goes with current item
                            split_pos = inline_num.start() + 1
                            item_text += " " + cc[:split_pos]
                            # The rest starts a new item — put it back
                            remaining = cc[split_pos:].strip()
                            line_texts[i] = remaining
                            break
                        item_text += " " + cc
                        i += 1
                    md.append(f"{item_num}. {clean_ocr_text(item_text)}")
                    continue

                # Detect paragraph break via y-gap
                # If there's a significant y-gap before this line, start new paragraph
                if para_lines and i > 0:
                    y_gap = line_y_positions[i] - line_y_positions[i - 1]
                    # Typical line spacing is ~16-17px; paragraph gap is ~22+ px
                    if y_gap > 20:
                        md.append(clean_ocr_text(" ".join(para_lines)))
                        md.append("")
                        para_lines = []

                para_lines.append(ct)
                i += 1

            if para_lines:
                md.append(clean_ocr_text(" ".join(para_lines)))
                md.append("")
            elif md and md[-1] != "":
                md.append("")  # Ensure blank line after section
            continue

        # Doctor signature block
        if t.upper().startswith("DR.") and is_mostly_upper(t):
            if not found_first_doctor:
                md.append("---")
                md.append("")
                found_first_doctor = True

            md.append(t)
            i += 1
            # Next line: title
            if i < n:
                title_t = line_texts[i]
                if title_t and is_mostly_upper(title_t):
                    md.append(title_t)
                    md.append("")
                    i += 1
                else:
                    md.append("")
            continue

        # Footer: disclaimer
        if "form used by" in t.lower() or "electronic medical" in t.lower():
            md.append("---")
            md.append("")
            md.append(f"*{clean_ocr_text(t)}*")
            md.append("")
            i += 1
            continue

        # Footer: page number
        if re.match(r'^\d+\s*/\s*\d+$', t):
            md.append(t)
            md.append("")
            i += 1
            continue

        # Footer: printed by
        if "printed by" in t.lower():
            md.append(t)
            md.append("")
            i += 1
            continue

        # Skip unrecognized
        i += 1

    return "\n".join(md).strip() + "\n"


# ---------------------------------------------------------------------------
# Regional OCR gap-filling
# ---------------------------------------------------------------------------

def _fill_ocr_gaps(words: list[dict], gray_img: np.ndarray,
                   img_height: int, img_width: int,
                   min_gap: int = 18, margin: int = 5) -> list[dict]:
    """Find vertical gaps in OCR coverage and re-scan those regions.

    Full-page Tesseract (PSM 3) sometimes misses lines that a targeted
    regional scan (PSM 6) on a cropped strip can recover. This function:
    1. Identifies vertical strips where no words were detected
    2. Crops each gap region from the image
    3. Runs Tesseract PSM 6 on the crop
    4. Merges recovered words back into the word list
    """
    if not words:
        return words

    # Sort words by y-position
    sorted_by_y = sorted(words, key=lambda w: w["y0"])

    # Find gaps between consecutive word y-ranges
    # A "gap" is a vertical strip with no word coverage > min_gap pixels
    covered_intervals = [(w["y0"], w["y1"]) for w in sorted_by_y]
    # Merge overlapping intervals
    merged_intervals = [covered_intervals[0]]
    for start, end in covered_intervals[1:]:
        if start <= merged_intervals[-1][1] + margin:
            merged_intervals[-1] = (merged_intervals[-1][0],
                                    max(merged_intervals[-1][1], end))
        else:
            merged_intervals.append((start, end))

    # Find gaps
    gaps = []
    for idx in range(len(merged_intervals) - 1):
        gap_start = merged_intervals[idx][1]
        gap_end = merged_intervals[idx + 1][0]
        if gap_end - gap_start > min_gap:
            gaps.append((gap_start, gap_end))

    # Also check gap between last word and bottom of image (for footer)
    if merged_intervals:
        last_end = merged_intervals[-1][1]
        if img_height - last_end > min_gap:
            gaps.append((last_end, min(last_end + 200, img_height)))

    if not gaps:
        return words

    print(f"  Found {len(gaps)} gap(s) to re-scan: {[(g[0], g[1]) for g in gaps]}")

    recovered_words = list(words)
    for gap_start, gap_end in gaps:
        # Add padding
        y0 = max(0, gap_start - margin)
        y1 = min(img_height, gap_end + margin)
        x0 = 80  # Left margin
        x1 = img_width - 80  # Right margin

        crop = gray_img[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        # Save temp crop and run Tesseract PSM 6
        crop_path = os.path.join(DATA_DIR, f"_tmp_gap_{gap_start}.png")
        cv2.imwrite(crop_path, crop)

        try:
            result = subprocess.run(
                ["tesseract", crop_path, "stdout", "hocr", "--psm", "6"],
                capture_output=True, text=True, errors="replace",
            )
            gap_words = parse_hocr(result.stdout)

            # Adjust coordinates back to full-image space
            for w in gap_words:
                w["x0"] += x0
                w["x1"] += x0
                w["y0"] += y0
                w["y1"] += y0

            if gap_words:
                print(f"    Gap y={gap_start}-{gap_end}: recovered {len(gap_words)} words")
                recovered_words.extend(gap_words)
        finally:
            if os.path.exists(crop_path):
                os.remove(crop_path)

    return recovered_words


def _rescan_impression(words: list[dict], gray_img: np.ndarray,
                       img_height: int, img_width: int) -> tuple[list[dict], list[str] | None]:
    """Re-OCR the impression region to recover missed numbered items.

    Full-page Tesseract often misses lines in dense numbered-list areas.
    This finds the IMPRESSION section, crops it, and re-scans with PSM 6.

    Returns (words, impression_lines) where impression_lines is the
    re-scanned text for the impression section (or None if not found).
    """
    # Find the "IMPRESSION" word to locate the section
    imp_y = None
    for w in words:
        if w["text"].upper().startswith("IMPRESSION"):
            imp_y = w["y0"]
            break
    if imp_y is None:
        return words, None

    # Find the next section boundary (DR. line or large gap)
    sorted_words = sorted(words, key=lambda w: w["y0"])
    imp_end = imp_y + 120  # default: 120px after IMPRESSION
    for w in sorted_words:
        if w["y0"] > imp_y + 20:
            if w["text"].upper().startswith("DR."):
                imp_end = w["y0"]
                break

    # Crop the impression region
    y0 = imp_y + 10  # skip the "IMPRESSION:" header
    y1 = min(imp_end, img_height)
    x0, x1 = 80, img_width - 80

    crop = gray_img[y0:y1, x0:x1]
    if crop.size == 0:
        return words, None

    # Run plain-text OCR (PSM 6) — don't merge words, use text directly
    crop_path = os.path.join(DATA_DIR, "_tmp_impression.png")
    cv2.imwrite(crop_path, crop)

    try:
        result = subprocess.run(
            ["tesseract", crop_path, "stdout", "--psm", "6"],
            capture_output=True, text=True, errors="replace",
        )
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if lines:
            print(f"  Impression re-scan: recovered {len(lines)} text lines")
            return words, lines
        return words, None
    finally:
        if os.path.exists(crop_path):
            os.remove(crop_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def convert_image(image_path: str) -> str:
    """Full pipeline: image -> HOCR -> bold detection -> markdown."""

    print(f"Processing: {image_path}")

    # Step 0: Preprocess (upscale 2x for better OCR on small text/digits)
    print("  Preprocessing image (2x upscale)...")
    preprocessed_path = preprocess_image(image_path)

    # Step 1: Dual OCR — run on both original and 2x, then merge
    # The 2x version captures dates/digits better; original captures more lines overall
    print("  Running Tesseract HOCR (original)...")
    hocr_orig = run_tesseract_hocr(image_path)
    words_orig = parse_hocr(hocr_orig)
    print(f"  Original: {len(words_orig)} words")

    print("  Running Tesseract HOCR (2x upscaled)...")
    hocr_2x = run_tesseract_hocr(preprocessed_path)
    words_2x = parse_hocr(hocr_2x)
    # Scale 2x coordinates back to original space
    for w in words_2x:
        w["x0"] //= UPSCALE_FACTOR
        w["y0"] //= UPSCALE_FACTOR
        w["x1"] //= UPSCALE_FACTOR
        w["y1"] //= UPSCALE_FACTOR
    print(f"  Upscaled: {len(words_2x)} words (scaled back)")

    # Merge: use 2x words for the header region (top 25% of image),
    # original words for the rest
    img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_height = img_orig.shape[0]
    img_width = img_orig.shape[1]
    header_cutoff = int(img_height * 0.25)  # Top 25% uses 2x OCR

    merged_words = []
    for w in words_2x:
        if w["y0"] < header_cutoff:
            merged_words.append(w)
    for w in words_orig:
        if w["y0"] >= header_cutoff:
            merged_words.append(w)
    print(f"  Merged: {len(merged_words)} words (2x header + original body)")

    # Step 1b: Regional OCR fallback — re-scan vertical gaps where full-page
    # layout analysis missed content (Tesseract PSM 3 loses lines that PSM 6
    # on a cropped region can recover)
    print("  Scanning for vertical gaps in OCR coverage...")
    merged_words = _fill_ocr_gaps(merged_words, img_orig, img_height, img_width)

    # Step 1c: Targeted impression re-scan — the impression section is
    # particularly prone to missed lines because numbered items are dense.
    # Re-OCR the impression block with PSM 6 to recover missing items.
    merged_words, impression_lines = _rescan_impression(
        merged_words, img_orig, img_height, img_width)

    # Group into lines
    lines = group_words_into_lines(merged_words)
    print(f"  Grouped into {len(lines)} lines")

    # Step 2: Bold detection via skeletonization (on original image)
    print("  Detecting bold text via skeletonization...")
    img = img_orig
    bold_results = detect_bold_lines(img, lines)

    # Reconstruct line texts and clean
    line_texts = [reconstruct_line_text(l) for l in lines]
    bold_flags = [r[0] for r in bold_results]

    # Print bold detection diagnostics
    print("\n  Bold detection results:")
    for idx, (line_words, (is_bold, ratios)) in enumerate(zip(lines, bold_results)):
        text = " ".join(w["text"] for w in line_words)
        marker = "BOLD" if is_bold else "    "
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        if is_bold or idx < 10:
            print(f"    [{marker}] L{idx:02d} (avg_sw={avg_ratio:.1f}): {text[:80]}")
    print()

    # Step 3: Clean line texts and compute y-positions
    cleaned_texts = [lt.strip() for lt in line_texts]
    line_y_positions = [
        sum(w["y0"] for w in line_words) / len(line_words)
        for line_words in lines
    ]

    # Step 4: Convert to markdown
    print("  Applying structural heuristics...")
    md = lines_to_markdown(cleaned_texts, bold_flags, line_y_positions,
                           impression_override=impression_lines)

    # Cleanup temp file
    if os.path.exists(preprocessed_path) and preprocessed_path != image_path:
        os.remove(preprocessed_path)

    return md


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    image_path = os.path.join(DATA_DIR, IMAGE_FILE)
    md_name = IMAGE_FILE.replace(".jpeg", ".md").replace(".jpg", ".md")
    out_path = os.path.join(OUT_DIR, md_name)
    truth_path = os.path.join(TRUTH_DIR, md_name)

    # Convert
    md_output = convert_image(image_path)

    # Write output
    with open(out_path, "w") as f:
        f.write(md_output)
    print(f"Wrote: {out_path}")

    # Compare with ground truth
    if os.path.exists(truth_path):
        with open(truth_path) as f:
            truth = f.read()

        lev_dist = levenshtein(md_output, truth)
        sim = similarity_ratio(md_output, truth)
        prog_tables = count_tables(md_output)
        truth_tables = count_tables(truth)

        print()
        print("=" * 70)
        print("COMPARISON VS GROUND TRUTH")
        print("=" * 70)
        print(f"  Levenshtein distance: {lev_dist}")
        print(f"  Similarity ratio:     {sim:.1%}")
        print(f"  Tables (programmatic): {prog_tables}")
        print(f"  Tables (ground truth): {truth_tables}")
        print("=" * 70)
    else:
        print(f"  No ground truth found at {truth_path}")


if __name__ == "__main__":
    main()
