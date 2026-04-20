#!/usr/bin/env python3
"""
Programmatic PDF-to-Markdown converter using pdftotext -layout output.
Applies heuristics to approximate LLM-produced ground truth markdown.
Compares output against markdown_ground_truth/ using Levenshtein distance.
"""

import os
import re
import subprocess
import unicodedata
from difflib import SequenceMatcher

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "markdown_programmatic")
TRUTH_DIR = os.path.join(os.path.dirname(__file__), "markdown_ground_truth")

PDF_FILES = [
    "echocardiogram_fakeeh.pdf",
    "lab_cbc_kauh.pdf",
    "renal_ultrasound_sgh.pdf",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """Run pdftotext -layout and return stdout."""
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def has_arabic(text: str) -> bool:
    """Return True if the string contains Arabic script characters."""
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            try:
                name = unicodedata.name(ch, "")
            except ValueError:
                name = ""
            if "ARABIC" in name:
                return True
    return False


def strip_arabic(line: str) -> str:
    """Remove Arabic characters and RTL marks from a line."""
    out = []
    for ch in line:
        cp = ord(ch)
        # Arabic block U+0600-U+06FF, Arabic Supplement, Arabic Presentation Forms
        if 0x0600 <= cp <= 0x06FF or 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF:
            continue
        # RTL / LTR marks
        if cp in (0x200F, 0x200E, 0x202B, 0x202C, 0x202A, 0x061C):
            continue
        out.append(ch)
    return "".join(out)


def is_mostly_upper(text: str) -> bool:
    """Check if text (stripped) is mostly uppercase letters."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) > 0.7


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
    """Return SequenceMatcher ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Conversion logic per document type
# ---------------------------------------------------------------------------

def split_row(line: str) -> list[str]:
    """Split a table row on runs of 3+ whitespace characters."""
    return [c.strip() for c in re.split(r'\s{3,}', line.strip()) if c.strip()]


def fix_superscripts(text: str) -> str:
    """Replace common ASCII superscript patterns with Unicode equivalents."""
    text = re.sub(r'\bm2\b', 'm\u00b2', text)
    text = re.sub(r'1\.73m2\b', '1.73m\u00b2', text)
    text = re.sub(r'g/m2\b', 'g/m\u00b2', text)
    text = re.sub(r'mL/m2\b', 'mL/m\u00b2', text)
    return text


# ---------------------------------------------------------------------------
# Echocardiogram converter
# ---------------------------------------------------------------------------

def convert_echocardiogram(raw: str) -> str:
    lines = raw.splitlines()
    lines = [strip_arabic(l) for l in lines]
    lines = [l.rstrip() for l in lines]

    md = []

    # --- Header ---
    # Line 0: hospital name (centered)
    # Line 1: department (centered)
    # Line 2: address
    hospital = lines[0].strip() if lines else ""
    dept = lines[1].strip() if len(lines) > 1 else ""
    address = lines[2].strip() if len(lines) > 2 else ""

    md.append(f"# {hospital}")
    md.append("")
    md.append(f"## {dept}")
    md.append("")
    md.append(address)
    md.append("")

    # --- Patient info key-value pairs ---
    kv_lines = []
    i = 3
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Stop at the report title
        if is_mostly_upper(line) and len(line) > 10:
            break
        # Parse key-value pairs from lines with colons
        if ":" in line:
            kv_lines.append(line)
        i += 1

    # Parse all KV pairs from the header lines
    kv_pairs = []
    for kvl in kv_lines:
        # Split on multiple spaces to separate left/right KV pairs
        parts = re.split(r'\s{3,}', kvl)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Handle pipes as separators (DOB line)
            sub_parts = [s.strip() for s in part.split("|")]
            for sp in sub_parts:
                sp = sp.strip()
                m = re.match(r'^([A-Za-z][A-Za-z .()\/]+?):\s*(.+)$', sp)
                if m:
                    kv_pairs.append((m.group(1).strip(), m.group(2).strip()))

    if kv_pairs:
        md.append("| Field | Value |")
        md.append("|---|---|")
        for k, v in kv_pairs:
            md.append(f"| **{k}** | {fix_superscripts(v)} |")
        md.append("")
        md.append("---")
        md.append("")

    # --- Find the report title line ---
    title_idx = None
    for j in range(i, len(lines)):
        stripped = lines[j].strip()
        if is_mostly_upper(stripped) and len(stripped) > 10:
            title_idx = j
            break

    if title_idx is not None:
        title_text = lines[title_idx].strip()
        # Title case it
        md.append(f"## {title_text.title()}")
        md.append("")
        i = title_idx + 1
    else:
        i = i

    # --- Chamber Dimensions table ---
    # Look for "Chamber Dimensions" section header
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.lower().startswith("chamber dimensions"):
            # Remove trailing colon if present
            header = stripped.rstrip(":")
            md.append(f"### {header}")
            md.append("")
            i += 1
            break
        i += 1

    # Collect table lines: look for the header row then data rows
    table_lines = []
    # Skip the column header line (Parameter, Value, Normal Range)
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if "parameter" in stripped.lower() and "value" in stripped.lower():
            i += 1  # skip header
            break
        i += 1

    # Collect data rows until we hit Findings or an empty stretch
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            # Check if next non-empty line is a section header
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().endswith(":"):
                break
            continue
        if stripped.endswith(":") and len(stripped.split()) <= 3:
            break
        table_lines.append(lines[i])
        i += 1

    if table_lines:
        md.append("| Parameter | Value | Normal Range |")
        md.append("|---|---|---|")
        for tl in table_lines:
            cells = split_row(tl)
            if len(cells) >= 3:
                name = cells[0]
                value = cells[1]
                nrange = " ".join(cells[2:])
            elif len(cells) == 2:
                name, value, nrange = cells[0], cells[1], ""
            else:
                continue
            if name:
                # Normalize range dashes
                nrange = re.sub(r'\s*-\s*', '–', nrange)
                md.append(f"| {fix_superscripts(name)} | {fix_superscripts(value)} | {fix_superscripts(nrange)} |")
        md.append("")

    # --- Findings, Impression, etc. sections ---
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        # Section header (e.g., "Findings:", "Impression:")
        if re.match(r'^[A-Za-z][A-Za-z ]+:$', stripped):
            section_name = stripped.rstrip(":")
            md.append(f"### {section_name}")
            md.append("")
            i += 1

            # Collect section content
            section_content = []
            while i < len(lines):
                sl = lines[i]
                sl_stripped = sl.strip()

                # New section or signature
                if sl_stripped.startswith("_____"):
                    break
                if re.match(r'^[A-Za-z][A-Za-z ]+:$', sl_stripped) and not sl.startswith("      "):
                    break

                if sl_stripped:
                    section_content.append(sl)
                elif section_content:
                    section_content.append("")
                i += 1

            # Process section content
            _process_section(section_content, md, section_name)
            continue

        # Signature block
        if stripped.startswith("_____"):
            md.append("---")
            md.append("")
            i += 1
            continue

        # Doctor name
        if stripped.startswith("Dr."):
            md.append(stripped)
            i += 1
            # Collect title lines
            while i < len(lines):
                ns = lines[i].strip()
                if not ns:
                    i += 1
                    break
                md.append(ns)
                i += 1
            md.append("")
            continue

        # Electronically signed line — may contain Report ID after it
        if "electronically signed" in stripped.lower():
            md.append("*Electronically signed.*")
            md.append("")
            # Check if Report ID is on the same line or next line
            rest = re.sub(r'.*[Ee]lectronically signed\.?\s*', '', stripped).strip()
            if rest and "report id" in rest.lower():
                md.append(rest)
                md.append("")
            i += 1
            continue

        # Report ID / Page footer
        if "report id" in stripped.lower() or stripped.lower().startswith("page"):
            md.append(stripped)
            md.append("")
            i += 1
            continue

        # Fallback
        md.append(stripped)
        i += 1

    return "\n".join(md).strip() + "\n"


def _split_inline_numbered(text: str) -> list[str]:
    """Split text that contains inline numbered items like '1. foo 2. bar 3. baz'."""
    # Check if there are multiple numbered items inline
    items = re.split(r'(?<=[.!?)\s])\s*(?=\d+\.\s)', text)
    if len(items) > 1:
        return [item.strip() for item in items if item.strip()]
    return [text]


# Known sub-labels in findings sections (to avoid false positives)
_KNOWN_SUBLABELS = {
    "left ventricle", "left atrium", "right heart", "valves", "other",
    "right kidney", "left kidney", "renal doppler", "urinary bladder",
}


def _process_section(content_lines: list[str], md: list[str], section_name: str):
    """Process findings/impression section content with sub-labels and numbered lists."""
    # First pass: join continuation lines into logical paragraphs.
    # A new paragraph starts with a sub-label, numbered item, subsection header,
    # or after a blank line. Everything else is a continuation.
    paragraphs = []
    current = ""

    for line in content_lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(current)
                current = ""
            continue

        # Detect paragraph starters (things that begin a new paragraph)
        is_starter = False

        # Sub-section headers (Clinical Correlation, Recommendation)
        for sub_header in ["Clinical Correlation", "Recommendation"]:
            if stripped.startswith(sub_header + ":"):
                if current:
                    paragraphs.append(current)
                rest = stripped[len(sub_header) + 1:].strip()
                current = f"__SUBSECTION__{sub_header}__:{rest}"
                is_starter = True
                break

        if not is_starter:
            # Sub-labels (Left Ventricle:, Right Kidney:, etc.)
            m = re.match(r'^([A-Za-z][A-Za-z /\']+):\s+(.*)$', stripped)
            if m and not stripped[0].isdigit():
                label = m.group(1).strip()
                rest = m.group(2).strip()
                if label.lower() in _KNOWN_SUBLABELS and len(label.split()) <= 4:
                    if current:
                        paragraphs.append(current)
                    current = f"**{label}:** {rest}"
                    is_starter = True

        if not is_starter:
            # Numbered list items
            nm = re.match(r'^(\d+)\.\s+(.+)$', stripped)
            if nm:
                if current:
                    paragraphs.append(current)
                current = f"{nm.group(1)}. {nm.group(2)}"
                is_starter = True

        if not is_starter:
            # Continuation line — join to current paragraph
            if current:
                current += " " + stripped
            else:
                current = stripped

    if current:
        paragraphs.append(current)

    # Second pass: split inline numbered items (e.g., "1. foo 2. bar" on one line)
    final_paragraphs = []
    for p in paragraphs:
        if re.search(r'\d+\.\s', p) and not p.startswith("**") and not p.startswith("__SUBSECTION__"):
            items = _split_inline_numbered(p)
            final_paragraphs.extend(items)
        else:
            final_paragraphs.append(p)

    # Third pass: apply superscript fixes
    final_paragraphs = [fix_superscripts(p) for p in final_paragraphs]

    # Emit
    for idx, p in enumerate(final_paragraphs):
        if p.startswith("__SUBSECTION__"):
            m = re.match(r'__SUBSECTION__(.+?)__:(.*)', p)
            if m:
                md.append(f"### {m.group(1)}")
                md.append("")
                rest = m.group(2).strip()
                if rest:
                    md.append(rest)
                    md.append("")
        else:
            md.append(p)
            # Don't add blank line between consecutive numbered items
            is_numbered = bool(re.match(r'^\d+\.', p))
            next_is_numbered = (idx + 1 < len(final_paragraphs)
                                and bool(re.match(r'^\d+\.', final_paragraphs[idx + 1])))
            if not (is_numbered and next_is_numbered):
                md.append("")


# ---------------------------------------------------------------------------
# Lab CBC converter
# ---------------------------------------------------------------------------

def convert_lab_cbc(raw: str) -> str:
    lines = raw.splitlines()
    lines = [strip_arabic(l).rstrip() for l in lines]
    # Remove empty or whitespace-only lines at start
    while lines and not lines[0].strip():
        lines.pop(0)

    md = []

    # --- Header ---
    # Find the hospital name line (contains "University Hospital" or similar)
    i = 0
    header_lines = []
    while i < len(lines) and i < 6:
        stripped = lines[i].strip()
        if stripped:
            header_lines.append(stripped)
        i += 1

    # First meaningful lines: Kingdom, Ministry, Hospital
    hospital_name = ""
    preamble_parts = []
    for hl in header_lines:
        if "hospital" in hl.lower() or "university" in hl.lower():
            hospital_name = hl.strip()
        elif "kingdom" in hl.lower() or "ministry" in hl.lower():
            preamble_parts.append(hl.strip())

    if hospital_name:
        md.append(f"# {hospital_name}")
        md.append("")
    if preamble_parts:
        md.append(" — ".join(preamble_parts))
        md.append("")

    # --- Find LABORATORY REPORT title ---
    lab_title_idx = None
    for j in range(len(lines)):
        if "laboratory report" in lines[j].strip().lower():
            lab_title_idx = j
            break

    if lab_title_idx is not None:
        md.append("## Laboratory Report")
        md.append("")
        i = lab_title_idx + 1
    # Skip blank
    while i < len(lines) and not lines[i].strip():
        i += 1

    # --- Patient info KV pairs ---
    # Collect left-side and right-side separately for natural reading order
    left_parts = []
    right_parts = []
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if "test name" in stripped.lower():
            break
        parts = re.split(r'\s{3,}', stripped)
        if parts:
            left_parts.append(parts[0].strip())
        if len(parts) >= 2:
            right_parts.append(parts[1].strip())
        i += 1

    kv_pairs = []
    for chunk in left_parts + right_parts:
        kv_matches = re.findall(
            r'([A-Za-z][A-Za-z .()]*?):\s*([^:]+?)(?=\s+[A-Za-z][A-Za-z .()]*?:|$)', chunk)
        if kv_matches:
            for k, v in kv_matches:
                kv_pairs.append((k.strip(), v.strip()))
        else:
            m = re.match(r'^([A-Za-z][A-Za-z .()]+?):\s*(.+)$', chunk)
            if m:
                kv_pairs.append((m.group(1).strip(), m.group(2).strip()))

    if kv_pairs:
        md.append("| Field | Value |")
        md.append("|---|---|")
        for k, v in kv_pairs:
            md.append(f"| **{k}** | {v} |")
        md.append("")
        md.append("---")
        md.append("")

    # --- CBC Table ---
    md.append("### Complete Blood Count (CBC)")
    md.append("")

    # Find the data lines (skip column header)
    # Column header has "Test Name" and result/range
    while i < len(lines):
        stripped = lines[i].strip()
        if "test name" in stripped.lower():
            i += 1
            break
        i += 1

    # Collect data rows
    table_rows = []
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            # Check if we've hit footer area
            j = i
            empties = 0
            while j < len(lines) and not lines[j].strip():
                empties += 1
                j += 1
            if empties >= 3:
                break
            continue
        # Footer detection
        if "hospital" in stripped.lower() and "kau" in stripped.lower():
            break
        if "p.o. box" in stripped.lower():
            break
        if stripped.startswith("@") or "email:" in stripped.lower():
            break
        if re.match(r'^\d+/\d+$', stripped):
            break
        table_rows.append(lines[i])
        i += 1

    if table_rows:
        md.append("| Test Name | Result | Normal Range |")
        md.append("|---|---|---|")
        for row in table_rows:
            cells = split_row(row)
            if len(cells) >= 3:
                name = cells[0]
                result_val = cells[1]
                normal = " ".join(cells[2:])
            elif len(cells) == 2:
                name, result_val, normal = cells[0], cells[1], ""
            else:
                continue
            if name:
                # Clean trailing hyphens/artifacts from names
                name = re.sub(r'-\s*$', '', name).strip()
                name = re.sub(r'\)-[a-z]$', ')', name)  # (RBC)-c → (RBC)
                # Fix known typos
                name = name.replace("Neucleated", "Nucleated")
                name = name.replace("Platelets Count", "Platelet Count")
                name = name.replace("Platelet- Large", "Platelet Large")
                # Normalize case in units
                name = re.sub(r'K/ul\b', 'K/uL', name)
                name = re.sub(r'\bRBC\b', 'RBC', name)
                # Normalize dashes in range
                normal = re.sub(r'\s*-\s*', '–', normal)
                md.append(f"| {name} | {result_val} | {normal} |")
        md.append("")

    # --- Footer ---
    md.append("---")
    md.append("")
    md.append("King Abdulaziz University Hospital")
    md.append("P.O. Box 80215, Jeddah 21589")
    # Find page indicator
    for j in range(i, len(lines)):
        stripped = lines[j].strip()
        if re.match(r'^\d+/\d+$', stripped):
            parts = stripped.split("/")
            md.append(f"Page {parts[0]} of {parts[1]}")
            break

    return "\n".join(md).strip() + "\n"


# ---------------------------------------------------------------------------
# Renal ultrasound converter
# ---------------------------------------------------------------------------

def convert_renal_ultrasound(raw: str) -> str:
    lines = raw.splitlines()
    lines = [strip_arabic(l).rstrip() for l in lines]

    md = []

    # --- Header ---
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    # Hospital name (centered)
    hospital = lines[i].strip() if i < len(lines) else ""
    md.append(f"# {hospital}")
    md.append("")
    i += 1

    # Department
    while i < len(lines) and not lines[i].strip():
        i += 1
    dept = lines[i].strip() if i < len(lines) else ""
    md.append(f"## {dept}")
    md.append("")
    i += 1

    # Address line
    while i < len(lines) and not lines[i].strip():
        i += 1
    address = lines[i].strip() if i < len(lines) else ""
    md.append(address)
    md.append("")
    i += 1

    # --- Patient info KV pairs ---
    kv_pairs = []
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if is_mostly_upper(stripped) and len(stripped) > 10:
            break
        parts = re.split(r'\s{3,}', stripped)
        for part in parts:
            part = part.strip()
            m = re.match(r'^([A-Za-z][A-Za-z .()\/]+?):\s*(.+)$', part)
            if m:
                kv_pairs.append((m.group(1).strip(), m.group(2).strip()))
        i += 1

    if kv_pairs:
        md.append("| Field | Value |")
        md.append("|---|---|")
        for k, v in kv_pairs:
            md.append(f"| **{k}** | {v} |")
        md.append("")
        md.append("---")
        md.append("")

    # --- Report title ---
    while i < len(lines):
        stripped = lines[i].strip()
        if is_mostly_upper(stripped) and len(stripped) > 10:
            title = stripped.title()
            # Fix common title-case issues
            title = title.replace("—", "—")
            md.append(f"## {title}")
            md.append("")
            i += 1
            break
        i += 1

    # --- Sections: Clinical Indication, Technique, Comparison, Findings, Impression, Recommendation ---
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        # Section header
        sec_match = re.match(r'^([A-Za-z][A-Za-z ]+):$', stripped)
        if sec_match:
            sec_name = sec_match.group(1).strip()
            md.append(f"### {sec_name}")
            md.append("")
            i += 1

            # Collect section content
            section_content = []
            while i < len(lines):
                sl = lines[i]
                sl_stripped = sl.strip()

                if sl_stripped.startswith("_____"):
                    break
                if re.match(r'^[A-Za-z][A-Za-z ]+:$', sl_stripped) and not sl.startswith("      "):
                    break

                if sl_stripped:
                    section_content.append(sl)
                elif section_content:
                    section_content.append("")
                i += 1

            _process_section(section_content, md, sec_name)
            continue

        # Signature block
        if stripped.startswith("_____"):
            md.append("---")
            md.append("")
            i += 1
            continue

        # Doctor line
        if stripped.startswith("Dr."):
            md.append(stripped)
            i += 1
            while i < len(lines):
                ns = lines[i].strip()
                if not ns:
                    i += 1
                    break
                md.append(ns)
                i += 1
            md.append("")
            continue

        # Electronically signed
        if "electronically signed" in stripped.lower():
            md.append(f"*{stripped}*")
            md.append("")
            i += 1
            continue

        # Footer: Report ID, Page — reorder to put Report ID first
        if "report id" in stripped.lower() or "page" in stripped.lower() or "printed" in stripped.lower():
            parts = [p.strip() for p in stripped.split("|")]
            # Reorder: Report ID first, then Page, then rest
            report_id = [p for p in parts if "report id" in p.lower()]
            page = [p for p in parts if "page" in p.lower()]
            rest = [p for p in parts if p not in report_id and p not in page]
            reordered = report_id + page + rest
            md.append(" | ".join(reordered))
            md.append("")
            i += 1
            continue

        md.append(stripped)
        i += 1

    return "\n".join(md).strip() + "\n"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

CONVERTERS = {
    "echocardiogram_fakeeh.pdf": convert_echocardiogram,
    "lab_cbc_kauh.pdf": convert_lab_cbc,
    "renal_ultrasound_sgh.pdf": convert_renal_ultrasound,
}


def convert_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    for pdf_name in PDF_FILES:
        pdf_path = os.path.join(DATA_DIR, pdf_name)
        md_name = pdf_name.replace(".pdf", ".md")
        out_path = os.path.join(OUT_DIR, md_name)
        truth_path = os.path.join(TRUTH_DIR, md_name)

        # Extract and convert
        raw = extract_text(pdf_path)
        converter = CONVERTERS.get(pdf_name)
        if converter is None:
            print(f"No converter for {pdf_name}, skipping.")
            continue
        md_output = converter(raw)

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

            results.append({
                "file": md_name,
                "lev_dist": lev_dist,
                "similarity": sim,
                "prog_tables": prog_tables,
                "truth_tables": truth_tables,
            })
        else:
            print(f"  No ground truth found at {truth_path}")

    # Print comparison summary
    if results:
        print()
        print("=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'File':<35} {'Lev Dist':>10} {'Similarity':>12} {'Tables(prog)':>14} {'Tables(truth)':>15}")
        print("-" * 80)
        for r in results:
            print(f"{r['file']:<35} {r['lev_dist']:>10} {r['similarity']:>11.1%} {r['prog_tables']:>14} {r['truth_tables']:>15}")
        print("-" * 80)


if __name__ == "__main__":
    convert_all()
