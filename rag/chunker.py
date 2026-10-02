"""
Parses real .docx source documents directly (via python-docx), reading
the actual document XML structure -- NOT a markdown/pandoc conversion.

DOCUMENT CONVENTIONS SUPPORTED (the four real source documents use a
mix of these -- this parser handles all of them):

Section boundaries:
  - A paragraph starting with "Section:" (any style) starts a new
    section -- resets source, audience, and chunk buffer.

Sub-chunk boundaries within a section:
  - A fully bold paragraph, OR a paragraph starting with a numbered/
    lettered/roman-numeral marker ("(i)", "(a)", "1.", "1st ..."), OR a
    short colon-ending phrase starting with a capital letter (e.g.
    "Skin substitutes:", "Management:") starts a new chunk. This third
    fallback exists because 3 of 4 real source documents don't apply
    bold formatting to their sub-headings consistently -- relying on
    bold alone silently absorbed unbolded headings into whatever
    content was being buffered.
  - A size cap also splits very long runs of body text.

Citations:
  - "[SOURCE: Document — Chapter/Section — pages]" (brackets optional,
    case-insensitive) is captured as that block's source metadata, not
    included in retrievable text. Citations may appear BEFORE or AFTER
    the content they describe (both patterns exist in the real
    documents) -- handled by a two-directional backfill pass.

Audience tagging -- TWO styles, both supported:
  1. STANDALONE marker paragraph: a paragraph whose ENTIRE text (after
     stripping the tag) is empty -- e.g. a paragraph that says only
     "[AUDIENCE: clinician_only]". This is a MODE SWITCH: every
     following paragraph AND table is tagged with that audience until
     the next Section heading (reset to patient_safe) or another such
     marker changes it again. IMPORTANT: this persists ACROSS
     sub-headings within the same section (verified necessary -- the
     Rehabilitation document's nutrition-restriction block spans
     multiple sub-headings and must stay restricted throughout). If a
     restricted block should end BEFORE the next Section heading, the
     source document must include an explicit
     "[AUDIENCE: patient_safe]" reset marker at that point -- the code
     cannot safely infer where such a block should end from formatting
     alone without breaking the verified-correct multi-subheading case.
  2. INLINE tag at the end of a paragraph that also has real content --
     e.g. "...until urine is clear of pigments. [AUDIENCE:
     clinician_only]". This tags THAT PARAGRAPH ONLY and does not
     change the audience of surrounding content.

Tables:
  - Real tables often have a MERGED TITLE ROW -- every cell in row 0
    holds the same text (e.g. "Dressing type: Hydrocolloid" or
    "Nutrition: Total Calorie Requirement"). This is extracted as the
    chunk's title and kept together with the table's body in ONE
    chunk -- no external merging with a preceding paragraph needed.
  - The body rows are formatted as "label: value" lines when they are
    a clean 2-column profile card (known field names like Function/
    Indications/Application/Notes), or as a generic "|"-joined line
    per row for messier multi-column data tables (e.g. calorie
    formulas with no consistent header).

Chunk boundary marker:
  - A paragraph that is only "[CHUNK]" forces a chunk boundary at that
    point (used to separate topics that share one long paragraph). It adds
    no words to the knowledge base.

Metadata added to every chunk:
  - section: the title of the enclosing "Section:" heading (kept for
    retrieval context and for grouping in reports).
  - topic: a coarse topic tag derived from the section title (see
    TOPIC_RULES), used to group guidance in generated reports.

Defence-in-depth safety net:
  - ANY chunk whose text matches DOSAGE_PATTERNS (numeric drug/fluid
    doses, or a named formula referencing body weight) is force-tagged
    clinician_only regardless of what the document itself says. A
    missing or misplaced tag in the source document should never be
    the single point of failure for keeping dosage content out of
    patient-facing retrieval.
"""

import re
from docx import Document

SECTION_HEADING_RE = re.compile(r"^\s*section\s*:\s*(.+)$", re.IGNORECASE)
SOURCE_LINE_RE = re.compile(r"^\s*\[?\s*source\s*:\s*(.+?)\s*\]?\s*$", re.IGNORECASE)
AUDIENCE_TAG_RE = re.compile(r"\[?\\?\[?\s*AUDIENCE\s*:\s*(\w+)\s*\]?\\?\]?", re.IGNORECASE)
SUBHEADING_RE = re.compile(
    r"^\s*(\(\s*[ivxlcdm]+\s*\)|\(\s*[a-z]\s*\)|\d+\s*[\.\)]\s|\d+(st|nd|rd|th)\s)",
    re.IGNORECASE,
)

DOSAGE_PATTERNS = re.compile(
    r"\b\d+\s*(mg|gm|g|ml|mcg|iu|units?)\b"
    r"|\bmg\s*/\s*(kg|day|hr)\b"
    r"|\bml\s*/\s*(kg|day|hr)\b"
    r"|\bcalorie requirement\b"
    r"|\bformula\b.*\bweight in kg\b",
    re.IGNORECASE,
)

CHUNK_BREAK_RE = re.compile(r"^\s*\[\s*chunk\s*\]\s*$", re.IGNORECASE)
NUMBERED_ITEM_RE = re.compile(r"^\s*\d+\s*[\.\)]\s")
_LEADING_GLYPH_RE = re.compile(r"(?m)^\s*[\u00b7\u2022\u25aa\u27a2\u2756\u25cf\u25e6]\s*")
_INLINE_GLYPH_RE = re.compile(r"\s[\u00b7\u2022\u25aa\u27a2\u2756\u25cf\u25e6]\s")

# (pattern on the lower-cased section title, topic). First match wins.
TOPIC_RULES = [
    (r"infection|toxic shock|warning sign", "warning_signs"),
    (r"child|paediatric|pediatric", "children"),
    (r"cooling|first aid|chemical|electrical|transport|tetanus", "first_aid"),
    (r"oral versus|fluid|nutrition", "nutrition_fluids"),
    (r"rehabilit|scar", "rehabilitation"),
    (r"referral|depth|triage|special area|inhalation|healing", "severity_triage"),
    (r"dressing|wound|escharotomy|skin substitute|biological", "wound_care"),
]

TOPIC_LABELS = {
    "first_aid": "First aid and emergency care",
    "children": "Burns in children",
    "warning_signs": "Warning signs and infection",
    "severity_triage": "Severity and triage",
    "wound_care": "Wound care and dressings",
    "rehabilitation": "Rehabilitation and scars",
    "nutrition_fluids": "Nutrition and fluids",
    "general": "General",
}


def _topic_for(section: str) -> str:
    title = (section or "").lower()
    for pattern, topic in TOPIC_RULES:
        if re.search(pattern, title):
            return topic
    return "general"


def _merge_sources(a: str, b: str) -> str:
    """Cite both sources of two merged fragments; the same source (text + its table) is cited once."""
    base = lambda x: x.replace(" (table)", "")
    if base(a) == base(b):
        return base(a) + (" (table)" if a.endswith("(table)") or b.endswith("(table)") else "")
    return a + "; " + b


def _normalize_glyphs(text: str) -> str:
    """Bullet glyphs copied from PDFs (dot, bullet, square, arrow, diamond...) become plain '- '."""
    text = _LEADING_GLYPH_RE.sub("- ", text)
    return _INLINE_GLYPH_RE.sub(" - ", text)


PROFILE_FIELD_NAMES = {
    "function", "indications", "application", "notes and precautions",
    "notes", "precautions", "examples",
}

MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 120


def _clean_text(text: str) -> str:
    return AUDIENCE_TAG_RE.sub("", text).strip()


def _looks_like_dosage(text: str) -> bool:
    return bool(DOSAGE_PATTERNS.search(text))


def _is_bold_paragraph(para) -> bool:
    runs = [r for r in para.runs if r.text.strip()]
    return bool(runs) and all(r.bold for r in runs)


def _looks_like_text_heading(text: str) -> bool:
    """
    Fallback heading detector for documents that don't apply bold
    formatting consistently (verified: 3 of 4 real source documents
    don't). A short, colon-ending phrase starting with a capital
    letter is treated as a sub-heading boundary regardless of
    formatting -- e.g. "Skin substitutes:", "Management:",
    "Micronutrient supplementation:". A word-count cap (rather than
    strict Title Case, which the real headings don't consistently use
    either -- "Skin substitutes:" only capitalizes the first word)
    distinguishes short heading phrases from long descriptive
    sentences that happen to end in a colon before a list.
    """
    if not text.endswith(":"):
        return False
    inner = text[:-1].strip()
    if not inner or not inner[0].isupper():
        return False
    return len(inner.split()) <= 8


def _extract_table_title(table):
    """
    Returns the title string if row 0's cells all hold the same
    non-empty text (a merged title cell), else None.
    """
    if not table.rows:
        return None
    cells = [c.text.strip() for c in table.rows[0].cells]
    if cells and all(c == cells[0] for c in cells) and cells[0]:
        return cells[0]
    return None


def _table_body_to_text(table, skip_first_row: bool) -> str:
    """
    Formats a table's body rows (excluding the title row, if present)
    as readable text -- "label: value" lines for a clean 2-column
    profile card, or a generic "|"-joined line per row otherwise.
    """
    rows = list(table.rows)[1:] if skip_first_row else list(table.rows)
    if not rows:
        return ""

    is_profile = len(table.columns) == 2 and any(
        r.cells[0].text.strip().lower() in PROFILE_FIELD_NAMES for r in rows
    )

    lines = []
    if is_profile:
        for row in rows:
            label = row.cells[0].text.strip()
            value = row.cells[1].text.strip()
            if label and value:
                lines.append(f"{label}: {value}")
    else:
        for row in rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))

    return "\n".join(lines)


def _table_to_chunk_text(table) -> str:
    """
    Builds one self-contained chunk of text for a whole table: its
    title (if a merged title row exists) followed by its formatted
    body, kept together always.
    """
    title = _extract_table_title(table)
    body = _table_body_to_text(table, skip_first_row=title is not None)
    if title and body:
        return f"{title}\n{body}"
    return title or body or ""


def parse_docx(filepath: str) -> list:
    doc = Document(filepath)
    raw_chunks = []

    current_source = "Unknown source"
    current_audience = "patient_safe"
    current_section = ""
    buffer = []

    def flush():
        nonlocal buffer
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        buffer = []
        if not text:
            return
        audience = "clinician_only" if _looks_like_dosage(text) else current_audience
        raw_chunks.append({"text": text, "audience": audience, "source": current_source,
                           "section": current_section})

    body = doc.element.body
    tables = doc.tables
    table_index = 0
    para_map = {p._p: p for p in doc.paragraphs}

    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]

        if tag == "p":
            para = para_map.get(child)
            text = para.text.strip() if para else ""
            if not text:
                continue

            section_match = SECTION_HEADING_RE.match(text)
            if section_match:
                flush()
                current_source = "Unknown source"
                current_audience = "patient_safe"
                current_section = section_match.group(1).strip().rstrip(":").strip()
                continue

            if CHUNK_BREAK_RE.match(text):
                flush()
                continue

            source_match = SOURCE_LINE_RE.search(text)
            if source_match:
                current_source = source_match.group(1).strip().rstrip("]").strip()
                flush()
                for prior in reversed(raw_chunks):
                    suffix = " (table)" if prior["source"].endswith("(table)") else ""
                    if prior["source"].replace(" (table)", "") != "Unknown source":
                        break
                    prior["source"] = current_source + suffix
                continue

            audience_match = AUDIENCE_TAG_RE.search(text)
            cleaned = _clean_text(text)

            if audience_match and not cleaned:
                # STANDALONE marker paragraph -> mode switch. Flush
                # whatever was buffered under the OLD audience first.
                flush()
                current_audience = audience_match.group(1).lower()
                continue

            if not cleaned:
                continue

            if audience_match:
                # INLINE tag with real content -> tags THIS paragraph
                # only, does not change current_audience going forward.
                flush()
                raw_chunks.append({
                    "text": cleaned,
                    "audience": audience_match.group(1).lower(),
                    "source": current_source,
                    "section": current_section,
                })
                continue

            is_boundary = (
                _is_bold_paragraph(para)
                or SUBHEADING_RE.match(cleaned)
                or _looks_like_text_heading(cleaned)
            )
            if is_boundary and NUMBERED_ITEM_RE.match(cleaned) and buffer and not _is_bold_paragraph(para):
                # "1." "2." "3." items belong to one list: do not cut between items,
                # nor between a list and the line that introduces it ("...are:").
                if NUMBERED_ITEM_RE.match(buffer[-1]) or buffer[-1].rstrip().endswith(":"):
                    is_boundary = False
            if is_boundary:
                flush()

            buffer.append(cleaned)

            if sum(len(b) for b in buffer) > MAX_CHUNK_CHARS:
                flush()

        elif tag == "tbl":
            flush()
            if table_index < len(tables):
                table_text = _table_to_chunk_text(tables[table_index])
                if table_text:
                    audience = "clinician_only" if _looks_like_dosage(table_text) else current_audience
                    raw_chunks.append({
                        "text": table_text,
                        "audience": audience,
                        "source": current_source + " (table)",
                        "section": current_section,
                    })
            table_index += 1

    flush()

    # Backward source backfill: citations that appear AFTER their
    # content (the other direction from the forward backfill above).
    next_known = "Unknown source"
    for chunk in reversed(raw_chunks):
        base = chunk["source"].replace(" (table)", "")
        if base == "Unknown source" and next_known != "Unknown source":
            suffix = " (table)" if chunk["source"].endswith("(table)") else ""
            chunk["source"] = next_known + suffix
        elif base != "Unknown source":
            next_known = base

    # Merge fragments too small to retrieve usefully into the next
    # chunk -- never across the patient_safe / clinician_only boundary.
        # Merge fragments too small to retrieve usefully into the next
    # chunk -- never across the patient_safe / clinician_only boundary.
    merged = []
    for chunk in raw_chunks:
        if (
            merged
            and len(merged[-1]["text"]) < MIN_CHUNK_CHARS
            and merged[-1]["audience"] == chunk["audience"]
            and merged[-1]["source"] == chunk["source"]
        ):
            merged[-1]["text"] += "\n" + chunk["text"]
        elif (
            merged
            and len(merged[-1]["text"]) < MIN_CHUNK_CHARS
            and merged[-1]["audience"] == chunk["audience"]
            and merged[-1]["section"] == chunk["section"]
            and merged[-1]["source"] != chunk["source"]
            and "Unknown source" not in merged[-1]["source"] + chunk["source"]
        ):
            # A tiny fragment whose neighbour cites a different source: merge and cite BOTH.
            merged[-1]["text"] += "\n" + chunk["text"]
            merged[-1]["source"] = _merge_sources(merged[-1]["source"], chunk["source"])
        else:
            merged.append(chunk)

    # Drop any chunk still under an absolute floor after merging --
    # these are orphaned fragments (e.g. a lone table caption like
    # "Table 1: Physical appearance of Burns") that had nothing
    # matching adjacent to merge into. Too short to carry retrievable
    # meaning on their own; better absent than surfaced as noise.
    ABSOLUTE_MIN_CHARS = 40
    final = [c for c in merged if len(c["text"]) >= ABSOLUTE_MIN_CHARS]

    for c in final:
        c["text"] = _normalize_glyphs(c["text"])
        c["topic"] = _topic_for(c["section"])

    return final


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m rag.chunker <path_to.docx>")
    else:
        result = parse_docx(sys.argv[1])
        print(f"TOTAL: {len(result)} chunks "
              f"({sum(1 for c in result if c['audience']=='clinician_only')} clinician_only)\n")
        for i, c in enumerate(result):
            print(f"--- {i} [{c['audience']}] ({c['source']}) ---")
            print(c["text"][:250])
            print()