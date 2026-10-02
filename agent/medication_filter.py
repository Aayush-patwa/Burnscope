"""
Keeps medication names out of patient-facing text.

Used in two places:
  * tools/rag_retrieve.py  -- drug terms are redacted from patient_safe passages
                              before the model sees them.
  * agent/loop.py          -- every chat reply is checked (drugs + dressing
                              products + over-the-counter wording + doses).

Generic wording ("pain relief", "medication", "a non-stick dressing") is
allowed: the system prompt's own safe phrasing uses it. Dressing-change
frequency ("change the dressing daily") is not treated as a medication
schedule.
"""

import re

_DRUG_NAMES = [
    "acetaminophen", "paracetamol", "tylenol", "panadol", "ibuprofen", "advil", "motrin",
    "naproxen", "aleve", "aspirin", "diclofenac", "ketorolac", "tramadol", "codeine",
    "morphine", "fentanyl", "oxycodone", "hydrocodone", "lidocaine", "benzocaine",
    "hydrocortisone", "bacitracin", "neosporin", "polysporin", "mupirocin",
    r"silver\s+sulph?adiazine", "silvadene", "sulph?adiazine", "mafenide",
    r"povidone[\s-]*iodine", "betadine", "chlorhexidine", r"silver\s+nitrate",
    "amoxicillin", "cephalexin", "ciprofloxacin", "gentamicin",
    r"(?-i:SSD)",
]

_DRUG_CLASSES = [
    r"nsaids?", r"non[\s-]*steroidal(?:[\s-]+anti[\s-]*inflammatory)?(?:\s+drugs?)?",
    r"anti[\s-]*inflammatory\s+(?:drugs?|medications?|medicines?|creams?)",
    r"opioids?", r"narcotics?", r"antibiotics?", r"analgesics?", r"corticosteroids?",
    r"antiseptics?", r"antimicrobials?", r"antihistamines?",
]

_PRODUCTS = [
    r"silver[\s-]+(?:impregnated|dressings?|creams?|based|containing|foam)",
    r"ionic\s+silver", r"colloidal\s+silver",
    r"hydrocolloids?", r"hydrogels?", r"paraffin", r"petroleum\s+jelly", "vaseline",
    "jelonet", "adaptic", "mepilex", "mepitel", "allevyn", "polymem", "acticoat", "aquacel",
    "duoderm", "comfeel", "tegaderm", "bactigras", "urgotul", "xeroform",
    r"over[\s-]*the[\s-]*counter", r"otc", r"pain[\s-]*killers?", r"pain[\s-]*relievers?",
    r"aloe(?:\s+vera)?",
]

_DOSES = [
    r"\d+(?:\.\d+)?\s?(?:mg|mcg|µg|ml|iu)",
    r"\d+\s?(?:mg|ml)\s*/\s*(?:kg|day|hr|hour)",
    r"mg\s*/\s*kg",
]


# The model often writes non-breaking / typographic hyphens (U+2010-U+2015, U+2212), so
# every "[\s-]" in the patterns above also has to match those.
_HYPHENS = r"[\s\-‐-―−]"


def _alt(parts):
    return "|".join(p.replace(r"[\s-]", _HYPHENS) for p in parts)


# Drug terms only (what patient_safe passages are redacted of).
DRUG_RE = re.compile(r"\b(?:%s)\b" % _alt(_DRUG_NAMES + _DRUG_CLASSES), re.IGNORECASE)
# Everything a chat reply must not contain.
_FULL_PATTERN = r"\b(?:%s)\b" % _alt(_DRUG_NAMES + _DRUG_CLASSES + _PRODUCTS + _DOSES)
FULL_RE = re.compile(_FULL_PATTERN, re.IGNORECASE)
# A parenthetical example list containing a flagged term, e.g. "(such as acetaminophen or ibuprofen)".
_PAREN_RE = re.compile(r"\s*\([^()]*?(?:%s)[^()]*?\)" % _FULL_PATTERN, re.IGNORECASE)

RETRIEVAL_PLACEHOLDER = "[medication]"
CHAT_PLACEHOLDER = "[medication - ask a clinician]"
_CHAT_PLACEHOLDER_RUN = re.compile(
    r"(?:%s)(?:\s*(?:,|/|or|and|&)?\s*(?:%s))+" % (re.escape(CHAT_PLACEHOLDER), re.escape(CHAT_PLACEHOLDER))
)


def find_drug_terms(text: str) -> list:
    return sorted({m.group(0).lower() for m in DRUG_RE.finditer(text or "")})


def find_medication_terms(text: str) -> list:
    """Every drug, drug class, dressing product/brand, OTC wording or dose in `text`."""
    return sorted({m.group(0).lower() for m in FULL_RE.finditer(text or "")})


def redact_drug_terms(text: str) -> str:
    """Replace drug names/classes with a neutral placeholder (retrieval path)."""
    return DRUG_RE.sub(RETRIEVAL_PLACEHOLDER, text or "")


def redact_medication_terms(text: str) -> str:
    """
    Deterministic last-resort cleanup of a chat reply. Example lists in
    parentheses are removed; other mentions become a visible placeholder
    so a reader can tell something was withheld.
    """
    cleaned = _PAREN_RE.sub("", text or "")
    cleaned = FULL_RE.sub(CHAT_PLACEHOLDER, cleaned)
    return _CHAT_PLACEHOLDER_RUN.sub(CHAT_PLACEHOLDER, cleaned)
