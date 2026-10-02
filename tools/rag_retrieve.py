"""
Tool: retrieve_treatment_guidance
Real implementation -- queries the local Chroma vector store (see
rag/vector_store.py) built from chunked, audience-tagged, real
source documents (rag/chunker.py + rag/build_index.py).

CRITICAL SAFETY PROPERTY: this chat-facing tool ALWAYS queries with
audience="patient_safe". This is not a prompt instruction the LLM
could ignore -- it is hardcoded here, so dosage/clinician-only content
is structurally unreachable from normal conversation. The only path
that may access clinician_only content is generate_escalation_report's
own internal lookup (tools/escalate.py), gated separately on whether
the report is genuinely being emailed to a professional.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.vector_store import query as vector_query
from agent.medication_filter import redact_drug_terms


# Chroma L2 distances measured on this corpus (all-mpnet-base-v2): burn questions
# land at 0.14-0.55, unrelated questions (sport, cars, recipes) at 0.80 and above.
# Anything past the floor is treated as "the knowledge base has nothing on this".
MAX_DISTANCE = 0.65
# Passages on these topics are only returned when the question is actually about them, so a fresh-burn query does not
# come back with months-long rehabilitation or nutrition text.
TOPIC_KEYWORDS = {
    "rehabilitation": ("rehab", "scar", "exercise", "physio", "contracture", "mobility", "splint", "stretch", "recover", "itch", "pressure garment"),
    "nutrition_fluids": ("nutrition", "diet", "food", "eat", "drink", "fluid", "protein", "vitamin", "calorie", "hydrat"),
}
RELATIVE_MARGIN = 0.20     # also drop passages this much further than the best match (they are usually off-topic)

NO_MATCH_NOTE = ("No relevant guidance was found in the knowledge base for this query. "
                 "Say so plainly instead of answering from memory, and suggest the user "
                 "ask a clinician.")


def _mostly_redacted(text: str) -> bool:
    """A passage that is about drugs (two or more names redacted) says nothing useful to a patient once they are
    removed, and reads as a list of '[medication]' markers."""
    return text.count("[medication]") >= 2


def _topic_allowed(topic: str, query: str, for_child: bool) -> bool:
    if topic == "children":
        return for_child
    words = TOPIC_KEYWORDS.get(topic)
    return words is None or any(w in query.lower() for w in words)


def retrieve_treatment_guidance(query: str, top_k: int = 3, for_child: bool = False) -> dict:
    """
    Args:
        query: natural-language treatment question, shaped by the
               agent's system prompt to fit the current triage branch
               or a standalone general Q&A question.
        top_k: how many passages to retrieve. Default 3.
        for_child: True only for a child or infant; otherwise child-specific passages are left out.

    Returns:
        dict with keys:
            passages: list of {text, source, section, topic, distance}, best first,
                      only those within MAX_DISTANCE, identical texts merged.
            note: present only when nothing relevant was found.
        Always restricted to patient_safe content -- see module
        docstring. Some patient_safe chunks still name drugs (the
        dosage safety net in rag/chunker.py only catches numeric doses),
        so drug names are redacted from the returned text; the index
        itself is untouched, so retrieval ranking is unchanged.
    """
    # Look a little deeper, drop passages that do not fit this patient or question, then keep the best top_k.
    found = [p for p in vector_query(query_text=query, top_k=top_k + 6, audience="patient_safe")
             if _topic_allowed(p.get("topic"), query, bool(for_child))]
    passages, seen = [], set()
    best = min((p["distance"] for p in found), default=0)
    for p in found:
        if p["distance"] > MAX_DISTANCE or p["distance"] > best + RELATIVE_MARGIN or p["text"] in seen:
            continue
        seen.add(p["text"])
        text = redact_drug_terms(p["text"])
        if _mostly_redacted(text):
            continue
        if len(passages) < top_k:
            passages.append({**p, "text": text})
    result = {"passages": passages}
    if not passages:
        result["note"] = NO_MATCH_NOTE
    return result
