"""
tools/escalate.py

Generates a consent-gated escalation report (PDF) and optionally emails it
to a clinician. This is the ONLY place in the system where clinician-only
RAG content may be pulled in -- and only when a real recipient_email is
provided (not merely when consent=True), since a patient may consent to a
"just for my records" report that should never contain clinician-only
dosage/clinical content.

Report content policy (see system_prompt.py "Report content: defaults vs.
specific requests"):
  - By default, only the standard branch fields are populated (severity,
    confidence, wound area, guidance passages, clinics). Keep it simple.
  - Optional fields (baseline_*, trend_summary, treatment_summary) are
    populated ONLY when the agent has explicitly gathered them because the
    user asked for something beyond the default -- e.g. "compare with my
    last visit" or "give me more detail on treatment".
  - This module never invents or infers these optional fields itself; it
    only renders what it's given.
"""

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, KeepTogether,
)

from agent.case_store import get_case
from agent.consent import is_valid_email
from agent.medication_filter import redact_medication_terms
from agent.reminders import send_mail
from agent.userctx import current_user_id, user_dir
from rag.chunker import TOPIC_LABELS
from rag.vector_store import query as rag_query

MAX_CLINICS_SHOWN = 4
OUTPUT_DIR = "outputs/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Styles and colours
# --------------------------------------------------------------------------- #

NAVY = colors.HexColor("#1a3a5c")
GRID = colors.HexColor("#d9dee5")
BAND = colors.HexColor("#f4f6f9")

# verdict -> (badge text, badge colour)
URGENCY = {
    "urgent_care": ("URGENT - SEEK CARE NOW", colors.HexColor("#b3261e")),
    "clinical_review": ("CLINICAL REVIEW ADVISED", colors.HexColor("#c77700")),
    "self_care": ("SELF-CARE", colors.HexColor("#2e7d32")),
}
SEVERITY_LABELS = {"1st_degree": "1st degree (superficial)", "2nd_degree": "2nd degree (partial thickness)",
                   "3rd_degree": "3rd degree (full thickness)"}
AREA_LABELS = {"none": "none detected", "small": "small", "moderate": "moderate", "large": "large"}

# What the person should do next, by verdict: (action, timeframe). Fixed, reviewed wording;
# the model does not write these.
ACTIONS = {
    "urgent_care": [
        ("Get an in-person medical assessment (emergency services if breathing, face, hands, genitals or a large area is involved).", "Today, now"),
        ("While arranging care: cool with cool running water, then cover loosely with a clean non-stick dressing.", "Immediately"),
        ("Give this report to the treating clinician.", "At the visit"),
        ("Photograph the wound again and re-check it.", "Within 24 hours"),
    ],
    "clinical_review": [
        ("Arrange a clinician review of the burn.", "Within 24-48 hours"),
        ("Keep the wound clean and covered as described in the guidance below.", "Ongoing"),
        ("Seek care sooner if pain worsens, redness spreads, there is pus, a bad smell or fever.", "Any time"),
        ("Re-photograph the wound for the follow-up check.", "On the follow-up date"),
    ],
    "self_care": [
        ("Care for the burn at home as described in the guidance below.", "Ongoing"),
        ("Watch for warning signs (spreading redness, pus, fever, worsening pain).", "Daily"),
        ("Re-photograph the wound and re-check; see a clinician if it is not improving.", "On the follow-up date"),
    ],
}

_styles = getSampleStyleSheet()


def _style(name, parent="BodyText", **kw):
    _styles.add(ParagraphStyle(name=name, parent=_styles[parent], **kw))


_style("HdrTitle", "Title", fontSize=17, leading=20, textColor=colors.white, alignment=0, spaceAfter=2)
_style("HdrMeta", fontSize=9, leading=12, textColor=colors.white)
_style("Badge", fontSize=10, leading=13, textColor=colors.white, alignment=1, fontName="Helvetica-Bold")
_style("SectionHeader", "Heading2", fontSize=12.5, spaceBefore=13, spaceAfter=5, textColor=NAVY)
_style("TopicHeader", "Heading3", fontSize=10, spaceBefore=6, spaceAfter=2, textColor=colors.HexColor("#444444"))
_style("BodyTextTight", fontSize=9.5, leading=13, spaceAfter=4)
_style("Cell", fontSize=8.5, leading=11)
_style("CellBold", fontSize=8.5, leading=11, fontName="Helvetica-Bold")
_style("CellHead", fontSize=8.5, leading=11, fontName="Helvetica-Bold", textColor=colors.white)
_style("Reason", fontSize=10.5, leading=14, spaceAfter=4)
_style("Caption", fontSize=8, leading=10, textColor=colors.HexColor("#666666"))
_style("Citation", fontSize=8, leading=11, textColor=colors.HexColor("#555555"), spaceAfter=2)
_style("Disclaimer", fontSize=8, leading=11, textColor=colors.HexColor("#777777"))


# --------------------------------------------------------------------------- #
# Content helpers
# --------------------------------------------------------------------------- #

def _esc(text) -> str:
    return escape(str(text)) if text is not None else ""


def _pct(value):
    return f"{value * 100:.0f}%" if value is not None else "not measured"


def derive_verdict(predicted_severity, decision, area_band, multi_region) -> str:
    """Fallback verdict when the agent did not pass its own; mirrors the rules in the system prompt."""
    abstain = decision == "ABSTAIN"
    if predicted_severity == "3rd_degree" and not abstain:
        return "urgent_care"
    if abstain and (area_band == "large" or multi_region):
        return "urgent_care"
    if (not abstain and predicted_severity == "1st_degree"
            and area_band in (None, "none", "small") and not multi_region):
        return "self_care"
    return "clinical_review"


def _reason_sentence(verdict, severity, confidence, decision, area_band, multi_region, followup_worse) -> str:
    sev = SEVERITY_LABELS.get(severity, "an unclassified severity")
    conf = f" (confidence {confidence * 100:.0f}%)" if confidence is not None else ""
    parts = []
    if decision == "ABSTAIN":
        parts.append(f"The image classifier was not confident enough to commit to a severity (best guess: {sev}{conf}), "
                     "so the conservative option was taken")
    elif verdict == "urgent_care":
        parts.append(f"The image analysis indicates a {sev} burn{conf}, which needs prompt in-person assessment")
    elif verdict == "clinical_review":
        parts.append(f"The image analysis indicates a {sev} burn{conf}, which should be reviewed by a clinician")
    else:
        parts.append(f"The image analysis indicates a {sev} burn{conf} that is suitable for self-care with monitoring")
    if area_band == "large":
        parts.append("the affected area looks large in the photo")
    if multi_region:
        parts.append("more than one separate burned region was detected")
    if followup_worse:
        parts.append("the follow-up shows a worsening trend")
    return "; ".join(parts) + "."


def _split_sources(source: str) -> list:
    return [s.strip() for s in (source or "").replace(" (table)", "").split(";") if s.strip()]


def _build_references(passage_lists) -> tuple:
    """Number every distinct source once. Returns (ordered source list, {source: number})."""
    order = {}
    for passages in passage_lists:
        for p in passages or []:
            for src in _split_sources(p.get("source") if isinstance(p, dict) else None):
                order.setdefault(src, len(order) + 1)
    return list(order), order


def _refs_for(passage, numbering) -> str:
    nums = sorted(numbering[s] for s in _split_sources(passage.get("source")) if s in numbering)
    return "".join(f"[{n}]" for n in nums)


def _group_by_topic(passages) -> list:
    """[(topic label, [passages])] in first-seen order; identical texts merged."""
    seen, groups = set(), {}
    for p in passages or []:
        if not isinstance(p, dict):
            p = {"text": str(p)}
        if p.get("text") in seen:
            continue
        seen.add(p.get("text"))
        label = TOPIC_LABELS.get(p.get("topic") or "general", TOPIC_LABELS["general"])
        groups.setdefault(label, []).append(p)
    return list(groups.items())


def _table(rows, widths, header=True, zebra=True, extra=None):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        style.append(("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, BAND]))
    t.setStyle(TableStyle(style + (extra or [])))
    return t


def _p(text, style="Cell"):
    return Paragraph(text, _styles[style])


# --------------------------------------------------------------------------- #
# PDF building
# --------------------------------------------------------------------------- #

def _on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(1.8 * cm, 0.8 * cm, "BurnScope - AI-assisted triage support. Not a diagnostic device.")
    canvas.drawRightString(A4[0] - 1.8 * cm, 0.8 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _build_pdf(
    case_id: str,
    pdf_path: str,
    predicted_severity: str = None,
    calibrated_confidence: float = None,
    wound_fraction: float = None,
    gradcam_overlay_path: str = None,
    guidance_passages: list = None,
    clinician_passages: list = None,
    clinics: list = None,
    baseline_predicted_severity: str = None,
    baseline_wound_fraction: float = None,
    trend_summary: str = None,
    treatment_summary: str = None,
    recipient_email: str = None,
    triage_verdict: str = None,
    decision: str = None,
    area_band: str = None,
    multi_region: bool = None,
    connected_components: int = None,
    quality: dict = None,
    next_followup: dict = None,
    created_date: str = None,
    include: set = None,
    patient_notes: str = None,
) -> None:
    inc = lambda key: include is None or key in include   # noqa: E731  (None = every section)
    width = A4[0] - 3.6 * cm
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4, title=f"BurnScope report {case_id[:8]}",
        topMargin=1.4 * cm, bottomMargin=1.6 * cm, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )
    story = []

    verdict = triage_verdict if triage_verdict in URGENCY else derive_verdict(
        predicted_severity, decision, area_band, multi_region)
    badge_text, badge_color = URGENCY[verdict]
    worse = bool(baseline_predicted_severity and predicted_severity and
                 predicted_severity != baseline_predicted_severity and
                 predicted_severity > baseline_predicted_severity)

    # --- 1. Header band: title, case id, dates, urgency badge ---
    left = [_p("BurnScope Assessment Report", "HdrTitle"),
            _p(f"Case ID: {_esc(case_id)}", "HdrMeta"),
            _p(f"Case opened: {_esc((created_date or '')[:10] or 'n/a')} &nbsp;|&nbsp; "
               f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "HdrMeta")]
    header = Table([[left, _p(badge_text, "Badge")]], colWidths=[width - 5.2 * cm, 5.2 * cm])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), NAVY), ("BACKGROUND", (1, 0), (1, 0), badge_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story += [header, Spacer(1, 10)]

    # --- 2. Key findings box ---
    q = quality or {}
    q_text = "not recorded"
    if q.get("verdict"):
        q_text = "acceptable" if q["verdict"] == "acceptable" else _esc(q.get("issue") or q["verdict"])
    regions = "not measured"
    if multi_region is not None:
        regions = "multiple separate regions" if multi_region else "single main region"
    findings = [
        ("Severity", SEVERITY_LABELS.get(predicted_severity, _esc(predicted_severity) or "not available")),
        ("Confidence", (f"{calibrated_confidence * 100:.0f}%" if calibrated_confidence is not None else "n/a")
         + (" - model was unsure (low reliability)" if decision == "ABSTAIN" else "")),
        ("Wound area (share of photo)", f"{AREA_LABELS.get(area_band, 'not measured')} ({_pct(wound_fraction)})"),
        ("Wound regions", regions),
        ("Image quality", q_text),
    ]
    story.append(_p("Key findings", "SectionHeader"))
    story.append(_table([[_p(k, "CellBold"), _p(v)] for k, v in findings], [5.2 * cm, width - 5.2 * cm],
                        header=False, extra=[("BACKGROUND", (0, 0), (0, -1), BAND)]))

    # --- 3. Why this is being flagged + recommended actions ---
    story.append(_p("Reason for this recommendation", "SectionHeader"))
    story.append(_p(_esc(_reason_sentence(verdict, predicted_severity, calibrated_confidence, decision,
                                          area_band, multi_region, worse)), "Reason"))
    actions = list(ACTIONS[verdict])
    due = (next_followup or {}).get("due")
    rows = [[_p("#", "CellHead"), _p("Recommended action", "CellHead"), _p("When", "CellHead")]]
    for i, (act, when) in enumerate(actions, 1):
        if when == "On the follow-up date" and due:
            when = f"On {_esc(due)}"
        rows.append([_p(str(i)), _p(_esc(act)), _p(_esc(when), "CellBold")])
    story.append(_table(rows, [0.8 * cm, width - 5 * cm, 4.2 * cm]))

    # --- 4. Trend table (follow-ups only) ---
    if (trend_summary or baseline_predicted_severity) and inc("trend"):
        story.append(_p("Change since the first visit", "SectionHeader"))
        if baseline_predicted_severity:
            rows = [[_p(h, "CellHead") for h in ("", "Baseline visit", "Current visit", "Change")]]
            sev_change = ("worse" if worse else "same" if baseline_predicted_severity == predicted_severity
                          else "improved")
            rows.append([_p("Severity", "CellBold"),
                         _p(SEVERITY_LABELS.get(baseline_predicted_severity, _esc(baseline_predicted_severity))),
                         _p(SEVERITY_LABELS.get(predicted_severity, _esc(predicted_severity or "-"))),
                         _p(sev_change, "CellBold")])
            if baseline_wound_fraction is not None or wound_fraction is not None:
                if baseline_wound_fraction is not None and wound_fraction is not None:
                    delta = (wound_fraction - baseline_wound_fraction) * 100
                    change = f"{delta:+.0f} points" if abs(delta) >= 1 else "about the same"
                else:
                    change = "-"
                rows.append([_p("Wound area", "CellBold"), _p(_pct(baseline_wound_fraction)),
                             _p(_pct(wound_fraction)), _p(change, "CellBold")])
            story.append(_table(rows, [3.2 * cm, 4.6 * cm, 4.6 * cm, width - 12.4 * cm]))
            story.append(Spacer(1, 4))
        if trend_summary:
            story.append(_p(_esc(trend_summary), "BodyTextTight"))

    # --- 4b. What the patient told us (typed by them, not verified) ---
    if patient_notes and inc("patient_notes"):
        story.append(_p("Patient-reported information", "SectionHeader"))
        story.append(_p(_esc(patient_notes).replace("\n", "<br/>"), "BodyTextTight"))
        story.append(_p("Typed by the patient; not verified by BurnScope.", "Caption"))

    # --- 5. Grad-CAM image with caption ---
    if inc("gradcam") and gradcam_overlay_path and os.path.exists(gradcam_overlay_path):
        try:
            w, h = ImageReader(gradcam_overlay_path).getSize()
            box = 7.5 * cm
            scale = min(box / w, box / h)
            img = RLImage(gradcam_overlay_path, width=w * scale, height=h * scale)
            story.append(KeepTogether([_p("What the model looked at", "SectionHeader"), img, Spacer(1, 3), _p(
                "Grad-CAM overlay: warmer colours mark the image regions that most influenced the "
                "severity prediction. It shows where the model looked, not a diagnosis.", "Caption")]))
        except Exception:
            story.append(_p("(The Grad-CAM overlay image could not be embedded.)", "Caption"))

    # --- Reference numbering across everything cited in this report ---
    ref_sources, numbering = _build_references([guidance_passages if inc("guidance") else None, clinician_passages])

    # --- 6. Guidance summary, grouped by topic, with numbered references ---
    if inc("guidance") and (treatment_summary or guidance_passages):
        story.append(_p("Guidance summary", "SectionHeader"))
        if treatment_summary:
            story.append(_p(_esc(treatment_summary), "BodyTextTight"))
        for label, group in _group_by_topic(guidance_passages):
            block = [_p(_esc(label), "TopicHeader")]
            for p in group:
                text = _esc(p.get("text", "")).replace("\n", "<br/>")
                block.append(_p(f"{text} <font color='#1a3a5c'><b>{_refs_for(p, numbering)}</b></font>", "BodyTextTight"))
            story.append(KeepTogether(block[:2]))
            story += block[2:]
        if guidance_passages and any("[medication" in str(p.get("text", "")) for p in guidance_passages if isinstance(p, dict)):
            story.append(_p("Medication names are deliberately left out of this report; "
                            "ask a clinician which treatments are appropriate.", "Caption"))

    # --- 7. Clinician reference table (only when a recipient was given) ---
    if clinician_passages:
        story.append(_p("Clinician reference", "SectionHeader"))
        story.append(_p("Clinical-detail material for the treating clinician only.", "Caption"))
        rows = [[_p(h, "CellHead") for h in ("Topic", "Detail", "Source")]]
        for label, group in _group_by_topic(clinician_passages):
            for p in group:
                topic = _esc(p.get("section") or label)
                rows.append([_p(topic, "CellBold"), _p(_esc(p.get("text", "")).replace("\n", "<br/>")),
                             _p(_refs_for(p, numbering) or "-")])
        story.append(_table(rows, [3.4 * cm, width - 5.4 * cm, 2 * cm]))

    # --- 8. Numbered references ---
    if ref_sources:
        story.append(_p("References", "SectionHeader"))
        for src in ref_sources:
            story.append(_p(f"[{numbering[src]}] {_esc(src)}", "Citation"))

    # --- 9. Nearby care ---
    if clinics and inc("clinics"):
        story.append(_p("Nearby care options", "SectionHeader"))
        rows = [[_p(h, "CellHead") for h in ("Facility", "Address", "Distance")]]
        # the closest four, nearest first
        for c in sorted(clinics, key=lambda c: c.get("distance_km") if c.get("distance_km") is not None else 1e9)[:MAX_CLINICS_SHOWN]:
            d = c.get("distance_km")
            rows.append([_p(_esc(c.get("name", "-")), "CellBold"), _p(_esc(c.get("address", "-"))),
                         _p(f"{d} km" if d is not None else "-")])
        story.append(_table(rows, [width * 0.36, width * 0.46, width * 0.18]))

    # --- 10. Follow-up plan ---
    if inc("followup"):
        story.append(_p("Follow-up", "SectionHeader"))
        if due:
            rows = [[_p("Next check-in", "CellBold"), _p(_esc(due)), _p("Re-photograph the wound and run the check again.")]]
            story.append(_table(rows, [3.2 * cm, 3.2 * cm, width - 6.4 * cm], header=False))
            story.append(_p("There is no automatic reminder; please set your own.", "Caption"))
        else:
            story.append(_p("No follow-up date has been scheduled yet.", "BodyTextTight"))

    # --- Footer disclaimer ---
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", color=GRID, thickness=1))
    story.append(Spacer(1, 4))
    footer = ("This is a machine-generated assessment that supports, and does not replace, clinical judgment. "
              "A qualified clinician should confirm severity and management in person.")
    if recipient_email:
        footer += f" Shared with the patient's consent to: {_esc(recipient_email)}."
    story.append(_p(footer, "Disclaimer"))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)


# --------------------------------------------------------------------------- #
# Email delivery
# --------------------------------------------------------------------------- #

def _send_email(recipient_email: str, pdf_path: str, case_id: str) -> bool:
    body = (
        "Please find attached a BurnScope report, shared with patient consent for clinical review.\n\n"
        "This report is generated by an AI-assisted triage support tool and is intended to support, "
        "not replace, clinical judgment.\n\n-- BurnScope"
    )
    with open(pdf_path, "rb") as f:
        pdf = f.read()
    return send_mail(recipient_email, f"BurnScope Report -- Case {case_id[-8:]}", body,
                     [(os.path.basename(pdf_path), pdf, "pdf")])


# --------------------------------------------------------------------------- #
# Report contents: what the user is told BEFORE anything is generated
# --------------------------------------------------------------------------- #

# Sections a user may leave out. The core (header, key findings, recommended actions) is always there.
OPTIONAL_SECTIONS = ("trend", "patient_notes", "gradcam", "guidance", "clinics", "followup", "clinician_reference")
MAX_NOTES_CHARS = 800


# What a user (and therefore the model) might call each section -> its key. First match wins.
_SECTION_WORDS = (
    ("clinician_reference", ("clinician", "doctor", "reference")),
    ("patient_notes", ("note", "patient")),
    ("clinics", ("clinic", "nearby", "hospital", "care")),
    ("followup", ("follow",)),
    ("trend", ("trend", "compar", "previous", "baseline", "change", "visit")),
    ("gradcam", ("grad", "heatmap", "attention", "overlay", "image", "photo")),
    ("guidance", ("guid", "treatment", "advice", "summary")),
)


def normalize_sections(values) -> list:
    """Map whatever names the model used ('nearby care', 'Grad-CAM image'...) to section keys; unknown names are dropped."""
    if isinstance(values, str):
        values = [values]
    keys = set()
    for v in values or []:
        text = str(v).strip().lower().replace("_", " ")
        if text.replace(" ", "_") in OPTIONAL_SECTIONS:
            keys.add(text.replace(" ", "_"))
            continue
        for key, words in _SECTION_WORDS:
            if any(w in text for w in words):
                keys.add(key)
                break
    return sorted(keys)


def report_signature(recipient_email, exclude_sections, patient_notes) -> str:
    """Identifies a report's options, so the system can require that THESE were previewed."""
    return "|".join([(recipient_email or "").strip().lower(),
                     ",".join(normalize_sections(exclude_sections)),
                     (patient_notes or "").strip()[:MAX_NOTES_CHARS]])


def preview_required(case_state: dict, tool_args: dict):
    """None if this exact report was previewed to the user; otherwise a result telling the model to preview."""
    sig = report_signature(tool_args.get("recipient_email"), tool_args.get("exclude_sections"), tool_args.get("patient_notes"))
    if case_state.get("report_preview_sig") == sig:
        return None
    return {"status": "preview_required",
            "message": ("Nothing was generated. First call generate_escalation_report with preview_only=true (same recipient, "
                        "exclude_sections and patient_notes), tell the user in plain words what the report will contain, "
                        "then ask whether they consent.")}


def _describe_plan(*, severity, confidence, verdict, area_band, wound_fraction, has_baseline, has_gradcam, n_guidance,
                   n_clinics, due, notes, n_clinician, recipient, exclude) -> list:
    """[(key, title, included, detail)] -- the honest list of what the report will hold."""
    sev = SEVERITY_LABELS.get(severity, severity or "not available")
    conf = f", confidence {confidence * 100:.0f}%" if confidence is not None else ""
    area = f", wound area {AREA_LABELS.get(area_band, 'not measured')}" + (
        f" ({wound_fraction * 100:.0f}% of the photo)" if wound_fraction is not None else "")
    plan = [
        ("core", "Header and urgency badge", True, f"case ID, dates and the badge: {URGENCY[verdict][0].title()}"),
        ("core", "Key findings", True, f"{sev}{conf}{area}, wound regions, image quality"),
        ("core", "Recommended actions", True, f"{len(ACTIONS[verdict])} numbered actions with timeframes"),
        ("trend", "Change since the first visit", has_baseline,
         "baseline vs current severity and wound area" if has_baseline else "no earlier visit to compare with"),
        ("patient_notes", "Patient-reported information", bool(notes),
         f"your own words ({len(notes)} characters), marked as unverified" if notes else "none added"),
        ("gradcam", "What the model looked at (Grad-CAM image)", has_gradcam,
         "the overlay image with a caption" if has_gradcam else "no image available"),
        ("guidance", "Guidance summary", n_guidance > 0,
         f"{n_guidance} passage(s) grouped by topic, with numbered references" if n_guidance else "no guidance retrieved yet"),
        ("clinics", "Nearby care options", n_clinics > 0,
         f"the closest {min(n_clinics, MAX_CLINICS_SHOWN)} facilities" if n_clinics
         else "no clinics searched yet (tell me your location to add them)"),
        ("followup", "Follow-up plan", True, f"next check-in on {due}" if due else "no follow-up date scheduled yet"),
        ("clinician_reference", "Clinician reference", bool(recipient) and n_clinician > 0,
         f"{n_clinician} clinical-detail passage(s) for the doctor (only because it is being emailed)"
         if recipient and n_clinician else "only added when the report is emailed to a doctor"),
    ]
    return [(k, t, inc and k not in exclude, "left out at your request" if (inc and k in exclude) else d)
            for k, t, inc, d in plan]


def _preview_text(plan: list, recipient) -> str:
    lines = ["This report will contain:"]
    for _key, title, included, detail in plan:
        if included:
            lines.append(f"- {title}: {detail}")
    left_out = [f"{t} ({d})" for _k, t, inc, d in plan if not inc]
    if left_out:
        lines.append("Not included: " + "; ".join(left_out) + ".")
    lines.append(f"It will be emailed to {recipient}, and you can also download it." if recipient
                 else "It will be ready for you to download; nobody is emailed.")
    lines.append("You can ask me to leave a section out or to add a note in your own words.")
    return "\n".join(lines)


def _default_trend(base_sev, base_frac, cur_sev, cur_frac) -> str:
    parts = []
    order = ["1st_degree", "2nd_degree", "3rd_degree"]
    if base_sev in order and cur_sev in order:
        if base_sev == cur_sev:
            parts.append(f"Severity is unchanged ({SEVERITY_LABELS.get(cur_sev, cur_sev)}).")
        else:
            went = "worse" if order.index(cur_sev) > order.index(base_sev) else "better"
            parts.append(f"Severity looks {went}: {SEVERITY_LABELS.get(base_sev, base_sev)} at the first visit, "
                         f"{SEVERITY_LABELS.get(cur_sev, cur_sev)} now.")
    if base_frac is not None and cur_frac is not None:
        parts.append(f"The wound covers {cur_frac * 100:.0f}% of the photo, against {base_frac * 100:.0f}% at the first visit "
                     "(relative to the photo, not a measurement).")
    return " ".join(parts)


def _is_own_address(recipient) -> bool:
    """True if `recipient` is the signed-in user's own account email."""
    if not recipient or not current_user_id():
        return False
    from agent.user_store import get_user
    user = get_user(current_user_id())
    return bool(user) and user["email"].lower() == recipient.strip().lower()


def _clinician_query(severity) -> str:
    """A real sentence to search with (a bare label like '3rd_degree' matches nothing above the relevance floor)."""
    label = {"1st_degree": "superficial (1st degree)", "2nd_degree": "partial thickness (2nd degree)",
             "3rd_degree": "full thickness (3rd degree)"}.get(severity, "")
    return f"{label} burn: clinical management, wound care, nutrition and referral to a specialist burns service".strip()


def _clinician_passages(query: str):
    try:
        results = rag_query(query_text=query, audience="clinician_only", top_k=4)
        return [r for r in results if r["distance"] <= 0.65] or None   # same relevance floor as the patient path
    except Exception as e:
        print(f"[escalate] Clinician-only RAG lookup failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# Main entry point (tool-callable)
# --------------------------------------------------------------------------- #

def generate_escalation_report(
    case_id: str,
    consent: bool,
    recipient_email: str = None,
    predicted_severity: str = None,
    calibrated_confidence: float = None,
    wound_fraction: float = None,
    gradcam_overlay_path: str = None,
    guidance_passages: list = None,
    clinics: list = None,
    # Optional, populated only when the user explicitly asked for more:
    baseline_predicted_severity: str = None,
    baseline_wound_fraction: float = None,
    trend_summary: str = None,
    treatment_summary: str = None,
    # Preview first, then choices:
    preview_only: bool = False,
    exclude_sections: list = None,
    patient_notes: str = None,
    # Filled automatically from the case (see agent/loop.py); the model does not send these:
    triage_verdict: str = None,
    decision: str = None,
    area_band: str = None,
    multi_region: bool = None,
    connected_components: int = None,
    quality: dict = None,
) -> dict:
    """
    Generate a PDF escalation report for a case, and email it to a clinician
    if recipient_email is provided and consent is True.

    Safety gate: clinician-only RAG content is fetched ONLY when
    recipient_email is a real, non-empty string. consent=True alone (e.g.
    "generate this for my own records") must never pull clinician-only
    content into the PDF.
    """
    if not consent and not preview_only:
        return {
            "status": "declined",
            "message": "Report generation requires explicit patient consent. "
                        "No report was generated.",
        }

    if recipient_email and not is_valid_email(recipient_email):
        return {
            "status": "error",
            "message": f"'{recipient_email}' is not a valid email address. "
                       "No report was generated or sent.",
        }

    case = get_case(case_id)
    if case is None:
        return {
            "status": "error",
            "message": f"No case found with id {case_id}.",
        }

    # A comparison is always available for a follow-up: take the baseline from the saved case.
    if case.get("followups") and not baseline_predicted_severity:
        base = case.get("baseline_result") or {}
        baseline_predicted_severity = (base.get("classify") or {}).get("predicted_severity")
        if baseline_wound_fraction is None:
            baseline_wound_fraction = (base.get("segment") or {}).get("wound_fraction")
    if baseline_predicted_severity and not trend_summary:
        trend_summary = _default_trend(baseline_predicted_severity, baseline_wound_fraction, predicted_severity, wound_fraction)
    # Model-written text is patient-facing: no drug names or doses.
    treatment_summary = redact_medication_terms(treatment_summary) if treatment_summary else treatment_summary
    trend_summary = redact_medication_terms(trend_summary) if trend_summary else trend_summary
    notes = (patient_notes or "").strip()[:MAX_NOTES_CHARS]
    exclude = set(normalize_sections(exclude_sections))

    # Clinician-only content is fetched ONLY when the report goes to someone else (a doctor), never on consent alone
    # and never when the user sends it to their own account address.
    to_self = _is_own_address(recipient_email)
    clinician_passages = (_clinician_passages(_clinician_query(predicted_severity))
                          if recipient_email and not to_self else None)

    verdict = triage_verdict if triage_verdict in URGENCY else derive_verdict(predicted_severity, decision, area_band, multi_region)
    plan = _describe_plan(
        severity=predicted_severity, confidence=calibrated_confidence, verdict=verdict, area_band=area_band,
        wound_fraction=wound_fraction, has_baseline=bool(baseline_predicted_severity or trend_summary),
        has_gradcam=bool(gradcam_overlay_path and os.path.exists(gradcam_overlay_path)),
        n_guidance=len(guidance_passages or []), n_clinics=len(clinics or []),
        due=(case.get("next_followup") or {}).get("due"), notes=notes,
        n_clinician=len(clinician_passages or []), recipient=recipient_email, exclude=exclude)

    if preview_only:
        return {
            "status": "preview",
            "recipient": recipient_email or None,
            "contents": [{"section": t, "included": inc, "detail": d} for _k, t, inc, d in plan],
            "signature": report_signature(recipient_email, list(exclude), notes),
            "message": _preview_text(plan, recipient_email),
        }
    include = {k for k, _t, inc, _d in plan if inc and k != "core"}
    if "clinician_reference" not in include:
        clinician_passages = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"report_{case_id[-8:]}_{timestamp}.pdf"
    report_dir = user_dir("reports") if current_user_id() else OUTPUT_DIR   # per-user folder when signed in
    os.makedirs(report_dir, exist_ok=True)
    pdf_path = os.path.join(report_dir, pdf_filename)

    _build_pdf(
        case_id=case_id,
        pdf_path=pdf_path,
        predicted_severity=predicted_severity,
        calibrated_confidence=calibrated_confidence,
        wound_fraction=wound_fraction,
        gradcam_overlay_path=gradcam_overlay_path,
        guidance_passages=guidance_passages,
        clinician_passages=clinician_passages,
        clinics=clinics,
        baseline_predicted_severity=baseline_predicted_severity,
        baseline_wound_fraction=baseline_wound_fraction,
        trend_summary=trend_summary,
        treatment_summary=treatment_summary,
        recipient_email=recipient_email,
        triage_verdict=triage_verdict,
        decision=decision,
        area_band=area_band,
        multi_region=multi_region,
        connected_components=connected_components,
        quality=quality,
        next_followup=case.get("next_followup"),
        created_date=case.get("created_date"),
        include=include | ({"followup"} if "followup" not in exclude else set()),
        patient_notes=notes,
    )

    email_sent = False
    if recipient_email:
        email_sent = _send_email(recipient_email, pdf_path, case_id)

    return {
        "status": "success",
        "pdf_path": pdf_path,
        "emailed": email_sent,
        "recipient_email": recipient_email if email_sent else None,
        "message": (
            "The report is ready; the user can download it with the Download button in the app (never quote a file path)."
            + (f" Emailed to {recipient_email}." if email_sent else
               " Not emailed (no recipient provided)." if not recipient_email else
               " Email delivery failed, so it is only available to download.")
        ),
    }