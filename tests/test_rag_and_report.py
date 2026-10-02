"""RAG retrieval quality and escalation-report layout. No Groq calls.
Uses the built index in rag/vector_store (rebuild with `python -m rag.build_index` if it is missing)."""
import os

import pytest

import tools.escalate as escalate
from agent.medication_filter import find_drug_terms
from rag.vector_store import _get_collection, query
from tools.rag_retrieve import MAX_DISTANCE, retrieve_treatment_guidance

# (question, audience, text the top passage's section title must contain)
TOP_HIT = [
    ("How many minutes should I hold a burn under cool running water?", "patient_safe", "COOLING A BURN (FIRST AID)"),
    ("How do I cool a fresh burn with water right after it happens?", "patient_safe", "COOLING A BURN (FIRST AID)"),
    ("My two year old toddler burned her hand on the stove, what do I do?", "patient_safe", "CHILDREN"),
    ("When should I call an ambulance for my child's burn?", "patient_safe", "BURNS IN CHILDREN"),
    ("What are the signs that my burn wound is infected?", "patient_safe", "INFECTION"),
    ("My child has a burn and is now vomiting, drowsy and has a red rash", "patient_safe", "TOXIC SHOCK"),
    ("I splashed a chemical on my skin, how long do I rinse it with water?", "patient_safe", "CHEMICAL"),
    ("Do I need a tetanus shot after a burn?", "patient_safe", "TETANUS"),
    ("Is there good evidence for how long to cool a burn?", "clinician_only", "EVIDENCE"),
    ("When should a child with a burn be referred to a burns service?", "clinician_only", "REFERRAL"),
]
OFF_TOPIC = ["who won the football world cup", "how do I fix my car engine", "best pasta recipe"]


@pytest.mark.parametrize("question,audience,section", TOP_HIT, ids=[t[0][:45] for t in TOP_HIT])
def test_top_passage_is_the_right_section(question, audience, section):
    top = query(question, 1, audience)[0]
    assert section in top["section"].upper(), top["section"]
    assert top["distance"] <= MAX_DISTANCE


@pytest.mark.parametrize("question", OFF_TOPIC)
def test_off_topic_returns_nothing_with_a_note(question):
    result = retrieve_treatment_guidance(question, 3)
    assert result["passages"] == []
    assert "No relevant guidance" in result["note"]


def test_patient_path_is_patient_safe_and_drug_free():
    meta = _get_collection().get(include=["documents", "metadatas"])
    clinician_texts = {d for d, m in zip(meta["documents"], meta["metadatas"]) if m["audience"] == "clinician_only"}
    assert clinician_texts, "index has no clinician_only chunks"
    for q, aud, _ in TOP_HIT:
        if aud == "patient_safe":
            for p in retrieve_treatment_guidance(q, 3)["passages"]:
                assert p["text"] not in clinician_texts
                assert not find_drug_terms(p["text"])


def test_index_has_expected_metadata():
    meta = _get_collection().get(include=["metadatas"])["metadatas"]
    assert len(meta) >= 60
    assert all(m["source"] and m["source"] != "Unknown source" and m["section"] and m["topic"] for m in meta)


# ------------------------------------------------------------------ report
@pytest.mark.parametrize("severity,decision,band,multi,expected", [
    ("3rd_degree", "PROCEED", "small", False, "urgent_care"),
    ("2nd_degree", "ABSTAIN", "large", False, "urgent_care"),
    ("1st_degree", "ABSTAIN", "small", True, "urgent_care"),
    ("1st_degree", "PROCEED", "small", False, "self_care"),
    ("1st_degree", "PROCEED", "large", False, "clinical_review"),
    ("2nd_degree", "PROCEED", "moderate", False, "clinical_review"),
])
def test_report_verdict_fallback_matches_prompt_rules(severity, decision, band, multi, expected):
    assert escalate.derive_verdict(severity, decision, band, multi) == expected


def test_report_pdf_is_built_with_all_sections(tmp_path, monkeypatch):
    monkeypatch.setattr(escalate, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(escalate, "get_case", lambda cid: {
        "case_id": cid, "created_date": "2026-09-14T10:00:00", "baseline_result": {}, "followups": [],
        "next_followup": {"due": "2026-09-22", "interval_days": 2}})
    passages = retrieve_treatment_guidance("cool a burn and watch for infection in a child", 5)["passages"]
    overlay = next((os.path.join(r, f) for r, _, fs in os.walk("outputs/gradcam") for f in fs if f.endswith(".png")), None)
    result = escalate.generate_escalation_report(
        "case_test0001", True, recipient_email=None, predicted_severity="2nd_degree", calibrated_confidence=0.83,
        wound_fraction=0.27, gradcam_overlay_path=overlay, guidance_passages=passages,
        clinics=[{"name": "City Burns Unit", "address": "12 Example Road", "distance_km": 3.2}],
        baseline_predicted_severity="1st_degree", baseline_wound_fraction=0.12,
        trend_summary="Larger than at the first visit.", triage_verdict="clinical_review", decision="ABSTAIN",
        area_band="moderate", multi_region=True, quality={"verdict": "acceptable"})
    assert result["status"] == "success"
    assert os.path.getsize(result["pdf_path"]) > 5000


def test_report_refuses_without_consent():
    assert escalate.generate_escalation_report("case_x", False)["status"] == "declined"


# ------------------------------------------------------------------ nearby care
def test_clinic_search_widens_until_four_options(monkeypatch):
    import tools.clinic_locator as cl
    calls = []

    def fake_find(lat, lon, radius):
        calls.append(radius)
        n = {5.0: 1, 15.0: 3, 30.0: 6}[radius]
        return [{"name": f"C{i}", "address": "-", "distance_km": float(i), "is_hospital": False} for i in range(n)]

    monkeypatch.setattr(cl, "_geocode_location", lambda loc: (1.0, 1.0))
    monkeypatch.setattr(cl, "_find_clinics", fake_find)
    result = cl.locate_nearby_clinics("anywhere", 5.0)
    assert calls == [5.0, 15.0, 30.0]
    assert len(result["clinics"]) == 6 and "widened to 30 km" in result["note"]
    monkeypatch.setattr(cl, "_find_clinics", lambda *a: fake_find(a[0], a[1], 30.0))
    assert "note" not in cl.locate_nearby_clinics("anywhere", 5.0)


def test_report_lists_only_the_closest_four_clinics():
    rows = [{"name": f"Clinic {i}", "address": "-", "distance_km": d} for i, d in enumerate([9, 2, 7, 1, 5, 3])]
    shown = sorted(rows, key=lambda c: c["distance_km"])[:escalate.MAX_CLINICS_SHOWN]
    assert [c["distance_km"] for c in shown] == [1, 2, 3, 5]


# ------------------------------------------------------------------ Option A additions
def test_mostly_redacted_passages_are_dropped():
    from tools.rag_retrieve import _mostly_redacted
    assert _mostly_redacted("Management:\n- [medication]\n- [medication]\n- [medication], and slow-release [medication], etc.")
    assert not _mostly_redacted("Keep the wound clean and covered with a non-stick dressing. Ask about [medication] for pain.")


def test_size_guard_shrinks_tool_results_inside_a_turn():
    import json
    import agent.context as C

    def tool_msg(cid, payload):
        return {"role": "tool", "tool_call_id": cid, "content": json.dumps(payload)}

    def call(cid, name):
        return {"role": "assistant", "content": None,
                "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}]}

    clinics = {"clinics": [{"name": f"Clinic {i}", "address": "x" * 200, "distance_km": i} for i in range(10)]}
    passages = {"passages": [{"text": "guidance " * 200, "source": "S", "section": "s", "topic": "t", "distance": 0.2} for _ in range(3)]}
    h = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
         call("c1", "locate_nearby_clinics"), tool_msg("c1", clinics),
         call("c2", "retrieve_treatment_guidance"), tool_msg("c2", passages)]
    before = C.estimate_tokens(h)
    assert C.fit_within_turn(h, before + 10) == 0                      # already fits: untouched
    saved = C.fit_within_turn(h, 900)
    assert saved > 0 and C.estimate_tokens(h) < before
    kept = json.loads(h[-1]["content"])["passages"]                     # newest guidance is shortened, not dropped
    assert len(kept) == 3 and all(0 < len(p["text"]) <= 600 for p in kept)
    assert "address" not in json.loads(h[3]["content"])["clinics"][0]   # older clinics list is compacted
    assert [m["role"] for m in h] == ["system", "user", "assistant", "tool", "assistant", "tool"]


# ------------------------------------------------------------------ report preview, choices, notes
BASE = dict(predicted_severity="2nd_degree", calibrated_confidence=0.83, wound_fraction=0.27, decision="PROCEED",
            area_band="moderate", multi_region=False, triage_verdict="clinical_review", quality={"verdict": "acceptable"})


@pytest.fixture()
def report_env(tmp_path, monkeypatch):
    built = {}
    monkeypatch.setattr(escalate, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(escalate, "get_case", lambda cid: {"case_id": cid, "created_date": "2026-09-14", "baseline_result": {},
                                                          "followups": [], "next_followup": {"due": "2026-09-22"}})
    monkeypatch.setattr(escalate, "_build_pdf", lambda **kw: built.update(kw) or open(kw["pdf_path"], "wb").write(b"%PDF"))
    monkeypatch.setattr(escalate, "send_mail", lambda *a, **k: True)
    monkeypatch.setattr(escalate, "_clinician_passages", lambda q: [{"text": "clinical detail", "source": "S", "section": "REFERRAL"}])
    return built


def test_preview_lists_contents_and_generates_nothing(report_env, tmp_path):
    r = escalate.generate_escalation_report("case_x", False, preview_only=True, clinics=[{"name": "A", "distance_km": 1}],
                                            guidance_passages=[{"text": "t", "source": "S"}], patient_notes="Allergic to latex", **BASE)
    assert r["status"] == "preview" and not report_env and not list(tmp_path.iterdir())     # no PDF, no consent needed
    by = {c["section"]: c for c in r["contents"]}
    assert by["Key findings"]["included"] and by["Nearby care options"]["included"] and by["Patient-reported information"]["included"]
    assert not by["Change since the first visit"]["included"] and not by["Clinician reference"]["included"]
    assert "nobody is emailed" in r["message"] and "Allergic" not in r["message"]            # says how many characters, not the text


def test_preview_with_recipient_adds_clinician_reference_and_exclusions(report_env):
    r = escalate.generate_escalation_report("case_x", False, preview_only=True, recipient_email="dr@example.com",
                                            exclude_sections=["clinics", "bogus"], clinics=[{"name": "A", "distance_km": 1}], **BASE)
    by = {c["section"]: c for c in r["contents"]}
    assert by["Clinician reference"]["included"] and not by["Nearby care options"]["included"]
    assert by["Nearby care options"]["detail"] == "left out at your request" and "dr@example.com" in r["message"]


def test_real_report_follows_the_chosen_sections_and_notes(report_env):
    r = escalate.generate_escalation_report("case_x", True, exclude_sections=["gradcam", "clinics"], patient_notes="  Allergic to latex  ",
                                            clinics=[{"name": "A", "distance_km": 1}], guidance_passages=[{"text": "t", "source": "S"}], **BASE)
    assert r["status"] == "success" and report_env["patient_notes"] == "Allergic to latex"
    assert "clinics" not in report_env["include"] and "gradcam" not in report_env["include"] and "guidance" in report_env["include"]
    assert report_env["clinician_passages"] is None                    # no recipient -> never clinician-only content


def test_clinician_content_only_with_recipient_and_only_if_kept(report_env):
    escalate.generate_escalation_report("case_x", True, recipient_email="dr@example.com", **BASE)
    assert report_env["clinician_passages"]
    escalate.generate_escalation_report("case_x", True, recipient_email="dr@example.com", exclude_sections=["clinician_reference"], **BASE)
    assert report_env["clinician_passages"] is None


def test_followup_comparison_is_filled_from_the_saved_case(report_env, monkeypatch):
    monkeypatch.setattr(escalate, "get_case", lambda cid: {
        "case_id": cid, "created_date": "2026-09-14", "next_followup": None, "followups": [{"date": "2026-09-16"}],
        "baseline_result": {"classify": {"predicted_severity": "3rd_degree"}, "segment": {"wound_fraction": 0.4}}})
    escalate.generate_escalation_report("case_x", True, **BASE)
    assert report_env["baseline_predicted_severity"] == "3rd_degree" and report_env["baseline_wound_fraction"] == 0.4
    assert "better" in report_env["trend_summary"] and "27%" in report_env["trend_summary"] and "40%" in report_env["trend_summary"]
    assert "trend" in report_env["include"]


def test_model_written_summaries_never_carry_drug_names(report_env):
    escalate.generate_escalation_report("case_x", True, treatment_summary="Take ibuprofen 400 mg twice a day and keep it covered.", **BASE)
    assert "ibuprofen" not in report_env["treatment_summary"].lower() and "keep it covered" in report_env["treatment_summary"]


def test_preview_is_required_for_exactly_these_options():
    args = {"recipient_email": "Dr@Example.com", "exclude_sections": ["clinics"], "patient_notes": "note"}
    assert escalate.preview_required({}, args)["status"] == "preview_required"
    sig = escalate.report_signature("dr@example.com", ["clinics"], "note")
    assert escalate.preview_required({"report_preview_sig": sig}, args) is None                    # same options, case-insensitive email
    assert escalate.preview_required({"report_preview_sig": sig}, {**args, "patient_notes": "other"})["status"] == "preview_required"
    assert escalate.preview_required({"report_preview_sig": sig}, {**args, "recipient_email": "x@y.org"})["status"] == "preview_required"
    assert escalate.preview_required({"report_preview_sig": sig}, {**args, "exclude_sections": []})["status"] == "preview_required"


def test_section_names_the_model_might_use_are_understood():
    n = escalate.normalize_sections
    assert n(["nearby care", "Grad-CAM image"]) == ["clinics", "gradcam"]
    assert n("follow-up plan") == ["followup"] and n(["comparison with last visit"]) == ["trend"]
    assert n(["my notes"]) == ["patient_notes"] and n(["treatment guidance"]) == ["guidance"] and n(["doctor reference"]) == ["clinician_reference"]
    assert n(["clinics", "clinics"]) == ["clinics"] and n(["xyz", None, ""]) == [] and n(None) == []
    assert escalate.report_signature("", ["nearby care"], "") == escalate.report_signature("", ["clinics"], "")   # same preview either way


def test_clinician_reference_is_really_found_for_every_severity():
    for sev in ("1st_degree", "2nd_degree", "3rd_degree", None):
        found = escalate._clinician_passages(escalate._clinician_query(sev))
        assert found and all(p["distance"] <= 0.65 for p in found), sev


def test_real_pdf_contains_exactly_the_chosen_sections(tmp_path, monkeypatch):
    """Builds the real PDF (nothing mocked except the case lookup and mail) and reads its text back."""
    pypdf = pytest.importorskip("pypdf")
    monkeypatch.setattr(escalate, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(escalate, "get_case", lambda cid: {"case_id": cid, "created_date": "2026-09-14", "baseline_result": {},
                                                          "followups": [], "next_followup": {"due": "2026-09-22"}})
    monkeypatch.setattr(escalate, "send_mail", lambda *a, **k: True)
    passages = retrieve_treatment_guidance("burn first aid and wound care in an adult", 5)["passages"]

    def text_of(**kw):
        r = escalate.generate_escalation_report("case_pdf", True, guidance_passages=passages,
                                                clinics=[{"name": "X", "address": "y", "distance_km": 1}], **BASE, **kw)
        assert r["status"] == "success"
        return "\n".join(p.extract_text() for p in pypdf.PdfReader(r["pdf_path"]).pages).lower()

    doctor = text_of(recipient_email="dr@example.com", patient_notes="Allergic to latex.", exclude_sections=["nearby care"])
    assert "clinician reference" in doctor and "latex" in doctor and "nearby care options" not in doctor
    assert "guidance summary" in doctor and "references" in doctor and "follow-up" in doctor
    own = text_of()
    assert "clinician reference" not in own and "nearby care options" in own and "latex" not in own     # nothing extra without a doctor


# ------------------------------------------------------------------ fixes from the first UI trial
def test_retrieval_drops_drug_talk_and_far_off_passages(monkeypatch):
    import tools.rag_retrieve as rr
    found = [
        {"text": "Cool the burn with running water.", "source": "S1", "section": "a", "topic": "first_aid", "distance": 0.20},
        {"text": "Keep the dressing clean and dry.", "source": "S2", "section": "b", "topic": "wound_care", "distance": 0.33},
        {"text": "Rehabilitation: leg press and bicycle exercise for months.", "source": "S3", "section": "c", "topic": "rehabilitation", "distance": 0.52},
        {"text": "Routine use of systemic ibuprofen and topical bacitracin is discussed.", "source": "S4", "section": "d", "topic": "wound_care", "distance": 0.25},
    ]
    monkeypatch.setattr(rr, "vector_query", lambda **kw: found)
    kept = [p["source"] for p in rr.retrieve_treatment_guidance("burn care", 4)["passages"]]
    assert kept == ["S1", "S2"]          # off-topic (0.52 > 0.20 + 0.20) and drug-only (2 redacted names) passages are gone


def test_self_send_report_needs_no_retyping_and_gets_no_clinician_content(report_env, monkeypatch):
    import agent.consent as consent
    own = "asha@example.com"
    ask = {"role": "assistant", "content": "Do you consent to me generating this report and emailing it to you?"}
    hist = [{"role": "system", "content": "s"}, ask, {"role": "user", "content": "yes please, mail it to me"}]
    assert consent.check_report_request(hist, {"recipient_email": own}, own_email=own) is None                       # own address: fine
    assert consent.check_report_request(hist, {"recipient_email": "dr@example.com"}, own_email=own)["status"] == "recipient_not_provided"
    assert consent.check_report_request(hist, {"recipient_email": own})["status"] == "recipient_not_provided"        # no session email known

    monkeypatch.setattr(escalate, "_is_own_address", lambda recipient: (recipient or "").lower() == own)
    r = escalate.generate_escalation_report("case_x", True, recipient_email=own, **BASE)
    assert r["status"] == "success" and report_env["clinician_passages"] is None                # personal copy: no clinician-only detail
    assert "file path" in r["message"] and "outputs" not in r["message"] and "\\" not in r["message"]    # never a path for the model to quote
    escalate.generate_escalation_report("case_x", True, recipient_email="dr@example.com", **BASE)
    assert report_env["clinician_passages"]                                                     # a doctor still gets it


def test_child_and_rehab_passages_only_when_they_fit_the_question(monkeypatch):
    import tools.rag_retrieve as rr

    def fake(**kw):
        return [{"text": f"text {t}", "source": t, "section": t, "topic": t, "distance": 0.30}
                for t in ("first_aid", "children", "rehabilitation", "nutrition_fluids", "wound_care")]

    monkeypatch.setattr(rr, "vector_query", fake)
    topics = lambda q, **kw: [p["topic"] for p in rr.retrieve_treatment_guidance(q, 5, **kw)["passages"]]
    assert topics("care for a fresh burn on the leg") == ["first_aid", "wound_care"]                 # adult, acute: no child/rehab/nutrition
    assert "children" in topics("care for a fresh burn", for_child=True)
    assert "rehabilitation" in topics("how do I reduce scarring and stiffness with exercises")
    assert "nutrition_fluids" in topics("what should I eat and drink while healing")


def test_recipient_is_validated_even_during_preview():
    """The bug this guards: a recipient must not reach even a preview unless it is the user's own account
    address or was actually typed by them -- previously only the real (consent) call was checked."""
    import agent.consent as consent
    own = "me@example.com"
    hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "share the report with me to mail"}]
    # the legitimate case: "send it to me" resolves to the signed-in account's own address -- must pass
    assert consent.validate_recipient(hist, {"recipient_email": own}, own_email=own) is None
    # a name/address the user never typed and that isn't their own account -- must be blocked
    assert consent.validate_recipient(hist, {"recipient_email": "someone.else@example.com"}, own_email=own)["status"] == "recipient_not_provided"
    # once the user actually types a different address, that one is fine
    hist2 = hist + [{"role": "user", "content": "send it on friend@example.com"}]
    assert consent.validate_recipient(hist2, {"recipient_email": "friend@example.com"}, own_email=own) is None
    assert consent.validate_recipient(hist, {"recipient_email": "not-an-email"}, own_email=own)["status"] == "invalid_email"
