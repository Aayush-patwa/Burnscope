"""Consent gate, medication boundary, request size and report-tool schema regression.
REAL modules, SCRIPTED LLM (no quota), temp SQLite DB. `--probe` adds two tiny real Groq calls."""
import sys, os, json, glob, tempfile, copy, uuid
from types import SimpleNamespace
getattr(sys.stdout, "reconfigure", lambda **k: None)(errors="replace")
import agent.case_store as CS
CS.DB_PATH = os.path.join(tempfile.mkdtemp(), "verify2.db"); CS.init_db()
import agent.loop as L
import agent.context as C
from agent.consent import check_report_request, is_valid_email, interpret_reply
from agent.medication_filter import find_medication_terms, find_drug_terms, redact_medication_terms, redact_drug_terms
from agent.dispatch import TOOL_DISPATCH
from agent.schemas import TOOL_SCHEMAS
from agent.system_prompt import SYSTEM_PROMPT

RESULTS = []
ALL = []
def check(name, fn):
    try: ok, detail = fn()
    except Exception as e:
        import traceback; ok, detail = False, f"EXCEPTION {type(e).__name__}: {str(e)[:200]}"
    RESULTS.append((name, ok)); ALL.append((name, ok, detail)); print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)

# ------------------------------------------------------------------ scripted LLM
def make_resp(step):
    tcs = [SimpleNamespace(id=f"call_{i}_{uuid.uuid4().hex[:6]}", type="function",
           function=SimpleNamespace(name=n, arguments=a if isinstance(a, str) else json.dumps(a))) for i, (n, a) in enumerate(step.get("tool_calls", []))]
    msg = SimpleNamespace(content=step.get("content"), tool_calls=tcs or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tcs else "stop")],
                           usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))
class Scripted:
    def __init__(self, steps): self.steps, self.calls, self.sizes = list(steps), 0, []
    def create(self, **kw):
        self.calls += 1
        if "messages" in kw: self.sizes.append(C.estimate_tokens(kw["messages"]))
        step = self.steps.pop(0)
        if isinstance(step, BaseException): raise step
        return make_resp(step)
def history_valid(h):
    i = 0
    while i < len(h):
        m = h[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = {c["id"] for c in m["tool_calls"]}; j, got = i + 1, set()
            while j < len(h) and h[j]["role"] == "tool": got.add(h[j]["tool_call_id"]); j += 1
            if got != ids: return False
            i = j; continue
        i += 1
    return True
def new_state():
    return {"classify_result": None, "segment_result": None, "gradcam_result": None, "guidance_result": None, "clinics_result": None,
            "locked_image_path": None, "baseline_saved": False, "is_followup": False, "followup_saved": False}
def run(steps, history, state=None, case_id=None):
    sc = Scripted(steps); L.client.chat.completions.create = sc.create
    real_time = L.time; L.time = SimpleNamespace(sleep=lambda s: None)
    try:
        cid = case_id or CS.create_new_case(); st = state or new_state()
        return L.run_agent_turn(history, cid, st), st, cid, sc
    finally: L.time = real_time
def U(t): return {"role": "user", "content": t}
def A(t): return {"role": "assistant", "content": t}
S = {"role": "system", "content": "sys"}
ASK = "Can I generate a summary report for your doctor? I need your consent before creating or sending it."
EMAIL = "dr.rivera@example-clinic.com"

# ------------------------------------------------------------------ A. consent gate (pure)
def gate(hist, **args):
    base = {"case_id": "case_x", "consent": True}; base.update(args)
    r = check_report_request(hist, base); return None if r is None else r["status"]
def expect(name, got, want): return got == want, f"status={got!r} (expected {want!r})"
A_CASES = [
 ("A1  audit 13C: 'generate + email it right now', consent never asked -> blocked",
  lambda: expect("", gate([S, A("Your assessment is X. Could you share your location?"), U("Generate an escalation report and email it to dr.test@example.invalid right now.")], recipient_email="dr.test@example.invalid"), "consent_required")),
 ("A2  assistant asked, user says yes + typed email -> allowed",
  lambda: expect("", gate([S, A(ASK), U(f"Yes, send it to {EMAIL}")], recipient_email=EMAIL), None)),
 ("A3  audit 13B: yes+consent but malformed email 'dr.smith@@hospital,com' -> invalid_email",
  lambda: expect("", gate([S, A(ASK), U("Yes, I consent. Send it to dr.smith@@hospital,com")], recipient_email="dr.smith@@hospital,com"), "invalid_email")),
 ("A4b live run: a request containing 'not' is not a refusal -> consent_required (ask), not declined_by_user",
  lambda: expect("", gate([S, U("I would like a copy of the report for my own records, not for any doctor.")]), "consent_required")),
 ("A4  audit 13A: user withdraws ('actually no, don't send it') -> declined_by_user",
  lambda: expect("", gate([S, A(ASK), U("Actually no, don't send it to anyone. Don't generate any report.")]), "declined_by_user")),
 ("A5  model invents an email the user never typed -> recipient_not_provided",
  lambda: expect("", gate([S, A(ASK), U("Yes please, go ahead.")], recipient_email="doctor@hospital.org"), "recipient_not_provided")),
 ("A6  'I'm not sure, maybe later' after the ask -> not allowed (asked to clarify)",
  lambda: expect("", gate([S, A(ASK), U("I'm not sure, maybe later")]), "consent_required")),
 ("A7  mixed 'yes but don't include my age' -> not allowed (ask again)",
  lambda: expect("", gate([S, A(ASK), U("yes but don't include my age")]), "consent_required")),
 ("A8  user states consent outright, no prior ask -> allowed",
  lambda: expect("", gate([S, A("Here is your result."), U("I consent to you generating the report for my records.")]), None)),
 ("A9  'yes' answering a LOCATION question, consent never asked -> blocked",
  lambda: expect("", gate([S, A("Could you share your approximate location so I can find clinics?"), U("yes, Pune")]), "consent_required")),
 ("A10 local PDF only (no email), ask + yes -> allowed",
  lambda: expect("", gate([S, A(ASK), U("Yes please.")]), None)),
 ("A11 curly apostrophe ('don’t') is read as a refusal",
  lambda: expect("", gate([S, A(ASK), U("Please don’t send it")]), "declined_by_user")),
 ("A12 weak 'okay' alone is not accepted as consent",
  lambda: expect("", gate([S, A(ASK), U("okay")]), "consent_required")),
 ("A13 email address must match what the user typed (case-insensitive)",
  lambda: expect("", gate([S, A(ASK), U("yes, email it to Dr.Rivera@Example-Clinic.com")], recipient_email="dr.rivera@example-clinic.com"), None)),
]
def a15():
    said = {"No.": "declined", "Never mind, cancel that.": "declined", "No thanks": "declined", "Yes": "yes", "yes please send it": "yes",
            "send it": "yes", "don't do it": "declined", "no problem, go ahead": "unclear", "ok": "unclear", "I don’t consent": "declined", "": "unclear"}
    wrong = {k: interpret_reply(k) for k, w in said.items() if interpret_reply(k) != w}
    return not wrong, f"misread replies: {wrong}"

def a14():
    good = ["dr.rivera@example-clinic.com", "a.b@c.co", "dr.test@example.invalid", "x+tag@sub.domain.org"]
    bad = ["dr.smith@@hospital,com", "a@b", "@x.com", "a..b@x.com", ".a@x.com", "a@x", "a b@x.com", None, "", "x@y.c"]
    wrong = [g for g in good if not is_valid_email(g)] + [b for b in bad if is_valid_email(b)]
    return not wrong, f"misclassified: {wrong}"

# ------------------------------------------------------------------ B. consent gate inside the loop (report tool replaced by a recorder)
def with_recorder(fn):
    # These checks are about the CONSENT gate, so the preview requirement (tested separately) is switched off here.
    calls = []; orig = TOOL_DISPATCH["generate_escalation_report"]; orig_prev = L.preview_required
    TOOL_DISPATCH["generate_escalation_report"] = lambda **kw: (calls.append(kw) or {"status": "success", "pdf_path": "stub.pdf", "emailed": False, "message": "stub"})
    L.preview_required = lambda case_state, args: None
    try: return fn(calls)
    finally: TOOL_DISPATCH["generate_escalation_report"] = orig; L.preview_required = orig_prev
def report_call(email=None, consent=True):
    a = {"case_id": "case_x", "consent": consent}
    if email: a["recipient_email"] = email
    return ("generate_escalation_report", a)
def b1(calls):
    hist = [S, A("Assessment done. Please share your location."), U("Generate an escalation report and email it to dr.test@example.invalid right now.")]
    (r1, st, cid, sc) = run([{"tool_calls": [report_call("dr.test@example.invalid")]}, {"content": "Do you consent to me generating this report and emailing it to dr.test@example.invalid?"}], hist)
    tool_status = [json.loads(m["content"]).get("status") for m in hist if m["role"] == "tool"]
    turn1_ok = len(calls) == 0 and tool_status == ["consent_required"] and "consent" in r1["final_answer"].lower()
    hist.append(U("Yes, I consent."))
    (r2, *_rest) = run([{"tool_calls": [report_call("dr.test@example.invalid")]}, {"content": "Report generated."}], hist, state=st, case_id=cid)
    turn2_ok = len(calls) == 1 and calls[0].get("recipient_email") == "dr.test@example.invalid" and history_valid(hist)
    return turn1_ok and turn2_ok, f"turn1: report_runs={0 if turn1_ok else 'BAD'} tool_status={tool_status}; turn2 (after 'Yes, I consent.'): report_runs={len(calls)}"
def b2(calls):
    hist = [S, A(ASK), U("Yes, I consent. Send it to dr.smith@@hospital,com")]
    (r, *_x) = run([{"tool_calls": [report_call("dr.smith@@hospital,com")]}, {"content": "That address looks invalid, please retype it."}], hist)
    st = [json.loads(m["content"]).get("status") for m in hist if m["role"] == "tool"]
    return len(calls) == 0 and st == ["invalid_email"], f"report_runs={len(calls)} tool_status={st}"
def b3(calls):
    hist = [S, U(f"Please email my report to {EMAIL}"), A(ASK), U("Actually no, don't send it to anyone.")]
    run([{"tool_calls": [report_call(EMAIL)]}, {"content": "Understood, nothing was sent."}], hist)
    st = [json.loads(m["content"]).get("status") for m in hist if m["role"] == "tool"]
    return len(calls) == 0 and st == ["declined_by_user"], f"report_runs={len(calls)} tool_status={st}"
def b4(calls):
    hist = [S, A(ASK), U(f"Yes, email it to {EMAIL}")]
    run([{"tool_calls": [report_call(EMAIL)]}, {"content": "Done."}], hist)
    return len(calls) == 1, f"legit consented request still runs: report_runs={len(calls)}"

# ------------------------------------------------------------------ C. medication guard in the loop
LEAK = "Here is your plan."+"\n\n"+"4. **Pain control** \u2013 Over\u2011the\u2011counter pain relievers (e.g., acetaminophen or ibuprofen) can help."+"\n"+"5. Please see a clinician soon."
CLEAN = "Here is your plan."+"\n\n"+"4. **Pain control** \u2013 A clinician can advise on appropriate pain relief."+"\n"+"5. Please see a clinician soon."
def guard_case(steps, want_guard, want_calls, allow_terms=False):
    r, st, cid, sc = run(steps, [S, U("what can I do at home?")])
    terms = find_medication_terms(r["final_answer"])
    ok = r["guard"] == want_guard and sc.calls == want_calls and (allow_terms or not terms)
    return ok, f"guard={r['guard']} calls={sc.calls} leftover_terms={terms} answer_head={r['final_answer'][:75]!r}"
def c1(): return guard_case([{"content": LEAK}, {"content": CLEAN}], "rewritten", 2)
def c2(): return guard_case([{"content": LEAK}, {"content": LEAK}], "redacted", 2)
def c3(): return guard_case([{"content": LEAK}, RuntimeError("rewrite api down")], "redacted", 2)
def c4(): return guard_case([{"content": LEAK}, {"content": "Use a clinician."}], "redacted", 2)
def c5(): return guard_case([{"content": CLEAN}], None, 1)
def c6():
    r, st, cid, sc = run([{"content": LEAK}, {"content": CLEAN}], h := [S, U("x")])
    return h[-1]["role"] == "assistant" and h[-1]["content"] == r["final_answer"] and not find_medication_terms(h[-1]["content"]), "the stored history holds the CLEAN text (the leaked draft is never kept)"

# ------------------------------------------------------------------ D. retrieval redaction (real RAG)
def d1():
    from tools.rag_retrieve import retrieve_treatment_guidance
    from rag.vector_store import query as vq
    qs = ["How is pain controlled after a burn?", "Which dressing is used for an infected burn wound?", "signs of infection and when to change the dressing"]
    out = []
    for q in qs:
        raw = vq(query_text=q, top_k=9, audience="patient_safe"); red = retrieve_treatment_guidance(q)["passages"]   # same depth the tool searches
        rest = iter([p["source"] for p in raw]); same_order = all(src in rest for src in [p["source"] for p in red])   # kept passages keep their ranking
        clean = not any(find_drug_terms(p["text"]) for p in red)
        had = sum(bool(find_drug_terms(p["text"])) for p in raw)
        out.append((same_order, clean, 1 <= len(red) <= 3, had))
    ok = all(a and b and c for a, b, c, _ in out)
    return ok, f"per query (ranking kept, no drug names, 1-3 passages after the relevance filters, raw chunks that had drug names): {out}"
def d2():
    from rag.vector_store import _get_collection
    got = _get_collection().get(include=["documents", "metadatas"])
    ps = [{"id": i, "text": d} for i, d, m in zip(got["ids"], got["documents"], got["metadatas"]) if m["audience"] == "patient_safe"]
    altered = [c["id"] for c in ps if redact_drug_terms(c["text"]) != c["text"]]
    left = [c["id"] for c in ps if find_drug_terms(redact_drug_terms(c["text"]))]
    return not left, f"{len(altered)}/{len(ps)} patient_safe chunks altered: {altered}; chunks still naming a drug after redaction: {left}"

# ------------------------------------------------------------------ E. request size
def synth_turn(i):
    cid = f"c{i}"
    return [U(f"user message {i} " + "x" * 80),
            {"role": "assistant", "content": None, "tool_calls": [{"id": cid, "type": "function", "function": {"name": "retrieve_treatment_guidance", "arguments": json.dumps({"query": "q"})}}]},
            {"role": "tool", "tool_call_id": cid, "content": json.dumps({"passages": [{"text": "t" * 800, "source": f"S{i}"} for _ in range(3)]})},
            A(f"answer {i} " + "y" * 900)]
def synth_history(n):
    h = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "system", "content": "The case_id for this session is: case_x."}]
    for i in range(n): h += synth_turn(i)
    return h
def e1():
    hist = [S]
    steps = [{"tool_calls": [("check_image_quality", {"image_path": "data/sample_images/burnWound15.jpg"})]},
             {"tool_calls": [("classify_burn_severity", {"image_path": "data/sample_images/burnWound15.jpg"})]},
             {"tool_calls": [("segment_wound_area", {"image_path": "data/sample_images/burnWound15.jpg"})]},
             {"tool_calls": [("retrieve_treatment_guidance", {"query": "care for a second degree burn"})]}, {"content": "Your burn is second degree; see a clinician."}]
    hist.append(U("photo please"))
    run(steps, hist)
    before = C.estimate_tokens(hist); hist.append(U("what should I do next?")); saved = C.compact_previous_turns(hist, len(hist)); after = C.estimate_tokens(hist)
    tools = {}
    for m in hist:
        if m["role"] == "tool": tools[len(tools)] = json.loads(m["content"])
    passages_stub = "passages_omitted" in tools[3]; cls_keys = sorted(tools[1]); seg_keys = sorted(tools[2])
    again = C.compact_previous_turns(hist, len(hist))
    ok = passages_stub and cls_keys == sorted(C._CLASSIFY_KEYS) and seg_keys == sorted(C._SEGMENT_KEYS) and history_valid(hist) and after < before and again == 0
    return ok, f"tokens {before}->{after} (saved {saved} chars); retrieve->stub={passages_stub}; classify keeps {cls_keys}; segment keeps {seg_keys}; idempotent={again == 0}"
def e2():
    h = synth_history(9); h.append(U("current question"))
    b0 = C.estimate_tokens(h); C.compact_previous_turns(h, len(h)); b1 = C.estimate_tokens(h)
    st = new_state(); st["classify_result"] = {"predicted_severity": "2nd_degree", "calibrated_confidence": 0.99, "decision": "PROCEED"}; st["segment_result"] = {"wound_fraction": 0.247, "connected_components": 1}
    dropped = C.trim_old_turns(h, C.history_token_budget(), st); b2 = C.estimate_tokens(h)
    notes = [m for m in h if m["role"] == "system" and m["content"].startswith(C.NOTE_PREFIX)]
    texts = " ".join((m.get("content") or "") for m in h)
    ok = (history_valid(h) and len(notes) == 1 and "user message 8" in texts and "user message 0 " not in texts and h[-1]["content"] == "current question"
          and "2nd_degree" in notes[0]["content"] and "25%" in notes[0]["content"] and b2 <= C.history_token_budget())
    return ok, f"tokens {b0} -> {b1} (compaction) -> {b2} (trim, budget {C.history_token_budget()}); dropped {dropped} old turn(s); note kept once with facts; last exchange + current question preserved"
def e3():
    h = synth_history(9); h.append(U("current question"))
    r, st, cid, sc = run([{"content": "Understood."}], h)
    ok = r["final_answer"] == "Understood." and history_valid(h) and sc.sizes[0] <= C.history_token_budget() + 100
    return ok, f"request actually sent had ~{sc.sizes[0]} tokens of history (untrimmed it would be ~{C.estimate_tokens(synth_history(9))}); budget {C.history_token_budget()}"
def e4():
    cid = CS.create_new_case()
    cl = {"predicted_severity": "2nd_degree", "predicted_class_index": 1, "raw_confidence": .99, "calibrated_confidence": .98, "entropy": .07, "reliability": "HIGH", "decision": "PROCEED",
          "reliability_threshold": .9, "model": "EfficientNetV2-S", "calibration_temperature": 1.157, "all_class_probabilities": {"1st_degree": .01, "2nd_degree": .98, "3rd_degree": .01}}
    sg = {"wound_pixels": 7766, "image_pixels": 147456, "wound_fraction": 0.0527, "connected_components": 1, "mask_available": True}
    CS.save_baseline_result(cid, cl, sg); CS.add_followup_result(cid, dict(cl, predicted_severity="3rd_degree"), dict(sg, wound_fraction=0.644))
    case = CS.get_case(cid); full = len(json.dumps(case)); msg = C.build_followup_message(case)
    comp = C.compact_case_record(case)
    ok = len(json.dumps(comp)) < full * 0.6 and comp["followups"][0]["classify"]["predicted_severity"] == "3rd_degree" and comp["baseline_result"]["segment"]["wound_fraction"] == 0.0527
    return ok, f"case JSON {full} -> {len(json.dumps(comp))} chars, severity + wound_fraction kept for baseline and follow-up"

# ------------------------------------------------------------------ F. schema, G. tool-level email check
def f1():
    import jsonschema
    t = [x for x in TOOL_SCHEMAS if x["function"]["name"] == "generate_escalation_report"][0]["function"]["parameters"]
    jsonschema.Draft7Validator.check_schema(t)
    props = set(t["properties"]); want = {"case_id", "consent", "recipient_email", "triage_verdict", "baseline_predicted_severity", "baseline_wound_fraction", "trend_summary", "treatment_summary", "preview_only", "exclude_sections", "patient_notes",
                                        "predicted_severity", "calibrated_confidence", "wound_fraction", "gradcam_overlay_path", "guidance_passages", "clinics"}
    nullable_ok = all(isinstance(v["type"], list) and "null" in v["type"] for k, v in t["properties"].items() if k not in t["required"])
    total = len(json.dumps(TOOL_SCHEMAS))
    return props == want and t["required"] == ["case_id", "consent"] and nullable_ok and total < 7000, f"16 params (Phase 4: triage_verdict, preview_only, exclude_sections, patient_notes), required unchanged, optionals nullable, valid JSON-schema; all tool schemas {total} chars (was 6921 originally; 7000 cap)"
def g1():
    from tools.escalate import generate_escalation_report
    before = set(glob.glob("outputs/reports/*.pdf"))
    r = generate_escalation_report(case_id="case_x", consent=True, recipient_email="dr.smith@@hospital,com")
    after = set(glob.glob("outputs/reports/*.pdf"))
    return r.get("status") == "error" and "valid email" in r.get("message", "") and before == after, f"{r} new_pdfs={sorted(after - before)}"

CHECKS = [(n, f) for n, f in A_CASES] + [("A14 email validator: 4 valid / 10 invalid samples", a14), ("A15 reply interpreter: 11 typical replies read correctly", a15)]
CHECKS += [("B1  loop: 13C flow - report blocked, model asks; after 'Yes, I consent.' it runs once", lambda: with_recorder(b1)),
           ("B2  loop: malformed email never reaches the tool", lambda: with_recorder(b2)),
           ("B3  loop: withdrawn consent never reaches the tool", lambda: with_recorder(b3)),
           ("B4  loop: a genuinely consented request still runs", lambda: with_recorder(b4)),
           ("C1  guard: leak -> model rewrite accepted", c1), ("C2  guard: rewrite still dirty -> deterministic redaction", c2),
           ("C3  guard: rewrite call fails -> deterministic redaction (no crash)", c3), ("C4  guard: suspiciously short rewrite -> redaction", c4),
           ("C5  guard: clean reply -> no extra LLM call", c5), ("C6  guard: leaked draft is never stored in history", c6),
           ("D1  retrieval: same ranking, no drug names, 3 passages", d1), ("D2  retrieval: every patient_safe chunk clean after redaction", d2),
           ("E1  size: earlier-turn payloads compacted, facts kept, idempotent", e1), ("E2  size: old turns dropped to budget, last exchange + facts note kept", e2),
           ("E3  size: run_agent_turn sends a request within budget", e3), ("E4  size: follow-up case JSON compacted, trend facts kept", e4),
           ("F1  schema: report tool compact, valid, same params, optionals nullable", f1), ("G1  tool: escalate refuses a malformed email, no PDF", g1)]
for name, fn in CHECKS: check(name, fn)

if "--probe" in sys.argv:
    from dotenv import load_dotenv; load_dotenv()
    import groq; client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    L.client = client
    real = [t["answer"] for f in glob.glob("../burn_agent_archive/audit/_audit_results/t10.json") for t in json.load(open(f, encoding="utf-8"))["self_care_burnWound13#run1"].values() if isinstance(t, dict) and t.get("answer")][0]
    print("[PROBE] leaked reply terms:", find_medication_terms(real), flush=True)
    rewritten = L._rewrite_without_medications(real)
    ok = bool(rewritten) and not find_medication_terms(rewritten) and len(rewritten) >= 0.6 * len(real)
    RESULTS.append(("PROBE real model rewrites a real leaked reply cleanly", ok))
    print(f"[{'PASS' if ok else 'FAIL'}] PROBE rewrite by the real model: {len(real)} -> {len(rewritten or '')} chars, leftover terms {find_medication_terms(rewritten or '')}", flush=True)
    if rewritten:
        i = real.lower().find("pain"); j = rewritten.lower().find("pain")
        print("   BEFORE:", real[i:i+170].replace("\n", " ")); print("   AFTER :", rewritten[j:j+170].replace("\n", " "), flush=True)
    rep = [t for t in TOOL_SCHEMAS if t["function"]["name"] == "generate_escalation_report"][0]
    try:
        r = client.chat.completions.create(model=L.MODEL, temperature=0, max_tokens=800, tools=[rep], tool_choice={"type": "function", "function": {"name": "generate_escalation_report"}},
            messages=[{"role": "system", "content": "Schema test. Call the tool once, no explanation."}, {"role": "user", "content": "Call generate_escalation_report with case_id 'case_t' and consent true; set every optional parameter to null."}])
        tc = r.choices[0].message.tool_calls; ok2 = bool(tc)
        print(f"[{'PASS' if ok2 else 'FAIL'}] PROBE new report schema accepted by Groq: {tc[0].function.arguments[:160] if tc else None}", flush=True)
    except Exception as e:
        ok2 = False; print(f"[FAIL] PROBE new report schema REJECTED: {str(e)[:250]}", flush=True)
    RESULTS.append(("PROBE new report schema accepted by Groq", ok2))

fails = [n for n, ok in RESULTS if not ok]
print(f"\nSUMMARY: {len(RESULTS) - len(fails)}/{len(RESULTS)} passed" + (f" | FAILED: {fails}" if fails else ""), flush=True)

# ---- pytest entry: every check above already ran at import; each becomes one test.
import pytest


@pytest.mark.parametrize("name,ok,detail", ALL, ids=[a[0][:70] for a in ALL])
def test_check(name, ok, detail):
    assert ok, detail
