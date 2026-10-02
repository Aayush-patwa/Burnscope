"""Image quality, case store migration, segmentation bands and forced retrieval regression. Real modules."""
import sys, os, json, glob, re, tempfile, sqlite3, uuid
from types import SimpleNamespace
import cv2, numpy as np
getattr(sys.stdout, "reconfigure", lambda **k: None)(errors="replace")
R = []
ALL = []
def check(name, fn):
    try: ok, d = fn()
    except Exception as e: ok, d = False, f"EXCEPTION {type(e).__name__}: {str(e)[:200]}"
    R.append(ok); ALL.append((name, ok, d)); print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {d}", flush=True)

# ---- 1. image quality (pure)
from tools.quality_check import check_image_quality
def nat(p):
    m = re.search(r"burnWound(\d*)\.jpg", p); return int(m.group(1)) if m and m.group(1) else 1
def q_samples():
    res = {os.path.basename(p): check_image_quality(p) for p in sorted(glob.glob("data/sample_images/*.jpg"), key=nat)}
    bad = {k: v["issue"] for k, v in res.items() if v["verdict"] != "acceptable"}
    return not bad, f"{len(res) - len(bad)}/{len(res)} real sample photos accepted; rejected: {bad}"
def q_synthetic():
    img = cv2.imread("data/sample_images/burnWound16.jpg"); d = tempfile.mkdtemp(); out = {}
    cases = {"blurred": cv2.GaussianBlur(img, (31, 31), 0), "dark": (img * 0.15).astype(np.uint8), "overexposed": cv2.add(img, (170, 170, 170, 0)),
             "flat gray": np.full_like(img, 128), "blue (no skin)": np.dstack([np.full(img.shape[:2], 200, np.uint8), np.full(img.shape[:2], 60, np.uint8), np.full(img.shape[:2], 30, np.uint8)]),
             "tiny 50x50": cv2.resize(img, (50, 50))}
    for k, im in cases.items():
        p = os.path.join(d, k.split()[0] + ".jpg"); cv2.imwrite(p, im); out[k] = check_image_quality(p)
    corrupt = os.path.join(d, "c.jpg"); open(corrupt, "wb").write(os.urandom(3000)); out["corrupt file"] = check_image_quality(corrupt)
    txt = os.path.join(d, "n.txt"); open(txt, "w").write("hello"); out["text file"] = check_image_quality(txt)
    out["missing path"] = check_image_quality("no/such/file.jpg")
    wrong = [k for k, v in out.items() if v["verdict"] != "retake_needed"]
    return not wrong, f"all 9 bad inputs rejected (wrongly accepted: {wrong}); e.g. blurred -> {out['blurred']['issue'][:60]!r}; corrupt -> {out['corrupt file']['issue'][:50]!r}"

# ---- 2. scheduler + case store (temp DB, including an OLD-schema database)
import agent.case_store as CS
def store():
    path = os.path.join(tempfile.mkdtemp(), "old.db"); CS.DB_PATH = path
    conn = sqlite3.connect(path)  # legacy schema: no next_followup column
    conn.execute("CREATE TABLE cases (case_id TEXT PRIMARY KEY, created_date TEXT, baseline_result TEXT, followups TEXT)")
    conn.execute("INSERT INTO cases VALUES ('case_old00001', date('now'), ?, '[]')", (json.dumps({"classify": {"predicted_severity": "2nd_degree"}, "segment": None}),))
    conn.commit(); conn.close()
    CS.init_db()
    cols = [r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(cases)")]
    from tools.scheduler import schedule_followup
    cid = CS.create_new_case(); CS.save_baseline_result(cid, {"predicted_severity": "3rd_degree"}, {"wound_fraction": .5})
    r_ok = schedule_followup(cid, 2); r_bad = schedule_followup(cid, 99); r_unknown = schedule_followup("case_nope", 2); r_str = schedule_followup(cid, "2")
    got = CS.get_case(cid)["next_followup"]; listed = CS.list_active_cases()
    empty = CS.create_new_case()   # no baseline yet -> must not be listed
    ids = [c["case_id"] for c in CS.list_active_cases()]
    ok = ("next_followup" in cols and r_ok.get("saved") is True and got and got["due"] == r_ok["scheduled_date"] and "error" in r_bad and "error" in r_unknown
          and "error" in r_str and empty not in ids and "case_old00001" in ids and any(c["case_id"] == cid and c["next_followup"] == r_ok["scheduled_date"] for c in listed))
    return ok, f"old DB migrated={('next_followup' in cols)}; saved due={got and got['due']}; interval 99 -> {r_bad['error'][:35]!r}; unknown case -> error; case with no baseline hidden={empty not in ids}; listed={[(c['case_id'][-4:], c['baseline_severity'], c['next_followup']) for c in listed]}"

# ---- 3. segmentation fields (real model)
def seg():
    from tools.segment import segment_wound_area
    rows = {}
    for n in ("burnWound3", "burnWound12", "burnWound15", "burnWound4", "burnWound17"):
        r = segment_wound_area(f"data/sample_images/{n}.jpg")
        rows[n] = (round(r["wound_fraction"], 3), r["connected_components"], r["significant_regions"], r["area_band"], r["multi_region"])
    ok = rows["burnWound12"][3] == "large" and rows["burnWound17"][3] == "small" and all(len(v) == 5 for v in rows.values())
    return ok, "(fraction, raw components, significant regions, band, multi_region): " + str(rows)

# ---- 4. forced retrieval (scripted LLM, real retrieval)
import agent.loop as L
def resp(step):
    tcs = [SimpleNamespace(id=f"c{i}{uuid.uuid4().hex[:5]}", type="function", function=SimpleNamespace(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(step.get("tool_calls", []))]
    m = SimpleNamespace(content=step.get("content"), tool_calls=tcs or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=m, finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))
class Scr:
    def __init__(self, steps): self.steps, self.calls = list(steps), 0
    def create(self, **kw): self.calls += 1; return resp(self.steps.pop(0))
def turn(steps):
    sc = Scr(steps); L.client.chat.completions.create = sc.create
    h = [{"role": "system", "content": "sys"}, {"role": "user", "content": "what should I do about the burn?"}]
    st = {"classify_result": {"predicted_severity": "2nd_degree"}, "segment_result": None, "gradcam_result": None, "guidance_result": None, "clinics_result": None,
          "locked_image_path": None, "baseline_saved": True, "is_followup": False, "followup_saved": False}
    r = L.run_agent_turn(h, "case_x", st); return r, h, sc
ADVICE = "Keep the wound clean, rinse with cool running water and cover it with a non-adhesive dressing."
def forced():
    r, h, sc = turn([{"content": ADVICE}, {"content": "Per the retrieved guidance: see a clinician and keep the area covered."}])
    tools = [json.loads(m["content"]) for m in h if m["role"] == "tool"]
    ok = sc.calls == 2 and len(tools) == 1 and len(tools[0].get("passages", [])) == 3 and "instruction" in tools[0] and r["final_answer"].startswith("Per the retrieved")
    return ok, f"calls={sc.calls}; injected retrieval passages={len(tools[0].get('passages', [])) if tools else 0}; final answer is the grounded rewrite={r['final_answer'][:30]!r}; history tool-call pairs valid={len([m for m in h if m.get('tool_calls')]) == len(tools)}"
def not_forced_without_advice():
    r, h, sc = turn([{"content": "Could you tell me your approximate location so I can look up clinics?"}])
    return sc.calls == 1 and not any(m["role"] == "tool" for m in h), f"calls={sc.calls} (no retrieval forced for a non-advice reply)"
def not_forced_when_model_retrieved():
    r, h, sc = turn([{"tool_calls": [("retrieve_treatment_guidance", {"query": "burn care"})]}, {"content": ADVICE}])
    tools = [m for m in h if m["role"] == "tool"]
    return sc.calls == 2 and len(tools) == 1 and r["final_answer"] == ADVICE, f"calls={sc.calls}; only the model's own retrieval ran ({len(tools)} tool result); answer kept as written"

# ---- 5. forced segmentation (scripted LLM, real segmentation model) -- a 2nd/3rd-degree reading must
# not be finalized without wound-area data (system prompt step 3); this is the code-level safety net.
def turn2(severity, segment_result, steps, locked="data/sample_images/burnWound3.jpg"):
    sc = Scr(steps); L.client.chat.completions.create = sc.create
    h = [{"role": "system", "content": "sys"}, {"role": "user", "content": "what should I do about the burn?"}]
    st = {"classify_result": {"predicted_severity": severity} if severity else None, "segment_result": segment_result,
          "gradcam_result": None, "guidance_result": None, "clinics_result": None, "locked_image_path": locked,
          "baseline_saved": True, "is_followup": False, "followup_saved": False}
    r = L.run_agent_turn(h, "case_x", st); return r, h, sc, st
def segment_forced():
    r, h, sc, st = turn2("2nd_degree", None, [{"content": ADVICE}, {"tool_calls": [("retrieve_treatment_guidance", {"query": "burn care"})]}, {"content": "Per the retrieved guidance: " + ADVICE}])
    calls = [c["function"]["name"] for m in h if m.get("tool_calls") for c in m["tool_calls"]]
    seg_tool_msgs = [json.loads(m["content"]) for i, m in enumerate(h) if m.get("role") == "tool" and h[i - 1]["tool_calls"][0]["function"]["name"] == "segment_wound_area"]
    ok = (sc.calls == 3 and calls == ["segment_wound_area", "retrieve_treatment_guidance"] and st["segment_result"] is not None
          and "area_band" in st["segment_result"] and len(seg_tool_msgs) == 1 and "instruction" in seg_tool_msgs[0]
          and r["final_answer"] == "Per the retrieved guidance: " + ADVICE)
    return ok, f"calls={sc.calls}; tool order={calls}; segment_result area_band={st['segment_result'] and st['segment_result'].get('area_band')}; instruction present in injected result={bool(seg_tool_msgs and 'instruction' in seg_tool_msgs[0])}"
def _no_segment_call(h):
    return not any(c["function"]["name"] == "segment_wound_area" for m in h if m.get("tool_calls") for c in m["tool_calls"])
def not_forced_1st_degree():
    # ADVICE still lacks a retrieval, so that guard fires regardless (a separate, already-tested behaviour);
    # what this checks is that segmentation specifically is never forced for an optional 1st-degree reading.
    r, h, sc, st = turn2("1st_degree", None, [{"content": ADVICE}, {"content": ADVICE}])
    return _no_segment_call(h) and st["segment_result"] is None, f"tool calls={[c['function']['name'] for m in h if m.get('tool_calls') for c in m['tool_calls']]}; segment_result stays None for 1st-degree (optional per the prompt)"
def not_forced_when_already_segmented():
    r, h, sc, st = turn2("3rd_degree", {"area_band": "large", "wound_fraction": 0.5, "multi_region": False}, [{"content": ADVICE}, {"content": ADVICE}])
    return _no_segment_call(h) and st["segment_result"]["area_band"] == "large", f"tool calls={[c['function']['name'] for m in h if m.get('tool_calls') for c in m['tool_calls']]}; no forced call when segment_result already present"
def not_forced_without_locked_image():
    r, h, sc, st = turn2("2nd_degree", None, [{"content": ADVICE}, {"content": ADVICE}], locked=None)
    return _no_segment_call(h) and st["segment_result"] is None, f"tool calls={[c['function']['name'] for m in h if m.get('tool_calls') for c in m['tool_calls']]}; no forced call with no locked image to segment"

for n, f in [("Q1 quality: all 18 real sample photos accepted", q_samples), ("Q2 quality: 9 kinds of bad input rejected with a plain instruction", q_synthetic),
             ("S1 store: old DB migrated; follow-up saved + validated; case list", store), ("G1 segment: area_band / regions returned", seg),
             ("R1 loop: care advice without retrieval -> retrieval forced, grounded rewrite", forced),
             ("R2 loop: non-advice reply not forced", not_forced_without_advice), ("R3 loop: model's own retrieval respected", not_forced_when_model_retrieved),
             ("F1 loop: 2nd/3rd-degree reading without segmentation -> segmentation forced before finalizing", segment_forced),
             ("F2 loop: 1st-degree reading -> segmentation not forced (optional)", not_forced_1st_degree),
             ("F3 loop: segmentation already present -> not forced again", not_forced_when_already_segmented),
             ("F4 loop: no locked image -> nothing to segment, not forced", not_forced_without_locked_image)]:
    check(n, f)
print(f"\nSUMMARY: {sum(R)}/{len(R)} passed", flush=True)

# ---- 6. forced segmentation also fires on a stated/hedged VERDICT, not just explicit care-advice
# wording -- this is the exact real-audit failure: "I'll move forward with a clinical-review
# recommendation" names a verdict but contains none of _gives_care_advice()'s trigger words.
HEDGE = ("The burn-severity model wasn't confident enough to give a reliable grade for this image. "
        "Could you take another, clearer photo? If that isn't practical, I'll move forward with a "
        "clinical-review recommendation as a precaution.")
def segment_forced_on_verdict_hedge():
    r, h, sc, st = turn2("2nd_degree", None, [{"content": HEDGE}, {"content": "Understood, I will wait for a clearer photo."}])
    calls = [c["function"]["name"] for m in h if m.get("tool_calls") for c in m["tool_calls"]]
    ok = "segment_wound_area" in calls and st["segment_result"] is not None and sc.calls == 2
    return ok, f"tool calls={calls}; segment_result={st['segment_result'] and st['segment_result'].get('area_band')}"
def no_verdict_no_advice_not_forced():
    # A genuine clarifying question with neither care advice nor a stated verdict must not force anything.
    r, h, sc, st = turn2("2nd_degree", None, [{"content": "How long ago did this happen, and is there any blistering?"}])
    return _no_segment_call(h) and st["segment_result"] is None and sc.calls == 1, f"calls={sc.calls}; no forced call for a plain clarifying question"

for n, f in [("F5 loop: forced segmentation also fires on a stated verdict hedge, not just care-advice wording", segment_forced_on_verdict_hedge),
             ("F6 loop: a plain clarifying question forces nothing", no_verdict_no_advice_not_forced)]:
    check(n, f)



# ---- 7. repeated-retrieval nudge and a real-data fallback for the tool-call safety cap (found live
# during the final audit: the model called retrieve_treatment_guidance 4 times in one turn with reworded
# queries, exhausted the tool-call budget, and the raw internal "[Agent stopped...]" string reached the user)
def turn_capped(case_state_extra=None, tool_name="retrieve_treatment_guidance", args_fn=None):
    steps = [{"tool_calls": [(tool_name, (args_fn or (lambda i: {"query": "burn care guidance"}))(i))]} for i in range(L.MAX_TOOL_CALLS_PER_TURN)]
    sc = Scr(steps); L.client.chat.completions.create = sc.create
    h = [{"role": "system", "content": "sys"}, {"role": "user", "content": "what should I do about the burn?"}]
    st = {"classify_result": None, "segment_result": None, "gradcam_result": None, "guidance_result": None, "clinics_result": None,
          "locked_image_path": None, "baseline_saved": True, "is_followup": False, "followup_saved": False}
    if case_state_extra:
        st.update(case_state_extra)
    r = L.run_agent_turn(h, "case_x", st)
    return r, h, sc

def retrieval_nudge_after_repeated_calls():
    r, h, sc = turn_capped()
    tool_msgs = [json.loads(m["content"]) for m in h if m["role"] == "tool"]
    notes = [i for i, t in enumerate(tool_msgs) if "stop_hint" in t]
    ok = len(tool_msgs) == L.MAX_TOOL_CALLS_PER_TURN and notes and notes[0] == L.MAX_RETRIEVALS_PER_TURN - 1
    return ok, f"{len(tool_msgs)} retrieval results; nudge first appears at call #{notes[0] + 1 if notes else None} (expected #{L.MAX_RETRIEVALS_PER_TURN})"

def tool_cap_falls_back_to_real_case_facts():
    r, h, sc = turn_capped(case_state_extra={
        "classify_result": {"predicted_severity": "3rd_degree", "decision": "PROCEED"},
        "segment_result": {"area_band": "large", "multi_region": False, "wound_fraction": 0.5}})
    ok = (r["error"] == "tool_call_cap" and "Agent stopped" not in r["final_answer"]
          and "3rd degree" in r["final_answer"].lower() and "urgent" in r["final_answer"].lower())
    return ok, f"error={r['error']}; final_answer={r['final_answer']!r}"

def tool_cap_generic_message_when_nothing_is_known():
    r, h, sc = turn_capped()   # no classify_result at all
    return r["error"] == "tool_call_cap" and r["final_answer"] == L.MSG_TOOL_CAP, f"final_answer={r['final_answer']!r}"

for n, f in [("F7 loop: repeated retrieve_treatment_guidance calls get a stop-and-answer nudge", retrieval_nudge_after_repeated_calls),
             ("F8 loop: hitting the tool-call cap falls back to real case facts, not raw internal text", tool_cap_falls_back_to_real_case_facts),
             ("F9 loop: with nothing known yet, the cap fallback stays the safe generic message", tool_cap_generic_message_when_nothing_is_known)]:
    check(n, f)


# ---- pytest entry: every check above already ran at import; each becomes one test.
import pytest


@pytest.mark.parametrize("name,ok,detail", ALL, ids=[a[0][:70] for a in ALL])
def test_check(name, ok, detail):
    assert ok, detail
