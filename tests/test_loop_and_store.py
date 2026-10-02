"""Agent loop + case store regression (H2/H3/H4/H6, follow-up persistence, retries).
REAL tools and DB code, a SCRIPTED stand-in for the LLM (no Groq quota), temp SQLite file (never data/cases.db).
`python tests/test_loop_and_store.py --probe` adds a tiny real Groq schema probe (uses quota)."""
import sys, os, json, tempfile, copy, traceback, uuid
from types import SimpleNamespace
getattr(sys.stdout, "reconfigure", lambda **k: None)(errors="replace")
import httpx, groq
import agent.case_store as CS
CS.DB_PATH = os.path.join(tempfile.mkdtemp(), "verify_cases.db")
CS.init_db()
import agent.loop as L
from agent.dispatch import execute_tool, TOOL_DISPATCH
from agent.schemas import TOOL_SCHEMAS

IMG = "data/sample_images/"
RESULTS = []
ALL = []
def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"EXCEPTION {type(e).__name__}: {str(e)[:160]}"
    RESULTS.append((name, ok)); ALL.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)

# ------------------------------------------------------------------ scripted LLM
def make_resp(step):
    tcs = [SimpleNamespace(id=f"call_{i}_{uuid.uuid4().hex[:6]}", type="function",
                           function=SimpleNamespace(name=n, arguments=a if isinstance(a, str) else json.dumps(a)))
           for i, (n, a) in enumerate(step.get("tool_calls", []))]
    msg = SimpleNamespace(content=step.get("content"), tool_calls=tcs or None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tcs else "stop")],
                           usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

class Scripted:
    def __init__(self, steps): self.steps, self.calls = list(steps), 0
    def create(self, **kw):
        self.calls += 1
        step = self.steps.pop(0)
        if isinstance(step, BaseException): raise step
        return make_resp(step)

def http_err(cls, status, text):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls(text, response=httpx.Response(status, request=req, text=text), body=None)

def conn_err():
    return groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x"))

def history_valid(h):
    i = 0
    while i < len(h):
        m = h[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = {c["id"] for c in m["tool_calls"]}
            j, got = i + 1, set()
            while j < len(h) and h[j]["role"] == "tool":
                got.add(h[j]["tool_call_id"]); j += 1
            if got != ids: return False
            i = j; continue
        i += 1
    return True

def new_state(followup=False):
    return {"classify_result": None, "segment_result": None, "gradcam_result": None, "guidance_result": None,
            "clinics_result": None, "locked_image_path": None, "baseline_saved": followup,
            "is_followup": followup, "followup_saved": False}

def run(steps, state=None, history=None, case_id=None):
    sc = Scripted(steps)
    L.client.chat.completions.create = sc.create
    real_time = L.time; L.time = SimpleNamespace(sleep=lambda s: None)
    try:
        h = history if history is not None else [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        cid = case_id or CS.create_new_case()
        st = state or new_state()
        res = L.run_agent_turn(h, cid, st)
        return res, h, st, cid, sc
    finally:
        L.time = real_time

# ------------------------------------------------------------------ C1 dispatch
corrupt = os.path.join(tempfile.gettempdir(), "verify_corrupt.jpg")
open(corrupt, "wb").write(os.urandom(2048))

def c1a():
    r = execute_tool("classify_burn_severity", {"image_path": corrupt})
    return isinstance(r, dict) and "error" in r, str(r)[:130]
def c1b():
    r = execute_tool("segment_wound_area", {"image_path": "does/not/exist.jpg"})
    return isinstance(r, dict) and "error" in r, str(r)[:130]
def c1c():
    r = execute_tool("no_such_tool", {}); r2 = execute_tool("check_image_quality", {"bogus": 1})
    return "error" in r and "error" in r2, f"{r} | {str(r2)[:80]}"
def c1d():
    orig = TOOL_DISPATCH["schedule_followup"]
    TOOL_DISPATCH["schedule_followup"] = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    try: r = execute_tool("schedule_followup", {"case_id": "x", "interval_days": 2})
    finally: TOOL_DISPATCH["schedule_followup"] = orig
    return isinstance(r, dict) and "error" in r, str(r)[:130]

# ------------------------------------------------------------------ C2 failed tool result is not persisted, turn survives
def c2():
    steps = [{"tool_calls": [("check_image_quality", {"image_path": corrupt}), ("classify_burn_severity", {"image_path": corrupt})]},
             {"content": "The analysis step failed; please retry."}]
    res, h, st, cid, sc = run(steps)
    base = CS.get_case(cid)["baseline_result"]
    ok = (res["final_answer"].startswith("The analysis step failed") and st["classify_result"] is None
          and base == {} and history_valid(h))
    return ok, f"answer ok={res['final_answer'][:30]!r} classify_result={st['classify_result']} baseline={base} hist_valid={history_valid(h)}"

# ------------------------------------------------------------------ C3 persistence of wound_fraction
def steps_for(img, seg=True):
    s = [{"tool_calls": [("check_image_quality", {"image_path": img})]},
         {"tool_calls": [("classify_burn_severity", {"image_path": img})]}]
    if seg: s.append({"tool_calls": [("segment_wound_area", {"image_path": img})]})
    s.append({"content": "done"})
    return s

def c3a():
    res, h, st, cid, sc = run(steps_for(IMG + "burnWound17.jpg"))
    b = CS.get_case(cid)["baseline_result"]
    wf = (b.get("segment") or {}).get("wound_fraction")
    return wf is not None and abs(wf - 0.0527) < 0.01, f"baseline wound_fraction={wf} (expect ~0.0527) sev={b.get('classify',{}).get('predicted_severity')}"
def c3b():
    img = IMG + "burnWound17.jpg"
    steps = [{"tool_calls": [("check_image_quality", {"image_path": img}), ("classify_burn_severity", {"image_path": img}),
                             ("segment_wound_area", {"image_path": img})]}, {"content": "done"}]
    res, h, st, cid, sc = run(steps)
    wf = (CS.get_case(cid)["baseline_result"].get("segment") or {}).get("wound_fraction")
    return wf is not None, f"same-batch baseline wound_fraction={wf}"
def c3c():
    cid = CS.create_new_case()
    CS.save_baseline_result(cid, {"predicted_severity": "2nd_degree"}, {"wound_fraction": 0.0527, "connected_components": 1})
    res, h, st, cid, sc = run(steps_for(IMG + "burnWound11.jpg"), state=new_state(followup=True), case_id=cid)
    fu = CS.get_case(cid)["followups"]
    wf = (fu[0].get("segment") or {}).get("wound_fraction") if fu else "no-followup"
    return isinstance(wf, float) and abs(wf - 0.6443) < 0.01, f"follow-up wound_fraction={wf} (expect ~0.6443) n_followups={len(fu)}"
def c3d():
    hist = [{"role": "system", "content": "sys"}, {"role": "user", "content": "photo A"}]
    st = new_state(); cid = CS.create_new_case()
    run(steps_for(IMG + "burnWound13.jpg", seg=False), state=st, history=hist, case_id=cid)
    hist.append({"role": "user", "content": "photo B"})
    run(steps_for(IMG + "burnWound15.jpg", seg=True), state=st, history=hist, case_id=cid)
    b = CS.get_case(cid)["baseline_result"]
    return b["classify"]["predicted_severity"] == "1st_degree" and b.get("segment") is None, \
        f"image-A record kept: sev={b['classify']['predicted_severity']} segment={b.get('segment')} (must stay None: B's area is not A's)"

# ------------------------------------------------------------------ C4 API / service errors
def api_case(steps, expect_answer_contains=None, expect_calls=None, expect_ok_text=None):
    res, h, st, cid, sc = run(steps)
    ans = res.get("final_answer") or ""
    good = history_valid(h)
    if expect_ok_text is not None: good = good and ans == expect_ok_text
    if expect_answer_contains: good = good and any(k in ans.lower() for k in expect_answer_contains)
    if expect_calls is not None: good = good and sc.calls == expect_calls
    return good, f"calls={sc.calls} err={res.get('error')} hist_valid={history_valid(h)} answer={ans[:110]!r}"

E413 = lambda: http_err(groq.APIStatusError, 413, "Request too large for model on tokens per minute (TPM): Limit 8000, Requested 8029")
E429M = lambda: http_err(groq.RateLimitError, 429, "Rate limit reached ... on tokens per minute (TPM): Limit 8000")
E429D = lambda: http_err(groq.RateLimitError, 429, "Rate limit reached ... on tokens per day (TPD): Limit 200000")
E400T = lambda: http_err(groq.BadRequestError, 400, "Tool call validation failed ... code tool_use_failed")
E400 = lambda: http_err(groq.BadRequestError, 400, "invalid request: something else")
E401 = lambda: http_err(groq.AuthenticationError, 401, "Invalid API Key")
E503 = lambda: http_err(groq.InternalServerError, 503, "service unavailable")
OK = {"content": "ok"}

def c4_bug():
    steps = [{"tool_calls": [("check_image_quality", {"image_path": corrupt})]}, RuntimeError("simulated bug"), ]
    res, h, st, cid, sc = run(steps)
    return history_valid(h) and bool(res.get("final_answer")) and h[-1]["role"] == "assistant" and h[-2]["role"] != "tool", \
        f"hist_valid={history_valid(h)} last_roles={[m['role'] for m in h[-3:]]} answer={str(res.get('final_answer'))[:80]!r}"

# ------------------------------------------------------------------ C5 null args stripped
def c5():
    steps = [{"tool_calls": [("retrieve_treatment_guidance", {"query": "first aid for a minor burn", "top_k": None})]}, {"content": "done"}]
    res, h, st, cid, sc = run(steps)
    tool_msgs = [json.loads(m["content"]) for m in h if m["role"] == "tool"]
    n = len((tool_msgs[0].get("passages") or [])) if tool_msgs else -1
    return n == 3, f"passages returned with top_k=null -> {n} (expect 3 via default) tool_result_keys={list(tool_msgs[0])[:3] if tool_msgs else None}"

# ------------------------------------------------------------------ C6 schema nullability
def c6a():
    bad = []
    for t in TOOL_SCHEMAS:
        p = t["function"]["parameters"]; req = set(p.get("required", []))
        for k, v in p["properties"].items():
            ty = v.get("type"); nullable = isinstance(ty, list) and "null" in ty
            if k in req and nullable: bad.append(f"{t['function']['name']}.{k} required-but-nullable")
            if k not in req and not nullable: bad.append(f"{t['function']['name']}.{k}")
    return not bad, f"{len(bad)} optional params not nullable: {bad[:6]}"
def c6b():
    import jsonschema
    p = [t for t in TOOL_SCHEMAS if t["function"]["name"] == "generate_escalation_report"][0]["function"]["parameters"]
    args = {"case_id": "case_x", "consent": True, "recipient_email": None, "baseline_predicted_severity": None,
            "baseline_wound_fraction": None, "trend_summary": None, "treatment_summary": None, "predicted_severity": "2nd_degree",
            "calibrated_confidence": 0.9, "wound_fraction": None, "gradcam_overlay_path": None, "guidance_passages": None, "clinics": None}
    v = jsonschema.Draft7Validator(p); errs = [e.message[:60] for e in v.iter_errors(args)]
    missing = jsonschema.Draft7Validator(p).is_valid({"case_id": "x"})
    return not errs and not missing, f"null-optional args validate: errors={errs[:3]}; missing required consent rejected={not missing}"

CHECKS = [
    ("C1a dispatch: corrupt image -> error dict, no raise", c1a),
    ("C1b dispatch: missing file -> error dict, no raise", c1b),
    ("C1c dispatch: unknown tool / bad args -> error dict", c1c),
    ("C1d dispatch: any tool exception -> error dict", c1d),
    ("C2  loop: failed tool does not crash turn or get saved as baseline", c2),
    ("C3a persist: baseline wound_fraction saved (classify and segment in separate LLM steps)", c3a),
    ("C3b persist: baseline wound_fraction saved (same batch)", c3b),
    ("C3c persist: follow-up wound_fraction saved", c3c),
    ("C3d persist: 2nd image's area is NOT written onto 1st image's record", c3d),
    ("C4a api: 413 too large -> graceful message, history valid", lambda: api_case([E413()], ["too long", "start a new"], 1)),
    ("C4b api: per-minute 429 x4 -> graceful after 4 retries", lambda: api_case([E429M(), E429M(), E429M(), E429M()], ["busy", "try again"], 4)),
    ("C4b2 api: daily-cap 429 -> graceful immediately (no useless retries)", lambda: api_case([E429D()], ["usage limit", "later"], 1)),
    ("C4c api: per-minute 429 then success -> retried transparently", lambda: api_case([E429M(), OK], expect_ok_text="ok", expect_calls=2)),
    ("C4d api: 400 tool_use_failed x2 then success -> retried", lambda: api_case([E400T(), E400T(), OK], expect_ok_text="ok", expect_calls=3)),
    ("C4e api: 400 tool_use_failed forever -> graceful", lambda: api_case([E400T()] * 4, ["try again", "rephrase", "trouble"], 4)),
    ("C4f api: other 400 -> graceful, no pointless retry", lambda: api_case([E400()], ["couldn't", "could not", "problem", "try again"], 1)),
    ("C4g api: connection error then success -> retried", lambda: api_case([conn_err(), OK], expect_ok_text="ok", expect_calls=2)),
    ("C4h api: 401 auth -> graceful, no retry", lambda: api_case([E401()], ["configuration", "credential", "problem", "api key"], 1)),
    ("C4i api: 503 x2 then success -> retried", lambda: api_case([E503(), E503(), OK], expect_ok_text="ok", expect_calls=3)),
    ("C4j loop: unexpected internal exception -> graceful + history rolled back", c4_bug),
    ("C4k api: empty model reply -> graceful fallback", lambda: api_case([{"content": None}], ["try again", "rephrase", "couldn't", "could not"], 1)),
    ("C5  loop: null optional tool args are stripped (top_k=null)", c5),
    ("C6a schema: every optional param nullable, required ones untouched", c6a),
    ("C6b schema: null-filled report args validate; missing required rejected", c6b),
]
for name, fn in CHECKS:
    check(name, fn)

if "--probe" in sys.argv:
    from dotenv import load_dotenv; load_dotenv()
    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    rep = copy.deepcopy([t for t in TOOL_SCHEMAS if t["function"]["name"] == "generate_escalation_report"][0])
    old = copy.deepcopy(rep)
    for k, v in old["function"]["parameters"]["properties"].items():
        if isinstance(v.get("type"), list): v["type"] = [t for t in v["type"] if t != "null"][0]
    def probe(label, schema):
        try:
            r = client.chat.completions.create(
                model=L.MODEL, temperature=0, max_tokens=1500, tools=[schema],
                tool_choice={"type": "function", "function": {"name": "generate_escalation_report"}},
                messages=[{"role": "system", "content": "You are a schema test. Call the tool once. Do not explain."},
                          {"role": "user", "content": "Call generate_escalation_report with case_id 'case_test0001' and consent true, and set EVERY optional parameter explicitly to null."}])
            tc = r.choices[0].message.tool_calls
            print(f"[PROBE {label}] accepted; tool_call args = {tc[0].function.arguments[:300] if tc else None}", flush=True)
            return True
        except Exception as e:
            print(f"[PROBE {label}] REJECTED {type(e).__name__}: {str(e)[:260]}", flush=True)
            return False
    probe("OLD non-nullable schema (control)", old)
    ok_new = probe("CURRENT schema", rep)
    RESULTS.append(("PROBE: current schema accepted by Groq with null optionals", ok_new))

fails = [n for n, ok in RESULTS if not ok]
print(f"\nSUMMARY: {len(RESULTS) - len(fails)}/{len(RESULTS)} passed" + (f" | FAILED: {len(fails)}" if fails else ""), flush=True)

# ---- pytest entry: every check above already ran at import; each becomes one test.
import pytest


@pytest.mark.parametrize("name,ok,detail", ALL, ids=[a[0][:70] for a in ALL])
def test_check(name, ok, detail):
    assert ok, detail
