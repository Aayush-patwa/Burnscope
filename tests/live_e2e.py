"""
LIVE end-to-end check (real Groq, real models, real email). Not collected by pytest (name does not start with test_).
Uses a temporary database and folders, so it never touches data/cases.db. The "doctor" it emails is the app's own
sender address (EMAIL_ADDRESS in .env), so nothing goes to a stranger.

    python tests/live_e2e.py [--skip-followup] [--report-only]

Costs Groq quota (roughly 60-90k tokens).
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
getattr(sys.stdout, "reconfigure", lambda **k: None)(errors="replace")

from dotenv import load_dotenv

load_dotenv(".env")

import agent.case_store as CS
import agent.service as service
import agent.user_store as US
import agent.userctx as UC

tmp = tempfile.mkdtemp(prefix="burnscope_live_")
CS.DB_PATH = os.path.join(tmp, "live.db")
roots = {"uploads": os.path.join(tmp, "data_users"), "gradcam": os.path.join(tmp, "out_users"), "reports": os.path.join(tmp, "out_users")}
UC._KIND_ROOTS.update(roots)
CS.init_db()

DOCTOR = os.environ["EMAIL_ADDRESS"]
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    print(f"   [{'PASS' if ok else 'FAIL'}] {name} {('- ' + str(detail)) if detail else ''}", flush=True)


def turn(session, label, fn, *args):
    seen = {c["id"] for m in session.history if m.get("tool_calls") for c in m["tool_calls"]}   # (history may be trimmed during a turn)
    t0 = time.time()
    out = fn(*args)
    tools = [c["function"]["name"] for m in session.history if m.get("tool_calls") for c in m["tool_calls"] if c["id"] not in seen]
    print(f"\n>> {label}  ({time.time() - t0:.0f}s)  tools={tools}  error={out['error']}  guard={out['guard']}")
    print("   " + out["answer"][:600].replace("\n", "\n   "), flush=True)
    return out, tools


# The tester's own address is the app's sender, so the calendar invite and reminder land in YOUR inbox.
user = US.register("Live Tester", DOCTOR, "live-pass-1")
s = service.Session.start(user, "new")
print(f"user={user['user_id']}  case={s.case_id}  temp={tmp}")

def photo_flow(session, label, sample):
    """Upload a photo; the agent asks for body part and age first (by design), so answer like a user would."""
    out, tools = turn(session, label, session.send_sample, sample)
    if session.case_state["classify_result"] is None:
        out2, tools2 = turn(session, label + " (answering the agent's questions)", session.send,
                            "It is on my forearm, and I am a 28 year old adult.")
        out, tools = out2, tools + tools2
    return out, tools


REPORT_ONLY = "--report-only" in sys.argv
if REPORT_ONLY:
    # Cheap mode: fill the case with the REAL tools directly (no model turns), then test only the chat report flow.
    from agent.dispatch import execute_tool
    with UC.acting_as(user["user_id"]):
        cs = s.case_state
        cs["locked_image_path"] = path = s.save_sample("burnWound11.jpg")
        cs["quality_result"] = execute_tool("check_image_quality", {"image_path": path})
        cs["classify_result"] = execute_tool("classify_burn_severity", {"image_path": path})
        cs["segment_result"] = execute_tool("segment_wound_area", {"image_path": path})
        cs["guidance_result"] = execute_tool("retrieve_treatment_guidance", {"query": "first aid for a burn on the forearm, adult"})
        cs["clinics_result"] = execute_tool("locate_nearby_clinics", {"location": "Vijay Nagar, Indore"})
        CS.save_baseline_result(s.case_id, cs["classify_result"], cs["segment_result"])
        print("seeded:", cs["classify_result"]["predicted_severity"], cs["segment_result"]["area_band"], len(cs["clinics_result"].get("clinics", [])), "clinics")
        print("follow-up:", execute_tool("schedule_followup", {"case_id": s.case_id, "interval_days": 2}))
    s._ensure_gradcam()
    # ...and the conversation a real session would already have, with real tool-call / tool-result messages
    # (payloads shrunk the same way earlier turns are), so the model knows this case is already analysed.
    from agent.context import compact_tool_payload
    s.history.append({"role": "user", "content": "Here is a photo of my burn. It is on my forearm and I am a 28 year old adult."})
    seeded = [("check_image_quality", {"image_path": cs["locked_image_path"]}, cs["quality_result"]),
              ("classify_burn_severity", {"image_path": cs["locked_image_path"]}, cs["classify_result"]),
              ("segment_wound_area", {"image_path": cs["locked_image_path"]}, cs["segment_result"]),
              ("retrieve_treatment_guidance", {"query": "first aid for a burn on the forearm, adult"}, cs["guidance_result"]),
              ("schedule_followup", {"case_id": s.case_id, "interval_days": 2}, {"scheduled_date": "2026-09-22", "saved": True}),
              ("locate_nearby_clinics", {"location": "Vijay Nagar, Indore"}, cs["clinics_result"])]
    for i, (name, args, res) in enumerate(seeded):
        cid = f"call_seed{i}"
        s.history.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        s.history.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(compact_tool_payload(name, res))})
    s.history.append({"role": "assistant", "content": (
        "I checked the photo: it looks like a 3rd-degree burn (high confidence) affecting a large part of the visible "
        "area, so I recommend urgent medical care. I scheduled a follow-up in 2 days and found nearby care centres.")})
else:
    out, tools = photo_flow(s, "1. upload a sample photo", "burnWound11.jpg")
    snap = s.snapshot()
    check("photo went through quality, classify and segmentation", {"check_image_quality", "classify_burn_severity", "segment_wound_area"} <= set(tools), tools)
    check("snapshot has severity, confidence, area band, verdict", all([snap["classify"], snap["segment"], snap["verdict"]]),
          f"{(snap['classify'] or {}).get('predicted_severity')} / {(snap['segment'] or {}).get('area_band')} / {snap['verdict']}")
    check("guidance retrieved", len(snap["guidance"]) > 0, f"{len(snap['guidance'])} passages")
    check("Grad-CAM image is in the user's folder and readable", snap["gradcam_path"] and s.read_file(snap["gradcam_path"], "gradcam"))
    check("follow-up date saved", bool(snap["next_followup"]), snap["next_followup"])
    check("calendar invite emailed for the follow-up",
          any(json.loads(m["content"]).get("reminder_email") == "sent" for m in s.history if m.get("role") == "tool" and m["content"].startswith("{") and "reminder_email" in m["content"]))
    check("no urgent banner missing for a 3rd-degree result", snap["urgent"] and snap["urgent_banner"])

    out, tools = turn(s, "2. chat question", s.send, "Why is the confidence not higher?")
    check("answered in chat", bool(out["answer"]) and out["error"] is None)

    out, tools = turn(s, "3. nearby care", s.send, "Can you find care centres near Vijay Nagar, Indore?")
    snap = s.snapshot()
    check("clinic lookup ran and returned options", "locate_nearby_clinics" in tools and len(snap["clinics"]) >= 1, f"{len(snap['clinics'])} options")

out, tools = turn(s, "4a. ask for a personal copy of the report", s.send, "I would like a copy of the report for my own records, not for any doctor.")
prev = s.case_state.get("report_preview") or {}
check("agent previewed the report contents first", prev.get("status") == "preview", (prev.get("message") or "")[:160])
check("nothing was generated before consent", s.report_path() is None)
out, tools = turn(s, "4a2. add a note and drop a section", s.send,
                  "Please add a note that I am allergic to latex, and leave out the nearby care section.")
prev = s.case_state.get("report_preview") or {}
by = {c["section"]: c for c in prev.get("contents", [])}
check("preview reflects the note and the removed section",
      by.get("Patient-reported information", {}).get("included") and not by.get("Nearby care options", {}).get("included"), prev.get("message"))
out, tools = turn(s, "4b. consent", s.send, "Yes, I consent.")
path = s.report_path()
check("download-only report created, in the user's folder", "generate_escalation_report" in tools and path, path)
check("download-only report was not emailed", not (s.case_state["report_result"] or {}).get("emailed"))
check("reply does not quote a file path", not any(x in out["answer"].lower() for x in ("outputs", ".pdf", "/users/", chr(92) + "users")), out["answer"][:160])

out, tools = turn(s, "5a. ask to mail it to me (own login address, not typed)", s.send, "Please mail the report to me.")
check("nothing sent before consent", not (s.case_state["report_result"] or {}).get("emailed") and (s.case_state.get("report_preview") or {}).get("recipient"))
out, tools = turn(s, "5b. consent to send", s.send, "Yes, I consent.")
rr = s.case_state["report_result"] or {}
check("report emailed to the doctor address", rr.get("emailed") is True, rr.get("message"))

if "--skip-followup" not in sys.argv and not REPORT_ONLY:
    s2 = service.Session.start(user, "followup", s.case_id)
    out, tools = photo_flow(s2, "6. follow-up visit with a new photo", "burnWound12.jpg")
    snap2 = s2.snapshot()
    check("follow-up saved and classified", s2.is_followup and snap2["classify"] is not None)
    out, tools = turn(s2, "7. compare", s2.send, "How does this compare with my last visit?")
    check("comparison answered", bool(out["answer"]) and out["error"] is None)
    with UC.acting_as(user["user_id"]):
        rec = CS.get_case(s.case_id)
    check("follow-up stored in the case history", len(rec["followups"]) == 1, f"{len(rec['followups'])} follow-up(s)")

other = US.register("Other User", "other.user@example.com", "other-pass-1")
with UC.acting_as(other["user_id"]):
    check("a second user cannot see this case", CS.get_case(s.case_id) is None and CS.list_active_cases() == [])

failed = [n for n, ok in CHECKS if not ok]
print(f"\nSUMMARY: {len(CHECKS) - len(failed)}/{len(CHECKS)} passed" + (f" | FAILED: {failed}" if failed else ""))
