"""Multi-user isolation, sign-in and the service layer. No Groq calls; temp DB and temp folders."""
import io
import os
import threading
import time

import pytest
from PIL import Image

import agent.case_store as CS
import agent.dispatch as dispatch
import agent.service as service
import agent.user_store as US
import agent.userctx as UC
from agent.userctx import acting_as


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(CS, "DB_PATH", str(tmp_path / "t.db"))
    roots = {"uploads": str(tmp_path / "data_users"), "gradcam": str(tmp_path / "out_users"), "reports": str(tmp_path / "out_users")}
    monkeypatch.setattr(UC, "_KIND_ROOTS", roots)
    monkeypatch.setattr(UC, "UPLOADS_ROOT", roots["uploads"])
    monkeypatch.setattr(US, "UPLOADS_ROOT", roots["uploads"])
    monkeypatch.setattr(US, "OUTPUTS_ROOT", roots["gradcam"])
    CS.init_db()
    a = US.register("Asha", "asha@example.com", "secret-A1")
    b = US.register("Ben", "ben@example.com", "secret-B1")
    return a, b


def jpeg(size=(120, 90), exif=False):
    buf = io.BytesIO()
    img = Image.new("RGB", size, (200, 120, 100))
    if exif:
        e = Image.Exif()
        e[0x010F] = "TestCam"
        img.save(buf, "JPEG", exif=e)
    else:
        img.save(buf, "JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------ sign-in
def test_register_and_authenticate(env):
    a, _ = env
    assert US.authenticate("ASHA@example.com ", "secret-A1")["user_id"] == a["user_id"]
    with pytest.raises(US.AuthError):
        US.authenticate("asha@example.com", "wrong")
    with pytest.raises(US.AuthError):
        US.authenticate("nobody@example.com", "secret-A1")


def test_register_rules_and_hashing(env):
    with pytest.raises(US.AuthError):
        US.register("X", "asha@example.com", "another-pass")          # duplicate email
    with pytest.raises(US.AuthError):
        US.register("X", "not-an-email", "another-pass")
    with pytest.raises(US.AuthError):
        US.register("X", "x@example.com", "123")                      # too short
    row = CS._get_connection().execute("SELECT * FROM users WHERE email='asha@example.com'").fetchone()
    assert "secret-A1" not in (row["pass_hash"] + row["salt"]) and len(row["pass_hash"]) == 64


def test_lockout_after_repeated_failures(env):
    for _ in range(US.MAX_FAILED_ATTEMPTS):
        with pytest.raises(US.AuthError):
            US.authenticate("ben@example.com", "nope")
    with pytest.raises(US.AuthError, match="Too many"):
        US.authenticate("ben@example.com", "secret-B1")               # even the right passcode is refused for now


# ------------------------------------------------------------------ data isolation
def make_case(user):
    with acting_as(user["user_id"]):
        cid = CS.create_new_case()
        CS.save_baseline_result(cid, {"predicted_severity": "2nd_degree"}, {"wound_fraction": 0.2})
    return cid


def test_users_cannot_read_or_change_each_others_cases(env):
    a, b = env
    cid = make_case(a)
    with acting_as(a["user_id"]):
        assert CS.get_case(cid) is not None and [c["case_id"] for c in CS.list_active_cases()] == [cid]
    with acting_as(b["user_id"]):
        assert CS.get_case(cid) is None
        assert CS.list_active_cases() == []
        assert CS.set_next_followup(cid, "2030-01-01", 3) is False
        CS.save_baseline_result(cid, {"predicted_severity": "3rd_degree"}, {})
        CS.add_followup_result(cid, {}, {})
    with acting_as(a["user_id"]):
        c = CS.get_case(cid)
        assert c["baseline_result"]["classify"]["predicted_severity"] == "2nd_degree"
        assert c["followups"] == [] and c["next_followup"] is None


def test_unowned_session_sees_no_user_cases(env):
    a, _ = env
    cid = make_case(a)
    assert CS.get_case(cid) is None and CS.list_active_cases() == []


def test_tools_refuse_another_users_case_and_files(env):
    a, b = env
    cid = make_case(a)
    with acting_as(b["user_id"]):
        assert "error" in dispatch.execute_tool("schedule_followup", {"case_id": cid, "interval_days": 2})
        r = dispatch.execute_tool("generate_escalation_report", {"case_id": cid, "consent": True})
        assert r["status"] == "error" and "No case found" in r["message"]
    a_photo = service.Session.start(a).save_photo(jpeg())
    with acting_as(b["user_id"]):
        assert "not one of this user" in dispatch.execute_tool("check_image_quality", {"image_path": a_photo})["error"]
        r = dispatch.execute_tool("generate_gradcam", {"image_path": "data/sample_images/burnWound.jpg", "predicted_class": "1st_degree"})
        assert "not one of this user" in r["error"]
    with acting_as(a["user_id"]):
        assert "error" not in dispatch.execute_tool("check_image_quality", {"image_path": a_photo})


def test_models_run_one_at_a_time(env, monkeypatch):
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    def fake(**kw):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return {"ok": True}

    monkeypatch.setitem(dispatch.TOOL_DISPATCH, "classify_burn_severity", fake)
    threads = [threading.Thread(target=lambda: dispatch.execute_tool("classify_burn_severity", {})) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max"] == 1


def test_delete_my_data(env):
    a, b = env
    make_case(a)
    cid_b = make_case(b)
    service.Session.start(a).save_photo(jpeg())
    folder = UC.user_dir("uploads", a["user_id"], create=False)
    assert os.path.isdir(folder)
    US.delete_user_data(a["user_id"])
    assert US.get_user(a["user_id"]) is None and not os.path.exists(folder)
    with acting_as(b["user_id"]):
        assert CS.get_case(cid_b) is not None          # the other user is untouched


# ------------------------------------------------------------------ service layer
def test_session_start_rules(env):
    a, b = env
    cid = make_case(a)
    s = service.Session.start(a, "followup", cid)
    assert s.is_followup and s.case_id == cid
    with pytest.raises(service.SessionError):
        service.Session.start(b, "followup", cid)         # not B's case
    fresh = service.Session.start(a, "new")
    with pytest.raises(service.SessionError):
        service.Session.start(a, "followup", fresh.case_id)   # no assessment saved yet


def test_photo_validation_and_privacy(env):
    a, _ = env
    s = service.Session.start(a)
    with pytest.raises(service.SessionError):
        s.save_photo(b"not an image")
    with pytest.raises(service.SessionError):
        s.save_photo(b"")
    with pytest.raises(service.SessionError):
        s.save_photo(b"x" * (service.MAX_UPLOAD_BYTES + 1))
    path = s.save_photo(jpeg((3000, 2000), exif=True))
    with acting_as(a["user_id"]):
        assert UC.path_is_users(path, "uploads")
    saved = Image.open(path)
    assert max(saved.size) <= service.MAX_SIDE_PX and not saved.getexif()      # resized, EXIF removed
    assert s.read_file(path, "uploads") and s.read_file("data/sample_images/burnWound.jpg", "uploads") is None


def test_send_runs_as_the_user_and_keeps_transcript(env, monkeypatch):
    a, _ = env
    seen = {}

    def fake_turn(history, case_id, case_state):
        seen["user"] = UC.current_user_id()
        history.append({"role": "assistant", "content": "ok"})
        return {"history": history, "final_answer": "ok", "error": None, "guard": None}

    monkeypatch.setattr(service.loop, "run_agent_turn", fake_turn)
    s = service.Session.start(a)
    out = s.send("hello")
    assert out["answer"] == "ok" and seen["user"] == a["user_id"]
    assert [m["role"] for m in s.transcript] == ["user", "assistant"]
    assert s.send("   ")["error"] == "empty"
    snap = s.snapshot()
    assert snap["case_id"] == s.case_id and snap["report_path"] is None and snap["verdict"] is None


# ------------------------------------------------------------------ reminders and the urgent banner
import agent.reminders as R
from tools.scheduler import schedule_followup


@pytest.fixture()
def mailbox(monkeypatch):
    sent = []
    monkeypatch.setattr(R, "send_mail", lambda to, subject, body, attachments=None: sent.append((to, subject, body, attachments)) or True)
    return sent


def test_scheduling_emails_a_calendar_invite_to_signed_in_users(env, mailbox):
    a, _ = env
    cid = make_case(a)
    with acting_as(a["user_id"]):
        r = schedule_followup(cid, 2)
    assert r["reminder_email"] == "sent" and r["saved"]
    to, subject, body, att = mailbox[0]
    ics = att[0][1].decode()
    assert to == "asha@example.com" and att[0][2] == "calendar" and r["scheduled_date"].replace("-", "") in ics
    assert "BEGIN:VALARM" in ics and "\r\n" in ics


def test_scheduling_respects_reminder_opt_out_and_cannot_touch_others_cases(env, mailbox):
    a, _ = env
    cid = make_case(a)
    US.update_profile(a["user_id"], reminders_enabled=False)
    with acting_as(a["user_id"]):
        assert schedule_followup(cid, 2)["reminder_email"] == "off"
    assert mailbox == []
    with acting_as(None):                       # unowned session cannot even see the case, and nothing is emailed
        assert "error" in schedule_followup(cid, 2)
    assert mailbox == []


def test_due_reminders_are_sent_once_and_only_to_the_owner(env, mailbox):
    a, b = env
    ca, cb = make_case(a), make_case(b)
    with acting_as(a["user_id"]):
        CS.set_next_followup(ca, "2020-01-01", 2)        # long overdue
    with acting_as(b["user_id"]):
        CS.set_next_followup(cb, "2999-01-01", 2)        # far in the future
    assert R.send_due_reminders(b["user_id"]) == []      # B has nothing due
    assert R.send_due_reminders(a["user_id"]) == [ca]
    assert [m[0] for m in mailbox] == ["asha@example.com"]
    assert "severity" not in mailbox[0][2].lower() and "degree" not in mailbox[0][2].lower()    # no health details in the email
    assert R.send_due_reminders() == []                  # the system sweep does not send it again
    assert [d["case_id"] for d in CS.due_followups_for_user(a["user_id"])] == [ca]      # still shown as due (banner)
    assert CS.due_followups_for_user(b["user_id"]) == []


def test_opted_out_users_get_no_due_email(env, mailbox):
    a, _ = env
    ca = make_case(a)
    with acting_as(a["user_id"]):
        CS.set_next_followup(ca, "2020-01-01", 2)
    US.update_profile(a["user_id"], reminders_enabled=False)
    assert R.send_due_reminders() == [] and mailbox == []


def test_ics_is_a_valid_all_day_event():
    ics = R.build_ics("case_12345678", "2026-09-22")
    assert ics.startswith("BEGIN:VCALENDAR") and ics.rstrip().endswith("END:VCALENDAR")
    assert "DTSTART;VALUE=DATE:20260922" in ics and "DTEND;VALUE=DATE:20260923" in ics and "TRIGGER:PT9H" in ics


def test_snapshot_shows_urgent_banner_and_report_preview(env):
    a, _ = env
    s = service.Session.start(a)
    assert s.snapshot()["urgent"] is False and s.snapshot()["report_preview"] is None
    s.case_state["classify_result"] = {"predicted_severity": "3rd_degree", "decision": "PROCEED", "calibrated_confidence": 0.99}
    s.case_state["segment_result"] = {"area_band": "large", "multi_region": False, "wound_fraction": 0.5}
    snap = s.snapshot()
    assert snap["urgent"] and "emergency" in snap["urgent_banner"].lower()
    s.case_state.update(report_preview={"status": "preview", "message": "This report will contain: ..."}, report_preview_sig="x")
    assert s.snapshot()["report_preview"]["status"] == "preview"


def test_check_reminders_on_login(env, mailbox):
    a, b = env
    ca = make_case(a)
    with acting_as(a["user_id"]):
        CS.set_next_followup(ca, "2020-01-01", 2)
    assert [d["case_id"] for d in service.Session.check_reminders(a)] == [ca] and len(mailbox) == 1
    assert service.Session.check_reminders(b) == []


def test_session_tells_the_agent_who_is_signed_in(env):
    a, _ = env
    s = service.Session.start(a)
    assert s.case_state["own_email"] == "asha@example.com"
    assert any("asha@example.com" in m["content"] for m in s.history if m["role"] == "system")


def test_the_model_never_sees_the_report_file_path(env, monkeypatch):
    """Runs the real loop with a scripted model: the tool result placed in history must not contain pdf_path."""
    import json
    from types import SimpleNamespace
    a, _ = env
    monkeypatch.setitem(dispatch.TOOL_DISPATCH, "generate_escalation_report",
                        lambda **kw: {"status": "success", "pdf_path": "C:/secret/place/report.pdf", "emailed": False, "message": "ready"})
    monkeypatch.setattr(service.loop, "preview_required", lambda cs, args: None)
    monkeypatch.setattr(service.loop, "check_report_request", lambda *a, **k: None)
    steps = iter([
        SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(
            name="generate_escalation_report", arguments=json.dumps({"case_id": "x", "consent": True})))]),
        SimpleNamespace(content="Your report is ready to download.", tool_calls=None)])
    monkeypatch.setattr(service.loop, "_call_with_retry", lambda history: SimpleNamespace(
        choices=[SimpleNamespace(message=next(steps), finish_reason="stop")]))
    s = service.Session.start(a)
    s.send("yes")
    tool_msgs = [m["content"] for m in s.history if m.get("role") == "tool"]
    assert tool_msgs and all("pdf_path" not in t and "secret" not in t for t in tool_msgs)
    assert s.case_state["report_result"]["pdf_path"] == "C:/secret/place/report.pdf"      # the app itself still has it for the download button


# ------------------------------------------------------------------ the model must never claim a report that was not made
def _scripted_session(env, monkeypatch, steps, report_result=None):
    import json
    from types import SimpleNamespace
    a, _ = env
    monkeypatch.setitem(dispatch.TOOL_DISPATCH, "generate_escalation_report",
                        lambda **kw: report_result or {"status": "success", "pdf_path": "x.pdf", "emailed": False, "message": "ready"})
    def msg(step):
        if isinstance(step, dict):
            return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id=f"c{len(seen)}", function=SimpleNamespace(
                name="generate_escalation_report", arguments=json.dumps(step)))])
        return SimpleNamespace(content=step, tool_calls=None)
    seen, it = [], iter(steps)
    def fake(history):
        seen.append([m for m in history if m.get("role") == "system"][-1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg(next(it)), finish_reason="stop")])
    monkeypatch.setattr(service.loop, "_call_with_retry", fake)
    return service.Session.start(a), seen


def test_false_report_claim_is_sent_back_for_correction(env, monkeypatch):
    # the real call is refused (never previewed); the model then wrongly says the report is ready
    s, seen = _scripted_session(env, monkeypatch, [
        {"case_id": "x", "consent": True},
        "Your PDF report is ready to download with the Download button.",
        "Here is what the report will contain: key findings and a follow-up plan. Do you consent to me generating it?"])
    out = s.send("yes")
    assert "ready to download" not in out["answer"] and "Do you consent" in out["answer"]
    assert any(t.startswith("CORRECTION:") for t in seen)


def test_a_real_report_can_be_announced_and_ordinary_replies_are_untouched(env, monkeypatch):
    monkeypatch.setattr(service.loop, "preview_required", lambda cs, args: None)
    monkeypatch.setattr(service.loop, "check_report_request", lambda *a, **k: None)
    s, seen = _scripted_session(env, monkeypatch, [{"case_id": "x", "consent": True}, "Your report is ready to download."])
    assert s.send("yes")["answer"] == "Your report is ready to download."
    assert not any(t.startswith("CORRECTION:") for t in seen)
    s2, seen2 = _scripted_session(env, monkeypatch, ["Sure. Do you consent to me generating the report for you?"])
    assert "consent" in s2.send("I want a report")["answer"] and not any(t.startswith("CORRECTION:") for t in seen2)
