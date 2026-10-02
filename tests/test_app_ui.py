"""Smoke tests of the Streamlit app through Streamlit's own test harness (no browser, no Groq). Temp DB and folders."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import agent.case_store as CS
import agent.user_store as US
import agent.userctx as UC
from agent.userctx import acting_as

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture()
def ui(tmp_path, monkeypatch):
    monkeypatch.setenv("BURNSCOPE_NO_WARMUP", "1")
    monkeypatch.setenv("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "test-key"))
    monkeypatch.setattr(CS, "DB_PATH", str(tmp_path / "ui.db"))
    roots = {"uploads": str(tmp_path / "du"), "gradcam": str(tmp_path / "ou"), "reports": str(tmp_path / "ou")}
    monkeypatch.setattr(UC, "_KIND_ROOTS", roots)
    monkeypatch.setattr(UC, "UPLOADS_ROOT", roots["uploads"])
    monkeypatch.setattr(US, "UPLOADS_ROOT", roots["uploads"])
    monkeypatch.setattr(US, "OUTPUTS_ROOT", roots["gradcam"])
    CS.init_db()
    return lambda: AppTest.from_file(APP, default_timeout=300).run()


def texts(at):
    return " | ".join([m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning] + [i.value for i in at.info])


def sign_up(at, name="Asha", email="asha@example.com", code="secret-A1"):
    at.text_input(key="su_email").set_value(email)
    at.text_input(key="su_code").set_value(code)
    at.text_input(key="su_again").set_value(code)
    [t for t in at.text_input if t.label == "Your name"][0].set_value(name)
    [b for b in at.button if b.label == "Create account"][0].click()
    return at.run()


def test_login_page_then_sign_up_and_dashboard(ui):
    at = ui()
    assert not at.exception and "Sign in to continue" in texts(at)
    at = sign_up(at)
    assert not at.exception and at.session_state["user"]["email"] == "asha@example.com"
    assert "Wound assessment" in texts(at)
    assert [r.value for r in at.sidebar.radio] == ["Dashboard"]
    assert any(b.label == "Logout" for b in at.sidebar.button)
    assert any(b.label == "Start a new case" for b in at.button)


def test_wrong_passcode_and_short_passcode_are_refused(ui):
    at = sign_up(ui(), code="123")
    assert "at least 6 characters" in texts(at) and "user" not in at.session_state
    at = sign_up(ui())
    at.session_state["user"] = None
    fresh = ui()
    [t for t in fresh.text_input if t.label == "Email address"][0].set_value("asha@example.com")
    [t for t in fresh.text_input if t.label == "Passcode"][0].set_value("wrong-pass")
    [b for b in fresh.button if b.label == "Sign in"][0].click()
    fresh.run()
    assert "do not match" in texts(fresh)


def test_dashboard_history_profile_and_logout(ui):
    import agent.service as service
    user = US.register("Asha", "asha@example.com", "secret-A1")
    at = ui()
    at.session_state["user"] = user
    at.session_state["nav"] = "Dashboard"
    at.session_state["sess"] = service.Session.start(user, "new")      # what "Start a new case" does
    at.run()
    assert not at.exception and "Ask BurnScope" in texts(at)
    assert any(b.label == "Analyse photo" for b in at.button) and any(b.label == "Change case" for b in at.button)
    at.session_state["nav"] = "History"
    at.run()
    assert not at.exception and "No saved cases yet" in texts(at)
    at.session_state["nav"] = "Profile"
    at.run()
    assert not at.exception and at.text_input[0].value == "Asha"
    [b for b in at.sidebar.button if b.label == "Logout"][0].click()
    at.run()
    assert "user" not in at.session_state and "Sign in to continue" in texts(at)


def test_start_button_creates_a_case_for_the_signed_in_user(ui):
    user = US.register("Asha", "asha@example.com", "secret-A1")
    at = ui()
    at.session_state["user"] = user
    at.session_state["nav"] = "Dashboard"
    at.run()
    [b for b in at.button if b.label == "Start a new case"][0].click()
    at.run()
    assert not at.exception and at.session_state.filtered_state.get("sess") is not None
    assert at.session_state.filtered_state["sess"].user["user_id"] == user["user_id"]


def test_history_shows_only_the_signed_in_users_cases(ui):
    a = US.register("Asha", "asha@example.com", "secret-A1")
    b = US.register("Ben", "ben@example.com", "secret-B1")
    with acting_as(a["user_id"]):
        cid = CS.create_new_case()
        CS.save_baseline_result(cid, {"predicted_severity": "2nd_degree"}, {"wound_fraction": 0.2})
    at = ui()
    at.session_state["user"] = b
    at.session_state["nav"] = "History"
    at.run()
    assert not at.exception and "No saved cases yet" in texts(at) and cid[-8:] not in texts(at)
    at.session_state["user"] = a
    at.run()
    assert not at.exception and cid[-8:] in " ".join(e.label for e in at.expander)
