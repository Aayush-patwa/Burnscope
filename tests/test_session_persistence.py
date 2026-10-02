"""A page reload must not sign the user out or lose the open case. Temp DB and folders; no Groq calls."""
import os
import time

import pytest

import agent.case_store as CS
import agent.service as service
import agent.user_store as US
import agent.userctx as UC
from agent.userctx import acting_as


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(CS, "DB_PATH", str(tmp_path / "p.db"))
    roots = {"uploads": str(tmp_path / "du"), "gradcam": str(tmp_path / "ou"), "reports": str(tmp_path / "ou")}
    monkeypatch.setattr(UC, "_KIND_ROOTS", roots)
    monkeypatch.setattr(UC, "UPLOADS_ROOT", roots["uploads"])
    monkeypatch.setattr(US, "UPLOADS_ROOT", roots["uploads"])
    monkeypatch.setattr(US, "OUTPUTS_ROOT", roots["gradcam"])
    CS.init_db()
    a = US.register("Asha", "asha@example.com", "secret-A1")
    b = US.register("Ben", "ben@example.com", "secret-B1")
    return a, b


# ------------------------------------------------------------------ login token
def test_token_signs_the_user_back_in(env):
    a, _ = env
    token = US.create_session_token(a["user_id"])
    assert US.user_for_token(token)["user_id"] == a["user_id"]
    assert US.user_for_token("not-a-token") is None
    assert US.user_for_token("") is None and US.user_for_token(None) is None


def test_token_is_stored_only_as_a_hash(env):
    a, _ = env
    token = US.create_session_token(a["user_id"])
    conn = CS._get_connection()
    try:
        stored = [r["token_hash"] for r in conn.execute("SELECT token_hash FROM auth_sessions")]
    finally:
        conn.close()
    assert stored and token not in stored and all(len(s) == 64 for s in stored)


def test_revoked_token_no_longer_works(env):
    a, _ = env
    token = US.create_session_token(a["user_id"])
    US.revoke_token(token)
    assert US.user_for_token(token) is None


def test_expired_token_is_rejected_and_removed(env):
    a, _ = env
    token = US.create_session_token(a["user_id"])
    conn = CS._get_connection()
    conn.execute("UPDATE auth_sessions SET expires_at = ?", (time.time() - 5,))
    conn.commit()
    conn.close()
    assert US.user_for_token(token) is None
    conn = CS._get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM auth_sessions").fetchone()["n"] == 0
    finally:
        conn.close()


def test_each_use_slides_the_expiry(env):
    a, _ = env
    token = US.create_session_token(a["user_id"])
    conn = CS._get_connection()
    conn.execute("UPDATE auth_sessions SET expires_at = ?", (time.time() + 60,))
    conn.commit()
    conn.close()
    assert US.user_for_token(token)
    conn = CS._get_connection()
    try:
        assert conn.execute("SELECT expires_at FROM auth_sessions").fetchone()["expires_at"] > time.time() + US.SESSION_SECONDS - 100
    finally:
        conn.close()


def test_active_case_pointer_and_delete_my_data(env):
    a, _ = env
    token = US.create_session_token(a["user_id"])
    assert US.get_active_case(token) is None
    US.set_active_case(token, "case_abc12345")
    assert US.get_active_case(token) == "case_abc12345"
    US.set_active_case(token, None)
    assert US.get_active_case(token) is None
    US.delete_user_data(a["user_id"])
    assert US.user_for_token(token) is None                # deleting the account ends its sessions


# ------------------------------------------------------------------ open case survives a reload
def test_open_case_round_trips_through_the_database(env):
    a, _ = env
    sess = service.Session.start(a, "new")
    sess.transcript += [{"role": "user", "content": "(uploaded a photo)"}, {"role": "assistant", "content": "Looks like a second-degree burn."}]
    sess.history.append({"role": "user", "content": "Here is a photo of my burn."})
    sess.case_state["classify_result"] = {"predicted_severity": "2nd_degree", "confidence": 0.93}
    sess.save()

    back = service.Session.resume(a, sess.case_id)
    assert back is not None and back.case_id == sess.case_id and back.is_followup is False
    assert back.transcript == sess.transcript
    assert back.history == sess.history
    assert back.case_state["classify_result"]["predicted_severity"] == "2nd_degree"
    assert back.case_state["own_email"] == a["email"]


def test_numpy_values_in_results_are_saved_as_plain_numbers(env):
    np = pytest.importorskip("numpy")
    a, _ = env
    sess = service.Session.start(a, "new")
    sess.case_state["classify_result"] = {"confidence": np.float32(0.9), "scores": np.array([0.1, 0.9])}
    sess.save()
    back = service.Session.resume(a, sess.case_id)
    assert back.case_state["classify_result"]["scores"] == [pytest.approx(0.1), pytest.approx(0.9)]


def test_save_if_changed_only_writes_when_something_changed(env, monkeypatch):
    a, _ = env
    sess = service.Session.start(a, "new")
    writes = []
    real = service.save_session_blob
    monkeypatch.setattr(service, "save_session_blob", lambda *args: (writes.append(1), real(*args))[1])
    sess.save_if_changed()
    assert writes == []                                    # nothing changed since start()
    sess.transcript.append({"role": "user", "content": "hi"})
    sess.save_if_changed()
    assert writes == [1]


def test_a_case_cannot_be_resumed_by_another_user_or_after_deletion(env):
    a, b = env
    sess = service.Session.start(a, "new")
    assert service.Session.resume(b, sess.case_id) is None
    assert service.Session.resume(a, "case_does_not_exist") is None
    assert service.Session.resume(a, None) is None
    conn = CS._get_connection()
    conn.execute("DELETE FROM cases WHERE case_id = ?", (sess.case_id,))
    conn.commit()
    conn.close()
    assert service.Session.resume(a, sess.case_id) is None


# ------------------------------------------------------------------ the app itself
def test_reload_with_a_valid_cookie_restores_sign_in_and_open_case(env, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    a, _ = env
    monkeypatch.setenv("BURNSCOPE_NO_WARMUP", "1")
    monkeypatch.setenv("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "test-key"))
    sess = service.Session.start(a, "new")
    sess.transcript.append({"role": "assistant", "content": "Welcome back test message"})
    sess.save()
    token = US.create_session_token(a["user_id"])
    US.set_active_case(token, sess.case_id)

    # AppTest has no browser, so the app's cookie read (st.context.cookies) is pointed at the token.
    import streamlit as st

    class _Ctx:
        cookies = {"burnscope_session": token}
    monkeypatch.setattr(st, "context", _Ctx(), raising=False)
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py"), default_timeout=300).run()
    assert not at.exception
    assert at.session_state["user"]["email"] == "asha@example.com"
    assert at.session_state["sess"] is not None and at.session_state["sess"].case_id == sess.case_id
    shown = " | ".join(m.value for cm in at.chat_message for m in cm.markdown)
    assert "Welcome back test message" in shown


def test_reload_with_a_revoked_cookie_shows_the_sign_in_page(env, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    a, _ = env
    monkeypatch.setenv("BURNSCOPE_NO_WARMUP", "1")
    monkeypatch.setenv("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "test-key"))
    token = US.create_session_token(a["user_id"])
    US.revoke_token(token)
    import streamlit as st

    class _Ctx:
        cookies = {"burnscope_session": token}
    monkeypatch.setattr(st, "context", _Ctx(), raising=False)
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py"), default_timeout=300).run()
    assert not at.exception
    assert "user" not in at.session_state or not at.session_state["user"]
    assert "Sign in to continue" in " | ".join(m.value for m in at.markdown)
