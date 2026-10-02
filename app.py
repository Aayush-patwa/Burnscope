"""
BurnScope -- Streamlit app.   Run:  streamlit run app.py

Pages (left sidebar): Dashboard, History, Profile, Logout. Everything a signed-in user sees comes from
agent/service.py, which only ever touches that user's own cases, photos and reports.
"""
import html
import os
import threading
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env")

st.set_page_config(page_title="BurnScope", page_icon="🩹", layout="wide")

import agent.user_store as US
from agent.case_store import init_db
from agent.reminders import start_reminder_daemon
from agent.userctx import MODEL_LOCK


def _load_heavy():
    """The agent, models and report code are big (TensorFlow, PyTorch, ChromaDB): import them only after sign-in,
    while the warm-up thread has been loading them in the background since the server started."""
    if "service" in globals():
        return
    try:
        import agent.service as service_module
        from tools import escalate
    except KeyError as missing:                                # the agent module needs GROQ_API_KEY at import
        st.error(f"Missing setting {missing} in the .env file. Add it and restart the app.")
        st.stop()
    globals().update(service=service_module, ACTIONS=escalate.ACTIONS, AREA_LABELS=escalate.AREA_LABELS,
                     SEVERITY_LABELS=escalate.SEVERITY_LABELS, URGENCY=escalate.URGENCY, _default_trend=escalate._default_trend)


ACC, ACC_DEEP, SOFT = "#0E7C86", "#0B5E66", "#E3F2F1"
TONES = {"red": ("#C13F2E", "#FBE6E2"), "amber": ("#B4740A", "#FBF0DC"), "green": ("#2F8F5B", "#E4F4EA"), "teal": (ACC_DEEP, SOFT)}
VERDICT_TONE = {"urgent_care": "red", "clinical_review": "amber", "self_care": "green"}
SEVERITY_TONE = {"1st_degree": "green", "2nd_degree": "amber", "3rd_degree": "red"}

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1350px; }}
  h1, h2, h3 {{ letter-spacing: .01em; }}
  .brand {{ display:flex; align-items:center; gap:10px; margin-bottom:.4rem; }}
  .mark {{ width:34px; height:34px; border-radius:9px; background:linear-gradient(155deg,{ACC},{ACC_DEEP});
           color:#fff; font-weight:700; display:flex; align-items:center; justify-content:center; }}
  .brandname {{ font-weight:700; font-size:1.15rem; }}
  .chip {{ display:inline-flex; align-items:center; gap:6px; padding:3px 11px; border-radius:999px; font-weight:700; font-size:.82rem; }}
  .pill {{ display:inline-block; padding:4px 13px; border-radius:999px; background:#EAF1F1; border:1px solid #D7E3E3; font-size:.85rem; }}
  .kv {{ width:100%; border-collapse:collapse; font-size:.92rem; }}
  .kv td {{ padding:9px 4px; border-bottom:1px solid #D7E3E3; vertical-align:middle; }}
  .kv td.k {{ color:#5B7278; width:42%; }}
  .kv tr:last-child td {{ border-bottom:none; }}
  .bar {{ display:inline-block; width:90px; height:7px; border-radius:99px; background:#EAF1F1; overflow:hidden; vertical-align:middle; margin-right:8px; }}
  .bar i {{ display:block; height:100%; background:{ACC}; }}
  .muted {{ color:#5B7278; font-size:.85rem; }}
  section[data-testid="stSidebar"] {{ border-right:1px solid #D7E3E3; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# One-time setup (per server process)
# --------------------------------------------------------------------------- #

def _warm_models():
    """Load the heavy models in the background so the first photo does not pay for it."""
    with MODEL_LOCK:
        try:
            import tools.classify as classify
            import tools.segment as segment
            from rag.vector_store import query
            classify._load_model()
            segment._load_model()
            query("burn", 1)
        except Exception as e:                                # never block the app on a warm-up problem
            print(f"[warm-up] {type(e).__name__}: {e}")


@st.cache_resource
def boot():
    init_db()
    try:
        from rag.build_index import ensure_index
        ensure_index()                                        # first run on a fresh checkout: build the guidance index
    except Exception as e:                                    # the app still opens; retrieval will report the problem
        print(f"[boot] could not prepare the guidance index: {type(e).__name__}: {e}")
    start_reminder_daemon()                                   # emails follow-ups as they come due while the app runs
    if not os.environ.get("BURNSCOPE_NO_WARMUP"):
        threading.Thread(target=_warm_models, daemon=True).start()
    return True


boot()


# --------------------------------------------------------------------------- #
# Small display helpers
# --------------------------------------------------------------------------- #

def chip(text, tone):
    fg, bg = TONES[tone]
    return f'<span class="chip" style="color:{fg};background:{bg}"><span style="width:7px;height:7px;border-radius:50%;background:{fg}"></span>{html.escape(text)}</span>'


def brand():
    st.markdown('<div class="brand"><div class="mark">BS</div><span class="brandname">BurnScope</span></div>', unsafe_allow_html=True)


def when(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return iso or "-"


def sev_label(sev):
    return SEVERITY_LABELS.get(sev, sev or "not available")


# --------------------------------------------------------------------------- #
# Sign in / create account
# --------------------------------------------------------------------------- #

def login_page():
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        brand()
        st.markdown("### Sign in to continue")
        st.caption("Your photos, cases and reports are private to your account and are never shown to other users.")
        tab_in, tab_new = st.tabs(["Sign in", "Create account"])
        with tab_in:
            with st.form("signin"):
                email = st.text_input("Email address")
                code = st.text_input("Passcode", type="password")
                if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                    try:
                        user = US.authenticate(email, code)
                    except US.AuthError as e:
                        st.error(str(e))
                    else:
                        start_user_session(user)
        with tab_new:
            with st.form("signup"):
                name = st.text_input("Your name")
                email = st.text_input("Email address", key="su_email", help="Follow-up reminders and reports you choose to send are emailed from here.")
                code = st.text_input("Choose a passcode (6+ characters)", type="password", key="su_code")
                again = st.text_input("Repeat the passcode", type="password", key="su_again")
                if st.form_submit_button("Create account", type="primary", use_container_width=True):
                    if code != again:
                        st.error("The two passcodes do not match.")
                    else:
                        try:
                            user = US.register(name, email, code)
                        except US.AuthError as e:
                            st.error(str(e))
                        else:
                            start_user_session(user)
        st.caption("BurnScope supports, and does not replace, evaluation by a qualified clinician.")


COOKIE_NAME = "burnscope_session"


def _cookie_token():
    try:
        value = st.context.cookies.get(COOKIE_NAME)
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def _cookie_script(value=None):
    """Set (or, with None, clear) the 'stay signed in' cookie in the browser. The value is a URL-safe random token,
    so it can sit in the script as is. It is written on the page that hosts the app, because the helper frame itself
    is not allowed to keep cookies."""
    max_age = 0 if value is None else US.SESSION_SECONDS
    script = f"""<script>
      (function () {{
        var host = window.parent;
        var secure = host.location.protocol === 'https:' ? '; Secure' : '';
        host.document.cookie = '{COOKIE_NAME}={value or ""}; Max-Age={max_age}; Path=/; SameSite=Lax' + secure;
      }})();
    </script>"""
    if hasattr(st, "iframe"):                                  # newer Streamlit; components.v1.html is deprecated there
        st.iframe(script, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(script, height=0)


def _open_session(user, token):
    """Common set-up once a user is signed in, whether by passcode or by cookie."""
    _load_heavy()
    st.session_state.user = user
    st.session_state.auth_token = token
    st.session_state.nav = "Dashboard"
    st.session_state.sess = None
    st.session_state.up_n = 0
    try:
        st.session_state.due = service.Session.check_reminders(user)   # emails anything due (once) and lists it for a banner
    except Exception as e:
        print(f"[reminders] login check failed: {e}")
        st.session_state.due = []


def start_user_session(user):
    token = US.create_session_token(user["user_id"])
    _cookie_script(token)                      # right away: the model import below can take a minute on a cold start
    _open_session(user, token)
    st.session_state._set_cookie = token       # and again on the next run, in case the first one was cut short
    st.rerun()


def restore_login():
    """A page reload gives Streamlit a brand-new, empty session. If the browser still holds a valid cookie, sign the
    user straight back in and reopen the case they were working on."""
    if st.session_state.get("user"):
        return
    token = _cookie_token()
    user = US.user_for_token(token) if token else None
    if not user:
        return
    _open_session(user, token)
    try:
        case_id = US.get_active_case(token)
        if case_id:
            st.session_state.sess = service.Session.resume(user, case_id)
    except Exception as e:                                     # worst case the user lands on the start panel
        print(f"[restore] could not reopen the active case: {type(e).__name__}: {e}")
    st.session_state._set_cookie = token                       # pushes the cookie's expiry forward as well


def logout():
    US.revoke_token(st.session_state.get("auth_token"))
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state._clear_cookie = True


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #

def start_case(user, mode, case_id=None):
    try:
        with st.spinner("Opening the case..."):
            st.session_state.sess = service.Session.start(user, mode, case_id)
        US.set_active_case(st.session_state.get("auth_token"), st.session_state.sess.case_id)
    except service.SessionError as e:
        st.error(str(e))


def start_panel(user):
    st.markdown("#### Start")
    mode = st.radio("What would you like to do?", ["New case", "Follow-up visit"], horizontal=True, label_visibility="collapsed", key="start_mode")
    if mode == "New case":
        st.caption("Upload a photo of a burn to get a severity read, guidance and next steps.")
        if st.button("Start a new case", type="primary"):
            start_case(user, "new")
            st.rerun()
        return
    cases = [c for c in service.list_cases(user) if c["baseline_severity"]]
    if not cases:
        st.info("You have no earlier cases yet. Start a new case first.")
        return
    labels = {c["case_id"]: f"#{c['case_id'][-8:]} · {when(c['created_date'])} · latest {sev_label(c['latest_severity'])} · {c['visits']} visit(s)" for c in cases}
    picked = st.selectbox("Which case?", list(labels), format_func=labels.get)
    if st.button("Start follow-up visit", type="primary"):
        start_case(user, "followup", picked)
        st.rerun()


def dashboard(user):
    st.markdown("## Wound assessment")

    pending = st.session_state.pop("pending_followup", None)
    if pending:
        start_case(user, "followup", pending)

    for item in st.session_state.get("due", []):
        st.warning(f"Your follow-up for case #{item['case_id'][-8:]} was due on {when(item['due'])}. "
                   "Choose **Follow-up visit** below and upload a new photo.")

    sess = st.session_state.get("sess")
    if sess is None:
        start_panel(user)
        return

    left, right = st.columns([1.55, 1], gap="large")
    snap = sess.snapshot()

    with left:
        top = st.columns([3, 1])
        top[0].markdown(f'<span class="pill">Case <b>#{sess.case_id[-8:]}</b> · {"follow-up visit" if sess.is_followup else "new case"}</span>', unsafe_allow_html=True)
        if top[1].button("Change case", use_container_width=True):
            st.session_state.sess = None
            US.set_active_case(st.session_state.get("auth_token"), None)
            st.rerun()
        if snap["urgent"]:
            st.error(snap["urgent_banner"], icon="🚨")
        upload_panel(sess)
        if snap["classify"]:
            results_panel(sess, snap)
            if sess.is_followup:
                compare_panel(user, sess, snap)

    with right:
        chat_panel(sess, snap)
        if snap["classify"]:
            steps_panel(snap)
        report_panel(sess, snap)

    try:
        sess.save_if_changed()                                 # safety net: whatever the user did, a reload finds it
    except Exception as e:
        print(f"[session] could not save case {sess.case_id}: {type(e).__name__}: {e}")


def upload_panel(sess):
    with st.container(border=True):
        st.markdown("**1 · Upload a burn photo**")
        c1, c2 = st.columns(2)
        part = c1.text_input("Body part (optional)", placeholder="e.g. forearm", key=f"part_{sess.case_id}")
        age = c2.text_input("Age (optional)", placeholder="e.g. 28, or child", key=f"age_{sess.case_id}")
        n = st.session_state.get("up_n", 0)
        file = st.file_uploader("Photo (JPG or PNG, one wound per photo)", type=["jpg", "jpeg", "png"], key=f"up_{n}")
        with st.expander("Tips for a good photo"):
            st.markdown("- **Good light, no flash glare**: daylight works best\n- **Fill the frame**: get close enough to see the wound\n"
                        "- **Hold steady**: a blurred photo is rejected by the quality check\n- **Plain background** behind the area")
        samples = sess.sample_names()
        sample = st.selectbox("...or try a sample photo", ["-"] + samples, key=f"sample_{sess.case_id}") if samples else "-"
        if st.button("Analyse photo", type="primary", disabled=not (file or sample != "-")):
            context = ", ".join(x for x in [f"It is on my {part.strip()}" if part.strip() else "", f"the person is {age.strip()}" if age.strip() else ""] if x)
            try:
                with st.spinner("Analysing the photo. On the free AI tier this can take a minute or two..."):
                    if file:
                        sess.send_photo(file.getvalue(), file.name, context)
                    else:
                        sess.send_sample(sample, context)
            except service.SessionError as e:
                st.error(str(e))
            else:
                st.session_state.up_n = n + 1
                st.rerun()


def results_panel(sess, snap):
    c, s, q = snap["classify"], snap["segment"] or {}, snap["quality"] or {}
    verdict = snap["verdict"]
    with st.container(border=True):
        head = st.columns([2, 1])
        head[0].markdown("**2 · Assessment results**")
        label, _ = URGENCY[verdict]
        head[1].markdown(chip(label.title(), VERDICT_TONE[verdict]), unsafe_allow_html=True)
        a, b = st.columns(2)
        photo = sess.read_file(snap["photo_path"], "uploads")
        if photo:
            a.image(photo, caption="Uploaded photo", use_container_width=True)
        conf = c.get("calibrated_confidence")
        area = AREA_LABELS.get(s.get("area_band"), "not measured")
        wf = s.get("wound_fraction")
        rows = [
            ("Predicted severity", chip(sev_label(c.get("predicted_severity")), SEVERITY_TONE.get(c.get("predicted_severity"), "teal"))),
            ("Model confidence", f'<span class="bar"><i style="width:{(conf or 0) * 100:.0f}%"></i></span><b>{(conf or 0) * 100:.0f}%</b>'),
            ("Wound area", f"<b>{html.escape(area)}</b>" + (f" ({wf * 100:.0f}% of the photo)" if wf is not None else "")),
            ("Wound regions", "multiple separate regions" if s.get("multi_region") else ("single main region" if s else "not measured")),
            ("Image quality", "Acceptable" if q.get("verdict") == "acceptable" else html.escape(q.get("issue") or "not recorded")),
        ]
        b.markdown("<table class='kv'>" + "".join(f"<tr><td class='k'>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>", unsafe_allow_html=True)
        if c.get("decision") == "ABSTAIN":
            st.warning("The model's confidence is below the review threshold, so this reading is provisional. A clinician review is recommended as a precaution.")
        cam = sess.read_file(snap["gradcam_path"], "gradcam")
        if cam:
            g1, _ = st.columns([1, 1])
            g1.image(cam, caption="Grad-CAM: warmer regions are where the model looked most. It shows attention, not a diagnosis.", use_container_width=True)


def steps_panel(snap):
    due = (snap.get("next_followup") or {}).get("due")
    with st.container(border=True):
        st.markdown("**Recommended next steps**")
        rows = ""
        for i, (action, timing) in enumerate(ACTIONS[snap["verdict"]], 1):
            timing = f"On {when(due)}" if timing == "On the follow-up date" and due else timing
            rows += f"<tr><td class='k' style='width:6%'>{i}</td><td>{html.escape(action)}</td><td style='width:24%'><b>{html.escape(timing)}</b></td></tr>"
        st.markdown(f"<table class='kv'>{rows}</table>", unsafe_allow_html=True)
        if due:
            st.caption(f"Next check-in: {when(due)}. A calendar invite was emailed if reminders are on in your profile.")


def report_panel(sess, snap):
    with st.container(border=True):
        st.markdown("**Report**")
        preview = snap.get("report_preview")
        path = snap.get("report_path")
        if preview:
            st.info(preview["message"])
            st.caption("Reply **yes, I consent** in the chat to generate it, or ask for changes.")
        if path:
            result = sess.case_state.get("report_result") or {}
            st.success("Your report is ready." + (f" It was emailed to {result.get('recipient_email')}." if result.get("emailed") else ""))
            st.download_button("Download report (PDF)", sess.read_file(path, "reports") or b"", file_name=os.path.basename(path), mime="application/pdf")
        elif not preview:
            st.caption("Ask in the chat: **\"I'd like a copy of my report\"**, or **\"send my report to Dr X at name@email.com\"**. "
                       "You will see exactly what it contains before anything is created or sent.")


def compare_panel(user, sess, snap):
    record = service.case_record(user, sess.case_id) or {}
    base = record.get("baseline_result") or {}
    b_cls, b_seg = base.get("classify") or {}, base.get("segment") or {}
    c_cls, c_seg = snap["classify"], snap["segment"] or {}
    with st.expander("Compare with previous visit", expanded=True):
        a, b = st.columns(2)
        for col, tag, path, cls, seg in ((a, "First visit", base.get("image_path"), b_cls, b_seg), (b, "This visit", snap["photo_path"], c_cls, c_seg)):
            img = service.read_user_file(user, path, "uploads")
            if img:
                col.image(img, caption=tag, use_container_width=True)
            wf = seg.get("wound_fraction")
            col.markdown(f"{sev_label(cls.get('predicted_severity'))} · {'%.0f%% area' % (wf * 100) if wf is not None else 'area not measured'}")
        trend = _default_trend(b_cls.get("predicted_severity"), b_seg.get("wound_fraction"), c_cls.get("predicted_severity"), c_seg.get("wound_fraction"))
        if trend:
            st.info(trend)


QUICK = ["Explain my result", "Find nearby care centres", "I'd like a copy of my report"]


def chat_panel(sess, snap):
    st.markdown("**Ask BurnScope**")
    st.caption("Care advice, nearby care and your report all happen here. Answers are grounded in retrieved guidance, not general knowledge.")
    box = st.container(height=600, border=True)
    with box:
        if not sess.transcript:
            st.chat_message("assistant").write("Hi. Upload a photo on the left, or ask me about burn care. "
                                               "I can also find nearby care, prepare a report for you to download, or email it to you or a doctor you name.")
        for m in sess.transcript:
            st.chat_message(m["role"]).write(m["content"])
    cols = st.columns(len(QUICK))
    quick = next((q for col, q in zip(cols, QUICK) if col.button(q, use_container_width=True, key=f"q_{q}")), None)
    prompt = st.chat_input("Ask about this result or burn care...") or quick
    if prompt:
        with box:
            st.chat_message("user").write(prompt)
            with st.spinner("Thinking. Replies can take up to a minute on the free AI tier..."):
                sess.send(prompt)
        st.rerun()
    sources = []
    for p in snap["guidance"]:
        for src in [x.strip() for x in p["source"].replace(" (table)", "").split(";") if x.strip()]:
            if src not in sources:
                sources.append(src)
    if sources:
        with st.expander(f"Sources behind the latest guidance ({len(sources)})"):
            for src in sources:
                st.caption(src)


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #

def open_followup(case_id):
    st.session_state.nav = "Dashboard"
    st.session_state.pending_followup = case_id


def history_page(user):
    st.markdown("## History")
    cases = service.list_cases(user)
    reports = service.my_reports(user)
    if not cases:
        st.info("No saved cases yet. Your assessments will appear here.")
    for c in cases:
        with st.expander(f"#{c['case_id'][-8:]} · {when(c['created_date'])} · {sev_label(c['latest_severity'])} · {c['visits']} visit(s)"):
            cols = st.columns(3)
            cols[0].markdown(f"**First visit**  \n{sev_label(c['baseline_severity'])}")
            cols[1].markdown(f"**Latest**  \n{sev_label(c['latest_severity'])}")
            cols[2].markdown(f"**Next check-in**  \n{when(c['next_followup']) if c['next_followup'] else 'not scheduled'}")
            record = service.case_record(user, c["case_id"]) or {}
            photo = service.read_user_file(user, (record.get("baseline_result") or {}).get("image_path"), "uploads")
            if photo:
                st.image(photo, caption="First photo", width=260)
            st.button("Start a follow-up visit", key=f"fu_{c['case_id']}", on_click=open_followup, args=(c["case_id"],))
            for r in (r for r in reports if r["case_suffix"] == c["case_id"][-8:]):
                st.download_button(f"Download report · {datetime.fromtimestamp(r['modified']).strftime('%d %b %Y %H:%M')}",
                                   service.read_user_file(user, r["path"], "reports") or b"", file_name=r["name"], mime="application/pdf",
                                   key=f"dl_{r['name']}")


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #

def profile_page(user):
    st.markdown("## Profile")
    with st.container(border=True):
        name = st.text_input("Name", value=user["name"])
        st.text_input("Email", value=user["email"], disabled=True)
        reminders = st.toggle("Email me follow-up reminders and calendar invites", value=user["reminders_enabled"])
        if st.button("Save changes", type="primary"):
            try:
                st.session_state.user = US.update_profile(user["user_id"], name=name, reminders_enabled=reminders)
                st.success("Saved.")
            except US.AuthError as e:
                st.error(str(e))
    with st.container(border=True):
        st.markdown("**Delete my data**")
        st.caption("Permanently removes your account, cases, photos and reports from this app. This cannot be undone.")
        confirm = st.text_input("Type DELETE to confirm", key="del_confirm")
        if st.button("Delete everything", disabled=confirm != "DELETE"):
            US.delete_user_data(user["user_id"])
            logout()
            st.rerun()


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #

restore_login()
user = st.session_state.get("user")
if not user:
    login_page()
    if st.session_state.pop("_clear_cookie", False):
        _cookie_script(None)
else:
    _load_heavy()
    cookie_to_set = st.session_state.pop("_set_cookie", None)
    if cookie_to_set:
        _cookie_script(cookie_to_set)
    with st.sidebar:
        brand()
        st.markdown(f"<span class='pill'>{html.escape(user['name'])}</span>", unsafe_allow_html=True)
        st.write("")
        st.radio("Navigate", ["Dashboard", "History", "Profile"], key="nav", label_visibility="collapsed")
        st.write("")
        if st.button("Logout", use_container_width=True):
            logout()
            st.rerun()
        st.caption("BurnScope supports triage and monitoring. It does not replace a clinician.")
    {"Dashboard": dashboard, "History": history_page, "Profile": profile_page}[st.session_state.get("nav", "Dashboard")](st.session_state.user)
