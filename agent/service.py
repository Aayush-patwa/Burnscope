"""
The application service layer: what a web UI calls instead of the terminal loop.

One `Session` = one signed-in user working on one case. Everything it does runs
inside `acting_as(user_id)`, so the case store, tools and files only ever touch
that user's data (see agent/userctx.py).

    session = Session.start(user, mode="new")            # or mode="followup", case_id=...
    session.send_photo(jpeg_bytes, "wound.jpg")           # quality check, classify, segment, guidance...
    session.send("Why is it not fully confident?")        # a chat turn
    session.snapshot()                                    # structured results for the dashboard cards
"""

import hashlib
import io
import json
import os
import uuid

from PIL import Image, UnidentifiedImageError

import agent.loop as loop
from agent.case_store import (create_new_case, get_case, init_db, list_active_cases,
                              load_session_blob, save_session_blob)
from agent.dispatch import execute_tool
from agent.context import build_followup_message, build_session_case_message
from agent.system_prompt import SYSTEM_PROMPT
from agent.userctx import acting_as, path_is_users, user_dir

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_SIDE_PX = 1600
SAMPLE_DIR = os.path.join("data", "sample_images")

# Shown as a full-width banner whenever the verdict is urgent (the UI decides how to draw it).
URGENT_BANNER = ("This may need urgent medical attention. Go to the nearest emergency department or call your local "
                 "emergency number now (112 in India, 911 in the US, 999 in the UK). Do not wait for a follow-up.")


def _json_default(obj):
    """Tool results can carry NumPy numbers; store them as plain JSON."""
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


class SessionError(Exception):
    """A problem the user can be told about (bad file, unknown case)."""


class Session:
    def __init__(self, user: dict, case_id: str, is_followup: bool, history: list):
        self.user = user
        self.case_id = case_id
        self.is_followup = is_followup
        self.history = history
        self.case_state = {
            "classify_result": None, "segment_result": None, "quality_result": None,
            "gradcam_result": None, "guidance_result": None, "clinics_result": None,
            "report_result": None, "report_preview": None, "report_preview_sig": None, "locked_image_path": None,
            "baseline_saved": is_followup, "is_followup": is_followup,
            "followup_saved": False, "segment_persisted": False, "persisted_image_path": None,
        }
        self.transcript = []     # [{"role": "user"|"assistant", "content": str}] for the chat window

    # ------------------------------------------------------------------ start
    @classmethod
    def start(cls, user: dict, mode: str = "new", case_id: str = None) -> "Session":
        with acting_as(user["user_id"]):
            init_db()
            existing = None
            if mode == "followup":
                existing = get_case(case_id) if case_id else None
                if existing is None:
                    raise SessionError("That case was not found in your history.")
                if not existing.get("baseline_result"):
                    raise SessionError("That case has no saved assessment yet, so it cannot be followed up.")
            else:
                case_id = create_new_case()
            is_followup = existing is not None
            history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": (
                    f"The case_id for this session is: {case_id}. Always use this EXACT case_id when calling "
                    f"schedule_followup or generate_escalation_report. Never invent your own case_id.")},
                {"role": "system", "content": build_session_case_message(case_id, is_followup)},
                {"role": "system", "content": (
                    f"The signed-in user's own account email is {user['email']}. If they ask for the report to be sent "
                    "to them / to their own email, use exactly that address as recipient_email.")},
            ]
            if is_followup:
                history.append({"role": "system", "content": build_followup_message(existing)})
        session = cls(user, case_id, is_followup, history)
        session.case_state["own_email"] = user["email"]
        session.save()
        return session

    # ------------------------------------------------------------------ persistence (a page reload must not lose the work)
    def _blob(self) -> str:
        return json.dumps({"is_followup": self.is_followup, "history": self.history,
                           "case_state": self.case_state, "transcript": self.transcript}, default=_json_default)

    def save(self) -> None:
        blob = self._blob()
        self._saved_digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()
        save_session_blob(self.case_id, self.user["user_id"], blob)

    def save_if_changed(self) -> None:
        if hashlib.sha1(self._blob().encode("utf-8")).hexdigest() != getattr(self, "_saved_digest", None):
            self.save()

    @classmethod
    def resume(cls, user: dict, case_id: str):
        """Reopen a case exactly where this user left it (chat, results, report preview), or None if it cannot be restored."""
        if not case_id:
            return None
        raw = load_session_blob(case_id, user["user_id"])
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        with acting_as(user["user_id"]):
            if get_case(case_id) is None:                      # deleted, or not this user's
                return None
        session = cls(user, case_id, bool(data.get("is_followup")), data.get("history") or [])
        session.case_state.update(data.get("case_state") or {})
        session.case_state["own_email"] = user["email"]
        session.transcript = data.get("transcript") or []
        session._saved_digest = hashlib.sha1(session._blob().encode("utf-8")).hexdigest()
        return session

    # ------------------------------------------------------------------ photos
    def save_photo(self, data: bytes, filename: str = "") -> str:
        """Validate an uploaded image and store a clean copy in this user's folder. Returns its path."""
        if not data:
            raise SessionError("The file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise SessionError(f"The photo is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Please use a smaller one.")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError, ValueError):
            raise SessionError("That file could not be read as an image. Please upload a JPG or PNG photo.")
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
        with acting_as(self.user["user_id"]):
            path = os.path.join(user_dir("uploads"), f"{self.case_id[-8:]}_{uuid.uuid4().hex[:8]}.jpg")
        img.save(path, "JPEG", quality=90)    # re-encoding also drops EXIF data such as GPS position
        return path

    def sample_names(self) -> list:
        try:
            return sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        except OSError:
            return []

    def save_sample(self, name: str) -> str:
        if name not in self.sample_names():
            raise SessionError("Unknown sample photo.")
        with open(os.path.join(SAMPLE_DIR, name), "rb") as f:
            return self.save_photo(f.read(), name)

    def send_photo(self, data: bytes, filename: str = "", context: str = "") -> dict:
        return self._photo_turn(self.save_photo(data, filename), context)

    def send_sample(self, name: str, context: str = "") -> dict:
        return self._photo_turn(self.save_sample(name), context)

    def _photo_turn(self, path: str, context: str = "") -> dict:
        """`context` is what the user typed about the burn (body part, age), so the agent need not ask for it."""
        context = (context or "").strip()
        text = "Here is a photo of my burn." + (f" {context}" if context else "") + f" image_path: {path}"
        return self.send(text, shown_as="(uploaded a photo)" + (f" - {context}" if context else ""))

    # ------------------------------------------------------------------ chat
    def send(self, text: str, shown_as: str = None) -> dict:
        """One chat turn. Returns {"answer", "error", "guard"}; never raises for service problems."""
        text = (text or "").strip()
        if not text:
            return {"answer": "", "error": "empty", "guard": None}
        self.transcript.append({"role": "user", "content": shown_as or text})
        self.history.append({"role": "user", "content": text})
        with acting_as(self.user["user_id"]):
            result = loop.run_agent_turn(self.history, self.case_id, self.case_state)
        self.history = result["history"]
        self._ensure_gradcam()
        self.transcript.append({"role": "assistant", "content": result["final_answer"]})
        try:
            self.save()
        except Exception as e:                                  # a storage hiccup must never lose the answer the user is waiting for
            print(f"[session] could not save case {self.case_id}: {type(e).__name__}: {e}")
        return {"answer": result["final_answer"], "error": result["error"], "guard": result["guard"]}

    def _ensure_gradcam(self) -> None:
        """The dashboard always shows the model-attention image, so make it here instead of leaving it to the model."""
        cs = self.case_state
        c = cs["classify_result"]
        if not c or not cs["locked_image_path"] or cs.get("gradcam_for") == cs["locked_image_path"]:
            return
        with acting_as(self.user["user_id"]):
            result = execute_tool("generate_gradcam", {"image_path": cs["locked_image_path"],
                                                       "predicted_class": c["predicted_severity"]})
        if "error" not in result:
            cs["gradcam_result"] = result
            cs["gradcam_for"] = cs["locked_image_path"]

    # ------------------------------------------------------------------ results for the UI
    def snapshot(self) -> dict:
        """Structured facts for the dashboard cards, taken from the real tool results."""
        from tools.escalate import derive_verdict
        cs = self.case_state
        c, s, q = cs["classify_result"], cs["segment_result"], cs["quality_result"]
        verdict = None
        if c:
            verdict = derive_verdict(c.get("predicted_severity"), c.get("decision"),
                                     (s or {}).get("area_band"), (s or {}).get("multi_region"))
        with acting_as(self.user["user_id"]):
            record = get_case(self.case_id)
        return {
            "case_id": self.case_id,
            "is_followup": self.is_followup,
            "quality": q,
            "classify": c,
            "segment": s,
            "verdict": verdict,
            "gradcam_path": (cs["gradcam_result"] or {}).get("overlay_path"),
            "photo_path": cs["locked_image_path"],
            "guidance": (cs["guidance_result"] or {}).get("passages", []),
            "clinics": (cs["clinics_result"] or {}).get("clinics", []),
            "next_followup": (record or {}).get("next_followup"),
            "urgent": verdict == "urgent_care",
            "urgent_banner": URGENT_BANNER if verdict == "urgent_care" else None,
            "report_preview": cs.get("report_preview") if cs.get("report_preview_sig") else None,
            "report_path": self.report_path(),
        }

    def report_path(self):
        rr = self.case_state.get("report_result")
        path = rr.get("pdf_path") if rr else None
        with acting_as(self.user["user_id"]):
            return path if path and path_is_users(path, "reports") and os.path.exists(path) else None

    def read_file(self, path: str, kind: str):
        """Bytes of one of this user's own files (for display/download); None for anything else."""
        with acting_as(self.user["user_id"]):
            if not path or not path_is_users(path, kind) or not os.path.isfile(path):
                return None
        with open(path, "rb") as f:
            return f.read()

    @staticmethod
    def check_reminders(user: dict) -> list:
        """On login: email this user's newly due follow-ups (once) and return every due/overdue one for a banner."""
        from agent.case_store import due_followups_for_user
        from agent.reminders import send_due_reminders
        send_due_reminders(user["user_id"])
        return due_followups_for_user(user["user_id"])

    def my_cases(self, limit: int = 50) -> list:
        with acting_as(self.user["user_id"]):
            return list_active_cases(limit)


# --------------------------------------------------------------------------- #
# Helpers for the History and Profile pages (they act for one signed-in user)
# --------------------------------------------------------------------------- #

def list_cases(user: dict, limit: int = 100) -> list:
    """All of this user's saved cases, newest first."""
    with acting_as(user["user_id"]):
        return list_active_cases(limit, all_time=True)


def case_record(user: dict, case_id: str):
    with acting_as(user["user_id"]):
        return get_case(case_id)


def my_reports(user: dict) -> list:
    """This user's generated reports: [{"name", "path", "case_suffix", "modified"}], newest first."""
    with acting_as(user["user_id"]):
        folder = user_dir("reports", create=False)
    if not os.path.isdir(folder):
        return []
    out = []
    for name in os.listdir(folder):
        if name.lower().endswith(".pdf"):
            path = os.path.join(folder, name)
            parts = name[:-4].split("_")           # report_<last 8 of case id>_<date>_<time>
            out.append({"name": name, "path": path, "case_suffix": parts[1] if len(parts) > 1 else "",
                        "modified": os.path.getmtime(path)})
    return sorted(out, key=lambda r: r["modified"], reverse=True)


def read_user_file(user: dict, path: str, kind: str):
    """Bytes of one of this user's own files ('uploads', 'gradcam' or 'reports'); None for anything else."""
    with acting_as(user["user_id"]):
        if not path or not path_is_users(path, kind) or not os.path.isfile(path):
            return None
    with open(path, "rb") as f:
        return f.read()
