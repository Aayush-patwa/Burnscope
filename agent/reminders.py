"""
Email sending and follow-up reminders (email only -- no SMS, no hosting needed).

Three layers, so a reminder still reaches the user when the app is not running at the due date:
  1. When a follow-up is scheduled, an email with a calendar invite (.ics, with an alarm) is sent, so
     the user's own calendar app does the reminding.
  2. While the app is open, a background thread emails any follow-up that has come due.
  3. When a user opens the app, anything due for THEM is emailed (once) and shown as a banner.

Reminder emails carry no health details: only the case number and the date.
"""

import os
import smtplib
import threading
import time
from datetime import date, datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from agent.case_store import all_due_followups, due_followups_for_user, mark_reminder_sent

SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 587


def mail_configured() -> bool:
    return bool(os.environ.get("EMAIL_ADDRESS") and os.environ.get("EMAIL_APP_PASSWORD"))


def send_mail(to: str, subject: str, body: str, attachments: list = None) -> bool:
    """
    Send one plain-text email. `attachments` is a list of (filename, bytes, subtype), e.g.
    ("report.pdf", data, "pdf") or ("followup.ics", data, "calendar"). Returns False (never raises)
    if mail is not configured or sending fails.
    """
    sender = os.environ.get("EMAIL_ADDRESS")
    app_password = os.environ.get("EMAIL_APP_PASSWORD")
    if not sender or not app_password:
        print("[mail] EMAIL_ADDRESS / EMAIL_APP_PASSWORD not set in .env -- skipping send.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = sender, to, subject
        msg.attach(MIMEText(body, "plain"))
        for filename, data, subtype in attachments or []:
            part = MIMEApplication(data, _subtype=subtype)
            part.add_header("Content-Disposition", "attachment", filename=filename)
            msg.attach(part)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"[mail] send failed: {type(e).__name__}: {e}")
        return False


def build_ics(case_id: str, due: str) -> str:
    """A one-event calendar file for the follow-up day, with an alarm at 09:00."""
    day = date.fromisoformat(due)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//BurnScope//Follow-up//EN", "CALSCALE:GREGORIAN", "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{case_id}-{due}@burnscope",
        f"DTSTAMP:{stamp}",
        f"DTSTART;VALUE=DATE:{day.strftime('%Y%m%d')}",
        f"DTEND;VALUE=DATE:{(day + timedelta(days=1)).strftime('%Y%m%d')}",
        "SUMMARY:BurnScope follow-up check-in",
        f"DESCRIPTION:Open BurnScope and upload a new photo of the burn (case {case_id[-8:]}).",
        "BEGIN:VALARM", "ACTION:DISPLAY", "DESCRIPTION:BurnScope follow-up check-in", "TRIGGER:PT9H", "END:VALARM",
        "END:VEVENT", "END:VCALENDAR",
    ]
    return "\r\n".join(lines) + "\r\n"


def send_followup_invite(user: dict, case_id: str, due: str) -> bool:
    body = (f"Hello {user['name']},\n\nYour BurnScope follow-up check-in for case {case_id[-8:]} is set for {due}. "
            "The attached calendar file adds it to your calendar with a reminder.\n\n"
            "On that day, open BurnScope and upload a new photo so the healing can be compared.\n\n"
            "BurnScope supports, and does not replace, a clinician's care. If the burn gets worse before then "
            "(spreading redness, pus, fever, more pain), seek medical care sooner.")
    return send_mail(user["email"], f"BurnScope follow-up on {due}", body,
                     [("burnscope_followup.ics", build_ics(case_id, due).encode("utf-8"), "calendar")])


def _send_due_email(item: dict, user: dict) -> bool:
    body = (f"Hello {user['name']},\n\nYour BurnScope follow-up for case {item['case_id'][-8:]} was due on {item['due']}. "
            "Please open BurnScope and upload a new photo to check how the burn is healing.\n\n"
            "If it is getting worse, do not wait: seek medical care.")
    return send_mail(user["email"], "BurnScope follow-up is due", body)


def send_due_reminders(user_id: str = None) -> list:
    """
    Email each due, not-yet-reminded follow-up once. With a user_id, only that user's cases (used when
    they open the app); without one, everyone's (the background thread). Returns the case IDs emailed.
    """
    from agent.user_store import get_user
    items = due_followups_for_user(user_id) if user_id else all_due_followups()
    sent = []
    for item in items:
        if item["reminder_sent"]:
            continue
        user = get_user(item["user_id"])
        if not user or not user["reminders_enabled"]:
            continue
        if _send_due_email(item, user):
            mark_reminder_sent(item["case_id"])
            sent.append(item["case_id"])
    return sent


_daemon = None


def start_reminder_daemon(interval_seconds: int = 1800) -> threading.Thread:
    """Start (once per process) a background thread that emails follow-ups as they come due."""
    global _daemon
    if _daemon and _daemon.is_alive():
        return _daemon

    def loop():
        while True:
            try:
                send_due_reminders()
            except Exception as e:
                print(f"[reminders] check failed: {type(e).__name__}: {e}")
            time.sleep(interval_seconds)

    _daemon = threading.Thread(target=loop, name="burnscope-reminders", daemon=True)
    _daemon.start()
    return _daemon
