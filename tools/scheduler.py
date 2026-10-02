"""
Tool: schedule_followup
Purpose: set a concrete follow-up check-in, interval decided case-by-case
(lighter-touch for self-care, tighter for clinical-review/urgent-care).

The date is saved with the case (data/cases.db), so it survives the session
and is shown when the user returns. For a signed-in user an email with a
calendar invite is sent (reminder_email: "sent" / "failed" / "off"); a due-date
email follows via agent/reminders.py while the app runs.
"""

from datetime import date, timedelta

from agent.case_store import get_case, set_next_followup
from agent.reminders import send_followup_invite
from agent.user_store import get_user
from agent.userctx import current_user_id

MIN_INTERVAL_DAYS = 1
MAX_INTERVAL_DAYS = 30


def schedule_followup(case_id: str, interval_days: int) -> dict:
    """
    Args:
        case_id: identifier for this patient/case.
        interval_days: how many days until the next check-in (1-30).

    Returns:
        dict with keys scheduled_date (ISO date) and saved (True), or
        {"error": ...} if the interval is invalid or the case is unknown.
    """
    if isinstance(interval_days, float) and interval_days.is_integer():
        interval_days = int(interval_days)
    if isinstance(interval_days, bool) or not isinstance(interval_days, int) \
            or not MIN_INTERVAL_DAYS <= interval_days <= MAX_INTERVAL_DAYS:
        return {"error": f"interval_days must be a whole number of days between "
                         f"{MIN_INTERVAL_DAYS} and {MAX_INTERVAL_DAYS}."}
    if get_case(case_id) is None:
        return {"error": f"No case found with id {case_id}."}

    scheduled = date.today() + timedelta(days=interval_days)
    set_next_followup(case_id, scheduled.isoformat(), interval_days)

    # Email a calendar invite so the user's own calendar reminds them (signed-in users only).
    reminder = "not_signed_in"
    user = get_user(current_user_id()) if current_user_id() else None
    if user:
        if not user["reminders_enabled"]:
            reminder = "off"
        else:
            reminder = "sent" if send_followup_invite(user, case_id, scheduled.isoformat()) else "failed"
    return {"scheduled_date": scheduled.isoformat(), "saved": True, "reminder_email": reminder}
