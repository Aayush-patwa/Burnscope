"""
Code-level gate for generate_escalation_report.

Consent used to be a boolean the model filled in, so a plain "generate a
report and email it to X" produced a sent report with no consent
question. This module decides from the conversation itself:

  granted  =  the user's latest message is an explicit yes / "I consent"
              with no negation
              AND (the assistant's previous message asked for consent,
                   OR the user stated their consent outright)

and the recipient address must be well-formed AND typed by the user
(the model may not invent, guess or "fix" one).

check_report_request() returns None when the call may run, otherwise a
result dict (no "error" key) telling the model what to do next.
"""

import re

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})

# Clear agreement. Imperatives such as "send it" only count when nothing negates them
# ("don't send it" must not read as agreement).
_AFFIRM_STRONG = re.compile(
    r"\b(?:yes|yeah|yep|yup|sure|please\s+do|go\s+ahead|proceed|absolutely|of\s+course|"
    r"i\s+agree|sounds\s+good)\b", re.IGNORECASE)
_AFFIRM_IMPERATIVE = re.compile(r"\b(?:do\s+it|send\s+it|generate\s+it)\b", re.IGNORECASE)
_SELF_CONSENT = re.compile(
    r"\b(?:i|we)\s+(?:do\s+|hereby\s+|fully\s+)?(?:consent|agree|authori[sz]e)\b"
    r"|\b(?:you\s+have|with)\s+my\s+(?:consent|permission)\b"
    r"|\bi\s+give\s+(?:you\s+)?(?:my\s+)?(?:consent|permission)\b", re.IGNORECASE)
_NEGATIVE = re.compile(
    r"\b(?:no|nope|nah|not|never|cancel|stop|decline|refuse|withdraw|revoke|hold\s+off|wait|"
    r"maybe|unsure|perhaps|rather\s+not)\b|n't\b", re.IGNORECASE)

_ASK_WORDING = re.compile(
    r"\b(?:consent|permission|authori[sz]e|okay\s+to|ok\s+to|(?:may|can|shall|should)\s+i|"
    r"would\s+you\s+like\s+me\s+to|do\s+you\s+want\s+me\s+to|"
    r"are\s+you\s+(?:happy|ok|okay)\s+(?:for\s+me\s+)?to)\b", re.IGNORECASE)
_ASK_TOPIC = re.compile(r"\b(?:report|e-?mail|send|share|generate|create|pdf|summary)\b", re.IGNORECASE)


def is_valid_email(address) -> bool:
    if not isinstance(address, str):
        return False
    a = address.strip()
    if not a or len(a) > 254 or not _EMAIL_RE.match(a):
        return False
    local = a.rsplit("@", 1)[0]
    return ".." not in a and not local.startswith(".") and not local.endswith(".")


def latest_user_message(history: list) -> str:
    for m in reversed(history):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def assistant_asked_for_consent(history: list) -> bool:
    """Did the assistant's last text message before the user's latest message ask for consent?"""
    last_user = None
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user":
            last_user = i
            break
    if last_user is None:
        return False
    for i in range(last_user - 1, -1, -1):
        m = history[i]
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            text = m["content"]
            return bool(_ASK_WORDING.search(text) and _ASK_TOPIC.search(text))
    return False


def email_typed_by_user(history: list, address: str) -> bool:
    needle = (address or "").strip().lower()
    return bool(needle) and any(
        needle in (m.get("content") or "").lower() for m in history if m.get("role") == "user"
    )


def interpret_reply(text: str) -> str:
    """'declined' (a clear no), 'yes' (explicit yes / own consent), or 'unclear'."""
    t = (text or "").translate(_APOSTROPHES)
    negative = bool(_NEGATIVE.search(t))
    strong_yes = bool(_AFFIRM_STRONG.search(t) or _SELF_CONSENT.search(t))
    if negative:
        return "unclear" if strong_yes else "declined"   # "yes but..." is ambiguous; a bare no is a refusal
    return "yes" if (strong_yes or _AFFIRM_IMPERATIVE.search(t)) else "unclear"


def validate_recipient(history: list, tool_args: dict, own_email: str = None):
    """
    None if recipient_email (if any) is acceptable; otherwise invalid_email / recipient_not_provided.

    Used for BOTH preview_only and the real call (see agent/loop.py), so a recipient the model invented
    is refused before it is even shown to the user in a preview -- not only when consent is requested.
    `own_email` is the signed-in user's account address: "send it to me" may use it without retyping it.
    """
    email = (tool_args.get("recipient_email") or "").strip()
    if not email:
        return None
    if not is_valid_email(email):
        return {"status": "invalid_email",
                "message": (f"'{email}' is not a valid email address, so NOTHING was generated, previewed or sent. "
                            "Ask the user to type the doctor's email address again. "
                            "Do not guess or correct it yourself.")}
    is_own = bool(own_email) and email.lower() == own_email.strip().lower()
    if not is_own and not email_typed_by_user(history, email):
        return {"status": "recipient_not_provided",
                "message": ("That email address was not typed by the user, so NOTHING was generated, previewed or "
                            "sent. Ask the user for the recipient's email address yourself; never invent, guess, "
                            "or use a placeholder address.")}
    return None


def check_report_request(history: list, tool_args: dict, own_email: str = None):
    """None if generate_escalation_report may run; otherwise a result dict for the model."""
    blocked = validate_recipient(history, tool_args, own_email)
    if blocked is not None:
        return blocked

    reply = latest_user_message(history)
    verdict = interpret_reply(reply)
    # A refusal only means something as the answer to a consent question. A request that merely
    # contains "not" ("a copy for my records, not for any doctor") is not a refusal.
    if verdict == "declined" and assistant_asked_for_consent(history):
        return {"status": "declined_by_user",
                "message": ("The user declined or withdrew consent. Do NOT generate or send any report. "
                            "Acknowledge their choice and offer other help.")}
    stated_own_consent = bool(_SELF_CONSENT.search(reply.translate(_APOSTROPHES)))
    if verdict == "yes" and (stated_own_consent or assistant_asked_for_consent(history)):
        return None
    email = (tool_args.get("recipient_email") or "").strip()
    target = f" and emailing it to {email}" if email else ""
    return {"status": "consent_required",
            "message": ("The report was NOT generated: the user has not explicitly consented. In one plain sentence, "
                        f"ask the user whether they consent to you generating this report{target}, "
                        "then wait for their reply. Do not call this tool again until they answer yes.")}
