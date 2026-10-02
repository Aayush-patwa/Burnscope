"""
User profiles for the demo app: name, email and a passcode (stored as a
salted PBKDF2 hash, never in clear), with created/updated/last-login
timestamps. Stored in Postgres when DATABASE_URL is set, else local SQLite.

This is demo-grade sign-in, not production security: the email address is not
verified here, and there is no password reset. It exists so that two or three
people can use the same install without seeing each other's data.
"""

import hashlib
import hmac
import os
import secrets
import shutil
import time
from datetime import date

from agent.case_store import IntegrityError, _get_connection, init_db
from agent.consent import is_valid_email
from agent.userctx import UPLOADS_ROOT, OUTPUTS_ROOT

MIN_PASSCODE_LEN = 6
PBKDF2_ROUNDS = 200_000
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300
SESSION_SECONDS = 14 * 24 * 3600     # a "stay signed in" cookie lasts 14 days after its last use


class AuthError(Exception):
    """Raised with a message that is safe to show the user."""


def _hash(passcode: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", passcode.encode("utf-8"), salt, PBKDF2_ROUNDS).hex()


def _public(row) -> dict:
    return {"user_id": row["user_id"], "name": row["name"], "email": row["email"],
            "reminders_enabled": bool(row["reminders_enabled"]), "created_date": row["created_date"]}


def register(name: str, email: str, passcode: str) -> dict:
    name, email = (name or "").strip(), (email or "").strip().lower()
    if not name:
        raise AuthError("Please enter your name.")
    if not is_valid_email(email):
        raise AuthError("Please enter a valid email address.")
    if len(passcode or "") < MIN_PASSCODE_LEN:
        raise AuthError(f"The passcode must be at least {MIN_PASSCODE_LEN} characters.")
    init_db()
    salt = secrets.token_bytes(16)
    user_id = f"u_{secrets.token_hex(8)}"
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO users (user_id, name, email, salt, pass_hash, created_date, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (user_id, name, email, salt.hex(), _hash(passcode, salt), date.today().isoformat()))
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    except IntegrityError:
        raise AuthError("An account with this email already exists. Please sign in.")
    finally:
        conn.close()
    return _public(row)


def authenticate(email: str, passcode: str) -> dict:
    email = (email or "").strip().lower()
    generic = AuthError("That email and passcode do not match.")
    conn = _get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            raise generic
        if row["locked_until"] and row["locked_until"] > time.time():
            raise AuthError("Too many wrong attempts. Please wait a few minutes and try again.")
        ok = hmac.compare_digest(_hash(passcode or "", bytes.fromhex(row["salt"])), row["pass_hash"])
        if not ok:
            failed = (row["failed_attempts"] or 0) + 1
            locked = time.time() + LOCKOUT_SECONDS if failed >= MAX_FAILED_ATTEMPTS else 0
            conn.execute("UPDATE users SET failed_attempts = ?, locked_until = ? WHERE user_id = ?",
                         (0 if locked else failed, locked, row["user_id"]))
            conn.commit()
            raise generic
        conn.execute("UPDATE users SET failed_attempts = 0, locked_until = 0, last_login = CURRENT_TIMESTAMP "
                     "WHERE user_id = ?", (row["user_id"],))
        conn.commit()
        return _public(row)
    finally:
        conn.close()


def _token_hash(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def create_session_token(user_id: str) -> str:
    """A random 'stay signed in' token for the browser cookie. Only its hash is stored, so a copy of the database
    cannot be used to sign in, and it can be revoked at any time."""
    token = secrets.token_urlsafe(32)
    now = time.time()
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (now,))
        conn.execute("INSERT INTO auth_sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
                     (_token_hash(token), user_id, now + SESSION_SECONDS, now))
        conn.commit()
    finally:
        conn.close()
    return token


def user_for_token(token: str):
    """The signed-in user a cookie token belongs to, or None if it is unknown, expired or revoked.
    Every successful use pushes the expiry forward (sliding window)."""
    if not isinstance(token, str) or not token:
        return None
    now = time.time()
    conn = _get_connection()
    try:
        row = conn.execute("SELECT user_id, expires_at FROM auth_sessions WHERE token_hash = ?", (_token_hash(token),)).fetchone()
        if row is None:
            return None
        if row["expires_at"] < now:
            conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_token_hash(token),))
            conn.commit()
            return None
        user = conn.execute("SELECT * FROM users WHERE user_id = ?", (row["user_id"],)).fetchone()
        if user is None:
            return None
        conn.execute("UPDATE auth_sessions SET expires_at = ? WHERE token_hash = ?", (now + SESSION_SECONDS, _token_hash(token)))
        conn.commit()
        return _public(user)
    finally:
        conn.close()


def revoke_token(token: str) -> None:
    if not token:
        return
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_token_hash(token),))
        conn.commit()
    finally:
        conn.close()


def set_active_case(token: str, case_id) -> None:
    """Remember which case this browser session had open, so a reload can reopen it (None = no open case)."""
    if not token:
        return
    conn = _get_connection()
    try:
        conn.execute("UPDATE auth_sessions SET active_case_id = ? WHERE token_hash = ?", (case_id, _token_hash(token)))
        conn.commit()
    finally:
        conn.close()


def get_active_case(token: str):
    if not token:
        return None
    conn = _get_connection()
    try:
        row = conn.execute("SELECT active_case_id FROM auth_sessions WHERE token_hash = ?", (_token_hash(token),)).fetchone()
        return row["active_case_id"] if row else None
    finally:
        conn.close()


def get_user(user_id: str):
    conn = _get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return _public(row) if row else None
    finally:
        conn.close()


def update_profile(user_id: str, name: str = None, reminders_enabled: bool = None) -> dict:
    conn = _get_connection()
    try:
        if name is not None:
            if not name.strip():
                raise AuthError("Please enter your name.")
            conn.execute("UPDATE users SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                         (name.strip(), user_id))
        if reminders_enabled is not None:
            conn.execute("UPDATE users SET reminders_enabled = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                         (int(bool(reminders_enabled)), user_id))
        conn.commit()
    finally:
        conn.close()
    return get_user(user_id)


def delete_user_data(user_id: str) -> None:
    """'Delete my data': the user's cases, photos, reports and profile."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM cases WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM case_sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()
    for root in (UPLOADS_ROOT, OUTPUTS_ROOT):
        shutil.rmtree(os.path.join(root, user_id), ignore_errors=True)
