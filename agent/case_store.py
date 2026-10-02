"""
Case history store -- generates real case IDs (never left to the LLM
to invent) and persists case data across runs.

Backend: Postgres when DATABASE_URL is set (e.g. a Neon pooled connection
string), otherwise a local SQLite file at DB_PATH. All SQL below is written
once with "?" placeholders; _PgConnection translates it for Postgres.
"""

import sqlite3
import time
import uuid
import json
from datetime import date, timedelta

import psycopg
from psycopg.rows import dict_row

from agent.userctx import current_user_id

import os

DB_PATH = os.environ.get("BURNSCOPE_DB", "data/cases.db")   # override to use a different database file
ACTIVE_CASE_DAYS = 90   # a saved case counts as "active" for this long after it was created

IntegrityError = (sqlite3.IntegrityError, psycopg.IntegrityError)


def _database_url():
    return os.environ.get("DATABASE_URL", "").strip()


class _PgConnection:
    """Gives a psycopg connection the small sqlite3-style surface this module uses."""

    def __init__(self, url):
        # prepare_threshold=None: Neon's pooler (PgBouncer, transaction mode) can't hold prepared statements.
        self._conn = psycopg.connect(url, prepare_threshold=None, row_factory=dict_row)

    @staticmethod
    def _translate(sql):
        return sql.replace("IS ?", "IS NOT DISTINCT FROM ?").replace("?", "%s")

    def execute(self, sql, params=()):
        cursor = self._conn.cursor()
        cursor.execute(self._translate(sql), params)
        return cursor

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


def _get_connection():
    url = _database_url()
    if url:
        return _PgConnection(url)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Every case belongs to one user (or to nobody, for the terminal CLI). All reads
# and writes below are restricted to the ACTIVE user's cases -- see agent/userctx.py.
# `IS ?` also matches NULL, so an unowned session only ever sees unowned cases.
OWNER_SQL = "user_id IS ?"


def init_db():
    """Create the tables if they don't exist yet and migrate older databases. Safe to call every run."""
    if _database_url():
        _init_postgres()
        return
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            created_date TEXT,
            baseline_result TEXT,
            followups TEXT
        )
    """)
    # Older databases were created without the follow-up column.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(cases)")}
    if "next_followup" not in columns:
        conn.execute("ALTER TABLE cases ADD COLUMN next_followup TEXT")
    if "user_id" not in columns:
        conn.execute("ALTER TABLE cases ADD COLUMN user_id TEXT")   # NULL = unowned (legacy / CLI)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            salt TEXT NOT NULL,
            pass_hash TEXT NOT NULL,
            created_date TEXT,
            reminders_enabled INTEGER DEFAULT 1,
            failed_attempts INTEGER DEFAULT 0,
            locked_until REAL DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            last_login TEXT
        )
    """)
    user_columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
    for column in ("created_at", "updated_at", "last_login"):
        if column not in user_columns:
            conn.execute(f"ALTER TABLE users ADD COLUMN {column} TEXT")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS auth_sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at REAL NOT NULL,
            active_case_id TEXT,
            created_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS case_sessions (
            case_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            data TEXT NOT NULL,
            updated_at REAL
        )
    """)
    conn.commit()
    conn.close()


def _init_postgres():
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                created_date TEXT,
                baseline_result TEXT,
                followups TEXT,
                next_followup TEXT,
                user_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                salt TEXT NOT NULL,
                pass_hash TEXT NOT NULL,
                created_date TEXT,
                reminders_enabled INTEGER DEFAULT 1,
                failed_attempts INTEGER DEFAULT 0,
                locked_until DOUBLE PRECISION DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_login TIMESTAMPTZ
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS cases_user_id_idx ON cases (user_id)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at DOUBLE PRECISION NOT NULL,
                active_case_id TEXT,
                created_at DOUBLE PRECISION
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS case_sessions (
                case_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                data TEXT NOT NULL,
                updated_at DOUBLE PRECISION
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_session_blob(case_id: str, user_id: str, data: str) -> None:
    """Store the in-progress conversation of one case so a page reload can pick it up again."""
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO case_sessions (case_id, user_id, data, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (case_id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at "
            "WHERE case_sessions.user_id = excluded.user_id",
            (case_id, user_id, data, time.time()))
        conn.commit()
    finally:
        conn.close()


def load_session_blob(case_id: str, user_id: str):
    """The saved conversation of this user's case, or None."""
    conn = _get_connection()
    try:
        row = conn.execute("SELECT data FROM case_sessions WHERE case_id = ? AND user_id = ?", (case_id, user_id)).fetchone()
        return row["data"] if row else None
    finally:
        conn.close()


def create_new_case() -> str:
    """
    Generates a brand-new, guaranteed-unique case ID and creates an
    empty case record. Call this ONCE at the start of a new user
    session -- never let the LLM invent its own case_id.
    """
    case_id = f"case_{uuid.uuid4().hex[:8]}"
    conn = _get_connection()
    conn.execute(
        "INSERT INTO cases (case_id, created_date, baseline_result, followups, user_id) VALUES (?, ?, ?, ?, ?)",
        (case_id, date.today().isoformat(), json.dumps({}), json.dumps([]), current_user_id()),
    )
    conn.commit()
    conn.close()
    return case_id


def save_baseline_result(case_id: str, classify_result: dict, segment_result: dict, image_path: str = None):
    """Store the first classify+segment result for a case as its baseline (with its photo, for later comparison)."""
    conn = _get_connection()
    baseline = {"classify": classify_result, "segment": segment_result, "image_path": image_path}
    conn.execute(
        f"UPDATE cases SET baseline_result = ? WHERE case_id = ? AND {OWNER_SQL}",
        (json.dumps(baseline), case_id, current_user_id()),
    )
    conn.commit()
    conn.close()


def add_followup_result(case_id: str, classify_result: dict, segment_result: dict, image_path: str = None):
    """Append a follow-up visit's result to this case's history."""
    conn = _get_connection()
    row = conn.execute(f"SELECT followups FROM cases WHERE case_id = ? AND {OWNER_SQL}",
                       (case_id, current_user_id())).fetchone()
    if row is None:
        conn.close()
        return
    followups = json.loads(row["followups"])
    followups.append({
        "date": date.today().isoformat(),
        "classify": classify_result,
        "segment": segment_result,
        "image_path": image_path,
    })
    conn.execute(
        f"UPDATE cases SET followups = ? WHERE case_id = ? AND {OWNER_SQL}",
        (json.dumps(followups), case_id, current_user_id()),
    )
    conn.commit()
    conn.close()


def update_segment_result(case_id: str, segment_result: dict, is_followup: bool) -> bool:
    """
    Attach a segmentation result to a visit that was already saved.

    Segmentation usually runs after classification, i.e. after the visit
    was first written, so the wound area has to be added afterwards. For
    a follow-up session the most recent follow-up entry is updated,
    otherwise the baseline. Returns False if there is no saved visit yet.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            f"SELECT baseline_result, followups FROM cases WHERE case_id = ? AND {OWNER_SQL}",
            (case_id, current_user_id())
        ).fetchone()
        if row is None:
            return False
        if is_followup:
            followups = json.loads(row["followups"])
            if not followups:
                return False
            followups[-1]["segment"] = segment_result
            conn.execute(
                f"UPDATE cases SET followups = ? WHERE case_id = ? AND {OWNER_SQL}",
                (json.dumps(followups), case_id, current_user_id()),
            )
        else:
            baseline = json.loads(row["baseline_result"])
            if not baseline:
                return False
            baseline["segment"] = segment_result
            conn.execute(
                f"UPDATE cases SET baseline_result = ? WHERE case_id = ? AND {OWNER_SQL}",
                (json.dumps(baseline), case_id, current_user_id()),
            )
        conn.commit()
        return True
    finally:
        conn.close()


def set_next_followup(case_id: str, due_date: str, interval_days: int) -> bool:
    """Record the next scheduled check-in (ISO date) for a case. Returns False if the case does not exist."""
    conn = _get_connection()
    try:
        record = json.dumps({"due": due_date, "interval_days": interval_days, "scheduled_on": date.today().isoformat(),
                             "reminder_sent": False})
        cursor = conn.execute(f"UPDATE cases SET next_followup = ? WHERE case_id = ? AND {OWNER_SQL}",
                              (record, case_id, current_user_id()))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _due_rows(sql_extra: str, params: tuple, today: str) -> list:
    conn = _get_connection()
    try:
        rows = conn.execute("SELECT case_id, user_id, next_followup FROM cases "
                            "WHERE next_followup IS NOT NULL AND user_id IS NOT NULL" + sql_extra, params).fetchall()
    finally:
        conn.close()
    due = []
    for row in rows:
        nxt = json.loads(row["next_followup"])
        if nxt.get("due") and nxt["due"] <= today:
            due.append({"case_id": row["case_id"], "user_id": row["user_id"], "due": nxt["due"],
                        "reminder_sent": bool(nxt.get("reminder_sent"))})
    return sorted(due, key=lambda d: d["due"])


def due_followups_for_user(user_id: str, today: str = None) -> list:
    """Follow-ups of ONE user that are due today or overdue."""
    return _due_rows(" AND user_id = ?", (user_id,), today or date.today().isoformat())


def all_due_followups(today: str = None) -> list:
    """SYSTEM USE ONLY (the reminder thread): due follow-ups of every user. Never expose this to a session."""
    return _due_rows("", (), today or date.today().isoformat())


def mark_reminder_sent(case_id: str) -> None:
    """SYSTEM USE ONLY: remember that the due-date email went out, so it is sent once."""
    conn = _get_connection()
    try:
        row = conn.execute("SELECT next_followup FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row and row["next_followup"]:
            record = json.loads(row["next_followup"])
            record["reminder_sent"] = True
            conn.execute("UPDATE cases SET next_followup = ? WHERE case_id = ?", (json.dumps(record), case_id))
            conn.commit()
    finally:
        conn.close()


def list_active_cases(limit: int = 5, all_time: bool = False) -> list:
    """
    Recent saved cases that have a baseline, newest first, as short summaries
    (for choosing a case in a follow-up session). Only the active user's cases.
    """
    cutoff = "0000-00-00" if all_time else (date.today() - timedelta(days=ACTIVE_CASE_DAYS)).isoformat()
    tiebreak = "created_at" if _database_url() else "rowid"   # newest insert first among same-day cases
    conn = _get_connection()
    try:
        rows = conn.execute(
            f"SELECT * FROM cases WHERE baseline_result != ? AND created_date >= ? AND {OWNER_SQL} "
            f"ORDER BY created_date DESC, {tiebreak} DESC LIMIT ?", (json.dumps({}), cutoff, current_user_id(), limit)
        ).fetchall()
    finally:
        conn.close()
    summaries = []
    for row in rows:
        baseline = json.loads(row["baseline_result"])
        followups = json.loads(row["followups"])
        latest = followups[-1] if followups else baseline
        nxt = json.loads(row["next_followup"]) if "next_followup" in row.keys() and row["next_followup"] else None
        summaries.append({
            "case_id": row["case_id"],
            "created_date": row["created_date"],
            "baseline_severity": (baseline.get("classify") or {}).get("predicted_severity"),
            "latest_severity": (latest.get("classify") or {}).get("predicted_severity"),
            "visits": 1 + len(followups),
            "next_followup": nxt.get("due") if nxt else None,
        })
    return summaries


def get_case(case_id: str) -> dict:
    """Everything stored for a case -- baseline + all follow-ups. None if it is not the active user's."""
    conn = _get_connection()
    row = conn.execute(f"SELECT * FROM cases WHERE case_id = ? AND {OWNER_SQL}",
                       (case_id, current_user_id())).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "case_id": row["case_id"],
        "created_date": row["created_date"],
        "baseline_result": json.loads(row["baseline_result"]),
        "followups": json.loads(row["followups"]),
        "next_followup": (json.loads(row["next_followup"])
                          if "next_followup" in row.keys() and row["next_followup"] else None),
    }