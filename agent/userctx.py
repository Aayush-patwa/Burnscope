"""
Who is acting, and where their files live.

Every request made on behalf of a signed-in user runs inside
`with acting_as(user_id):`. The case store, the tools and the report
generator read the current user from here, so one user's cases, photos and
reports are never reachable from another user's session -- even if the model
is asked (or tricked) into passing another user's case ID or file path.

With no user set (the terminal CLI, tests), only cases that have NO owner
are visible, so an unowned session can never see a signed-in user's data.
"""

import contextlib
import contextvars
import os
import re
import threading

_current_user = contextvars.ContextVar("burnscope_user", default=None)

# The classifier, segmenter, Grad-CAM and embedding model are shared by every
# session in the process and are not safe to run from two threads at once.
MODEL_LOCK = threading.RLock()

UPLOADS_ROOT = os.path.join("data", "users")
OUTPUTS_ROOT = os.path.join("outputs", "users")
_KIND_ROOTS = {"uploads": UPLOADS_ROOT, "gradcam": OUTPUTS_ROOT, "reports": OUTPUTS_ROOT}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_]{1,40}$")


@contextlib.contextmanager
def acting_as(user_id):
    token = _current_user.set(user_id)
    try:
        yield
    finally:
        _current_user.reset(token)


def current_user_id():
    return _current_user.get()


def user_dir(kind: str, user_id: str = None, create: bool = True) -> str:
    """Folder for one user's uploads / gradcam images / reports."""
    user_id = user_id or current_user_id()
    if kind not in _KIND_ROOTS:
        raise ValueError(f"unknown folder kind {kind!r}")
    if not user_id or not _SAFE_ID.match(user_id):
        raise ValueError("no valid user is active")
    path = os.path.join(_KIND_ROOTS[kind], user_id, kind)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def path_is_users(path: str, kind: str) -> bool:
    """True if `path` really lies inside the active user's `kind` folder (symlinks and '..' resolved)."""
    try:
        base = os.path.realpath(user_dir(kind, create=False))
        target = os.path.realpath(path)
        return os.path.commonpath([base, target]) == base
    except (ValueError, OSError):
        return False
