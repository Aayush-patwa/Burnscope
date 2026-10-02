"""
Maps tool names (as the LLM will name them) to the actual Python
functions that implement them. This is the ONLY file that needs to
change when a tool is added or renamed (also add its schema in
agent/schemas.py).

execute_tool() never raises: any failure inside a tool comes back to the
agent as an {"error": ...} dict, so the agent can tell the user the step
failed instead of the whole conversation crashing.
"""

import os

from agent.userctx import MODEL_LOCK, current_user_id, path_is_users
from tools.quality_check import check_image_quality
from tools.classify import classify_burn_severity
from tools.segment import segment_wound_area
from tools.gradcam import generate_gradcam
from tools.rag_retrieve import retrieve_treatment_guidance
from tools.scheduler import schedule_followup
from tools.clinic_locator import locate_nearby_clinics
from tools.escalate import generate_escalation_report

TOOL_DISPATCH = {
    "check_image_quality": check_image_quality,
    "classify_burn_severity": classify_burn_severity,
    "segment_wound_area": segment_wound_area,
    "generate_gradcam": generate_gradcam,
    "retrieve_treatment_guidance": retrieve_treatment_guidance,
    "schedule_followup": schedule_followup,
    "locate_nearby_clinics": locate_nearby_clinics,
    "generate_escalation_report": generate_escalation_report,
}

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _short_error(exc: Exception, limit: int = 200) -> str:
    """One-line, length-capped error text with the project's absolute path removed."""
    text = " ".join(str(exc).split()).replace(_PROJECT_ROOT, "<project>")
    return text if len(text) <= limit else text[: limit - 3] + "..."


# Tools that use the shared models run one at a time across all sessions.
_MODEL_TOOLS = {"classify_burn_severity", "segment_wound_area", "generate_gradcam", "retrieve_treatment_guidance"}
# Which of the active user's folders each file argument must lie in.
_PATH_ARGS = {"image_path": "uploads", "gradcam_overlay_path": "gradcam"}


def _foreign_path(args: dict):
    """Name of a file argument that points outside the active user's folders, else None."""
    if current_user_id() is None:
        return None
    for arg, kind in _PATH_ARGS.items():
        if args.get(arg) and not path_is_users(str(args[arg]), kind):
            return arg
    return None


def execute_tool(name: str, args: dict) -> dict:
    """Look up and run the named tool with the given arguments."""
    if name not in TOOL_DISPATCH:
        return {"error": f"Unknown tool '{name}'"}
    bad = _foreign_path(args)
    if bad:
        return {"error": f"{bad} is not one of this user's files; the tool was not run.", "tool": name}
    try:
        if name in _MODEL_TOOLS:
            with MODEL_LOCK:
                return TOOL_DISPATCH[name](**args)
        return TOOL_DISPATCH[name](**args)
    except TypeError as e:
        return {"error": f"Bad arguments for '{name}': {e}"}
    except Exception as e:
        print(f"  [tool error] {name} raised {type(e).__name__}: {e}")
        return {"error": f"{name} failed: {type(e).__name__}: {_short_error(e)}", "tool": name}
