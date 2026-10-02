"""
The agent loop -- this IS the agent. There is no separate reasoning
engine: we call the LLM repeatedly, and its response is either a tool
call, a final text answer, or (via a plain-text convention) a question
back to the user.

Run this file directly to try one full case interactively:
    python -m agent.loop
"""

import os
import re
import sys
import json
import time
import uuid
import traceback
from dotenv import load_dotenv
from groq import (
    Groq, APIConnectionError, APIStatusError, BadRequestError, RateLimitError,
)

from agent.schemas import TOOL_SCHEMAS
from agent.system_prompt import SYSTEM_PROMPT
from agent.dispatch import execute_tool
from agent.case_store import (
    init_db, create_new_case, save_baseline_result, get_case, add_followup_result,
    update_segment_result, list_active_cases,
)
from agent.consent import check_report_request, validate_recipient
from tools.escalate import preview_required
from agent.context import (
    compact_previous_turns, fit_within_turn, trim_old_turns, history_token_budget, build_followup_message,
    build_session_case_message,
)
from agent.medication_filter import find_medication_terms, redact_medication_terms

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"   # Groq-hosted, tool-calling capable (as of Aug 2026)

MAX_REPLY_TOKENS = 1800   # reserved for the reply; Groq counts it against the per-minute limit
MAX_TOOL_CALLS_PER_TURN = 8   # safety cap -- prevents infinite tool-call loops
DEBUG = os.environ.get("BURNSCOPE_DEBUG", "").lower() in ("1", "true", "yes")   # raw model messages / finish reasons

# What the user is told when the AI service (not the burn logic) fails.
MSG_RATE_LIMIT_MINUTE = "The AI service is busy right now. Please wait a minute and try again."
MSG_RATE_LIMIT_DAILY = "The AI service's usage limit has been reached for now. Please try again later."
MSG_TOO_LONG = ("This conversation has grown too long for the AI service's per-request size limit. "
                "Please start a new session; your saved case history is kept.")
MSG_BAD_TOOL_CALL = "I had trouble preparing that action. Please try again, or rephrase your request."
MSG_BAD_REQUEST = "I couldn't process that request because of a problem talking to the AI service. Please try again."
MSG_CONNECTION = "I couldn't reach the AI service. Please check your connection and try again."
MSG_AUTH = ("The AI service rejected our credentials or configuration, so I can't continue. "
            "Please contact the administrator.")
MSG_SERVER = "The AI service is having problems right now. Please try again shortly."
MSG_EMPTY = "I wasn't able to put together a response. Please try again or rephrase your message."
MSG_INTERNAL = "Something went wrong on my side while handling that message. Please try again."
MSG_TOOL_CAP = "[Agent stopped: too many tool calls without a final answer.]"
MAX_RETRIEVALS_PER_TURN = 2   # after this many, nudge the model to answer with what it already has

_REWRITE_SYSTEM = (
    "You edit patient-facing chat messages for a burn-triage assistant. Rewrite the message you are given so that it "
    "contains NO medication names, drug classes (for example anti-inflammatories, antibiotics, opioids), "
    "over-the-counter product wording, specific dressing products or ingredients (for example silver, hydrogel, "
    "paraffin gauze), doses, or medication schedules. Replace them with general wording such as "
    "'appropriate pain relief chosen by a clinician' or 'a suitable non-stick dressing'. Keep EVERYTHING else "
    "unchanged: meaning, structure, formatting, tone, numbers, citations and advice. Output ONLY the rewritten message."
)


class AgentServiceError(Exception):
    """An LLM-service failure the loop could not recover from.
    `user_message` is safe to show to the end user."""

    def __init__(self, user_message: str, kind: str = "service"):
        super().__init__(user_message)
        self.user_message = user_message
        self.kind = kind


def _call_with_retry(history, max_retries=4):
    """
    Calls the Groq API, retrying failures that are worth retrying and
    turning the rest into an AgentServiceError with a user-safe message.

    - 429 per-minute limit: exponential backoff (2s, 4s, 8s, capped at
      10s). The free tier caps at 8000 tokens/minute; a long conversation
      (many tool calls + retrieved passages accumulating in history) can
      briefly exceed this. A 429 on the DAILY cap is not retried: waiting
      seconds cannot help.
    - 400 tool_use_failed (the model produced a tool call that does not
      match the schema): asked again, since sampling usually differs.
    - 5xx and connection errors: backoff and retry.
    - 413 (request larger than the per-request token limit): cannot be
      fixed by waiting -- reported as "conversation too long".
    - 401/403/404 and any other 400: configuration/request problem, no
      retry.
    """
    for attempt in range(1, max_retries + 1):
        last = attempt == max_retries
        wait_seconds = min(2 ** attempt, 10)  # 2s, 4s, 8s, capped at 10s
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=history,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
        except RateLimitError as e:
            if "per day" in str(e).lower():
                raise AgentServiceError(MSG_RATE_LIMIT_DAILY, "rate_limit_daily") from e
            if last:
                raise AgentServiceError(MSG_RATE_LIMIT_MINUTE, "rate_limit") from e
            print(f"  [rate limit] Hit Groq's rate limit, retrying in "
                  f"{wait_seconds}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait_seconds)
        except BadRequestError as e:
            if "tool_use_failed" not in str(e):
                raise AgentServiceError(MSG_BAD_REQUEST, "bad_request") from e
            if last:
                raise AgentServiceError(MSG_BAD_TOOL_CALL, "bad_tool_call") from e
            print(f"  [retry] The model produced an invalid tool call; asking again "
                  f"(attempt {attempt}/{max_retries})...")
        except APIStatusError as e:
            if e.status_code == 413:
                raise AgentServiceError(MSG_TOO_LONG, "too_long") from e
            if e.status_code in (401, 403, 404):
                raise AgentServiceError(MSG_AUTH, "auth") from e
            if e.status_code >= 500 and not last:
                print(f"  [retry] AI service error {e.status_code}, retrying in "
                      f"{wait_seconds}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_seconds)
            else:
                raise AgentServiceError(MSG_SERVER, "service") from e
        except APIConnectionError as e:
            if last:
                raise AgentServiceError(MSG_CONNECTION, "connection") from e
            print(f"  [retry] Could not reach the AI service, retrying in "
                  f"{wait_seconds}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait_seconds)


# Words that show a reply is giving care advice (two distinct ones = advice).
_CARE_TERMS = [
    r"dressing", r"bandage", r"gauze", r"blister", r"rinse", r"wound care", r"first aid", r"home care",
    r"non[\s\-‐-―]*adhesive", r"moist", r"elevat", r"cool(?:ing)?\s+(?:running\s+)?water",
    r"keep the (?:area|wound|burn) clean",
]
_SEVERITY_WORDS = {"1st_degree": "first-degree", "2nd_degree": "second-degree", "3rd_degree": "third-degree"}

# A reply that names or leans toward a triage verdict -- even as a conditional/fallback ("if a clearer
# photo isn't practical, I'll move forward with a clinical-review recommendation") -- without giving
# explicit care instructions. _gives_care_advice() alone misses this: a hedge naming a verdict is not
# phrased as "dressing/rinse/cool water" advice, but it is still reasoning toward a verdict.
_VERDICT_WORDS_RE = re.compile(
    r"\bself[\s\-]?care\b|\bclinical[\s\-]?review\b|\burgent[\s\-]?care\b|\bsee a clinician\b|"
    r"\bclinician (?:promptly|review)\b|\bemergency (?:care|department|room)\b", re.IGNORECASE)


def _states_verdict(text: str) -> bool:
    return bool(_VERDICT_WORDS_RE.search(text))


# A reply that says the report is ready / generated / sent / downloadable.
_REPORT_DONE_RE = re.compile(
    r"(?:\b(?:report|pdf)\b.{0,80}\b(?:ready|generated|created|sent|emailed|attached|saved)\b"
    r"|\b(?:ready|generated|created|sent|emailed)\b.{0,40}\b(?:report|pdf)\b"
    r"|download button)", re.IGNORECASE | re.DOTALL)

REPORT_CORRECTION = (
    "CORRECTION: your draft says the report is ready or sent, but NO report was generated in this turn (the call was "
    "blocked, or it only previewed). Do not say it is ready. If the user has clearly consented to exactly the options "
    "you previewed, call generate_escalation_report again with consent=true and the same options. Otherwise tell the "
    "user in plain words what the report will contain and ask whether they consent."
)


def _claims_report_done(text: str) -> bool:
    return bool(_REPORT_DONE_RE.search(text or ""))


def _gives_care_advice(text: str) -> bool:
    t = text.lower()
    return sum(1 for pattern in _CARE_TERMS if re.search(pattern, t)) >= 2


def _inject_forced_retrieval(history: list, case_state: dict):
    """
    Care advice must come from retrieved guidance, and earlier retrievals are
    compacted away between turns, so a reply that gives advice without a
    retrieval in the SAME turn is sent back for a grounded rewrite: we run
    the retrieval ourselves and put it in the history as a normal tool call.
    """
    user = next((m.get("content") or "" for m in reversed(history) if m.get("role") == "user"), "")
    severity = (case_state.get("classify_result") or {}).get("predicted_severity")
    prefix = f"{_SEVERITY_WORDS[severity]} burn care guidance: " if severity in _SEVERITY_WORDS else "burn care guidance: "
    args = {"query": (prefix + user.strip())[:300]}
    call_id = f"call_forced_{uuid.uuid4().hex[:8]}"
    print(f"  [retrieval guard] care advice without retrieval this turn; retrieving: {args['query']!r}")
    history.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "retrieve_treatment_guidance", "arguments": json.dumps(args)}}],
    })
    result = execute_tool("retrieve_treatment_guidance", args)
    if isinstance(result, dict) and "error" not in result:
        case_state["guidance_result"] = result
        result = dict(result, instruction=(
            "Your draft gave care advice without retrieved guidance. Rewrite your answer using ONLY these "
            "passages, cite the sources, and do not add advice that is not in them."))
    history.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})


def _inject_forced_segment(history: list, case_state: dict):
    """
    A 2nd/3rd-degree reading needs the wound's spatial extent (area_band, multi_region) to set the right
    triage verdict -- see the system prompt's verdict rules -- but the model sometimes reasons about
    severity and gives care advice without having called segment_wound_area first. Same pattern as
    _inject_forced_retrieval: we run the tool ourselves and hand the result back as a normal tool call,
    then ask the model to re-state its verdict with it, rather than trusting the prompt instruction alone.
    """
    image_path = case_state.get("locked_image_path")
    args = {"image_path": image_path}
    call_id = f"call_forced_{uuid.uuid4().hex[:8]}"
    severity = (case_state.get("classify_result") or {}).get("predicted_severity")
    print(f"  [segmentation guard] {severity} reading without segmentation this turn; running segment_wound_area")
    history.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "segment_wound_area", "arguments": json.dumps(args)}}],
    })
    result = execute_tool("segment_wound_area", args)
    if isinstance(result, dict) and "error" not in result:
        case_state["segment_result"] = result
        result = dict(result, instruction=(
            "The wound's spatial extent is now known. Re-state your triage verdict using area_band and "
            "multi_region together with the severity, per the verdict rules, then continue your answer."))
    history.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})


def _rewrite_without_medications(text: str):
    """One cheap call (the draft only, no history, no tools). Returns None on any failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=MAX_REPLY_TOKENS,
            messages=[{"role": "system", "content": _REWRITE_SYSTEM}, {"role": "user", "content": text}],
        )
        return (response.choices[0].message.content or "").strip() or None
    except Exception as e:
        print(f"  [medication guard] rewrite call failed ({type(e).__name__}); falling back to redaction")
        return None


def _enforce_medication_boundary(text: str):
    """
    Chat replies are patient-facing: they must not name medications, drug
    classes, dressing products or doses (see the system prompt). The model
    obeyed this only when asked directly, so it is enforced here.
    Returns (text, guard) with guard in (None, "rewritten", "redacted").
    """
    terms = find_medication_terms(text)
    if not terms:
        return text, None
    print(f"  [medication guard] reply named {terms}; rewriting")
    rewritten = _rewrite_without_medications(text)
    # A rewrite that shrank a lot has probably dropped content: do not trust it.
    if rewritten and len(rewritten) >= 0.6 * len(text) and not find_medication_terms(rewritten):
        return rewritten, "rewritten"
    fallback = rewritten if rewritten and len(rewritten) >= 0.6 * len(text) else text
    return redact_medication_terms(fallback), "redacted"


def _persist_case_results(case_id: str, case_state: dict):
    """
    Write this session's results to the case store, and complete them as
    they arrive.

    The classification is saved as soon as it exists, so a later failure
    cannot lose it. Segmentation normally runs AFTER classification (a
    later step of the loop), so its wound area is added to the same
    saved visit when it arrives -- but only if it belongs to the image
    that was saved (a second photo in the same session must not have its
    area written onto the first photo's record).
    """
    classify_result = case_state.get("classify_result")
    if classify_result is None:
        return
    segment_result = case_state.get("segment_result")
    is_followup = bool(case_state.get("is_followup"))
    saved_flag = "followup_saved" if is_followup else "baseline_saved"

    if not case_state.get(saved_flag):
        if is_followup:
            add_followup_result(case_id, classify_result, segment_result, case_state.get("locked_image_path"))
        else:
            save_baseline_result(case_id, classify_result, segment_result, case_state.get("locked_image_path"))
        case_state[saved_flag] = True
        case_state["persisted_image_path"] = case_state.get("locked_image_path")
        case_state["segment_persisted"] = segment_result is not None
        print(f"  [case_store] {'follow-up result' if is_followup else 'baseline'} saved for {case_id}")
    elif (
        segment_result is not None
        and not case_state.get("segment_persisted")
        and case_state.get("locked_image_path") == case_state.get("persisted_image_path")
    ):
        if update_segment_result(case_id, segment_result, is_followup):
            case_state["segment_persisted"] = True
            print(f"  [case_store] wound area added to the saved "
                  f"{'follow-up' if is_followup else 'baseline'} for {case_id}")


def run_agent_turn(history: list, case_id: str, case_state: dict) -> dict:
    """
    Runs the loop for one user turn: keeps calling the LLM and executing
    tools until it produces a final plain-text answer, or hits the
    safety cap. Never raises for service or tool failures: those come
    back as a readable final_answer, and `history` is always left in a
    state that is valid to send to the API again.

    Args:
        history: list of chat messages so far (OpenAI/Groq message format).
        case_id: the real case ID generated at session start (NEVER
                 invented by the LLM -- see agent/case_store.py).
        case_state: mutable dict tracking this case's tool results
                    across turns. Shape:
                    {"classify_result": dict|None,
                     "segment_result": dict|None,
                     "gradcam_result": dict|None,
                     "guidance_result": dict|None,
                     "clinics_result": dict|None,
                     "locked_image_path": str|None,
                     "baseline_saved": bool,
                     "is_followup": bool,
                     "followup_saved": bool,
                     "segment_persisted": bool,        # optional
                     "persisted_image_path": str|None}  # optional

    Returns:
        dict: {"history": updated history, "final_answer": str,
               "error": None | short error kind (e.g. "too_long"),
               "guard": None | "rewritten" | "redacted"  (medication filter)}
    """
    # Keep the request small: shrink tool payloads from EARLIER turns, then
    # drop the oldest whole turns if still over budget (see agent/context.py).
    # The caller has already appended the user's message.
    try:
        compact_previous_turns(history, len(history))
        dropped = trim_old_turns(history, history_token_budget(), case_state)
        if dropped:
            print(f"  [context] dropped {dropped} old exchange(s) to stay within the request-size budget")
    except Exception:
        traceback.print_exc()   # size management is an optimisation; never let it break the turn
    turn_start = len(history)
    try:
        final_answer, error, guard = _agent_turn_loop(history, case_id, case_state)
    except AgentServiceError as e:
        print(f"  [service error] {e.kind}: {str(e.__cause__)[:300]}")
        # History is consistent here (tool results are appended before the
        # next API call), so keep this turn's progress and record the failure.
        history.append({"role": "assistant", "content": e.user_message})
        return {"history": history, "final_answer": e.user_message, "error": e.kind, "guard": None}
    except Exception:
        traceback.print_exc()
        # An unexpected bug: drop this turn's partial messages so no
        # tool call is left without its result (the API rejects that).
        del history[turn_start:]
        history.append({"role": "assistant", "content": MSG_INTERNAL})
        return {"history": history, "final_answer": MSG_INTERNAL, "error": "internal", "guard": None}
    return {"history": history, "final_answer": final_answer, "error": error, "guard": guard}


def _agent_turn_loop(history: list, case_id: str, case_state: dict):
    """The tool-calling loop for one turn. Returns (final_answer, error_kind, guard)."""
    retrieved_this_turn = False
    forced_retrieval_done = False
    retrieval_calls_this_turn = 0
    segment_forced_done = False
    report_called_this_turn = False
    report_made_this_turn = False
    report_guard_done = False
    for _ in range(MAX_TOOL_CALLS_PER_TURN):
        fit_within_turn(history, history_token_budget())   # tool results may have grown the request mid-turn
        response = _call_with_retry(history)
        message = response.choices[0].message
        if DEBUG:
            print(f"  [debug] finish_reason={response.choices[0].finish_reason}")
            print(f"  [debug] raw message: {message!r}")

        # Case 1: the LLM wants to call one or more tools
        if message.tool_calls:
            # append a CLEAN assistant message -- only the fields the API
            # accepts back. message.model_dump() includes extra SDK-only
            # fields (like "annotations") that Groq rejects if echoed back.
            history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ],
            })

            for call in message.tool_calls:
                tool_name = call.function.name
                try:
                    tool_args = json.loads(call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}
                if not isinstance(tool_args, dict):
                    tool_args = {}
                # The model fills unused optional parameters with null; treat
                # those as "not provided" so tool defaults apply.
                tool_args = {k: v for k, v in tool_args.items() if v is not None}

                # --- Safety net: LOCK the image path the moment
                # check_image_quality is called, and force every
                # subsequent image-consuming tool in this cycle to use
                # that EXACT path -- regardless of what the LLM sends.
                # This eliminates truncation/hallucination risk entirely
                # (observed: the LLM once shortened/altered a real
                # filename mid-call), rather than just reducing it via
                # prompt instructions alone.
                if tool_name == "check_image_quality" and "image_path" in tool_args:
                    case_state["locked_image_path"] = tool_args["image_path"]
                elif tool_name in (
                    "classify_burn_severity", "segment_wound_area", "generate_gradcam"
                ) and case_state.get("locked_image_path"):
                    if tool_args.get("image_path") != case_state["locked_image_path"]:
                        print(f"  [safety net] Overriding image_path "
                              f"'{tool_args.get('image_path')}' -> "
                              f"'{case_state['locked_image_path']}' "
                              f"(locked from check_image_quality)")
                    tool_args["image_path"] = case_state["locked_image_path"]

                # --- Safety net: backfill any fields the LLM omitted
                # when calling generate_escalation_report, using the
                # most recent real results from earlier in this case.
                # This does not rely on the LLM remembering correctly --
                # it guarantees a complete report regardless.
                if tool_name == "generate_escalation_report":
                    classify_result = case_state.get("classify_result")
                    segment_result = case_state.get("segment_result")
                    gradcam_result = case_state.get("gradcam_result")
                    guidance_result = case_state.get("guidance_result")
                    clinics_result = case_state.get("clinics_result")
                    if not tool_args.get("predicted_severity") and classify_result:
                        tool_args["predicted_severity"] = classify_result.get("predicted_severity")
                    if not tool_args.get("calibrated_confidence") and classify_result:
                        tool_args["calibrated_confidence"] = classify_result.get("calibrated_confidence")
                    if not tool_args.get("wound_fraction") and segment_result:
                        tool_args["wound_fraction"] = segment_result.get("wound_fraction")
                    if not tool_args.get("gradcam_overlay_path") and gradcam_result:
                        tool_args["gradcam_overlay_path"] = gradcam_result.get("overlay_path")
                    if not tool_args.get("guidance_passages") and guidance_result:
                        tool_args["guidance_passages"] = guidance_result.get("passages")
                    if not tool_args.get("clinics") and clinics_result:
                        tool_args["clinics"] = clinics_result.get("clinics")
                    # extra facts for the report layout (not model-facing parameters)
                    if classify_result:
                        tool_args["decision"] = classify_result.get("decision")
                    if segment_result:
                        for key in ("area_band", "multi_region", "connected_components"):
                            tool_args[key] = segment_result.get(key)
                    if case_state.get("quality_result"):
                        tool_args["quality"] = case_state["quality_result"]

                print(f"  [tool call] {tool_name}({tool_args})")
                # Consent and recipient are enforced HERE, from the conversation,
                # not from the boolean the model supplies (see agent/consent.py).
                blocked = None
                if tool_name == "generate_escalation_report":
                    # The recipient is checked FIRST and even for a preview: a recipient the model
                    # invented must never reach the user, not even inside a "here's what will be sent"
                    # preview message.
                    blocked = validate_recipient(history, tool_args, own_email=case_state.get("own_email"))
                    if blocked is None and not tool_args.get("preview_only"):
                        # The user must have been shown what the report contains (a preview of these
                        # exact options) before consent is even considered.
                        blocked = preview_required(case_state, tool_args)
                        if blocked is None and tool_args.get("consent"):
                            blocked = check_report_request(history, tool_args, own_email=case_state.get("own_email"))
                    if blocked is not None:
                        print(f"  [report gate] report NOT run: {blocked['status']}")
                result = blocked if blocked is not None else execute_tool(tool_name, tool_args)
                print(f"  [tool result] {result}")

                # Track every SUCCESSFUL result as it comes in, so later
                # tool calls (especially generate_escalation_report) can be
                # backfilled from real, already-gathered case data. A failed
                # tool ({"error": ...}) must never be recorded as a result,
                # or it would be saved to the case as if it were real data.
                tool_ok = not (isinstance(result, dict) and "error" in result)
                if tool_ok and tool_name == "retrieve_treatment_guidance":
                    retrieved_this_turn = True
                if tool_name == "generate_escalation_report":
                    report_called_this_turn = True
                    report_made_this_turn = report_made_this_turn or (isinstance(result, dict) and result.get("status") == "success")
                if tool_ok:
                    if tool_name == "check_image_quality":
                        case_state["quality_result"] = result
                    elif tool_name == "classify_burn_severity":
                        case_state["classify_result"] = result
                    elif tool_name == "segment_wound_area":
                        case_state["segment_result"] = result
                    elif tool_name == "generate_gradcam":
                        case_state["gradcam_result"] = result
                    elif tool_name == "retrieve_treatment_guidance":
                        case_state["guidance_result"] = result
                    elif tool_name == "locate_nearby_clinics":
                        case_state["clinics_result"] = result
                    elif tool_name == "generate_escalation_report" and result.get("status") == "success":
                        case_state["report_result"] = result
                        case_state["report_preview_sig"] = None       # a new report needs a new preview
                    elif tool_name == "generate_escalation_report" and result.get("status") == "preview":
                        case_state["report_preview"] = result
                        case_state["report_preview_sig"] = result.get("signature")

                # append each tool's result back into history
                shown = result
                if tool_name == "generate_escalation_report" and isinstance(result, dict):
                    shown = {k: v for k, v in result.items() if k != "pdf_path"}   # the user downloads it from the app, not by path
                if tool_ok and tool_name == "retrieve_treatment_guidance":
                    # Observed: the model sometimes re-queries with slightly reworded text when a result
                    # doesn't feel complete, burning through the whole per-turn tool-call budget and
                    # falling back to an internal-sounding message. Nudge it to stop after a couple of tries.
                    retrieval_calls_this_turn += 1
                    if retrieval_calls_this_turn >= MAX_RETRIEVALS_PER_TURN and isinstance(shown, dict):
                        # A distinct key from compact_tool_payload's own "note" (used for a different,
                        # unrelated reason -- shrinking OLDER results -- and its wording nudges the
                        # model to retrieve again, the opposite of what this is for).
                        shown = dict(shown, stop_hint=(
                            "You have retrieved guidance more than once this turn. Do not call "
                            "retrieve_treatment_guidance again this turn -- write your final answer now using "
                            "the passages you already have."))
                history.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(shown),
                })

            # Persist case data as soon as we have at least a
            # classification -- segmentation is OPTIONAL for 1st-degree
            # cases and must not block baseline/follow-up persistence --
            # and add the wound area when segmentation arrives later.
            # A storage problem must not crash the conversation.
            try:
                _persist_case_results(case_id, case_state)
            except Exception as e:
                print(f"  [case_store] WARNING: could not save results for {case_id}: "
                      f"{type(e).__name__}: {e}")

            continue  # loop again so the LLM can see the tool results

        # Case 2: the LLM produced a final plain-text answer
        final_text = (message.content or "").strip()
        if not final_text:
            history.append({"role": "assistant", "content": MSG_EMPTY})
            return MSG_EMPTY, "empty_response", None
        if (not report_guard_done and not report_made_this_turn and _claims_report_done(final_text)
                and (report_called_this_turn or case_state.get("report_result") is None)):
            # Never tell the user a report exists when none was made: send the model back once to correct itself.
            report_guard_done = True
            print("  [report guard] draft claimed a report that was not generated; asking the model to correct it")
            history.append({"role": "system", "content": REPORT_CORRECTION})
            continue
        classify_result = case_state.get("classify_result")
        if (not segment_forced_done and case_state.get("segment_result") is None
                and case_state.get("locked_image_path") and classify_result
                and classify_result.get("predicted_severity") in ("2nd_degree", "3rd_degree")
                and (_gives_care_advice(final_text) or _states_verdict(final_text))):
            # A 2nd/3rd-degree reading without segmentation: force it before letting the model
            # finalize a verdict or advice on severity alone (see the system prompt's step 3).
            segment_forced_done = True
            _inject_forced_segment(history, case_state)
            try:
                _persist_case_results(case_id, case_state)
            except Exception as e:
                print(f"  [case_store] WARNING: could not save results for {case_id}: {type(e).__name__}: {e}")
            continue
        if not retrieved_this_turn and not forced_retrieval_done and _gives_care_advice(final_text):
            forced_retrieval_done = True
            _inject_forced_retrieval(history, case_state)
            continue
        final_text, guard = _enforce_medication_boundary(final_text)
        history.append({"role": "assistant", "content": final_text})
        return final_text, None, guard

    # Safety cap hit (usually the model re-querying a tool, e.g. retrieval, instead of answering).
    # Never show the user raw internal debug text: fall back to whatever real facts are already
    # gathered, so the turn still gives them something useful rather than an error-shaped message.
    fallback = _fallback_summary(case_state)
    history.append({"role": "assistant", "content": fallback})
    return fallback, "tool_call_cap", None


def _fallback_summary(case_state: dict) -> str:
    c, s = case_state.get("classify_result"), case_state.get("segment_result")
    if not c or not c.get("predicted_severity"):
        return MSG_TOOL_CAP
    from tools.escalate import AREA_LABELS, SEVERITY_LABELS, derive_verdict
    severity = SEVERITY_LABELS.get(c["predicted_severity"], c["predicted_severity"])
    verdict = derive_verdict(c.get("predicted_severity"), c.get("decision"),
                             (s or {}).get("area_band"), (s or {}).get("multi_region"))
    verdict_text = {"urgent_care": "seek urgent medical care", "clinical_review": "have a clinician review it soon",
                    "self_care": "it is suitable for self-care with monitoring"}[verdict]
    area = f", affecting a {AREA_LABELS.get(s['area_band'], 'unmeasured')} part of the photo" if s and s.get("area_band") else ""
    return (f"I had trouble finishing that reply, but here is what I found so far: this looks like a {severity} "
            f"burn{area}, so I recommend you {verdict_text}. Please ask me again if you would like more detail.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        # Windows consoles (cp1252) cannot print characters the model often
        # uses (e.g. non-breaking hyphens); replace them instead of crashing.
        sys.stdout.reconfigure(errors="replace")

    init_db()

    mode = input("New case or follow-up visit? (new/followup): ").strip().lower()

    if mode == "followup":
        saved_cases = list_active_cases()
        if saved_cases:
            print("Recent saved cases:")
            for i, c in enumerate(saved_cases, 1):
                nxt = f", next check-in {c['next_followup']}" if c["next_followup"] else ""
                print(f"  {i}. {c['case_id']}  created {c['created_date']}  baseline {c['baseline_severity']}"
                      f"  latest {c['latest_severity']}  visits {c['visits']}{nxt}")
        entered = input("Enter the case_id (or its number from the list): ").strip()
        existing_case_id = (saved_cases[int(entered) - 1]["case_id"]
                            if entered.isdigit() and 1 <= int(entered) <= len(saved_cases) else entered)
        existing_case = get_case(existing_case_id)
        if existing_case is None:
            print(f"No case found with id '{existing_case_id}'. Starting a new case instead.\n")
            case_id = create_new_case()
            is_followup = False
        else:
            case_id = existing_case_id
            is_followup = True
    else:
        case_id = create_new_case()
        is_followup = False

    case_state = {
        "classify_result": None,
        "segment_result": None,
        "quality_result": None,
        "gradcam_result": None,
        "guidance_result": None,
        "clinics_result": None,
        "report_result": None,
        "report_preview": None,
        "report_preview_sig": None,
        "locked_image_path": None,
        "baseline_saved": is_followup,  # baseline already exists if this is a follow-up
        "is_followup": is_followup,
        "followup_saved": False,
        "segment_persisted": False,
        "persisted_image_path": None,
    }

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                f"The case_id for this session is: {case_id}. "
                f"Always use this EXACT case_id when calling schedule_followup "
                f"or generate_escalation_report. Never invent your own case_id."
            ),
        },
    ]

    history.append({"role": "system", "content": build_session_case_message(case_id, is_followup)})
    if is_followup:
        history.append({"role": "system", "content": build_followup_message(existing_case)})

    print(f"BurnScope agent -- case_id: {case_id} "
          f"({'follow-up' if is_followup else 'new case'}) -- type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() == "quit":
            break

        history.append({"role": "user", "content": user_input})
        print("  ...sending to Groq, waiting for response...")
        result = run_agent_turn(history, case_id, case_state)
        history = result["history"]
        print(f"\nAgent: {result['final_answer']}\n")
