"""
Keeps each LLM request small.

Every call re-sends the whole conversation (plus ~5.4k tokens of system
prompt and tool schemas), so history growth ends in HTTP 413 on models
with a per-request token cap. Two behaviour-preserving steps, applied at
the start of every user turn:

  1. compact_previous_turns(): tool payloads from EARLIER turns shrink to
     the few facts a later turn can still need (retrieved passages become
     a count and sources; classify/segment JSON keeps severity/confidence/area).
     The current turn's results are never touched.
  2. trim_old_turns(): if the history is still over budget, whole old
     turns are dropped, oldest first, always keeping the most recent
     exchange, and a condensed case-facts note is kept in their place.

Whole turns only, so a tool call is never separated from its result.
"""

import json
import os

# Sized for the 8,000-tokens-per-minute free tier: the request is this history (system prompt included)
# + ~1,500 tokens of tool schemas + the 1,800 reserved for the reply, with ~10% estimation margin.
DEFAULT_HISTORY_TOKEN_BUDGET = 4300   # raise it on larger tiers
NOTE_PREFIX = "CONVERSATION NOTE (condensed):"

_CLASSIFY_KEYS = ("predicted_severity", "calibrated_confidence", "reliability", "decision")
_SEGMENT_KEYS = ("wound_fraction", "area_band", "multi_region", "connected_components")


def history_token_budget() -> int:
    try:
        return int(os.environ.get("BURNSCOPE_HISTORY_TOKEN_BUDGET", DEFAULT_HISTORY_TOKEN_BUDGET))
    except ValueError:
        return DEFAULT_HISTORY_TOKEN_BUDGET


def estimate_tokens(messages: list) -> int:
    """Rough size (about 4 characters per token, matches the API's own counts within ~10%)."""
    return len(json.dumps(messages, ensure_ascii=False)) // 4


def _pick(record, keys):
    return {k: record[k] for k in keys if isinstance(record, dict) and k in record}


def compact_tool_payload(tool_name, payload):
    """Shrink one tool result to what a later turn can still need. Errors and unknown tools pass through."""
    if not isinstance(payload, dict) or "error" in payload:
        return payload
    if tool_name == "retrieve_treatment_guidance" and isinstance(payload.get("passages"), list):
        sources = []
        for p in payload["passages"]:
            src = p.get("source") if isinstance(p, dict) else None
            if src and src not in sources:
                sources.append(src)
        compacted = {"passages_omitted": len(payload["passages"]), "sources": sources[:4],
                    "note": "older retrieval omitted to save space; call retrieve_treatment_guidance again if needed"}
        if "stop_hint" in payload:
            # The per-turn repeated-retrieval nudge (agent/loop.py) must survive compaction: it is the
            # opposite instruction to this compaction's own "note" ("call again if needed"), and losing
            # it would let a long turn re-trigger the exact runaway-retrieval loop it exists to prevent.
            compacted["stop_hint"] = payload["stop_hint"]
        return compacted
    if tool_name == "classify_burn_severity":
        return _pick(payload, _CLASSIFY_KEYS)
    if tool_name == "segment_wound_area":
        return _pick(payload, _SEGMENT_KEYS)
    if tool_name == "check_image_quality":
        return _pick(payload, ("verdict", "issue"))
    if tool_name == "generate_gradcam":
        return _pick(payload, ("overlay_path",))
    if tool_name == "locate_nearby_clinics" and isinstance(payload.get("clinics"), list):
        out = {"clinics": [_pick(c, ("name", "distance_km")) for c in payload["clinics"]]}
        if "note" in payload:
            out["note"] = payload["note"]
        return out
    return payload


def _compact_report_args(arguments: str) -> str:
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments
    if not isinstance(args, dict):
        return arguments
    for key in ("guidance_passages", "clinics"):
        args.pop(key, None)
    for key in ("treatment_summary", "trend_summary"):
        if isinstance(args.get(key), str) and len(args[key]) > 300:
            args[key] = args[key][:297] + "..."
    return json.dumps(args)


def compact_previous_turns(history: list, turn_start: int) -> int:
    """
    Compact tool payloads in every turn BEFORE the current one. `turn_start`
    is len(history) at the start of the current turn (the current user
    message is at turn_start - 1). Returns the number of characters saved.
    """
    limit = min(max(turn_start - 1, 0), len(history))
    names, saved = {}, 0
    for i in range(limit):
        m = history[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for call in m["tool_calls"]:
                names[call["id"]] = call["function"]["name"]
                if call["function"]["name"] == "generate_escalation_report":
                    old = call["function"]["arguments"]
                    new = _compact_report_args(old)
                    if len(new) < len(old):
                        call["function"]["arguments"] = new
                        saved += len(old) - len(new)
        elif m.get("role") == "tool":
            try:
                payload = json.loads(m.get("content") or "")
            except (json.JSONDecodeError, TypeError):
                continue
            new = json.dumps(compact_tool_payload(names.get(m.get("tool_call_id")), payload))
            if len(new) < len(m["content"]):
                saved += len(m["content"]) - len(new)
                m["content"] = new
    return saved


def fit_within_turn(history: list, budget_tokens: int) -> int:
    """
    A single turn can add several large tool results (guidance passages, a list of clinics) after the
    start-of-turn trim. If the request would now exceed the budget, shrink tool results oldest first
    and shorten the newest guidance passages as a last resort instead of dropping them.
    Returns the number of characters saved.
    """
    saved = 0
    if estimate_tokens(history) <= budget_tokens:
        return 0
    names, tool_idx = {}, []
    for i, m in enumerate(history):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for call in m["tool_calls"]:
                names[call["id"]] = call["function"]["name"]
        elif m.get("role") == "tool":
            tool_idx.append(i)
    for i in tool_idx:                                   # oldest first
        if estimate_tokens(history) <= budget_tokens:
            break
        m = history[i]
        try:
            payload = json.loads(m.get("content") or "")
        except (json.JSONDecodeError, TypeError):
            continue
        name = names.get(m.get("tool_call_id"))
        if name == "retrieve_treatment_guidance" and isinstance(payload.get("passages"), list) and i == tool_idx[-1]:
            new_payload = dict(payload, passages=[dict(p, text=p["text"][:600]) for p in payload["passages"]])
        else:
            new_payload = compact_tool_payload(name, payload)
        new = json.dumps(new_payload)
        if len(new) < len(m["content"]):
            saved += len(m["content"]) - len(new)
            m["content"] = new
    return saved


def case_facts(case_state: dict) -> str:
    """One line of the analysis facts gathered so far, for the condensed note."""
    if not case_state:
        return ""
    parts = []
    c, s = case_state.get("classify_result"), case_state.get("segment_result")
    if isinstance(c, dict) and c.get("predicted_severity"):
        details = []
        if isinstance(c.get("calibrated_confidence"), (int, float)):
            details.append(f"calibrated confidence {c['calibrated_confidence']:.2f}")
        if c.get("decision"):
            details.append(f"decision {c['decision']}")
        parts.append(f"photo analysis: {c['predicted_severity']}" + (f" ({', '.join(details)})" if details else ""))
    if isinstance(s, dict) and isinstance(s.get("wound_fraction"), (int, float)):
        parts.append(f"wound area {s['wound_fraction']:.0%} of the image in {s.get('connected_components')} region(s)")
    if case_state.get("locked_image_path"):
        parts.append(f"image: {case_state['locked_image_path']}")
    return "; ".join(parts)


def _set_note(history: list, case_state: dict):
    text = f"{NOTE_PREFIX} earlier exchanges were removed to keep the request small."
    facts = case_facts(case_state)
    if facts:
        text += f" Facts from them: {facts}."
    first_conv = next((i for i, m in enumerate(history) if m.get("role") != "system"), len(history))
    for m in history[:first_conv]:
        if isinstance(m.get("content"), str) and m["content"].startswith(NOTE_PREFIX):
            m["content"] = text
            return
    history.insert(first_conv, {"role": "system", "content": text})


def trim_old_turns(history: list, budget_tokens: int, case_state: dict = None, keep_last_turns: int = 1) -> int:
    """
    Drop whole old turns (oldest first) until the history fits the budget,
    keeping the last `keep_last_turns` previous exchanges and the current
    turn. Returns the number of turns dropped.
    """
    dropped = 0
    while estimate_tokens(history) > budget_tokens:
        first_conv = next((i for i, m in enumerate(history) if m.get("role") != "system"), len(history))
        current = len(history) - 1 if history and history[-1].get("role") == "user" else len(history)
        starts = [i for i in range(first_conv, current) if history[i].get("role") == "user"]
        if len(starts) <= keep_last_turns:
            break
        end = starts[1] if len(starts) > 1 else current
        del history[starts[0]:end]
        dropped += 1
    if dropped:
        _set_note(history, case_state)
    return dropped


def compact_case_record(case: dict) -> dict:
    """The stored case history with only the fields trend analysis needs (for the follow-up system message)."""
    def visit(v):
        return {"classify": _pick(v.get("classify"), _CLASSIFY_KEYS), "segment": _pick(v.get("segment"), _SEGMENT_KEYS)}
    base = case.get("baseline_result") or {}
    return {
        "case_id": case.get("case_id"),
        "created_date": case.get("created_date"),
        "baseline_result": visit(base) if base else {},
        "followups": [{"date": f.get("date"), **visit(f)} for f in case.get("followups") or []],
    }


def build_followup_message(case: dict) -> str:
    return (
        f"This is a FOLLOW-UP visit for an existing case. "
        f"Here is the case's history so far (baseline visit "
        f"and any prior follow-ups): {json.dumps(compact_case_record(case))}. "
        f"When the new image's classify/segment results come in, "
        f"compare them against this baseline (and most recent "
        f"follow-up, if any) and explicitly describe the trend as "
        f"improving, stable, or worsening, per your instructions "
        f"on longitudinal visits."
    )


def build_session_case_message(case_id: str, is_followup: bool) -> str:
    """
    Tells the agent which case this conversation is about, so the system
    prompt's "exactly one active case: use it, do not ask" rule is driven
    by real session data rather than left to the model to infer.
    """
    kind = "an existing case (follow-up visit)" if is_followup else "a NEW case"
    return (f"SESSION CASES: exactly one case is active in this conversation: case_id={case_id}, {kind}. "
            f"Use it automatically; do not ask the user which case this is.")
