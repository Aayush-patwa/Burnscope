"""
Tool schemas exposed to the LLM, in OpenAI/Groq/Gemini-style
function-calling format. This list is sent alongside the conversation
on every single agent loop call -- the LLM does not remember tools
between calls, it must be reminded every time.

Keep descriptions precise: this is literally how the LLM decides
WHEN to call WHICH tool. Vague wording causes wrong or missed calls.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_image_quality",
            "description": (
                "Check whether an uploaded burn photo is usable -- blur, "
                "brightness, and whether the wound is visible. ALWAYS "
                "call this first, before any other tool, on every new image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path or ID of the uploaded image.",
                    }
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_burn_severity",
            "description": (
                "Predict burn degree (1st/2nd/3rd) with a confidence "
                "score, plus a built-in reliability gate (calibrated "
                "confidence, entropy, and a PROCEED/ABSTAIN decision). "
                "Call this immediately after check_image_quality passes "
                "with verdict 'acceptable'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path or ID of the quality-checked image.",
                    }
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "segment_wound_area",
            "description": (
                "Segment the wound region and return its RELATIVE "
                "proportion of the image (wound_fraction) and number of "
                "separate wound regions (connected_components). Does "
                "NOT return a real-world cm^2 measurement -- there is no "
                "scale-reference calibration in this system. Call when "
                "decision from classify_burn_severity was 'PROCEED' and "
                "predicted_class is 2nd/3rd-degree. Optional for 1st-degree."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path or ID of the quality-checked image.",
                    }
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_gradcam",
            "description": (
                "Generate a heatmap overlay showing which image regions "
                "drove the classifier's prediction. Call mainly when "
                "compiling a clinician-facing report, not for every case."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "predicted_class": {
                        "type": "string",
                        "description": "Class returned by classify_burn_severity.",
                    },
                },
                "required": ["image_path", "predicted_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_treatment_guidance",
            "description": (
                "Retrieve cited treatment guidance from the curated "
                "medical document corpus. NEVER answer treatment "
                "questions from general knowledge -- always call this "
                "tool instead. Shape the query differently depending on "
                "audience: a self-care checklist query for self-care "
                "cases, a clinician-oriented query for clinical-review "
                "or urgent-care cases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language treatment question.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many passages to retrieve. Default 3.",
                    },
                    "for_child": {
                        "type": "boolean",
                        "description": "true only when the patient is a child or infant (then child-specific passages are included).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_followup",
            "description": (
                "Schedule the next check-in for this case. Use a short "
                "interval (light-touch, e.g. 5-7 days) for self-care "
                "cases, and a tighter, more structured interval "
                "(e.g. 2-3 days) for clinical-review or urgent-care cases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "interval_days": {"type": "integer"},
                },
                "required": ["case_id", "interval_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "locate_nearby_clinics",
            "description": (
                "Find nearby clinics/urgent-care/hospital burn units. "
                "ONLY call this when the triage verdict is "
                "'clinical_review' or 'urgent_care' -- never for "
                "self-care cases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "User's approximate location.",
                    },
                    "radius_km": {"type": "number"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_escalation_report",
            "description": (
                "Make the case report (PDF), optionally emailed to a doctor. ALWAYS call "
                "it first with preview_only=true and tell the user what it contains. The real "
                "call is REFUSED unless that exact report was previewed and the user then "
                "explicitly consented. Case results are attached automatically. Offer the "
                "report proactively for 'urgent_care' or a worsening trend (ask consent "
                "first); for 'clinical_review' only when asked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "consent": {
                        "type": "boolean",
                        "description": "True only if the user explicitly consented in their latest reply.",
                    },
                    "recipient_email": {
                        "type": "string",
                        "description": "Doctor's email exactly as the user typed it. Optional; if omitted only a local PDF is made.",
                    },
                    "baseline_predicted_severity": {
                        "type": "string",
                        "description": "Follow-up cases only: the baseline visit's predicted_severity from the case history.",
                    },
                    "baseline_wound_fraction": {
                        "type": "number",
                        "description": "Follow-up cases only: the baseline visit's wound_fraction.",
                    },
                    "trend_summary": {
                        "type": "string",
                        "description": "Follow-up cases: YOUR OWN plain-language comparison of baseline vs current findings. Required for a follow-up report.",
                    },
                    "preview_only": {
                        "type": "boolean",
                        "description": "true = only list what the report would contain; nothing is generated or sent.",
                    },
                    "exclude_sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sections the user wants left out, e.g. [\"clinics\"] for nearby care, \"gradcam\" for the model-attention image, \"trend\", \"guidance\", \"followup\", \"patient_notes\", \"clinician_reference\".",
                    },
                    "patient_notes": {
                        "type": "string",
                        "description": "Text the user asked to add, in THEIR words (symptoms, allergies, how it happened). Not medical advice.",
                    },
                    "treatment_summary": {
                        "type": "string",
                        "description": "YOUR OWN readable treatment summary based on the retrieved guidance, not raw quotes.",
                    },
                    "triage_verdict": {
                        "type": "string",
                        "enum": ["self_care", "clinical_review", "urgent_care"],
                        "description": "Your triage verdict for this case, so the report matches what you told the user.",
                    },
                    "predicted_severity": {"type": "string", "description": "Optional; filled automatically."},
                    "calibrated_confidence": {"type": "number", "description": "Optional; filled automatically."},
                    "wound_fraction": {"type": "number", "description": "Optional; filled automatically."},
                    "gradcam_overlay_path": {"type": "string", "description": "Optional; filled automatically."},
                    "guidance_passages": {
                        "type": "array",
                        "description": "Optional; filled automatically.",
                        "items": {"type": "object"},
                    },
                    "clinics": {
                        "type": "array",
                        "description": "Optional; filled automatically.",
                        "items": {"type": "object"},
                    },
                },
                "required": ["case_id", "consent"],
            },
        },
    },
]


def _make_optional_params_nullable(schemas):
    """
    The model often fills unused optional parameters with null. The API
    validates tool calls against these schemas server-side and rejects a
    null for a non-nullable type with HTTP 400 (tool_use_failed), which
    used to abort the turn. Optional parameters are therefore declared
    nullable; required ones are left strict.
    """
    for tool in schemas:
        params = tool["function"]["parameters"]
        required = set(params.get("required", []))
        for name, prop in params.get("properties", {}).items():
            if name not in required and isinstance(prop.get("type"), str):
                prop["type"] = [prop["type"], "null"]


_make_optional_params_nullable(TOOL_SCHEMAS)
