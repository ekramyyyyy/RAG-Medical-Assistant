"""
[NEW FILE - "Guardrail #2" from the plan]

Catches questions that are too thin to search or answer usefully -
e.g. "عندي كحة" / "I have a cough" with nothing else - and asks the
user for the missing details INSTEAD of running retrieval + Gemini on
a single word.

Design (per items #5-#9 of the plan):
- This is NOT a blacklist of symptom words. A single symptom word is
  fine; it only becomes "vague" when it also lacks any of: duration,
  severity, or an associated symptom/detail, AND the query is short.
- Layer 1 (always on, deterministic, free): `is_vague()`.
- Layer 2 (optional, only called by rag_pipeline.py when Layer 1 is
  genuinely ambiguous - not for every request): `llm_vagueness_check()`,
  which asks the LLM for a structured JSON verdict. This keeps the
  common case free/instant and only spends an LLM call on edge cases.
- Returns a structured dict (`{"allowed": ..., "reason": ..., "missing": ...,
  "message": ...}`) as recommended in item #9, instead of a bare bool.
"""

import json
import re

# A handful of single-symptom mentions that are common but say almost
# nothing on their own (both AR and EN forms).
BARE_SYMPTOM_TERMS = [
    "كحة", "cough", "صداع", "headache", "دوخة", "dizziness", "dizzy",
    "تعبان", "tired", "fatigue", "حرارة", "fever", "وجعان", "بطني", "stomach ache",
    "ألم", "pain", "غثيان", "nausea", "قيء", "vomiting", "طفح", "rash",
    "زكام", "cold", "إمساك", "constipation", "إسهال", "diarrhea",
]

# Presence of ANY of these means the user already gave useful context,
# even if the message is short.
DETAIL_INDICATORS = [
    # duration
    "من", "منذ", "يوم", "أيام", "أسبوع", "أسابيع", "شهر", "ساعات", "ساعة",
    "since", "for", "day", "days", "week", "weeks", "hour", "hours", "month",
    # severity
    "شديد", "شديدة", "خفيف", "خفيفة", "بسيط", "severe", "mild", "intense",
    # location / qualifiers that add specificity
    "يمين", "شمال", "left", "right", "مستمر", "متقطع", "constant", "intermittent",
]

MIN_WORDS_CONSIDERED_DETAILED = 8  # a long-enough message is assumed detailed regardless

CLARIFICATION_MESSAGE_AR = (
    "المعلومة دي لوحدها مش كفاية عشان أقدر أفيدك بشكل دقيق. "
    "ممكن تقولي: من إمتى بدأ العرض، هل هو شديد ولا خفيف، وهل معاه أعراض تانية "
    "(زي حرارة، ضيق تنفس، ألم، دم)؟"
)
CLARIFICATION_MESSAGE_EN = (
    "That's not quite enough for me to help accurately. Could you tell me: "
    "when it started, how severe it is, and whether there are any other symptoms "
    "with it (e.g. fever, shortness of breath, pain, bleeding)?"
)


def _is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def is_vague(query: str) -> dict:
    """Layer 1: deterministic rule-based check. Cheap, always runs."""
    normalized = query.strip().lower()
    words = normalized.split()

    if len(words) >= MIN_WORDS_CONSIDERED_DETAILED:
        return {"allowed": True, "reason": None, "missing": [], "message": None}

    mentions_bare_symptom = any(term in normalized for term in BARE_SYMPTOM_TERMS)
    has_detail = any(term in normalized for term in DETAIL_INDICATORS)

    if mentions_bare_symptom and not has_detail and len(words) <= 6:
        message = CLARIFICATION_MESSAGE_AR if _is_arabic(query) else CLARIFICATION_MESSAGE_EN
        return {
            "allowed": False,
            "reason": "vague",
            "missing": ["duration", "severity", "associated_symptoms"],
            "message": message,
        }

    return {"allowed": True, "reason": None, "missing": [], "message": None}


LLM_VAGUENESS_PROMPT = """You are a triage assistant deciding whether a patient's message \
gives enough information to search a medical knowledge base usefully.

Message: "{query}"

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"is_vague": true or false, "missing": ["duration", "severity", "associated_symptoms", "location"], "reason": "short reason"}}
Only list fields in "missing" that are actually absent from the message."""


def llm_vagueness_check(query: str, call_llm_fn) -> dict:
    """Layer 2 (optional): ask the LLM for a structured verdict on
    borderline cases. `call_llm_fn` is any `str -> str` callable
    (e.g. RAGPipeline.call_llm), injected so this module has no
    dependency on which LLM SDK is used.

    Only call this for cases Layer 1 leaves ambiguous - don't run it on
    every single message, it costs a real LLM call.
    """
    try:
        raw = call_llm_fn(LLM_VAGUENESS_PROMPT.format(query=query))
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        verdict = json.loads(cleaned)
    except Exception:
        # If the LLM/JSON parsing fails, fail open (treat as not vague) -
        # a guardrail that silently blocks everyone on error is worse
        # than one that occasionally lets a vague query through.
        return {"allowed": True, "reason": None, "missing": [], "message": None}

    if verdict.get("is_vague"):
        message = CLARIFICATION_MESSAGE_AR if _is_arabic(query) else CLARIFICATION_MESSAGE_EN
        return {
            "allowed": False,
            "reason": "vague",
            "missing": verdict.get("missing", []),
            "message": message,
        }
    return {"allowed": True, "reason": None, "missing": [], "message": None}
