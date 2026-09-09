"""
Emergency Detection - a fast, deterministic pre-check that runs BEFORE
the RAG pipeline. If the user's question matches patterns associated with
medical emergencies, we skip retrieval and the LLM entirely and return a
fixed safety response immediately.

=== EDIT LOG ===
- [Reorg / item #8] Moved as-is from the old top-level emergency_check.py
  into guardrails/emergency.py, to sit next to vagueness.py and
  confidence.py as "Guardrail #1" (see rag_pipeline.py's answer()).
  No logic changed here.
- [SAFETY FIX] Added Arabic text normalization before matching. Plain
  substring matching was silently missing real emergencies whenever the
  user's spelling didn't byte-for-byte match the pattern list - most
  commonly hamza variants: everyday colloquial Arabic typing very often
  drops the hamza (e.g. "الم" instead of "ألم", "ازمة" instead of "أزمة"),
  which are DIFFERENT unicode codepoints (U+0627 'ا' vs U+0623 'أ') and
  therefore never matched at all. Diacritics (tashkeel) are stripped for
  the same reason.
- [SAFETY FIX] Keyword rules for "صدر"/"حلق" etc. now match the bare root
  instead of the definite-article form ("الصدر"), so possessive/indefinite
  forms ("صدري", "بصدري") are caught too.
- [REVERTED - embedding semantic layer removed] A second layer using
  cosine similarity against multilingual-e5-base reference sentences was
  added, then removed: with only 5-10 reference sentences per category
  and an uncalibrated threshold (guessed, then lowered from 0.88 to 0.82
  off a single example), e5's similarity on short same-domain Arabic
  medical sentences turned out to be compressed enough that it started
  firing on essentially anything medical-sounding, including plain
  informational questions ("ايه اعراض الانيميا؟") and mild/common
  symptoms ("عندي برد وسخونية وكحة"). A single untested threshold change
  is not something to ship on a safety guardrail - this whole approach is
  replaced below by an LLM classification fallback, which reasons about
  the query directly instead of relying on an unmeasured similarity cutoff.
- [NEW] Layer 2 is now check_emergency_llm(): a single call to the SAME
  call_llm used elsewhere in the pipeline (Groq -> Gemini chain, with its
  existing timeouts/retries), only invoked when the keyword layer (Layer 1)
  finds nothing. It fails OPEN: any exception, malformed response, or all
  providers being unavailable is treated as "not an emergency" and control
  returns to the normal pipeline - this guardrail must never be the reason
  a request hangs or crashes, and blocking everyone when the LLM happens to
  be down would be worse than occasionally missing a paraphrase the keyword
  layer would also have missed.
"""

import json
import os
import re

EMERGENCY_NUMBER = "123"  # Egypt ambulance - change if deploying elsewhere

# [SAFETY FIX] Normalize Arabic hamza/alef variants and strip diacritics so
# matching is robust to how a real user actually types, not just to the
# exact spelling used in EMERGENCY_PATTERNS below. Applied to BOTH the
# user's query and every keyword at match time, so it's safe regardless of
# which variant (with or without hamza) shows up in either side.
_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # tashkeel + tatweel
_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})


def _normalize_arabic(text: str) -> str:
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = text.translate(_ALEF_VARIANTS)
    return text


# Each category is a list of RULES. A rule is a list of substrings that
# must ALL appear somewhere in the query (any order, any words between
# them). The category triggers if ANY rule matches.
EMERGENCY_PATTERNS = {
    "cardiac": [
        ["chest", "pain"], ["chest", "tight"], ["chest", "crushing"],["dying"],
        # [SAFETY FIX] Was "الصدر" (with the definite article) - matched only
        # "in THE chest", never "صدري" (my chest) or any other possessive/
        # indefinite form. "صدر" (the bare root) is a substring of ALL of
        # those forms ("صدري", "الصدر", "بصدري", "لصدري"...), so it catches
        # the way people actually phrase this, deterministically.
        ["heart attack"], ["صدر", "ألم"], ["صدر", "ضيق"], ["صدر", "ضغط"],
        # [SAFETY FIX] Colloquial Arabic very commonly describes chest pain
        # as pain "in the heart" ("ألم في القلب") rather than "in the chest".
        ["قلب", "ألم"], ["قلب", "ضيق"], ["قلب", "ضغط"],
        ["جلطة قلبية"], ["أزمة قلبية"], ["نوبة قلبية"],["بيحتضر"],["يحتضر"],
    ],
    "stroke": [
        ["face", "droop"], ["speech", "slur"], ["numbness", "sudden"],["dying"],
        ["weakness", "one side"], ["تنميل", "مفاجئ"], ["تدلي", "وجه"],
        ["كلام", "مفهوم"], ["ضعف مفاجئ"], ["جلطة دماغية"], ["سكتة دماغية"],["بيحتضر"],["يحتضر"],
    ],
    "breathing": [
        ["can't breathe"], ["cant breathe"], ["not breathing"], ["choking"],
        ["breathing", "severe"], ["مش قادر", "اتنفس"], ["لا أستطيع التنفس"],
        ["اختناق"], ["توقف التنفس"],["بيحتضر"],["يحتضر"],
    ],
    "suicide_self_harm": [
        ["kill myself"], ["want to die"], ["end my life"], ["suicide"], ["suicidal"],
        ["انتحار"], ["عايز اموت"], ["اقتل نفسي"], ["افكار انتحارية"], ["مش عايز اعيش"],
    ],
    "anaphylaxis": [
        ["anaphylaxis"], ["allergic reaction", "severe"], ["throat", "closing"],
        # [SAFETY FIX] Same "ال" issue as the chest keyword above.
        ["throat", "swelling"], ["حساسية", "شديدة"], ["تورم", "حلق"], ["صدمة تحسسية"],
    ],
    "unconscious_bleeding": [
        ["unconscious"], ["unresponsive"], ["won't wake up"],
        ["bleeding", "heavy"], ["bleeding", "heavily"],
        ["فقدان الوعي"], ["مايستجيبش"], ["نزيف", "شديد"], ["نزيف", "حاد"],
    ],
}

EMERGENCY_MESSAGES_AR = {
    "cardiac": (
        "⚠️ الأعراض اللي وصفتها ممكن تكون علامة على حالة قلبية طارئة. "
        f"اتصل بالإسعاف فوراً ({EMERGENCY_NUMBER}) أو روح لأقرب طوارئ - متستناش. "
        "المساعد ده مش بديل عن رعاية طبية طارئة."
    ),
    "stroke": (
        "⚠️ الأعراض اللي وصفتها ممكن تكون علامة على سكتة دماغية. الوقت حرج جداً. "
        f"اتصل بالإسعاف فوراً ({EMERGENCY_NUMBER}) ولا تنتظر لحظة."
    ),
    "breathing": (
        "⚠️ صعوبة شديدة في التنفس أو الاختناق حالة طارئة. "
        f"اتصل بالإسعاف فوراً ({EMERGENCY_NUMBER})."
    ),
    "suicide_self_harm": (
        "أنا قلقان عليك من اللي بتقوله. لو بتفكر تأذي نفسك، محتاج تتكلم مع حد فوراً - "
        f"اتصل بالإسعاف ({EMERGENCY_NUMBER}) أو روح لأقرب طوارئ. مش لازم تكون لوحدك دلوقتي."
    ),
    "anaphylaxis": (
        "⚠️ الأعراض دي ممكن تكون صدمة تحسسية شديدة. "
        f"اتصل بالإسعاف فوراً ({EMERGENCY_NUMBER})، واستخدم قلم الإبينفرين لو متوفر."
    ),
    "unconscious_bleeding": (
        "⚠️ ده وصف لحالة طارئة (فقدان وعي أو نزيف شديد). "
        f"اتصل بالإسعاف فوراً ({EMERGENCY_NUMBER})."
    ),
}

# [FIX] These were missing entirely before - every emergency response was
# hardcoded Arabic only, regardless of the language the user asked in. This
# is the one guardrail in the whole pipeline that bypasses the LLM (by
# design - it must never depend on it), so unlike the LLM-generated answer
# elsewhere it can't "naturally" answer in the question's language on its
# own; it needs an explicit English variant, matching how confidence.py and
# vagueness.py each already carry both languages.
EMERGENCY_MESSAGES_EN = {
    "cardiac": (
        "⚠️ The symptoms you're describing could be a sign of a cardiac emergency. "
        f"Call emergency services right away ({EMERGENCY_NUMBER}) or go to the nearest ER - don't wait. "
        "This assistant is not a substitute for emergency medical care."
    ),
    "stroke": (
        "⚠️ The symptoms you're describing could be a sign of a stroke. Time is critical. "
        f"Call emergency services right away ({EMERGENCY_NUMBER}) and don't wait even a moment."
    ),
    "breathing": (
        "⚠️ Severe difficulty breathing or choking is a medical emergency. "
        f"Call emergency services right away ({EMERGENCY_NUMBER})."
    ),
    "suicide_self_harm": (
        "I'm concerned about what you're describing. If you're thinking about harming yourself, "
        f"please talk to someone right now - call emergency services ({EMERGENCY_NUMBER}) or go to "
        "the nearest ER. You don't have to be alone with this right now."
    ),
    "anaphylaxis": (
        "⚠️ This could be a severe allergic reaction (anaphylaxis). "
        f"Call emergency services right away ({EMERGENCY_NUMBER}), and use an epinephrine "
        "auto-injector if one is available."
    ),
    "unconscious_bleeding": (
        "⚠️ This describes a medical emergency (loss of consciousness or severe bleeding). "
        f"Call emergency services right away ({EMERGENCY_NUMBER})."
    ),
}

# [FIX] When the user is reporting that someone ELSE said they want to die
# (e.g. "حد قال لي انه عايز يموت"), the standard message above ("أنا قلقان
# عليك"، "متستناش لوحدك") wrongly addresses the reporting user as if THEY
# were the one at risk, which is confusing and undermines trust in the
# guardrail. This variant is used instead when the LLM classifier
# identifies the subject as "other" - it stays supportive, doesn't ask the
# user to decode/diagnose, and gives concrete steps for both the at-risk
# person and the person reporting it.
EMERGENCY_MESSAGE_SUICIDE_OTHER_AR = (
    "قلقان لما بسمع إن حد قريب منك بيتكلم كده. الأهم دلوقتي إنه ميفضلش لوحده. "
    f"لو حاسس إن في خطر فوري عليه دلوقتي، اتصل بالإسعاف ({EMERGENCY_NUMBER}) أو روح بيه لأقرب طوارئ. "
    "حاول تفضلوا معاه حد يقدر يطمّن عليه، وشجّعه يتكلم مع مختص أو خط مساعدة نفسية في أقرب وقت. "
    "ولو انت كمان حاسس إن الموقف ده تقيل عليك، متترددش تتكلم مع حد أو تطلب مساعدة لنفسك كمان."
)
EMERGENCY_MESSAGE_SUICIDE_OTHER_EN = (
    "I'm concerned to hear that someone close to you is saying this. The most important thing "
    "right now is that they aren't left alone. "
    f"If you feel there's immediate danger, call emergency services ({EMERGENCY_NUMBER}) or take "
    "them to the nearest ER. Try to stay with them or have someone else stay with them, and "
    "encourage them to talk to a mental health professional or a crisis line as soon as possible. "
    "And if this situation feels heavy for you too, don't hesitate to talk to someone or get "
    "support for yourself as well."
)


def check_emergency(query: str):
    """Layer 1: exact/normalized keyword matching. Zero-cost, instant,
    never depends on any external service - this is the guardrail's
    guaranteed floor even if the LLM layer below is completely down."""
    normalized = _normalize_arabic(query.lower())
    for category, rules in EMERGENCY_PATTERNS.items():
        for rule in rules:
            if all(_normalize_arabic(keyword.lower()) in normalized for keyword in rule):
                return category
    return None


def get_emergency_response(category: str, subject: str = "self", query: str = "") -> str:
    # Same detection approach used in confidence.py / vagueness.py: presence
    # of Arabic script decides the language, since this guardrail bypasses
    # the LLM and so can't infer the question's language the way the normal
    # generated answer does.
    is_ar = bool(re.search(r"[\u0600-\u06FF]", query))
    if category == "suicide_self_harm" and subject == "other":
        return EMERGENCY_MESSAGE_SUICIDE_OTHER_AR if is_ar else EMERGENCY_MESSAGE_SUICIDE_OTHER_EN
    messages = EMERGENCY_MESSAGES_AR if is_ar else EMERGENCY_MESSAGES_EN
    return messages.get(category, messages["cardiac"])


# ============================================================================
# Layer 2: LLM classification fallback (replaces the old embedding-based
# semantic layer - see EDIT LOG above for why). Only runs when Layer 1
# (check_emergency) finds nothing.
# ============================================================================

_VALID_CATEGORIES = {
    "cardiac", "stroke", "breathing",
    "suicide_self_harm", "anaphylaxis", "unconscious_bleeding", "none",
}

EMERGENCY_LLM_PROMPT = """You are a medical emergency triage classifier. \
Decide whether the user's message describes symptoms consistent with ONE of \
these acute emergency categories, or none of them.

- cardiac: chest pain/pressure/tightness, symptoms suggesting a heart attack
- stroke: sudden face drooping, sudden slurred speech, sudden numbness or \
weakness on one side of the body
- breathing: severe difficulty breathing, choking, not breathing
- suicide_self_harm: suicidal ideation, wanting to end one's life, intent to \
self-harm
- anaphylaxis: severe allergic reaction, throat closing or swelling
- unconscious_bleeding: unconsciousness, unresponsiveness, heavy/severe \
bleeding that won't stop
- none: anything else - general questions, mild or chronic symptoms, \
informational questions, symptoms that are uncomfortable but not acutely \
life-threatening (e.g. a general question about anemia, a common cold, \
fatigue, a headache)

Only pick a category other than "none" if the message describes an ACUTE, \
currently-happening emergency - not a general question about a condition, \
not a mild/chronic symptom, and not just because the message is medical in \
topic.

Also decide who the emergency is about:
- "self": the person writing the message is describing themselves having \
these symptoms or being at risk
- "other": the person writing the message is describing someone else (a \
family member, friend, partner, etc.) having these symptoms or being at risk
If this is unclear, use "self" - it is the safer default.

Message: "{query}"

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"category": "cardiac" | "stroke" | "breathing" | "suicide_self_harm" | "anaphylaxis" | "unconscious_bleeding" | "none", "subject": "self" | "other"}}"""


def check_emergency_llm(query: str, call_llm_fn):
    """Layer 2: LLM-based classification, called only when check_emergency()
    (keywords) found nothing. `call_llm_fn` is any `str -> str | None`
    callable - pass RAGPipeline.call_llm so this reuses its existing
    provider chain (Groq -> Gemini), timeouts, retries, and reasoning-strip
    logic instead of maintaining a separate model/client.

    Returns a (category, subject) tuple. category is None for BOTH
    "genuinely not an emergency" AND "the LLM call/parse failed" - both
    cases must behave identically (fail open) so a transient LLM outage
    can never turn into every user being blocked, and can never turn into
    a crash. subject is "self" or "other", defaulting to "self" (the
    safer assumption) whenever it can't be determined.
    """
    try:
        raw = call_llm_fn(EMERGENCY_LLM_PROMPT.format(query=query))
        if not raw:
            return None, "self"  # all providers failed/timed out -> fail open
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        verdict = json.loads(cleaned)
        category = verdict.get("category")
        subject = verdict.get("subject") if verdict.get("subject") in ("self", "other") else "self"
        if category in _VALID_CATEGORIES and category != "none":
            return category, subject
        return None, "self"
    except Exception:
        # Malformed JSON, unexpected shape, network error, etc. -> fail
        # open. Never let a parsing bug in this guardrail block a request
        # or crash the pipeline.
        return None, "self"
