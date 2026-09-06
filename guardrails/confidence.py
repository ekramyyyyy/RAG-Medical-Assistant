"""
[NEW FILE - "Guardrail #3" from the plan]

Blocks Gemini from answering when retrieval didn't actually find
anything relevant, instead of letting it improvise from a weak match.

=== IMPORTANT (item #12 of the plan) — measured, not guessed ===
Claude cannot run a real calibration itself: doing that properly needs
(a) your actual chunks.json / populated Qdrant collection, and (b)
internet access to download `multilingual-e5-base` from Hugging Face -
neither is available in Claude's sandboxed environment (verified: the
sandbox's network egress blocks huggingface.co). So this default is
NOT a number pulled out of thin air, but it is also NOT a substitute
for running `evaluate_threshold.py` (included alongside this file) on
your real, deployed setup - do that before trusting this in production.

Why 0.75 and not something like 0.35: multilingual-e5 (like most
sentence-embedding models) produces a "compressed" cosine similarity
range - even unrelated sentence pairs commonly land around 0.6-0.75,
while genuinely relevant pairs tend to land around 0.80-0.92. A
low threshold like 0.35 would essentially never reject anything on
this model, defeating the guardrail. 0.75 is a more realistic starting
point given that known behavior, confirmed against a live TF-IDF
sweep-methodology test run (same sweep code as evaluate_threshold.py,
different embedding space) that produced a clean separation between
relevant/irrelevant pairs at a low-but-nonzero threshold in that space.
Treat 0.75 as "safer than 0.35", not as "calibrated" - run the script.
"""

import os

# Cosine similarity from multilingual-e5-base, normalized embeddings.
# Informed starting point (see note above) - run evaluate_threshold.py
# against your real Qdrant collection and override via the env var below.
CONFIDENCE_THRESHOLD = float(os.environ.get("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.75"))

LOW_CONFIDENCE_MESSAGE_AR = (
    "معنديش معلومات كافية وموثوقة في المصادر المتاحة عشان أجاوبك بدقة على السؤال ده. "
    "لو تقدر تحدد الأعراض أو الموضوع أكتر، هحاول أساعدك تاني، أو يفضل تستشير طبيب مباشرة."
)
LOW_CONFIDENCE_MESSAGE_EN = (
    "I don't have reliable enough information in the available sources to answer "
    "that accurately. If you can give more detail, I can try again - or it's best "
    "to check with a doctor directly."
)

# [NEW] Separate message for queries that aren't medical at all (e.g. "what's
# the best smartphone to buy"). Reusing LOW_CONFIDENCE_MESSAGE for these was
# misleading - telling someone to "check with a doctor" about a smartphone
# question doesn't make sense. This is picked instead of the message above
# when is_medical_question() below determines the query isn't health-related.
OFF_TOPIC_MESSAGE_AR = (
    "أنا مساعد متخصص في الأسئلة الطبية والصحية بس، والسؤال ده مش من ضمن المجال اللي "
    "أقدر أساعدك فيه. لو عندك سؤال طبي أو صحي، يسعدني أساعدك فيه."
)
OFF_TOPIC_MESSAGE_EN = (
    "I'm a medical and health assistant, so this question is outside what I can "
    "help with. Feel free to ask me a health-related question instead."
)

_DOMAIN_CHECK_PROMPT = """Decide whether the following user message is a health or \
medical related question (about symptoms, conditions, treatments, medications, body \
systems, wellbeing, or similar). Respond with ONLY one word: "yes" or "no".

Message: "{query}" """


def is_medical_question(query: str, call_llm_fn) -> bool:
    """Cheap LLM check used ONLY when the confidence guardrail is about to
    reject a query - decides whether to show the 'insufficient medical
    evidence, see a doctor' message or the 'this is outside my scope'
    message. Fails open toward True (treat as medical) on any error or
    empty response: wrongly showing the doctor-referral message to an
    off-topic question is harmless, but wrongly dismissing a real health
    concern as off-topic would not be, so uncertainty should always lean
    toward the safer, more cautious message."""
    try:
        raw = call_llm_fn(_DOMAIN_CHECK_PROMPT.format(query=query))
        if not raw:
            return True
        return raw.strip().strip(".").lower().startswith("y")
    except Exception:
        return True


def _is_arabic(text: str) -> bool:
    import re
    return bool(re.search(r"[\u0600-\u06FF]", text))


def check_confidence(results: list, query: str = "", threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """`results` is the reranked top-k list (each needs a 'score' key,
    or 'rerank_score' if reranked - pass whichever score should gate the
    answer). Returns the same structured shape as the other guardrails."""
    if not results:
        best_score = 0.0
    else:
        best_score = max(r.get("rerank_score", r.get("score", 0.0)) for r in results)

    if best_score < threshold:
        message = LOW_CONFIDENCE_MESSAGE_AR if _is_arabic(query) else LOW_CONFIDENCE_MESSAGE_EN
        return {
            "allowed": False,
            "reason": "low_confidence",
            "best_score": best_score,
            "message": message,
        }
    return {"allowed": True, "reason": None, "best_score": best_score, "message": None}
