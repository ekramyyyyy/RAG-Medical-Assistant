"""
RAG pipeline for the medical assistant using Google GenAI SDK.
"""

import os
import re
import time
import concurrent.futures
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from search import load_data, search
from reranker import Reranker
from guardrails.emergency import (
    check_emergency,
    check_emergency_llm,   # [CHANGED] replaces build_emergency_index/check_emergency_semantic -
                            # the embedding-based semantic layer was firing on essentially any
                            # medical-sounding query (see guardrails/emergency.py's edit log).
                            # This reuses the pipeline's own call_llm as a classification fallback
                            # instead of an uncalibrated similarity threshold.
    get_emergency_response,
)
from guardrails.vagueness import is_vague
from guardrails.confidence import (
    check_confidence,
    is_medical_question,
    OFF_TOPIC_MESSAGE_AR,
    OFF_TOPIC_MESSAGE_EN,
)

# تحميل متغيرات البيئة تلقائياً من ملف .env
load_dotenv()

# ===== Configuration =====
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# ترتيب حسب الأولوية: كل عنصر (provider, model_name).
# Qwen (عبر Groq) الأساسي دلوقتي، وGemini احتياطي لو Groq مش متاح أو فشل.
LLM_PROVIDER_CHAIN = [
    ("groq", "qwen/qwen3.6-27b"),
    ("gemini", "gemini-3.6-flash"),
    ("gemini", "gemini-3.5-flash"),
]
PER_CALL_TIMEOUT_SECONDS = 12  # [FIX] 8s was too tight for a cold-start Gemini client (TLS+auth handshake on first call)
MAX_TOTAL_LLM_SECONDS = 25     # [FIX] must comfortably fit 2+ retries at the new per-call timeout above
MAX_CONTEXT_TOKENS = 3500
LLM_MAX_TOKENS = 1024
# [FIX] Groq's on_demand tier enforces an Output-Tokens-Per-Minute (OTPM) cap
# that can be as low as 1000 - without an explicit max_tokens, Qwen3 was
# requesting ~1023 output tokens and hitting a 429 rate_limit_exceeded on
# every single call. Keep this comfortably under that cap; override via env
# var if your org's quota differs.
GROQ_MAX_OUTPUT_TOKENS = int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "900"))
RETRIEVAL_TOP_K = 20   # [item #19] wide candidate pool for the reranker
FINAL_TOP_K = 5        # [item #19] what actually goes into the prompt

FALLBACK_MESSAGE = "هذه المعلومات غير متوفرة في المصادر الحالية. يرجى استشارة طبيب متخصص."

# [item #14-17] Rewritten to sound natural and to give a clear, non-hedgy
# instruction for the "not enough evidence" case instead of "the provided
# excerpts do not mention...".
SYSTEM_PROMPT_TEMPLATE = """You are a medical information assistant speaking directly to the user. \
Answer in the same language as the question (Arabic question -> Arabic answer, English -> English), \
and answer naturally and directly, the way a knowledgeable person would - never mention "the provided \
excerpts", "the retrieved context", "the sources above", vector search, or any detail of how you found \
the information.

Use the retrieved sources below as your factual evidence for this answer. Treat them as authoritative:
- Do not introduce medical claims that are not supported by them.
- Do not give precise medication dosages even if mentioned in the text - direct the user to a doctor or pharmacist for that.
- Do not state a definitive diagnosis. If the evidence mentions matching conditions, present them as general possibilities discussed in medical literature (e.g. "this can be seen with asthma or bronchitis"), and note that a specialist can confirm.
- Do not include any inline citation, source tag, or bracketed reference (e.g. "[Source: ...]", "(Source: ...)") anywhere in your answer. The retrieved evidence below is only for your own grounding - the app already shows the underlying sources to the user in a separate panel, so repeating them in your answer is redundant and must not happen.
- If the evidence genuinely does not support an answer to this specific question, say plainly that you don't have enough reliable information to answer that, and suggest what extra detail (duration, other symptoms, severity) would help - do NOT say the excerpts/sources "don't mention" it, just say you don't have enough information.
- End every response with a brief reminder that this is not a substitute for seeing a qualified doctor.

Retrieved evidence:
{context}

User question: {query}

Answer:"""


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)
# [FIX] Defensive second layer for the citation instruction above: even
# with the prompt telling the model never to emit inline source tags, a
# model can still copy the "[Source: ...]" pattern it sees in the context
# it was given (this happened in testing - it copied a filename verbatim
# as if it were a real citation). Strip any such bracket that slips
# through so the app's own Sources panel remains the single place sources
# are shown, and users never see a raw filename presented as a citation.
_SOURCE_TAG_RE = re.compile(r"\s*[\[(]\s*[Ss]ource\s*:.*?[\])]")


def _strip_source_tags(text: str) -> str:
    if not text:
        return text
    return _SOURCE_TAG_RE.sub("", text).strip()


def _strip_reasoning(text: str) -> str:
    """[FIX] Reasoning models (e.g. Qwen3 via Groq) can emit their internal
    chain-of-thought wrapped in <think>...</think> before the real answer.
    This must never reach the user - not the final answer text, not the
    rewritten search query, nothing. We ask the Groq API to hide it
    server-side (reasoning_format="hidden" in _call_provider), but this
    strip is a defensive second layer in case a provider/model ignores
    that param or leaks reasoning in a different call path."""
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # Defensive: if a <think> tag was opened but never closed (e.g. the
    # response got truncated mid-reasoning), drop everything from it on
    # rather than showing a dangling raw tag + half a thought to the user.
    cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
    return cleaned.strip()


def _needs_rewrite(query: str) -> bool:
    """[item #18] Skip the extra LLM call when the query is already
    clear enough for vector search: pure-English queries of a decent
    length are assumed to already use reasonable medical/search terms.
    Short queries and anything containing Arabic (colloquial symptom
    descriptions) still get rewritten/translated."""
    import re
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", query))
    if has_arabic:
        return True
    return len(query.split()) < 4


class RAGPipeline:
    """Encapsulates guardrails, retrieval, reranking, prompt construction,
    and LLM calls for the medical assistant."""

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        llm_max_tokens: int = LLM_MAX_TOKENS,
    ):
        self.max_context_tokens = max_context_tokens
        self.llm_max_tokens = llm_max_tokens

        print("Connecting to vector DB and loading embedding model...")
        self.qdrant_client, self.collection_name = load_data()
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # [REMOVED] The emergency semantic embedding index used to be built
        # here (build_emergency_index). It's gone - see guardrails/emergency.py's
        # edit log: an uncalibrated embedding-similarity threshold started
        # flagging plain informational questions as cardiac emergencies.
        # Emergency detection now falls back to check_emergency_llm(), which
        # needs no precomputed index - self.embedding_model stays here only
        # for retrieval.

        self.reranker = Reranker()
        _ = self.reranker.model  # force download/load now, not mid-query
        print("Loaded successfully.")

        self._llm_client = None
        self._groq_client = None

    @property
    def llm_client(self):
        """Lazily initialize the Gemini client so import time stays fast."""
        if self._llm_client is None:
            from google import genai
            self._llm_client = genai.Client()
        return self._llm_client

    @property
    def groq_client(self):
        """Lazily initialize the Groq client (last-resort fallback, reads GROQ_API_KEY)."""
        if self._groq_client is None:
            from groq import Groq
            self._groq_client = Groq()
        return self._groq_client

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count estimate based on whitespace-separated words."""
        return int(len(text.split()) * 1.3)

    def rewrite_query(self, query: str) -> str:
        """Translates and expands colloquial user queries into standard
        English medical terminology to optimize vector search cosine
        similarity. [item #18] Only called when _needs_rewrite() is True."""
        rewrite_prompt = f"""You are a medical search query generator.
Convert the following user symptom complaint or question into concise, standard English medical terminology keywords suitable for searching a medical textbook index or vector database.
Keep it strictly to key clinical terms/symptoms without conversational filler.

User Input: {query}
Search Keywords (in English):"""
        try:
            expanded_query = self.call_llm(rewrite_prompt)
            print(f"\n[Query Rewritten for Search]: {expanded_query.strip()}")
            return expanded_query.strip()
        except Exception:
            return query  # Fallback to original query on error

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K):
        """Runs search() against the Qdrant collection to get the raw
        top-k candidates (before reranking)."""
        return search(
            query,
            self.embedding_model,
            self.qdrant_client,
            self.collection_name,
            top_k=top_k,
        )

    def build_context(self, search_query: str, reranked_results: list):
        """Trims the already-reranked top results to stay within the
        max token budget. Returns (context_text, sources_list)."""
        context_parts = []
        sources = []
        total_tokens = 0

        for result in reranked_results:
            chunk_text = result.get("text", "")
            chunk_tokens = self.estimate_tokens(chunk_text)

            if total_tokens + chunk_tokens > self.max_context_tokens:
                break

            source_name = result.get("source_file") or result.get("topic") or "Unknown Source"

            context_parts.append(f"[Source: {source_name}]\n{chunk_text}")
            sources.append({
                "source": source_name,
                "score": round(float(result.get("score", 0.0)), 3),
                "rerank_score": round(float(result.get("rerank_score", 0.0)), 3),
            })
            total_tokens += chunk_tokens

        context_text = "\n\n---\n\n".join(context_parts)
        return context_text, sources

    def build_prompt(self, query: str, context_text: str) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(context=context_text, query=query)

    def _call_provider(self, provider: str, model_name: str, prompt: str) -> str:
        """نداء واحد لمزوّد واحد (groq أو gemini)، بحد أقصى زمني صارم عن طريق thread."""
        if provider == "groq":
            # [FIX] reasoning_format="hidden" tells Groq's API to drop the
            # model's internal chain-of-thought server-side for reasoning
            # models (Qwen3, DeepSeek-R1, etc.) instead of prepending it to
            # `content` as a <think>...</think> block. Without this, the
            # raw reasoning (sometimes even in the wrong language) leaks
            # straight into what the user sees.
            # [FIX] max_tokens is now explicitly capped - without it, Qwen3
            # was requesting ~1023 output tokens and hitting Groq's 1000
            # OTPM (output-tokens-per-minute) rate limit on every call.
            # [FIX #2] reasoning_effort="none" turns off the model's internal
            # chain-of-thought entirely. This task (rewriting a search query,
            # writing an answer from retrieved context, or classifying a
            # message) doesn't need step-by-step reasoning. Without this, the
            # model was spending the *entire* max_tokens budget on hidden
            # reasoning and returning an EMPTY final answer, and taking long
            # enough to also trip the 12s timeout.
            # [FIX #3] temperature is now capped low - without it, Qwen3 was
            # occasionally leaking stray tokens from other languages into an
            # otherwise-Arabic answer. A low, near-deterministic temperature
            # is also the right choice generally for a RAG assistant meant
            # to stick closely to retrieved facts rather than write
            # creatively, and for a classification task like emergency
            # detection where we want a stable, repeatable verdict.
            call_fn = lambda: self.groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                reasoning_format="hidden",
                reasoning_effort="none",
                max_tokens=GROQ_MAX_OUTPUT_TOKENS,
                temperature=0.3,
            ).choices[0].message.content
        else:  # "gemini"
            from google.genai import types
            call_fn = lambda: self.llm_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=self.llm_max_tokens,
                    temperature=0.3,  # [FIX #3] same reasoning as the Groq call above
                ),
            ).text

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call_fn)
            return future.result(timeout=PER_CALL_TIMEOUT_SECONDS)

    def call_llm(self, prompt: str) -> str:
        """
        يجرب كل مزوّد في LLM_PROVIDER_CHAIN بالترتيب (Qwen عبر Groq أولًا،
        Gemini احتياطي). أي فشل (503، rate limit، timeout، مفتاح ناقص...)
        يروح فورًا للي بعده بدل ما يستنى أو يكسّر الـ request كله.
        لو الوقت الكلي عدّى MAX_TOTAL_LLM_SECONDS أو كل المزوّدين فشلوا،
        يرجع None (وanswer() بترجع رسالة واضحة للمستخدم بدل الكراش، أو
        بالنسبة للـ emergency guardrail، بترجع None فتبقى "مش طوارئ").
        """
        start = time.monotonic()

        for provider, model_name in LLM_PROVIDER_CHAIN:
            if time.monotonic() - start > MAX_TOTAL_LLM_SECONDS:
                break

            if provider == "groq" and not os.environ.get("GROQ_API_KEY"):
                continue  # المفتاح مش متاح، سيبه وكمّل للي بعده
            if provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
                continue

            try:
                raw = self._call_provider(provider, model_name, prompt)
                # [FIX] Defensive second layer - strip any leaked reasoning
                # regardless of provider, in case reasoning_format is ever
                # ignored or a different model/provider leaks it another way.
                return _strip_source_tags(_strip_reasoning(raw))
            except concurrent.futures.TimeoutError:
                print(f"\n[{model_name} بطيء جدًا، بجرب اللي بعده]")
                continue
            except Exception as e:
                print(f"\n[{model_name} فشل ({e}), بجرب اللي بعده]")
                continue

        return None

    def answer(self, query: str) -> dict:
        """
        Full pipeline, guardrails-first (item #10):
          1. Emergency guardrail - two layers:
             (a) exact/normalized keywords (check_emergency) - deterministic,
                 zero-cost, never depends on any external service. Always
                 checked first and is the guaranteed floor.
             (b) LLM classification fallback (check_emergency_llm) - only
                 runs when (a) finds nothing. Reuses self.call_llm (same
                 Groq->Gemini chain as the rest of the pipeline). Fails
                 open: any error or unavailable provider => treated as
                 "not an emergency", never blocks the request.
          2. Vagueness guardrail (rule-based, LLM fallback only if ambiguous)
          3. Retrieval (top-20) + reranking (top-5)
          4. Confidence guardrail on the reranked top score
          5. Prompt + Gemini call
        """
        # 1. Emergency
        emergency_category = check_emergency(query)
        subject = "self"
        if not emergency_category:
            emergency_category, subject = check_emergency_llm(query, self.call_llm)
        if emergency_category:
            return {
                "answer": get_emergency_response(emergency_category, subject, query),
                "sources": [],
                "emergency": True,
                "guardrail": None,
            }

        # 2. Vagueness (rule-based first; LLM fallback left available for
        # you to wire in for genuinely ambiguous middle-length queries -
        # see guardrails/vagueness.llm_vagueness_check)
        vague_check = is_vague(query)
        if not vague_check["allowed"]:
            return {
                "answer": vague_check["message"],
                "sources": [],
                "emergency": False,
                "guardrail": vague_check["reason"],
            }

        # 3. Retrieval + rerank
        search_query = self.rewrite_query(query) if _needs_rewrite(query) else query
        candidates = self.retrieve(search_query, top_k=RETRIEVAL_TOP_K)
        reranked = self.reranker.rerank(search_query, candidates, top_k=FINAL_TOP_K)

        # 4. Confidence guardrail
        confidence_check = check_confidence(reranked, query=query)
        if not confidence_check["allowed"]:
            # [FIX] The score threshold alone can't tell "a real medical
            # question with weak retrieval" apart from "not a medical
            # question at all" - both land here with a low score. Ask once,
            # cheaply, only in this failure path, which message fits.
            is_ar = bool(re.search(r"[\u0600-\u06FF]", query))
            if is_medical_question(query, self.call_llm):
                message = confidence_check["message"]
            else:
                message = OFF_TOPIC_MESSAGE_AR if is_ar else OFF_TOPIC_MESSAGE_EN
            return {
                "answer": message,
                "sources": [],
                "emergency": False,
                "guardrail": confidence_check["reason"],
            }

        context_text, sources = self.build_context(search_query, reranked)
        if not context_text.strip():
            return {"answer": FALLBACK_MESSAGE, "sources": [], "emergency": False, "guardrail": "no_context"}

        # 5. Generation
        prompt = self.build_prompt(query, context_text)
        answer_text = self.call_llm(prompt)

        # [FIX] Treat an empty string the same as None - a reasoning model
        # that burns its whole token budget on hidden thinking can return ""
        # instead of raising, which would otherwise slip past this guard and
        # show the user a blank answer instead of the fallback message.
        if not answer_text:
            is_ar = bool(__import__("re").search(r"[\u0600-\u06FF]", query))
            message = (
                "الخدمة مشغولة جدًا دلوقتي، جرب تاني بعد شوية. لو الأعراض مقلقة، متأخرش في استشارة طبيب."
                if is_ar else
                "The service is very busy right now — please try again shortly. If your symptoms are concerning, don't delay seeing a doctor."
            )
            return {
                "answer": message,
                "sources": [],
                "emergency": False,
                "guardrail": "llm_unavailable",
            }

        return {"answer": answer_text, "sources": sources, "emergency": False, "guardrail": None}


def run_interactive_session(pipeline: RAGPipeline) -> None:
    """Simple REPL loop for testing the pipeline from the terminal."""
    print("Type your question (or 'exit' to quit):")

    while True:
        query = input("\nQuestion: ").strip()

        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        result = pipeline.answer(query)

        print("\n=== Answer ===")
        print(result["answer"])
        print("\n=== Sources ===")
        print(result["sources"])


if __name__ == "__main__":
    rag_pipeline = RAGPipeline()
    run_interactive_session(rag_pipeline)
