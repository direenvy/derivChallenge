"""ANSWER stage: compose an answer strictly from retrieved evidence.

Two interchangeable composers share one interface:

* ``rule``  -- extractive. The answer is verbatim sentences from the retrieved
  passages, so it is grounded by construction.
* ``llm``   -- an Anthropic model rewrites the retrieved passages, then a
  grounding check verifies the result against those same passages. Any
  failure (no SDK, no credentials, API error, failed check) falls back to the
  extractive answer.

This module imports neither ``kb`` nor ``retrieval``: it can only see the
:class:`~support_answer.support.Evidence` handed to it, which is what makes
"retrieval before generation" a structural guarantee rather than a convention.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .config import Settings
from .support import Evidence, EvidenceSentence
from .text import content_terms, truncate

LLM_SYSTEM_PROMPT = (
    "You are a customer-support answer composer. You will be given CONTEXT "
    "passages from a support knowledge base and a user QUESTION.\n"
    "Rules:\n"
    "1. Use only facts stated in the CONTEXT. Never add steps, timings, "
    "policies, product behaviour or reassurance that is not written there.\n"
    "2. Do not give legal, tax or financial advice, and never promise or "
    "guarantee an outcome or a processing time.\n"
    "3. Answer in at most three short sentences, in plain language.\n"
    "4. If the CONTEXT does not contain the answer, reply with exactly: "
    "INSUFFICIENT_CONTEXT\n"
    "Return the answer text only, with no preamble and no citations."
)


class LlmUnavailable(RuntimeError):
    """Raised when the LLM composer cannot be used; triggers the fallback."""


@dataclass
class AnswerResult:
    """A composed answer plus the citations that support it."""

    text: str
    citations: List[Dict[str, str]]
    answerer_used: str
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# selection shared by both composers
# --------------------------------------------------------------------------


def select_supporting_sentences(
    evidence: Evidence, settings: Settings
) -> List[EvidenceSentence]:
    """Highest-scoring sentences, capped per answer and per document.

    Returned in reading order (best passage first, original sentence order
    within a passage) so the composed answer flows like the source article.
    """
    if not evidence.sentences:
        return []
    # Relative floor: a sentence that barely overlaps the question would pad
    # the answer and add an irrelevant citation.
    floor = evidence.sentences[0].score * settings.min_sentence_score_ratio
    top_retrieval = evidence.top_score or 1.0
    chosen: List[EvidenceSentence] = []
    doc_ids: List[str] = []
    for sentence in evidence.sentences:
        if sentence.score < floor:
            break
        if len(chosen) >= settings.max_answer_sentences:
            break
        if sentence.passage_rank > 0:
            # A second document may only contribute if it was retrieved
            # competitively. Otherwise one shared low-IDF word is enough to
            # attach an off-topic citation.
            rank_score = evidence.retrieved[sentence.passage_rank].score
            if rank_score < top_retrieval * settings.min_secondary_passage_ratio:
                continue
        if sentence.doc_id not in doc_ids:
            if len(doc_ids) >= settings.max_answer_docs:
                continue
            doc_ids.append(sentence.doc_id)
        chosen.append(sentence)
    chosen.sort(key=lambda s: (s.passage_rank, s.position))
    return chosen


def build_citations(
    sentences: Sequence[EvidenceSentence], settings: Settings
) -> List[Dict[str, str]]:
    """One citation per contributing document, snippet = the sentences used.

    Grouping by parent document id means chunking long articles never leaks
    into the response contract.
    """
    grouped: Dict[str, Dict[str, object]] = {}
    for sentence in sentences:
        entry = grouped.setdefault(
            sentence.doc_id,
            {"id": sentence.doc_id, "title": sentence.title, "parts": []},
        )
        entry["parts"].append(sentence.text)  # type: ignore[union-attr]
    citations = []
    for entry in grouped.values():
        snippet = " ".join(entry["parts"])  # type: ignore[arg-type]
        citations.append(
            {
                "id": str(entry["id"]),
                "title": str(entry["title"]),
                "snippet": truncate(snippet, settings.snippet_max_chars),
            }
        )
    return citations


# --------------------------------------------------------------------------
# composers
# --------------------------------------------------------------------------


def compose_extractive(
    sentences: Sequence[EvidenceSentence], settings: Settings
) -> str:
    """Join the supporting sentences verbatim."""
    parts = []
    for sentence in sentences:
        text = " ".join(sentence.text.split())
        if text and text[-1] not in ".!?":
            text += "."
        parts.append(text)
    return " ".join(parts)


def grounding_ratio(answer: str, context: str) -> float:
    """Fraction of the answer's content terms that occur in the context.

    1.0 means every content word of the answer is present in the retrieved
    passages. Used to reject an LLM answer that introduced outside facts.
    """
    answer_terms = set(content_terms(answer))
    if not answer_terms:
        return 0.0
    context_terms = set(content_terms(context))
    return len(answer_terms & context_terms) / len(answer_terms)


def compose_llm(evidence: Evidence, settings: Settings) -> str:
    """Compose via the Anthropic Messages API, using only the retrieved context.

    Raises :class:`LlmUnavailable` for every failure path so the caller can
    fall back to the extractive answer.
    """
    try:
        import anthropic  # noqa: PLC0415  (optional dependency, imported lazily)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise LlmUnavailable("anthropic SDK is not installed") from exc

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise LlmUnavailable("no ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN in environment")

    prompt = (
        f"CONTEXT:\n{evidence.context_text}\n\n"
        f"QUESTION: {evidence.question}\n\n"
        "Answer using only the CONTEXT above."
    )
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
    except Exception as exc:  # pragma: no cover - network/credential dependent
        raise LlmUnavailable(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

    if not text:
        raise LlmUnavailable("LLM returned an empty answer")
    if "INSUFFICIENT_CONTEXT" in text:
        raise LlmUnavailable("LLM reported insufficient context")
    return text


def compose(evidence: Evidence, settings: Settings) -> AnswerResult:
    """Compose the answer with the configured answering layer.

    ``rule`` is used directly; ``llm`` and ``auto`` try the model first and
    degrade to the extractive answer, recording why in ``notes``.
    """
    sentences = select_supporting_sentences(evidence, settings)
    citations = build_citations(sentences, settings)
    extractive = compose_extractive(sentences, settings)

    if settings.answerer == "rule":
        return AnswerResult(text=extractive, citations=citations, answerer_used="rule")

    notes: List[str] = []
    try:
        text = compose_llm(evidence, settings)
    except LlmUnavailable as exc:
        notes.append(f"llm_fallback: {exc}")
        return AnswerResult(
            text=extractive, citations=citations, answerer_used="rule", notes=notes
        )

    ratio = grounding_ratio(text, evidence.context_text)
    notes.append(f"llm_grounding_ratio: {round(ratio, 3)}")
    if ratio < settings.llm_grounding_min:
        notes.append(
            f"llm_fallback: grounding ratio below {settings.llm_grounding_min}"
        )
        return AnswerResult(
            text=extractive, citations=citations, answerer_used="rule", notes=notes
        )
    return AnswerResult(text=text, citations=citations, answerer_used="llm", notes=notes)
