"""CHECK_SUPPORT stage: decide whether the retrieved evidence can support an
answer, or whether the service must abstain.

The rule is a conjunction of six independent gates (G1-G6). Every gate is
cheap, deterministic and individually explainable, and each one is reported in
``debug.gates`` so a failure can be traced. See DESIGN.md section 4.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

from .config import Settings
from .retrieval import Bm25Index, Retrieved
from .text import (
    adjacent_pairs,
    content_terms,
    numeric_tokens,
    split_sentences,
    stem,
    tokenize,
    unique,
)

#: Negation / limitation cues used by gate G3.
NEGATION_CUES: Set[str] = {
    "not", "cannot", "cant", "never", "no", "nor", "without", "beyond",
    "unable", "neither",
}

#: Advice- or provision-giving cues used by gate G3 (stemmed).
#: Only verbs about *supplying an answer or assurance* count. Verbs about
#: product behaviour ("view", "send", "submit") deliberately do not, because
#: "support agents cannot view passwords" is a good grounded answer.
ADVICE_CUE_WORDS: Set[str] = {
    "provide", "offer", "advice", "advise", "guarantee", "recommend",
    "promise", "assurance", "assure", "opinion",
}

#: Stemmed at import time so the cue set cannot drift from the stemmer.
ADVICE_CUE_STEMS: Set[str] = {stem(word) for word in ADVICE_CUE_WORDS}


@dataclass(frozen=True)
class EvidenceSentence:
    """One candidate supporting sentence, scored against the question."""

    doc_id: str
    title: str
    text: str
    score: float
    passage_rank: int
    position: int


@dataclass
class Evidence:
    """Everything retrieval produced, and nothing else.

    This is the only object the answer composers receive, which is what stops
    an answer from being produced before retrieval has run.
    """

    question: str
    query_terms: List[str]
    retrieved: List[Retrieved]
    sentences: List[EvidenceSentence] = field(default_factory=list)

    @property
    def retrieved_ids(self) -> List[str]:
        """Parent document ids of the retrieved passages, de-duplicated."""
        return unique([r.passage.doc_id for r in self.retrieved])

    @property
    def top_score(self) -> float:
        return self.retrieved[0].score if self.retrieved else 0.0

    @property
    def context_text(self) -> str:
        """Retrieved passages as a single block, for the LLM composer."""
        return "\n\n".join(
            f"[{r.passage.doc_id}] {r.passage.title}\n{r.passage.text}"
            for r in self.retrieved
        )


@dataclass(frozen=True)
class SupportDecision:
    """Outcome of the support check."""

    decision: str  # "answer" | "abstain"
    support_score: float
    norm_score: float
    coverage: float
    gates: Dict[str, object]
    reason: str

    @property
    def should_answer(self) -> bool:
        return self.decision == "answer"


# --------------------------------------------------------------------------
# evidence selection
# --------------------------------------------------------------------------


def select_evidence(index: Bm25Index, question: str, retrieved: Sequence[Retrieved]) -> Evidence:
    """Score the sentences of the retrieved passages against the question.

    Sentence score is the IDF-weighted overlap with the question terms, with a
    small bonus for sentences from a higher-ranked passage so that ties resolve
    toward the best-matching document.
    """
    query_terms = index.query_terms(question)
    evidence = Evidence(question=question, query_terms=query_terms, retrieved=list(retrieved))
    if not query_terms:
        return evidence

    wanted = set(query_terms)
    sentences: List[EvidenceSentence] = []
    for rank, hit in enumerate(retrieved):
        bonus = 1.0 + (0.05 if rank == 0 else 0.0)
        for position, raw in enumerate(split_sentences(hit.passage.text)):
            terms = set(content_terms(raw))
            overlap = wanted & terms
            if not overlap:
                continue
            score = sum(index.idf(term) for term in overlap) * bonus
            sentences.append(
                EvidenceSentence(
                    doc_id=hit.passage.doc_id,
                    title=hit.passage.title,
                    text=raw,
                    score=score,
                    passage_rank=rank,
                    position=position,
                )
            )
    sentences.sort(key=lambda s: (-s.score, s.passage_rank, s.position))
    evidence.sentences = sentences
    return evidence


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------


def passage_coverage(index: Bm25Index, hit: Retrieved, query_terms: Sequence[str]) -> float:
    """IDF-weighted fraction of the question's *known* terms covered by one passage.

    Terms absent from the whole knowledge base are excluded from the
    denominator: an unknown word says nothing about which passage is on topic,
    and it is already penalised by gate G1 (where it inflates the maximum
    achievable score and so drives the normalised score down). What remains
    measures whether a *single* passage is about the whole question.
    """
    known = [t for t in query_terms if index.document_freq.get(t, 0) > 0]
    total = sum(index.idf(t) for t in known)
    if total <= 0.0:
        return 0.0
    covered = sum(index.idf(t) for t in known if t in hit.indexed.term_freq)
    return covered / total


def is_conduct_limitation(sentence: str) -> bool:
    """True when a sentence forbids *giving this kind of answer* (gate G3).

    Requires a negation cue together with an advice/provision cue in the same
    sentence, e.g. "should not provide legal advice, tax advice, or
    guarantees...". A negation about product behaviour ("cannot view or send
    existing passwords") does not qualify.
    """
    raw = tokenize(sentence)
    if not any(token in NEGATION_CUES for token in raw):
        return False
    return any(stem(token) in ADVICE_CUE_STEMS for token in raw)


def check_support(index: Bm25Index, evidence: Evidence, settings: Settings) -> SupportDecision:
    """Apply gates G1-G6. Any failure means abstain."""
    max_possible = index.max_possible_score(evidence.query_terms)
    norm_score = (evidence.top_score / max_possible) if max_possible > 0 else 0.0
    norm_score = min(1.0, norm_score)

    coverage = 0.0
    best_hit = None
    for hit in evidence.retrieved:
        value = passage_coverage(index, hit, evidence.query_terms)
        if best_hit is None or value > coverage:
            coverage, best_hit = value, hit

    best_sentence = evidence.sentences[0] if evidence.sentences else None
    limitation = bool(best_sentence) and is_conduct_limitation(best_sentence.text)

    evidence_numbers = set()
    for hit in evidence.retrieved:
        evidence_numbers.update(numeric_tokens(hit.passage.text))
    question_numbers = set(numeric_tokens(evidence.question))
    unsupported_numbers = sorted(question_numbers - evidence_numbers)

    # G6 -- any phrase the question names that the knowledge base knows must
    # appear in the passage the answer would come from. Catches a question
    # whose best passage is a neighbouring topic ("email address" asked, but
    # the best passage only covers "residential address").
    question_pairs = adjacent_pairs(content_terms(evidence.question))
    known_pairs = question_pairs & index.kb_pairs
    missing_pairs = sorted(
        "+".join(sorted(pair))
        for pair in known_pairs
        if best_hit is not None and pair not in best_hit.indexed.pairs
    )

    gates = {
        "G1_retrieval_strength": {
            "value": round(norm_score, 4),
            "threshold": settings.min_norm_score,
            "pass": norm_score >= settings.min_norm_score,
        },
        "G2_evidence_coverage": {
            "value": round(coverage, 4),
            "threshold": settings.min_coverage,
            "pass": coverage >= settings.min_coverage,
        },
        "G3_not_conduct_limitation": {
            "value": limitation,
            "pass": not limitation,
        },
        "G4_no_unsupported_quantity": {
            "value": unsupported_numbers,
            "pass": not unsupported_numbers,
        },
        "G5_has_evidence_sentence": {
            "value": len(evidence.sentences),
            "pass": bool(evidence.sentences),
        },
        "G6_phrase_consistency": {
            "value": missing_pairs,
            "pass": not missing_pairs,
        },
    }

    support_score = round(0.5 * norm_score + 0.5 * coverage, 4)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    if failed:
        return SupportDecision(
            decision="abstain",
            support_score=support_score,
            norm_score=round(norm_score, 4),
            coverage=round(coverage, 4),
            gates=gates,
            reason="failed " + ", ".join(failed),
        )
    return SupportDecision(
        decision="answer",
        support_score=support_score,
        norm_score=round(norm_score, 4),
        coverage=round(coverage, 4),
        gates=gates,
        reason="all gates passed",
    )
