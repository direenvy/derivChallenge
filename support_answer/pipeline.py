"""The query flow, in order:

    LOAD_KB -> RETRIEVE -> CHECK_SUPPORT -> ANSWER | ABSTAIN

``SupportAnswerService.from_path`` performs LOAD_KB once (at startup, so the
HTTP server does not re-read the file per request); ``ask`` performs the other
three stages per question. The composers in ``answer.py`` are only reachable
from the ANSWER branch, and only ever receive retrieved evidence.
"""

from typing import Any, Dict, List, Optional, Sequence

from .answer import compose
from .config import DEFAULT_KB_PATH, Settings
from .kb import Document, build_passages, load_kb
from .retrieval import Bm25Index
from .schema import build_response
from .support import check_support, select_evidence

STAGES = ("LOAD_KB", "RETRIEVE", "CHECK_SUPPORT", "ANSWER|ABSTAIN")


class SupportAnswerService:
    """A loaded knowledge base plus the query flow over it."""

    def __init__(
        self,
        documents: Sequence[Document],
        settings: Optional[Settings] = None,
        kb_path: Optional[str] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.documents: List[Document] = list(documents)
        self.kb_path = kb_path
        self.passages = build_passages(self.documents, self.settings)
        self.index = Bm25Index(self.passages, self.settings)

    @classmethod
    def from_path(
        cls, kb_path: str = DEFAULT_KB_PATH, settings: Optional[Settings] = None
    ) -> "SupportAnswerService":
        """STAGE LOAD_KB -- read and validate the knowledge base from disk."""
        documents = load_kb(kb_path)
        return cls(documents, settings=settings, kb_path=kb_path)

    # ------------------------------------------------------------------ ask

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question, or abstain. Returns the response contract."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        question = question.strip()

        # STAGE RETRIEVE -- must complete before any answer is composed.
        retrieved = self.index.search(question)
        evidence = select_evidence(self.index, question, retrieved)

        # STAGE CHECK_SUPPORT
        support = check_support(self.index, evidence, self.settings)

        debug: Dict[str, Any] = {
            "retrieved_ids": evidence.retrieved_ids,
            "support_score": support.support_score,
            "norm_score": support.norm_score,
            "coverage": support.coverage,
            "retrieved_scores": [
                {"id": hit.passage.doc_id, "score": round(hit.score, 4)}
                for hit in evidence.retrieved
            ],
            "gates": support.gates,
            "decision_reason": support.reason,
            "kb": {"path": self.kb_path, "documents": len(self.documents),
                   "passages": len(self.passages)},
            "stages": list(STAGES),
        }

        # STAGE ABSTAIN
        if not support.should_answer:
            debug["answerer_used"] = "none"
            return build_response(
                question=question,
                decision="abstain",
                answer=self.settings.abstain_message,
                citations=[],
                debug=debug,
            )

        # STAGE ANSWER
        result = compose(evidence, self.settings)
        debug["answerer_used"] = result.answerer_used
        if result.notes:
            debug["answerer_notes"] = result.notes

        # Safety net: the support check guarantees evidence, but never return a
        # bare "answer" decision without citations or text.
        if not result.citations or not result.text.strip():
            debug["answerer_used"] = "none"
            debug["decision_reason"] = "composer produced no cited answer"
            return build_response(
                question=question,
                decision="abstain",
                answer=self.settings.abstain_message,
                citations=[],
                debug=debug,
            )

        return build_response(
            question=question,
            decision="answer",
            answer=result.text,
            citations=result.citations,
            debug=debug,
        )
