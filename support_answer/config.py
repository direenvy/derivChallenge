"""Tunable settings for the support-answer service.

Every threshold the pipeline uses lives here, so the support rule can be
re-tuned without touching logic. Defaults are the values documented in
README.md and DESIGN.md.
"""

from dataclasses import dataclass, replace

DEFAULT_KB_PATH = "kb.json"

ABSTAIN_MESSAGE = (
    "I do not have enough supporting information in the knowledge base to "
    "answer that reliably. Please contact human support so an agent can help "
    "with this request."
)


@dataclass(frozen=True)
class Settings:
    """Immutable configuration bundle. Use ``Settings().with_(top_k=5)``."""

    # --- retrieval -------------------------------------------------------
    top_k: int = 3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    title_boost: int = 2

    # --- support check (gates G1-G6, see DESIGN.md section 4) ------------
    min_norm_score: float = 0.12
    min_coverage: float = 0.55

    # --- answer composition ---------------------------------------------
    max_answer_sentences: int = 3
    max_answer_docs: int = 2
    min_sentence_score_ratio: float = 0.4
    min_secondary_passage_ratio: float = 0.5
    snippet_max_chars: int = 300

    # --- chunking (see DESIGN.md section 6) -----------------------------
    chunk_threshold_chars: int = 1200
    chunk_sentences: int = 6
    chunk_overlap_sentences: int = 2

    # --- answering layer -------------------------------------------------
    answerer: str = "rule"  # "rule" | "llm" | "auto"
    llm_model: str = "claude-opus-5-5"
    llm_max_tokens: int = 512
    llm_grounding_min: float = 0.80

    abstain_message: str = ABSTAIN_MESSAGE

    def with_(self, **overrides: object) -> "Settings":
        """Return a copy with the given fields replaced."""
        return replace(self, **overrides)  # type: ignore[arg-type]


ANSWERER_MODES = ("rule", "llm", "auto")
