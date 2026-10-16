"""Tunable settings for the support-answer service.

Every threshold the pipeline uses lives here, so the support rule can be
re-tuned without touching logic. Defaults are the values documented in
README.md and DESIGN.md.
"""

import os
from dataclasses import dataclass, replace
from typing import List

DEFAULT_KB_PATH = "kb.json"
DEFAULT_ENV_FILE = ".env"

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
    llm_provider: str = "auto"  # "auto" | "gemini" | "anthropic"
    llm_model: str = ""  # empty means the provider default
    llm_max_tokens: int = 512
    llm_timeout: float = 30.0
    llm_grounding_min: float = 0.80

    abstain_message: str = ABSTAIN_MESSAGE

    def with_(self, **overrides: object) -> "Settings":
        """Return a copy with the given fields replaced."""
        return replace(self, **overrides)  # type: ignore[arg-type]


ANSWERER_MODES = ("rule", "llm", "auto")
LLM_PROVIDERS = ("auto", "gemini", "anthropic")

#: Default model per provider. Overridable with --llm-model or the
#: GEMINI_MODEL / ANTHROPIC_MODEL environment variables.
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-opus-5-5",
}

#: Credential environment variables checked per provider, in order.
PROVIDER_ENV_KEYS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
}


def load_env_file(path: str = DEFAULT_ENV_FILE) -> List[str]:
    """Load ``KEY=VALUE`` lines from a local ``.env`` into the environment.

    Convenience only, for the optional LLM layer: a real environment variable
    always wins, missing files are fine, and nothing here is required for the
    rule-based path. Called by the entry points rather than at import time, so
    importing the package never mutates the environment.

    Returns the names (never the values) of the variables it set.
    """
    loaded: List[str] = []
    if not os.path.isfile(path):
        return loaded
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return loaded
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded
