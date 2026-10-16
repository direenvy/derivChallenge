"""RETRIEVE stage: an Okapi BM25 index over the knowledge-base passages.

Hand-written so the core path has no third-party dependencies, and so the IDF
weights are available to the support check (see ``support.py``).
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Sequence, Set

from .config import Settings
from .kb import Passage
from .text import acronym_variants, adjacent_pairs, content_terms, unique


@dataclass
class IndexedPassage:
    """A passage plus its indexed term statistics."""

    passage: Passage
    terms: List[str]
    term_freq: Counter = field(default_factory=Counter)
    pairs: Set[FrozenSet[str]] = field(default_factory=set)

    @property
    def length(self) -> int:
        return len(self.terms)

    @property
    def term_set(self) -> Set[str]:
        return set(self.term_freq)


@dataclass(frozen=True)
class Retrieved:
    """A scored search hit."""

    indexed: IndexedPassage
    score: float

    @property
    def passage(self) -> Passage:
        return self.indexed.passage


class Bm25Index:
    """Okapi BM25 over passages, with a title boost and KB-derived acronyms."""

    def __init__(self, passages: Sequence[Passage], settings: Settings) -> None:
        self.settings = settings
        self.passages: List[IndexedPassage] = []
        document_freq: Counter = Counter()

        self.kb_pairs: Set[FrozenSet[str]] = set()
        for passage in passages:
            terms = self._index_terms(passage, settings)
            pairs = adjacent_pairs(content_terms(passage.text)) | adjacent_pairs(
                content_terms(passage.title)
            )
            indexed = IndexedPassage(
                passage=passage, terms=terms, term_freq=Counter(terms), pairs=pairs
            )
            self.passages.append(indexed)
            self.kb_pairs |= pairs
            for term in indexed.term_set:
                document_freq[term] += 1

        self.document_freq: Dict[str, int] = dict(document_freq)
        self.count = len(self.passages)
        total_length = sum(p.length for p in self.passages)
        self.avg_length = (total_length / self.count) if self.count else 0.0
        self._idf_cache: Dict[str, float] = {}

    @staticmethod
    def _index_terms(passage: Passage, settings: Settings) -> List[str]:
        """Terms for one passage: body + boosted title + title acronyms.

        The title is repeated ``title_boost`` times because an FAQ title is the
        topic label and carries more signal per word than the body.
        """
        terms = content_terms(passage.text)
        title_terms = content_terms(passage.title)
        terms.extend(title_terms * max(1, settings.title_boost))
        terms.extend(acronym_variants(passage.title))
        return terms

    # ------------------------------------------------------------------ IDF

    def idf(self, term: str) -> float:
        """BM25 inverse document frequency, always positive."""
        if term in self._idf_cache:
            return self._idf_cache[term]
        if not self.count:
            return 0.0
        df = self.document_freq.get(term, 0)
        value = math.log(1.0 + (self.count - df + 0.5) / (df + 0.5))
        self._idf_cache[term] = value
        return value

    def query_terms(self, question: str) -> List[str]:
        """Unique content terms of the question, in order."""
        return unique(content_terms(question))

    def max_possible_score(self, query_terms: Sequence[str]) -> float:
        """Upper bound on the BM25 score this query could reach.

        Used to normalise the top score into [0, 1] so one threshold works
        across questions and across knowledge bases of different sizes.
        """
        k1 = self.settings.bm25_k1
        return sum(self.idf(term) * (k1 + 1.0) for term in query_terms)

    # --------------------------------------------------------------- search

    def score(self, indexed: IndexedPassage, query_terms: Sequence[str]) -> float:
        k1 = self.settings.bm25_k1
        b = self.settings.bm25_b
        if not indexed.length or not self.avg_length:
            return 0.0
        norm = indexed.length / self.avg_length
        total = 0.0
        for term in query_terms:
            tf = indexed.term_freq.get(term, 0)
            if not tf:
                continue
            total += self.idf(term) * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * norm))
        return total

    def search(self, question: str, top_k: int = 0) -> List[Retrieved]:
        """Top-k passages for the question, highest score first.

        Ties break on passage key so results are deterministic.
        """
        limit = top_k or self.settings.top_k
        terms = self.query_terms(question)
        if not terms or not self.passages:
            return []
        hits = [Retrieved(indexed=p, score=self.score(p, terms)) for p in self.passages]
        hits = [h for h in hits if h.score > 0.0]
        hits.sort(key=lambda h: (-h.score, h.passage.key))
        return hits[:limit]
