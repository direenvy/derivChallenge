"""Text utilities: tokenisation, light stemming, stopwords, sentence splitting,
and KB-derived acronym expansion.

Deliberately dependency-free and deterministic. Everything here is generic --
nothing is specialised to the sample knowledge base.
"""

import re
from typing import Dict, FrozenSet, List, Sequence, Set

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

VOWELS = frozenset("aeiou")

#: Common English function words. Removed before indexing and before any
#: overlap/coverage arithmetic so that scores reflect content words only.
STOPWORDS: Set[str] = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "but", "by",
    "can", "cannot", "could", "did", "do", "does", "doing", "done", "for",
    "from", "get", "got", "had", "has", "have", "he", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "may", "me", "might", "mine", "more", "most", "must", "my", "need", "no",
    "nor", "not", "of", "on", "one", "only", "or", "other", "our", "ours",
    "out", "over", "own", "please", "same", "shall", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "us", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours",
}

#: Leading number words are also emitted as digits when building acronyms, so a
#: title like "Two-factor authentication" is reachable by the query "2FA".
NUMBER_WORDS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric runs. ``2fa`` stays a single token."""
    return _TOKEN_RE.findall(text.lower())


def _ends_cvc(word: str) -> bool:
    """True when the word ends consonant-vowel-consonant (last not w/x/y)."""
    if len(word) < 3:
        return False
    a, b, c = word[-3], word[-2], word[-1]
    return (
        a not in VOWELS
        and b in VOWELS
        and c not in VOWELS
        and c not in "wxy"
    )


def _measure(word: str) -> int:
    """Porter measure: the number of vowel-group/consonant-group alternations.

    ``clos`` -> 1, ``deliver`` -> 3. Used to keep the silent-e restoration to
    short stems, so ``closed`` -> ``close`` but ``delivered`` -> ``deliver``.
    """
    groups = []
    for ch in word:
        kind = "v" if ch in VOWELS else "c"
        if not groups or groups[-1] != kind:
            groups.append(kind)
    return sum(1 for i in range(1, len(groups)) if groups[i] == "c" and groups[i - 1] == "v")


def _restore_e(stem_: str) -> str:
    """Re-attach the silent ``e`` dropped by ``-ed``/``-ing`` stripping.

    ``clos`` -> ``close``, ``stat`` -> ``state``, ``enabl`` -> ``enable``.
    This is what makes *closed* and *close* share a stem, which matters for
    coverage scoring.
    """
    if stem_.endswith(("at", "bl", "iz")):
        return stem_ + "e"
    if _measure(stem_) == 1 and _ends_cvc(stem_):
        return stem_ + "e"
    return stem_


def _depluralise(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _deconjugate(word: str) -> str:
    if word.endswith("ing") and len(word) - 3 >= 3:
        return _restore_e(word[:-3])
    if word.endswith("ed") and len(word) - 2 >= 3:
        return _restore_e(word[:-2])
    return word


def stem(word: str) -> str:
    """Conservative suffix stemmer (see DESIGN.md section 3).

    Applied in Porter-like order: plurals, then verb endings, then adverbs.
    Tokens containing digits are left alone so ``2fa`` and ``24`` survive.
    """
    if len(word) <= 3 or any(ch.isdigit() for ch in word):
        return word
    w = _depluralise(word)
    w = _deconjugate(w)
    if w.endswith("ly") and len(w) - 2 >= 3:
        w = w[:-2]
    # Drop a trailing silent 'e' on longer stems, so "damage" and "damaged"
    # (which loses it) agree. Short stems keep it: "close" stays "close".
    if w.endswith("e") and _measure(w[:-1]) > 1:
        w = w[:-1]
    return w


STOPWORD_STEMS: Set[str] = {stem(w) for w in STOPWORDS}


def content_terms(text: str) -> List[str]:
    """Stemmed content terms, in order, duplicates kept (term frequency)."""
    out = []
    for token in tokenize(text):
        if token in STOPWORDS:
            continue
        s = stem(token)
        if len(s) < 2 or s in STOPWORD_STEMS:
            continue
        out.append(s)
    return out


def split_sentences(text: str) -> List[str]:
    """Split on sentence-final punctuation followed by a capital/quote/digit."""
    chunks = _SENTENCE_SPLIT_RE.split(text.strip())
    return [c.strip() for c in chunks if c.strip()]


def acronym_variants(title: str) -> List[str]:
    """Acronyms derived from a document title, e.g. ``2fa``/``tfa``.

    Generic: it uses the initials of the title's content words, plus a digit
    form when the first word is a number word. No hardcoded synonym list, so
    it works on whatever titles a replacement knowledge base provides.
    """
    words = [w for w in tokenize(title) if w not in STOPWORDS]
    if not 2 <= len(words) <= 5:
        return []
    initials = "".join(w[0] for w in words)
    if len(initials) < 2:
        return []
    variants = {initials}
    first = words[0]
    if first in NUMBER_WORDS:
        variants.add(NUMBER_WORDS[first] + initials[1:])
    return sorted(variants)


def numeric_tokens(text: str) -> List[str]:
    """Standalone numbers in the text. ``2fa`` is not a number; ``2`` is."""
    return [t for t in tokenize(text) if t.isdigit()]


def unique(seq: Sequence[str]) -> List[str]:
    """Order-preserving de-duplication."""
    seen: Set[str] = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def truncate(text: str, limit: int) -> str:
    """Trim to ``limit`` characters on a word boundary, adding an ellipsis."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rstrip()
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


def adjacent_pairs(terms: Sequence[str]) -> Set[FrozenSet[str]]:
    """Unordered pairs of neighbouring content terms.

    Used by gate G6 as a cheap phrase signal: "email address" and "address
    email" both map to the same pair, so word order does not matter, but a
    phrase the user names must appear in the passage the answer comes from.
    """
    return {
        frozenset((a, b))
        for a, b in zip(terms, terms[1:])
        if a != b
    }
