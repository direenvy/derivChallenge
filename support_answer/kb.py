"""LOAD_KB stage: read and validate ``kb.json`` from disk, then split it into
retrievable passages.

The knowledge base is *always* read from the filesystem -- there is no embedded
fallback copy anywhere in this package. A replacement file using the same
schema is picked up with no code change.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, List, Sequence

from .config import Settings
from .text import split_sentences

REQUIRED_FIELDS = ("id", "title", "text")


class KbError(Exception):
    """Raised when kb.json is missing, unreadable, or does not match schema."""


@dataclass(frozen=True)
class Document:
    """One knowledge-base article, exactly as stored on disk."""

    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Passage:
    """A retrieval unit. For short articles this is the whole document.

    ``doc_id``/``title`` always refer to the parent document, so citations stay
    document-level even when a long article is chunked.
    """

    doc_id: str
    title: str
    text: str
    chunk_index: int = 0

    @property
    def key(self) -> str:
        return self.doc_id if self.chunk_index == 0 else f"{self.doc_id}#{self.chunk_index}"


def load_kb(path: str) -> List[Document]:
    """Load and validate the knowledge base at ``path``.

    Raises :class:`KbError` with an actionable message for every failure mode:
    missing file, unreadable file, invalid JSON, non-array root, non-object
    entry, missing/blank/non-string required field, duplicate id.
    """
    if not os.path.isfile(path):
        raise KbError(
            f"knowledge base not found: {path!r} "
            "(expected a JSON array of {id, title, text} objects)"
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw: Any = json.load(fh)
    except json.JSONDecodeError as exc:
        raise KbError(f"{path!r} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise KbError(f"cannot read {path!r}: {exc}") from exc

    if not isinstance(raw, list):
        raise KbError(
            f"{path!r} must contain a JSON array at the top level, "
            f"got {type(raw).__name__}"
        )

    docs: List[Document] = []
    seen_ids = set()
    for position, entry in enumerate(raw):
        where = f"{path!r} entry #{position}"
        if not isinstance(entry, dict):
            raise KbError(f"{where} must be an object, got {type(entry).__name__}")
        for field in REQUIRED_FIELDS:
            if field not in entry:
                raise KbError(f"{where} is missing required field {field!r}")
            if not isinstance(entry[field], str):
                raise KbError(
                    f"{where} field {field!r} must be a string, "
                    f"got {type(entry[field]).__name__}"
                )
        doc_id = entry["id"].strip()
        if not doc_id:
            raise KbError(f"{where} has a blank 'id'")
        if doc_id in seen_ids:
            raise KbError(f"{where} repeats id {doc_id!r}; ids must be unique")
        seen_ids.add(doc_id)
        docs.append(
            Document(id=doc_id, title=entry["title"].strip(), text=entry["text"].strip())
        )
    return docs


def build_passages(docs: Sequence[Document], settings: Settings) -> List[Passage]:
    """Turn documents into retrieval passages.

    Short articles (the common case for FAQ content) stay whole: they are
    already single-topic chunks, and keeping them intact keeps citations
    document-level. Articles longer than ``chunk_threshold_chars`` are split
    into overlapping sentence windows so one relevant sentence is not diluted
    by a long document's length normalisation.
    """
    passages: List[Passage] = []
    for doc in docs:
        if len(doc.text) <= settings.chunk_threshold_chars:
            passages.append(Passage(doc_id=doc.id, title=doc.title, text=doc.text))
            continue
        for index, window in enumerate(_windows(doc.text, settings)):
            passages.append(
                Passage(
                    doc_id=doc.id,
                    title=doc.title,
                    text=window,
                    chunk_index=index,
                )
            )
    return passages


def _windows(text: str, settings: Settings) -> List[str]:
    """Overlapping sentence windows over a long document."""
    sentences = split_sentences(text)
    size = max(1, settings.chunk_sentences)
    overlap = min(max(0, settings.chunk_overlap_sentences), size - 1)
    stride = max(1, size - overlap)
    if len(sentences) <= size:
        return [" ".join(sentences)]
    windows = []
    for start in range(0, len(sentences), stride):
        window = sentences[start : start + size]
        if not window:
            break
        windows.append(" ".join(window))
        if start + size >= len(sentences):
            break
    return windows
