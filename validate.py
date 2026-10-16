#!/usr/bin/env python3
"""Validation script -- standard library only, no pytest, no network.

    python validate.py
    python validate.py --kb other_kb.json

Checks, in order:

1. the knowledge base loads from disk;
2. an answerable question returns ``decision: answer`` with >= 1 citation;
3. an unsupported question returns ``decision: abstain`` with no citations;
4. both responses satisfy the output contract;
5. answers and citations are grounded in the knowledge-base text;
6. the service answers from a **replacement** knowledge base written at
   runtime, proving the KB is read from disk rather than embedded.

Deliberately KB-agnostic: the answerable question is derived from the loaded
knowledge base (a document title is the strongest possible query for that
document), so this script still passes when ``kb.json`` is swapped out.

Exit code 0 if every check passes, 1 otherwise.
"""

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from support_answer.config import DEFAULT_KB_PATH, Settings
from support_answer.kb import KbError, load_kb
from support_answer.pipeline import SupportAnswerService
from support_answer.schema import validate_response
from support_answer.text import content_terms

UNSUPPORTED_QUESTION = (
    "What is your official legal and tax guidance on the resale value of a "
    "vintage trombone in Ulaanbaatar?"
)

REPLACEMENT_KB = [
    {
        "id": "zz_1",
        "title": "Greenhouse watering schedule",
        "text": (
            "Tomato beds in the greenhouse are watered every second morning. "
            "Seedlings younger than three weeks are watered by hand only. "
            "Volunteers should never water the beds after sunset."
        ),
    },
    {
        "id": "zz_2",
        "title": "Compost bin rotation",
        "text": (
            "The compost bins are rotated on the first Monday of each month. "
            "Finished compost is screened before it is added to the raised beds."
        ),
    },
]


class Checks:
    """Tiny assertion collector so one run reports every failure."""

    def __init__(self) -> None:
        self.results: List[Tuple[bool, str, str]] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        self.results.append((bool(ok), label, detail))
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {label}"
        if detail:
            line += f"\n         {detail}"
        print(line)
        return bool(ok)

    @property
    def failed(self) -> int:
        return sum(1 for ok, _, _ in self.results if not ok)


def first_answerable(service: SupportAnswerService) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Find a question this knowledge base can answer, using document titles."""
    for document in service.documents:
        for question in (document.title, f"{document.title}?"):
            if not question.strip():
                continue
            response = service.ask(question)
            if response["decision"] == "answer":
                return question, response
    return None


def grounded_in_kb(text: str, kb_text: str) -> float:
    """Fraction of the text's content terms that occur in the knowledge base."""
    terms = set(content_terms(text))
    if not terms:
        return 0.0
    return len(terms & set(content_terms(kb_text))) / len(terms)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="validate.py", description=__doc__)
    parser.add_argument("--kb", default=DEFAULT_KB_PATH, help="path to kb.json")
    args = parser.parse_args(argv)

    checks = Checks()
    print(f"validating with knowledge base {args.kb!r}\n")

    # 1. knowledge base loads from disk ------------------------------------
    try:
        documents = load_kb(args.kb)
    except KbError as exc:
        checks.check(False, "knowledge base loads from disk", str(exc))
        print(f"\n{checks.failed} check(s) failed")
        return 1
    checks.check(
        bool(documents),
        "knowledge base loads from disk",
        f"{len(documents)} document(s) from {os.path.abspath(args.kb)}",
    )

    service = SupportAnswerService.from_path(args.kb, Settings())
    kb_text = " ".join(f"{d.title} {d.text}" for d in documents)

    # 2. an answerable question answers, with citations --------------------
    found = first_answerable(service)
    if not found:
        checks.check(False, "an answerable question returns decision=answer", "no document title produced an answer")
        print(f"\n{checks.failed} check(s) failed")
        return 1
    question, answered = found
    checks.check(
        answered["decision"] == "answer",
        "an answerable question returns decision=answer",
        f"q={question!r} score={answered['debug']['support_score']}",
    )
    checks.check(
        len(answered["citations"]) >= 1,
        "answered question returns at least one citation",
        f"citations={[c['id'] for c in answered['citations']]}",
    )
    checks.check(
        all(c["id"] in {d.id for d in documents} for c in answered["citations"]),
        "citation ids exist in the knowledge base",
    )

    # 3. an unsupported question abstains ---------------------------------
    abstained = service.ask(UNSUPPORTED_QUESTION)
    checks.check(
        abstained["decision"] == "abstain",
        "an unsupported question returns decision=abstain",
        f"reason={abstained['debug']['decision_reason']}",
    )
    checks.check(
        abstained["citations"] == [],
        "abstained question returns no citations",
    )
    checks.check(
        "support" in abstained["answer"].lower(),
        "abstain message points the user to human support",
    )

    # 4. output contract ---------------------------------------------------
    for label, payload in (("answered", answered), ("abstained", abstained)):
        errors = validate_response(payload)
        checks.check(
            not errors,
            f"{label} response matches the output contract",
            "; ".join(errors) if errors else "keys, types and ranges ok",
        )
    checks.check(
        json.loads(json.dumps(answered)) == answered,
        "response is JSON-serialisable",
    )

    # 5. grounding ---------------------------------------------------------
    answer_grounding = grounded_in_kb(answered["answer"], kb_text)
    checks.check(
        answer_grounding == 1.0,
        "answer text is grounded in knowledge-base text",
        f"content-term grounding = {answer_grounding:.2f}",
    )
    snippets_ok = all(
        c["snippet"].rstrip("…").strip()[:60] in " ".join(kb_text.split())
        for c in answered["citations"]
    )
    checks.check(snippets_ok, "citation snippets are verbatim knowledge-base text")

    # 6. replacement knowledge base ---------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "kb.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(REPLACEMENT_KB, fh)
        swapped = SupportAnswerService.from_path(path, Settings())
        response = swapped.ask("When are the greenhouse tomato beds watered?")
        checks.check(
            response["decision"] == "answer"
            and response["citations"]
            and response["citations"][0]["id"] == "zz_1"
            and "every second morning" in response["answer"],
            "answers from a replacement kb.json written at runtime",
            f"decision={response['decision']} citations="
            f"{[c['id'] for c in response['citations']]}",
        )
        original_topic = swapped.ask(question)
        checks.check(
            original_topic["decision"] == "abstain",
            "replacement knowledge base no longer answers the original question",
            f"q={question!r} -> {original_topic['decision']}",
        )

    total = len(checks.results)
    if checks.failed:
        print(f"\n{checks.failed} of {total} check(s) FAILED")
        return 1
    print(f"\nall {total} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
