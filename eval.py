#!/usr/bin/env python3
"""Evaluation harness.

    python eval.py
    python eval.py --answerer llm          # needs ANTHROPIC_API_KEY, falls back if absent
    python eval.py --cases other_cases.json --kb other_kb.json

Reads question/expected-decision pairs from ``eval_questions.json`` and reports
decision accuracy, the answer/abstain breakdown, and citation presence rate.
Exit code 0 when every case matches, 1 otherwise.
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from support_answer.config import ANSWERER_MODES, DEFAULT_KB_PATH, Settings
from support_answer.kb import KbError
from support_answer.pipeline import SupportAnswerService

DEFAULT_CASES = "eval_questions.json"


def load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cases = payload["cases"] if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path!r} contains no cases")
    return cases


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.py", description=__doc__)
    parser.add_argument("--kb", default=DEFAULT_KB_PATH)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--answerer", default="rule", choices=list(ANSWERER_MODES))
    parser.add_argument("--verbose", action="store_true", help="print each answer in full")
    args = parser.parse_args(argv)

    try:
        service = SupportAnswerService.from_path(args.kb, Settings(answerer=args.answerer))
    except KbError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cases = load_cases(args.cases)

    rows = []
    correct = 0
    cited = 0
    answered = 0
    per_class = {"answer": [0, 0], "abstain": [0, 0]}  # [correct, total]
    citation_hits = 0
    citation_checked = 0

    for case in cases:
        question = case["question"]
        expected = case["expected"]
        response = service.ask(question)
        got = response["decision"]
        ok = got == expected
        correct += ok
        per_class.setdefault(expected, [0, 0])
        per_class[expected][1] += 1
        per_class[expected][0] += ok

        citation_ids = [c["id"] for c in response["citations"]]
        if got == "answer":
            answered += 1
            cited += bool(citation_ids)
        if ok and got == "answer" and case.get("expect_citations"):
            citation_checked += 1
            citation_hits += bool(set(case["expect_citations"]) & set(citation_ids))

        rows.append(
            {
                "question": question,
                "expected": expected,
                "got": got,
                "ok": ok,
                "score": response["debug"]["support_score"],
                "citations": citation_ids,
                "reason": response["debug"]["decision_reason"],
                "answer": response["answer"],
                "answerer": response["debug"].get("answerer_used"),
            }
        )

    print(f"knowledge base : {args.kb}")
    print(f"cases          : {args.cases} ({len(cases)} cases)")
    print(f"answerer       : {args.answerer}\n")
    print(f"{'':3}{'expected':9} {'got':9} {'score':>6}  {'citations':22} question")
    print("-" * 100)
    for row in rows:
        mark = "ok " if row["ok"] else "XX "
        cites = ",".join(row["citations"]) or "-"
        print(
            f"{mark}{row['expected']:9} {row['got']:9} {row['score']:>6.3f}  "
            f"{cites:22} {row['question']}"
        )
        if not row["ok"]:
            print(f"{'':3}-> {row['reason']}")
        if args.verbose:
            print(f"{'':3}-> {row['answer']}")

    total = len(rows)
    accuracy = correct / total if total else 0.0
    citation_rate = (cited / answered) if answered else 1.0
    print("\nmetrics")
    print(f"  decision accuracy       : {correct}/{total} = {accuracy:.0%}")
    for label, (hit, count) in per_class.items():
        if count:
            print(f"    {label:<8} accuracy    : {hit}/{count} = {hit / count:.0%}")
    print(f"  citation presence rate  : {cited}/{answered} = {citation_rate:.0%}"
          " (answered cases with >=1 citation)")
    if citation_checked:
        print(
            f"  expected-citation hit   : {citation_hits}/{citation_checked} = "
            f"{citation_hits / citation_checked:.0%}"
        )
    failures = total - correct
    print(f"\n{'PASS' if not failures else 'FAIL'}: {failures} mismatch(es)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
