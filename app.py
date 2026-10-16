#!/usr/bin/env python3
"""CLI entry point.

    python app.py "How do I reset my password?"
    python app.py --kb other_kb.json "How do I enable 2FA?"
    echo '{"question": "How long do withdrawal reviews take?"}' | python app.py --stdin

Prints the response contract as JSON on stdout. Exit codes: 0 success,
2 usage or knowledge-base error.
"""

import argparse
import json
import sys
from typing import List, Optional

from support_answer.config import (
    ANSWERER_MODES,
    DEFAULT_KB_PATH,
    LLM_PROVIDERS,
    Settings,
    load_env_file,
)
from support_answer.kb import KbError
from support_answer.pipeline import SupportAnswerService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Answer a support question from a local knowledge base, "
        "with citations, or abstain when the evidence is insufficient.",
    )
    parser.add_argument("question", nargs="?", help="the user question")
    parser.add_argument(
        "--kb", default=DEFAULT_KB_PATH, help=f"path to kb.json (default: {DEFAULT_KB_PATH})"
    )
    parser.add_argument(
        "--answerer",
        default="rule",
        choices=list(ANSWERER_MODES),
        help="answer composition layer (default: rule, no API key needed)",
    )
    parser.add_argument(
        "--llm-provider",
        default="auto",
        choices=list(LLM_PROVIDERS),
        help="LLM provider for --answerer llm (default: auto, whichever key is set)",
    )
    parser.add_argument(
        "--llm-model", default=None, help="override the provider's default model"
    )
    parser.add_argument("--top-k", type=int, default=None, help="passages to retrieve")
    parser.add_argument(
        "--min-coverage", type=float, default=None, help="override gate G2 threshold"
    )
    parser.add_argument(
        "--min-score", type=float, default=None, help="override gate G1 threshold"
    )
    parser.add_argument(
        "--stdin", action="store_true", help='read {"question": "..."} from stdin'
    )
    parser.add_argument(
        "--compact", action="store_true", help="single-line JSON output"
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings(answerer=args.answerer, llm_provider=args.llm_provider)
    overrides = {}
    if args.llm_model:
        overrides["llm_model"] = args.llm_model
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.min_coverage is not None:
        overrides["min_coverage"] = args.min_coverage
    if args.min_score is not None:
        overrides["min_norm_score"] = args.min_score
    return settings.with_(**overrides) if overrides else settings


def read_question(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.stdin:
        raw = sys.stdin.read().strip()
        if not raw:
            parser.error("no input on stdin")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            parser.error(f"stdin is not valid JSON: {exc}")
        if not isinstance(payload, dict) or not isinstance(payload.get("question"), str):
            parser.error('stdin JSON must be an object of the form {"question": "..."}')
        question = payload["question"]
    else:
        question = args.question or ""
    if not question.strip():
        parser.error('a non-empty question is required, e.g. app.py "How do I reset my password?"')
    return question.strip()


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    question = read_question(args, parser)
    load_env_file()  # optional local .env, for the LLM layer only

    try:
        service = SupportAnswerService.from_path(args.kb, settings_from_args(args))
    except KbError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    response = service.ask(question)
    if args.compact:
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
