"""The response contract: construction in one place, validation in another.

``validate_response`` is used by the tests, by ``validate.py`` and by the HTTP
layer, so the shape in SPECS.md section 5 is checked rather than assumed.
"""

from typing import Any, Dict, List, Mapping, Sequence

RESPONSE_KEYS = ("question", "decision", "answer", "citations", "debug")
CITATION_KEYS = ("id", "title", "snippet")
DEBUG_REQUIRED_KEYS = ("retrieved_ids", "support_score")
DECISIONS = ("answer", "abstain")


def build_response(
    question: str,
    decision: str,
    answer: str,
    citations: Sequence[Mapping[str, str]],
    debug: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble the response dict with exactly the contract's top-level keys."""
    return {
        "question": question,
        "decision": decision,
        "answer": answer,
        "citations": [
            {key: str(citation[key]) for key in CITATION_KEYS} for citation in citations
        ],
        "debug": dict(debug),
    }


def validate_response(payload: Any) -> List[str]:
    """Return a list of contract violations; empty means valid."""
    errors: List[str] = []
    if not isinstance(payload, dict):
        return [f"response must be an object, got {type(payload).__name__}"]

    missing = [key for key in RESPONSE_KEYS if key not in payload]
    if missing:
        errors.append(f"missing top-level key(s): {', '.join(missing)}")
    extra = [key for key in payload if key not in RESPONSE_KEYS]
    if extra:
        errors.append(f"unexpected top-level key(s): {', '.join(sorted(extra))}")

    if not isinstance(payload.get("question"), str):
        errors.append("'question' must be a string")
    if payload.get("decision") not in DECISIONS:
        errors.append(f"'decision' must be one of {DECISIONS}, got {payload.get('decision')!r}")
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("'answer' must be a non-empty string")

    citations = payload.get("citations")
    if not isinstance(citations, list):
        errors.append("'citations' must be a list")
    else:
        for index, citation in enumerate(citations):
            if not isinstance(citation, dict):
                errors.append(f"citation #{index} must be an object")
                continue
            for key in CITATION_KEYS:
                if key not in citation:
                    errors.append(f"citation #{index} is missing {key!r}")
                elif not isinstance(citation[key], str):
                    errors.append(f"citation #{index} field {key!r} must be a string")
        if payload.get("decision") == "answer" and not citations:
            errors.append("'citations' must contain at least one entry when answering")
        if payload.get("decision") == "abstain" and citations:
            errors.append("'citations' must be empty when abstaining")

    debug = payload.get("debug")
    if not isinstance(debug, dict):
        errors.append("'debug' must be an object")
    else:
        for key in DEBUG_REQUIRED_KEYS:
            if key not in debug:
                errors.append(f"'debug' is missing {key!r}")
        retrieved_ids = debug.get("retrieved_ids")
        if not isinstance(retrieved_ids, list) or not all(
            isinstance(item, str) for item in retrieved_ids
        ):
            errors.append("'debug.retrieved_ids' must be a list of strings")
        score = debug.get("support_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            errors.append("'debug.support_score' must be a number")
        elif not 0.0 <= float(score) <= 1.0:
            errors.append(f"'debug.support_score' must be within [0, 1], got {score}")
    return errors
