"""Tests for the grounded support-answer service.

    python -m pytest -q

Covers the four cases the brief requires (answerable -> answer, unsupported ->
abstain, citations present when answering, the KB is read from disk) plus the
edge cases from SPECS.md section 3 and DESIGN.md section 7.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support_answer.config import Settings  # noqa: E402
from support_answer.kb import KbError, build_passages, load_kb  # noqa: E402
from support_answer.pipeline import SupportAnswerService  # noqa: E402
from support_answer.schema import validate_response  # noqa: E402
from support_answer.support import is_conduct_limitation  # noqa: E402
from support_answer.text import acronym_variants, content_terms, stem  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(REPO_ROOT, "kb.json")

ANSWERABLE = [
    "How do I reset my password?",
    "How do I enable two-factor authentication?",
    "How do I enable 2FA?",
    "How long does a withdrawal review take?",
    "What do I need to do before closing my account?",
    "Can I update my residential address?",
    "What happens if I lose access to my authenticator app?",
]

UNSUPPORTED = [
    "Can you guarantee my withdrawal will finish in 2 hours?",
    "What tax advice do you provide for closed accounts?",
    "Can support change my email address for me?",
    "What is the capital of France?",
    "zzz qqq xyzzy",
]


@pytest.fixture(scope="module")
def service():
    return SupportAnswerService.from_path(KB_PATH, Settings())


# --------------------------------------------------------------------- KB

def test_kb_is_loaded_from_disk_not_embedded(service):
    """The service must read kb.json, so a different file gives different answers."""
    assert service.kb_path == KB_PATH
    assert os.path.isfile(KB_PATH)
    source = open(os.path.join(REPO_ROOT, "support_answer", "pipeline.py"), encoding="utf-8").read()
    assert "Users can reset their password" not in source


def test_kb_load_rejects_bad_input(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(KbError, match="not found"):
        load_kb(str(missing))

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(KbError, match="not valid JSON"):
        load_kb(str(bad_json))

    not_array = tmp_path / "obj.json"
    not_array.write_text('{"id": "a"}', encoding="utf-8")
    with pytest.raises(KbError, match="array"):
        load_kb(str(not_array))

    missing_field = tmp_path / "partial.json"
    missing_field.write_text('[{"id": "a", "title": "t"}]', encoding="utf-8")
    with pytest.raises(KbError, match="missing required field 'text'"):
        load_kb(str(missing_field))

    duplicate = tmp_path / "dupe.json"
    duplicate.write_text(
        '[{"id":"a","title":"t","text":"x"},{"id":"a","title":"t2","text":"y"}]',
        encoding="utf-8",
    )
    with pytest.raises(KbError, match="unique"):
        load_kb(str(duplicate))


def test_empty_kb_loads_and_always_abstains(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text("[]", encoding="utf-8")
    svc = SupportAnswerService.from_path(str(path), Settings())
    response = svc.ask("How do I reset my password?")
    assert response["decision"] == "abstain"
    assert response["citations"] == []
    assert not validate_response(response)


def test_replacement_kb_drives_the_answers(tmp_path):
    """Same schema, unrelated domain: answers must come from the new file."""
    path = tmp_path / "kb.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "p1",
                    "title": "Pizza delivery times",
                    "text": (
                        "Pizza orders are delivered within 40 minutes inside the city ring. "
                        "Deliveries outside the ring can take up to 70 minutes."
                    ),
                },
                {
                    "id": "p2",
                    "title": "Refund policy",
                    "text": "A cold pizza can be refunded if it is reported on the same day.",
                },
            ]
        ),
        encoding="utf-8",
    )
    svc = SupportAnswerService.from_path(str(path), Settings())

    response = svc.ask("How long does pizza delivery take inside the city ring?")
    assert response["decision"] == "answer"
    assert [c["id"] for c in response["citations"]] == ["p1"]
    assert "40 minutes" in response["answer"]

    # A question the previous knowledge base answered must now abstain.
    assert svc.ask("How do I reset my password?")["decision"] == "abstain"


def test_long_documents_are_chunked_but_cited_by_parent_id(tmp_path):
    body = " ".join(f"Rule {i} states that step {i} must be completed." for i in range(40))
    path = tmp_path / "kb.json"
    path.write_text(
        json.dumps([{"id": "long_1", "title": "Operating rules", "text": body}]),
        encoding="utf-8",
    )
    svc = SupportAnswerService.from_path(str(path), Settings())
    assert len(svc.passages) > 1, "a long document should be chunked"
    assert all(p.doc_id == "long_1" for p in svc.passages)

    response = svc.ask("What does rule 17 state?")
    assert response["decision"] == "answer"
    assert [c["id"] for c in response["citations"]] == ["long_1"]
    assert "#" not in response["citations"][0]["id"]


def test_short_documents_are_not_chunked(service):
    assert len(service.passages) == len(service.documents)


# ------------------------------------------------------------ answer path

@pytest.mark.parametrize("question", ANSWERABLE)
def test_answerable_questions_answer_with_citations(service, question):
    response = service.ask(question)
    assert response["decision"] == "answer", response["debug"]["decision_reason"]
    assert len(response["citations"]) >= 1
    for citation in response["citations"]:
        assert citation["id"] and citation["title"] and citation["snippet"]
    assert not validate_response(response)


def test_answer_is_grounded_in_the_cited_documents(service):
    response = service.ask("How long does a withdrawal review take?")
    kb_by_id = {d.id: d for d in service.documents}
    corpus = " ".join(
        " ".join(f"{kb_by_id[c['id']].title} {kb_by_id[c['id']].text}".split())
        for c in response["citations"]
    )
    answer_terms = set(content_terms(response["answer"]))
    assert answer_terms <= set(content_terms(corpus)), "answer introduced unsupported terms"


def test_citations_reference_retrieved_documents_only(service):
    response = service.ask("How do I enable 2FA?")
    retrieved = set(response["debug"]["retrieved_ids"])
    assert {c["id"] for c in response["citations"]} <= retrieved


# ----------------------------------------------------------- abstain path

@pytest.mark.parametrize("question", UNSUPPORTED)
def test_unsupported_questions_abstain(service, question):
    response = service.ask(question)
    assert response["decision"] == "abstain", response["debug"]
    assert response["citations"] == []
    assert "support" in response["answer"].lower()
    assert not validate_response(response)


def test_abstain_reason_names_a_failed_gate(service):
    response = service.ask("Can you guarantee my withdrawal will finish in 2 hours?")
    assert response["debug"]["decision_reason"].startswith("failed ")
    failed = [name for name, gate in response["debug"]["gates"].items() if not gate["pass"]]
    assert "G4_no_unsupported_quantity" in failed


def test_conduct_limitation_detector_distinguishes_verb_classes():
    assert is_conduct_limitation(
        "Support answers should not provide legal advice, tax advice, or guarantees "
        "about processing times beyond what is stated in the knowledge base."
    )
    # Product behaviour, not a refusal to advise -> still answerable.
    assert not is_conduct_limitation(
        "Support agents cannot manually view or send existing passwords."
    )
    assert not is_conduct_limitation(
        "Users should not submit duplicate withdrawal requests during review."
    )


# -------------------------------------------------------------- contract

def test_blank_question_is_rejected(service):
    for bad in ("", "   ", "\n"):
        with pytest.raises(ValueError):
            service.ask(bad)


def test_response_has_exactly_the_contract_keys(service):
    response = service.ask("How do I reset my password?")
    assert list(response) == ["question", "decision", "answer", "citations", "debug"]
    assert {"retrieved_ids", "support_score"} <= set(response["debug"])
    assert 0.0 <= response["debug"]["support_score"] <= 1.0


def test_validate_response_rejects_a_broken_payload():
    assert validate_response({"question": "q"})
    assert validate_response(
        {
            "question": "q",
            "decision": "answer",
            "answer": "a",
            "citations": [],  # answering without citations is invalid
            "debug": {"retrieved_ids": [], "support_score": 0.5},
        }
    )


# ------------------------------------------------------- answering layers

def test_rule_mode_needs_no_network_or_key(service, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = service.ask("How do I reset my password?")
    assert response["debug"]["answerer_used"] == "rule"


def test_llm_mode_falls_back_to_rule_without_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    svc = SupportAnswerService.from_path(KB_PATH, Settings(answerer="llm"))
    response = svc.ask("How do I reset my password?")
    assert response["decision"] == "answer"
    assert response["debug"]["answerer_used"] == "rule"
    assert any("llm_fallback" in note for note in response["debug"]["answerer_notes"])


# ------------------------------------------------------------------ text

def test_stemmer_and_acronyms_are_generic():
    assert stem("withdrawals") == stem("withdrawal")
    assert stem("closed") == stem("close")
    assert stem("enabled") == stem("enable")
    assert stem("2fa") == "2fa"
    assert "2fa" in acronym_variants("Two-factor authentication")
    assert acronym_variants("Password") == []


def test_build_passages_respects_threshold():
    from support_answer.kb import Document

    doc = Document(id="d", title="t", text="Short sentence. Another one.")
    assert len(build_passages([doc], Settings())) == 1
