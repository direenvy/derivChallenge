# SPECS — Grounded AI Support Answer Service

Status: written before implementation; updated once during build where the
implementation forced a change (noted inline as **[revised]**).

## 1. Purpose

A small, runnable service that answers user support questions **only** from a
local FAQ-style knowledge base, returns citations to the passages it used, and
**abstains** when the retrieved evidence does not support an answer.

Non-goals: chat history, multi-turn dialogue, authentication, a web UI,
vector databases, model fine-tuning.

## 2. Functional requirements

| ID | Requirement |
|----|-------------|
| FR-1 | Load `kb.json` from disk at runtime. No KB content embedded in source. |
| FR-2 | Work with any file matching the KB schema (replacement KB, different domain, different size). |
| FR-3 | Retrieve candidate passages for a question before any answer is composed. |
| FR-4 | Compose the answer strictly from retrieved passages. |
| FR-5 | Return >= 1 citation whenever `decision == "answer"`. |
| FR-6 | Abstain with an explicit fallback message when support is insufficient. |
| FR-7 | Expose the flow through a CLI **and** an HTTP `POST /ask` endpoint. |
| FR-8 | Emit the fixed JSON response contract (section 5). |
| FR-9 | Run with zero third-party dependencies and zero secrets. |
| FR-10 | Offer an optional LLM answer-composition mode that degrades to the non-LLM path automatically. |

## 3. Input contract

`kb.json` is a JSON array of documents:

```json
[{ "id": "doc_1", "title": "Password reset", "text": "Users can reset ..." }]
```

Rules:

- `id`, `title`, `text` are required and must be strings; `id` must be unique.
- Unknown extra fields are ignored (forward compatible).
- A non-array root, invalid JSON, a missing file, or a document missing a
  required field is a **fatal load error** with a clear message and non-zero
  exit code (FR-1 must fail loudly, not silently answer from nothing).
- An empty array loads successfully; every question then abstains.

## 4. Pipeline

```
LOAD_KB -> RETRIEVE -> CHECK_SUPPORT -> ANSWER | ABSTAIN
```

Stage order is enforced structurally, not by convention:

- The answer composers receive only an `Evidence` object (retrieved passages +
  selected sentences). They are not given the KB, the index, or the raw
  question-to-document mapping, so an answer cannot be produced before
  retrieval has run.
- `CHECK_SUPPORT` runs on the retrieval output and returns a decision; the
  `ANSWER` branch is unreachable when the decision is `abstain`.

## 5. Output contract

```json
{
  "question": "string",
  "decision": "answer | abstain",
  "answer": "string",
  "citations": [{ "id": "string", "title": "string", "snippet": "string" }],
  "debug": { "retrieved_ids": ["string"], "support_score": 0.0 }
}
```

- Top-level keys are exactly these five, always present, in this order.
- `decision` is the literal string `answer` or `abstain`.
- `citations` is `[]` when abstaining, and has >= 1 entry when answering.
- `debug.support_score` is a float in `[0, 1]` in both branches.
- `debug` carries additional diagnostic keys (gate values, answerer used).
  Extra keys inside `debug` are allowed by this spec; top-level keys are not.

## 6. Support rule (summary)

Answer only if **all** gates pass. Any failure -> abstain.

Gates G1-G5 were specified up front; **G6 was added during the build**, when
G2 alone could not separate "can support change my email address" (must
abstain) from "I forgot my password, how do I get into my account" (should
answer) without a knife-edge threshold.

| Gate | Check |
|------|-------|
| G1 Retrieval strength | normalised BM25 top score >= `min_norm_score` |
| G2 Evidence coverage | IDF-weighted coverage of question terms by a single retrieved passage >= `min_coverage` |
| G3 Conduct limitation | the best evidence sentence must not be a policy sentence that forbids giving this kind of answer |
| G4 Unsupported quantity | every standalone number in the question must appear in the retrieved evidence |
| G5 Extractive evidence | at least one evidence sentence selected (guarantees >= 1 citation) |
| G6 Phrase consistency **[revised]** | any adjacent term-pair the question names that exists in the KB must also exist in the passage the answer comes from |

Rationale and tuning live in `DESIGN.md` section 4; thresholds live in
`support_answer/config.py`.

## 7. Interfaces

CLI:

```
python app.py "How do I reset my password?"
python app.py --kb other_kb.json --answerer llm "How do I enable 2FA?"
echo '{"question":"..."}' | python app.py --stdin
```

HTTP:

```
python server.py --port 8000
POST /ask   {"question": "..."}   -> 200 response contract
GET  /healthz                     -> {"status":"ok","docs":6}
```

Errors: malformed body or missing/blank `question` -> HTTP 400 with
`{"error": "..."}`. A blank question on the CLI is a usage error.

## 8. Answering modes

| Mode | Behaviour |
|------|-----------|
| `rule` (default) | Extractive composition: the answer is verbatim sentences selected from retrieved passages. Hallucination-free by construction. |
| `llm` | A hosted model composes from the retrieved context only, then a grounding check verifies the output against that context. Failure, missing SDK, or missing credentials -> automatic fallback to `rule`. |
| `auto` | Use `llm` if usable, else `rule`, silently. |

Providers **[revised]**: `gemini` (`GEMINI_API_KEY`, no package required) and
`anthropic` (`ANTHROPIC_API_KEY`, needs the SDK), selected by `--llm-provider`
or auto-detected from the environment with Gemini preferred. Gemini was added
after the first build so the LLM path is runnable on a free key.

The LLM path never widens retrieval and never sees documents outside the
retrieved set.

## 9. Validation

`python validate.py` (stdlib only) must assert:

1. KB loads from disk successfully.
2. An answerable question returns `decision: answer` with >= 1 citation.
3. An unsupported question returns `decision: abstain` with `citations == []`.
4. Response JSON matches the contract in section 5.
5. Answers follow a **replacement** KB written at runtime to a temp directory
   (proves FR-1/FR-2: no hardcoded answers).

`python -m pytest` runs the same assertions plus edge cases (empty KB, blank
question, malformed KB, long-document chunking, citation-grounding).

`python eval.py` scores 8 fixed questions: decision accuracy, citation
presence rate, and per-class breakdown.

## 10. Acceptance criteria

- Clean checkout, Python >= 3.9, no install step, no network, no API key: all
  commands in section 7 and section 9 succeed.
- Swapping `kb.json` for an unrelated domain changes the answers accordingly
  and does not require code changes.
- The three sample adversarial questions abstain; the sample answerable
  questions answer with correct citations.
