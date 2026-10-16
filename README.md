# Grounded AI Support Answer Service

Answers support questions **only** from a local FAQ knowledge base, returns
**citations** to the passages it used, and **abstains** when the retrieved
evidence does not support an answer.

```
LOAD_KB -> RETRIEVE -> CHECK_SUPPORT -> ANSWER | ABSTAIN
```

- **No dependencies.** Python 3.9+ standard library only. No install step, no
  network, no API key.
- **The knowledge base is read from disk** every run. Swap `kb.json` for any
  file with the same schema and the answers follow it; nothing is hardcoded.
- **Retrieval always runs before generation**, enforced structurally: the
  answer composers only ever receive retrieved evidence.

Design notes: [`SPECS.md`](SPECS.md) (requirements and contracts) and
[`DESIGN.md`](DESIGN.md) (architecture, retrieval choice, support rule,
chunking, tradeoffs).

## Setup

```bash
python --version        # 3.9 or newer
```

That is the whole setup. Optional extras:

```bash
pip install -r requirements-optional.txt   # pytest (tests) + anthropic (LLM mode)
```

`python validate.py` and `python eval.py` need neither.

## Run

```bash
python app.py "How do I reset my password?"
```

```bash
echo '{"question": "How long does a withdrawal review take?"}' | python app.py --stdin
```

HTTP, on the standard library:

```bash
python server.py --port 8000
```

```bash
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question": "How do I enable 2FA?"}'
```

Useful flags: `--kb PATH`, `--answerer rule|llm|auto`, `--top-k N`,
`--min-coverage F`, `--min-score F`, `--compact`. `GET /healthz` reports the
loaded document count.

## Validate

```bash
python validate.py        # 14 checks, standard library only, exit 0 on success
python -m pytest -q       # 29 tests (needs pytest)
python eval.py            # 8-question eval with metrics
```

`validate.py` checks that the knowledge base loads, that an answerable question
returns citations, that an unsupported question abstains, that the output JSON
matches the contract, that answers are verbatim knowledge-base text, and that a
**replacement `kb.json` written at runtime** drives the answers. It derives its
answerable question from whatever KB is loaded, so it still passes after the
knowledge base is swapped.

## Swapping the knowledge base

```bash
python app.py --kb /path/to/other_kb.json "a question about that content"
python validate.py --kb /path/to/other_kb.json
```

Schema (extra fields are ignored):

```json
[{ "id": "doc_1", "title": "Password reset", "text": "Users can reset ..." }]
```

A missing file, invalid JSON, a non-array root, a missing field or a duplicate
id is a fatal error with an explicit message and exit code 2 — the service
never silently answers from an empty index.

## Response contract

```json
{
  "question": "How do I reset my password?",
  "decision": "answer",
  "answer": "Users can reset their password from the login page by clicking 'Forgot password'. ...",
  "citations": [
    { "id": "doc_1", "title": "Password reset", "snippet": "Users can reset their password ..." }
  ],
  "debug": {
    "retrieved_ids": ["doc_1"],
    "support_score": 0.8674,
    "coverage": 1.0,
    "gates": { "G1_retrieval_strength": { "value": 0.7348, "threshold": 0.12, "pass": true }, "...": {} },
    "decision_reason": "all gates passed",
    "answerer_used": "rule",
    "stages": ["LOAD_KB", "RETRIEVE", "CHECK_SUPPORT", "ANSWER|ABSTAIN"]
  }
}
```

Top-level keys are always exactly these five. `citations` is `[]` when
abstaining and holds at least one entry when answering. `debug` carries extra
diagnostic keys (the gate values, the reason, the answerer used) so a decision
can be traced.

## The support rule

Retrieval is **BM25** (hand-written, k1=1.5, b=0.75) over the documents, with
the title indexed twice and acronyms derived from each title (so a question
about "2FA" finds a document titled "Two-factor authentication").

An answer is returned only when **all six gates pass**. Any failure abstains.
Each gate is reported under `debug.gates`, and `debug.decision_reason` names
the ones that failed.

| Gate | Rule | Default | Catches |
|------|------|---------|---------|
| **G1** retrieval strength | top BM25 score as a fraction of the best score this query could achieve (`top1 / Σ idf·(k1+1)`) | ≥ `0.12` | nothing relevant in the KB; off-topic and gibberish questions |
| **G2** evidence coverage | IDF-weighted share of the question's known terms covered by a **single** passage | ≥ `0.55` | questions only partly addressed by any one article |
| **G3** conduct limitation | the strongest evidence sentence must not combine a negation cue with an advice/provision cue | — | *"What tax advice do you provide…"* — the KB forbids giving it |
| **G4** unsupported quantity | every standalone number in the question must appear in the retrieved evidence | — | *"…finish in 2 hours?"* — answering would invent a figure |
| **G5** evidence sentence | at least one sentence overlaps the question | — | guarantees every answer has ≥1 citation |
| **G6** phrase consistency | any adjacent term-pair the question names that exists in the KB must exist in the passage answered from | — | *"change my email address"* when the best passage is about *residential* address |

Why a conjunction and not one threshold: *"Can you guarantee my withdrawal
will finish in 2 hours?"* retrieves the withdrawals article with a **high**
lexical score, so a score-only rule answers it and invents a timing guarantee.
G4 stops it. Full rationale in [`DESIGN.md`](DESIGN.md) section 4.

`support_score = 0.5·norm_score + 0.5·coverage` is reported for transparency;
the **decision comes from the gates**, so a response can show a middling score
and still abstain.

Thresholds live in [`support_answer/config.py`](support_answer/config.py) and
can be overridden per run (`--min-coverage`, `--min-score`).

Tuning bias is **toward abstention**: in a support product a confident wrong
answer costs more than "I do not have that information, contact support".
`eval.py` scores both classes so the bias stays measurable.

## Answering layers

Retrieval and answer composition are separate modules, so the answering layer
is swappable:

| Mode | Behaviour |
|------|-----------|
| `rule` (default) | **Extractive.** The answer is verbatim sentences selected from the retrieved passages, so it cannot hallucinate by construction. |
| `llm` | Sends **only** the retrieved passages plus the question to a hosted model, then runs a grounding check (answer content terms must occur in the context at ≥0.80) and falls back to extractive if it fails. |
| `auto` | `llm` when a key is present, otherwise `rule`. |

Two providers are supported, picked automatically from whichever key is set
(Gemini first, since its free tier means the LLM path can be run without
paid credentials):

| Provider | Key | Default model | Install |
|----------|-----|---------------|---------|
| Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-2.5-flash` | none — plain `urllib` REST call |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-5-5` | `pip install anthropic` |

```bash
# PowerShell
$env:GEMINI_API_KEY = "..."
python app.py --answerer llm "How long does a withdrawal review take?"
```

```bash
# bash
export GEMINI_API_KEY=...
python app.py --answerer llm --llm-provider gemini "How long does a withdrawal review take?"
```

Override the model with `--llm-model` (or `GEMINI_MODEL` / `ANTHROPIC_MODEL`).
`--llm-provider gemini|anthropic` forces one; naming a provider with no key is
an error rather than a silent switch, so a misconfigured run is visible.

Verified live against Gemini `gemini-2.5-flash`: `debug.answerer_used` reports
`llm:gemini` with `llm_grounding_ratio: 1.0`, and the answer tightens to
*"Withdrawals usually complete within 24 hours. Reviews can take longer on
weekends or holidays."* — shorter than the extractive version, same facts.
Free-tier quota (HTTP 429) was also exercised: it falls back to the extractive
answer and records the error, so a quota-limited key degrades instead of
failing.

Keys can also go in a local `.env` (gitignored, see `.env.example`); real
environment variables always win over it.

**The LLM layer is strictly optional.** With no key, no SDK, an API error, or a
failed grounding check, `--answerer llm` still returns an answer: it falls back
to the extractive composer and records why in `debug.answerer_notes`, with
`debug.answerer_used` showing `rule` instead of `llm:gemini`. The model never
widens retrieval, never sees documents outside the retrieved set, and never
supplies the citations — those come from the retrieved evidence either way — so
the grounding guarantee is identical in both modes.

## Chunking

Short FAQ articles are **not** split: each is already a single-topic chunk, and
keeping it whole preserves the topical context BM25 needs and keeps citations
document-level. Sentences are the unit for evidence selection and snippets.

Documents longer than `chunk_threshold_chars` (1200) are split into 6-sentence
windows with 2-sentence overlap, so a replacement KB with long articles does
not dilute one relevant sentence across a long document. Chunks keep their
parent id and citations de-duplicate by parent, so chunking never shows up in
the response. The sample KB never triggers it; a test fixture covers that path.

## Layout

```
kb.json                  sample knowledge base
app.py                   CLI
server.py                stdlib HTTP server (POST /ask, GET /healthz)
validate.py              dependency-free validation (14 checks)
eval.py                  eval harness + metrics
eval_questions.json      8 cases with expected decisions
tests/test_service.py    pytest suite (29 tests)
support_answer/
  config.py   thresholds and settings      retrieval.py  BM25 index
  text.py     tokenise/stem/sentences      support.py    gates G1-G6
  kb.py       load, validate, chunk        answer.py     extractive + LLM composers
  schema.py   contract + validator         pipeline.py   the four stages
```

## Known limits

- Lexical retrieval cannot bridge a pure paraphrase that shares no vocabulary
  with the KB ("I cannot get in" vs "password reset"); that surfaces as a safe
  abstain rather than a wrong answer.
- Extractive answers read like the source article rather than like an agent,
  and can carry one clause more than strictly needed. `--answerer llm`
  addresses phrasing without loosening grounding.
- `eval_questions.json` expectations are tied to the sample `kb.json`;
  `validate.py` is the knowledge-base-agnostic check.
