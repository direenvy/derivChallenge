# DESIGN — Grounded AI Support Answer Service

Companion to `SPECS.md`. Point form: what the system is made of, and why each
choice was made.

## 1. Constraints driving the design

- Evaluator runs from a **clean checkout** and may **swap `kb.json`** → zero
  third-party deps in the core path, no build step, no network, no secrets.
- **Over-answering is the failure mode being graded** → extractive answers by
  default, support check = several independent gates, not one threshold.
- Must be readable in one sitting → ~700 lines, one module per stage.

## 2. Layout

```
app.py          CLI entry point
server.py       stdlib HTTP server, POST /ask
validate.py     dependency-free validation script
eval.py         8-question eval harness
support_answer/
  config.py     thresholds + settings (single source of truth)
  text.py       tokenise, stem, stopwords, sentences, acronyms
  kb.py         load + validate kb.json, chunk long docs
  retrieval.py  BM25 index + search
  support.py    support check (gates G1-G6)
  answer.py     extractive + LLM composers, grounding check
  pipeline.py   LOAD_KB -> RETRIEVE -> CHECK_SUPPORT -> ANSWER|ABSTAIN
  schema.py     response construction + contract validator
```

- One-way dependencies: `pipeline` → {`retrieval`, `support`, `answer`} →
  {`text`, `config`} (`answer` also imports `support` for the `Evidence` type).
- `answer.py` imports neither `kb` nor `retrieval`, and only receives an
  `Evidence` object → "no answer before retrieval" is **structural**, not a
  promise in a comment.

## 3. Retrieval — hand-written Okapi BM25 (k1=1.5, b=0.75)

Why not the alternatives:

| Option | Why not |
|--------|---------|
| Hosted embeddings | needs key + network; evaluator runs without secrets |
| Local embedding model | hundreds of MB on a clean checkout |
| `sklearn` TF-IDF | a dependency for ~60 lines of arithmetic; BM25 handles uneven article lengths better |
| Plain keyword overlap | no IDF → common words dominate, score not discriminative |

BM25 also produces the IDF signal the support check reuses, and it is exact and
deterministic.

Preprocessing (all generic, nothing KB-specific):

- Lowercase, tokens = `[a-z0-9]+` runs → `2fa` stays one token.
- Conservative suffix stemmer (`-s -es -ies -ing -ed -ly`) so
  withdrawal/withdrawals, review/reviews match. Not Porter: no dep, fewer
  surprises.
- 90-word stopword list.
- **Title boost** — title tokens indexed twice (an FAQ title is the topic label).
- **Acronym expansion derived from the KB itself** — initials of title words
  are indexed, and a leading number-word also emits its digit, so
  "Two-factor authentication" indexes `tfa` and `2fa`. Works on whatever
  titles a replacement KB has; no hardcoded synonym list.

Known limit: pure paraphrase with no shared vocabulary ("I cannot get in" vs
"password reset") will not retrieve. Given abstention is the designed
behaviour, that shows up as a safe abstain, not a wrong answer. LLM mode does
not fix it — the LLM never widens retrieval.

## 4. Support check (the core of the task)

One threshold is not enough: *"Can you guarantee my withdrawal will finish in
2 hours?"* scores **high** on the withdrawals doc, so a score-only rule answers
it and invents a guarantee. Hence five gates, all must pass, any failure →
abstain.

Where `Q` = question content terms (stemmed, deduped), `idf(t)` from the index,
top-k = 3:

| Gate | Rule | What it catches |
|------|------|-----------------|
| **G1** retrieval strength | `norm_score = top1_bm25 / Σ idf(t)·(k1+1)` ≥ `0.12` | nothing relevant in the KB |
| **G2** evidence coverage | `max over passages of (Σ idf(t) for t in Q ∩ p) / Σ idf(t)` ≥ `0.55` | question only half-covered |
| **G3** conduct limitation | best evidence sentence must not be a negation + advice/provision cue | "what tax advice do you give?" |
| **G4** unsupported quantity | every standalone number in the question must appear in the evidence | "finish in 2 hours" |
| **G5** extractive evidence | ≥1 sentence scores above zero | guarantees ≥1 citation |
| **G6** phrase consistency | any adjacent term-pair the question names that exists in the KB must exist in the best passage | "change my **email address**" when the best passage is *residential* address |

Key design points:

- **G1 is normalised by the query own IDF mass**, so the threshold transfers
  across questions *and* across replacement KBs of different sizes. A raw BM25
  cutoff would need retuning per KB.
- **G2 is per-passage, not over the union.** *"What tax advice do you provide
  for closed accounts?"* is fully covered by the *union* of the policy doc and
  the closure doc, but by neither alone — it straddles two topics the KB does
  not join. This is the brief "top passages must agree on the same topic" rule
  expressed as a number.
- **G3 distinguishes verb classes.** Negation + `provide/offer/advice/advise/
  guarantee/recommend/promise` = refusing to give advice → abstain. Negation +
  `view/send` (e.g. *"Support agents cannot manually view or send existing
  passwords"*) = a statement of product behaviour → still a good grounded
  answer ("no, they cannot").
- **G4 only counts purely numeric tokens**, so `2fa` is unaffected.
- **G6 was added during the build, not designed up front.** G2 put "can
  support change my email address" at coverage 0.646 and "I forgot my password,
  how do I get into my account" at 0.667 — the first must abstain, the second
  should answer, and no threshold separates 0.021 of margin robustly. Pairs are
  unordered, so "change address" matches a title of "Address changes", and the
  gate only fires for a phrase the KB knows somewhere else, so an unknown
  phrase never blocks an answer.
- `support_score = 0.5·norm_score + 0.5·best_coverage` is reported for
  transparency, but the **decision comes from the gates** — so a response can
  show a middling score and still abstain. `debug.gates` names the gate that
  failed.
- **Bias is toward abstention**: a confident wrong answer is worse than
  "contact support" in a support product. `eval.py` keeps that bias from going
  too far by scoring both classes.

## 5. Answer composition

**Extractive (default)**

- Score sentences in top-k by IDF-weighted overlap with `Q`, small bonus for
  the top-1 passage.
- Take ≤3 sentences, from ≤2 documents, in original order, joined verbatim.
- Two precision filters, both added after seeing off-topic citations on a
  swapped KB: a sentence must score ≥40% of the best sentence, and a **second**
  document may only contribute if it was retrieved at ≥50% of the top score.
  Without them, one shared low-IDF word ("bike") attaches an unrelated article
  to the answer.
- Grounded **by construction** — the answer is literally KB text.
- Citations: one per contributing doc, snippet = the sentences used.
- Tradeoff: reads like the FAQ rather than like an agent, and can carry one
  extra clause. Correct side to err on for this brief.

**LLM (opt-in, `--answerer llm`)**

- Sends only retrieved passages + question to the Anthropic Messages API
  (`claude-opus-5-5`), system prompt restricted to the supplied context.
- Output passes a **grounding check**: answer content tokens must appear in the
  context at ≥0.80, else the extractive answer is used.
- Silent fallback to extractive on: SDK missing, no credentials, API error,
  grounding failure. Recorded in `debug.answerer_used`, so the evaluator run is
  identical with or without secrets.

## 6. Chunking

**Document = retrieval unit; sentence = evidence unit; long docs windowed.**

- Sample KB articles are 2-4 sentences on one topic — already the natural
  chunk. Splitting them would *hurt*: fragments lose topical context, IDF is
  computed over near-duplicates, and citations would point at fragments instead
  of the article a human agent would cite.
- But a replacement KB may hold 20-paragraph articles, where one document is
  too coarse (length normalisation dilutes the one relevant sentence → wrong
  abstains). So docs over `chunk_threshold_chars` (1200) split into **6-sentence
  windows with 2-sentence overlap**; the overlap stops evidence being cut in
  half at a boundary.
- Chunks keep the parent `id`/`title` and citations de-dupe by parent id →
  chunking is invisible in the output contract.
- Accepted tradeoff: the sample KB never triggers it, so a test fixture with a
  long document covers that path.

## 7. Failure modes

| Situation | Behaviour |
|-----------|-----------|
| `kb.json` missing / bad JSON / not an array / field missing | fatal, explicit message, exit 2 |
| Empty KB array | loads; everything abstains (G1) |
| Blank question | CLI usage error; HTTP 400 |
| Gibberish or other-language question | abstain (no overlap) |
| Two topics, neither answering | abstain (G2) |
| Asks for a figure not in the KB | abstain (G4) |
| Asks for advice the KB forbids | abstain (G3) |
| LLM mode without a key | extractive answer, noted in `debug` |

## 8. With more time

- Query expansion from KB co-occurrence stats, to narrow the paraphrase gap
  without a model download.
- Auto-tune `min_coverage` from an abstention set generated out of the KB.
- Span-level citation offsets (`start_char`/`end_char`) for UI highlighting.
