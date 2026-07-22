# langgraph-doc-pipeline

[![CI](https://github.com/saivarun161/langgraph-doc-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/saivarun161/langgraph-doc-pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **multi-agent document-processing pipeline built on [LangGraph](https://langchain-ai.github.io/langgraph/)**. An unstructured document flows through a stateful graph of agents that **classify** it, **extract** its fields, **validate** the result against per-type rules, and **summarize** it — with a **self-correcting retry loop** that re-extracts when validation fails and flags a document for human review only after it has genuinely exhausted its options.

It runs with **no API key**: the default reasoning engine is deterministic rule-based logic, so the whole pipeline and its test suite work offline. Swap in the OpenAI engine for production without touching the graph.

```text
             ┌───────────────────────────── self-correcting loop ─────────────────────────────┐
             │                                                                                  │
START ─► classify ─┬─(unknown / low confidence)─────────────────────────────────────────► summarize ─► END
                   │                                                                        ▲   │
                   └─(trusted type)─► extract ─► validate ─┬─(errors & budget remains)─► ──┘   │
                                       ▲                   └─(clean, or budget spent)──────────┘
                                       │                                                    status:
                                  engine.extract                                       ok | needs_review
                              (rule-based | OpenAI)
```

## Why this design

Real document pipelines fail on the messy 10%: a field labeled `CC:` instead of `Chief Complaint:`, a scan that classified fine but extracted badly. A straight-line `classify → extract → done` chain has nowhere to recover. Modeling the flow as a **graph with conditional edges** gives it three properties that matter:

- **It self-corrects.** When validation finds missing required fields, the graph routes *back* to extraction for a second, more lenient pass instead of emitting a broken record.
- **It knows when to stop.** Retries are bounded; a document that can't be completed is marked `needs_review` rather than looped forever or trusted silently.
- **It's auditable.** Every node appends to a `trace`, so each result carries the exact path it took — which recruiters, and on-call engineers, both appreciate.

## Quickstart — no API key

```bash
git clone https://github.com/saivarun161/langgraph-doc-pipeline.git
cd langgraph-doc-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

docpipeline --samples          # run the 6 bundled synthetic documents
```

Example output for one document:

```text
── note-02-abbrev — type=clinical_note status=ok
  fields:
    patient_name: Aisha Khan
    chief_complaint: Shortness of breath on exertion
    ...
  trace:
    • classify → clinical_note (confidence 1.00, budget 2)
    • extract → 5 field(s): assessment, dob, mrn, patient_name, plan
    • validate → 1 error(s), 0 warning(s)
    • extract (retry 1) → 6 field(s): assessment, chief_complaint, dob, mrn, patient_name, plan
    • validate → clean, 0 warning(s)
    • summarize → status=ok
```

That trace is the whole point: the first extraction missed the `CC:` abbreviation, validation caught it, and the retry recovered it — automatically.

## Batch runs and metrics

A single document's trace explains that document. A corpus needs different questions answered — what share came out clean, and is the retry loop actually rescuing anything or just burning passes?

```bash
docpipeline --dir ./inbox --metrics --workers 8
```

```text
── batch — 6 document(s) in 0.01s (531.0/s)
  status:       ok=5  needs_review=1 (17% review rate)
  types:        clinical_note=2, discharge_summary=1, lab_report=1, referral=1, unknown=1
  retries:      1 retried, 1 recovered by the retry loop
  skipped:      1 never reached extraction
  mean conf:    0.79   mean passes: 1.00
  completeness: 100% of required fields extracted
```

`retried` vs `recovered_by_retry` is the pair worth watching: retries that never recover are pure cost, and the gap between them tells you whether to raise `--max-attempts` or tighten `--min-confidence`.

The aggregate hides the thing you actually act on, though — a 17% review rate is a very different problem when it is one type failing most of the time than when it is every type failing occasionally. So `--metrics` also breaks the run down by type:

```text
── by type
  clinical_note      n=2    review=  0%  conf=1.00  passes=1.50  recovered=1/1
  discharge_summary  n=1    review=  0%  conf=1.00  passes=1.00  recovered=0/0
  lab_report         n=1    review=  0%  conf=1.00  passes=1.00  recovered=0/0
  referral           n=1    review=  0%  conf=0.75  passes=1.00  recovered=0/0
  unknown            n=1    review=100%  conf=0.00  passes=0.00  recovered=0/0
```

The same numbers are available as data:

```python
from docpipeline import run_batch

batch = run_batch(docs, workers=8)          # results stay in input order
batch.metrics.review_rate                   # 0.1667
batch.metrics.per_type["lab_report"]        # TypeStats(documents=1, ok=1, ...)
batch.metrics.as_dict()                     # JSON-ready, with derived rates
```

Add `--json --metrics` to get `{"results": [...], "metrics": {...}}` from the CLI. Without a reporting flag the JSON stays a bare list, so existing consumers are unaffected.

## Confidence-weighted routing

Extraction passes are **earned, not granted**. The retry pass is deliberately more lenient than the first — it accepts `CC:` for chief complaint, `Impression:` for assessment — and that leniency is only a good trade when the document was typed correctly to begin with. Spending it on a probably-misclassified document manufactures plausible-but-wrong fields, which is strictly worse than an honest `needs_review`.

So the budget scales with the classifier's confidence:

| Confidence | Passes (ceiling 2) | Behavior |
|---|---|---|
| `< 0.35` | 0 | never extracted; routed to review with the reason attached |
| `0.35 – 0.50` | 1 | one strict pass; no lenient retry |
| `> 0.50` | 2 | earns the retry that rescues messy-but-real documents |

Both knobs are tunable — `--min-confidence` for the trust threshold, `--max-attempts` for the ceiling — which is exactly what the batch metrics exist to tune against.

### One threshold per type

The types are not equally separable. A lab report is nearly unmistakable; a discharge summary shares most of its vocabulary with a clinical note. A single global threshold has to be set for the worst type, which means either wasting lenient passes on the easy ones or sending the hard ones to review too eagerly.

So `--min-confidence` is repeatable and accepts `TYPE=VALUE`: set a default, then override only the types that have earned something different.

```bash
docpipeline --dir ./inbox --min-confidence 0.30 --min-confidence discharge_summary=0.70
```

The same policy works from the library, as a mapping with an optional `"default"` entry:

```python
run_batch(docs, min_confidence={"discharge_summary": 0.70, "default": 0.30})
```

A policy is validated when the pipeline is built, so a mistyped key fails immediately instead of silently doing nothing. `"unknown"` is rejected outright: those documents never reach extraction under any threshold, so a value there could only ever be a no-op — and a knob that does nothing is worse than one that refuses to exist.

### Calibrating those thresholds

Picking per-type numbers by hand is guesswork. `--calibrate` fits them to a run you have already measured:

```bash
docpipeline --dir ./inbox --calibrate
```

```text
── calibration — suggested --min-confidence per type
  clinical_note      0.35 → 0.35  (unchanged)  [3/5 routed correctly, 3 ok]
  discharge_summary  0.35 → 0.80  [9/9 routed correctly, 5 ok]
```

Read the second row as: every discharge summary that scored below 0.80 was headed for review anyway, so extracting them was wasted work — raise the bar and all nine documents get routed correctly.

For each type it fits a one-dimensional decision stump over the observed confidences: a document is *routed correctly* if it scored at or above the threshold and finished `ok`, or scored below it and was headed for review anyway. The threshold with the most correct calls wins — and it must beat the one already in force *strictly*, so a fit that explains the data no better than the status quo is reported as `(unchanged)` rather than dressed up as advice.

It suggests; it never applies. Two reasons, both visible in the output above:

- **Confidence is not always the lever.** A clinical note with no `Plan:` anywhere in it goes to review no matter how confidently it was typed. The `3/5` on the first row is the tell: no threshold separates that type's outcomes, because the failures have nothing to do with classification.
- **Small samples fit noise.** Types with fewer than three documents are left out entirely rather than calibrated on a handful.

`calibrate_thresholds(results)` returns the same suggestions as data, and `--json --calibrate` puts them under a `"calibration"` key.

## Use it as a library

```python
from docpipeline import run_document

result = run_document("Referred To: Cardiology\nReason for Referral: afib\nPatient Name: D O'Brien")
print(result["doc_type"])   # 'referral'
print(result["status"])     # 'ok'
print(result["fields"])     # {'patient_name': "D O'Brien", 'referred_to': 'Cardiology', ...}
```

## Production engine (OpenAI)

The graph calls a pluggable `Engine`; the default is keyless and rule-based. For LLM-quality classification and extraction, install the extra and select the OpenAI engine — **the graph is unchanged**, only the reasoning improves:

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
docpipeline --samples --engine openai
```

## Recognized document types

| Type | Required fields (must be present to pass) |
|---|---|
| `clinical_note` | patient_name, chief_complaint, assessment, plan |
| `lab_report` | patient_name, results |
| `discharge_summary` | patient_name, admission_date, discharge_date, diagnosis |
| `referral` | patient_name, referred_to, reason |
| `unknown` | — (routed straight to review) |

Missing demographics (`mrn`, `dob`) are **warnings**, not errors — a document is still usable without them. All sample documents are fictional and contain no real patient data.

## Architecture

```text
src/docpipeline/
├── state.py      # DocState: the shared graph state (trace uses an additive reducer)
├── engine.py     # pluggable reasoning: RuleBasedEngine (keyless) | OpenAIEngine
├── agents.py     # node functions (classify/extract/validate/summarize) + routing
│                 #   and attempt_budget: confidence → extraction passes
├── graph.py      # assembles the StateGraph, conditional edges, retry loop
├── batch.py      # corpus runner, BatchMetrics + TypeStats, threshold calibration
├── samples.py    # bundled synthetic documents
├── cli.py        # docpipeline command
└── data/         # sample_docs.jsonl
tests/            # engine, agents, graph (incl. retry & routing), batch, CLI, samples
.github/workflows/ci.yml   # ruff + pytest on Python 3.11 and 3.12
```

**Design choices worth calling out:**

- **Reasoning is pluggable; orchestration is not.** Classification and extraction go through an `Engine` interface, so the deterministic and LLM backends are interchangeable. The graph — the part that's actually the contribution — stays identical.
- **Validation is code, not a model call.** Checking required fields is deterministic business logic, so it lives in plain Python the tests pin exactly. Only the fuzzy work (classify, extract) is delegated to the engine.
- **The retry changes behavior, so it can't loop pointlessly.** The second extraction pass enables more lenient patterns (e.g. accepting `CC:`); bounded attempts guarantee termination.
- **`unknown` is a first-class outcome.** A document that matches nothing is flagged for review, never forced into the closest wrong bucket.
- **Retries are rationed by confidence, not handed out flat.** A weak classification buys fewer lenient passes — and below the threshold, none at all. Being wrong loudly beats being wrong plausibly.
- **Thresholds are per type, and fitted rather than guessed.** The types are not equally separable, so they do not share one number; `--calibrate` derives each from measured outcomes. It suggests rather than applies, and refuses to suggest a change that does not demonstrably route more documents correctly.
- **The batch runner shares one compiled graph.** The pipeline and the bundled engines hold no per-document state, so a corpus can be run across a thread pool; results are returned in input order regardless of completion order.

## Roadmap

- [x] LangGraph state machine with conditional edges + bounded retry loop
- [x] Pluggable rule-based (keyless) and OpenAI engines
- [x] Per-type validation with error/warning distinction
- [x] CLI + library API, full test suite, CI
- [x] Confidence-weighted routing and a batch runner with metrics
- [x] Per-type confidence thresholds calibrated from batch metrics
- [ ] Streaming/token-level progress via LangGraph events
- [ ] Human-in-the-loop interrupt on `needs_review` (LangGraph checkpointer)

## License

MIT — see [LICENSE](LICENSE).

Built by [Varun Kammadanam](https://www.linkedin.com/in/varun-kammadanam-a823a6196) — backend + GenAI engineer (Java, Python, AWS, agentic systems).
