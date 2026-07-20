# langgraph-doc-pipeline

[![CI](https://github.com/saivarun161/langgraph-doc-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/saivarun161/langgraph-doc-pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **multi-agent document-processing pipeline built on [LangGraph](https://langchain-ai.github.io/langgraph/)**. An unstructured document flows through a stateful graph of agents that **classify** it, **extract** its fields, **validate** the result against per-type rules, and **summarize** it — with a **self-correcting retry loop** that re-extracts when validation fails and flags a document for human review only after it has genuinely exhausted its options.

It runs with **no API key**: the default reasoning engine is deterministic rule-based logic, so the whole pipeline and its test suite work offline. Swap in the OpenAI engine for production without touching the graph.

```text
             ┌───────────────────────────── self-correcting loop ─────────────────────────────┐
             │                                                                                  │
START ─► classify ─┬─(unknown)──────────────────────────────────────────────────────────► summarize ─► END
                   │                                                                        ▲   │
                   └─(recognized)─► extract ─► validate ─┬─(errors & attempts remain)─► ────┘   │
                                       ▲                 └─(clean, or retries exhausted)─────────┘
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
    • classify → clinical_note (confidence 1.00)
    • extract → 5 field(s): assessment, dob, mrn, patient_name, plan
    • validate → 1 error(s), 0 warning(s)
    • extract (retry 1) → 6 field(s): assessment, chief_complaint, dob, mrn, patient_name, plan
    • validate → clean, 0 warning(s)
    • summarize → status=ok
```

That trace is the whole point: the first extraction missed the `CC:` abbreviation, validation caught it, and the retry recovered it — automatically.

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
├── graph.py      # assembles the StateGraph, conditional edges, retry loop
├── samples.py    # bundled synthetic documents
├── cli.py        # docpipeline command
└── data/         # sample_docs.jsonl
tests/            # engine, agents, graph (incl. retry & routing), CLI, samples
.github/workflows/ci.yml   # ruff + pytest on Python 3.11 and 3.12
```

**Design choices worth calling out:**

- **Reasoning is pluggable; orchestration is not.** Classification and extraction go through an `Engine` interface, so the deterministic and LLM backends are interchangeable. The graph — the part that's actually the contribution — stays identical.
- **Validation is code, not a model call.** Checking required fields is deterministic business logic, so it lives in plain Python the tests pin exactly. Only the fuzzy work (classify, extract) is delegated to the engine.
- **The retry changes behavior, so it can't loop pointlessly.** The second extraction pass enables more lenient patterns (e.g. accepting `CC:`); bounded attempts guarantee termination.
- **`unknown` is a first-class outcome.** A document that matches nothing is flagged for review, never forced into the closest wrong bucket.

## Roadmap

- [x] LangGraph state machine with conditional edges + bounded retry loop
- [x] Pluggable rule-based (keyless) and OpenAI engines
- [x] Per-type validation with error/warning distinction
- [x] CLI + library API, full test suite, CI
- [ ] Streaming/token-level progress via LangGraph events
- [ ] Human-in-the-loop interrupt on `needs_review` (LangGraph checkpointer)
- [ ] Confidence-weighted routing and a batch runner with metrics

## License

MIT — see [LICENSE](LICENSE).

Built by [Varun Kammadanam](https://www.linkedin.com/in/varun-kammadanam-a823a6196) — backend + GenAI engineer (Java, Python, AWS, agentic systems).
