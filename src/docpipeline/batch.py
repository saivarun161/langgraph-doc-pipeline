"""Run many documents through the pipeline and measure the run.

A single document's ``trace`` explains that document. A corpus needs different
questions answered: what share came out clean, how often did the retry loop
actually rescue something, where is the pipeline spending its passes? Those are
the numbers you tune ``max_attempts`` and ``min_confidence`` against, so they are
computed here rather than left to whoever is reading the output.

The graph is compiled once and shared across the batch — the compiled pipeline
and the bundled engines hold no per-document state, so documents can be run
concurrently over a thread pool without interfering with each other.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from .agents import DEFAULT_MAX_ATTEMPTS, DEFAULT_MIN_CONFIDENCE, REQUIRED_FIELDS
from .engine import Engine
from .graph import build_pipeline, initial_state
from .state import STATUS_NEEDS_REVIEW, STATUS_OK


@dataclass(frozen=True)
class BatchMetrics:
    """Aggregate outcome of one batch run."""

    documents: int = 0
    ok: int = 0
    needs_review: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    # Documents that needed more than one extraction pass, and the subset of
    # those that finished clean — i.e. the retry loop earning its keep.
    retried: int = 0
    recovered_by_retry: int = 0
    # Documents that never reached extraction: unknown type, or a classification
    # below the confidence threshold.
    skipped_extraction: int = 0
    mean_confidence: float = 0.0
    mean_attempts: float = 0.0
    # Mean share of a type's required fields that were actually extracted,
    # over documents whose type declares required fields.
    field_completeness: float = 0.0
    wall_seconds: float = 0.0

    @property
    def review_rate(self) -> float:
        """Share of documents handed to a human, in [0, 1]."""
        return self.needs_review / self.documents if self.documents else 0.0

    @property
    def docs_per_second(self) -> float:
        return self.documents / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable view, including the derived rates."""
        return {
            **asdict(self),
            "review_rate": round(self.review_rate, 4),
            "docs_per_second": round(self.docs_per_second, 2),
        }

    def render(self) -> str:
        """Human-readable block, in the style of the per-document output."""
        types = ", ".join(f"{t}={n}" for t, n in sorted(self.by_type.items())) or "none"
        return "\n".join(
            [
                f"── batch — {self.documents} document(s) in {self.wall_seconds:.2f}s "
                f"({self.docs_per_second:.1f}/s)",
                f"  status:       ok={self.ok}  needs_review={self.needs_review} "
                f"({self.review_rate:.0%} review rate)",
                f"  types:        {types}",
                f"  retries:      {self.retried} retried, "
                f"{self.recovered_by_retry} recovered by the retry loop",
                f"  skipped:      {self.skipped_extraction} never reached extraction",
                f"  mean conf:    {self.mean_confidence:.2f}   "
                f"mean passes: {self.mean_attempts:.2f}",
                f"  completeness: {self.field_completeness:.0%} of required fields extracted",
            ]
        )


@dataclass(frozen=True)
class BatchResult:
    """Per-document results (in input order) plus the aggregate metrics."""

    results: list[dict[str, Any]]
    metrics: BatchMetrics

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)


def _normalize(doc: Mapping[str, Any] | str, index: int) -> tuple[str, str]:
    """Accept either a raw string or a mapping with ``text`` and an optional ``id``."""
    if isinstance(doc, str):
        return f"doc-{index + 1}", doc
    try:
        text = doc["text"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"document {index} has no 'text' key: {doc!r}") from exc
    return str(doc.get("id") or f"doc-{index + 1}"), text


def _completeness(result: Mapping[str, Any]) -> float | None:
    """Share of this document's required fields that were extracted, or None if
    its type declares none (``unknown``), which would otherwise be a free 100%."""
    required = REQUIRED_FIELDS.get(result.get("doc_type", "unknown"), ())
    if not required:
        return None
    fields = result.get("fields") or {}
    return sum(1 for name in required if fields.get(name)) / len(required)


def summarize_batch(results: list[dict[str, Any]], wall_seconds: float = 0.0) -> BatchMetrics:
    """Compute metrics over already-collected results.

    Split out from ``run_batch`` so metrics can also be taken over results that
    were produced elsewhere — a resumed run, or a subset filtered by type.
    """
    if not results:
        return BatchMetrics(wall_seconds=round(wall_seconds, 4))

    attempts = [r.get("attempts", 0) for r in results]
    confidences = [float(r.get("classification_confidence", 0.0)) for r in results]
    scored = [c for c in (_completeness(r) for r in results) if c is not None]

    return BatchMetrics(
        documents=len(results),
        ok=sum(1 for r in results if r.get("status") == STATUS_OK),
        needs_review=sum(1 for r in results if r.get("status") == STATUS_NEEDS_REVIEW),
        by_type=dict(Counter(r.get("doc_type", "unknown") for r in results)),
        retried=sum(1 for a in attempts if a > 1),
        recovered_by_retry=sum(
            1 for r in results if r.get("attempts", 0) > 1 and r.get("status") == STATUS_OK
        ),
        skipped_extraction=sum(1 for a in attempts if a == 0),
        mean_confidence=round(sum(confidences) / len(confidences), 4),
        mean_attempts=round(sum(attempts) / len(attempts), 4),
        field_completeness=round(sum(scored) / len(scored), 4) if scored else 0.0,
        wall_seconds=round(wall_seconds, 4),
    )


def run_batch(
    docs: Iterable[Mapping[str, Any] | str],
    engine: Engine | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    workers: int = 1,
) -> BatchResult:
    """Run every document through one shared pipeline and measure the run.

    Args:
        docs: documents as raw strings, or mappings with ``text`` and optional ``id``.
        engine: reasoning engine (defaults to the keyless rule-based one).
        max_attempts: ceiling on extraction passes per document.
        min_confidence: classifications below this score skip extraction.
        workers: thread-pool size. The default of 1 runs documents sequentially;
            raise it when the engine spends its time waiting on a network call.

    Returns:
        A ``BatchResult`` whose ``results`` preserve input order regardless of
        the order the pool happened to finish them in.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    prepared = [_normalize(doc, i) for i, doc in enumerate(docs)]
    pipeline = build_pipeline(
        engine=engine, max_attempts=max_attempts, min_confidence=min_confidence
    )

    def run_one(item: tuple[str, str]) -> dict[str, Any]:
        doc_id, text = item
        return pipeline.invoke(initial_state(text, doc_id))

    started = time.perf_counter()
    if workers == 1 or len(prepared) < 2:
        results = [run_one(item) for item in prepared]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(prepared))) as pool:
            # Executor.map yields in submission order, so input order is kept.
            results = list(pool.map(run_one, prepared))
    elapsed = time.perf_counter() - started

    return BatchResult(results=results, metrics=summarize_batch(results, elapsed))
