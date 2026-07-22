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
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from .agents import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MIN_CONFIDENCE,
    REQUIRED_FIELDS,
    ConfidencePolicy,
    resolve_min_confidence,
)
from .engine import Engine
from .graph import build_pipeline, initial_state
from .state import STATUS_NEEDS_REVIEW, STATUS_OK

# A type needs at least this many documents before its outcomes say anything
# trustworthy about where its threshold belongs.
DEFAULT_MIN_SAMPLES = 3


@dataclass(frozen=True)
class TypeStats:
    """One document type's slice of a batch.

    The whole-batch numbers hide the thing you actually tune against: a 20%
    review rate is a very different problem when it is one type failing most of
    the time than when it is every type failing occasionally.
    """

    doc_type: str = ""
    documents: int = 0
    ok: int = 0
    needs_review: int = 0
    retried: int = 0
    recovered_by_retry: int = 0
    skipped_extraction: int = 0
    mean_confidence: float = 0.0
    mean_attempts: float = 0.0
    field_completeness: float = 0.0

    @property
    def review_rate(self) -> float:
        """Share of this type's documents handed to a human, in [0, 1]."""
        return self.needs_review / self.documents if self.documents else 0.0


@dataclass(frozen=True)
class BatchMetrics:
    """Aggregate outcome of one batch run."""

    documents: int = 0
    ok: int = 0
    needs_review: int = 0
    # The same counters, sliced by document type.
    per_type: dict[str, TypeStats] = field(default_factory=dict)
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
    def by_type(self) -> dict[str, int]:
        """Document count per type — the headline slice of ``per_type``."""
        return {name: stats.documents for name, stats in self.per_type.items()}

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
            "by_type": self.by_type,
            "review_rate": round(self.review_rate, 4),
            "docs_per_second": round(self.docs_per_second, 2),
            "per_type": {
                name: {**asdict(stats), "review_rate": round(stats.review_rate, 4)}
                for name, stats in self.per_type.items()
            },
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

    def render_types(self) -> str:
        """Per-type breakdown, one aligned row per type."""
        if not self.per_type:
            return "── by type — no documents"
        width = max(len(name) for name in self.per_type)
        rows = [
            f"  {name:<{width}}  n={s.documents:<4} review={s.review_rate:>4.0%}  "
            f"conf={s.mean_confidence:.2f}  passes={s.mean_attempts:.2f}  "
            f"recovered={s.recovered_by_retry}/{s.retried}"
            for name, s in self.per_type.items()
        ]
        return "\n".join(["── by type", *rows])


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


def _tally(results: list[dict[str, Any]]) -> dict[str, Any]:
    """The counters the whole batch and each per-type slice both need.

    Computed once here so the aggregate and the breakdown can never drift apart
    — a per-type ``recovered_by_retry`` that means something subtly different
    from the batch-wide one would be worse than not reporting it.
    """
    attempts = [r.get("attempts", 0) for r in results]
    confidences = [float(r.get("classification_confidence", 0.0)) for r in results]
    scored = [c for c in (_completeness(r) for r in results) if c is not None]

    return {
        "documents": len(results),
        "ok": sum(1 for r in results if r.get("status") == STATUS_OK),
        "needs_review": sum(1 for r in results if r.get("status") == STATUS_NEEDS_REVIEW),
        "retried": sum(1 for a in attempts if a > 1),
        "recovered_by_retry": sum(
            1 for r in results if r.get("attempts", 0) > 1 and r.get("status") == STATUS_OK
        ),
        "skipped_extraction": sum(1 for a in attempts if a == 0),
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "mean_attempts": round(sum(attempts) / len(attempts), 4) if attempts else 0.0,
        "field_completeness": round(sum(scored) / len(scored), 4) if scored else 0.0,
    }


def _group_by_type(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Results bucketed by document type, in a stable (sorted) key order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result.get("doc_type", "unknown"), []).append(result)
    return dict(sorted(grouped.items()))


def summarize_batch(results: list[dict[str, Any]], wall_seconds: float = 0.0) -> BatchMetrics:
    """Compute metrics over already-collected results.

    Split out from ``run_batch`` so metrics can also be taken over results that
    were produced elsewhere — a resumed run, or a subset filtered by type.
    """
    if not results:
        return BatchMetrics(wall_seconds=round(wall_seconds, 4))

    per_type = {
        name: TypeStats(doc_type=name, **_tally(subset))
        for name, subset in _group_by_type(results).items()
    }
    return BatchMetrics(
        per_type=per_type,
        wall_seconds=round(wall_seconds, 4),
        **_tally(results),
    )


# --------------------------------------------------------------------------- #
# Threshold calibration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ThresholdSuggestion:
    """A confidence threshold fitted to one type's observed outcomes."""

    doc_type: str
    threshold: float
    current: float
    documents: int
    ok: int
    # How many of this type's documents the suggested threshold would have
    # routed correctly — extracted the ones that finished clean, skipped the
    # ones that ended up in review anyway.
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.documents if self.documents else 0.0

    @property
    def changed(self) -> bool:
        """Whether this differs from the threshold currently in force."""
        return abs(self.threshold - self.current) >= 0.005

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "accuracy": round(self.accuracy, 4),
            "changed": self.changed,
        }


def calibrate_thresholds(
    results: list[dict[str, Any]],
    min_confidence: ConfidencePolicy = DEFAULT_MIN_CONFIDENCE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict[str, ThresholdSuggestion]:
    """Fit a per-type confidence threshold to the outcomes of a batch.

    For each type, this fits a one-dimensional decision stump over the
    classification confidences: for every candidate threshold ``t``, a document
    is *called correctly* if it scored at or above ``t`` and finished ``ok``, or
    scored below ``t`` and would have been sent to review regardless. The
    threshold with the most correct calls wins, and it must beat the threshold
    already in force *strictly* — a fit that explains the data no better than
    the status quo is reported as unchanged rather than dressed up as advice.
    That also settles ties toward the lowest threshold, which is the safer
    choice when several explain the data equally well.

    Candidates are the confidences actually observed for that type, plus one
    just above the highest — which expresses "stop extracting this type at all".

    Two honest caveats, which is why this *suggests* rather than applies:

    * A document can end in review for reasons confidence knows nothing about
      (a clinical note with no ``Plan:`` anywhere in it). Skipping it earlier
      saves wasted passes but does not make it come out clean, so a type whose
      failures are all of that kind will pull its threshold up for no gain.
    * Types with fewer than ``min_samples`` documents are left out entirely
      rather than fitted to noise.

    Args:
        results: per-document results, e.g. ``run_batch(...).results``.
        min_confidence: the policy currently in force, reported as ``current``.
        min_samples: documents a type needs before it is fitted at all.

    Returns:
        A ``{doc_type: ThresholdSuggestion}`` mapping, sorted by type. Types
        that were skipped for lack of samples are absent.
    """
    suggestions: dict[str, ThresholdSuggestion] = {}

    for doc_type, subset in _group_by_type(results).items():
        # 'unknown' never reaches extraction, so it has no threshold to fit.
        if doc_type == "unknown" or len(subset) < min_samples:
            continue

        samples = [
            (float(r.get("classification_confidence", 0.0)), r.get("status") == STATUS_OK)
            for r in subset
        ]

        def routed_correctly(threshold: float, samples: list[tuple[float, bool]] = samples) -> int:
            return sum(1 for confidence, ok in samples if (confidence >= threshold) == ok)

        confidences = sorted({confidence for confidence, _ in samples})
        candidates = [*confidences, round(confidences[-1] + 0.01, 4)]

        # Start from the threshold already in force and only move off it for a
        # strict improvement. A fit that routes no more documents correctly than
        # the status quo is not a recommendation — and since candidates ascend,
        # requiring strictly-better also settles ties toward the lowest.
        current = resolve_min_confidence(min_confidence, doc_type)
        best, best_correct = current, routed_correctly(current)
        for candidate in candidates:
            if (correct := routed_correctly(candidate)) > best_correct:
                best, best_correct = candidate, correct

        suggestions[doc_type] = ThresholdSuggestion(
            doc_type=doc_type,
            threshold=round(best, 2),
            current=round(current, 2),
            documents=len(subset),
            ok=sum(1 for _, ok in samples if ok),
            correct=best_correct,
        )

    return suggestions


def render_calibration(suggestions: Mapping[str, ThresholdSuggestion]) -> str:
    """Human-readable calibration block, in the style of the metrics output."""
    if not suggestions:
        return (
            f"── calibration — no type had the {DEFAULT_MIN_SAMPLES}+ documents "
            "needed to fit a threshold"
        )
    width = max(len(name) for name in suggestions)
    rows = [
        f"  {s.doc_type:<{width}}  {s.current:.2f} → {s.threshold:.2f}"
        f"{'' if s.changed else '  (unchanged)'}"
        f"  [{s.correct}/{s.documents} routed correctly, {s.ok} ok]"
        for s in suggestions.values()
    ]
    return "\n".join(["── calibration — suggested --min-confidence per type", *rows])


def run_batch(
    docs: Iterable[Mapping[str, Any] | str],
    engine: Engine | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_confidence: ConfidencePolicy = DEFAULT_MIN_CONFIDENCE,
    workers: int = 1,
) -> BatchResult:
    """Run every document through one shared pipeline and measure the run.

    Args:
        docs: documents as raw strings, or mappings with ``text`` and optional ``id``.
        engine: reasoning engine (defaults to the keyless rule-based one).
        max_attempts: ceiling on extraction passes per document.
        min_confidence: classifications below this score skip extraction — one
            threshold, or a ``{doc_type: threshold}`` mapping.
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
