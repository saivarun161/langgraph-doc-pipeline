"""Node-by-node progress events, streamed while a document is still in flight.

``run_document`` hands back a finished result: the trace explains what happened,
but only once everything already has. That is the wrong shape for anything a
person is waiting on — a CLI run over a directory, a UI showing a document being
processed — where the interesting moment is *the retry firing*, not the record
that it fired.

So this module exposes the same run as a stream of :class:`PipelineEvent`s, one
per node completion, built on LangGraph's ``stream(..., stream_mode="updates")``.
Each event carries the partial state its node returned, the trace lines it
appended, and — the part that makes the stream usable on its own — an
accumulated snapshot of the state so far. The last event's ``state`` is exactly
what ``run_document`` would have returned, so a caller never has to choose
between watching the run and getting its result.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .agents import DEFAULT_MAX_ATTEMPTS, DEFAULT_MIN_CONFIDENCE, ConfidencePolicy
from .engine import Engine
from .graph import build_pipeline, initial_state
from .state import ADDITIVE_CHANNELS

__all__ = ["PipelineEvent", "stream_document", "stream_pipeline"]

# LangGraph reserves double-underscore keys in the update stream (interrupts and
# other control channels). They are not agent nodes, so they are not events.
_RESERVED_PREFIX = "__"


@dataclass(frozen=True)
class PipelineEvent:
    """One node finishing, as seen from outside the graph.

    Attributes:
        doc_id: the document being processed.
        step: 1-based position in this document's stream.
        node: the node that just ran (``classify``/``extract``/``validate``/``summarize``).
        visit: how many times *this* node has run for this document — the retry
            loop is the only thing that pushes it past 1, which makes ``visit``
            the cheapest way for a consumer to spot a self-correction happening.
        update: the partial state the node returned.
        lines: the trace lines the node appended this visit.
        state: the accumulated state after applying ``update``.
        elapsed: seconds from the start of the run to this node finishing.
    """

    doc_id: str
    step: int
    node: str
    visit: int
    update: dict[str, Any]
    lines: tuple[str, ...]
    state: dict[str, Any]
    elapsed: float

    @property
    def is_final(self) -> bool:
        """Whether this is the last event of the run.

        ``status`` is written only by ``summarize``, which is the graph's one
        terminal node, so its presence marks the end without the stream having
        to look ahead.
        """
        return "status" in self.state

    def render(self) -> str:
        """One indented, timestamped line per trace line the node emitted."""
        labels = self.lines or (f"{self.node} → done",)
        return "\n".join(f"    [{self.elapsed:6.3f}s] {label}" for label in labels)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable view. ``raw_text`` is dropped from the state
        snapshot — it is the same document on every event, and repeating a whole
        document per node would swamp the thing being streamed."""
        return {
            "doc_id": self.doc_id,
            "step": self.step,
            "node": self.node,
            "visit": self.visit,
            "update": self.update,
            "lines": list(self.lines),
            "state": {k: v for k, v in self.state.items() if k != "raw_text"},
            "elapsed": round(self.elapsed, 4),
            "is_final": self.is_final,
        }


def _merge(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Apply one node's update to the accumulated state.

    Mirrors LangGraph's own channel semantics: the channels listed in
    ``ADDITIVE_CHANNELS`` accumulate, everything else is last-write-wins. The
    result is a new dict, so snapshots already handed to a consumer are never
    mutated out from under it.
    """
    merged = dict(state)
    for key, value in update.items():
        if key in ADDITIVE_CHANNELS:
            merged[key] = [*merged.get(key, []), *value]
        else:
            merged[key] = value
    return merged


def stream_pipeline(pipeline: Any, text: str, doc_id: str = "doc") -> Iterator[PipelineEvent]:
    """Stream one document through an already-compiled pipeline.

    Takes the compiled graph rather than building one, so a caller working
    through a corpus pays the build cost once — the same reason the batch runner
    compiles a single shared pipeline.

    Args:
        pipeline: a graph from :func:`docpipeline.graph.build_pipeline`.
        text: the raw document.
        doc_id: identifier carried on every event.

    Yields:
        A :class:`PipelineEvent` per node completion, in execution order.
    """
    state: dict[str, Any] = dict(initial_state(text, doc_id))
    visits: dict[str, int] = {}
    step = 0
    started = time.perf_counter()

    for chunk in pipeline.stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            if node.startswith(_RESERVED_PREFIX) or not isinstance(update, dict):
                continue
            step += 1
            visits[node] = visits.get(node, 0) + 1
            state = _merge(state, update)
            yield PipelineEvent(
                doc_id=doc_id,
                step=step,
                node=node,
                visit=visits[node],
                update=dict(update),
                lines=tuple(update.get("trace", ())),
                state=state,
                elapsed=time.perf_counter() - started,
            )


def stream_document(
    text: str,
    doc_id: str = "doc",
    engine: Engine | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_confidence: ConfidencePolicy = DEFAULT_MIN_CONFIDENCE,
) -> Iterator[PipelineEvent]:
    """Run one document through a freshly built pipeline, streaming its progress.

    The streaming counterpart of :func:`docpipeline.graph.run_document`, and it
    takes the same tuning arguments. The final event's ``state`` is what
    ``run_document`` would have returned for the same input::

        for event in stream_document(text):
            print(event.render())
            if event.is_final:
                result = event.state
    """
    pipeline = build_pipeline(
        engine=engine, max_attempts=max_attempts, min_confidence=min_confidence
    )
    yield from stream_pipeline(pipeline, text, doc_id)
