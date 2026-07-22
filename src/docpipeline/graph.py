"""Assemble the agent nodes into a compiled LangGraph pipeline.

START → classify ─┬─(unknown)────────────────► summarize → END
                  └─(recognized)─► extract → validate ─┬─(errors, tries left)─► extract
                                                       └─(clean or exhausted)─► summarize
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from .agents import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MIN_CONFIDENCE,
    ConfidencePolicy,
    classify_node,
    extract_node,
    route_after_classify,
    route_after_validate,
    summarize_node,
    validate_confidence_policy,
    validate_node,
)
from .engine import Engine, RuleBasedEngine
from .state import DocState

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MIN_CONFIDENCE",
    "build_pipeline",
    "initial_state",
    "run_document",
]


def build_pipeline(
    engine: Engine | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_confidence: ConfidencePolicy = DEFAULT_MIN_CONFIDENCE,
):
    """Build and compile the document-processing graph.

    Args:
        engine: reasoning engine to bind to the nodes (defaults to the keyless
            ``RuleBasedEngine``).
        max_attempts: ceiling on extraction passes. The actual budget a document
            receives is weighted by its classification confidence.
        min_confidence: classifications below this score are not extracted at
            all; the document is routed straight to review. Either one threshold
            for every type, or a ``{doc_type: threshold}`` mapping with an
            optional ``"default"`` entry — validated here so a mistyped key
            fails at build time rather than silently doing nothing.
    """
    engine = engine or RuleBasedEngine()
    validate_confidence_policy(min_confidence)

    graph = StateGraph(DocState)
    graph.add_node(
        "classify",
        partial(
            classify_node,
            engine=engine,
            max_attempts=max_attempts,
            min_confidence=min_confidence,
        ),
    )
    graph.add_node("extract", partial(extract_node, engine=engine))
    graph.add_node("validate", validate_node)
    graph.add_node("summarize", partial(summarize_node, engine=engine))

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify", route_after_classify, {"extract": "extract", "summarize": "summarize"}
    )
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"extract": "extract", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)

    return graph.compile()


def run_document(
    text: str,
    doc_id: str = "doc",
    engine: Engine | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_confidence: ConfidencePolicy = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Convenience wrapper: run one document through a freshly built pipeline and
    return the final state (doc_type, fields, errors, summary, status, trace)."""
    pipeline = build_pipeline(
        engine=engine, max_attempts=max_attempts, min_confidence=min_confidence
    )
    return pipeline.invoke(initial_state(text, doc_id))


def initial_state(text: str, doc_id: str = "doc") -> DocState:
    """The starting state for one document. Shared with the batch runner so both
    entry points seed exactly the same channels."""
    return {"doc_id": doc_id, "raw_text": text, "attempts": 0, "trace": []}
