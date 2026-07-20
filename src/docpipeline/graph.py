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
    classify_node,
    extract_node,
    route_after_classify,
    route_after_validate,
    summarize_node,
    validate_node,
)
from .engine import Engine, RuleBasedEngine
from .state import DocState

DEFAULT_MAX_ATTEMPTS = 2


def build_pipeline(engine: Engine | None = None, max_attempts: int = DEFAULT_MAX_ATTEMPTS):
    """Build and compile the document-processing graph.

    Args:
        engine: reasoning engine to bind to the nodes (defaults to the keyless
            ``RuleBasedEngine``).
        max_attempts: how many extraction passes to allow before a document with
            validation errors is flagged for review.
    """
    engine = engine or RuleBasedEngine()

    graph = StateGraph(DocState)
    graph.add_node("classify", partial(classify_node, engine=engine))
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
        partial(route_after_validate, max_attempts=max_attempts),
        {"extract": "extract", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)

    return graph.compile()


def run_document(
    text: str,
    doc_id: str = "doc",
    engine: Engine | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Convenience wrapper: run one document through a freshly built pipeline and
    return the final state (doc_type, fields, errors, summary, status, trace)."""
    pipeline = build_pipeline(engine=engine, max_attempts=max_attempts)
    initial: DocState = {"doc_id": doc_id, "raw_text": text, "attempts": 0, "trace": []}
    return pipeline.invoke(initial)
