"""langgraph-doc-pipeline — a multi-agent document-processing pipeline on LangGraph.

An unstructured document flows through a stateful agent graph: it is classified,
its fields are extracted, the extraction is validated against per-type rules, and
a summary is produced. When validation fails, the graph loops back for a second,
more lenient extraction pass before giving up and flagging the document for review.

The reasoning is done by a pluggable engine: a deterministic, dependency-free
rule-based engine (the default, so everything runs with no API key) or an
OpenAI-backed engine for production.
"""

from .batch import (
    BatchMetrics,
    BatchResult,
    ThresholdSuggestion,
    TypeStats,
    calibrate_thresholds,
    run_batch,
    summarize_batch,
)
from .graph import build_pipeline, run_document
from .state import DOC_TYPES, DocState
from .stream import PipelineEvent, stream_document, stream_pipeline

__version__ = "0.4.0"

__all__ = [
    "DOC_TYPES",
    "BatchMetrics",
    "BatchResult",
    "DocState",
    "PipelineEvent",
    "ThresholdSuggestion",
    "TypeStats",
    "build_pipeline",
    "calibrate_thresholds",
    "run_batch",
    "run_document",
    "stream_document",
    "stream_pipeline",
    "summarize_batch",
]
