"""The shared state that flows through the agent graph.

LangGraph merges each node's returned dict into this state. Most channels are
last-write-wins; ``trace`` uses an additive reducer so every node can append an
audit line without clobbering the others.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

# The document classes the pipeline recognizes. "unknown" is a real outcome:
# a document that matches nothing is routed straight to review, not forced
# into a wrong bucket.
DOC_TYPES = (
    "clinical_note",
    "lab_report",
    "discharge_summary",
    "referral",
    "unknown",
)

# Terminal statuses.
STATUS_OK = "ok"
STATUS_NEEDS_REVIEW = "needs_review"

# Channels whose node updates accumulate instead of replacing what is there.
# LangGraph applies these reducers itself when it runs the graph; the streaming
# layer has to reproduce them by hand, because it merges the per-node updates as
# they arrive rather than waiting for the final state. Named here, next to the
# annotations they mirror, so the two cannot quietly drift apart — a test pins
# that this tuple matches the reducers declared on ``DocState``.
ADDITIVE_CHANNELS = ("trace",)


class DocState(TypedDict, total=False):
    # --- input ---
    doc_id: str
    raw_text: str
    # --- classify ---
    doc_type: str
    classification_confidence: float
    # --- extract ---
    fields: dict[str, Any]
    attempts: int
    # How many extraction passes this document has earned, derived from the
    # classification confidence at classify time (see agents.attempt_budget).
    retry_budget: int
    # --- validate ---
    errors: list[str]
    warnings: list[str]
    # --- summarize / output ---
    summary: str
    status: str
    # --- audit log (appended to by every node) ---
    trace: Annotated[list[str], operator.add]
