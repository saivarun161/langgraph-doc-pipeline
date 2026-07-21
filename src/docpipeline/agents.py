"""The agent nodes and the routing logic that wires them together.

Each node is a small function ``(state, engine) -> partial state``. The engine is
bound at graph-build time (see ``graph.py``), so the nodes match LangGraph's
single-argument node signature while staying easy to unit-test in isolation.

Validation is deliberately *not* an engine call — checking that required fields
are present and well-formed is deterministic business logic, so it lives here as
plain code the tests can pin exactly.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .engine import Engine
from .state import STATUS_NEEDS_REVIEW, STATUS_OK, DocState

# Ceiling on extraction passes for any single document.
DEFAULT_MAX_ATTEMPTS = 2

# Classifications weaker than this are not trusted enough to extract against.
# The rule-based engine reports a type's share of all keyword hits, so a score
# below roughly a third means the runner-up types were nearly as plausible.
DEFAULT_MIN_CONFIDENCE = 0.35

# Fields every recognized type must carry to be considered complete. Demographic
# fields (mrn, dob) are treated as warnings, not errors — a document is still
# usable without them.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "clinical_note": ("patient_name", "chief_complaint", "assessment", "plan"),
    "lab_report": ("patient_name", "results"),
    "discharge_summary": ("patient_name", "admission_date", "discharge_date", "diagnosis"),
    "referral": ("patient_name", "referred_to", "reason"),
    "unknown": (),
}

_DATE_RE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")


# --------------------------------------------------------------------------- #
# Confidence-weighted retry budget
# --------------------------------------------------------------------------- #


def attempt_budget(
    confidence: float,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> int:
    """How many extraction passes a classification of this confidence has earned.

    The retry pass is deliberately *more lenient* than the first one, which is
    only a good trade when we are confident the document was typed correctly.
    Spending a lenient pass on a probably-misclassified document manufactures
    plausible-but-wrong fields — strictly worse than an honest ``needs_review``.
    So the budget scales with confidence:

    * below ``min_confidence`` → 0 passes; the document goes straight to review
    * otherwise → ``ceil(confidence * max_attempts)``, at least one pass

    With the default ceiling of 2, that means a document classified at 0.5 or
    below gets a single strict pass, and only a confident classification earns
    the lenient retry.
    """
    if confidence < min_confidence:
        return 0
    return max(1, math.ceil(confidence * max_attempts))


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def classify_node(
    state: DocState,
    engine: Engine,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    doc_type, confidence = engine.classify(state["raw_text"])
    budget = attempt_budget(confidence, max_attempts, min_confidence)
    out: dict[str, Any] = {
        "doc_type": doc_type,
        "classification_confidence": confidence,
        "retry_budget": budget,
        "trace": [f"classify → {doc_type} (confidence {confidence:.2f}, budget {budget})"],
    }
    # A recognized type we do not actually trust is an error, not a silent pass:
    # it carries the document to needs_review with the reason attached.
    if budget == 0 and doc_type != "unknown":
        out["errors"] = [
            f"classification confidence {confidence:.2f} below threshold {min_confidence:.2f}"
        ]
        out["trace"].append("classify → below confidence threshold, skipping extraction")
    return out


def extract_node(state: DocState, engine: Engine) -> dict[str, Any]:
    attempt = state.get("attempts", 0) + 1
    fields = engine.extract(state["raw_text"], state["doc_type"], attempt)
    label = "extract" if attempt == 1 else f"extract (retry {attempt - 1})"
    return {
        "fields": fields,
        "attempts": attempt,
        "trace": [f"{label} → {len(fields)} field(s): {', '.join(sorted(fields)) or 'none'}"],
    }


def validate_node(state: DocState) -> dict[str, Any]:
    doc_type = state.get("doc_type", "unknown")
    fields = state.get("fields", {})
    required = REQUIRED_FIELDS.get(doc_type, ())

    errors = [f"missing required field: {name}" for name in required if not fields.get(name)]

    warnings: list[str] = []
    for demographic in ("mrn", "dob"):
        if not fields.get(demographic):
            warnings.append(f"missing {demographic}")
    if (dob := fields.get("dob")) and not _DATE_RE.match(dob):
        warnings.append(f"dob not a recognizable date: {dob!r}")

    verdict = "clean" if not errors else f"{len(errors)} error(s)"
    return {
        "errors": errors,
        "warnings": warnings,
        "trace": [f"validate → {verdict}, {len(warnings)} warning(s)"],
    }


def summarize_node(state: DocState, engine: Engine) -> dict[str, Any]:
    doc_type = state.get("doc_type", "unknown")
    fields = state.get("fields", {})
    summary = engine.summarize(doc_type, fields)
    # Anything with outstanding validation errors, or that never classified,
    # is handed to a human rather than trusted downstream.
    needs_review = bool(state.get("errors")) or doc_type == "unknown"
    status = STATUS_NEEDS_REVIEW if needs_review else STATUS_OK
    return {
        "summary": summary,
        "status": status,
        "trace": [f"summarize → status={status}"],
    }


# --------------------------------------------------------------------------- #
# Routing (conditional edges)
# --------------------------------------------------------------------------- #


def route_after_classify(state: DocState) -> str:
    """Send a document to extraction only if it earned a pass.

    Two kinds of document skip straight to summarize (and therefore review): one
    that matched nothing (``unknown``), and one whose classification was too weak
    to trust — both arrive with a ``retry_budget`` of 0.
    """
    if state.get("doc_type") == "unknown" or state.get("retry_budget", 0) < 1:
        return "summarize"
    return "extract"


def route_after_validate(state: DocState) -> str:
    """Loop back for another extraction pass while there are errors and the
    document's confidence-weighted budget still has room; otherwise finish. This
    is the self-correcting core of the pipeline."""
    if state.get("errors") and state.get("attempts", 0) < state.get("retry_budget", 0):
        return "extract"
    return "summarize"
