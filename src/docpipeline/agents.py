"""The agent nodes and the routing logic that wires them together.

Each node is a small function ``(state, engine) -> partial state``. The engine is
bound at graph-build time (see ``graph.py``), so the nodes match LangGraph's
single-argument node signature while staying easy to unit-test in isolation.

Validation is deliberately *not* an engine call — checking that required fields
are present and well-formed is deterministic business logic, so it lives here as
plain code the tests can pin exactly.
"""

from __future__ import annotations

import re
from typing import Any

from .engine import Engine
from .state import STATUS_NEEDS_REVIEW, STATUS_OK, DocState

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
# Nodes
# --------------------------------------------------------------------------- #


def classify_node(state: DocState, engine: Engine) -> dict[str, Any]:
    doc_type, confidence = engine.classify(state["raw_text"])
    return {
        "doc_type": doc_type,
        "classification_confidence": confidence,
        "trace": [f"classify → {doc_type} (confidence {confidence:.2f})"],
    }


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
    """An unrecognized document has nothing to extract — send it straight to
    summarize, where it will be flagged for review."""
    return "summarize" if state.get("doc_type") == "unknown" else "extract"


def route_after_validate(state: DocState, max_attempts: int) -> str:
    """Loop back for another extraction pass while there are errors and attempts
    remain; otherwise finish. This is the self-correcting core of the pipeline."""
    if state.get("errors") and state.get("attempts", 0) < max_attempts:
        return "extract"
    return "summarize"
