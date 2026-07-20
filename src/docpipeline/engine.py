"""Pluggable reasoning engines.

The graph never talks to a model directly — it calls an ``Engine`` with three
capabilities: classify a document, extract its fields, and summarize the result.
Two implementations ship:

* ``RuleBasedEngine`` — deterministic keyword/regex logic, zero dependencies and
  zero credentials. It is the default so the whole pipeline (and its test suite)
  runs offline. It is intentionally not "smart"; it is predictable, which is what
  makes the graph's orchestration testable.
* ``OpenAIEngine`` — backs the same interface with an LLM for production use.

Both return plain data; all validation and routing live in the graph, not here.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from .state import DOC_TYPES


@runtime_checkable
class Engine(Protocol):
    id: str

    def classify(self, text: str) -> tuple[str, float]:
        """Return (doc_type, confidence in [0, 1])."""

    def extract(self, text: str, doc_type: str, attempt: int) -> dict[str, Any]:
        """Return a dict of extracted fields. ``attempt`` starts at 1; a value >= 2
        signals a retry, on which the engine may use more lenient heuristics."""

    def summarize(self, doc_type: str, fields: dict[str, Any]) -> str:
        """Return a short human-readable summary of the extracted fields."""


# --------------------------------------------------------------------------- #
# Rule-based engine
# --------------------------------------------------------------------------- #

# Keyword signatures for classification. Counted case-insensitively; the type
# with the most hits wins, and confidence is its share of all hits.
_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "discharge_summary": ("discharge", "admission date", "disposition", "hospital course"),
    "lab_report": ("reference range", "specimen", "hemoglobin", "wbc", "glucose", "result"),
    "referral": ("referral", "referred to", "refer to", "consult"),
    "clinical_note": ("chief complaint", "assessment", "plan", "history of present illness"),
}

# Shared demographic patterns.
_PATIENT = re.compile(r"(?:patient name|patient|name)\s*:\s*([A-Za-z][A-Za-z .,'-]+)", re.I)
_MRN = re.compile(r"\bMRN\s*:?\s*([A-Za-z0-9-]{3,})", re.I)
_DOB = re.compile(r"\bDOB\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.I)


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1).strip().rstrip(".") if m else None


def _line_value(label: str, text: str, *, aliases: tuple[str, ...] = ()) -> str | None:
    """Grab the rest of the line after ``Label:`` (or one of its aliases).

    Anchored to the start of a line (MULTILINE) so a short alias like "CC" can't
    match the middle of an unrelated word.
    """
    labels = "|".join(re.escape(x) for x in (label, *aliases))
    m = re.search(rf"^\s*(?:{labels})\s*:\s*(.+)", text, re.I | re.M)
    return m.group(1).strip().rstrip(".") if m else None


class RuleBasedEngine:
    """Deterministic, dependency-free engine. Predictable by design."""

    id = "rule-based-v1"

    def classify(self, text: str) -> tuple[str, float]:
        low = text.lower()
        scores = {
            doc_type: sum(low.count(kw) for kw in kws) for doc_type, kws in _TYPE_KEYWORDS.items()
        }
        best = max(scores, key=lambda k: scores[k])
        total = sum(scores.values())
        if total == 0 or scores[best] == 0:
            return "unknown", 0.0
        return best, round(scores[best] / total, 2)

    def extract(self, text: str, doc_type: str, attempt: int) -> dict[str, Any]:
        loose = attempt >= 2
        fields: dict[str, Any] = {}
        name = _first(_PATIENT, text)
        if name:
            fields["patient_name"] = name
        if mrn := _first(_MRN, text):
            fields["mrn"] = mrn
        if dob := _first(_DOB, text):
            fields["dob"] = dob

        if doc_type == "clinical_note":
            # On a retry, also accept the "CC:" abbreviation for chief complaint.
            cc_aliases = ("cc",) if loose else ()
            if cc := _line_value("chief complaint", text, aliases=cc_aliases):
                fields["chief_complaint"] = cc
            if a := _line_value("assessment", text, aliases=("impression",) if loose else ()):
                fields["assessment"] = a
            if p := _line_value("plan", text):
                fields["plan"] = p

        elif doc_type == "lab_report":
            results: dict[str, str] = {}
            for analyte in ("hemoglobin", "wbc", "platelets", "glucose", "sodium", "potassium"):
                m = re.search(rf"\b{analyte}\b\s*:?\s*([\d.]+)\s*([A-Za-z/%^0-9]+)?", text, re.I)
                if m:
                    unit = f" {m.group(2)}" if m.group(2) else ""
                    results[analyte] = f"{m.group(1)}{unit}".strip()
            if results:
                fields["results"] = results

        elif doc_type == "discharge_summary":
            if ad := _line_value("admission date", text):
                fields["admission_date"] = ad
            if dd := _line_value("discharge date", text):
                fields["discharge_date"] = dd
            if dx := _line_value("discharge diagnosis", text, aliases=("diagnosis",)):
                fields["diagnosis"] = dx
            if disp := _line_value("disposition", text):
                fields["disposition"] = disp

        elif doc_type == "referral":
            if to := _line_value("referred to", text, aliases=("refer to", "referral to")):
                fields["referred_to"] = to
            if reason := _line_value("reason for referral", text, aliases=("reason",)):
                fields["reason"] = reason

        return fields

    def summarize(self, doc_type: str, fields: dict[str, Any]) -> str:
        who = fields.get("patient_name", "Unknown patient")
        pretty = doc_type.replace("_", " ")
        if doc_type == "clinical_note":
            cc = fields.get("chief_complaint", "unspecified complaint")
            return f"{pretty.capitalize()} for {who}: presents with {cc}."
        if doc_type == "lab_report":
            n = len(fields.get("results", {}))
            return f"{pretty.capitalize()} for {who}: {n} analyte result(s) captured."
        if doc_type == "discharge_summary":
            dx = fields.get("diagnosis", "unspecified diagnosis")
            return f"{pretty.capitalize()} for {who}: discharged with {dx}."
        if doc_type == "referral":
            to = fields.get("referred_to", "an unspecified service")
            return f"Referral for {who} to {to}."
        return f"Unclassified document for {who}; no structured summary available."


# --------------------------------------------------------------------------- #
# OpenAI engine (optional)
# --------------------------------------------------------------------------- #


class OpenAIEngine:
    """LLM-backed engine using LangChain's chat model interface.

    Requires the ``openai`` extra and an OPENAI_API_KEY. The prompts ask for the
    same shapes the rule-based engine returns, so the graph is identical either
    way — only the reasoning quality changes.
    """

    id = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise RuntimeError(
                "OpenAI support is not installed. Install it with: "
                "pip install 'langgraph-doc-pipeline[openai]'"
            ) from exc
        self._model = ChatOpenAI(model=model, temperature=0)
        self.id = f"openai:{model}"

    def _ask(self, system: str, user: str) -> str:
        return self._model.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ).content.strip()

    def classify(self, text: str) -> tuple[str, float]:
        options = ", ".join(DOC_TYPES)
        raw = self._ask(
            f"Classify the clinical document into exactly one of: {options}. "
            "Reply with only the label.",
            text,
        ).lower()
        for doc_type in DOC_TYPES:
            if doc_type in raw:
                return doc_type, 0.9 if doc_type != "unknown" else 0.0
        return "unknown", 0.0

    def extract(self, text: str, doc_type: str, attempt: int) -> dict[str, Any]:
        import json

        raw = self._ask(
            f"Extract the key fields of this {doc_type.replace('_', ' ')} as flat JSON. "
            "Use snake_case keys. Reply with only JSON.",
            text,
        )
        try:
            data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
            return data if isinstance(data, dict) else {}
        except (ValueError, json.JSONDecodeError):
            return {}

    def summarize(self, doc_type: str, fields: dict[str, Any]) -> str:
        import json

        return self._ask(
            "Summarize the extracted fields in one sentence for a clinician.",
            json.dumps({"doc_type": doc_type, "fields": fields}),
        )


def get_engine(name: str = "rule") -> Engine:
    """Build an engine by short name: ``rule`` (default) or ``openai``."""
    name = name.strip().lower()
    if name in ("rule", "rule-based", "rulebased"):
        return RuleBasedEngine()
    if name == "openai":
        return OpenAIEngine()
    raise ValueError(f"Unknown engine {name!r} (expected 'rule' or 'openai')")
