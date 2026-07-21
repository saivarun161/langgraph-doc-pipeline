import pytest

from docpipeline.agents import (
    DEFAULT_MAX_ATTEMPTS,
    REQUIRED_FIELDS,
    attempt_budget,
    classify_node,
    extract_node,
    route_after_classify,
    route_after_validate,
    summarize_node,
    validate_node,
)
from docpipeline.engine import RuleBasedEngine

ENGINE = RuleBasedEngine()


def test_classify_node_records_trace():
    out = classify_node({"raw_text": "Chief Complaint: x\nAssessment: y\nPlan: z"}, ENGINE)
    assert out["doc_type"] == "clinical_note"
    assert out["trace"] and out["trace"][0].startswith("classify")


def test_classify_node_sets_retry_budget_from_confidence():
    out = classify_node({"raw_text": "Chief Complaint: x\nAssessment: y\nPlan: z"}, ENGINE)
    assert out["classification_confidence"] == 1.0
    assert out["retry_budget"] == DEFAULT_MAX_ATTEMPTS
    assert "budget" in out["trace"][0]
    assert "errors" not in out  # a confident classification is not an error


def test_classify_node_below_threshold_reports_an_error():
    class Unsure:
        id = "unsure"

        def classify(self, text):
            return "lab_report", 0.10

    out = classify_node({"raw_text": "anything"}, Unsure(), min_confidence=0.35)
    assert out["retry_budget"] == 0
    assert out["errors"] and "below threshold" in out["errors"][0]
    assert any("skipping extraction" in line for line in out["trace"])


def test_classify_node_unknown_is_not_double_reported():
    # "unknown" already routes to review on its own; it should not also acquire a
    # confidence error, which would be a second reason for the same thing.
    out = classify_node({"raw_text": "logistics and shipping notes"}, ENGINE)
    assert out["doc_type"] == "unknown"
    assert out["retry_budget"] == 0
    assert "errors" not in out


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.0, 0),  # no signal at all
        (0.20, 0),  # below the default threshold
        (0.35, 1),  # exactly at the threshold: one strict pass, no retry
        (0.50, 1),  # ceil(1.0) -> still a single pass
        (0.75, 2),  # ceil(1.5) -> earns the lenient retry
        (1.00, 2),  # capped by max_attempts
    ],
)
def test_attempt_budget_scales_with_confidence(confidence, expected):
    assert attempt_budget(confidence, max_attempts=2, min_confidence=0.35) == expected


def test_attempt_budget_is_monotonic_and_capped():
    budgets = [attempt_budget(c / 20, max_attempts=4) for c in range(21)]
    assert budgets == sorted(budgets)
    assert max(budgets) == 4


def test_extract_node_increments_attempts():
    state = {"raw_text": "Patient Name: A B", "doc_type": "clinical_note", "attempts": 0}
    first = extract_node(state, ENGINE)
    assert first["attempts"] == 1
    second = extract_node({**state, "attempts": 1}, ENGINE)
    assert second["attempts"] == 2
    assert "retry" in second["trace"][0]


def test_validate_flags_missing_required_fields():
    state = {"doc_type": "clinical_note", "fields": {"patient_name": "A B"}}
    out = validate_node(state)
    # chief_complaint, assessment, plan are missing
    assert len(out["errors"]) == 3
    assert all("missing required field" in e for e in out["errors"])


def test_validate_missing_demographics_are_warnings_not_errors():
    complete = {"patient_name": "A B", "chief_complaint": "c", "assessment": "a", "plan": "p"}
    out = validate_node({"doc_type": "clinical_note", "fields": complete})
    assert out["errors"] == []
    assert any("mrn" in w for w in out["warnings"])
    assert any("dob" in w for w in out["warnings"])


def test_validate_warns_on_malformed_dob():
    fields = {
        "patient_name": "A B",
        "chief_complaint": "c",
        "assessment": "a",
        "plan": "p",
        "mrn": "X1",
        "dob": "not-a-date",
    }
    out = validate_node({"doc_type": "clinical_note", "fields": fields})
    assert any("dob not a recognizable date" in w for w in out["warnings"])


def test_summarize_status_reflects_errors_and_unknown():
    ok = summarize_node({"doc_type": "referral", "fields": {"patient_name": "x"}}, ENGINE)
    assert ok["status"] == "ok"
    bad = summarize_node({"doc_type": "referral", "fields": {}, "errors": ["e"]}, ENGINE)
    assert bad["status"] == "needs_review"
    unknown = summarize_node({"doc_type": "unknown", "fields": {}}, ENGINE)
    assert unknown["status"] == "needs_review"


def test_route_after_classify():
    assert route_after_classify({"doc_type": "unknown", "retry_budget": 0}) == "summarize"
    assert route_after_classify({"doc_type": "lab_report", "retry_budget": 2}) == "extract"


def test_route_after_classify_skips_extraction_without_a_budget():
    # Recognized type, but the classification was too weak to spend a pass on.
    assert route_after_classify({"doc_type": "lab_report", "retry_budget": 0}) == "summarize"


def test_route_after_validate_retries_then_gives_up():
    # errors + budget remaining -> retry
    assert route_after_validate({"errors": ["e"], "attempts": 1, "retry_budget": 2}) == "extract"
    # errors but budget exhausted -> stop
    assert route_after_validate({"errors": ["e"], "attempts": 2, "retry_budget": 2}) == "summarize"
    # no errors -> stop
    assert route_after_validate({"errors": [], "attempts": 1, "retry_budget": 2}) == "summarize"


def test_route_after_validate_respects_a_one_pass_budget():
    # A 0.5-confidence document gets one strict pass and no lenient retry, even
    # though the global ceiling would have allowed two.
    assert route_after_validate({"errors": ["e"], "attempts": 1, "retry_budget": 1}) == "summarize"


def test_required_fields_cover_all_non_unknown_types():
    for doc_type, required in REQUIRED_FIELDS.items():
        if doc_type != "unknown":
            assert "patient_name" in required
