from docpipeline.agents import (
    REQUIRED_FIELDS,
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
    assert route_after_classify({"doc_type": "unknown"}) == "summarize"
    assert route_after_classify({"doc_type": "lab_report"}) == "extract"


def test_route_after_validate_retries_then_gives_up():
    # errors + attempts remaining -> retry
    assert route_after_validate({"errors": ["e"], "attempts": 1}, max_attempts=2) == "extract"
    # errors but attempts exhausted -> stop
    assert route_after_validate({"errors": ["e"], "attempts": 2}, max_attempts=2) == "summarize"
    # no errors -> stop
    assert route_after_validate({"errors": [], "attempts": 1}, max_attempts=2) == "summarize"


def test_required_fields_cover_all_non_unknown_types():
    for doc_type, required in REQUIRED_FIELDS.items():
        if doc_type != "unknown":
            assert "patient_name" in required
