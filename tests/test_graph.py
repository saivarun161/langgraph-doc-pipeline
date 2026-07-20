"""End-to-end tests that drive the compiled LangGraph pipeline."""

from docpipeline.graph import build_pipeline, run_document
from docpipeline.samples import load_samples


def test_all_samples_reach_expected_type_and_status():
    for doc in load_samples():
        result = run_document(doc["text"], doc_id=doc["id"])
        assert result["doc_type"] == doc["expected_type"], doc["id"]
        assert result["status"] == doc["expected_status"], doc["id"]


def test_retry_loop_recovers_and_records_two_attempts():
    text = (
        "Patient Name: Aisha Khan\nMRN: A9981123\nDOB: 11/02/1990\n"
        "CC: Shortness of breath\nAssessment: asthma\nPlan: inhaler"
    )
    result = run_document(text)
    assert result["attempts"] == 2  # first pass failed validation, retry recovered
    assert result["status"] == "ok"
    assert result["fields"]["chief_complaint"] == "Shortness of breath"


def test_unknown_skips_extraction():
    result = run_document("random meeting notes about logistics and shipping")
    assert result["doc_type"] == "unknown"
    assert result["status"] == "needs_review"
    assert result.get("attempts", 0) == 0
    assert result.get("fields", {}) == {}


def test_exhausted_retries_flag_for_review():
    # A clinical note missing its Plan entirely: no pass can recover it, so after
    # max_attempts the document is flagged rather than looping forever.
    text = "Chief Complaint: cough\nAssessment: bronchitis\nPatient Name: Sam Roe"
    result = run_document(text, max_attempts=2)
    assert result["doc_type"] == "clinical_note"
    assert result["attempts"] == 2  # tried the maximum, then stopped
    assert result["status"] == "needs_review"
    assert any("plan" in e for e in result["errors"])


def test_trace_accumulates_across_nodes():
    result = run_document(load_samples()[0]["text"])
    kinds = [step.split(" ")[0] for step in result["trace"]]
    assert kinds[0] == "classify"
    assert "extract" in kinds
    assert "validate" in kinds
    assert kinds[-1] == "summarize"


def test_pipeline_is_reusable_across_documents():
    pipeline = build_pipeline()
    a = pipeline.invoke({"raw_text": "Referred To: Cardiology\nReason: afib", "attempts": 0})
    b = pipeline.invoke({"raw_text": "Hemoglobin: 12 g/dL\nspecimen blood", "attempts": 0})
    assert a["doc_type"] == "referral"
    assert b["doc_type"] == "lab_report"
