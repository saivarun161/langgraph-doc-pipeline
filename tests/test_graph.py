"""End-to-end tests that drive the compiled LangGraph pipeline."""

import pytest

from docpipeline.graph import build_pipeline, run_document
from docpipeline.samples import load_samples

CLEAN_NOTE = "Chief Complaint: cough\nAssessment: bronchitis\nPlan: rest\nPatient Name: Sam Roe"
CLEAN_LAB = "Patient Name: Sam Roe\nSpecimen: blood\nResults: Hemoglobin 12 g/dL"


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


def test_low_confidence_document_skips_extraction_entirely():
    # min_confidence of 1.01 makes every classification untrusted, so no document
    # reaches extraction no matter how well it matches.
    text = "Chief Complaint: cough\nAssessment: bronchitis\nPlan: rest\nPatient Name: Sam Roe"
    result = run_document(text, min_confidence=1.01)
    assert result["doc_type"] == "clinical_note"
    assert result.get("attempts", 0) == 0
    assert result["status"] == "needs_review"
    assert any("below threshold" in e for e in result["errors"])


def test_confident_document_still_extracts_under_the_default_threshold():
    text = "Chief Complaint: cough\nAssessment: bronchitis\nPlan: rest\nPatient Name: Sam Roe"
    result = run_document(text)
    assert result["retry_budget"] == 2
    assert result["status"] == "ok"


def test_retry_budget_scales_with_max_attempts():
    # ref-01 classifies at 0.75 confidence, so it earns ceil(0.75 * ceiling)
    # passes rather than the full ceiling.
    ref = next(d for d in load_samples() if d["id"] == "ref-01")
    assert run_document(ref["text"])["classification_confidence"] == 0.75
    assert run_document(ref["text"], max_attempts=4)["retry_budget"] == 3
    assert run_document(ref["text"], max_attempts=2)["retry_budget"] == 2


def test_per_type_threshold_gates_one_type_without_touching_another():
    # One policy, two documents that both classify confidently: the note's own
    # threshold is set above its score, the lab report's is not.
    policy = {"clinical_note": 1.01, "default": 0.35}
    note = run_document(CLEAN_NOTE, min_confidence=policy)
    lab = run_document(CLEAN_LAB, min_confidence=policy)

    assert note.get("attempts", 0) == 0
    assert note["status"] == "needs_review"
    assert any("for clinical_note" in e for e in note["errors"])

    assert lab["doc_type"] == "lab_report"
    assert lab["status"] == "ok"
    assert lab["attempts"] >= 1


def test_per_type_policy_falls_back_to_default_for_unnamed_types():
    # The policy names only clinical_note, so the lab report is judged by
    # "default" — set high enough here to stop it.
    lab = run_document(CLEAN_LAB, min_confidence={"clinical_note": 0.1, "default": 1.01})
    assert lab.get("attempts", 0) == 0
    assert lab["status"] == "needs_review"


def test_build_pipeline_rejects_a_bad_policy_before_running_anything():
    with pytest.raises(ValueError, match="unknown document type"):
        build_pipeline(min_confidence={"lab_reports": 0.5})


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
