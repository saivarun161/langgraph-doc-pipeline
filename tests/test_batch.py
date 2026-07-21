"""Tests for the batch runner and its metrics."""

import pytest

from docpipeline import run_batch
from docpipeline.batch import BatchMetrics, summarize_batch
from docpipeline.samples import load_samples

CLEAN_NOTE = (
    "Patient Name: Sam Roe\nMRN: X1\nDOB: 01/02/1990\n"
    "Chief Complaint: cough\nAssessment: bronchitis\nPlan: rest"
)
ABBREV_NOTE = (
    "Patient Name: Aisha Khan\nMRN: A9981123\nDOB: 11/02/1990\n"
    "CC: Shortness of breath\nAssessment: asthma\nPlan: inhaler"
)
GIBBERISH = "random meeting notes about logistics and shipping"


def test_batch_preserves_input_order_and_ids():
    batch = run_batch([{"id": "a", "text": CLEAN_NOTE}, {"id": "b", "text": GIBBERISH}])
    assert [r["doc_id"] for r in batch.results] == ["a", "b"]
    assert batch.results[0]["doc_type"] == "clinical_note"
    assert batch.results[1]["doc_type"] == "unknown"


def test_batch_accepts_raw_strings_and_generates_ids():
    batch = run_batch([CLEAN_NOTE, GIBBERISH])
    assert [r["doc_id"] for r in batch.results] == ["doc-1", "doc-2"]


def test_batch_rejects_documents_without_text():
    with pytest.raises(ValueError, match="no 'text' key"):
        run_batch([{"id": "a"}])


def test_batch_result_is_iterable_and_sized():
    batch = run_batch([CLEAN_NOTE, GIBBERISH])
    assert len(batch) == 2
    assert [r["doc_id"] for r in batch] == ["doc-1", "doc-2"]


def test_metrics_count_statuses_types_and_retries():
    batch = run_batch([CLEAN_NOTE, ABBREV_NOTE, GIBBERISH])
    m = batch.metrics
    assert m.documents == 3
    assert m.ok == 2
    assert m.needs_review == 1
    assert m.by_type == {"clinical_note": 2, "unknown": 1}
    assert m.retried == 1  # only the CC: note needed a second pass
    assert m.recovered_by_retry == 1  # ...and that pass rescued it
    assert m.skipped_extraction == 1  # the unknown document never extracted
    assert m.review_rate == pytest.approx(1 / 3)


def test_metrics_separate_retried_from_recovered():
    # A note with no Plan at all: it retries, and the retry cannot save it.
    unrecoverable = "Chief Complaint: cough\nAssessment: bronchitis\nPatient Name: Sam Roe"
    m = run_batch([unrecoverable]).metrics
    assert m.retried == 1
    assert m.recovered_by_retry == 0
    assert m.needs_review == 1


def test_field_completeness_ignores_types_without_required_fields():
    # The unknown document declares no required fields; scoring it would hand the
    # batch a free 100%, so it is excluded from the average entirely.
    only_unknown = run_batch([GIBBERISH]).metrics
    assert only_unknown.field_completeness == 0.0

    partial = "Chief Complaint: cough\nAssessment: bronchitis\nPatient Name: Sam Roe"
    mixed = run_batch([CLEAN_NOTE, partial, GIBBERISH]).metrics
    # 4/4 required for the clean note, 3/4 for the one missing its plan.
    assert mixed.field_completeness == pytest.approx((1.0 + 0.75) / 2)


def test_metrics_mean_confidence_and_attempts():
    m = run_batch([CLEAN_NOTE, GIBBERISH]).metrics
    assert m.mean_confidence == pytest.approx(0.5)  # 1.00 and 0.00
    assert m.mean_attempts == pytest.approx(0.5)  # one pass and none


def test_low_confidence_threshold_shows_up_as_skipped_extraction():
    m = run_batch([CLEAN_NOTE, ABBREV_NOTE], min_confidence=1.01).metrics
    assert m.skipped_extraction == 2
    assert m.needs_review == 2
    assert m.field_completeness == 0.0


def test_workers_produce_the_same_results_as_sequential():
    docs = load_samples()
    sequential = run_batch(docs, workers=1)
    parallel = run_batch(docs, workers=4)
    assert [r["doc_id"] for r in parallel.results] == [r["doc_id"] for r in sequential.results]
    assert [r["status"] for r in parallel.results] == [r["status"] for r in sequential.results]
    # .get: a document that skipped extraction never sets the fields channel.
    assert [r.get("fields") for r in parallel.results] == [
        r.get("fields") for r in sequential.results
    ]
    assert parallel.metrics.as_dict()["by_type"] == sequential.metrics.as_dict()["by_type"]


def test_workers_must_be_positive():
    with pytest.raises(ValueError, match="workers must be >= 1"):
        run_batch([CLEAN_NOTE], workers=0)


def test_empty_batch_is_all_zeros_not_a_crash():
    m = run_batch([]).metrics
    assert m == BatchMetrics(wall_seconds=m.wall_seconds)
    assert m.documents == 0
    assert m.review_rate == 0.0
    assert m.docs_per_second == 0.0


def test_summarize_batch_works_on_externally_collected_results():
    results = run_batch([CLEAN_NOTE, GIBBERISH]).results
    only_notes = summarize_batch([r for r in results if r["doc_type"] == "clinical_note"])
    assert only_notes.documents == 1
    assert only_notes.ok == 1
    assert only_notes.by_type == {"clinical_note": 1}


def test_metrics_as_dict_is_json_serializable_with_derived_rates():
    import json

    data = json.loads(json.dumps(run_batch(load_samples()).metrics.as_dict()))
    assert data["documents"] == 6
    assert data["ok"] + data["needs_review"] == 6
    assert 0.0 <= data["review_rate"] <= 1.0
    assert data["docs_per_second"] > 0


def test_metrics_render_is_human_readable():
    text = run_batch(load_samples()).metrics.render()
    assert "6 document(s)" in text
    assert "needs_review=" in text
    assert "recovered by the retry loop" in text
