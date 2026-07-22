"""Tests for the batch runner and its metrics."""

import pytest

from docpipeline import run_batch
from docpipeline.batch import (
    BatchMetrics,
    calibrate_thresholds,
    render_calibration,
    summarize_batch,
)
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


# --------------------------------------------------------------------------- #
# Per-type breakdown
# --------------------------------------------------------------------------- #


def test_per_type_slices_agree_with_the_batch_totals():
    m = run_batch(load_samples()).metrics
    assert sum(s.documents for s in m.per_type.values()) == m.documents
    assert sum(s.ok for s in m.per_type.values()) == m.ok
    assert sum(s.needs_review for s in m.per_type.values()) == m.needs_review
    assert sum(s.retried for s in m.per_type.values()) == m.retried
    assert sum(s.skipped_extraction for s in m.per_type.values()) == m.skipped_extraction


def test_by_type_stays_a_plain_count_map():
    m = run_batch([CLEAN_NOTE, ABBREV_NOTE, GIBBERISH]).metrics
    assert m.by_type == {"clinical_note": 2, "unknown": 1}


def test_per_type_isolates_the_failing_type():
    # Two healthy clinical notes and a lab report that cannot pass validation:
    # the batch review rate is 33%, but it is entirely one type's problem.
    bad_lab = "Specimen: blood\nCollected: 01/02/2024"
    m = run_batch([CLEAN_NOTE, ABBREV_NOTE, bad_lab]).metrics
    assert m.per_type["clinical_note"].review_rate == 0.0
    assert m.per_type["lab_report"].review_rate == 1.0
    assert m.per_type["clinical_note"].recovered_by_retry == 1


def test_per_type_keys_are_sorted_for_stable_output():
    m = run_batch([GIBBERISH, CLEAN_NOTE]).metrics
    assert list(m.per_type) == sorted(m.per_type)


def test_per_type_survives_the_json_round_trip():
    import json

    data = json.loads(json.dumps(run_batch(load_samples()).metrics.as_dict()))
    assert data["per_type"]["clinical_note"]["documents"] == 2
    assert 0.0 <= data["per_type"]["clinical_note"]["review_rate"] <= 1.0
    assert data["by_type"] == {k: v["documents"] for k, v in data["per_type"].items()}


def test_render_types_lists_every_type():
    text = run_batch(load_samples()).metrics.render_types()
    assert "── by type" in text
    for doc_type in run_batch(load_samples()).metrics.per_type:
        assert doc_type in text


def test_render_types_handles_an_empty_batch():
    assert "no documents" in run_batch([]).metrics.render_types()


# --------------------------------------------------------------------------- #
# Threshold calibration
# --------------------------------------------------------------------------- #


def fake_results(doc_type, pairs):
    """Minimal result dicts: (confidence, status) pairs for one type."""
    return [
        {"doc_type": doc_type, "classification_confidence": c, "status": s, "attempts": 1}
        for c, s in pairs
    ]


def test_calibration_finds_the_separating_threshold():
    # Everything at 0.8 came out clean; everything at 0.4 went to review. The
    # threshold that routes all four correctly sits at the lowest good score.
    results = fake_results(
        "lab_report",
        [(0.4, "needs_review"), (0.4, "needs_review"), (0.8, "ok"), (0.8, "ok")],
    )
    suggestion = calibrate_thresholds(results)["lab_report"]
    assert suggestion.threshold == 0.8
    assert suggestion.correct == 4
    assert suggestion.accuracy == 1.0
    assert suggestion.changed is True


def test_calibration_prefers_the_lowest_threshold_when_scores_tie():
    # Every document came out clean, so any threshold at or below the minimum
    # routes all of them correctly — take the least aggressive one.
    results = fake_results("referral", [(0.5, "ok"), (0.7, "ok"), (0.9, "ok")])
    assert calibrate_thresholds(results)["referral"].threshold == 0.5


def test_calibration_can_suggest_never_extracting_a_hopeless_type():
    # Nothing of this type ever finished clean, so the fitted threshold lands
    # just above the best score it ever produced.
    results = fake_results(
        "discharge_summary",
        [(0.5, "needs_review"), (0.6, "needs_review"), (0.7, "needs_review")],
    )
    suggestion = calibrate_thresholds(results)["discharge_summary"]
    assert suggestion.threshold == 0.71
    assert suggestion.ok == 0


def test_calibration_reports_the_threshold_currently_in_force():
    results = fake_results("lab_report", [(0.4, "ok"), (0.6, "ok"), (0.8, "ok")])
    policy = {"lab_report": 0.4, "default": 0.9}
    assert calibrate_thresholds(results, min_confidence=policy)["lab_report"].current == 0.4
    assert calibrate_thresholds(results)["lab_report"].current == 0.35  # the built-in default


def test_calibration_marks_an_unchanged_threshold():
    results = fake_results("lab_report", [(0.35, "ok"), (0.6, "ok"), (0.8, "ok")])
    assert calibrate_thresholds(results, min_confidence=0.35)["lab_report"].changed is False


def test_calibration_skips_types_with_too_few_samples():
    results = fake_results("referral", [(0.5, "ok"), (0.9, "ok")])
    assert calibrate_thresholds(results, min_samples=3) == {}
    assert "referral" in calibrate_thresholds(results, min_samples=2)


def test_calibration_never_fits_unknown():
    # 'unknown' documents never reach extraction, so there is no threshold to fit.
    results = fake_results("unknown", [(0.0, "needs_review")] * 5)
    assert calibrate_thresholds(results) == {}


def test_calibration_on_a_real_batch_stays_within_observed_confidences():
    results = run_batch(load_samples() * 2).results
    for doc_type, suggestion in calibrate_thresholds(results).items():
        observed = [r["classification_confidence"] for r in results if r["doc_type"] == doc_type]
        assert min(observed) <= suggestion.threshold <= max(observed) + 0.01
        assert suggestion.correct <= suggestion.documents


def test_calibration_suggestion_is_json_serializable():
    import json

    results = fake_results("lab_report", [(0.4, "needs_review"), (0.8, "ok"), (0.9, "ok")])
    data = json.loads(
        json.dumps({k: v.as_dict() for k, v in calibrate_thresholds(results).items()})
    )
    assert data["lab_report"]["threshold"] == 0.8
    assert data["lab_report"]["changed"] is True


def test_render_calibration_is_human_readable():
    results = fake_results("lab_report", [(0.4, "needs_review"), (0.8, "ok"), (0.9, "ok")])
    text = render_calibration(calibrate_thresholds(results))
    assert "lab_report" in text
    assert "0.35 → 0.80" in text
    assert "routed correctly" in text


def test_render_calibration_explains_an_empty_result():
    assert "no type had" in render_calibration({})
