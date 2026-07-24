"""Tests for the streaming view of a pipeline run.

The contract worth pinning is that streaming is a different *view* of the same
run, not a different run: the events arrive in execution order, and the state
they accumulate ends up identical to what ``run_document`` returns.
"""

import json
import operator
import typing
from itertools import pairwise

from docpipeline import stream_document
from docpipeline.graph import build_pipeline, run_document
from docpipeline.samples import load_samples
from docpipeline.state import ADDITIVE_CHANNELS, DocState
from docpipeline.stream import stream_pipeline

CLEAN_NOTE = "Chief Complaint: cough\nAssessment: bronchitis\nPlan: rest\nPatient Name: Sam Roe"
# The abbreviated note the strict first pass misses and the lenient retry recovers.
RETRY_NOTE = (
    "Patient Name: Aisha Khan\nMRN: A9981123\nDOB: 11/02/1990\n"
    "CC: Shortness of breath\nAssessment: asthma\nPlan: inhaler"
)


def test_clean_document_streams_one_event_per_node():
    events = list(stream_document(CLEAN_NOTE, doc_id="note"))
    assert [e.node for e in events] == ["classify", "extract", "validate", "summarize"]
    assert [e.step for e in events] == [1, 2, 3, 4]
    assert all(e.visit == 1 for e in events)
    assert all(e.doc_id == "note" for e in events)


def test_retry_loop_shows_up_as_a_second_visit_to_extract():
    events = list(stream_document(RETRY_NOTE))
    assert [e.node for e in events] == [
        "classify",
        "extract",
        "validate",
        "extract",
        "validate",
        "summarize",
    ]
    extracts = [e for e in events if e.node == "extract"]
    assert [e.visit for e in extracts] == [1, 2]
    # The self-correction is visible as it happens: the first validate reports an
    # error, and the retry that follows recovers the missed field.
    assert events[2].update["errors"] == ["missing required field: chief_complaint"]
    assert extracts[1].update["fields"]["chief_complaint"] == "Shortness of breath"
    assert events[-1].state["status"] == "ok"


def test_unknown_document_never_streams_an_extract_event():
    events = list(stream_document("random meeting notes about logistics and shipping"))
    assert [e.node for e in events] == ["classify", "summarize"]
    assert events[-1].state["status"] == "needs_review"


def test_low_confidence_document_skips_extraction_in_the_stream():
    events = list(stream_document(CLEAN_NOTE, min_confidence=1.01))
    assert [e.node for e in events] == ["classify", "summarize"]
    assert any("below threshold" in e for e in events[-1].state["errors"])


def test_final_state_matches_a_non_streamed_run_for_every_sample():
    for doc in load_samples():
        events = list(stream_document(doc["text"], doc_id=doc["id"]))
        assert events[-1].state == run_document(doc["text"], doc_id=doc["id"]), doc["id"]


def test_only_the_last_event_is_final():
    events = list(stream_document(RETRY_NOTE))
    assert [e.is_final for e in events] == [False] * (len(events) - 1) + [True]


def test_state_accumulates_rather_than_replacing():
    events = list(stream_document(RETRY_NOTE))
    traces = [e.state["trace"] for e in events]
    # Every snapshot extends the previous one, and carries the input forward.
    for earlier, later in pairwise(traces):
        assert later[: len(earlier)] == earlier
        assert len(later) > len(earlier)
    assert traces[-1] == events[-1].state["trace"]
    assert all(e.state["raw_text"] == RETRY_NOTE for e in events)
    # Last-write-wins channels replace: attempts counts up, it does not collect.
    assert [e.state["attempts"] for e in events] == [0, 1, 1, 2, 2, 2]


def test_snapshots_are_not_mutated_by_later_events():
    events = list(stream_document(RETRY_NOTE))
    first_extract = next(e for e in events if e.node == "extract")
    # The snapshot taken at the first extraction still shows one attempt and the
    # fields as they were then, even though a retry overwrote both afterwards.
    assert first_extract.state["attempts"] == 1
    assert "chief_complaint" not in first_extract.state["fields"]


def test_elapsed_never_goes_backwards():
    events = list(stream_document(RETRY_NOTE))
    assert all(e.elapsed >= 0 for e in events)
    assert [e.elapsed for e in events] == sorted(e.elapsed for e in events)


def test_render_timestamps_each_trace_line():
    events = list(stream_document(CLEAN_NOTE))
    rendered = events[0].render()
    assert "classify → clinical_note" in rendered
    assert "s]" in rendered
    # The retry pass appends exactly one line, so it renders as exactly one line.
    assert len(events[1].render().splitlines()) == 1


def test_as_dict_is_json_serializable_and_drops_the_raw_document():
    event = list(stream_document(CLEAN_NOTE, doc_id="note"))[-1]
    payload = json.loads(json.dumps(event.as_dict(), default=str))
    assert payload["doc_id"] == "note"
    assert payload["node"] == "summarize"
    assert payload["is_final"] is True
    assert payload["state"]["status"] == "ok"
    assert "raw_text" not in payload["state"]
    assert payload["lines"] == ["summarize → status=ok"]


def test_a_compiled_pipeline_can_be_reused_across_documents():
    pipeline = build_pipeline()
    first = list(stream_pipeline(pipeline, CLEAN_NOTE, "a"))
    second = list(stream_pipeline(pipeline, RETRY_NOTE, "b"))
    # Neither run leaks into the other: step and visit counters restart, and the
    # traces stay separate.
    assert [e.step for e in first] == [1, 2, 3, 4]
    assert second[0].step == 1
    assert second[-1].state["doc_id"] == "b"
    assert len(second[-1].state["trace"]) == 6


def test_tuning_arguments_reach_the_streamed_pipeline():
    events = list(stream_document(RETRY_NOTE, max_attempts=1))
    # One pass only: the lenient retry is never earned, so the miss stands.
    assert [e.node for e in events] == ["classify", "extract", "validate", "summarize"]
    assert events[-1].state["status"] == "needs_review"


def test_additive_channels_match_the_reducers_declared_on_doc_state():
    """The streaming merge mirrors DocState's reducers by hand, so pin the mirror.

    If a channel gains an additive reducer and ``ADDITIVE_CHANNELS`` is not
    updated with it, streamed snapshots would silently start dropping history.
    """
    declared = {
        name
        for name, hint in typing.get_type_hints(DocState, include_extras=True).items()
        if operator.add in getattr(hint, "__metadata__", ())
    }
    assert declared == set(ADDITIVE_CHANNELS)
