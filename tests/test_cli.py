import json
import sys

import pytest

from docpipeline import cli


def run_cli(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["docpipeline", *args])
    cli.main()


def write_docs(tmp_path):
    """A small on-disk corpus: two recognizable documents and one that is not."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "note.txt").write_text(
        "Patient Name: Sam Roe\nChief Complaint: cough\nAssessment: bronchitis\nPlan: rest"
    )
    (tmp_path / "nested" / "ref.md").write_text(
        "Referred To: Cardiology\nReason for Referral: afib\nPatient Name: D O"
    )
    (tmp_path / "misc.txt").write_text("logistics and shipping notes")
    (tmp_path / "ignored.pdf").write_bytes(b"%PDF-not a document we read")
    return tmp_path


def test_cli_samples_human_output(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples"])
    out = capsys.readouterr().out
    assert "Processed 6 document(s)" in out
    assert "flagged for review" in out
    assert "clinical_note" in out


def test_cli_samples_json(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 6
    assert {d["doc_type"] for d in data} >= {"clinical_note", "lab_report", "referral"}
    assert all("raw_text" not in d for d in data)


def test_cli_file(monkeypatch, capsys, tmp_path):
    doc = tmp_path / "note.txt"
    doc.write_text("Referred To: Cardiology\nReason for Referral: afib\nPatient Name: D O")
    run_cli(monkeypatch, ["--file", str(doc)])
    out = capsys.readouterr().out
    assert "referral" in out
    assert "Cardiology" in out


def test_cli_dir_walks_recursively_and_skips_other_extensions(monkeypatch, capsys, tmp_path):
    root = write_docs(tmp_path)
    run_cli(monkeypatch, ["--dir", str(root), "--json"])
    data = json.loads(capsys.readouterr().out)
    # note.txt, misc.txt, nested/ref.md — but not ignored.pdf
    assert [d["doc_id"] for d in data] == ["misc", "nested/ref", "note"]
    assert {d["doc_type"] for d in data} == {"clinical_note", "referral", "unknown"}


def test_cli_dir_rejects_a_missing_directory(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, ["--dir", str(tmp_path / "nope")])


def test_cli_dir_rejects_a_directory_with_no_documents(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, ["--dir", str(tmp_path)])


def test_cli_metrics_block_is_appended_to_human_output(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples", "--metrics"])
    out = capsys.readouterr().out
    assert "Processed 6 document(s)" in out  # per-document output is unchanged
    assert "── batch — 6 document(s)" in out
    assert "review rate" in out


def test_cli_json_stays_a_bare_list_without_metrics(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples", "--json"])
    assert isinstance(json.loads(capsys.readouterr().out), list)


def test_cli_json_with_metrics_wraps_results(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples", "--json", "--metrics"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["results"]) == 6
    assert payload["metrics"]["documents"] == 6
    assert payload["metrics"]["ok"] + payload["metrics"]["needs_review"] == 6


def test_cli_workers_does_not_change_results(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples", "--json"])
    sequential = json.loads(capsys.readouterr().out)
    run_cli(monkeypatch, ["--samples", "--json", "--workers", "4"])
    parallel = json.loads(capsys.readouterr().out)
    assert [d["doc_id"] for d in parallel] == [d["doc_id"] for d in sequential]
    assert [d["status"] for d in parallel] == [d["status"] for d in sequential]


def test_cli_min_confidence_flag_forces_review(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples", "--json", "--metrics", "--min-confidence", "1.01"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["needs_review"] == 6
    assert payload["metrics"]["skipped_extraction"] == 6


@pytest.mark.parametrize("bad", [["--workers", "0"], ["--max-attempts", "0"]])
def test_cli_rejects_nonsense_numeric_flags(monkeypatch, bad):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, ["--samples", *bad])
