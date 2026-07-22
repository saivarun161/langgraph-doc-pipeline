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


def test_cli_metrics_includes_the_per_type_breakdown(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples", "--metrics"])
    out = capsys.readouterr().out
    assert "── by type" in out
    assert "clinical_note" in out
    assert "recovered=" in out


def test_cli_json_metrics_carries_per_type(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples", "--json", "--metrics"])
    payload = json.loads(capsys.readouterr().out)
    per_type = payload["metrics"]["per_type"]
    assert sum(s["documents"] for s in per_type.values()) == 6
    assert per_type["clinical_note"]["documents"] == 2


# --------------------------------------------------------------------------- #
# Per-type --min-confidence
# --------------------------------------------------------------------------- #


def test_cli_min_confidence_accepts_a_per_type_override(monkeypatch, capsys):
    # Gate clinical notes only; every other type keeps the permissive default.
    run_cli(
        monkeypatch,
        [
            "--samples",
            "--json",
            "--metrics",
            "--min-confidence",
            "0.3",
            "--min-confidence",
            "clinical_note=1.01",
        ],
    )
    per_type = json.loads(capsys.readouterr().out)["metrics"]["per_type"]
    assert per_type["clinical_note"]["needs_review"] == per_type["clinical_note"]["documents"]
    assert per_type["clinical_note"]["skipped_extraction"] == 2
    assert per_type["lab_report"]["needs_review"] == 0
    assert per_type["referral"]["needs_review"] == 0


def test_cli_bare_min_confidence_still_applies_to_everything(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples", "--json", "--metrics", "--min-confidence", "1.01"])
    metrics = json.loads(capsys.readouterr().out)["metrics"]
    assert metrics["skipped_extraction"] == 6


def test_cli_last_min_confidence_for_a_type_wins(monkeypatch, capsys):
    run_cli(
        monkeypatch,
        [
            "--samples",
            "--json",
            "--metrics",
            "--min-confidence",
            "clinical_note=1.01",
            "--min-confidence",
            "clinical_note=0.1",
        ],
    )
    per_type = json.loads(capsys.readouterr().out)["metrics"]["per_type"]
    assert per_type["clinical_note"]["skipped_extraction"] == 0


@pytest.mark.parametrize(
    "bad", ["abc", "clinical_note=abc", "clinical_notes=0.5", "unknown=0.5", "-0.5"]
)
def test_cli_rejects_malformed_min_confidence(monkeypatch, capsys, bad):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, ["--samples", "--min-confidence", bad])
    assert "--min-confidence" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# --calibrate
# --------------------------------------------------------------------------- #


def test_cli_calibrate_reports_when_there_are_too_few_samples(monkeypatch, capsys):
    # The 6 bundled documents span 5 types, so no type clears the sample floor.
    run_cli(monkeypatch, ["--samples", "--calibrate"])
    out = capsys.readouterr().out
    assert "── calibration" in out
    assert "needed to fit a threshold" in out


def test_cli_calibrate_fits_a_type_with_enough_documents(monkeypatch, capsys, tmp_path):
    for i in range(4):
        (tmp_path / f"note{i}.txt").write_text(
            f"Patient Name: Sam Roe {i}\nChief Complaint: cough\nAssessment: bronchitis\nPlan: rest"
        )
    run_cli(monkeypatch, ["--dir", str(tmp_path), "--calibrate", "--json"])
    calibration = json.loads(capsys.readouterr().out)["calibration"]
    assert calibration["clinical_note"]["documents"] == 4
    assert calibration["clinical_note"]["ok"] == 4
    # All four are clean at the same confidence, so nothing beats the status quo.
    assert calibration["clinical_note"]["changed"] is False


def test_cli_calibrate_reports_the_policy_actually_in_force(monkeypatch, capsys, tmp_path):
    for i in range(4):
        (tmp_path / f"note{i}.txt").write_text(
            f"Patient Name: Sam Roe {i}\nChief Complaint: cough\nAssessment: bronchitis\nPlan: rest"
        )
    run_cli(
        monkeypatch,
        ["--dir", str(tmp_path), "--calibrate", "--json", "--min-confidence", "clinical_note=0.6"],
    )
    calibration = json.loads(capsys.readouterr().out)["calibration"]
    assert calibration["clinical_note"]["current"] == 0.6


def test_cli_json_stays_a_bare_list_without_metrics_or_calibration(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples", "--json"])
    assert isinstance(json.loads(capsys.readouterr().out), list)


def test_cli_calibrate_alone_wraps_results_without_a_metrics_key(monkeypatch, capsys):
    run_cli(monkeypatch, ["--samples", "--json", "--calibrate"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["results"]) == 6
    assert "calibration" in payload
    assert "metrics" not in payload
