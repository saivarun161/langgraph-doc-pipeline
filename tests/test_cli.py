import json
import sys

from docpipeline import cli


def run_cli(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["docpipeline", *args])
    cli.main()


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
