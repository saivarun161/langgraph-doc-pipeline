from docpipeline.engine import RuleBasedEngine, get_engine


def test_classify_recognizes_each_type():
    engine = RuleBasedEngine()
    assert engine.classify("Chief Complaint: cough\nAssessment: x\nPlan: y")[0] == "clinical_note"
    assert engine.classify("Hemoglobin: 13 g/dL\nreference range 13-17")[0] == "lab_report"
    assert engine.classify("Admission Date: 1/1\nDisposition: home")[0] == "discharge_summary"
    assert engine.classify("Referred To: cardiology\nreferral")[0] == "referral"


def test_classify_unknown_and_confidence_bounds():
    engine = RuleBasedEngine()
    doc_type, confidence = engine.classify("the quarterly logistics meeting notes")
    assert doc_type == "unknown"
    assert confidence == 0.0

    doc_type, confidence = engine.classify("Chief Complaint: pain\nAssessment: a\nPlan: b")
    assert 0.0 < confidence <= 1.0


def test_extract_pulls_shared_demographics():
    engine = RuleBasedEngine()
    fields = engine.extract(
        "Patient Name: Jane A. Carter\nMRN: A1042283\nDOB: 03/14/1978", "clinical_note", attempt=1
    )
    assert fields["patient_name"] == "Jane A. Carter"
    assert fields["mrn"] == "A1042283"
    assert fields["dob"] == "03/14/1978"


def test_extract_retry_recovers_abbreviated_label():
    engine = RuleBasedEngine()
    text = "Patient Name: A K\nCC: Shortness of breath\nAssessment: asthma\nPlan: inhaler"
    strict = engine.extract(text, "clinical_note", attempt=1)
    loose = engine.extract(text, "clinical_note", attempt=2)
    assert "chief_complaint" not in strict  # "CC:" not recognized on first pass
    assert loose["chief_complaint"] == "Shortness of breath"


def test_extract_lab_results_are_structured():
    engine = RuleBasedEngine()
    fields = engine.extract(
        "Patient Name: R L\nHemoglobin: 13.5 g/dL\nGlucose: 98 mg/dL", "lab_report", attempt=1
    )
    assert fields["results"]["hemoglobin"] == "13.5 g/dL"
    assert fields["results"]["glucose"] == "98 mg/dL"


def test_summaries_mention_patient_and_are_type_specific():
    engine = RuleBasedEngine()
    s = engine.summarize("referral", {"patient_name": "D O", "referred_to": "Cardiology"})
    assert "D O" in s and "Cardiology" in s


def test_get_engine_factory():
    assert isinstance(get_engine("rule"), RuleBasedEngine)
    assert get_engine("rule").id == "rule-based-v1"
    try:
        get_engine("nope")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown engine")
