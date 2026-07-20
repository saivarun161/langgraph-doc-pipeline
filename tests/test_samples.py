from docpipeline.samples import load_samples
from docpipeline.state import DOC_TYPES


def test_samples_have_required_keys_and_valid_labels():
    samples = load_samples()
    assert len(samples) >= 6
    for s in samples:
        assert s["id"] and s["text"].strip()
        assert s["expected_type"] in DOC_TYPES
        assert s["expected_status"] in ("ok", "needs_review")


def test_sample_ids_are_unique():
    ids = [s["id"] for s in load_samples()]
    assert len(ids) == len(set(ids))


def test_samples_cover_every_doc_type():
    seen = {s["expected_type"] for s in load_samples()}
    assert seen == set(DOC_TYPES)
