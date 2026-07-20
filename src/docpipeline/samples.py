"""Access to the bundled synthetic sample documents.

The samples are fictional and contain no real patient information. They exist so
the pipeline can be demonstrated and tested end-to-end with no external data.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

_DATA = files("docpipeline").joinpath("data", "sample_docs.jsonl")


def load_samples() -> list[dict[str, Any]]:
    """Return the bundled documents as dicts with id, text, and expected_* labels."""
    records: list[dict[str, Any]] = []
    for line in _DATA.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
