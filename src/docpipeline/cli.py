"""Command-line runner for the document pipeline.

docpipeline --samples            # run all bundled sample documents
docpipeline --file note.txt      # run one document from a file
cat note.txt | docpipeline       # or pipe a document on stdin
docpipeline --samples --json     # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .engine import get_engine
from .graph import run_document
from .samples import load_samples


def _render(result: dict[str, Any]) -> str:
    lines = [
        f"── {result.get('doc_id', 'doc')} — type={result.get('doc_type')} "
        f"status={result.get('status')}"
    ]
    if fields := result.get("fields"):
        lines.append("  fields:")
        lines.extend(f"    {k}: {v}" for k, v in fields.items())
    if errors := result.get("errors"):
        lines.append(f"  errors: {errors}")
    if warnings := result.get("warnings"):
        lines.append(f"  warnings: {warnings}")
    lines.append(f"  summary: {result.get('summary')}")
    lines.append("  trace:")
    lines.extend(f"    • {step}" for step in result.get("trace", []))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="docpipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--samples", action="store_true", help="run all bundled sample documents")
    source.add_argument("--file", help="path to a document to process")
    parser.add_argument("--engine", default="rule", choices=["rule", "openai"])
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON output")
    args = parser.parse_args()

    engine = get_engine(args.engine)

    docs: list[dict[str, str]] = []
    if args.file:
        path = Path(args.file)
        docs.append({"id": path.stem, "text": path.read_text(encoding="utf-8")})
    elif args.samples:
        docs = load_samples()
    else:
        piped = sys.stdin.read()
        if not piped.strip():
            parser.error("provide --samples, --file PATH, or pipe a document on stdin")
        docs.append({"id": "stdin", "text": piped})

    results = [run_document(d["text"], doc_id=d["id"], engine=engine) for d in docs]

    if args.as_json:
        serializable = [{k: v for k, v in r.items() if k != "raw_text"} for r in results]
        print(json.dumps(serializable, indent=2, default=str))
    else:
        for result in results:
            print(_render(result))
            print()
        flagged = sum(1 for r in results if r.get("status") == "needs_review")
        print(f"Processed {len(results)} document(s); {flagged} flagged for review.")


if __name__ == "__main__":
    main()
