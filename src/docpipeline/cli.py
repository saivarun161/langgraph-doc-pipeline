"""Command-line runner for the document pipeline.

docpipeline --samples                 # run all bundled sample documents
docpipeline --file note.txt           # run one document from a file
docpipeline --dir ./inbox             # run every document in a directory
cat note.txt | docpipeline            # or pipe a document on stdin
docpipeline --samples --json          # machine-readable output
docpipeline --dir ./inbox --metrics --workers 8   # batch with aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .agents import DEFAULT_MAX_ATTEMPTS, DEFAULT_MIN_CONFIDENCE
from .batch import run_batch
from .engine import get_engine
from .samples import load_samples

# Extensions treated as documents when reading a directory.
DOC_SUFFIXES = (".txt", ".md")


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


def _read_directory(root: Path) -> list[dict[str, str]]:
    """Load every document file under ``root``, sorted for a reproducible order."""
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in DOC_SUFFIXES)
    return [
        {"id": str(p.relative_to(root).with_suffix("")), "text": p.read_text(encoding="utf-8")}
        for p in paths
    ]


def _collect_docs(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[dict[str, str]]:
    if args.file:
        path = Path(args.file)
        return [{"id": path.stem, "text": path.read_text(encoding="utf-8")}]
    if args.dir:
        root = Path(args.dir)
        if not root.is_dir():
            parser.error(f"--dir {args.dir!r} is not a directory")
        docs = _read_directory(root)
        if not docs:
            parser.error(f"no {'/'.join(DOC_SUFFIXES)} documents found under {args.dir!r}")
        return docs
    if args.samples:
        return load_samples()
    piped = sys.stdin.read()
    if not piped.strip():
        parser.error("provide --samples, --file PATH, --dir PATH, or pipe a document on stdin")
    return [{"id": "stdin", "text": piped}]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="docpipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--samples", action="store_true", help="run all bundled sample documents")
    source.add_argument("--file", help="path to a document to process")
    source.add_argument("--dir", help="directory of documents to process as a batch")
    parser.add_argument("--engine", default="rule", choices=["rule", "openai"])
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON output")
    parser.add_argument(
        "--metrics", action="store_true", help="report aggregate metrics for the run"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="documents to process concurrently (default: 1)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"ceiling on extraction passes per document (default: {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=(
            "classifications below this confidence skip extraction and are "
            f"flagged for review (default: {DEFAULT_MIN_CONFIDENCE})"
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be >= 1")

    docs = _collect_docs(args, parser)
    batch = run_batch(
        docs,
        engine=get_engine(args.engine),
        max_attempts=args.max_attempts,
        min_confidence=args.min_confidence,
        workers=args.workers,
    )
    results = batch.results

    if args.as_json:
        serializable = [{k: v for k, v in r.items() if k != "raw_text"} for r in results]
        # Without --metrics the payload stays a bare list, so existing consumers
        # that index into it keep working.
        payload: Any = (
            {"results": serializable, "metrics": batch.metrics.as_dict()}
            if args.metrics
            else serializable
        )
        print(json.dumps(payload, indent=2, default=str))
    else:
        for result in results:
            print(_render(result))
            print()
        flagged = sum(1 for r in results if r.get("status") == "needs_review")
        print(f"Processed {len(results)} document(s); {flagged} flagged for review.")
        if args.metrics:
            print()
            print(batch.metrics.render())


if __name__ == "__main__":
    main()
