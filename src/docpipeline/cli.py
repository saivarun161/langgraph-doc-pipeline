"""Command-line runner for the document pipeline.

docpipeline --samples                 # run all bundled sample documents
docpipeline --file note.txt           # run one document from a file
docpipeline --dir ./inbox             # run every document in a directory
cat note.txt | docpipeline            # or pipe a document on stdin
docpipeline --samples --json          # machine-readable output
docpipeline --dir ./inbox --metrics --workers 8   # batch with aggregate metrics
docpipeline --dir ./inbox --calibrate             # suggest per-type thresholds
docpipeline --dir ./inbox --min-confidence 0.3 --min-confidence lab_report=0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .agents import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MIN_CONFIDENCE,
    POLICY_DEFAULT_KEY,
    ConfidencePolicy,
    validate_confidence_policy,
)
from .batch import calibrate_thresholds, render_calibration, run_batch
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


def _parse_confidence_policy(
    values: list[str] | None, parser: argparse.ArgumentParser
) -> ConfidencePolicy:
    """Turn repeated ``--min-confidence`` values into a scalar or a per-type policy.

    A bare number sets the threshold for every type; ``TYPE=NUMBER`` sets it for
    one. Mixing them is the point — a default, plus the types that have earned
    something different. When only a bare number is given the result stays a
    plain float, so the simple case never pays for the general one.
    """
    if not values:
        return DEFAULT_MIN_CONFIDENCE

    policy: dict[str, float] = {}
    for raw in values:
        doc_type, _, number = raw.rpartition("=")
        try:
            policy[doc_type or POLICY_DEFAULT_KEY] = float(number)
        except ValueError:
            parser.error(f"--min-confidence {raw!r} is not a NUMBER or a TYPE=NUMBER pair")

    resolved: ConfidencePolicy = (
        policy[POLICY_DEFAULT_KEY] if set(policy) == {POLICY_DEFAULT_KEY} else policy
    )
    try:
        return validate_confidence_policy(resolved)
    except ValueError as exc:
        parser.error(f"--min-confidence: {exc}")


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
        "--calibrate",
        action="store_true",
        help="suggest a per-type --min-confidence fitted to this run's outcomes",
    )
    parser.add_argument(
        "--min-confidence",
        action="append",
        metavar="[TYPE=]VALUE",
        help=(
            "classifications below this confidence skip extraction and are "
            "flagged for review. Repeatable: a bare number sets the default, "
            "TYPE=NUMBER overrides one document type "
            f"(default: {DEFAULT_MIN_CONFIDENCE})"
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be >= 1")
    min_confidence = _parse_confidence_policy(args.min_confidence, parser)

    docs = _collect_docs(args, parser)
    batch = run_batch(
        docs,
        engine=get_engine(args.engine),
        max_attempts=args.max_attempts,
        min_confidence=min_confidence,
        workers=args.workers,
    )
    results = batch.results
    suggestions = calibrate_thresholds(results, min_confidence) if args.calibrate else {}

    if args.as_json:
        serializable = [{k: v for k, v in r.items() if k != "raw_text"} for r in results]
        # With neither --metrics nor --calibrate the payload stays a bare list,
        # so existing consumers that index into it keep working.
        payload: Any = serializable
        if args.metrics or args.calibrate:
            payload = {"results": serializable}
            if args.metrics:
                payload["metrics"] = batch.metrics.as_dict()
            if args.calibrate:
                payload["calibration"] = {k: v.as_dict() for k, v in suggestions.items()}
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
            print()
            print(batch.metrics.render_types())
        if args.calibrate:
            print()
            print(render_calibration(suggestions))


if __name__ == "__main__":
    main()
