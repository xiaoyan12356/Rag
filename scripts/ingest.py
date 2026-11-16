#!/usr/bin/env python
"""Ingestion script for the Modular RAG MCP Server.

This script takes documents from the disk into the knowledge hub. It works on
a single file or on a whole folder, and it remembers what it has already
taken in, so a second run leaves the unchanged documents alone.

Usage:
    # Take a single PDF into a collection
    python scripts/ingest.py --path documents/report.pdf --collection contracts

    # Take every PDF of a folder into a collection
    python scripts/ingest.py --path documents/ --collection technical_docs

    # Take a document in again, even though it was taken in before
    python scripts/ingest.py --path documents/report.pdf --collection contracts --force

    # Use another configuration file
    python scripts/ingest.py --path documents/ --config custom_settings.yaml

Exit codes:
    0 - every file was taken in
    1 - some files were taken in and some failed
    2 - every file failed, or the script could not start
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

if sys.platform == "win32":  # pragma: no cover - depends on the terminal
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from core.settings import SettingsError, load_settings  # noqa: E402
from core.trace.trace_context import TraceContext  # noqa: E402
from ingestion.pipeline import IngestionPipeline, PipelineError, PipelineResult  # noqa: E402

SUPPORTED_EXTENSIONS = (".pdf",)
EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_FAILED = 2


@dataclass
class Outcome:
    """What happened to one file."""

    file_path: str
    result: PipelineResult | None = None
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.error == "" and self.result is not None

    @property
    def skipped(self) -> bool:
        return self.result is not None and self.result.skipped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the arguments of the command line."""
    parser = argparse.ArgumentParser(
        description="Take documents into the Modular RAG knowledge hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--path",
        "-p",
        required=True,
        help="The file or folder to take in. A folder is searched for PDFs.",
    )
    parser.add_argument(
        "--collection",
        "-c",
        default="default",
        help="The collection to put the documents in (default: 'default')",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Take a document in again, even though it was taken in before",
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="The configuration file to use (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Say more about what is going on",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Name the files that would be taken in, without taking any in",
    )
    return parser.parse_args(argv)


def discover_files(path: str | Path) -> list[Path]:
    """Return the files that are to be taken in.

    A file is taken as it is; a folder is searched for PDFs.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"The path does not exist: {target}")

    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Only {', '.join(SUPPORTED_EXTENSIONS)} files can be taken in, "
                f"and this is a {target.suffix or 'file without a suffix'}"
            )
        return [target]

    found: set[Path] = set()
    for extension in SUPPORTED_EXTENSIONS:
        found.update(target.rglob(f"*{extension}"))
        found.update(target.rglob(f"*{extension.upper()}"))
    return sorted(found)


def print_summary(outcomes: list[Outcome], verbose: bool = False) -> None:
    """Say what the run did."""
    successful = sum(1 for outcome in outcomes if outcome.succeeded)
    skipped = sum(1 for outcome in outcomes if outcome.skipped)
    failed = len(outcomes) - successful

    chunks = sum(outcome.result.chunk_count for outcome in outcomes if outcome.succeeded)
    images = sum(outcome.result.image_count for outcome in outcomes if outcome.succeeded)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files: {len(outcomes)}")
    print(f"  Taken in: {successful - skipped}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"Chunks: {chunks}")
    print(f"Images: {images}")

    if verbose:
        for outcome in outcomes:
            if outcome.error:
                print(f"  [FAIL] {outcome.file_path}: {outcome.error}")
            elif outcome.skipped:
                print(f"  [SKIP] {outcome.file_path}")
            else:
                print(
                    f"  [OK] {outcome.file_path}: "
                    f"{outcome.result.chunk_count} chunks, "
                    f"{outcome.result.image_count} images"
                )
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """Take the documents in and say how it went."""
    args = parse_args(argv)

    print("[*] Modular RAG ingestion")
    print("=" * 60)

    try:
        settings = load_settings(args.config)
    except SettingsError as exc:
        print(f"[FAIL] The configuration could not be read: {exc}")
        return EXIT_FAILED
    print(f"[OK] Configuration read from {args.config}")

    try:
        files = discover_files(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return EXIT_FAILED

    print(f"[INFO] {len(files)} file(s) to take in")
    for file_path in files:
        print(f"   - {file_path}")

    if args.dry_run:
        print("\n[INFO] Nothing was taken in")
        return EXIT_OK
    if not files:
        print("\n[WARN] There was nothing to take in")
        return EXIT_OK

    print(f"\n[INFO] Collection: {args.collection}")
    print(f"[INFO] Force: {args.force}")

    try:
        pipeline = IngestionPipeline(
            settings=settings,
            collection=args.collection,
            force=args.force,
        )
    except Exception as exc:
        print(f"[FAIL] The pipeline could not be started: {exc}")
        return EXIT_FAILED

    outcomes: list[Outcome] = []
    for number, file_path in enumerate(files, 1):
        print(f"\n[{number}/{len(files)}] {file_path}")
        trace = TraceContext(trace_type="ingestion")
        trace.metadata["source_path"] = str(file_path)
        try:
            result = pipeline.run(file_path, trace=trace)
        except PipelineError as exc:
            outcomes.append(Outcome(file_path=str(file_path), error=str(exc)))
            print(f"   [FAIL] {exc}")
            continue
        except Exception as exc:
            outcomes.append(Outcome(file_path=str(file_path), error=str(exc)))
            print(f"   [FAIL] {exc}")
            continue

        outcomes.append(Outcome(file_path=str(file_path), result=result))
        if result.skipped:
            print("   [SKIP] It was taken in before")
        else:
            print(f"   [OK] {result.chunk_count} chunks, {result.image_count} images")

    print_summary(outcomes, args.verbose)

    successful = sum(1 for outcome in outcomes if outcome.succeeded)
    if successful == len(outcomes):
        return EXIT_OK
    if successful > 0:
        return EXIT_PARTIAL
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
