#!/usr/bin/env python
"""Query script for the Modular RAG MCP Server.

This script asks the knowledge hub a question and prints what it found. It
runs the same search the MCP tool runs, so what it prints is what an agent
would get.

Usage:
    # Ask a question about the documents of a collection
    python scripts/query.py --query "how do I configure Azure?"

    # Ask for more results, and for the collection the documents live in
    python scripts/query.py --query "chunking" --collection contracts --top-k 5

    # See what every stage of the search found
    python scripts/query.py --query "chunking" --verbose

    # Skip the reranking stage
    python scripts/query.py --query "chunking" --no-rerank

The collection has to be the one the documents were ingested into; see
scripts/ingest.py.

Exit codes:
    0 - the search ran and found something
    1 - the search ran and found nothing
    2 - the search could not run
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

if sys.platform == "win32":  # pragma: no cover - depends on the terminal
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from core.query_engine.hybrid_search import HybridSearch  # noqa: E402
from core.query_engine.reranker import Reranker  # noqa: E402
from core.settings import SettingsError, load_settings  # noqa: E402
from core.trace.trace_context import TraceContext  # noqa: E402
from core.types import RetrievalResult  # noqa: E402

DEFAULT_COLLECTION = "default"
DEFAULT_TOP_K = 10
SUMMARY_LENGTH = 240

EXIT_OK = 0
EXIT_NO_RESULTS = 1
EXIT_FAILED = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the arguments of the command line."""
    parser = argparse.ArgumentParser(
        description="Search the Modular RAG knowledge hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="The question to search for.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many results to show (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--collection",
        "-c",
        default=DEFAULT_COLLECTION,
        help=f"The collection to search (default: '{DEFAULT_COLLECTION}')",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the reranking stage",
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
        help="Show what every stage of the search found",
    )
    return parser.parse_args(argv)


def summarize(text: str, length: int = SUMMARY_LENGTH) -> str:
    """Return a short, single-line version of the text."""
    flattened = " ".join(text.split())
    if len(flattened) <= length:
        return flattened
    return flattened[: length - 3].rstrip() + "..."


def describe_source(metadata: dict) -> str:
    """Say where a chunk came from."""
    parts: list[str] = []
    source = metadata.get("source_path")
    if source:
        parts.append(str(source))
    doc_type = metadata.get("doc_type")
    if doc_type:
        parts.append(str(doc_type))
    page = metadata.get("page", metadata.get("page_number"))
    if page is not None:
        parts.append(f"page {page}")
    return "  ".join(parts) if parts else "(unknown source)"


def print_result(number: int, result: RetrievalResult) -> None:
    """Print one result."""
    print(f"[{number}] score={result.score:.4f}  {result.chunk_id}")
    print(f"    {describe_source(result.metadata)}")
    print(f"    {summarize(result.text)}")
    print()


def print_stage(title: str, chunk_ids: list[str], results: list[RetrievalResult]) -> None:
    """Print what one stage of the search found."""
    print(f"--- {title} ({len(chunk_ids)}) ---")
    if not chunk_ids:
        print("(nothing)")
        print()
        return

    by_id = {result.chunk_id: result for result in results}
    for chunk_id in chunk_ids:
        result = by_id.get(chunk_id)
        if result is None:
            print(f"  {chunk_id}")
            continue
        print(f"  {chunk_id}  score={result.score:.4f}  {summarize(result.text, 90)}")
    print()


def main(argv: list[str] | None = None) -> int:
    """Search the knowledge hub and print what was found."""
    args = parse_args(argv)

    print("[*] Modular RAG query")
    print("=" * 60)

    try:
        settings = load_settings(args.config)
    except SettingsError as exc:
        print(f"[FAIL] The configuration could not be read: {exc}")
        return EXIT_FAILED

    print(f"[OK] Configuration read from {args.config}")
    print(f"[INFO] Collection: {args.collection}")
    print(f"[INFO] Query: {args.query}")

    trace = TraceContext(trace_type="query")
    try:
        search = HybridSearch(settings, collection=args.collection)
        results = search.search(args.query, args.top_k, trace=trace)
    except Exception as exc:
        print(f"[FAIL] The search could not be run: {exc}")
        return EXIT_FAILED

    if results and not args.no_rerank:
        try:
            reranker = Reranker(settings)
            results = reranker.rerank(args.query, results, trace=trace)
            if reranker.fallback_used:
                print("[WARN] The reranker failed, the order of the search is kept")
        except Exception as exc:
            print(f"[WARN] The reranker could not be run: {exc}")

    if args.verbose:
        print()
        print_stage(
            "Dense results",
            (trace.get_stage_data("dense_retrieval") or {}).get("chunk_ids", []),
            results,
        )
        print_stage(
            "Sparse results",
            (trace.get_stage_data("sparse_retrieval") or {}).get("chunk_ids", []),
            results,
        )
        print_stage(
            "Fused results",
            (trace.get_stage_data("fusion") or {}).get("chunk_ids", []),
            results,
        )

    print("=" * 60)
    print(f"RESULTS ({len(results)})")
    print("=" * 60)

    if not results:
        print()
        print("Nothing was found.")
        print(
            "Have the documents been ingested? Run scripts/ingest.py with the "
            "same collection."
        )
        return EXIT_NO_RESULTS

    for number, result in enumerate(results, 1):
        print_result(number, result)

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
