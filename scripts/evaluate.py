#!/usr/bin/env python
"""Evaluation script for the Modular RAG MCP Server.

This script runs a golden test set against the knowledge hub and prints how
well the search did. It asks every question of the golden test set, counts how
often a golden chunk was found and how high it was ranked, and lets the
configured evaluator score the answers.

Usage:
    # Run the golden test set of the repository
    python scripts/evaluate.py

    # Run another golden test set
    python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json

    # Ask for more chunks per question
    python scripts/evaluate.py --top-k 20

    # Use another configuration file
    python scripts/evaluate.py --config custom_settings.yaml

Exit codes:
    0 - the evaluation ran
    1 - the evaluation ran and hit nothing
    2 - the evaluation could not run
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

if sys.platform == "win32":  # pragma: no cover - depends on the terminal
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from core.query_engine.hybrid_search import HybridSearch  # noqa: E402
from core.settings import SettingsError, load_settings  # noqa: E402
from core.types import RetrievalResult  # noqa: E402
from libs.evaluator.evaluator_factory import EvaluatorFactory  # noqa: E402
from observability.evaluation.eval_runner import EvalError, EvalRunner  # noqa: E402

DEFAULT_TEST_SET = "tests/fixtures/golden_test_set.json"
DEFAULT_TOP_K = 10

EXIT_OK = 0
EXIT_NO_HITS = 1
EXIT_FAILED = 2


class MockSearch:
    """A search that answers every question with a placeholder chunk.

    It is used by `--no-search`, where the point is to check that the
    evaluation framework runs, not that the corpus is any good: the chunk it
    hands back carries the id the golden test set expects, so the whole path
    of the runner is walked through.
    """

    def __init__(self, test_set_path: str) -> None:
        """Set up the mock from the golden test set."""
        from observability.evaluation.eval_runner import load_test_set

        self._answers = {
            case.query: case.expected_chunk_ids[0]
            for case in load_test_set(test_set_path)
            if case.expected_chunk_ids
        }

    def search(self, query: str, top_k: int | None = None) -> list[Any]:
        """Return the placeholder chunk of the question."""
        chunk_id = self._answers.get(query)
        if chunk_id is None:
            return []
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                score=1.0,
                text="A placeholder chunk. The search was not really run.",
                metadata={"source_path": "mock"},
            )
        ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the arguments of the command line."""
    parser = argparse.ArgumentParser(
        description="Run a golden test set against the Modular RAG knowledge hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-set",
        "-t",
        default=str(_REPO_ROOT / DEFAULT_TEST_SET),
        help=f"The golden test set to run (default: {DEFAULT_TEST_SET})",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many chunks to ask for per question (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--collection",
        "-c",
        default=None,
        help="The collection to search (default: the configured one)",
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="The configuration file to use (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON instead of as text",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help=(
            "Skip the real search and answer every question with a placeholder "
            "chunk. It checks that the evaluation framework runs, not that the "
            "corpus is any good."
        ),
    )
    return parser.parse_args(argv)


def print_report(report, verbose: bool = False) -> None:
    """Say what the evaluation found."""
    if verbose:
        print("=" * 60)
        print("QUESTIONS")
        print("=" * 60)
        for number, result in enumerate(report.query_results, 1):
            mark = "HIT " if result.hit else "MISS"
            rank = result.rank if result.rank is not None else "-"
            print(f"[{number}] {mark} rank={rank}  {result.query}")
            print(f"    retrieved: {', '.join(result.retrieved_chunk_ids) or '(nothing)'}")
            for key, value in result.metrics.items():
                print(f"    {key}: {value:.4f}")

    print()
    print("=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"Questions: {len(report.query_results)}")
    print(f"Hit rate: {report.hit_rate:.4f}")
    print(f"MRR: {report.mrr:.4f}")
    for key, value in report.metrics.items():
        if key in ("hit_rate", "mrr"):
            continue
        print(f"{key}: {value:.4f}")
    print(f"Took: {report.total_elapsed_ms:.1f} ms")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and say what it found."""
    args = parse_args(argv)

    print("[*] Modular RAG evaluation")
    print("=" * 60)

    try:
        settings = load_settings(args.config)
    except SettingsError as exc:
        print(f"[FAIL] The configuration could not be read: {exc}")
        return EXIT_FAILED
    print(f"[OK] Configuration read from {args.config}")

    search = None
    if not args.no_search:
        try:
            search = HybridSearch(settings, collection=args.collection)
        except Exception as exc:
            print(f"[FAIL] The search could not be started: {exc}")
            return EXIT_FAILED

    try:
        evaluator = EvaluatorFactory.create(_with_evaluation_on(settings))
    except Exception as exc:
        print(f"[FAIL] The evaluator could not be started: {exc}")
        return EXIT_FAILED
    print(f"[OK] Evaluator: {type(evaluator).__name__}")

    runner = EvalRunner(
        settings=settings,
        hybrid_search=search if search is not None else MockSearch(args.test_set),
        evaluator=evaluator,
        top_k=args.top_k,
    )

    if search is None:
        print("[*] The search was mocked, so the numbers say nothing about the corpus.")

    try:
        report = runner.run(args.test_set)
    except EvalError as exc:
        print(f"[FAIL] The evaluation could not run: {exc}")
        return EXIT_FAILED
    except Exception as exc:
        print(f"[FAIL] The evaluation could not run: {exc}")
        return EXIT_FAILED

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print_report(report, verbose=True)

    written = _write_report(report)
    if written is not None:
        print(f"[OK] The report was written to {written}")

    if report.hit_rate <= 0.0 and search is not None:
        return EXIT_NO_HITS
    return EXIT_OK


def _with_evaluation_on(settings):
    """Return the configuration with the evaluation switched on.

    The evaluation is off in the configuration of the project, because a
    server has no reason to judge its own answers. This script exists to do
    exactly that, so it switches it on.
    """
    from dataclasses import replace

    provider = (settings.evaluation.provider or "").strip().lower()
    if provider in ("", "none"):
        provider = "custom"

    return replace(
        settings,
        evaluation=replace(
            settings.evaluation,
            enabled=True,
            provider=provider,
            backends=[provider],
        ),
    )


def _write_report(report) -> str | None:
    """Write the report out, so the dashboard can show it as well."""
    try:
        from observability.dashboard.services.evaluation_service import (
            EvaluationService,
        )

        return str(EvaluationService().save_report(report))
    except Exception as exc:
        print(f"[WARN] The report could not be written: {exc}")
        return None


if __name__ == "__main__":
    sys.exit(main())
