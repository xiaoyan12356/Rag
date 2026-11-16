"""Evaluation runner.

The runner takes a golden test set, asks the hybrid search every question of
it, and lets an evaluator score what came back. What comes out is a report
that says how often the search hit a golden chunk, how high that chunk was
ranked, and what every single question scored.

The report is built so that a regression can be seen at a glance: the
aggregate numbers at the top, the detail of every question underneath.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10


class EvalError(Exception):
    """Raised when the evaluation cannot be run."""


@dataclass
class GoldenTestCase:
    """One question of the golden test set, with what it should find."""

    query: str
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenTestCase:
        """Build a test case out of one entry of the golden test set."""
        return cls(
            query=str(data.get("query", "")),
            expected_chunk_ids=[str(item) for item in data.get("expected_chunk_ids", [])],
            expected_sources=[str(item) for item in data.get("expected_sources", [])],
        )


@dataclass
class QueryResult:
    """What one question of the golden test set scored."""

    query: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    hit: bool = False
    rank: int | None = None
    reciprocal_rank: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary."""
        return {
            "query": self.query,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "hit": self.hit,
            "rank": self.rank,
            "reciprocal_rank": self.reciprocal_rank,
            "metrics": dict(self.metrics),
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


@dataclass
class EvalReport:
    """What the whole golden test set scored.

    Attributes:
        query_results: The result of every question.
        hit_rate: The share of the questions that found a golden chunk.
        mrr: The mean reciprocal rank of the golden chunks.
        metrics: The mean of every metric the evaluators reported.
        total_elapsed_ms: How long the whole run took.
        evaluator_name: The evaluator that scored the run.
        test_set_path: The golden test set that was run.
    """

    query_results: list[QueryResult] = field(default_factory=list)
    hit_rate: float = 0.0
    mrr: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0
    evaluator_name: str = ""
    test_set_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a plain dictionary."""
        return {
            "evaluator_name": self.evaluator_name,
            "test_set_path": self.test_set_path,
            "query_count": len(self.query_results),
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "metrics": {key: round(value, 4) for key, value in self.metrics.items()},
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "query_results": [result.to_dict() for result in self.query_results],
        }


def load_test_set(path: str | Path) -> list[GoldenTestCase]:
    """Read the golden test set from a JSON file.

    Args:
        path: The file the golden test set lives in.

    Returns:
        The test cases of the file.

    Raises:
        EvalError: the file is missing or does not have the expected shape.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise EvalError(f"The golden test set is not there: {file_path}")

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalError(f"The golden test set cannot be read: {exc}") from exc

    if not isinstance(data, dict) or "test_cases" not in data:
        raise EvalError(
            "The golden test set must be a mapping with a 'test_cases' key"
        )

    cases = data["test_cases"]
    if not isinstance(cases, list):
        raise EvalError("The 'test_cases' of the golden test set must be a list")

    return [GoldenTestCase.from_dict(case) for case in cases]


class EvalRunner:
    """Runs a golden test set through the search and an evaluator.

    Args:
        settings: The settings of the project. They are kept so that the
            report can say which configuration it was run with.
        hybrid_search: The search that is asked every question.
        evaluator: The evaluator that scores what comes back.
        top_k: How many chunks are asked for, when the caller leaves it out.
    """

    def __init__(
        self,
        settings: Any = None,
        hybrid_search: Any = None,
        evaluator: BaseEvaluator | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Set up the runner."""
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        self.top_k = top_k

    def run(self, test_set_path: str | Path) -> EvalReport:
        """Run every question of the golden test set.

        Args:
            test_set_path: The file the golden test set lives in.

        Returns:
            The report of the run.

        Raises:
            EvalError: the test set is empty, or the search cannot be run.
        """
        test_cases = load_test_set(test_set_path)
        if not test_cases:
            raise EvalError("The golden test set is empty")

        report = EvalReport(
            evaluator_name=type(self.evaluator).__name__ if self.evaluator else "",
            test_set_path=str(test_set_path),
        )

        started = time.monotonic()
        for case in test_cases:
            report.query_results.append(self._run_case(case))
        report.total_elapsed_ms = (time.monotonic() - started) * 1000.0

        report.hit_rate = _mean([1.0 if result.hit else 0.0 for result in report.query_results])
        report.mrr = _mean([result.reciprocal_rank for result in report.query_results])
        report.metrics = self._average_metrics(report.query_results)
        report.metrics.setdefault("hit_rate", report.hit_rate)
        report.metrics.setdefault("mrr", report.mrr)

        logger.info(
            "The evaluation ran %d questions: hit_rate=%.3f, mrr=%.3f",
            len(report.query_results),
            report.hit_rate,
            report.mrr,
        )
        return report

    def _run_case(self, case: GoldenTestCase) -> QueryResult:
        """Run one question and score what came back."""
        started = time.monotonic()
        result = QueryResult(
            query=case.query,
            expected_chunk_ids=list(case.expected_chunk_ids),
        )

        chunks = self._retrieve(case.query)
        result.retrieved_chunk_ids = [self._chunk_id(chunk) for chunk in chunks]

        rank = self._first_golden_rank(result.retrieved_chunk_ids, case.expected_chunk_ids)
        result.rank = rank
        result.hit = rank is not None
        result.reciprocal_rank = 0.0 if rank is None else 1.0 / rank

        if self.evaluator is not None and chunks:
            try:
                result.metrics = self.evaluator.evaluate(
                    query=case.query,
                    retrieved_chunks=chunks,
                    generated_answer=self._answer(chunks),
                    ground_truth={"ids": case.expected_chunk_ids},
                )
            except Exception as exc:
                logger.warning(
                    "The evaluation of '%s' failed: %s", case.query[:60], exc
                )
                result.metrics = {}

        result.elapsed_ms = (time.monotonic() - started) * 1000.0
        return result

    def _retrieve(self, query: str) -> list[Any]:
        """Ask the search for the chunks that answer the question."""
        if self.hybrid_search is None:
            raise EvalError("The EvalRunner needs a search to run the questions against")

        try:
            results = self.hybrid_search.search(query, self.top_k)
        except Exception as exc:
            raise EvalError(f"The search failed for '{query[:60]}': {exc}") from exc

        return list(results or [])

    @staticmethod
    def _first_golden_rank(retrieved_ids: list[str], golden_ids: list[str]) -> int | None:
        """Return the place the first golden chunk was found at."""
        if not golden_ids:
            return None
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in golden_ids:
                return rank
        return None

    @staticmethod
    def _chunk_id(chunk: Any) -> str:
        """Return the id of a chunk, whatever shape it arrives in."""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            for key in ("chunk_id", "id"):
                if key in chunk:
                    return str(chunk[key])
            return ""
        chunk_id = getattr(chunk, "chunk_id", None)
        if chunk_id:
            return str(chunk_id)
        chunk_id = getattr(chunk, "id", None)
        if chunk_id:
            return str(chunk_id)
        return ""

    @staticmethod
    def _answer(chunks: list[Any]) -> str:
        """Return an answer built from the chunks that were retrieved."""
        texts: list[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                texts.append(str(chunk.get("text", "")))
            else:
                texts.append(str(getattr(chunk, "text", "")))
        return " ".join(text for text in texts if text)

    @staticmethod
    def _average_metrics(results: list[QueryResult]) -> dict[str, float]:
        """Return the mean of every metric the queries reported."""
        keys: set[str] = set()
        for result in results:
            keys.update(result.metrics)

        averages: dict[str, float] = {}
        for key in sorted(keys):
            values = [result.metrics[key] for result in results if key in result.metrics]
            averages[key] = _mean(values)
        return averages


def _mean(values: list[float]) -> float:
    """Return the mean of the values; an empty list has the mean of zero."""
    if not values:
        return 0.0
    return sum(values) / len(values)
