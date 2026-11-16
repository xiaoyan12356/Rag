"""Evaluation service for the dashboard.

The evaluation panel needs two things: the golden test sets that can be run,
and a way to run one of them. Both live here, so the page does not have to
know how the evaluator is built or where the report is kept.

A run is written to `reports/evaluation.json` when it is over, so the panel
can show the last result after the page is reloaded.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.query_engine.hybrid_search import HybridSearch
from core.settings import DEFAULT_SETTINGS_PATH, Settings, load_settings, resolve_path
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.eval_runner import EvalReport, EvalRunner

logger = logging.getLogger(__name__)

DEFAULT_TEST_SET = "tests/fixtures/golden_test_set.json"
REPORT_PATH = "reports/evaluation.json"


class EvaluationService:
    """Run the golden test sets and keep the result of the last run.

    Args:
        settings: The configuration of the project; the default one is read
            when it is left out.
        settings_path: The configuration file to read, when the settings
            themselves are not handed in.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        settings_path: str | Path | None = None,
    ) -> None:
        """Set up the service without building anything yet."""
        self._settings = settings
        self._settings_path = settings_path
        self._search: HybridSearch | None = None
        self._search_collection = ""

    @property
    def settings(self) -> Settings:
        """Return the configuration, reading the file the first time."""
        if self._settings is None:
            self._settings = load_settings(self._settings_path or DEFAULT_SETTINGS_PATH)
        return self._settings

    def list_test_sets(self) -> list[str]:
        """Return the golden test sets that can be run."""
        folder = resolve_path("tests/fixtures")
        if not folder.exists():
            return [str(resolve_path(DEFAULT_TEST_SET))]

        found = sorted(
            str(path) for path in folder.glob("golden*_test_set.json") if path.is_file()
        )
        return found or [str(resolve_path(DEFAULT_TEST_SET))]

    def available_backends(self) -> list[str]:
        """Return the evaluators that can be used."""
        backends = ["custom", "none"]
        for name in EvaluatorFactory.registered_providers():
            if name not in backends:
                backends.append(name)
        return backends

    def run(
        self,
        test_set_path: str | Path,
        backend: str = "custom",
        top_k: int = 10,
        collection: str | None = None,
    ) -> EvalReport:
        """Run one golden test set and return the report.

        Args:
            test_set_path: The golden test set to run.
            backend: The evaluator to score the answers with.
            top_k: How many chunks to ask for per question.
            collection: The collection to search.

        Returns:
            The report of the run.

        Raises:
            Exception: The run could not be made.
        """
        settings = self._settings_for(backend)
        evaluator = EvaluatorFactory.create(settings)
        runner = EvalRunner(
            settings=settings,
            hybrid_search=self._get_search(collection),
            evaluator=evaluator,
            top_k=top_k,
        )
        report = runner.run(test_set_path)
        self.save_report(report)
        return report

    def evaluate_answer(
        self,
        query: str,
        chunks: list[Any],
        backend: str = "custom",
    ) -> dict[str, float]:
        """Score one answer with the chosen evaluator.

        Args:
            query: The question that was asked.
            chunks: The chunks the answer was built from.
            backend: The evaluator to score the answer with.

        Returns:
            The metrics of the evaluator.

        Raises:
            Exception: The answer could not be scored.
        """
        settings = self._settings_for(backend)
        evaluator = EvaluatorFactory.create(settings)
        answer = " ".join(str(getattr(chunk, "text", "")) for chunk in chunks)
        return evaluator.evaluate(
            query=query,
            retrieved_chunks=chunks,
            generated_answer=answer,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Any]:
        """Return the chunks that answer a question.

        Args:
            query: The question to answer.
            top_k: How many chunks to ask for.

        Returns:
            The chunks the search found.
        """
        return list(self._get_search(None).search(query, top_k=top_k))

    def save_report(self, report: EvalReport) -> Path:
        """Write the report out, so the next visit can show it."""
        path = resolve_path(REPORT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def last_report(self) -> dict[str, Any] | None:
        """Return the report of the last run, or None when there is none."""
        path = resolve_path(REPORT_PATH)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("The last evaluation report could not be read: %s", exc)
            return None
        return payload if isinstance(payload, dict) else None

    def _settings_for(self, backend: str) -> Settings:
        """Return the configuration with the chosen evaluator switched on."""
        settings = self.settings
        return replace(
            settings,
            evaluation=replace(
                settings.evaluation,
                enabled=True,
                provider=backend,
                backends=[backend],
            ),
        )

    def _get_search(self, collection: str | None) -> HybridSearch:
        """Return the search, built for the collection when it changed."""
        target = collection or self.settings.vector_store.collection_name
        if self._search is not None and self._search_collection == target:
            return self._search

        self._search = HybridSearch(self.settings, collection=target)
        self._search_collection = target
        return self._search
