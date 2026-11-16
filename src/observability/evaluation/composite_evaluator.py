"""Composite evaluator.

The composite evaluator holds several evaluators and asks all of them the
same question. Their answers are merged into one set of metrics, so a caller
gets the counts of the custom evaluator and the judgements of the ragas
evaluator from a single call.

One evaluator failing does not stop the others: the failure is reported and
the metrics of the rest still come back. Only when every evaluator fails is
the whole call a failure.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from core.settings import EvaluationSettings, Settings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# The providers that name the composite evaluator itself are left out, so the
# composite never builds itself.
SELF_PROVIDERS = frozenset({"composite", "none", "disabled"})


class CompositeEvaluator(BaseEvaluator):
    """Runs several evaluators and merges what they say.

    Args:
        config: The evaluation settings of the project.
        evaluators: The evaluators to run. They are built from the backends
            of the settings when they are left out.
        settings: The whole configuration, needed to build the evaluators.
    """

    def __init__(
        self,
        config: EvaluationSettings,
        evaluators: Sequence[BaseEvaluator] | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Set up the composite.

        Raises:
            EvaluatorError: there is no evaluator to run.
        """
        super().__init__(config)

        self.settings = settings

        if evaluators is not None:
            self._evaluators: list[BaseEvaluator] = list(evaluators)
        elif settings is not None:
            self._evaluators = self._build_from_settings(settings)
        else:
            self._evaluators = []

        if not self._evaluators:
            raise EvaluatorError(
                "The CompositeEvaluator needs at least one evaluator. Give it "
                "evaluators, or name them in 'evaluation.backends'."
            )

    @property
    def evaluators(self) -> list[BaseEvaluator]:
        """Return the evaluators that are run."""
        return list(self._evaluators)

    def evaluate(
        self,
        query: str,
        retrieved_chunks: Sequence[Any],
        generated_answer: str | None = None,
        ground_truth: Any = None,
        trace: TraceContext | None = None,
    ) -> dict[str, float]:
        """Run every evaluator and merge their metrics.

        Args:
            query: The question that was asked.
            retrieved_chunks: The chunks the answer was built from.
            generated_answer: The answer that has to be judged.
            ground_truth: The golden chunks or the golden answer.
            trace: A trace to record the work in.

        Returns:
            The metrics of every evaluator, merged into one dictionary.

        Raises:
            EvaluatorError: every evaluator failed.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        merged: dict[str, float] = {}
        failures: list[str] = []

        for evaluator in self._evaluators:
            name = type(evaluator).__name__
            try:
                metrics = evaluator.evaluate(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    generated_answer=generated_answer,
                    ground_truth=ground_truth,
                    trace=trace,
                )
            except Exception as exc:
                logger.warning("%s failed: %s", name, exc)
                failures.append(f"{name}: {exc}")
                continue

            for key, value in metrics.items():
                if key in merged:
                    logger.warning(
                        "The metric '%s' came from more than one evaluator, "
                        "the one of %s is used",
                        key,
                        name,
                    )
                merged[key] = value

        if not merged and failures:
            raise EvaluatorError(
                "Every evaluator failed:\n" + "\n".join(failures)
            )

        return merged

    @staticmethod
    def _build_from_settings(settings: Settings) -> list[BaseEvaluator]:
        """Build the evaluators named by `evaluation.backends`."""
        from dataclasses import replace

        from libs.evaluator.evaluator_factory import EvaluatorFactory

        evaluation = settings.evaluation
        backends = [str(name).strip().lower() for name in (evaluation.backends or [])]
        backends = [name for name in backends if name and name not in SELF_PROVIDERS]
        if not backends:
            return []

        evaluators: list[BaseEvaluator] = []
        for backend in backends:
            backend_settings = replace(
                settings,
                evaluation=replace(
                    evaluation,
                    enabled=True,
                    provider=backend,
                    backends=[],
                ),
            )
            try:
                evaluators.append(EvaluatorFactory.create(backend_settings))
            except Exception as exc:
                logger.warning(
                    "The backend '%s' could not be built: %s", backend, exc
                )

        return evaluators
