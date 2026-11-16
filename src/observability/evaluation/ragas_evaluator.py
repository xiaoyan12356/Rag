"""Ragas evaluator.

Ragas judges an answer the way a person would: it asks a model whether the
answer sticks to the chunks that were retrieved, whether the answer is about
the question, and whether the chunks that came back are the relevant ones.

The three judgements are the metrics this evaluator reports: `faithfulness`,
`answer_relevancy` and `context_precision`. Ragas is an optional dependency,
so the evaluator says plainly what to install when it is missing instead of
failing with a bare `ModuleNotFoundError`.
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

FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = (FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION)

INSTALL_HINT = (
    "The 'ragas' package is needed for the RagasEvaluator. "
    "Install it with: uv pip install --python .venv/Scripts/python.exe "
    "ragas datasets"
)


def ragas_is_available() -> bool:
    """Say whether the ragas package can be imported."""
    try:
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


class RagasEvaluator(BaseEvaluator):
    """Scores an answer with the Ragas metrics.

    Args:
        config: The evaluation settings; the metrics it names are the ones
            that get computed.
        settings: The whole configuration, needed to reach the model that
            does the judging. It may be left out when the model is handed
            over directly instead.
        llm: The model ragas uses to judge. It is built from `settings` when
            it is left out.
        embeddings: The model ragas uses to compare the answer and the
            question.
    """

    def __init__(
        self,
        config: EvaluationSettings,
        settings: Settings | None = None,
        llm: Any | None = None,
        embeddings: Any | None = None,
    ) -> None:
        """Set up the evaluator and work out which metrics it computes.

        Raises:
            ImportError: ragas is not installed.
            EvaluatorError: a requested metric is not one ragas can compute.
        """
        super().__init__(config)

        if not ragas_is_available():
            raise ImportError(INSTALL_HINT)

        self.settings = settings
        self._llm = llm
        self._embeddings = embeddings

        requested = [str(metric).strip().lower() for metric in (config.metrics or [])]
        self.metrics = [
            metric for metric in requested if metric in SUPPORTED_METRICS
        ] or list(SUPPORTED_METRICS)

        unsupported = [metric for metric in requested if metric not in SUPPORTED_METRICS]
        if unsupported:
            raise EvaluatorError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(SUPPORTED_METRICS)}"
            )

    def evaluate(
        self,
        query: str,
        retrieved_chunks: Sequence[Any],
        generated_answer: str | None = None,
        ground_truth: Any = None,
        trace: TraceContext | None = None,
    ) -> dict[str, float]:
        """Return the ragas metrics of one answer.

        Args:
            query: The question that was asked.
            retrieved_chunks: The chunks the answer was built from.
            generated_answer: The answer that has to be judged.
            ground_truth: Not used by ragas; it judges without a reference.
            trace: A trace to record the work in.

        Returns:
            The metrics that were computed, keyed by metric name.

        Raises:
            ValueError: the question or the retrieved chunks are missing.
            EvaluatorError: ragas could not be run.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator needs a non-empty 'generated_answer' - it judges "
                "the answer against the chunks that were retrieved"
            )

        contexts = self._extract_texts(retrieved_chunks)

        try:
            scores = self._score(query, contexts, generated_answer)
        except EvaluatorError:
            raise
        except Exception as exc:
            logger.error("The ragas evaluation failed: %s", exc, exc_info=True)
            raise EvaluatorError(f"The ragas evaluation failed: {exc}") from exc

        if trace is not None:
            trace.record_stage("ragas_evaluation", dict(scores))

        return scores

    def _score(
        self,
        query: str,
        contexts: list[str],
        answer: str,
    ) -> dict[str, float]:
        """Ask ragas for the value of every metric."""
        llm = self._resolve_llm()
        embeddings = self._resolve_embeddings()

        scores: dict[str, float] = {}
        for metric in self.metrics:
            scores[metric] = self._score_one(metric, llm, embeddings, query, contexts, answer)
        return scores

    def _score_one(
        self,
        metric: str,
        llm: Any,
        embeddings: Any,
        query: str,
        contexts: list[str],
        answer: str,
    ) -> float:
        """Return the value of one ragas metric."""
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
            Faithfulness,
        )

        if metric == FAITHFULNESS:
            scorer = Faithfulness(llm=llm)
            result = scorer.score(
                user_input=query,
                response=answer,
                retrieved_contexts=contexts,
            )
        elif metric == ANSWER_RELEVANCY:
            scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
            result = scorer.score(user_input=query, response=answer)
        elif metric == CONTEXT_PRECISION:
            scorer = ContextPrecisionWithoutReference(llm=llm)
            result = scorer.score(
                user_input=query,
                response=answer,
                retrieved_contexts=contexts,
            )
        else:
            raise EvaluatorError(f"Unsupported ragas metric: {metric}")

        return _as_float(result)

    def _resolve_llm(self) -> Any:
        """Return the model ragas judges with."""
        if self._llm is not None:
            return self._llm
        if self.settings is None:
            raise EvaluatorError(
                "The RagasEvaluator needs the settings of the project, or a model "
                "to judge with, before it can score anything"
            )
        self._llm = build_ragas_llm(self.settings)
        return self._llm

    def _resolve_embeddings(self) -> Any:
        """Return the model ragas compares the answer and the question with."""
        if self._embeddings is not None:
            return self._embeddings
        if self.settings is None:
            raise EvaluatorError(
                "The RagasEvaluator needs the settings of the project, or an "
                "embedding model, before it can score anything"
            )
        self._embeddings = build_ragas_embeddings(self.settings)
        return self._embeddings

    @staticmethod
    def _extract_texts(chunks: Sequence[Any]) -> list[str]:
        """Return the text of every chunk, whatever shape it arrives in."""
        texts: list[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content")
                texts.append(str(text) if text else "")
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts


def build_ragas_llm(settings: Settings) -> Any:
    """Build the model ragas judges with, from the project settings.

    Args:
        settings: The settings of the project.

    Returns:
        The ragas model.

    Raises:
        EvaluatorError: the configured provider is not one ragas can use.
    """
    from ragas.llms import InstructorLLM

    llm_settings = settings.llm
    provider = (llm_settings.provider or "").strip().lower()
    endpoint = (llm_settings.azure_endpoint or "").strip()

    if provider == "azure" or (provider == "openai" and endpoint):
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            api_key=llm_settings.api_key,
            azure_endpoint=endpoint,
            api_version=llm_settings.api_version or "2024-02-15-preview",
        )
        return InstructorLLM(client=client, model=llm_settings.model, provider="azure")

    if provider == "openai":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=llm_settings.api_key)
        return InstructorLLM(client=client, model=llm_settings.model, provider="openai")

    raise EvaluatorError(
        f"Unsupported LLM provider for ragas: '{provider}'. Supported: azure, openai"
    )


def build_ragas_embeddings(settings: Settings) -> Any:
    """Build the embedding model ragas compares with, from the settings.

    Args:
        settings: The settings of the project.

    Returns:
        The ragas embedding model.

    Raises:
        EvaluatorError: the configured provider is not one ragas can use.
    """
    from ragas.embeddings import OpenAIEmbeddings

    embedding_settings = settings.embedding
    provider = (embedding_settings.provider or "").strip().lower()
    endpoint = (embedding_settings.azure_endpoint or "").strip()

    if provider == "azure" or (provider == "openai" and endpoint):
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            api_key=embedding_settings.api_key,
            azure_endpoint=endpoint,
            api_version=embedding_settings.api_version or "2024-02-15-preview",
        )
        return OpenAIEmbeddings(model=embedding_settings.model, client=client)

    if provider == "openai":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=embedding_settings.api_key)
        return OpenAIEmbeddings(model=embedding_settings.model, client=client)

    raise EvaluatorError(
        f"Unsupported embedding provider for ragas: '{provider}'. "
        "Supported: azure, openai"
    )


def _as_float(result: Any) -> float:
    """Return the value of a ragas result as a plain float."""
    value = getattr(result, "value", result)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
