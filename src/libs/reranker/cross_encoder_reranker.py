"""Cross-encoder reranking.

A cross-encoder reads the query and a passage together and scores how well
they match, which is more accurate than comparing their embeddings. The
`sentence-transformers` package is only needed when no scorer is injected, so
the reranker can be created - and tested - without the heavy dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from core.settings import RerankSettings
from libs.reranker.base_reranker import BaseReranker, RerankerError

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# A scorer reads the query and the passages and returns one score per passage.
Scorer = Callable[[str, Sequence[str]], Sequence[float]]


class CrossEncoderReranker(BaseReranker):
    """Rerank the top candidates with a cross-encoder model."""

    def __init__(self, config: RerankSettings, scorer: Scorer | None = None) -> None:
        super().__init__(config)
        self.model_name = config.model or DEFAULT_MODEL
        self._scorer = scorer

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return `candidates` with the top ones reordered by cross-encoder score."""
        self.validate_query(query)
        self.validate_candidates(candidates)

        by_id = {_candidate_id(candidate): dict(candidate) for candidate in candidates}
        if len(by_id) == 1:
            return list(by_id.values())

        scored_ids = list(by_id)[: self.config.top_k]
        passages = [_candidate_text(by_id[candidate_id]) for candidate_id in scored_ids]

        scorer = self._get_scorer()
        try:
            scores = [float(score) for score in scorer(query, passages)]
        except Exception:
            self.fallback_used = True
            return list(by_id.values())

        if len(scores) != len(passages):
            self.fallback_used = True
            return list(by_id.values())

        order = sorted(range(len(scored_ids)), key=lambda index: scores[index], reverse=True)
        self.fallback_used = False
        return [by_id[scored_ids[index]] for index in order] + [
            by_id[candidate_id] for candidate_id in list(by_id)[len(scored_ids) :]
        ]

    def _get_scorer(self) -> Scorer:
        if self._scorer is None:
            self._scorer = _load_scorer(self.model_name)
        return self._scorer


def _load_scorer(model_name: str) -> Scorer:
    """Load the cross-encoder model, failing with a readable error."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankerError(
            "The cross_encoder backend needs the sentence-transformers package. "
            "Install it with 'pip install sentence-transformers'."
        ) from exc

    model = CrossEncoder(model_name)

    def score(query: str, passages: Sequence[str]) -> Sequence[float]:
        pairs = [(query, passage) for passage in passages]
        return [float(value) for value in model.predict(pairs)]

    return score


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    candidate_id = candidate.get("id")
    if candidate_id is None or not str(candidate_id).strip():
        raise RerankerError("A candidate record carries no id to rank it by")
    return str(candidate_id)


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    text = candidate.get("text")
    if not isinstance(text, str):
        metadata = candidate.get("metadata")
        text = metadata.get("text") if isinstance(metadata, Mapping) else None
    return text if isinstance(text, str) else ""
