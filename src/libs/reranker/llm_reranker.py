"""LLM reranking.

The LLM is asked to order the candidate ids and answers with a JSON array of
ids. The instruction lives in `config/prompts/rerank.txt` so it can be tuned
without touching the code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.settings import RerankSettings, resolve_path
from libs.llm.base_llm import BaseLLM, ChatMessage, LLMError
from libs.reranker.base_reranker import BaseReranker, RerankerError

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

DEFAULT_PROMPT_PATH = "config/prompts/rerank.txt"


class LLMReranker(BaseReranker):
    """Rerank candidates by asking an LLM to order their ids."""

    def __init__(
        self,
        config: RerankSettings,
        llm: BaseLLM,
        *,
        prompt_template: str | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        super().__init__(config)
        self.llm = llm
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else _read_prompt(prompt_path or DEFAULT_PROMPT_PATH)
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return `candidates` in the order the LLM ranked them."""
        self.validate_query(query)
        self.validate_candidates(candidates)

        by_id = {_candidate_id(candidate): dict(candidate) for candidate in candidates}
        if len(by_id) == 1:
            return list(by_id.values())

        messages: list[ChatMessage] = [
            {"role": "user", "content": self._build_prompt(query, by_id)}
        ]

        try:
            answer = self.llm.chat(messages, trace)
        except LLMError:
            self.fallback_used = True
            return list(by_id.values())

        try:
            ranked = self._rank(answer, by_id)
        except RerankerError:
            self.fallback_used = True
            raise

        self.fallback_used = False
        return ranked

    def _build_prompt(self, query: str, by_id: Mapping[str, dict[str, Any]]) -> str:
        lines = [
            f"{candidate_id}: {_candidate_text(candidate)}"
            for candidate_id, candidate in by_id.items()
        ]
        return (
            self.prompt_template.replace("{query}", query)
            .replace("{candidates}", "\n".join(lines))
        )

    def _rank(
        self, answer: str, by_id: Mapping[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Turn the LLM answer into the reordered candidates."""
        ranked_ids = _parse_ids(answer)

        unknown = [ranked_id for ranked_id in ranked_ids if ranked_id not in by_id]
        if unknown:
            raise RerankerError(
                f"The rerank answer names unknown candidate ids: {', '.join(unknown)}"
            )

        repeated = {ranked_id for ranked_id in ranked_ids if ranked_ids.count(ranked_id) > 1}
        if repeated:
            raise RerankerError(
                f"The rerank answer repeats candidate ids: {', '.join(sorted(repeated))}"
            )

        missing = [candidate_id for candidate_id in by_id if candidate_id not in ranked_ids]
        if missing:
            raise RerankerError(
                f"The rerank answer leaves out candidate ids: {', '.join(missing)}"
            )

        return [by_id[ranked_id] for ranked_id in ranked_ids]


def _read_prompt(path: str | Path) -> str:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise RerankerError(f"The rerank prompt file was not found: {path}")
    return resolved.read_text(encoding="utf-8")


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


def _parse_ids(answer: str) -> list[str]:
    """Read the JSON array of candidate ids out of the LLM answer."""
    text = _strip_code_fence(answer.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RerankerError(
            f"The rerank answer is not valid JSON: {exc}. Answer: {text[:200]}"
        ) from exc

    if not isinstance(parsed, list):
        raise RerankerError(
            "The rerank answer must be a JSON array of candidate ids, got "
            f"{type(parsed).__name__}. Answer: {text[:200]}"
        )

    if not all(isinstance(item, str) for item in parsed):
        raise RerankerError(
            f"The rerank answer must only hold candidate ids as strings. Answer: {text[:200]}"
        )

    return list(parsed)


def _strip_code_fence(text: str) -> str:
    """Remove the markdown fence a model may wrap its answer in."""
    if not text.startswith("```"):
        return text
    text = text[3:]
    if text.lower().startswith("json"):
        text = text[4:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
