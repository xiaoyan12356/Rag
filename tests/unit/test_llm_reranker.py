"""Tests for the LLM reranker.

The LLM is replaced by a fake that answers with a prepared string, so the
prompt shaping and the response validation are covered without a network call.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from core.settings import LLMSettings, RerankSettings, Settings
from libs.llm.base_llm import BaseLLM, ChatMessage, LLMError
from libs.reranker.base_reranker import RerankerError
from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.reranker_factory import RerankerFactory


class FakeLLM(BaseLLM):
    """Answers with a prepared string and records the prompts it received."""

    def __init__(self, answer: str = '["b", "a"]', error: Exception | None = None) -> None:
        super().__init__(LLMSettings(provider="fake", model="fake-model"))
        self.answer = answer
        self.error = error
        self.prompts: list[str] = []

    def chat(self, messages: Sequence[ChatMessage], trace=None) -> str:
        self.prompts.append(messages[-1]["content"])
        if self.error is not None:
            raise self.error
        return self.answer


def make_candidates(*ids: str) -> list[dict]:
    return [{"id": candidate_id, "text": f"text of {candidate_id}"} for candidate_id in ids]


def make_reranker(answer: str = '["b", "a"]', error: Exception | None = None) -> LLMReranker:
    return LLMReranker(
        RerankSettings(enabled=True, provider="llm", model="fake", top_k=5),
        FakeLLM(answer=answer, error=error),
        prompt_template="Query:\n{query}\n\nCandidates:\n{candidates}\n\nOrder them.",
    )


def make_settings(provider: str = "llm", enabled: bool = True) -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o", api_key="fake-key"),
        rerank=RerankSettings(enabled=enabled, provider=provider, model="gpt-4o", top_k=5),
    )


class TestPrompt:
    def test_the_prompt_carries_the_query_and_the_candidates(self):
        reranker = make_reranker()

        reranker.rerank("what is rag", make_candidates("a", "b"))

        prompt = reranker.llm.prompts[0]
        assert "what is rag" in prompt
        assert "a: text of a" in prompt
        assert "b: text of b" in prompt

    def test_the_prompt_uses_the_candidate_text_from_the_metadata(self):
        reranker = make_reranker()

        reranker.rerank(
            "query",
            [
                {"id": "a", "metadata": {"text": "from metadata"}},
                {"id": "b", "text": "plain text"},
            ],
        )

        assert "from metadata" in reranker.llm.prompts[0]

    def test_the_prompt_file_is_used_when_no_template_is_injected(self):
        reranker = LLMReranker(
            RerankSettings(enabled=True, provider="llm", model="fake"),
            FakeLLM(answer='["b", "a"]'),
        )

        reranker.rerank("what is rag", make_candidates("a", "b"))

        prompt = reranker.llm.prompts[0]
        assert "what is rag" in prompt
        assert "a: text of a" in prompt


class TestRanking:
    def test_the_candidates_are_returned_in_the_llm_order(self):
        reranker = make_reranker('["b", "a"]')

        ranked = reranker.rerank("query", make_candidates("a", "b"))

        assert [candidate["id"] for candidate in ranked] == ["b", "a"]

    def test_the_candidate_records_are_preserved(self):
        reranker = make_reranker('["b", "a"]')

        ranked = reranker.rerank("query", make_candidates("a", "b"))

        assert ranked[0] == {"id": "b", "text": "text of b"}

    def test_a_successful_ranking_is_not_a_fallback(self):
        reranker = make_reranker('["b", "a"]')

        reranker.rerank("query", make_candidates("a", "b"))

        assert reranker.fallback_used is False

    def test_a_single_candidate_is_returned_unchanged(self):
        reranker = make_reranker()

        ranked = reranker.rerank("query", make_candidates("a"))

        assert [candidate["id"] for candidate in ranked] == ["a"]
        assert reranker.llm.prompts == []

    def test_a_fenced_answer_is_accepted(self):
        reranker = make_reranker('```json\n["b", "a"]\n```')

        ranked = reranker.rerank("query", make_candidates("a", "b"))

        assert [candidate["id"] for candidate in ranked] == ["b", "a"]


class TestSchemaValidation:
    def test_an_answer_that_is_not_json_is_rejected(self):
        reranker = make_reranker("the most relevant one is b")

        with pytest.raises(RerankerError, match="not valid JSON"):
            reranker.rerank("query", make_candidates("a", "b"))

    def test_an_answer_that_is_not_an_array_is_rejected(self):
        reranker = make_reranker('{"id": "b"}')

        with pytest.raises(RerankerError, match="JSON array"):
            reranker.rerank("query", make_candidates("a", "b"))

    def test_an_answer_with_non_string_ids_is_rejected(self):
        reranker = make_reranker('[{"id": "b", "score": 1}]')

        with pytest.raises(RerankerError, match="only hold candidate ids as strings"):
            reranker.rerank("query", make_candidates("a", "b"))

    def test_an_answer_naming_an_unknown_id_is_rejected(self):
        reranker = make_reranker('["b", "z"]')

        with pytest.raises(RerankerError, match="unknown candidate ids: z"):
            reranker.rerank("query", make_candidates("a", "b"))

    def test_an_answer_repeating_an_id_is_rejected(self):
        reranker = make_reranker('["b", "b", "a"]')

        with pytest.raises(RerankerError, match="repeats candidate ids"):
            reranker.rerank("query", make_candidates("a", "b"))

    def test_an_answer_leaving_out_a_candidate_is_rejected(self):
        reranker = make_reranker('["b"]')

        with pytest.raises(RerankerError, match="leaves out candidate ids: a"):
            reranker.rerank("query", make_candidates("a", "b"))

    def test_a_rejected_answer_marks_the_fallback(self):
        reranker = make_reranker("not json at all")

        with pytest.raises(RerankerError):
            reranker.rerank("query", make_candidates("a", "b"))

        assert reranker.fallback_used is True


class TestFallback:
    def test_a_failing_llm_keeps_the_original_order(self):
        reranker = make_reranker(error=LLMError("provider is down"))

        ranked = reranker.rerank("query", make_candidates("a", "b"))

        assert [candidate["id"] for candidate in ranked] == ["a", "b"]

    def test_a_failing_llm_marks_the_fallback(self):
        reranker = make_reranker(error=LLMError("provider is down"))

        reranker.rerank("query", make_candidates("a", "b"))

        assert reranker.fallback_used is True


class TestInputValidation:
    def test_an_empty_query_is_rejected(self):
        with pytest.raises(ValueError):
            make_reranker().rerank("   ", make_candidates("a"))

    def test_an_empty_candidate_list_is_rejected(self):
        with pytest.raises(ValueError):
            make_reranker().rerank("query", [])

    def test_a_candidate_without_an_id_is_rejected(self):
        with pytest.raises(RerankerError, match="carries no id"):
            make_reranker().rerank("query", [{"text": "no id here"}])


class TestFactoryRouting:
    def test_the_factory_creates_the_llm_reranker(self):
        reranker = RerankerFactory.create(make_settings("llm"))

        assert isinstance(reranker, LLMReranker)

    def test_the_strategy_name_is_matched_case_insensitively(self):
        reranker = RerankerFactory.create(make_settings("LLM"))

        assert isinstance(reranker, LLMReranker)

    def test_a_disabled_rerank_falls_back_to_the_none_reranker(self):
        reranker = RerankerFactory.create(make_settings("llm", enabled=False))

        assert reranker.__class__.__name__ == "NoneReranker"
