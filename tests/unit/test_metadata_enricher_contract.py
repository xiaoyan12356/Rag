"""Contract tests for the metadata enricher.

The enricher adds a title, a summary and tags to every chunk. The tests pin
two things: the rules always produce all three, and the model is only ever an
addition - when it fails, the rules carry the chunk on their own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.settings import LLMSettings, Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.llm.base_llm import BaseLLM

ANSWER = json.dumps(
    {
        "title": "Vector upsert idempotency",
        "summary": "Every chunk is written under an id derived from its content.",
        "tags": ["upsert", "idempotency"],
    }
)


class FakeLLM(BaseLLM):
    """Model that answers what the test tells it to."""

    def __init__(self, answer: str = ANSWER, error: Exception | None = None) -> None:
        super().__init__(LLMSettings(provider="fake", model="fake"))
        self.answer = answer
        self.error = error
        self.calls: list[list[dict]] = []

    def chat(self, messages, trace=None) -> str:
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.answer


def make_settings(use_llm: bool = False) -> Settings:
    settings = Settings()
    settings.ingestion.metadata_enricher.use_llm = use_llm
    return settings


def make_chunk(text: str = "Some text to describe.", chunk_id: str = "chunk_1") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "data/documents/sample.pdf", "chunk_index": 0},
        source_ref="doc_1",
    )


class TestRuleBasedEnrichment:
    def test_the_title_comes_from_the_heading(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform([make_chunk("# Introduction to Machine Learning")])

        assert enriched[0].metadata["title"] == "Introduction to Machine Learning"

    def test_the_title_comes_from_the_first_line(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform([make_chunk("Quick Start Guide\n\nFollow these steps.")])

        assert enriched[0].metadata["title"] == "Quick Start Guide"

    def test_the_title_comes_from_the_first_sentence(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform(
            [make_chunk("This is a test document about Python. It covers basic concepts.")]
        )

        assert enriched[0].metadata["title"] == "This is a test document about Python"

    def test_the_summary_holds_the_first_sentences(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform([make_chunk("One. Two. Three. Four. Five.")])

        assert enriched[0].metadata["summary"] == "One. Two. Three."

    def test_the_tags_hold_the_words_of_the_chunk(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform(
            [make_chunk("The Gateway Service routes traffic to the Authentication Service.")]
        )

        tags = enriched[0].metadata["tags"]
        assert "Gateway Service" in tags
        assert "Authentication Service" in tags

    def test_the_tags_hold_the_identifiers_of_the_chunk(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform(
            [make_chunk("The getUserById function calls async_fetch under the hood.")]
        )

        tags = enriched[0].metadata["tags"]
        assert "getUserById" in tags
        assert "async_fetch" in tags

    def test_the_tags_hold_the_emphasised_words(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform([make_chunk("A **bold** word and an *italic* one.")])

        tags = enriched[0].metadata["tags"]
        assert "bold" in tags
        assert "italic" in tags

    def test_the_common_words_are_not_tags(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform(
            [make_chunk("This is the document and the text that it has.")]
        )

        assert "the" not in [tag.lower() for tag in enriched[0].metadata["tags"]]

    def test_every_chunk_gets_a_title_a_summary_and_tags(self):
        enricher = MetadataEnricher(make_settings())
        chunks = [
            make_chunk("# A heading\n\nSome text about the ingestion pipeline."),
            make_chunk("Plain prose that says something useful about retrieval."),
        ]

        enriched = enricher.transform(chunks)

        for chunk in enriched:
            assert chunk.metadata["title"]
            assert chunk.metadata["summary"]
            assert chunk.metadata["tags"]

    def test_an_empty_chunk_gets_a_title(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform([make_chunk("")])

        assert enriched[0].metadata["title"] == "Untitled"
        assert enriched[0].metadata["summary"] == ""
        assert enriched[0].metadata["tags"] == []

    def test_the_metadata_says_the_rules_did_it(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform([make_chunk("Text.")])

        assert enriched[0].metadata["enriched_by"] == "rule"


class TestTransform:
    def test_every_chunk_comes_back(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform([make_chunk("First."), make_chunk("Second.")])

        assert len(enriched) == 2

    def test_no_chunks_gives_no_chunks(self):
        enricher = MetadataEnricher(make_settings())

        assert enricher.transform([]) == []

    def test_the_text_is_left_alone(self):
        enricher = MetadataEnricher(make_settings())
        chunk = make_chunk("The text stays as it is.")

        enriched = enricher.transform([chunk])

        assert enriched[0].text == chunk.text

    def test_the_id_and_the_source_are_kept(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform([make_chunk("Text.")])

        assert enriched[0].id == "chunk_1"
        assert enriched[0].source_ref == "doc_1"

    def test_the_metadata_of_the_chunk_is_kept(self):
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform([make_chunk("Text.")])

        assert enriched[0].metadata["source_path"] == "data/documents/sample.pdf"
        assert enriched[0].metadata["chunk_index"] == 0

    def test_the_chunk_that_came_in_is_left_alone(self):
        enricher = MetadataEnricher(make_settings())
        chunk = make_chunk("Text.")

        enricher.transform([chunk])

        assert "title" not in chunk.metadata
        assert "enriched_by" not in chunk.metadata

    def test_the_trace_records_what_happened(self):
        enricher = MetadataEnricher(make_settings())
        trace = TraceContext()

        enricher.transform([make_chunk("Text.")], trace)

        data = trace.get_stage_data("metadata_enricher")
        assert data["total_chunks"] == 1
        assert data["enriched_by_rule"] == 1
        assert data["enriched_by_llm"] == 0
        assert data["failed"] == 0


class TestLlmMode:
    def test_the_model_is_asked_when_the_switch_is_on(self):
        llm = FakeLLM()
        enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

        enricher.transform([make_chunk("Text.")])

        assert len(llm.calls) == 1

    def test_the_metadata_of_the_model_is_used(self):
        llm = FakeLLM()
        enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

        enriched = enricher.transform([make_chunk("Text.")])

        assert enriched[0].metadata["title"] == "Vector upsert idempotency"
        assert enriched[0].metadata["summary"] == (
            "Every chunk is written under an id derived from its content."
        )
        assert enriched[0].metadata["tags"] == ["upsert", "idempotency"]

    def test_the_metadata_says_the_model_did_it(self):
        llm = FakeLLM()
        enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

        enriched = enricher.transform([make_chunk("Text.")])

        assert enriched[0].metadata["enriched_by"] == "llm"

    def test_the_model_is_given_the_text_of_the_chunk(self):
        llm = FakeLLM()
        enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

        enricher.transform([make_chunk("A chunk about the ingestion pipeline.")])

        prompt = llm.calls[0][0]["content"]
        assert "A chunk about the ingestion pipeline." in prompt

    def test_the_prompt_asks_for_the_metadata(self):
        llm = FakeLLM()
        enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

        enricher.transform([make_chunk("Text.")])

        assert "title" in llm.calls[0][0]["content"]
        assert "summary" in llm.calls[0][0]["content"]
        assert "tags" in llm.calls[0][0]["content"]

    def test_the_model_is_not_asked_when_the_switch_is_off(self):
        llm = FakeLLM()
        enricher = MetadataEnricher(make_settings(use_llm=False), llm=llm)

        enricher.transform([make_chunk("Text.")])

        assert llm.calls == []

    def test_a_fenced_answer_is_read(self):
        llm = FakeLLM(answer=f"```json\n{ANSWER}\n```")
        enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

        enriched = enricher.transform([make_chunk("Text.")])

        assert enriched[0].metadata["title"] == "Vector upsert idempotency"


class TestFallback:
    def test_a_failing_model_falls_back_to_the_rules(self):
        llm = FakeLLM(error=RuntimeError("the model is down"))
        enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

        enriched = enricher.transform([make_chunk("# A heading\n\nThe body of the chunk.")])

        assert enriched[0].metadata["title"] == "A heading"
        assert enriched[0].metadata["summary"] == "The body of the chunk."

    def test_a_failing_model_is_marked_as_a_fallback(self):
        llm = FakeLLM(error=RuntimeError("the model is down"))
        enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

        enriched = enricher.transform([make_chunk("Text.")])

        assert enriched[0].metadata["enriched_by"] == "rule"
        assert enriched[0].metadata["enrich_fallback_reason"] == "llm_failed"

    def test_an_empty_answer_falls_back_to_the_rules(self):
        llm = FakeLLM(answer="   ")
        enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

        enriched = enricher.transform([make_chunk("Text.")])

        assert enriched[0].metadata["enriched_by"] == "rule"
        assert enriched[0].metadata["enrich_fallback_reason"] == "llm_failed"

    def test_an_answer_that_is_not_json_falls_back_to_the_rules(self):
        llm = FakeLLM(answer="I am afraid I cannot do that.")
        enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

        enriched = enricher.transform([make_chunk("Text.")])

        assert enriched[0].metadata["enriched_by"] == "rule"
        assert enriched[0].metadata["enrich_fallback_reason"] == "llm_failed"

    def test_a_missing_prompt_falls_back_to_the_rules(self, tmp_path: Path):
        llm = FakeLLM()
        enricher = MetadataEnricher(
            make_settings(use_llm=True),
            llm=llm,
            prompt_path=tmp_path / "nowhere.txt",
        )

        enriched = enricher.transform([make_chunk("Text.")])

        assert enriched[0].metadata["enriched_by"] == "rule"
        assert llm.calls == []

    def test_a_prompt_without_a_placeholder_falls_back_to_the_rules(self, tmp_path: Path):
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("Describe this chunk.", encoding="utf-8")
        llm = FakeLLM()
        enricher = MetadataEnricher(
            make_settings(use_llm=True),
            llm=llm,
            prompt_path=prompt,
        )

        enriched = enricher.transform([make_chunk("Text.")])

        assert enriched[0].metadata["enriched_by"] == "rule"
        assert llm.calls == []

    def test_a_chunk_that_breaks_keeps_its_text(self, monkeypatch):
        def explode(self, text: str) -> dict:
            raise ValueError("this chunk is broken")

        monkeypatch.setattr(MetadataEnricher, "_rule_based_enrich", explode)
        enricher = MetadataEnricher(make_settings())

        enriched = enricher.transform([make_chunk("BOOM")])

        assert enriched[0].text == "BOOM"

    def test_one_broken_chunk_does_not_stop_the_others(self, monkeypatch):
        def explode(self, text: str) -> dict:
            if "BOOM" in text:
                raise ValueError("this chunk is broken")
            return {"title": "t", "summary": "s", "tags": ["a"]}

        monkeypatch.setattr(MetadataEnricher, "_rule_based_enrich", explode)
        enricher = MetadataEnricher(make_settings())
        chunks = [make_chunk("BOOM", "chunk_1"), make_chunk("Fine text.", "chunk_2")]

        enriched = enricher.transform(chunks)

        assert enriched[0].text == "BOOM"
        assert enriched[0].metadata["enriched_by"] == "failed"
        assert enriched[1].text == "Fine text."
        assert enriched[1].metadata["title"] == "t"

    def test_the_trace_counts_the_fallbacks(self):
        llm = FakeLLM(error=RuntimeError("the model is down"))
        enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)
        trace = TraceContext()

        enricher.transform([make_chunk("Text.")], trace)

        data = trace.get_stage_data("metadata_enricher")
        assert data["enriched_by_rule"] == 1
        assert data["enriched_by_llm"] == 0


class TestPrompt:
    def test_the_prompt_of_the_repository_is_used(self):
        enricher = MetadataEnricher(make_settings())

        template = enricher._load_prompt()

        assert template is not None
        assert "{chunk_text}" in template

    def test_the_prompt_is_read_once(self):
        enricher = MetadataEnricher(make_settings())

        first = enricher._load_prompt()
        second = enricher._load_prompt()

        assert first is second

    def test_a_missing_prompt_is_reported(self, tmp_path: Path):
        enricher = MetadataEnricher(make_settings(), prompt_path=tmp_path / "nowhere.txt")

        assert enricher._load_prompt() is None


class TestBaseTransform:
    def test_the_base_class_cannot_be_used_on_its_own(self):
        with pytest.raises(TypeError, match="abstract"):
            BaseTransform()

    def test_an_enricher_is_a_transform(self):
        assert isinstance(MetadataEnricher(make_settings()), BaseTransform)
