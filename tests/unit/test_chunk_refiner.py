"""Unit tests for the chunk refiner.

The refiner takes the marks a PDF leaves on its pages off a chunk. The
tests pin two things: the rules clean what they can recognise and leave
everything else alone, and the model is only ever an addition - when it
fails, the chunk still comes out the other side.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.settings import LLMSettings, Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
from libs.llm.base_llm import BaseLLM

FIXTURES = Path(__file__).parent.parent / "fixtures" / "noisy_chunks.json"


@pytest.fixture(scope="module")
def scenarios() -> dict:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


class FakeLLM(BaseLLM):
    """Model that answers what the test tells it to."""

    def __init__(self, answer: str = "The refined text.", error: Exception | None = None) -> None:
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
    settings.ingestion.chunk_refiner.use_llm = use_llm
    return settings


def make_chunk(text: str = "Some text to refine.") -> Chunk:
    return Chunk(
        id="chunk_1",
        text=text,
        metadata={"source_path": "data/documents/sample.pdf", "chunk_index": 0},
        source_ref="doc_1",
    )


class TestRuleBasedRefine:
    @pytest.mark.parametrize(
        "scenario",
        [
            "typical_noise_scenario",
            "ocr_errors",
            "page_header_footer",
            "excessive_whitespace",
            "format_markers",
            "clean_text",
            "code_blocks",
            "mixed_noise",
        ],
    )
    def test_the_scenario_is_cleaned_as_expected(self, scenarios: dict, scenario: str):
        refiner = ChunkRefiner(make_settings())
        case = scenarios[scenario]

        assert refiner._rule_based_refine(case["input"]) == case["expected_clean"]

    def test_the_page_header_is_removed(self, scenarios: dict):
        refiner = ChunkRefiner(make_settings())

        cleaned = refiner._rule_based_refine(scenarios["page_header_footer"]["input"])

        assert "Page 42" not in cleaned
        assert "Company Name" not in cleaned

    def test_the_page_footer_is_removed(self, scenarios: dict):
        refiner = ChunkRefiner(make_settings())

        cleaned = refiner._rule_based_refine(scenarios["typical_noise_scenario"]["input"])

        assert "Confidential" not in cleaned

    def test_the_html_comment_is_removed(self, scenarios: dict):
        refiner = ChunkRefiner(make_settings())

        cleaned = refiner._rule_based_refine(scenarios["format_markers"]["input"])

        assert "comment" not in cleaned

    def test_the_html_tag_is_removed(self, scenarios: dict):
        refiner = ChunkRefiner(make_settings())

        cleaned = refiner._rule_based_refine(scenarios["format_markers"]["input"])

        assert "<div" not in cleaned

    def test_the_runs_of_spaces_are_collapsed(self, scenarios: dict):
        refiner = ChunkRefiner(make_settings())

        cleaned = refiner._rule_based_refine(scenarios["excessive_whitespace"]["input"])

        assert "    " not in cleaned

    def test_the_separator_line_is_removed(self, scenarios: dict):
        refiner = ChunkRefiner(make_settings())

        cleaned = refiner._rule_based_refine(scenarios["page_header_footer"]["input"])

        assert "─" not in cleaned

    def test_the_code_block_is_kept_as_it_is(self, scenarios: dict):
        refiner = ChunkRefiner(make_settings())

        cleaned = refiner._rule_based_refine(scenarios["code_blocks"]["input"])

        assert "def    hello_world():" in cleaned
        assert 'print("Hello,    World!")' in cleaned

    def test_the_markdown_is_kept(self, scenarios: dict):
        refiner = ChunkRefiner(make_settings())

        cleaned = refiner._rule_based_refine(scenarios["mixed_noise"]["input"])

        assert "## Real Heading" in cleaned
        assert "- Extra spaces" in cleaned

    def test_text_that_is_already_clean_is_left_alone(self, scenarios: dict):
        refiner = ChunkRefiner(make_settings())

        cleaned = refiner._rule_based_refine(scenarios["clean_text"]["input"])

        assert cleaned == scenarios["clean_text"]["input"]

    def test_the_words_of_an_ocr_text_are_left_alone(self, scenarios: dict):
        refiner = ChunkRefiner(make_settings())

        cleaned = refiner._rule_based_refine(scenarios["ocr_errors"]["input"])

        assert cleaned == scenarios["ocr_errors"]["input"]

    def test_an_empty_text_stays_empty(self):
        refiner = ChunkRefiner(make_settings())

        assert refiner._rule_based_refine("") == ""

    def test_a_text_of_blanks_becomes_empty(self):
        refiner = ChunkRefiner(make_settings())

        assert refiner._rule_based_refine("   \n\n  ") == ""

    def test_a_nested_list_keeps_its_indentation(self):
        refiner = ChunkRefiner(make_settings())

        cleaned = refiner._rule_based_refine("- First item\n  - Nested item\n")

        assert "  - Nested item" in cleaned


class TestTransform:
    def test_every_chunk_comes_back(self):
        refiner = ChunkRefiner(make_settings())
        chunks = [make_chunk("First text."), make_chunk("Second text.")]

        refined = refiner.transform(chunks)

        assert len(refined) == 2

    def test_the_text_is_cleaned(self, scenarios: dict):
        refiner = ChunkRefiner(make_settings())

        refined = refiner.transform(
            [make_chunk(scenarios["typical_noise_scenario"]["input"])]
        )

        assert refined[0].text == scenarios["typical_noise_scenario"]["expected_clean"]

    def test_the_metadata_says_the_rules_did_it(self):
        refiner = ChunkRefiner(make_settings())

        refined = refiner.transform([make_chunk("Text.")])

        assert refined[0].metadata["refined_by"] == "rule"

    def test_the_metadata_of_the_chunk_is_kept(self):
        refiner = ChunkRefiner(make_settings())

        refined = refiner.transform([make_chunk("Text.")])

        assert refined[0].metadata["source_path"] == "data/documents/sample.pdf"
        assert refined[0].metadata["chunk_index"] == 0

    def test_the_id_and_the_source_are_kept(self):
        refiner = ChunkRefiner(make_settings())

        refined = refiner.transform([make_chunk("Text.")])

        assert refined[0].id == "chunk_1"
        assert refined[0].source_ref == "doc_1"

    def test_the_chunk_that_came_in_is_left_alone(self):
        refiner = ChunkRefiner(make_settings())
        chunk = make_chunk("Text.")

        refiner.transform([chunk])

        assert "refined_by" not in chunk.metadata

    def test_no_chunks_gives_no_chunks(self):
        refiner = ChunkRefiner(make_settings())

        assert refiner.transform([]) == []

    def test_the_trace_records_what_happened(self):
        refiner = ChunkRefiner(make_settings())
        trace = TraceContext()

        refiner.transform([make_chunk("Text.")], trace)

        data = trace.get_stage_data("chunk_refiner")
        assert data["total_chunks"] == 1
        assert data["refined_by_rule"] == 1
        assert data["refined_by_llm"] == 0


class TestLlmMode:
    def test_the_model_is_asked_when_the_switch_is_on(self):
        llm = FakeLLM()
        refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

        refiner.transform([make_chunk("Text.")])

        assert len(llm.calls) == 1

    def test_the_rewritten_text_is_used(self):
        llm = FakeLLM(answer="A self-contained sentence.")
        refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

        refined = refiner.transform([make_chunk("Text.")])

        assert refined[0].text == "A self-contained sentence."

    def test_the_metadata_says_the_model_did_it(self):
        llm = FakeLLM()
        refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

        refined = refiner.transform([make_chunk("Text.")])

        assert refined[0].metadata["refined_by"] == "llm"

    def test_the_model_is_given_the_cleaned_text(self, scenarios: dict):
        llm = FakeLLM()
        refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

        refiner.transform([make_chunk(scenarios["typical_noise_scenario"]["input"])])

        prompt = llm.calls[0][0]["content"]
        assert "Page 15" not in prompt
        assert "valuable paragraph" in prompt

    def test_the_prompt_asks_for_the_text_to_be_rewritten(self):
        llm = FakeLLM()
        refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

        refiner.transform([make_chunk("Text.")])

        assert "self-contained" in llm.calls[0][0]["content"]

    def test_the_model_is_not_asked_when_the_switch_is_off(self):
        llm = FakeLLM()
        refiner = ChunkRefiner(make_settings(use_llm=False), llm=llm)

        refiner.transform([make_chunk("Text.")])

        assert llm.calls == []

    def test_the_model_is_not_asked_when_the_switch_is_off_and_none_was_given(self):
        refiner = ChunkRefiner(make_settings(use_llm=False))

        refined = refiner.transform([make_chunk("Text.")])

        assert refined[0].metadata["refined_by"] == "rule"


class TestFallback:
    def test_a_failing_model_falls_back_to_the_rules(self, scenarios: dict):
        llm = FakeLLM(error=RuntimeError("the model is down"))
        refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

        refined = refiner.transform(
            [make_chunk(scenarios["typical_noise_scenario"]["input"])]
        )

        assert refined[0].text == scenarios["typical_noise_scenario"]["expected_clean"]

    def test_a_failing_model_is_marked_as_a_fallback(self):
        llm = FakeLLM(error=RuntimeError("the model is down"))
        refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

        refined = refiner.transform([make_chunk("Text.")])

        assert refined[0].metadata["refined_by"] == "rule"
        assert refined[0].metadata["refine_fallback_reason"] == "llm_failed"

    def test_an_empty_answer_falls_back_to_the_rules(self):
        llm = FakeLLM(answer="   ")
        refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

        refined = refiner.transform([make_chunk("Text.")])

        assert refined[0].metadata["refined_by"] == "rule"
        assert refined[0].metadata["refine_fallback_reason"] == "llm_failed"

    def test_a_missing_prompt_falls_back_to_the_rules(self, tmp_path: Path):
        llm = FakeLLM()
        refiner = ChunkRefiner(
            make_settings(use_llm=True),
            llm=llm,
            prompt_path=tmp_path / "nowhere.txt",
        )

        refined = refiner.transform([make_chunk("Text.")])

        assert refined[0].metadata["refined_by"] == "rule"

    def test_a_prompt_without_a_placeholder_falls_back_to_the_rules(self, tmp_path: Path):
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("Rewrite this chunk.", encoding="utf-8")
        llm = FakeLLM()
        refiner = ChunkRefiner(
            make_settings(use_llm=True),
            llm=llm,
            prompt_path=prompt,
        )

        refined = refiner.transform([make_chunk("Text.")])

        assert refined[0].metadata["refined_by"] == "rule"
        assert llm.calls == []

    def test_a_chunk_that_breaks_keeps_its_text(self, monkeypatch):
        def explode(self, text: str) -> str:
            if "BOOM" in text:
                raise ValueError("this chunk is broken")
            return text

        monkeypatch.setattr(ChunkRefiner, "_rule_based_refine", explode)
        refiner = ChunkRefiner(make_settings())

        refined = refiner.transform([make_chunk("BOOM")])

        assert refined[0].text == "BOOM"

    def test_one_broken_chunk_does_not_stop_the_others(self, monkeypatch):
        def explode(self, text: str) -> str:
            if "BOOM" in text:
                raise ValueError("this chunk is broken")
            return text

        monkeypatch.setattr(ChunkRefiner, "_rule_based_refine", explode)
        refiner = ChunkRefiner(make_settings())
        chunks = [make_chunk("BOOM"), make_chunk("Fine text.")]

        refined = refiner.transform(chunks)

        assert refined[0].text == "BOOM"
        assert refined[1].text == "Fine text."

    def test_the_trace_counts_the_fallbacks(self):
        llm = FakeLLM(error=RuntimeError("the model is down"))
        refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)
        trace = TraceContext()

        refiner.transform([make_chunk("Text.")], trace)

        data = trace.get_stage_data("chunk_refiner")
        assert data["refined_by_rule"] == 1
        assert data["refined_by_llm"] == 0


class TestPrompt:
    def test_the_prompt_of_the_repository_is_used(self):
        refiner = ChunkRefiner(make_settings())

        template = refiner._load_prompt()

        assert template is not None
        assert "{text}" in template

    def test_the_prompt_is_read_once(self):
        refiner = ChunkRefiner(make_settings())

        first = refiner._load_prompt()
        second = refiner._load_prompt()

        assert first is second

    def test_a_missing_prompt_is_reported(self, tmp_path: Path):
        refiner = ChunkRefiner(make_settings(), prompt_path=tmp_path / "nowhere.txt")

        assert refiner._load_prompt() is None


class TestBaseTransform:
    def test_the_base_class_cannot_be_used_on_its_own(self):
        with pytest.raises(TypeError, match="abstract"):
            BaseTransform()

    def test_a_refiner_is_a_transform(self):
        assert isinstance(ChunkRefiner(make_settings()), BaseTransform)
