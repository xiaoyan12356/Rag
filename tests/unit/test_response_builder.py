"""Unit tests for the response of a tool.

The tests pin what a caller may rely on: the text carries the citation
markers `[1]`, `[2]`, ... that point at the structured citations, every
citation names the source, the page, the chunk and the score, and a search
that found nothing says so instead of answering with an empty list.
"""

from __future__ import annotations

import pytest

from core.response.citation_generator import CitationGenerator
from core.response.multimodal_assembler import MultimodalAssembler
from core.response.response_builder import ResponseBuilder
from core.types import RetrievalResult


def make_results() -> list[RetrievalResult]:
    """Return two results that come from different documents."""
    return [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.91,
            text="The first passage answers the question.",
            metadata={"source_path": "docs/guide.pdf", "page": 3},
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            score=0.75,
            text="The second passage says something else.",
            metadata={"source_path": "docs/other.md"},
        ),
    ]


class TestTheText:
    def test_the_passages_are_in_the_text(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "how does it work?")

        assert "The first passage answers the question." in result.content[0]["text"]

    def test_the_first_passage_is_marked_with_one(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "how does it work?")

        assert "[1]" in result.content[0]["text"]

    def test_the_second_passage_is_marked_with_two(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "how does it work?")

        assert "[2]" in result.content[0]["text"]

    def test_the_question_is_in_the_text(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "how does it work?")

        assert "how does it work?" in result.content[0]["text"]

    def test_a_long_passage_is_shortened(self):
        builder = ResponseBuilder(snippet_max_length=20)
        results = [
            RetrievalResult(
                chunk_id="long",
                score=0.5,
                text="word " * 50,
                metadata={"source_path": "a.txt"},
            )
        ]

        result = builder.build(results, "q")

        text = result.content[0]["text"]
        assert len(text) < 400
        assert "..." in text

    def test_the_content_block_is_a_text_block(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "q")

        assert result.content[0]["type"] == "text"


class TestTheCitations:
    def test_there_is_one_citation_per_result(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "q")

        assert len(result.structured_content["citations"]) == 2

    def test_a_citation_names_the_source(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "q")

        assert result.structured_content["citations"][0]["source"] == "guide.pdf"

    def test_a_citation_names_the_page(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "q")

        assert result.structured_content["citations"][0]["page"] == 3

    def test_a_citation_names_the_chunk(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "q")

        assert result.structured_content["citations"][0]["chunk_id"] == "chunk-1"

    def test_a_citation_carries_the_score(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "q")

        assert result.structured_content["citations"][0]["score"] == pytest.approx(0.91)

    def test_the_citations_are_numbered_from_one(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "q")

        citations = result.structured_content["citations"]
        assert [citation["index"] for citation in citations] == [1, 2]

    def test_a_result_without_a_page_has_no_page(self):
        builder = ResponseBuilder()

        result = builder.build(make_results(), "q")

        assert result.structured_content["citations"][1]["page"] is None

    def test_the_generator_numbers_the_citations_from_one(self):
        generator = CitationGenerator()

        citations = generator.generate(make_results())

        assert [citation.index for citation in citations] == [1, 2]

    def test_the_generator_takes_the_page_of_an_image_when_there_is_none(self):
        generator = CitationGenerator()
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.5,
                text="text",
                metadata={"images": [{"id": "img-1", "path": "a.png", "page": 7}]},
            )
        ]

        citations = generator.generate(results)

        assert citations[0].page == 7

    def test_a_result_without_a_source_is_unknown(self):
        generator = CitationGenerator()

        citations = generator.generate([RetrievalResult(chunk_id="c", score=0.1)])

        assert citations[0].source == "unknown"


class TestNothingFound:
    def test_the_text_says_that_nothing_was_found(self):
        builder = ResponseBuilder()

        result = builder.build([], "what is the answer?")

        assert "No results" in result.content[0]["text"]

    def test_the_text_is_not_an_empty_list(self):
        builder = ResponseBuilder()

        result = builder.build([], "what is the answer?")

        assert result.content[0]["text"].strip() != ""

    def test_the_citations_are_empty(self):
        builder = ResponseBuilder()

        result = builder.build([], "q")

        assert result.structured_content["citations"] == []

    def test_the_number_of_results_is_zero(self):
        builder = ResponseBuilder()

        result = builder.build([], "q")

        assert result.structured_content["result_count"] == 0

    def test_the_result_is_not_an_error(self):
        builder = ResponseBuilder()

        result = builder.build([], "q")

        assert result.is_error is False


class TestTheMultimodalPart:
    def test_an_image_of_a_passage_becomes_an_image_block(self, tmp_path):
        image = tmp_path / "figure.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.5,
                text="text",
                metadata={"images": [{"id": "img-1", "path": str(image)}]},
            )
        ]

        blocks = assembler.assemble(results)

        assert blocks[0]["type"] == "image"
        assert blocks[0]["mimeType"] == "image/png"

    def test_the_data_of_an_image_is_base64(self, tmp_path):
        image = tmp_path / "figure.png"
        image.write_bytes(b"hello")
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.5,
                text="text",
                metadata={"images": [{"id": "img-1", "path": str(image)}]},
            )
        ]

        blocks = assembler.assemble(results)

        assert blocks[0]["data"] == "aGVsbG8="

    def test_an_image_that_is_not_there_is_left_out(self, tmp_path):
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.5,
                text="text",
                metadata={"images": [{"id": "img-1", "path": str(tmp_path / "gone.png")}]},
            )
        ]

        assert assembler.assemble(results) == []

    def test_the_same_image_is_not_returned_twice(self, tmp_path):
        image = tmp_path / "figure.png"
        image.write_bytes(b"hello")
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.5,
                text="text",
                metadata={"images": [{"id": "img-1", "path": str(image)}]},
            ),
            RetrievalResult(
                chunk_id="chunk-2",
                score=0.4,
                text="text",
                metadata={"images": [{"id": "img-1", "path": str(image)}]},
            ),
        ]

        assert len(assembler.assemble(results)) == 1
