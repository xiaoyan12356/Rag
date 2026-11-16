"""Unit tests for the sparse encoder.

The encoder counts the terms of every chunk, which is what the BM25 index
is built from. The tests pin the shape of that count and the behaviour on
text that holds nothing.
"""

from __future__ import annotations

import pytest

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.sparse_encoder import SparseEncoder


def make_chunk(text: str = "Some text to count.", chunk_id: str = "chunk_1") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "data/documents/sample.pdf"},
        source_ref="doc_1",
    )


class TestEncode:
    def test_every_chunk_gets_statistics(self):
        encoder = SparseEncoder()

        stats = encoder.encode([make_chunk("one"), make_chunk("two")])

        assert len(stats) == 2

    def test_the_statistics_name_the_chunk(self):
        encoder = SparseEncoder()

        stats = encoder.encode([make_chunk("text", "chunk_7")])

        assert stats[0]["chunk_id"] == "chunk_7"

    def test_the_terms_are_counted(self):
        encoder = SparseEncoder()

        stats = encoder.encode([make_chunk("hello world hello")])

        assert stats[0]["term_frequencies"]["hello"] == 2
        assert stats[0]["term_frequencies"]["world"] == 1

    def test_the_length_of_the_chunk_is_counted(self):
        encoder = SparseEncoder()

        stats = encoder.encode([make_chunk("hello world hello")])

        assert stats[0]["doc_length"] == 3

    def test_the_number_of_different_terms_is_counted(self):
        encoder = SparseEncoder()

        stats = encoder.encode([make_chunk("hello world hello")])

        assert stats[0]["unique_terms"] == 2

    def test_the_terms_are_lowercased(self):
        encoder = SparseEncoder()

        stats = encoder.encode([make_chunk("Hello World")])

        assert "hello" in stats[0]["term_frequencies"]
        assert "Hello" not in stats[0]["term_frequencies"]

    def test_the_case_can_be_kept(self):
        encoder = SparseEncoder(lowercase=False)

        stats = encoder.encode([make_chunk("Hello World")])

        assert "Hello" in stats[0]["term_frequencies"]

    def test_the_punctuation_is_not_a_term(self):
        encoder = SparseEncoder()

        stats = encoder.encode([make_chunk("hello, world!")])

        assert "," not in stats[0]["term_frequencies"]
        assert "!" not in stats[0]["term_frequencies"]

    def test_a_short_term_is_left_out(self):
        encoder = SparseEncoder(min_term_length=3)

        stats = encoder.encode([make_chunk("a bb ccc")])

        assert "bb" not in stats[0]["term_frequencies"]
        assert "ccc" in stats[0]["term_frequencies"]

    def test_the_chinese_text_is_cut_into_words(self):
        encoder = SparseEncoder()

        stats = encoder.encode([make_chunk("机器学习是人工智能的一个分支")])

        assert stats[0]["doc_length"] > 1
        assert len(stats[0]["term_frequencies"]) > 1

    def test_a_chunk_without_text_holds_no_terms(self):
        encoder = SparseEncoder()

        stats = encoder.encode([make_chunk("")])

        assert stats[0]["term_frequencies"] == {}
        assert stats[0]["doc_length"] == 0
        assert stats[0]["unique_terms"] == 0

    def test_a_chunk_of_blanks_holds_no_terms(self):
        encoder = SparseEncoder()

        stats = encoder.encode([make_chunk("   \n  ")])

        assert stats[0]["term_frequencies"] == {}
        assert stats[0]["doc_length"] == 0

    def test_the_statistics_keep_the_order_of_the_chunks(self):
        encoder = SparseEncoder()
        chunks = [make_chunk("alpha", "chunk_1"), make_chunk("beta", "chunk_2")]

        stats = encoder.encode(chunks)

        assert [item["chunk_id"] for item in stats] == ["chunk_1", "chunk_2"]

    def test_no_chunks_gives_no_statistics(self):
        encoder = SparseEncoder()

        with pytest.raises(ValueError, match="empty"):
            encoder.encode([])

    def test_a_minimum_length_of_zero_is_refused(self):
        with pytest.raises(ValueError, match="min_term_length"):
            SparseEncoder(min_term_length=0)


class TestCorpusStats:
    def test_the_number_of_documents_is_counted(self):
        encoder = SparseEncoder()
        stats = encoder.encode([make_chunk("alpha"), make_chunk("beta")])

        corpus = encoder.get_corpus_stats(stats)

        assert corpus["num_docs"] == 2

    def test_the_average_length_is_counted(self):
        encoder = SparseEncoder()
        stats = encoder.encode([make_chunk("one two"), make_chunk("three four five")])

        corpus = encoder.get_corpus_stats(stats)

        assert corpus["avg_doc_length"] == 2.5

    def test_a_term_of_every_document_is_counted_once_per_document(self):
        encoder = SparseEncoder()
        stats = encoder.encode([make_chunk("shared alpha"), make_chunk("shared beta")])

        corpus = encoder.get_corpus_stats(stats)

        assert corpus["document_frequency"]["shared"] == 2
        assert corpus["document_frequency"]["alpha"] == 1

    def test_nothing_gives_empty_statistics(self):
        encoder = SparseEncoder()

        corpus = encoder.get_corpus_stats([])

        assert corpus == {"num_docs": 0, "avg_doc_length": 0.0, "document_frequency": {}}


class TestTrace:
    def test_the_trace_records_what_happened(self):
        encoder = SparseEncoder()
        trace = TraceContext()

        encoder.encode([make_chunk("hello world"), make_chunk("hello there")], trace)

        data = trace.get_stage_data("sparse_encoder")
        assert data["total_chunks"] == 2
        assert data["total_terms"] == 4
        assert data["unique_terms"] == 3
