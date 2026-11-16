"""Unit tests for the BM25 index.

The index turns the term counts of the sparse encoder into something that can
be searched. The tests pin the rarity of a term, that the index survives a
trip through disk, and that a document can be added and taken away again.
"""

from __future__ import annotations

import math

import pytest

from core.trace.trace_context import TraceContext
from ingestion.storage.bm25_indexer import BM25Indexer


def make_stats(chunk_id: str, text: str) -> dict:
    """Return what the sparse encoder would say about a chunk of text."""
    terms = text.split()
    frequencies: dict[str, int] = {}
    for term in terms:
        frequencies[term] = frequencies.get(term, 0) + 1
    return {
        "chunk_id": chunk_id,
        "term_frequencies": frequencies,
        "doc_length": len(terms),
        "unique_terms": len(frequencies),
    }


CORPUS = [
    make_stats("doc1_0000", "the quick brown fox jumps over the lazy dog"),
    make_stats("doc1_0001", "the fox is quick and the dog is lazy"),
    make_stats("doc2_0000", "a completely different sentence about cats"),
]


@pytest.fixture
def indexer(tmp_path):
    return BM25Indexer(index_dir=tmp_path)


class TestBuild:
    def test_the_index_holds_every_chunk(self, indexer):
        indexer.build(CORPUS, "kb")

        assert indexer.get_document_count() == 3

    def test_the_index_is_written_to_disk(self, indexer):
        indexer.build(CORPUS, "kb")

        assert (indexer.index_dir / "kb_bm25.json").exists()

    def test_an_empty_corpus_is_refused(self, indexer):
        with pytest.raises(ValueError, match="empty"):
            indexer.build([], "kb")

    def test_a_stat_without_a_chunk_id_is_refused(self, indexer):
        with pytest.raises(ValueError, match="chunk_id"):
            indexer.build([{"term_frequencies": {}, "doc_length": 0}], "kb")

    def test_a_stat_without_a_document_length_is_refused(self, indexer):
        with pytest.raises(ValueError, match="doc_length"):
            indexer.build([{"chunk_id": "a", "term_frequencies": {}}], "kb")

    def test_a_negative_document_length_is_refused(self, indexer):
        with pytest.raises(ValueError, match="doc_length"):
            indexer.build([{"chunk_id": "a", "term_frequencies": {}, "doc_length": -1}], "kb")


class TestIdf:
    def test_the_rarity_of_a_term_is_the_formula(self, indexer):
        indexer.build(CORPUS, "kb")

        expected = math.log((3 - 2 + 0.5) / (2 + 0.5))
        postings = indexer._index["the"]["postings"]
        assert len(postings) == 2
        assert indexer._index["the"]["idf"] == pytest.approx(expected)

    def test_a_term_in_every_chunk_is_less_worth_than_a_rare_one(self, indexer):
        indexer.build(CORPUS, "kb")

        assert indexer._index["cats"]["idf"] > indexer._index["the"]["idf"]

    def test_the_number_of_documents_is_counted(self, indexer):
        indexer.build(CORPUS, "kb")

        metadata = indexer.get_metadata()
        assert metadata["num_docs"] == 3
        assert metadata["total_terms"] == 16


class TestRoundTrip:
    def test_an_index_can_be_read_back(self, indexer):
        indexer.build(CORPUS, "kb")

        other = BM25Indexer(index_dir=indexer.index_dir)
        assert other.load("kb") is True
        assert other.get_document_count() == 3

    def test_the_query_answers_the_same_after_a_round_trip(self, indexer):
        indexer.build(CORPUS, "kb")
        before = indexer.query(["fox"])

        other = BM25Indexer(index_dir=indexer.index_dir)
        other.load("kb")
        after = other.query(["fox"])

        assert before == after

    def test_an_index_that_is_not_there_is_reported(self, indexer):
        assert indexer.load("missing") is False

    def test_a_broken_index_file_is_refused(self, indexer):
        path = indexer.index_dir / "broken_bm25.json"
        indexer.index_dir.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(ValueError, match="cannot be read"):
            indexer.load("broken")

    def test_an_index_of_the_wrong_shape_is_refused(self, indexer):
        path = indexer.index_dir / "wrong_bm25.json"
        indexer.index_dir.mkdir(parents=True, exist_ok=True)
        path.write_text('{"something": "else"}', encoding="utf-8")

        with pytest.raises(ValueError, match="expected shape"):
            indexer.load("wrong")


class TestQuery:
    def test_the_chunk_with_the_term_is_found(self, indexer):
        indexer.build(CORPUS, "kb")

        results = indexer.query(["cats"])

        assert results[0]["chunk_id"] == "doc2_0000"

    def test_the_chunks_that_hold_the_term_are_found(self, indexer):
        indexer.build(CORPUS, "kb")

        results = indexer.query(["fox"])

        assert [item["chunk_id"] for item in results] == ["doc1_0000", "doc1_0001"]

    def test_a_chunk_that_says_the_term_more_often_scores_higher(self, indexer):
        # The term has to stay rare in the corpus, otherwise its rarity
        # turns negative and a short chunk wins over a chunk that repeats it.
        indexer.build(
            [
                make_stats("a_0000", "fox fox fox"),
                make_stats("b_0000", "fox"),
                make_stats("c_0000", "cat"),
                make_stats("d_0000", "dog"),
                make_stats("e_0000", "bird"),
            ],
            "kb",
        )

        results = indexer.query(["fox"])

        assert results[0]["chunk_id"] == "a_0000"
        assert results[0]["score"] > results[1]["score"]

    def test_a_term_that_is_not_there_finds_nothing(self, indexer):
        indexer.build(CORPUS, "kb")

        assert indexer.query(["elephant"]) == []

    def test_the_number_of_results_is_capped(self, indexer):
        indexer.build(CORPUS, "kb")

        assert len(indexer.query(["the"], top_k=1)) == 1

    def test_the_query_is_case_insensitive(self, indexer):
        indexer.build(CORPUS, "kb")

        assert indexer.query(["FOX"]) == indexer.query(["fox"])

    def test_an_index_that_is_empty_cannot_be_queried(self, indexer):
        with pytest.raises(ValueError, match="empty"):
            indexer.query(["fox"])

    def test_a_query_without_terms_is_refused(self, indexer):
        indexer.build(CORPUS, "kb")

        with pytest.raises(ValueError, match="query_terms"):
            indexer.query([])

    def test_a_top_k_of_zero_is_refused(self, indexer):
        indexer.build(CORPUS, "kb")

        with pytest.raises(ValueError, match="top_k"):
            indexer.query(["fox"], top_k=0)

    def test_the_same_query_gives_the_same_answer(self, indexer):
        indexer.build(CORPUS, "kb")

        assert indexer.query(["the", "fox"]) == indexer.query(["the", "fox"])


class TestIncremental:
    def test_a_document_can_be_added(self, indexer):
        indexer.build(CORPUS, "kb")

        indexer.add_documents([make_stats("doc3_0000", "fox fox fox")], "kb", doc_id="doc3")

        assert indexer.get_document_count() == 4

    def test_a_document_can_be_added_to_an_empty_index(self, indexer):
        indexer.add_documents([make_stats("doc1_0000", "hello world")], "kb")

        assert indexer.get_document_count() == 1
        assert indexer.query(["hello"])[0]["chunk_id"] == "doc1_0000"

    def test_adding_the_same_document_twice_leaves_one_copy(self, indexer):
        indexer.build(CORPUS, "kb")

        indexer.add_documents(
            [make_stats("doc1_0000", "rewritten text")], "kb", doc_id="doc1"
        )

        assert indexer.get_document_count() == 2
        assert indexer.query(["brown"]) == []

    def test_a_document_can_be_taken_away(self, indexer):
        indexer.build(CORPUS, "kb")

        assert indexer.remove_document("doc1", "kb") is True
        assert indexer.get_document_count() == 1

    def test_taking_away_a_document_that_is_not_there_reports_nothing(self, indexer):
        indexer.build(CORPUS, "kb")

        assert indexer.remove_document("nothing", "kb") is False

    def test_the_index_can_be_built_again(self, indexer):
        indexer.build(CORPUS, "kb")
        indexer.add_documents([make_stats("doc3_0000", "new text")], "kb", doc_id="doc3")

        indexer.rebuild(CORPUS, "kb")

        assert indexer.get_document_count() == 3

    def test_nothing_to_add_leaves_the_index_alone(self, indexer):
        indexer.build(CORPUS, "kb")

        indexer.add_documents([], "kb")

        assert indexer.get_document_count() == 3


class TestTrace:
    def test_the_trace_records_what_happened(self, indexer):
        trace = TraceContext()

        indexer.build(CORPUS, "kb", trace)

        data = trace.get_stage_data("bm25_indexer")
        assert data["collection"] == "kb"
        assert data["num_docs"] == 3

    def test_the_trace_records_the_query(self, indexer):
        indexer.build(CORPUS, "kb")
        trace = TraceContext()

        indexer.query(["fox"], trace=trace)

        data = trace.get_stage_data("bm25_query")
        assert data["query_terms"] == ["fox"]
        assert data["matched_chunks"] == 2
