"""Integration tests for the hybrid search.

The hybrid search puts the dense route and the sparse route together. These
tests run it for real: the vectors go into a Chroma collection, the terms go
into a BM25 index on the disk, and only the embedding provider is replaced,
because that is the one part that would need a network.

The corpus is small on purpose, so that every result can be said out loud.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.query_engine.hybrid_search import HybridSearch
from core.settings import (
    EmbeddingSettings,
    IngestionSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.vector_upserter import VectorUpserter
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory

DIMENSION = 8
COLLECTION = "kb_test"

CORPUS = [
    (
        "Azure OpenAI Service gives managed access to the OpenAI models. "
        "Configure the endpoint and the api key to use the azure provider.",
        {"doc_type": "pdf", "title": "Azure Setup"},
    ),
    (
        "The Chroma vector store keeps the embeddings on the disk. "
        "Use the chroma provider to write and to query the vectors.",
        {"doc_type": "pdf", "title": "Vector Stores"},
    ),
    (
        "The BM25 index counts the terms of every chunk. "
        "The sparse route looks the words of the question up in this index.",
        {"doc_type": "md", "title": "Sparse Retrieval"},
    ),
    (
        "Another chunk about the index and about the terms it counts, "
        "so that the sparse route has more than one candidate to offer.",
        {"doc_type": "md", "title": "Sparse Retrieval Notes"},
    ),
]


class FakeEmbedding(BaseEmbedding):
    """Provider that answers with a vector built from the text itself."""

    def __init__(self) -> None:
        super().__init__(
            EmbeddingSettings(provider="fake", model="fake", dimensions=DIMENSION)
        )

    def embed(self, texts, trace=None) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * DIMENSION
            for position, character in enumerate(text):
                vector[position % DIMENSION] += (ord(character) % 17) / 17.0
            vectors.append(vector)
        return vectors


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        embedding=EmbeddingSettings(provider="fake", model="fake", dimensions=DIMENSION),
        vector_store=VectorStoreSettings(
            provider="chroma",
            persist_directory=str(tmp_path / "chroma"),
            collection_name=COLLECTION,
        ),
        retrieval=RetrievalSettings(dense_top_k=10, sparse_top_k=10, fusion_top_k=10),
        ingestion=IngestionSettings(bm25_index_dir=str(tmp_path / "bm25")),
    )


@pytest.fixture
def fake_provider():
    """Make the embedding factory answer with the fake provider."""
    EmbeddingFactory.register("fake", lambda settings: FakeEmbedding())
    yield
    EmbeddingFactory.unregister("fake")


@pytest.fixture
def corpus(settings, fake_provider) -> dict[str, str]:
    """Fill the collection and the index with the corpus."""
    chunks = [
        Chunk(
            id=f"chunk_{number}",
            text=text,
            metadata={
                "source_path": f"doc_{number}.pdf" if number < 3 else f"doc_{number}.md",
                "chunk_index": number,
                **extra,
            },
        )
        for number, (text, extra) in enumerate(CORPUS)
    ]

    upserter = VectorUpserter(settings, collection_name=COLLECTION)
    vectors = FakeEmbedding().embed([chunk.text for chunk in chunks])
    vector_ids = upserter.upsert(chunks, vectors)

    encoder = SparseEncoder()
    term_stats = encoder.encode(chunks)
    for stat, vector_id in zip(term_stats, vector_ids):
        stat["chunk_id"] = vector_id

    indexer = BM25Indexer(index_dir=settings.ingestion.bm25_index_dir)
    indexer.build(term_stats, collection=COLLECTION)

    return dict(zip(vector_ids, [chunk.text for chunk in chunks]))


@pytest.fixture
def search(settings, corpus) -> HybridSearch:
    return HybridSearch(settings, collection=COLLECTION)


class TestSearch:
    def test_the_search_finds_something(self, search):
        results = search.search("how do I configure azure")

        assert results

    def test_the_chunk_about_azure_comes_first(self, search):
        results = search.search("how do I configure azure openai")

        assert results[0].text.startswith("Azure OpenAI Service")

    def test_the_chunk_about_the_index_is_found(self, search):
        results = search.search("bm25 index terms")

        assert any("BM25 index" in result.text for result in results)

    def test_every_result_carries_the_text(self, search):
        results = search.search("chroma vector store")

        assert all(result.text for result in results)

    def test_every_result_carries_the_metadata(self, search):
        results = search.search("chroma vector store")

        assert all(result.metadata.get("source_path") for result in results)

    def test_the_results_are_numbered(self, search):
        results = search.search("index terms")

        assert [result.rank for result in results] == list(range(1, len(results) + 1))

    def test_the_number_of_results_can_be_capped(self, search):
        results = search.search("index terms", top_k=2)

        assert len(results) <= 2

    def test_a_question_that_shares_no_words_finds_nothing_in_the_index(self, search):
        # The dense route always hands back its nearest chunks, so the words
        # are what decide whether the index has something to offer.
        results = search.sparse_retriever.retrieve(["kubernetes", "helm"])

        assert results == []

    def test_the_score_of_both_routes_is_kept(self, search):
        results = search.search("bm25 index terms")

        assert any(result.sparse_score is not None for result in results)
        assert any(result.dense_score is not None for result in results)


class TestFilters:
    def test_a_filter_leaves_out_the_other_documents(self, search):
        results = search.search("index terms chunk", filters={"doc_type": "md"})

        assert results
        assert all(result.metadata["doc_type"] == "md" for result in results)

    def test_a_filter_that_nothing_matches_gives_no_result(self, search):
        results = search.search("azure", filters={"doc_type": "unknown"})

        assert results == []

    def test_a_filter_named_in_the_question_is_used(self, search):
        results = search.search("azure doc_type:md")

        assert all(result.metadata["doc_type"] == "md" for result in results)

    def test_the_title_can_be_filtered_on(self, search):
        results = search.search("azure", filters={"title": "Azure Setup"})

        assert len(results) == 1
        assert results[0].metadata["title"] == "Azure Setup"


class TestDegradation:
    def test_the_sparse_route_carries_the_search_when_the_dense_route_fails(self, settings, corpus):
        class BrokenDense:
            collection = COLLECTION

            def retrieve(self, query, top_k=None, filters=None, trace=None):
                raise RuntimeError("the embedding provider is down")

        search = HybridSearch(
            settings,
            dense_retriever=BrokenDense(),
            collection=COLLECTION,
        )

        results = search.search("bm25 index terms")

        assert results
        assert all(result.sparse_score is not None for result in results)

    def test_the_dense_route_carries_the_search_when_the_sparse_route_fails(self, settings, corpus):
        class BrokenSparse:
            collection = COLLECTION

            def retrieve(self, keywords, top_k=None, trace=None):
                raise RuntimeError("the index cannot be read")

        search = HybridSearch(
            settings,
            sparse_retriever=BrokenSparse(),
            collection=COLLECTION,
        )

        results = search.search("how do I configure azure")

        assert results
        assert all(result.dense_score is not None for result in results)

    def test_the_search_fails_when_both_routes_fail(self, settings, corpus):
        class Broken:
            def retrieve(self, *args: Any, **kwargs: Any):
                raise RuntimeError("the route is down")

        search = HybridSearch(
            settings,
            dense_retriever=Broken(),
            sparse_retriever=Broken(),
            collection=COLLECTION,
        )

        with pytest.raises(RuntimeError, match="Both retrieval routes failed"):
            search.search("azure")


class TestFailures:
    def test_a_question_that_is_empty_is_refused(self, search):
        with pytest.raises(ValueError, match="empty"):
            search.search("   ")

    def test_a_top_k_of_zero_is_refused(self, search):
        with pytest.raises(ValueError, match="top_k"):
            search.search("azure", top_k=0)


class TestTrace:
    def test_the_trace_records_every_stage(self, search):
        trace = TraceContext()

        search.search("bm25 index terms", trace=trace)

        assert trace.get_stage_data("query_processing") is not None
        assert trace.get_stage_data("dense_retrieval") is not None
        assert trace.get_stage_data("sparse_retrieval") is not None
        assert trace.get_stage_data("fusion") is not None
