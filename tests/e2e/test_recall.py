"""End-to-end recall regression.

These tests run the real search over a small corpus and check that every
question of the golden test set still finds its chunk. The corpus is built
inside the test, so the run needs no network and no ingested documents: only
the embedding provider is replaced.

The threshold is written into the tests, so that a change that makes the
search worse fails here instead of slipping through.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.query_engine.hybrid_search import HybridSearch
from core.settings import (
    EmbeddingSettings,
    IngestionSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from core.types import Chunk
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.vector_upserter import VectorUpserter
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from observability.evaluation.eval_runner import EvalRunner, load_test_set

DIMENSION = 16
COLLECTION = "golden_recall"
TOP_K = 5

# The threshold the regression holds: every question has to find its chunk
# within the first TOP_K results.
MIN_HIT_RATE = 1.0
MIN_MRR = 0.8

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "golden_test_set.json"

# The corpus the golden test set was written against. The source paths and
# the texts decide the chunk ids, so they have to stay as they are.
CORPUS = [
    (
        "docs/azure_setup.md",
        "Azure OpenAI gives managed access to the OpenAI models. "
        "Configure the azure endpoint and the api key to use the azure provider.",
    ),
    (
        "docs/vector_store.md",
        "The Chroma vector store keeps the embeddings on the disk. "
        "Use the chroma provider to write and to query the vectors.",
    ),
    (
        "docs/sparse_index.md",
        "The BM25 index counts the terms of every chunk. "
        "The sparse route looks the words of the question up in this index.",
    ),
    (
        "docs/ingestion.md",
        "The ingestion pipeline reads a PDF, splits it into chunks and writes "
        "the chunks into the vector store and the BM25 index.",
    ),
    (
        "docs/mcp_server.md",
        "The MCP server offers three tools: query_knowledge_hub searches the "
        "hub, list_collections lists the collections and get_document_summary "
        "describes one document.",
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
    """Fill the collection and the index with the corpus of the golden set."""
    chunks = [
        Chunk(
            id=f"chunk_{number}",
            text=text,
            metadata={"source_path": source_path, "chunk_index": 0},
        )
        for number, (source_path, text) in enumerate(CORPUS)
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


@pytest.fixture
def golden_cases() -> list:
    return load_test_set(GOLDEN_SET_PATH)


class TestGoldenTestSet:
    def test_the_golden_test_set_is_there(self):
        assert GOLDEN_SET_PATH.exists()

    def test_every_case_names_a_question_and_the_chunks_it_should_find(self, golden_cases):
        assert golden_cases

        for case in golden_cases:
            assert case.query.strip()
            assert case.expected_chunk_ids

    def test_the_golden_test_set_has_the_shape_the_spec_asks_for(self):
        data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))

        assert "test_cases" in data
        for case in data["test_cases"]:
            assert "query" in case
            assert "expected_chunk_ids" in case
            assert "expected_sources" in case


class TestRecall:
    def test_every_question_of_the_golden_set_finds_its_chunk(self, search, golden_cases):
        for case in golden_cases:
            results = search.search(case.query, TOP_K)

            found = [result.chunk_id for result in results]
            assert any(chunk_id in case.expected_chunk_ids for chunk_id in found), (
                f"the question '{case.query}' found none of {case.expected_chunk_ids}"
            )

    def test_the_golden_chunk_is_among_the_first_results(self, search, golden_cases):
        for case in golden_cases:
            results = search.search(case.query, TOP_K)

            found = [result.chunk_id for result in results]
            assert any(chunk_id in case.expected_chunk_ids for chunk_id in found), (
                f"the question '{case.query}' found none of {case.expected_chunk_ids}"
            )

    def test_the_hit_rate_of_the_run_reaches_the_threshold(self, search, golden_cases):
        runner = EvalRunner(hybrid_search=search, top_k=TOP_K)

        report = runner.run(GOLDEN_SET_PATH)

        assert report.hit_rate >= MIN_HIT_RATE

    def test_the_mean_reciprocal_rank_of_the_run_reaches_the_threshold(
        self, search, golden_cases
    ):
        runner = EvalRunner(hybrid_search=search, top_k=TOP_K)

        report = runner.run(GOLDEN_SET_PATH)

        assert report.mrr >= MIN_MRR

    def test_the_report_names_the_questions_that_were_asked(self, search, golden_cases):
        runner = EvalRunner(hybrid_search=search, top_k=TOP_K)

        report = runner.run(GOLDEN_SET_PATH)

        assert [result.query for result in report.query_results] == [
            case.query for case in golden_cases
        ]

    def test_the_report_says_where_every_golden_chunk_was_found(self, search, golden_cases):
        runner = EvalRunner(hybrid_search=search, top_k=TOP_K)

        report = runner.run(GOLDEN_SET_PATH)

        for result in report.query_results:
            assert result.hit
            assert result.rank is not None
            assert result.rank <= TOP_K

    def test_the_metrics_of_the_report_hold_the_hit_rate_and_the_mrr(self, search, golden_cases):
        runner = EvalRunner(hybrid_search=search, top_k=TOP_K)

        report = runner.run(GOLDEN_SET_PATH)

        assert report.metrics["hit_rate"] == pytest.approx(report.hit_rate)
        assert report.metrics["mrr"] == pytest.approx(report.mrr)
