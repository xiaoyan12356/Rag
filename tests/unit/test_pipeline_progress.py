"""Unit tests for the progress callback of the ingestion pipeline.

The callback tells whoever is watching how far the ingestion has come. These
tests pin that every step announces itself, that the steps are numbered, that
the batches of the embedding step are announced as well, and that nothing
changes when no callback is handed in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import (
    EmbeddingSettings,
    IngestionSettings,
    Settings,
    VectorStoreSettings,
)
from core.types import Chunk, Document
from ingestion.embedding.batch_processor import BatchResult
from ingestion.pipeline import STAGES, IngestionPipeline

CHUNK_COUNT = 5
BATCH_SIZE = 2


class FakeEmbedding:
    """Provider that answers with a vector built from the text itself."""

    def embed(self, texts, trace=None) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


class FakeLoader:
    """Loader that hands back a document without reading a file."""

    def load(self, path: Path) -> Document:
        return Document(
            id="doc_1",
            text="a document with some text in it",
            metadata={"source_path": str(path)},
        )


class FakeChunker:
    """Splitter that hands back a fixed number of chunks."""

    def split_document(self, document: Document) -> list[Chunk]:
        return [
            Chunk(
                id=f"chunk_{number}",
                text=f"chunk {number}",
                metadata={"source_path": document.metadata["source_path"]},
                source_ref=document.id,
            )
            for number in range(CHUNK_COUNT)
        ]


class FakeTransformer:
    """Transformer that hands the chunks back as they are."""

    def transform(self, chunks):
        return list(chunks)


class FakeBatchProcessor:
    """Processor that answers with a vector for every chunk."""

    def __init__(self, batch_size: int = BATCH_SIZE) -> None:
        self.batch_size = batch_size

    def get_batch_count(self, total_chunks: int) -> int:
        if total_chunks <= 0:
            return 0
        return (total_chunks + self.batch_size - 1) // self.batch_size

    def process(self, chunks, trace=None) -> BatchResult:
        return BatchResult(
            dense_vectors=[[1.0] for _ in chunks],
            sparse_stats=[{"term": "a"} for _ in chunks],
            batch_count=self.get_batch_count(len(chunks)),
            successful_chunks=len(chunks),
        )


class FakeUpserter:
    """Upserter that hands back an id for every chunk."""

    def upsert(self, chunks, vectors) -> list[str]:
        return [chunk.id for chunk in chunks]


class FakeIndexer:
    """Index that keeps what it was given."""

    def add_documents(self, documents, collection=None, doc_id=None) -> None:
        return None


class FakeImageStorage:
    """Image storage that keeps nothing."""

    def register_image(self, **kwargs) -> None:
        return None


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings that keep everything inside the test's own folder."""
    return Settings(
        embedding=EmbeddingSettings(provider="fake", model="fake", dimensions=1),
        vector_store=VectorStoreSettings(
            provider="chroma",
            persist_directory=str(tmp_path / "chroma"),
            collection_name="kb_test",
        ),
        ingestion=IngestionSettings(
            chunk_size=400,
            chunk_overlap=50,
            batch_size=BATCH_SIZE,
            bm25_index_dir=str(tmp_path / "bm25"),
            images_root=str(tmp_path / "images"),
            image_index_db=str(tmp_path / "image_index.db"),
            history_db=str(tmp_path / "history.db"),
        ),
    )


@pytest.fixture
def pipeline(settings) -> IngestionPipeline:
    """A pipeline whose parts are all fakes, so no network is needed."""
    built = IngestionPipeline.__new__(IngestionPipeline)
    built.settings = settings
    built.collection = "kb_test"
    built.force = False

    built.integrity_checker = _FakeIntegrityChecker()
    built.loader = FakeLoader()
    built.chunker = FakeChunker()
    built.chunk_refiner = FakeTransformer()
    built.metadata_enricher = FakeTransformer()
    built.image_captioner = FakeTransformer()
    built.embedding = FakeEmbedding()
    built.dense_encoder = None
    built.sparse_encoder = None
    built.batch_processor = FakeBatchProcessor()
    built.vector_upserter = FakeUpserter()
    built.bm25_indexer = FakeIndexer()
    built.image_storage = FakeImageStorage()
    return built


class _FakeIntegrityChecker:
    """Checker that never skips a document."""

    def compute_sha256(self, path: str) -> str:
        return "hash-1"

    def should_skip(self, file_hash: str) -> bool:
        return False

    def mark_success(self, file_hash: str, file_path: str, chunk_count: int = 0) -> None:
        return None

    def mark_failed(self, file_hash: str, reason: str) -> None:
        return None


@pytest.fixture
def document(tmp_path) -> Path:
    """The file that is handed to the pipeline."""
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4")
    return path


def _watched(pipeline, document) -> list[tuple[str, int]]:
    """Run the pipeline and return what the callback was told."""
    seen: list[tuple[str, int]] = []
    pipeline.run(document, on_progress=lambda stage, current, total: seen.append((stage, total)))
    return seen


class TestProgressCallback:
    def test_every_step_announces_itself(self, pipeline, document):
        seen: list[str] = []

        pipeline.run(document, on_progress=lambda stage, current, total: seen.append(stage))

        assert [stage for stage in seen if stage in STAGES] == list(STAGES)

    def test_the_steps_are_numbered(self, pipeline, document):
        steps: list[tuple[int, int]] = []

        pipeline.run(document, on_progress=lambda stage, current, total: steps.append((current, total)))

        assert steps[0] == (1, len(STAGES))
        assert (len(STAGES), len(STAGES)) in steps

    def test_the_number_of_steps_is_the_number_of_stages(self, pipeline, document):
        step_totals = [
            total for stage, total in _watched(pipeline, document) if stage in STAGES
        ]

        assert step_totals
        assert all(total == len(STAGES) for total in step_totals)

    def test_the_batches_of_the_embedding_step_are_announced(self, pipeline, document):
        batches: list[tuple[int, int]] = []

        def watch(stage: str, current: int, total: int) -> None:
            if stage == "embed_batch":
                batches.append((current, total))

        pipeline.run(document, on_progress=watch)

        expected = pipeline.batch_processor.get_batch_count(CHUNK_COUNT)
        assert batches == [(index, expected) for index in range(1, expected + 1)]

    def test_the_callback_is_asked_for_nothing_when_it_is_none(self, pipeline, document):
        result = pipeline.run(document, on_progress=None)

        assert result.chunk_count == CHUNK_COUNT


class TestWithoutCallback:
    def test_the_run_works_the_same_without_a_callback(self, pipeline, document):
        result = pipeline.run(document)

        assert result.chunk_count == CHUNK_COUNT
        assert result.skipped is False

    def test_the_stages_are_the_same_without_a_callback(self, pipeline, document):
        result = pipeline.run(document)

        assert set(result.stages) == set(STAGES)
