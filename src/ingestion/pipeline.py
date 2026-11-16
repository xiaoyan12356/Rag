"""Ingestion pipeline.

The pipeline walks a document from the file on disk to the places where it
can be found again. The text is loaded, cut into chunks, cleaned, described,
turned into vectors and written to the stores.

The steps run one after the other: integrity, load, split, transform, encode,
store. A step that fails stops the run with an error that names the step, so
the reason a document did not make it is never a guess.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, Document
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessor, BatchResult
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "default"
STAGES = ("integrity", "load", "split", "transform", "encode", "store")


class PipelineError(Exception):
    """Raised when a step of the pipeline fails.

    The name of the step is kept, so the error says where the run stopped
    rather than only what went wrong.
    """

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"The {stage} step failed: {message}")
        self.stage = stage
        self.message = message


@dataclass
class PipelineResult:
    """What one run of the pipeline did."""

    file_path: str
    doc_id: str | None = None
    chunk_count: int = 0
    image_count: int = 0
    vector_ids: list[str] = field(default_factory=list)
    skipped: bool = False
    stages: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary."""
        return {
            "file_path": self.file_path,
            "doc_id": self.doc_id,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "vector_ids_count": len(self.vector_ids),
            "skipped": self.skipped,
            "stages": self.stages,
        }


class IngestionPipeline:
    """Run a document through the whole ingestion.

    Every part of the pipeline is built from the settings, so switching a
    provider or moving a data folder is a change in the configuration file.
    The embedding provider can be handed in, which is what lets the pipeline
    run without a network.
    """

    def __init__(
        self,
        settings: Settings,
        collection: str = DEFAULT_COLLECTION,
        force: bool = False,
        embedding: BaseEmbedding | None = None,
    ) -> None:
        """Set up the pipeline and everything it needs."""
        self.settings = settings
        self.collection = collection
        self.force = force

        self.integrity_checker = SQLiteIntegrityChecker(
            db_path=settings.ingestion.history_db
        )
        self.loader = PdfLoader(
            extract_images=True,
            image_storage_dir=settings.ingestion.images_root,
        )
        self.chunker = DocumentChunker(settings)
        self.chunk_refiner = ChunkRefiner(settings)
        self.metadata_enricher = MetadataEnricher(settings)
        self.image_captioner = ImageCaptioner(settings)

        self.embedding = embedding or EmbeddingFactory.create(settings)
        batch_size = settings.ingestion.batch_size
        self.dense_encoder = DenseEncoder(self.embedding, batch_size=batch_size)
        self.sparse_encoder = SparseEncoder()
        self.batch_processor = BatchProcessor(
            self.dense_encoder,
            self.sparse_encoder,
            batch_size=batch_size,
        )

        self.vector_upserter = VectorUpserter(settings, collection_name=collection)
        self.bm25_indexer = BM25Indexer(index_dir=settings.ingestion.bm25_index_dir)
        self.image_storage = ImageStorage(
            db_path=settings.ingestion.image_index_db,
            images_root=settings.ingestion.images_root,
        )

    def run(
        self,
        file_path: str | Path,
        trace: TraceContext | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> PipelineResult:
        """Take the document at `file_path` through the whole pipeline.

        Args:
            file_path: The document to ingest.
            trace: A trace to record the run in.
            on_progress: Called with the name of a step, the number of the
                step and the number of steps, once the step is done.

        Raises:
            PipelineError: When a step of the pipeline fails.
        """
        path = Path(file_path)
        stages: dict[str, Any] = {}
        file_hash = ""
        chunks: list[Chunk] = []
        vector_ids: list[str] = []

        def notify(stage: str, step: int) -> None:
            logger.info("Step %d/%d: %s", step, len(STAGES), stage)
            if on_progress is not None:
                on_progress(stage, step, len(STAGES))

        try:
            with self._step("integrity", stages, notify, 1):
                file_hash = self.integrity_checker.compute_sha256(str(path))
                already_done = (
                    not self.force and self.integrity_checker.should_skip(file_hash)
                )
            stages["integrity"].update({"file_hash": file_hash, "skipped": already_done})
            self._record(trace, "integrity", stages)

            if already_done:
                logger.info("The document was ingested before, skipping it")
                return PipelineResult(
                    file_path=str(path),
                    doc_id=file_hash,
                    skipped=True,
                    stages=stages,
                )

            with self._step("load", stages, notify, 2):
                document = self.loader.load(path)
            stages["load"].update(
                {
                    "doc_id": document.id,
                    "text_length": len(document.text),
                    "image_count": len(document.metadata.get("images", []) or []),
                }
            )
            self._record(trace, "load", stages)

            with self._step("split", stages, notify, 3):
                chunks = self.chunker.split_document(document)
            stages["split"].update(
                {
                    "chunk_count": len(chunks),
                    "average_chunk_size": (
                        sum(len(chunk.text) for chunk in chunks) // len(chunks)
                        if chunks
                        else 0
                    ),
                }
            )
            self._record(trace, "split", stages)

            with self._step("transform", stages, notify, 4):
                chunks = self._transform(chunks)
            stages["transform"].update(
                {
                    "refined_by_rule": sum(
                        1 for chunk in chunks if chunk.metadata.get("refined_by") == "rule"
                    ),
                    "enriched_by_rule": sum(
                        1 for chunk in chunks if chunk.metadata.get("enriched_by") == "rule"
                    ),
                    "captioned_chunks": sum(
                        1 for chunk in chunks if chunk.metadata.get("image_captions")
                    ),
                }
            )
            self._record(trace, "transform", stages)

            with self._step("encode", stages, notify, 5):
                encoded = self._encode(chunks)
            stages["encode"].update(
                {
                    "dense_vector_count": len(encoded.dense_vectors),
                    "sparse_doc_count": len(encoded.sparse_stats),
                    "failed_chunks": encoded.failed_chunks,
                }
            )
            self._record(trace, "encode", stages)

            with self._step("store", stages, notify, 6):
                vector_ids = self._store(chunks, encoded, document, file_hash)
            stages["store"].update(
                {
                    "vector_count": len(vector_ids),
                    "bm25_docs": len(encoded.sparse_stats),
                    "images_indexed": len(document.metadata.get("images", []) or []),
                }
            )
            self._record(trace, "store", stages)
        except PipelineError:
            if file_hash:
                self._mark_failed(file_hash, str(path))
            raise

        self.integrity_checker.mark_success(file_hash, str(path), chunk_count=len(chunks))
        logger.info(
            "The document was ingested: %d chunks, %d images",
            len(chunks),
            stages["load"]["image_count"],
        )
        return PipelineResult(
            file_path=str(path),
            doc_id=file_hash,
            chunk_count=len(chunks),
            image_count=stages["load"]["image_count"],
            vector_ids=vector_ids,
            stages=stages,
        )

    @contextmanager
    def _step(
        self,
        stage: str,
        stages: dict[str, Any],
        notify: Callable[[str, int], None],
        step: int,
    ) -> Iterator[None]:
        """Run one step of the pipeline, and name it when it fails."""
        started = time.monotonic()
        try:
            yield
        except PipelineError:
            raise
        except Exception as exc:
            logger.error("The %s step failed: %s", stage, exc)
            raise PipelineError(stage, str(exc)) from exc
        finally:
            stages.setdefault(stage, {})["duration_ms"] = round(
                (time.monotonic() - started) * 1000.0, 2
            )
        notify(stage, step)

    @staticmethod
    def _record(
        trace: TraceContext | None,
        stage: str,
        stages: dict[str, Any],
    ) -> None:
        """Write what a step did into the trace."""
        if trace is None:
            return
        details = dict(stages.get(stage, {}))
        elapsed_ms = details.pop("duration_ms", None)
        trace.record_stage(stage, details, elapsed_ms=elapsed_ms)

    def _transform(self, chunks: list[Chunk]) -> list[Chunk]:
        """Clean the chunks, describe them and caption their images."""
        chunks = self.chunk_refiner.transform(chunks)
        chunks = self.metadata_enricher.transform(chunks)
        chunks = self.image_captioner.transform(chunks)
        logger.info("Cleaned and described %d chunks", len(chunks))
        return chunks

    def _encode(self, chunks: list[Chunk]) -> BatchResult:
        """Turn the chunks into vectors and term statistics.

        A chunk that could not be encoded is a failure of the run: the
        stores would otherwise hold a text without a vector beside it.
        """
        result = self.batch_processor.process(chunks)
        if result.failed_chunks:
            raise RuntimeError(
                f"{result.failed_chunks} of {len(chunks)} chunks could not be encoded"
            )
        logger.info("Encoded %d chunks into %d vectors", len(chunks), len(result.dense_vectors))
        return result

    def _store(
        self,
        chunks: list[Chunk],
        encoded: BatchResult,
        document: Document,
        file_hash: str,
    ) -> list[str]:
        """Write the vectors, the index and the images to their stores."""
        vector_ids = self.vector_upserter.upsert(chunks, encoded.dense_vectors)
        logger.info("Wrote %d vectors", len(vector_ids))

        # The index has to speak the same ids as the vector store, so that a
        # chunk found by the index can be looked up where its text is kept.
        for statistics, vector_id in zip(encoded.sparse_stats, vector_ids):
            statistics["chunk_id"] = vector_id
        self.bm25_indexer.add_documents(
            encoded.sparse_stats,
            collection=self.collection,
            doc_id=document.id,
        )
        logger.info("Indexed %d chunks for the sparse search", len(encoded.sparse_stats))

        images = document.metadata.get("images", []) or []
        indexed = 0
        for image in images:
            image_path = Path(image["path"])
            if not image_path.exists():
                logger.warning("The image %s is not on disk, skipping it", image["id"])
                continue
            self.image_storage.register_image(
                image_id=image["id"],
                file_path=image_path,
                collection=self.collection,
                doc_hash=file_hash,
                page_num=image.get("page"),
            )
            indexed += 1
        logger.info("Indexed %d images", indexed)
        return vector_ids

    def _mark_failed(self, file_hash: str, file_path: str) -> None:
        """Note that the document did not make it, so it is tried again later."""
        try:
            self.integrity_checker.mark_failed(file_hash, "the pipeline failed")
        except Exception as exc:
            logger.warning("Could not note the failure of %s: %s", file_path, exc)


def run_pipeline(
    file_path: str | Path,
    settings: Settings,
    collection: str = DEFAULT_COLLECTION,
    force: bool = False,
    trace: TraceContext | None = None,
) -> PipelineResult:
    """Take one document through the whole pipeline."""
    pipeline = IngestionPipeline(settings, collection=collection, force=force)
    return pipeline.run(file_path, trace=trace)
