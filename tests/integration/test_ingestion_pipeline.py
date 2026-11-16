"""Integration tests for the ingestion pipeline.

The pipeline walks a document from the file on disk to the stores. These
tests run the whole way with a provider that answers without a network, so
the loader, the splitter, the transformers and the stores are all really
used.

The test document is built when the test runs: several sections, several
pages and a picture on every page, so that the image path is walked too.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw

from core.settings import (
    EmbeddingSettings,
    IngestionSettings,
    Settings,
    VectorStoreSettings,
)
from core.trace.trace_context import TraceContext
from ingestion.pipeline import STAGES, IngestionPipeline, PipelineError
from libs.embedding.base_embedding import BaseEmbedding

DIMENSION = 8


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


def make_png(width: int = 320, height: int = 200) -> bytes:
    """Return a picture, so the document holds something to extract."""
    buffer = io.BytesIO()
    image = Image.new("RGB", (width, height), (238, 240, 248))
    drawing = ImageDraw.Draw(image)
    drawing.rectangle([16, 16, width - 16, height - 16], outline=(40, 80, 160), width=4)
    drawing.line([16, height - 16, width - 16, 16], fill=(200, 60, 60), width=3)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_document(path: Path, pages: int = 3) -> Path:
    """Write a document with sections, paragraphs and a picture on every page."""
    picture = make_png()
    document = pymupdf.open()

    for number in range(1, pages + 1):
        page = document.new_page()
        page.insert_text((72, 80), f"Chapter {number}: The Pipeline", fontsize=18)
        page.insert_text(
            (72, 110),
            f"This chapter explains step {number} of the ingestion and what it does.",
            fontsize=11,
        )
        page.insert_text(
            (72, 130),
            "The text is loaded, cut into chunks, cleaned, turned into vectors "
            "and written to the stores.",
            fontsize=11,
        )
        page.insert_text((72, 160), f"Figure {number}: the flow of the step", fontsize=11)
        page.insert_image(pymupdf.Rect(72, 175, 392, 375), stream=picture)
        page.insert_text(
            (72, 400),
            "Every stage records how long it took, so a slow step is easy to find.",
            fontsize=11,
        )

    document.save(str(path))
    document.close()
    return path


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings that keep everything inside the test's own folder."""
    return Settings(
        embedding=EmbeddingSettings(provider="fake", model="fake", dimensions=DIMENSION),
        vector_store=VectorStoreSettings(
            provider="chroma",
            persist_directory=str(tmp_path / "chroma"),
            collection_name="kb_test",
        ),
        ingestion=IngestionSettings(
            chunk_size=400,
            chunk_overlap=50,
            batch_size=2,
            bm25_index_dir=str(tmp_path / "bm25"),
            images_root=str(tmp_path / "images"),
            image_index_db=str(tmp_path / "image_index.db"),
            history_db=str(tmp_path / "history.db"),
        ),
    )


@pytest.fixture
def document(tmp_path) -> Path:
    return build_document(tmp_path / "technical_doc.pdf")


@pytest.fixture
def pipeline(settings) -> IngestionPipeline:
    return IngestionPipeline(settings, collection="kb_test", embedding=FakeEmbedding())


class TestRun:
    def test_the_document_comes_out_in_chunks(self, pipeline, document):
        result = pipeline.run(document)

        assert result.chunk_count > 0
        assert result.skipped is False

    def test_every_chunk_gets_an_id(self, pipeline, document):
        result = pipeline.run(document)

        assert len(result.vector_ids) == result.chunk_count
        assert len(set(result.vector_ids)) == result.chunk_count

    def test_the_pictures_are_found(self, pipeline, document):
        result = pipeline.run(document)

        assert result.image_count == 3

    def test_the_vectors_reach_the_store(self, pipeline, document):
        result = pipeline.run(document)

        found = pipeline.vector_upserter.vector_store.get_by_ids(result.vector_ids)
        assert len(found) == result.chunk_count

    def test_the_index_is_written_to_disk(self, pipeline, document, settings):
        pipeline.run(document)

        index_path = Path(settings.ingestion.bm25_index_dir) / "kb_test_bm25.json"
        assert index_path.exists()

    def test_the_index_can_be_looked_up(self, pipeline, document):
        result = pipeline.run(document)

        results = pipeline.bm25_indexer.query(["pipeline"], top_k=5)

        assert results
        assert results[0]["chunk_id"] in result.vector_ids

    def test_the_images_are_written_to_disk(self, pipeline, document, settings):
        pipeline.run(document)

        images = pipeline.image_storage.list_images(collection="kb_test")
        assert len(images) == 3
        for image in images:
            assert Path(image["file_path"]).exists()

    def test_the_images_are_noted_in_the_database(self, pipeline, document):
        pipeline.run(document)

        images = pipeline.image_storage.list_images(collection="kb_test")
        assert all(image["collection"] == "kb_test" for image in images)

    def test_the_run_says_what_it_did(self, pipeline, document):
        result = pipeline.run(document)

        assert set(result.stages) == {
            "integrity",
            "load",
            "split",
            "transform",
            "embed",
            "upsert",
        }

    def test_every_stage_is_timed(self, pipeline, document):
        result = pipeline.run(document)

        assert all("duration_ms" in result.stages[stage] for stage in result.stages)


class TestSkipping:
    def test_a_document_that_was_ingested_before_is_skipped(self, pipeline, document):
        pipeline.run(document)

        second = pipeline.run(document)

        assert second.skipped is True
        assert second.chunk_count == 0

    def test_a_forced_run_ingests_the_document_again(self, pipeline, document, settings):
        pipeline.run(document)

        forced = IngestionPipeline(settings, collection="kb_test", force=True, embedding=FakeEmbedding())
        second = forced.run(document)

        assert second.skipped is False
        assert second.chunk_count > 0

    def test_the_chunks_are_not_written_twice(self, pipeline, document):
        first = pipeline.run(document)
        pipeline.run(document)

        found = pipeline.vector_upserter.vector_store.get_by_ids(first.vector_ids)
        assert len(found) == len(first.vector_ids)


class TestFailures:
    def test_a_document_that_is_not_there_names_the_step(self, pipeline, tmp_path):
        with pytest.raises(PipelineError) as error:
            pipeline.run(tmp_path / "missing.pdf")

        assert error.value.stage == "integrity"

    def test_a_file_that_is_not_a_pdf_names_the_step(self, pipeline, tmp_path):
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"this is not a pdf")

        with pytest.raises(PipelineError) as error:
            pipeline.run(broken)

        assert error.value.stage == "load"

    def test_the_error_says_which_step_failed(self, pipeline, tmp_path):
        with pytest.raises(PipelineError, match="integrity"):
            pipeline.run(tmp_path / "missing.pdf")

    def test_a_failed_document_is_tried_again(self, pipeline, tmp_path):
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"this is not a pdf")

        with pytest.raises(PipelineError):
            pipeline.run(broken)

        checker = pipeline.integrity_checker
        assert checker.should_skip(checker.compute_sha256(str(broken))) is False


class TestProgress:
    def test_every_step_is_announced(self, pipeline, document):
        seen: list[str] = []

        pipeline.run(document, on_progress=lambda stage, step, total: seen.append(stage))

        assert [stage for stage in seen if stage in STAGES] == list(STAGES)

    def test_the_steps_are_numbered(self, pipeline, document):
        steps: list[tuple[int, int]] = []

        pipeline.run(document, on_progress=lambda stage, step, total: steps.append((step, total)))

        assert steps[0] == (1, 6)
        assert (6, 6) in steps


class TestTrace:
    def test_the_trace_records_every_step(self, pipeline, document):
        trace = TraceContext()

        pipeline.run(document, trace=trace)

        for stage in ("integrity", "load", "split", "transform", "embed", "upsert"):
            assert trace.get_stage_data(stage) is not None

    def test_the_trace_records_what_the_split_did(self, pipeline, document):
        trace = TraceContext()

        pipeline.run(document, trace=trace)

        data = trace.get_stage_data("split")
        assert data["chunk_count"] > 0

    def test_the_trace_is_an_ingestion(self, pipeline, document):
        trace = TraceContext()

        pipeline.run(document, trace=trace)

        assert trace.to_dict()["trace_type"] == "ingestion"

    def test_every_step_names_the_method_that_did_the_work(self, pipeline, document):
        trace = TraceContext()

        pipeline.run(document, trace=trace)

        for stage in ("load", "split", "transform", "embed", "upsert"):
            data = trace.get_stage_data(stage)
            assert data["method"], stage
            assert data["provider"], stage

    def test_every_step_says_how_long_it_took(self, pipeline, document):
        trace = TraceContext()

        pipeline.run(document, trace=trace)

        for stage in ("load", "split", "transform", "embed", "upsert"):
            assert trace.get_stage(stage)["elapsed_ms"] >= 0.0, stage

    def test_the_trace_can_be_written_out(self, pipeline, document):
        trace = TraceContext()

        pipeline.run(document, trace=trace)
        trace.finish()

        written = json.dumps(trace.to_dict())

        assert json.loads(written)["trace_type"] == "ingestion"
