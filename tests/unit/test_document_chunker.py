"""Unit tests for the document chunker.

The chunker is the adapter between the splitter and the pipeline, so the
tests pin what it adds on top of a plain text split: stable ids, the
metadata of the document, and the images each chunk really refers to.
"""

from __future__ import annotations

import hashlib

import pytest

from core.settings import Settings
from core.types import Chunk, Document
from ingestion.chunking import DocumentChunker
from libs.splitter.base_splitter import BaseSplitter, SplitterError
from libs.splitter.splitter_factory import SplitterFactory

CHUNK_SEPARATOR = "\n---\n"


class FakeSplitter(BaseSplitter):
    """Splitter that cuts the text where the test asks it to."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.seen: list[str] = []

    def split_text(self, text: str, trace=None) -> list[str]:
        self.seen.append(text)
        return [part.strip() for part in text.split(CHUNK_SEPARATOR) if part.strip()]


def make_settings(splitter: str = "fake", chunk_size: int = 1000, chunk_overlap: int = 0):
    """Return settings that ask for the given splitter."""
    settings = Settings()
    settings.ingestion.splitter = splitter
    settings.ingestion.chunk_size = chunk_size
    settings.ingestion.chunk_overlap = chunk_overlap
    return settings


@pytest.fixture
def fake_splitter():
    """Register a splitter whose cuts the test controls."""
    SplitterFactory.register("fake", lambda settings: FakeSplitter(settings.ingestion))
    yield
    SplitterFactory.unregister("fake")


@pytest.fixture
def chunker(fake_splitter: None) -> DocumentChunker:
    return DocumentChunker(make_settings())


@pytest.fixture
def document() -> Document:
    return Document(
        id="doc_sample_001",
        text=(
            "First paragraph content.\n"
            "---\n"
            "Second paragraph mentions [IMAGE: img_002].\n"
            "---\n"
            "Third paragraph content."
        ),
        metadata={
            "source_path": "data/documents/sample.pdf",
            "doc_type": "pdf",
            "title": "Sample Document",
            "page_count": 3,
            "images": [
                {"id": "img_001", "path": "data/images/a/img_001.png", "page": 1},
                {"id": "img_002", "path": "data/images/a/img_002.png", "page": 2},
            ],
        },
    )


class TestChunkIds:
    def test_every_chunk_gets_its_own_id(self, chunker: DocumentChunker, document: Document):
        chunks = chunker.split_document(document)

        ids = [chunk.id for chunk in chunks]
        assert len(set(ids)) == len(ids)

    def test_the_same_document_gives_the_same_ids(
        self, chunker: DocumentChunker, document: Document
    ):
        first = chunker.split_document(document)
        second = chunker.split_document(document)

        assert [chunk.id for chunk in first] == [chunk.id for chunk in second]

    def test_the_id_names_the_document_the_place_and_the_text(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        for index, chunk in enumerate(chunks):
            digest = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:8]
            assert chunk.id == f"doc_sample_001_{index:04d}_{digest}"

    def test_two_chunks_with_the_same_text_have_different_ids(self, chunker: DocumentChunker):
        document = Document(
            id="doc_repeat",
            text="Same text.\n---\nSame text.",
            metadata={"source_path": "data/documents/repeat.pdf"},
        )

        chunks = chunker.split_document(document)

        assert chunks[0].id != chunks[1].id

    def test_the_place_in_the_document_is_part_of_the_id(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        assert chunks[0].id.startswith("doc_sample_001_0000_")
        assert chunks[1].id.startswith("doc_sample_001_0001_")


class TestMetadata:
    def test_the_document_metadata_is_handed_to_the_chunk(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        for chunk in chunks:
            assert chunk.metadata["source_path"] == "data/documents/sample.pdf"
            assert chunk.metadata["doc_type"] == "pdf"
            assert chunk.metadata["title"] == "Sample Document"
            assert chunk.metadata["page_count"] == 3

    def test_the_place_in_the_document_is_recorded(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]

    def test_the_metadata_of_the_document_is_left_alone(
        self, chunker: DocumentChunker, document: Document
    ):
        chunker.split_document(document)

        assert "chunk_index" not in document.metadata
        assert "image_refs" not in document.metadata

    def test_the_chunk_does_not_change_the_document(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        chunks[0].metadata["source_path"] = "changed.pdf"

        assert document.metadata["source_path"] == "data/documents/sample.pdf"

    def test_a_document_without_images_is_handed_down_whole(
        self, chunker: DocumentChunker
    ):
        document = Document(
            id="doc_plain",
            text="Only text here.",
            metadata={
                "source_path": "data/documents/plain.pdf",
                "doc_type": "pdf",
                "custom_field": "value",
            },
        )

        chunks = chunker.split_document(document)

        assert chunks[0].metadata["custom_field"] == "value"
        assert chunks[0].metadata["doc_type"] == "pdf"
        assert "images" not in chunks[0].metadata


class TestSourceRef:
    def test_every_chunk_points_back_at_the_document(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        for chunk in chunks:
            assert chunk.source_ref == "doc_sample_001"


class TestImageDistribution:
    def test_a_chunk_carries_the_images_it_refers_to(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        assert chunks[1].metadata["image_refs"] == ["img_002"]
        assert len(chunks[1].metadata["images"]) == 1
        assert chunks[1].metadata["images"][0]["id"] == "img_002"

    def test_a_chunk_without_a_placeholder_carries_no_images(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        assert "images" not in chunks[0].metadata
        assert "images" not in chunks[2].metadata

    def test_a_chunk_without_a_placeholder_still_says_it_has_none(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        assert chunks[0].metadata["image_refs"] == []

    def test_the_images_of_the_document_are_not_handed_down_whole(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        for chunk in chunks:
            referenced = {image["id"] for image in chunk.metadata.get("images", [])}
            assert referenced <= {"img_002"}

    def test_a_chunk_does_not_share_the_image_records_of_the_document(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        chunks[1].metadata["images"][0]["path"] = "changed.png"

        assert document.metadata["images"][1]["path"] == "data/images/a/img_002.png"

    def test_a_placeholder_of_an_unknown_image_is_still_reported(
        self, chunker: DocumentChunker
    ):
        document = Document(
            id="doc_unknown",
            text="Text with [IMAGE: img_missing] in it.",
            metadata={"source_path": "data/documents/unknown.pdf"},
        )

        chunks = chunker.split_document(document)

        assert chunks[0].metadata["image_refs"] == ["img_missing"]
        assert "images" not in chunks[0].metadata


class TestConfigurationDrivesTheChunks:
    def test_a_smaller_chunk_size_gives_more_chunks(self, fake_splitter: None):
        document = Document(
            id="doc_long",
            text=" ".join(f"word{index}" for index in range(200)),
            metadata={"source_path": "data/documents/long.pdf"},
        )

        large = DocumentChunker(make_settings("recursive", chunk_size=1000)).split_document(
            document
        )
        small = DocumentChunker(make_settings("recursive", chunk_size=50)).split_document(
            document
        )

        assert len(small) > len(large)

    def test_the_chunk_size_shows_in_the_length_of_the_chunks(self, fake_splitter: None):
        document = Document(
            id="doc_long",
            text=" ".join(f"word{index}" for index in range(200)),
            metadata={"source_path": "data/documents/long.pdf"},
        )

        chunks = DocumentChunker(make_settings("recursive", chunk_size=60)).split_document(
            document
        )

        assert all(len(chunk.text) <= 60 for chunk in chunks)

    def test_the_splitter_of_the_configuration_is_used(self, fake_splitter: None):
        document = Document(
            id="doc_any",
            text="A text with a marker.",
            metadata={"source_path": "data/documents/any.pdf"},
        )

        chunks = DocumentChunker(make_settings("recursive")).split_document(document)

        assert len(chunks) == 1

    def test_an_unknown_splitter_is_reported(self):
        with pytest.raises(SplitterError):
            DocumentChunker(make_settings("no-such-splitter"))


class TestRefusals:
    def test_a_document_without_text_is_reported(self, chunker: DocumentChunker):
        document = Document(
            id="doc_empty",
            text="   ",
            metadata={"source_path": "data/documents/empty.pdf"},
        )

        with pytest.raises(ValueError, match="no text"):
            chunker.split_document(document)

    def test_a_splitter_that_returns_nothing_is_reported(self, monkeypatch):
        class SilentSplitter(BaseSplitter):
            def __init__(self, config) -> None:
                super().__init__(config)

            def split_text(self, text: str, trace=None) -> list[str]:
                return []

        SplitterFactory.register("silent", lambda settings: SilentSplitter(settings.ingestion))
        try:
            document = Document(
                id="doc_silent",
                text="Some text.",
                metadata={"source_path": "data/documents/silent.pdf"},
            )

            with pytest.raises(ValueError, match="no chunks"):
                DocumentChunker(make_settings("silent")).split_document(document)
        finally:
            SplitterFactory.unregister("silent")


class TestTypeContract:
    def test_the_result_is_a_list_of_chunks(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_every_chunk_carries_the_source_path(self, chunker: DocumentChunker, document: Document):
        chunks = chunker.split_document(document)

        for chunk in chunks:
            assert "source_path" in chunk.metadata

    def test_a_chunk_can_be_serialised_and_restored(
        self, chunker: DocumentChunker, document: Document
    ):
        chunks = chunker.split_document(document)

        data = chunks[1].to_dict()
        restored = Chunk.from_dict(data)

        assert restored.id == chunks[1].id
        assert restored.text == chunks[1].text
        assert restored.metadata == chunks[1].metadata
        assert restored.source_ref == chunks[1].source_ref

    def test_the_text_of_the_document_is_handed_to_the_splitter(
        self, document: Document
    ):
        created: list[FakeSplitter] = []

        def build(settings):
            splitter = FakeSplitter(settings.ingestion)
            created.append(splitter)
            return splitter

        SplitterFactory.register("capturing", build)
        try:
            DocumentChunker(make_settings("capturing")).split_document(document)
        finally:
            SplitterFactory.unregister("capturing")

        assert created[0].seen == [document.text]
