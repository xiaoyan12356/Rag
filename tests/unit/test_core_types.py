"""Unit tests for the core data types.

The types are the contract every pipeline stage passes records through, so
the tests pin the fields, the serialization and the metadata conventions.
"""

from __future__ import annotations

import json

import pytest

from core.types import Chunk, ChunkRecord, Document, image_placeholder


def make_image_ref(image_id: str = "abc123_1_0", **overrides) -> dict:
    values = {
        "id": image_id,
        "path": f"data/images/collection/{image_id}.png",
        "page": 1,
        "text_offset": 20,
        "text_length": len(image_placeholder(image_id)),
        "position": {"x": 100, "y": 200, "width": 400, "height": 300},
    }
    values.update(overrides)
    return values


class TestDocument:
    def test_the_fields_are_kept(self):
        document = Document(
            id="doc_123",
            text="# Title\n\nContent here",
            metadata={"source_path": "data/test.pdf"},
        )

        assert document.id == "doc_123"
        assert document.text == "# Title\n\nContent here"
        assert document.metadata["source_path"] == "data/test.pdf"

    def test_a_missing_source_path_is_rejected(self):
        with pytest.raises(ValueError, match="must contain 'source_path'"):
            Document(id="doc_123", text="Content", metadata={})

    def test_the_extra_metadata_fields_are_kept(self):
        document = Document(
            id="doc_123",
            text="Content",
            metadata={
                "source_path": "data/test.pdf",
                "doc_type": "pdf",
                "title": "Test Document",
                "page_count": 10,
            },
        )

        assert document.metadata["doc_type"] == "pdf"
        assert document.metadata["title"] == "Test Document"
        assert document.metadata["page_count"] == 10

    def test_it_serialises_to_a_dictionary(self):
        document = Document(
            id="doc_123",
            text="Content",
            metadata={"source_path": "data/test.pdf", "title": "Test"},
        )

        data = document.to_dict()

        assert data["id"] == "doc_123"
        assert data["text"] == "Content"
        assert data["metadata"]["source_path"] == "data/test.pdf"

    def test_it_deserialises_from_a_dictionary(self):
        original = Document(
            id="doc_123",
            text="Content",
            metadata={"source_path": "data/test.pdf", "title": "Test"},
        )

        restored = Document.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata

    def test_the_serialised_fields_keep_their_order(self):
        document = Document(id="doc_123", text="Content", metadata={"source_path": "a"})

        assert list(document.to_dict()) == ["id", "text", "metadata"]

    def test_it_serialises_to_json(self):
        document = Document(
            id="doc_123",
            text="Content",
            metadata={"source_path": "data/test.pdf", "page_count": 3},
        )

        restored = json.loads(json.dumps(document.to_dict()))

        assert restored["id"] == "doc_123"
        assert restored["metadata"]["page_count"] == 3

    def test_the_metadata_can_be_extended_without_breaking_compatibility(self):
        document = Document(
            id="doc_123",
            text="Content",
            metadata={
                "source_path": "data/test.pdf",
                "custom_field_1": "value1",
                "custom_field_2": 123,
                "custom_field_3": ["list", "values"],
            },
        )

        restored = Document.from_dict(document.to_dict())

        assert restored.metadata["custom_field_1"] == "value1"
        assert restored.metadata["custom_field_2"] == 123
        assert restored.metadata["custom_field_3"] == ["list", "values"]


class TestChunk:
    def test_the_fields_are_kept(self):
        chunk = Chunk(
            id="chunk_123_001",
            text="## Section 1\n\nFirst paragraph",
            metadata={"source_path": "data/test.pdf", "chunk_index": 0},
        )

        assert chunk.id == "chunk_123_001"
        assert chunk.text == "## Section 1\n\nFirst paragraph"
        assert chunk.metadata["chunk_index"] == 0

    def test_the_offsets_and_the_source_reference_default_to_nothing(self):
        chunk = Chunk(id="chunk_1", text="Content", metadata={"source_path": "a"})

        assert chunk.start_offset is None
        assert chunk.end_offset is None
        assert chunk.source_ref is None

    def test_the_offsets_are_kept(self):
        chunk = Chunk(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf"},
            start_offset=0,
            end_offset=100,
        )

        assert chunk.start_offset == 0
        assert chunk.end_offset == 100

    def test_the_source_reference_is_kept(self):
        chunk = Chunk(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf"},
            source_ref="doc_123",
        )

        assert chunk.source_ref == "doc_123"

    def test_a_missing_source_path_is_rejected(self):
        with pytest.raises(ValueError, match="must contain 'source_path'"):
            Chunk(id="chunk_123", text="Content", metadata={"chunk_index": 0})

    def test_a_negative_offset_is_rejected(self):
        with pytest.raises(ValueError, match="start_offset"):
            Chunk(
                id="chunk_123",
                text="Content",
                metadata={"source_path": "a"},
                start_offset=-1,
            )

    def test_an_end_before_the_start_is_rejected(self):
        with pytest.raises(ValueError, match="end_offset"):
            Chunk(
                id="chunk_123",
                text="Content",
                metadata={"source_path": "a"},
                start_offset=100,
                end_offset=50,
            )

    def test_it_serialises_and_deserialises(self):
        original = Chunk(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=100,
            source_ref="doc_123",
        )

        data = original.to_dict()
        restored = Chunk.from_dict(data)

        assert list(data) == [
            "id",
            "text",
            "metadata",
            "start_offset",
            "end_offset",
            "source_ref",
        ]
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.start_offset == original.start_offset
        assert restored.end_offset == original.end_offset
        assert restored.source_ref == original.source_ref


class TestChunkRecord:
    def test_the_fields_are_kept(self):
        record = ChunkRecord(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf", "chunk_index": 0},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"word1": 0.5, "word2": 0.3},
        )

        assert record.id == "chunk_123_001"
        assert record.dense_vector == [0.1, 0.2, 0.3]
        assert record.sparse_vector == {"word1": 0.5, "word2": 0.3}

    def test_the_vectors_are_optional(self):
        record = ChunkRecord(
            id="chunk_123_001", text="Content", metadata={"source_path": "a"}
        )

        assert record.dense_vector is None
        assert record.sparse_vector is None

    def test_a_missing_source_path_is_rejected(self):
        with pytest.raises(ValueError, match="must contain 'source_path'"):
            ChunkRecord(id="chunk_123", text="Content", metadata={})

    def test_it_serialises_and_deserialises(self):
        original = ChunkRecord(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf", "title": "Section 1"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"word": 0.5},
        )

        data = original.to_dict()
        restored = ChunkRecord.from_dict(data)

        assert list(data) == [
            "id",
            "text",
            "metadata",
            "dense_vector",
            "sparse_vector",
        ]
        assert restored.id == original.id
        assert restored.dense_vector == original.dense_vector
        assert restored.sparse_vector == original.sparse_vector

    def test_it_is_built_from_a_chunk(self):
        chunk = Chunk(
            id="chunk_123_001",
            text="Content",
            metadata={"source_path": "data/test.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=100,
        )

        record = ChunkRecord.from_chunk(chunk, [0.1, 0.2, 0.3], {"word": 0.5})

        assert record.id == chunk.id
        assert record.text == chunk.text
        assert record.metadata == chunk.metadata
        assert record.dense_vector == [0.1, 0.2, 0.3]
        assert record.sparse_vector == {"word": 0.5}

    def test_the_metadata_is_copied_and_not_shared_with_the_chunk(self):
        chunk = Chunk(
            id="chunk_123",
            text="Content",
            metadata={"source_path": "data/test.pdf", "key": "original"},
        )

        record = ChunkRecord.from_chunk(chunk)
        record.metadata["key"] = "modified"

        assert chunk.metadata["key"] == "original"
        assert record.metadata["key"] == "modified"


class TestImageSupport:
    def test_the_placeholder_names_the_image(self):
        assert image_placeholder("abc123_1_0") == "[IMAGE: abc123_1_0]"

    def test_the_placeholder_length_matches_the_recorded_text_length(self):
        image_id = "abc123_1_0"
        expected = len("[IMAGE: ") + len(image_id) + len("]")

        assert len(image_placeholder(image_id)) == expected

    def test_a_document_carries_its_images(self):
        document = Document(
            id="doc_with_img",
            text="Here is some text.\n\n[IMAGE: abc123_1_0]\n\nMore text after image.",
            metadata={
                "source_path": "data/test.pdf",
                "images": [make_image_ref()],
            },
        )

        assert "[IMAGE: abc123_1_0]" in document.text
        assert document.metadata["images"][0]["id"] == "abc123_1_0"
        assert document.metadata["images"][0]["text_offset"] == 20
        assert document.metadata["images"][0]["text_length"] == len(
            image_placeholder("abc123_1_0")
        )

    def test_a_document_can_carry_several_images(self):
        document = Document(
            id="doc_multi_img",
            text="Text [IMAGE: img1] middle [IMAGE: img2] end",
            metadata={
                "source_path": "data/test.pdf",
                "images": [make_image_ref("img1"), make_image_ref("img2")],
            },
        )

        assert len(document.metadata["images"]) == 2
        assert document.text.count("[IMAGE:") == 2

    def test_a_document_without_images_is_allowed(self):
        document = Document(
            id="doc_no_img",
            text="Plain text document",
            metadata={"source_path": "data/test.txt", "images": []},
        )

        assert document.metadata["images"] == []

    def test_a_chunk_carries_the_images_it_refers_to(self):
        chunk = Chunk(
            id="chunk_with_img",
            text="Section content [IMAGE: abc123_1_0] continues here",
            metadata={
                "source_path": "data/test.pdf",
                "chunk_index": 0,
                "images": [make_image_ref()],
            },
        )

        assert "[IMAGE: abc123_1_0]" in chunk.text
        assert len(chunk.metadata["images"]) == 1

    def test_a_record_carries_the_generated_captions(self):
        record = ChunkRecord(
            id="record_with_caption",
            text="Architecture diagram [IMAGE: diagram_001] shows the system",
            metadata={
                "source_path": "data/test.pdf",
                "images": [make_image_ref("diagram_001", page=5)],
                "image_captions": {
                    "diagram_001": "System architecture showing a three-tier design"
                },
            },
            dense_vector=[0.1, 0.2, 0.3],
        )

        assert record.metadata["image_captions"]["diagram_001"]
        assert "architecture" in record.metadata["image_captions"]["diagram_001"].lower()

    def test_an_image_reference_without_an_id_is_rejected(self):
        with pytest.raises(ValueError, match="'id'"):
            Document(
                id="doc_1",
                text="Content",
                metadata={
                    "source_path": "a",
                    "images": [{"path": "data/images/collection/img1.png"}],
                },
            )

    def test_an_image_reference_without_a_path_is_rejected(self):
        with pytest.raises(ValueError, match="'path'"):
            Document(
                id="doc_1",
                text="Content",
                metadata={"source_path": "a", "images": [{"id": "img1"}]},
            )

    def test_an_image_reference_that_is_not_a_mapping_is_rejected(self):
        with pytest.raises(ValueError, match="mapping"):
            Document(
                id="doc_1",
                text="Content",
                metadata={"source_path": "a", "images": ["img1.png"]},
            )

    def test_the_images_field_must_be_a_list(self):
        with pytest.raises(ValueError, match="list"):
            Document(
                id="doc_1",
                text="Content",
                metadata={"source_path": "a", "images": {"id": "img1"}},
            )

    def test_a_negative_text_offset_is_rejected(self):
        with pytest.raises(ValueError, match="text_offset"):
            Document(
                id="doc_1",
                text="Content",
                metadata={
                    "source_path": "a",
                    "images": [make_image_ref(text_offset=-1)],
                },
            )

    def test_a_position_that_is_not_a_mapping_is_rejected(self):
        with pytest.raises(ValueError, match="position"):
            Document(
                id="doc_1",
                text="Content",
                metadata={
                    "source_path": "a",
                    "images": [make_image_ref(position=[1, 2])],
                },
            )

    def test_the_optional_fields_may_be_left_out(self):
        document = Document(
            id="doc_1",
            text="Content",
            metadata={
                "source_path": "a",
                "images": [{"id": "img1", "path": "data/images/collection/img1.png"}],
            },
        )

        assert document.metadata["images"][0]["id"] == "img1"


class TestMetadataPropagation:
    def test_the_metadata_travels_from_the_document_to_the_record(self):
        document = Document(
            id="doc_123",
            text="Full document text",
            metadata={
                "source_path": "data/report.pdf",
                "doc_type": "pdf",
                "title": "Annual Report",
            },
        )

        chunk_metadata = document.metadata.copy()
        chunk_metadata.update({"chunk_index": 0, "page": 1})
        chunk = Chunk(
            id="chunk_123_001",
            text="First section",
            metadata=chunk_metadata,
            source_ref="doc_123",
        )

        record_metadata = chunk.metadata.copy()
        record_metadata.update({"summary": "Introduction section"})
        record = ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            metadata=record_metadata,
            dense_vector=[0.1, 0.2, 0.3],
        )

        assert record.metadata["source_path"] == document.metadata["source_path"]
        assert record.metadata["title"] == document.metadata["title"]
        assert record.metadata["chunk_index"] == 0
        assert record.metadata["summary"] == "Introduction section"

    def test_every_type_needs_a_source_path(self):
        with pytest.raises(ValueError):
            Document(id="d1", text="t", metadata={})
        with pytest.raises(ValueError):
            Chunk(id="c1", text="t", metadata={})
        with pytest.raises(ValueError):
            ChunkRecord(id="r1", text="t", metadata={})
