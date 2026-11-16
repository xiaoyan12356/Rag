"""Unit tests for the tool that describes one document.

The tests pin that the title, the summary and the tags of a document are
read off its chunks, that a document can be named by its id, by the hash of
its file or by its path, and that an unknown id gives an error instead of an
answer.
"""

from __future__ import annotations

from typing import Any

import pytest

from mcp_server.tools.get_document_summary import (
    DocumentLookup,
    get_document_summary,
)
from mcp_server.tools.list_collections import ToolArgumentError


class FakeCollection:
    """A collection that holds a fixed set of records."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        """Set up the collection.

        Args:
            records: The records it holds, each with an id, a text and a
                metadata.
        """
        self.records = records

    def get(self, include: list[str] | None = None) -> dict[str, Any]:
        """Return every record of the collection."""
        return {
            "ids": [record["id"] for record in self.records],
            "metadatas": [record["metadata"] for record in self.records],
            "documents": [record["text"] for record in self.records],
        }


class FakeStore:
    """A store that holds one collection."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        """Set up the store.

        Args:
            records: The records of the collection.
        """
        self.collection = FakeCollection(records)


def make_records() -> list[dict[str, Any]]:
    """Return the records of a document that was taken in."""
    return [
        {
            "id": "abc12345_0000_deadbeef",
            "text": "The first chunk.",
            "metadata": {
                "source_path": "docs/guide.pdf",
                "doc_hash": "abcdef1234567890",
                "title": "The Guide",
                "summary": "A guide about everything.",
                "tags": ["guide", "manual"],
                "page_count": 12,
                "chunk_index": 0,
            },
        },
        {
            "id": "abc12345_0001_cafe",
            "text": "The second chunk.",
            "metadata": {
                "source_path": "docs/guide.pdf",
                "doc_hash": "abcdef1234567890",
                "chunk_index": 1,
            },
        },
    ]


def make_lookup(records: list[dict[str, Any]] | None = None) -> DocumentLookup:
    """Return a lookup that reads from a fake store."""
    store = FakeStore(records if records is not None else make_records())
    lookup = DocumentLookup.__new__(DocumentLookup)
    lookup.settings = None  # type: ignore[assignment]
    lookup.collection = "knowledge_hub"
    lookup._store = store
    return lookup


class TestTheSummary:
    def test_the_title_is_reported(self):
        result = get_document_summary("abcdef1234567890", lookup=make_lookup())

        assert result.structured_content["title"] == "The Guide"

    def test_the_summary_is_reported(self):
        result = get_document_summary("abcdef1234567890", lookup=make_lookup())

        assert result.structured_content["summary"] == "A guide about everything."

    def test_the_tags_are_reported(self):
        result = get_document_summary("abcdef1234567890", lookup=make_lookup())

        assert result.structured_content["tags"] == ["guide", "manual"]

    def test_the_number_of_chunks_is_reported(self):
        result = get_document_summary("abcdef1234567890", lookup=make_lookup())

        assert result.structured_content["chunk_count"] == 2

    def test_the_source_path_is_reported(self):
        result = get_document_summary("abcdef1234567890", lookup=make_lookup())

        assert result.structured_content["source_path"] == "docs/guide.pdf"

    def test_the_text_shows_the_title(self):
        result = get_document_summary("abcdef1234567890", lookup=make_lookup())

        assert "The Guide" in result.content[0]["text"]

    def test_the_text_shows_the_tags(self):
        result = get_document_summary("abcdef1234567890", lookup=make_lookup())

        assert "guide" in result.content[0]["text"]

    def test_the_result_is_not_an_error(self):
        result = get_document_summary("abcdef1234567890", lookup=make_lookup())

        assert result.is_error is False


class TestHowADocumentIsNamed:
    def test_the_hash_of_the_file_finds_the_document(self):
        result = get_document_summary("abcdef1234567890", lookup=make_lookup())

        assert result.structured_content["found"] is True

    def test_the_beginning_of_the_hash_finds_the_document(self):
        result = get_document_summary("abcdef12", lookup=make_lookup())

        assert result.structured_content["found"] is True

    def test_the_path_finds_the_document(self):
        result = get_document_summary("docs/guide.pdf", lookup=make_lookup())

        assert result.structured_content["found"] is True

    def test_the_name_of_the_file_finds_the_document(self):
        result = get_document_summary("guide.pdf", lookup=make_lookup())

        assert result.structured_content["found"] is True

    def test_the_id_of_a_chunk_finds_the_document(self):
        result = get_document_summary("abc12345_0000_deadbeef", lookup=make_lookup())

        assert result.structured_content["found"] is True

    def test_the_stored_doc_id_finds_the_document(self):
        records = make_records()
        records[0]["metadata"]["doc_id"] = "doc_abc"
        records[1]["metadata"]["doc_id"] = "doc_abc"

        result = get_document_summary("doc_abc", lookup=make_lookup(records))

        assert result.structured_content["found"] is True


class TestAnUnknownDocument:
    def test_the_tool_reports_the_error(self):
        result = get_document_summary("nothing", lookup=make_lookup())

        assert result.is_error is True

    def test_the_tool_says_the_document_is_missing(self):
        result = get_document_summary("nothing", lookup=make_lookup())

        assert "nothing" in result.content[0]["text"]

    def test_the_tool_reports_that_it_was_not_found(self):
        result = get_document_summary("nothing", lookup=make_lookup())

        assert result.structured_content["found"] is False


class TestTheArguments:
    def test_an_empty_id_is_refused(self):
        with pytest.raises(ToolArgumentError, match="doc_id"):
            get_document_summary("  ", lookup=make_lookup())

    def test_an_id_that_is_not_a_text_is_refused(self):
        with pytest.raises(ToolArgumentError, match="doc_id"):
            get_document_summary(42, lookup=make_lookup())

    def test_an_empty_collection_is_refused(self):
        with pytest.raises(ToolArgumentError, match="collection"):
            get_document_summary("abc", collection="  ", lookup=make_lookup())


class TestTheLookup:
    def test_the_lookup_reads_the_collection(self):
        lookup = make_lookup()

        chunks = lookup.chunks_of("abcdef1234567890")

        assert len(chunks) == 2

    def test_the_lookup_leaves_out_the_other_documents(self):
        lookup = make_lookup()

        chunks = lookup.chunks_of("something-else")

        assert chunks == []


def test_the_tags_of_a_document_stored_as_a_text_are_split() -> None:
    records = [
        {
            "id": "a_0000_b",
            "text": "text",
            "metadata": {"tags": "guide, manual", "chunk_index": 0},
        }
    ]

    result = get_document_summary("a", lookup=make_lookup(records))

    assert result.structured_content["tags"] == ["guide", "manual"]


def test_a_document_without_tags_has_none() -> None:
    records = [{"id": "a_0000_b", "text": "text", "metadata": {"chunk_index": 0}}]

    result = get_document_summary("a", lookup=make_lookup(records))

    assert result.structured_content["tags"] == []
