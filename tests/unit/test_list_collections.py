"""Unit tests for the tool that lists the collections.

The tests pin that every folder under the documents root becomes a
collection, that the number of documents and the description of a collection
are reported, that the collections of the vector store are reported as well,
and that an empty root gives a friendly message instead of an empty list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.settings import Settings, VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory
from mcp_server.tools.list_collections import list_collections


@pytest.fixture
def documents_root(tmp_path: Path) -> Path:
    """Return a folder with two collections in it."""
    root = tmp_path / "documents"

    contracts = root / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "agreement.pdf").write_bytes(b"%PDF-1.4")
    (contracts / "notes.md").write_text("# Notes", encoding="utf-8")
    (contracts / "description.txt").write_text("The contracts of the company.", encoding="utf-8")

    manuals = root / "manuals"
    manuals.mkdir(parents=True)
    (manuals / "guide.docx").write_bytes(b"docx")

    return root


class TestTheCollections:
    def test_every_folder_becomes_a_collection(self, documents_root: Path):
        result = list_collections(documents_root)

        names = [item["name"] for item in result.structured_content["collections"]]
        assert names == ["contracts", "manuals"]

    def test_the_number_of_documents_is_reported(self, documents_root: Path):
        result = list_collections(documents_root)

        collections = {item["name"]: item for item in result.structured_content["collections"]}
        assert collections["contracts"]["document_count"] == 2

    def test_the_description_of_a_collection_is_reported(self, documents_root: Path):
        result = list_collections(documents_root)

        collections = {item["name"]: item for item in result.structured_content["collections"]}
        assert collections["contracts"]["description"] == "The contracts of the company."

    def test_a_collection_without_a_description_has_an_empty_one(self, documents_root: Path):
        result = list_collections(documents_root)

        collections = {item["name"]: item for item in result.structured_content["collections"]}
        assert collections["manuals"]["description"] == ""

    def test_a_file_is_not_a_collection(self, documents_root: Path):
        (documents_root / "readme.txt").write_text("hello", encoding="utf-8")

        result = list_collections(documents_root)

        names = [item["name"] for item in result.structured_content["collections"]]
        assert "readme.txt" not in names

    def test_a_hidden_folder_is_not_a_collection(self, documents_root: Path):
        (documents_root / ".hidden").mkdir()

        result = list_collections(documents_root)

        names = [item["name"] for item in result.structured_content["collections"]]
        assert ".hidden" not in names

    def test_the_text_names_every_collection(self, documents_root: Path):
        result = list_collections(documents_root)

        text = result.content[0]["text"]
        assert "contracts" in text
        assert "manuals" in text


class TestNothingThere:
    def test_an_empty_root_gives_a_friendly_message(self, tmp_path: Path):
        result = list_collections(tmp_path)

        assert "no collection" in result.content[0]["text"].lower()

    def test_an_empty_root_gives_no_collection(self, tmp_path: Path):
        result = list_collections(tmp_path)

        assert result.structured_content["collections"] == []

    def test_a_root_that_is_not_there_gives_a_friendly_message(self, tmp_path: Path):
        result = list_collections(tmp_path / "gone")

        assert "no collection" in result.content[0]["text"].lower()


class FakeStore(BaseVectorStore):
    """Store that knows the collections it was given."""

    def __init__(self, config: VectorStoreSettings, collections: list[dict[str, Any]]):
        super().__init__(config)
        self.collections = collections

    def upsert(self, records, trace=None) -> None:
        raise NotImplementedError

    def query(self, vector, top_k=10, filters=None, trace=None):
        raise NotImplementedError

    def get_by_ids(self, ids, trace=None):
        raise NotImplementedError

    def delete_by_metadata(self, filters) -> int:
        raise NotImplementedError

    def list_collections(self) -> list[dict[str, Any]]:
        return self.collections


@pytest.fixture
def store_provider():
    """Register a store that holds one collection, and take it away again."""
    VectorStoreFactory.register(
        "fake",
        lambda settings: FakeStore(
            settings.vector_store, [{"name": "kb", "chunk_count": 12}]
        ),
    )
    yield "fake"
    VectorStoreFactory.unregister("fake")


def make_settings(provider: str) -> Settings:
    """Return a configuration that points at the fake store."""
    return Settings(
        vector_store=VectorStoreSettings(
            provider=provider, persist_directory="./tmp", collection_name="kb"
        )
    )


class TestTheStore:
    """The collections of the vector store are reported as well."""

    def test_a_collection_of_the_store_is_reported(self, documents_root: Path, store_provider):
        result = list_collections(documents_root, settings=make_settings(store_provider))

        names = [item["name"] for item in result.structured_content["collections"]]
        assert "kb" in names

    def test_the_number_of_chunks_is_reported(self, documents_root: Path, store_provider):
        result = list_collections(documents_root, settings=make_settings(store_provider))

        collections = {item["name"]: item for item in result.structured_content["collections"]}
        assert collections["kb"]["chunk_count"] == 12

    def test_the_folders_are_still_reported(self, documents_root: Path, store_provider):
        result = list_collections(documents_root, settings=make_settings(store_provider))

        names = [item["name"] for item in result.structured_content["collections"]]
        assert "contracts" in names
        assert "manuals" in names

    def test_a_folder_that_is_also_a_collection_keeps_both_counts(
        self, documents_root: Path, store_provider
    ):
        (documents_root / "kb").mkdir()

        result = list_collections(documents_root, settings=make_settings(store_provider))

        collections = {item["name"]: item for item in result.structured_content["collections"]}
        assert collections["kb"]["chunk_count"] == 12
        assert collections["kb"]["document_count"] == 0

    def test_the_text_names_the_collection_of_the_store(
        self, documents_root: Path, store_provider
    ):
        result = list_collections(documents_root, settings=make_settings(store_provider))

        assert "kb" in result.content[0]["text"]
        assert "12 chunk(s)" in result.content[0]["text"]

    def test_a_store_that_cannot_be_opened_leaves_the_folders(
        self, documents_root: Path
    ):
        result = list_collections(documents_root, settings=make_settings("not_a_provider"))

        names = [item["name"] for item in result.structured_content["collections"]]
        assert names == ["contracts", "manuals"]

    def test_a_configuration_that_is_not_there_leaves_the_folders(
        self, documents_root: Path
    ):
        result = list_collections(documents_root, settings=None)

        names = [item["name"] for item in result.structured_content["collections"]]
        assert names == ["contracts", "manuals"]


def test_the_result_is_not_an_error(documents_root: Path) -> None:
    result = list_collections(documents_root)

    assert result.is_error is False
