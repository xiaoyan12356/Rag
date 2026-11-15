"""Roundtrip tests for the Chroma vector store.

The store is exercised against a real Chroma database in a temporary
directory, so the upsert and query paths are covered end to end and cleaned up
afterwards.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from vector_store_contract import VectorStoreContract, make_record

from core.settings import Settings, VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory

pytestmark = pytest.mark.integration


@pytest.fixture
def persist_directory(tmp_path: Path) -> Path:
    """A directory the store may persist into, removed by pytest afterwards."""
    return tmp_path / "chroma"


@pytest.fixture
def store(persist_directory: Path) -> ChromaStore:
    return ChromaStore(
        VectorStoreSettings(
            provider="chroma",
            persist_directory=str(persist_directory),
            collection_name="test_collection",
        )
    )


class TestChromaStoreContract(VectorStoreContract):
    """Runs the shared contract suite against a real Chroma database."""

    @pytest.fixture(autouse=True)
    def _contract_directory(self, tmp_path: Path):
        self.contract_directory = tmp_path / "contract"

    def make_store(self) -> BaseVectorStore:
        return ChromaStore(
            VectorStoreSettings(
                provider="chroma",
                persist_directory=str(self.contract_directory),
                collection_name="contract_collection",
            )
        )


class TestChromaStoreRoundtrip:
    def test_the_factory_creates_the_chroma_store(self, persist_directory: Path):
        settings = Settings(
            vector_store=VectorStoreSettings(
                provider="chroma",
                persist_directory=str(persist_directory),
                collection_name="factory_collection",
            )
        )

        store = VectorStoreFactory.create(settings)

        assert isinstance(store, ChromaStore)

    def test_upsert_then_query_returns_the_record(self, store: ChromaStore):
        store.upsert(
            [
                make_record(
                    "doc1_chunk0",
                    [0.1, 0.2, 0.3],
                    source="doc1.pdf",
                    text="the first chunk",
                )
            ]
        )

        matches = store.query([0.1, 0.2, 0.3], top_k=5)

        assert [match["id"] for match in matches] == ["doc1_chunk0"]
        assert matches[0]["metadata"]["text"] == "the first chunk"
        assert matches[0]["score"] > 0.9

    def test_the_closest_record_ranks_first(self, store: ChromaStore):
        store.upsert(
            [
                make_record("far", [0.0, 1.0], source="a.pdf"),
                make_record("close", [1.0, 0.0], source="a.pdf"),
            ]
        )

        matches = store.query([1.0, 0.0], top_k=2)

        assert [match["id"] for match in matches] == ["close", "far"]

    def test_top_k_limits_the_number_of_matches(self, store: ChromaStore):
        store.upsert(
            [make_record(f"chunk{index}", [1.0, 0.0], source="a.pdf") for index in range(5)]
        )

        assert len(store.query([1.0, 0.0], top_k=2)) == 2

    def test_a_metadata_filter_narrows_the_candidates(self, store: ChromaStore):
        store.upsert(
            [
                make_record("wanted", [1.0, 0.0], source="wanted.pdf"),
                make_record("other", [1.0, 0.0], source="other.pdf"),
            ]
        )

        matches = store.query([1.0, 0.0], top_k=10, filters={"source": "wanted.pdf"})

        assert [match["id"] for match in matches] == ["wanted"]

    def test_upserting_the_same_id_replaces_the_record(self, store: ChromaStore):
        store.upsert([make_record("a", [0.1, 0.2], text="first")])
        store.upsert([make_record("a", [0.3, 0.4], text="second")])

        matches = store.query([0.3, 0.4], top_k=5)

        assert len(matches) == 1
        assert matches[0]["metadata"]["text"] == "second"

    def test_the_records_survive_a_new_store_on_the_same_directory(
        self, persist_directory: Path
    ):
        config = VectorStoreSettings(
            provider="chroma",
            persist_directory=str(persist_directory),
            collection_name="persistent_collection",
        )
        ChromaStore(config).upsert([make_record("kept", [1.0, 0.0], text="stored")])

        matches = ChromaStore(config).query([1.0, 0.0], top_k=5)

        assert [match["id"] for match in matches] == ["kept"]

    def test_an_empty_store_answers_with_no_matches(self, store: ChromaStore):
        assert store.query([1.0, 0.0], top_k=5) == []

    def test_a_record_without_metadata_is_still_stored(self, store: ChromaStore):
        store.upsert([{"id": "plain", "vector": [1.0, 0.0]}])

        matches = store.query([1.0, 0.0], top_k=5)

        assert [match["id"] for match in matches] == ["plain"]
