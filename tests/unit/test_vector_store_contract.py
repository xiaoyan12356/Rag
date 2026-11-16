"""Contract tests for the vector store contract and its provider factory."""

from __future__ import annotations

from typing import Any

import pytest
from vector_store_contract import VectorStoreContract, cosine_similarity

from core.settings import Settings, VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError
from libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    """In-memory store that honours the contract, so it can be exercised offline."""

    def __init__(self, config: VectorStoreSettings) -> None:
        super().__init__(config)
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(self, records, trace=None) -> None:
        self.validate_records(records)
        for record in records:
            self.records[record["id"]] = dict(record)

    def query(self, vector, top_k=10, filters=None, trace=None):
        self.validate_query_vector(vector, top_k)
        candidates = [
            record
            for record in self.records.values()
            if filters is None
            or all(
                record.get("metadata", {}).get(key) == value
                for key, value in filters.items()
            )
        ]
        scored = [
            (cosine_similarity(vector, record["vector"]), record) for record in candidates
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"id": record["id"], "score": score, "metadata": record.get("metadata", {})}
            for score, record in scored[:top_k]
        ]

    def get_by_ids(self, ids, trace=None):
        wanted = list(ids)
        return [
            {
                "id": record["id"],
                "text": record.get("metadata", {}).get("text", ""),
                "metadata": record.get("metadata", {}),
            }
            for record in self.records.values()
            if record["id"] in wanted
        ]

    def delete_by_metadata(self, filters):
        self.validate_delete_filters(filters)
        keep = {
            key: record
            for key, record in self.records.items()
            if not all(record.get("metadata", {}).get(k) == v for k, v in filters.items())
        }
        removed = len(self.records) - len(keep)
        self.records = keep
        return removed

    def list_collections(self):
        return [{"name": self.collection_name, "chunk_count": len(self.records)}]


class TestFakeVectorStoreContract(VectorStoreContract):
    """Runs the contract suite against the in-memory fake."""

    def make_store(self) -> BaseVectorStore:
        return FakeVectorStore(VectorStoreSettings(provider="fake", persist_directory="./tmp"))


class TestBaseVectorStore:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseVectorStore(VectorStoreSettings())

    def test_subclass_must_implement_upsert_and_query(self):
        class IncompleteStore(BaseVectorStore):
            pass

        with pytest.raises(TypeError):
            IncompleteStore(VectorStoreSettings())

    def test_config_is_available_to_the_implementation(self):
        config = VectorStoreSettings(provider="fake", persist_directory="./data/db/chroma")
        store = FakeVectorStore(config)

        assert store.config is config
        assert store.config.collection_name == "knowledge_hub"

    def test_validate_records_rejects_a_non_mapping_record(self):
        store = FakeVectorStore(VectorStoreSettings())

        with pytest.raises(ValueError, match="not a mapping"):
            store.validate_records(["not a record"])

    def test_validate_records_rejects_a_vector_of_the_wrong_type(self):
        store = FakeVectorStore(VectorStoreSettings())

        with pytest.raises(ValueError, match="invalid vector type"):
            store.validate_records([{"id": "a", "vector": "not a vector"}])

    def test_validate_records_rejects_an_empty_vector(self):
        store = FakeVectorStore(VectorStoreSettings())

        with pytest.raises(ValueError, match="empty vector"):
            store.validate_records([{"id": "a", "vector": []}])

    def test_validate_query_vector_rejects_a_non_positive_top_k(self):
        store = FakeVectorStore(VectorStoreSettings())

        with pytest.raises(ValueError, match="top_k"):
            store.validate_query_vector([1.0, 0.0], -1)

    def test_validate_delete_filters_rejects_empty_filters(self):
        store = FakeVectorStore(VectorStoreSettings())

        with pytest.raises(ValueError, match="filters cannot be empty"):
            store.validate_delete_filters({})


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        vector_store=VectorStoreSettings(
            provider=provider,
            persist_directory="./data/db/chroma",
            collection_name="knowledge_hub",
        )
    )


@pytest.fixture
def register_provider():
    """Register providers and remove them again once the test finishes."""
    registered: list[str] = []

    def _register(name: str, builder) -> str:
        VectorStoreFactory.register(name, builder)
        registered.append(name)
        return name

    yield _register

    for name in registered:
        VectorStoreFactory.unregister(name)


class TestVectorStoreFactory:
    def test_create_returns_the_registered_provider(self, register_provider):
        register_provider("fake", lambda settings: FakeVectorStore(settings.vector_store))

        store = VectorStoreFactory.create(make_settings("fake"))

        assert isinstance(store, FakeVectorStore)

    def test_create_passes_the_vector_store_configuration_to_the_provider(
        self, register_provider
    ):
        register_provider("fake", lambda settings: FakeVectorStore(settings.vector_store))

        store = VectorStoreFactory.create(make_settings("fake"))

        assert store.config.persist_directory == "./data/db/chroma"
        assert store.config.collection_name == "knowledge_hub"

    def test_provider_name_is_matched_case_insensitively(self, register_provider):
        register_provider("fake", lambda settings: FakeVectorStore(settings.vector_store))

        store = VectorStoreFactory.create(make_settings("FAKE"))

        assert isinstance(store, FakeVectorStore)

    def test_unknown_provider_raises_a_readable_error(self):
        with pytest.raises(VectorStoreError) as exc_info:
            VectorStoreFactory.create(make_settings("does-not-exist"))

        message = str(exc_info.value)
        assert "does-not-exist" in message
        assert "Registered providers" in message

    def test_missing_provider_raises_a_readable_error(self):
        with pytest.raises(VectorStoreError) as exc_info:
            VectorStoreFactory.create(make_settings("   "))

        assert "vector_store.provider" in str(exc_info.value)

    def test_builder_returning_a_non_store_is_rejected(self, register_provider):
        register_provider("broken", lambda settings: "not a store")

        with pytest.raises(VectorStoreError) as exc_info:
            VectorStoreFactory.create(make_settings("broken"))

        assert "BaseVectorStore" in str(exc_info.value)

    def test_registering_a_provider_twice_replaces_the_builder(self, register_provider):
        register_provider("fake", lambda settings: FakeVectorStore(settings.vector_store))
        register_provider("fake", lambda settings: FakeVectorStore(settings.vector_store))

        store = VectorStoreFactory.create(make_settings("fake"))

        assert isinstance(store, FakeVectorStore)

    def test_registered_providers_lists_registered_names(self, register_provider):
        register_provider("fake", lambda settings: FakeVectorStore(settings.vector_store))

        assert "fake" in VectorStoreFactory.registered_providers()

    def test_unregister_removes_the_provider(self, register_provider):
        register_provider("fake", lambda settings: FakeVectorStore(settings.vector_store))

        VectorStoreFactory.unregister("fake")

        assert "fake" not in VectorStoreFactory.registered_providers()

    def test_registering_an_empty_provider_name_is_rejected(self):
        with pytest.raises(ValueError):
            VectorStoreFactory.register("", lambda settings: FakeVectorStore(settings.vector_store))

    def test_registering_a_provider_normalises_the_name(self, register_provider):
        register_provider("  Fake  ", lambda settings: FakeVectorStore(settings.vector_store))

        assert "fake" in VectorStoreFactory.registered_providers()

    def test_unregistering_a_provider_normalises_the_name(self, register_provider):
        register_provider("fake", lambda settings: FakeVectorStore(settings.vector_store))

        VectorStoreFactory.unregister("  FAKE  ")

        assert "fake" not in VectorStoreFactory.registered_providers()

    def test_unregistering_an_unknown_provider_is_ignored(self):
        VectorStoreFactory.unregister("never-registered")

        assert "never-registered" not in VectorStoreFactory.registered_providers()

    def test_a_provider_name_with_spaces_is_accepted(self, register_provider):
        register_provider("fake", lambda settings: FakeVectorStore(settings.vector_store))

        store = VectorStoreFactory.create(make_settings("  fake  "))

        assert isinstance(store, FakeVectorStore)

    def test_a_builder_that_fails_is_not_hidden(self, register_provider):
        def broken(settings):
            raise RuntimeError("the database is not there")

        register_provider("fake", broken)

        with pytest.raises(RuntimeError, match="the database is not there"):
            VectorStoreFactory.create(make_settings("fake"))
