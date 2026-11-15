"""The contract every vector store backend has to satisfy.

Keeping the checks in one place lets the in-memory fake and a real database
be measured against the same behaviour.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from libs.vector_store.base_vector_store import BaseVectorStore


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Similarity of two vectors, in the [-1, 1] range."""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(a * a for a in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def make_record(record_id: str, vector: list[float], **metadata: Any) -> dict[str, Any]:
    """Build a record in the shape `upsert` consumes."""
    return {"id": record_id, "vector": vector, "metadata": metadata}


class VectorStoreContract:
    """Checks an implementation satisfies the store contract.

    Subclass this and implement `make_store` to reuse the checks for a real
    backend.
    """

    def make_store(self) -> BaseVectorStore:
        raise NotImplementedError

    def test_upsert_accepts_valid_records(self):
        store = self.make_store()

        store.upsert([make_record("a", [0.1, 0.2], source="doc.pdf")])

        matches = store.query([0.1, 0.2], top_k=1)
        assert [match["id"] for match in matches] == ["a"]

    def test_upsert_replaces_a_record_with_the_same_id(self):
        store = self.make_store()

        store.upsert([make_record("a", [0.1, 0.2], source="doc.pdf")])
        store.upsert([make_record("a", [0.3, 0.4], source="doc.pdf")])

        matches = store.query([0.3, 0.4], top_k=10)
        assert [match["id"] for match in matches] == ["a"]

    def test_upsert_rejects_an_empty_record_list(self):
        store = self.make_store()

        with pytest.raises(ValueError):
            store.upsert([])

    def test_upsert_rejects_a_record_without_an_id(self):
        store = self.make_store()

        with pytest.raises(ValueError):
            store.upsert([{"vector": [0.1], "metadata": {}}])

    def test_upsert_rejects_a_record_without_a_vector(self):
        store = self.make_store()

        with pytest.raises(ValueError):
            store.upsert([{"id": "a", "metadata": {}}])

    def test_query_returns_at_most_top_k_matches(self):
        store = self.make_store()
        store.upsert(
            [
                make_record("a", [1.0, 0.0]),
                make_record("b", [0.9, 0.1]),
                make_record("c", [0.8, 0.2]),
            ]
        )

        matches = store.query([1.0, 0.0], top_k=2)

        assert len(matches) == 2

    def test_query_returns_the_closest_match_first(self):
        store = self.make_store()
        store.upsert(
            [
                make_record("far", [0.0, 1.0]),
                make_record("close", [1.0, 0.0]),
            ]
        )

        matches = store.query([1.0, 0.0], top_k=2)

        assert [match["id"] for match in matches] == ["close", "far"]
        assert matches[0]["score"] > matches[1]["score"]

    def test_query_returns_the_metadata_of_a_match(self):
        store = self.make_store()
        store.upsert([make_record("a", [1.0, 0.0], source="doc.pdf", text="hello")])

        matches = store.query([1.0, 0.0], top_k=1)

        assert matches[0]["metadata"]["source"] == "doc.pdf"
        assert matches[0]["metadata"]["text"] == "hello"

    def test_query_filters_candidates_by_metadata(self):
        store = self.make_store()
        store.upsert(
            [
                make_record("wanted", [1.0, 0.0], source="wanted.pdf"),
                make_record("other", [1.0, 0.0], source="other.pdf"),
            ]
        )

        matches = store.query([1.0, 0.0], top_k=10, filters={"source": "wanted.pdf"})

        assert [match["id"] for match in matches] == ["wanted"]

    def test_query_rejects_an_empty_vector(self):
        store = self.make_store()

        with pytest.raises(ValueError):
            store.query([], top_k=5)

    def test_query_rejects_a_non_positive_top_k(self):
        store = self.make_store()

        with pytest.raises(ValueError):
            store.query([1.0, 0.0], top_k=0)

    def test_get_by_ids_returns_the_records_of_those_ids(self):
        store = self.make_store()
        store.upsert(
            [
                make_record("a", [1.0, 0.0], text="first"),
                make_record("b", [0.0, 1.0], text="second"),
            ]
        )

        records = store.get_by_ids(["b"])

        assert [record["id"] for record in records] == ["b"]
        assert records[0]["text"] == "second"

    def test_get_by_ids_leaves_out_an_unknown_id(self):
        store = self.make_store()
        store.upsert([make_record("a", [1.0, 0.0], text="first")])

        records = store.get_by_ids(["a", "missing"])

        assert [record["id"] for record in records] == ["a"]

    def test_get_by_ids_of_nothing_gives_nothing(self):
        store = self.make_store()

        assert store.get_by_ids([]) == []
