"""Vector store contract.

A store owns the durable side of retrieval: it accepts chunk records and
answers similarity queries. Backends differ in how they persist and rank, so
callers depend on `BaseVectorStore` and never on a concrete database.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from core.settings import VectorStoreSettings

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class VectorStoreError(Exception):
    """Raised when no provider can serve a request or a provider call fails."""


class BaseVectorStore(ABC):
    """Contract that every vector database implementation follows."""

    def __init__(self, config: VectorStoreSettings) -> None:
        self.config = config
        self.collection_name = config.collection_name or "knowledge_hub"

    @abstractmethod
    def upsert(
        self,
        records: Sequence[Mapping[str, Any]],
        trace: TraceContext | None = None,
    ) -> None:
        """Insert or replace `records`.

        Every record carries `id`, `vector` and `metadata`; the chunk text
        travels in `metadata["text"]`. Re-upserting an id must replace that
        record instead of duplicating it.
        """

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return the `top_k` records closest to `vector`, closest first.

        Every match carries `id`, `score` and `metadata`; `score` is higher for
        a closer record. `filters` narrows the candidates by metadata
        equality.
        """

    @abstractmethod
    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return the records with the given ids.

        Every record carries `id`, `text` and `metadata`. An id that is not
        in the store is left out, so the caller can tell how many of the
        requested records were found.
        """

    @abstractmethod
    def delete_by_metadata(
        self,
        filters: Mapping[str, Any],
    ) -> int:
        """Delete every record whose metadata matches `filters`.

        A record is deleted when every field of `filters` equals the field of
        the same name in the metadata of the record. The number of records
        that were deleted is returned; filters that match nothing delete
        nothing and report zero.
        """

    @abstractmethod
    def list_collections(self) -> list[dict[str, Any]]:
        """Return the collections the store holds and how much is in them.

        Every entry carries a `name` and a `chunk_count`. The collection the
        store was opened with is always among them, so a caller can ask what
        can be searched without knowing the name of the store.
        """

    def validate_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Raise `ValueError` unless every record carries the required fields."""
        if not records:
            raise ValueError("records cannot be empty")

        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"record at index {index} is not a mapping "
                    f"(type: {type(record).__name__})"
                )
            if "id" not in record:
                raise ValueError(f"record at index {index} is missing required field 'id'")
            if "vector" not in record:
                raise ValueError(
                    f"record at index {index} is missing required field 'vector'"
                )

            vector = record["vector"]
            if not isinstance(vector, (list, tuple)):
                raise ValueError(
                    f"record at index {index} has invalid vector type: "
                    f"{type(vector).__name__}. Expected a list or tuple of floats."
                )
            if not vector:
                raise ValueError(f"record at index {index} has an empty vector")

    def validate_query_vector(self, vector: list[float], top_k: int) -> None:
        """Raise `ValueError` unless the query vector and `top_k` are usable."""
        if not isinstance(vector, (list, tuple)):
            raise ValueError(
                f"query vector must be a list or tuple, got {type(vector).__name__}"
            )
        if not vector:
            raise ValueError("query vector cannot be empty")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

    def validate_delete_filters(self, filters: Mapping[str, Any]) -> None:
        """Raise `ValueError` unless `filters` names the records to delete.

        Filters that match everything would empty the whole store, so an
        empty filter is refused.
        """
        if not filters:
            raise ValueError("filters cannot be empty")
