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
