"""Transform abstraction.

A transform takes the chunks of a document and gives them back in a better
shape: cleaned, enriched or described. The contract is the same for every
transform, so the pipeline can chain them without knowing which one it has.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.trace.trace_context import TraceContext
from core.types import Chunk


class BaseTransform(ABC):
    """Contract that every transform follows."""

    @abstractmethod
    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Return the transformed chunks, in the order they came in.

        A transform must not lose a chunk: when it cannot do better than the
        text it was given, it hands that text back unchanged.
        """
