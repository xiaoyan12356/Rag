"""Core data types shared by the whole pipeline.

Ingestion, retrieval and the MCP tools all pass the same records around, so
the contracts live here instead of in whichever module happens to produce
them. Every type turns into a plain dictionary and back, which is what lets a
chunk travel from the splitter to the vector store and out again.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

Metadata = dict[str, Any]
Vector = list[float]
SparseVector = dict[str, float]

IMAGE_PLACEHOLDER_TEMPLATE = "[IMAGE: {image_id}]"


def image_placeholder(image_id: str) -> str:
    """Return the marker that stands in for an image in the document text.

    The length of this marker is what `metadata.images[].text_length` records,
    so a producer and a consumer that both use this helper agree on where the
    image sits in the text.
    """
    return IMAGE_PLACEHOLDER_TEMPLATE.format(image_id=image_id)


@dataclass
class Document:
    """A document as a loader produced it, before it was split.

    Images stay in the text as `[IMAGE: {image_id}]` markers, and
    `metadata["images"]` describes each of them.
    """

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_source_path("Document", self.metadata)
        _validate_image_refs(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return the document as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Document:
        """Rebuild a document from a dictionary."""
        return cls(**data)


@dataclass
class Chunk:
    """A piece of a document, as a splitter produced it.

    The offsets say where the chunk sits in the document text, and
    `source_ref` names the document it came from.
    """

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)
    start_offset: int | None = None
    end_offset: int | None = None
    source_ref: str | None = None

    def __post_init__(self) -> None:
        _require_source_path("Chunk", self.metadata)
        _validate_offsets(self.start_offset, self.end_offset)
        _validate_image_refs(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return the chunk as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Chunk:
        """Rebuild a chunk from a dictionary."""
        return cls(**data)


@dataclass
class ChunkRecord:
    """A chunk with its vectors, as the vector store keeps it.

    The vectors are filled in along the way, so a record may carry only one
    of them, or none at all.
    """

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)
    dense_vector: Vector | None = None
    sparse_vector: SparseVector | None = None

    def __post_init__(self) -> None:
        _require_source_path("ChunkRecord", self.metadata)
        _validate_image_refs(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChunkRecord:
        """Rebuild a record from a dictionary."""
        return cls(**data)

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        dense_vector: Vector | None = None,
        sparse_vector: SparseVector | None = None,
    ) -> ChunkRecord:
        """Build a record from a chunk and its vectors.

        The metadata is copied, so a later change to the record does not reach
        back into the chunk it came from.
        """
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )


@dataclass
class RetrievalResult:
    """A chunk the search found, with the score that brought it back.

    `score` is the score of the route the result came from, or of the fusion
    that put it there. The scores of the single routes are kept beside it, so
    a result can be explained after the fact.
    """

    chunk_id: str
    score: float
    text: str = ""
    metadata: Metadata = field(default_factory=dict)
    dense_score: float | None = None
    sparse_score: float | None = None
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetrievalResult:
        """Rebuild a result from a dictionary."""
        return cls(**data)


def _require_source_path(owner: str, metadata: Mapping[str, Any]) -> None:
    if "source_path" not in metadata:
        raise ValueError(f"{owner} metadata must contain 'source_path'")


def _validate_offsets(start_offset: int | None, end_offset: int | None) -> None:
    for name, value in (("start_offset", start_offset), ("end_offset", end_offset)):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        start_offset is not None
        and end_offset is not None
        and end_offset < start_offset
    ):
        raise ValueError("end_offset cannot be smaller than start_offset")


def _validate_image_refs(metadata: Mapping[str, Any]) -> None:
    images = metadata.get("images")
    if images is None:
        return
    if not isinstance(images, list):
        raise ValueError("metadata['images'] must be a list of image references")
    for image in images:
        _validate_image_ref(image)


def _validate_image_ref(image: Any) -> None:
    if not isinstance(image, Mapping):
        raise ValueError("each image reference must be a mapping")
    for key in ("id", "path"):
        if not image.get(key):
            raise ValueError(f"an image reference needs a non-empty '{key}'")
    page = image.get("page")
    if page is not None and not isinstance(page, int):
        raise ValueError("image reference field 'page' must be an integer")
    for key in ("text_offset", "text_length"):
        value = image.get(key)
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError(
                f"image reference field '{key}' must be a non-negative integer"
            )
    position = image.get("position")
    if position is not None and not isinstance(position, Mapping):
        raise ValueError("image reference field 'position' must be a mapping")
