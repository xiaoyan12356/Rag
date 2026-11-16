"""Document chunking.

The splitter cuts text into pieces and knows nothing about the document the
text came from. This module is the adapter between the two: it gives every
chunk a stable id, hands the metadata of the document to the chunk, and
collects the images the chunk actually refers to.

The image part matters. A document with twenty images becomes chunks that
mention one or two of them each, so a chunk that cannot say which images it
holds leaves the captioner with nothing to describe.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from core.settings import Settings
from core.types import Chunk, Document
from libs.splitter.splitter_factory import SplitterFactory

IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


class DocumentChunker:
    """Turn a document into the chunks the pipeline stores."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._splitter = SplitterFactory.create(settings)

    def split_document(self, document: Document) -> list[Chunk]:
        """Return the chunks of `document`, in reading order."""
        if not document.text or not document.text.strip():
            raise ValueError(f"Document {document.id} has no text to split")

        fragments = self._splitter.split_text(document.text)
        if not fragments:
            raise ValueError(f"Splitter returned no chunks for document {document.id}")

        return [
            Chunk(
                id=self._generate_chunk_id(document.id, index, text),
                text=text,
                metadata=self._inherit_metadata(document, index, text),
                source_ref=document.id,
            )
            for index, text in enumerate(fragments)
        ]

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """Return the id of a chunk: its document, its place in it and its text.

        The same document split twice gives the same ids, so a chunk that was
        ingested before is recognised the next time around.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{digest}"

    def _inherit_metadata(
        self,
        document: Document,
        chunk_index: int,
        chunk_text: str,
    ) -> dict[str, Any]:
        """Return the metadata of a chunk: the metadata of the document, plus its own.

        The images of the document are not handed down as they are. A chunk
        only carries the images it refers to, because the captioner should
        only describe what the chunk is about.
        """
        metadata = document.metadata.copy()
        document_images = metadata.pop("images", []) or []

        metadata["chunk_index"] = chunk_index

        image_refs = _image_refs(chunk_text)
        metadata["image_refs"] = image_refs

        if image_refs:
            lookup = {image.get("id"): image for image in document_images}
            images = [dict(lookup[ref]) for ref in image_refs if ref in lookup]
            if images:
                metadata["images"] = images

        return metadata


def _image_refs(text: str) -> list[str]:
    """Return the ids of the images the text refers to, in the order they appear."""
    refs: list[str] = []
    for match in IMAGE_PLACEHOLDER_PATTERN.findall(text):
        image_id = match.strip()
        if image_id and image_id not in refs:
            refs.append(image_id)
    return refs
