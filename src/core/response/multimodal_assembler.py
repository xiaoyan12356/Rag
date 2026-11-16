"""Images in a response.

A passage that talks about an image points at it by id. The image itself
lives on disk, so the assembler reads the file, encodes it and hands it over
as the image content MCP knows.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from core.types import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_MIME_TYPE = "image/png"
MAX_IMAGES = 5


class MultimodalAssembler:
    """Collect the images the retrieved passages point at."""

    def __init__(
        self,
        image_storage: Any | None = None,
        max_images: int = MAX_IMAGES,
    ) -> None:
        """Set up the assembler.

        Args:
            image_storage: The store that knows where an image id lives; it
                is used for the images the passages only point at.
            max_images: How many images one answer may carry.
        """
        self.image_storage = image_storage
        self.max_images = max_images

    def assemble(self, retrieval_results: list[RetrievalResult]) -> list[dict[str, Any]]:
        """Return the image content of the results, ready to be sent.

        Args:
            retrieval_results: The passages the search returned.

        Returns:
            The image blocks, one per image that could be read.
        """
        blocks: list[dict[str, Any]] = []
        seen: set[str] = set()

        for result in retrieval_results:
            for image_id, path in self._images_of(result):
                if image_id in seen:
                    continue
                block = self._image_block(image_id, path)
                if block is None:
                    continue
                seen.add(image_id)
                blocks.append(block)
                if len(blocks) >= self.max_images:
                    return blocks
        return blocks

    def _images_of(self, result: RetrievalResult) -> Iterator[tuple[str, str]]:
        """Yield the (id, path) of every image the passage points at.

        A store that keeps only plain values turns a list into a comma
        separated text, so both shapes are read.
        """
        images = result.metadata.get("images")
        if isinstance(images, list):
            for image in images:
                if isinstance(image, dict) and image.get("id"):
                    yield str(image["id"]), str(image.get("path") or "")

        for reference in _as_list(result.metadata.get("image_refs")):
            yield reference, ""

    def _image_block(self, image_id: str, path: str) -> dict[str, Any] | None:
        """Return the image content of one image, or None when it is not there.

        The store is asked first: it knows where the image ended up, while the
        path of the passage may be relative to the folder the server was
        started in.
        """
        file_path = self._stored_path(image_id) or path
        if not file_path:
            return None

        image_file = Path(file_path)
        try:
            data = image_file.read_bytes()
        except OSError as exc:
            logger.warning("The image %s could not be read: %s", image_id, exc)
            return None

        mime_type = mimetypes.guess_type(image_file.name)[0] or DEFAULT_MIME_TYPE
        return {
            "type": "image",
            "data": base64.b64encode(data).decode("ascii"),
            "mimeType": mime_type,
        }

    def _stored_path(self, image_id: str) -> str:
        """Return where an image id lives, when the store knows."""
        if self.image_storage is None:
            return ""
        try:
            return str(self.image_storage.get_image_path(image_id) or "")
        except Exception as exc:
            logger.warning("The image %s could not be looked up: %s", image_id, exc)
            return ""


def _as_list(value: Any) -> list[str]:
    """Return the values of a list, whether it is a list or a joined text."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []
