"""Vision LLM abstraction.

A vision model answers about an image, so a request carries both the text and
the image. The image arrives as a path on disk or as bytes already in memory,
and `prepare_image` is where an implementation shrinks or converts it before
the request is encoded.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.settings import VisionLLMSettings
from libs.llm.base_llm import ChatResponse, LLMError

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class VisionLLMError(LLMError):
    """Raised when no vision provider can serve a request or a call fails."""


@dataclass
class ImageInput:
    """An image on its way to a model."""

    data: bytes
    mime_type: str = "image/png"


class BaseVisionLLM(ABC):
    """Contract that every vision model implementation follows."""

    def __init__(self, config: VisionLLMSettings) -> None:
        self.config = config

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: TraceContext | None = None,
    ) -> ChatResponse:
        """Answer `text` about the image given as a path or as bytes."""

    def validate_text(self, text: str) -> None:
        """Raise `ValueError` unless `text` says something."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

    def validate_image(self, image_path: str | bytes) -> None:
        """Raise `ValueError` unless `image_path` carries an image."""
        if isinstance(image_path, bytes):
            if not image_path:
                raise ValueError("image bytes cannot be empty")
            return
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(
                "image_path must be a file path, a base64 payload or the image bytes"
            )

    def prepare_image(self, image_path: str | bytes) -> ImageInput:
        """Return the image as bytes plus its mime type.

        A string names a file on disk; when no such file exists it is read as
        a base64 payload. Subclasses override this to shrink or convert the
        image before it is encoded.
        """
        self.validate_image(image_path)

        if isinstance(image_path, bytes):
            return ImageInput(data=image_path, mime_type=_detect_mime_type(image_path))

        path = Path(image_path)
        if path.is_file():
            data = path.read_bytes()
        else:
            try:
                data = base64.b64decode(image_path, validate=True)
            except (ValueError, TypeError) as exc:
                raise VisionLLMError(
                    f"The image path '{image_path}' is neither a readable file nor a "
                    f"base64 payload"
                ) from exc
        return ImageInput(data=data, mime_type=_detect_mime_type(data))


def _detect_mime_type(data: bytes) -> str:
    """Guess the mime type from the magic bytes of the image."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"
