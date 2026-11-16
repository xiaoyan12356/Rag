"""Azure OpenAI image understanding.

Azure serves the OpenAI chat completions format under a deployment-scoped URL
and authenticates with an `api-key` header. The image travels as an inline
data URL, so an image that is larger than the configured limit is shrunk
before the request is built.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

import httpx

from core.settings import VisionLLMSettings
from libs.http_client import JSONHttpClient
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM, ImageInput, VisionLLMError

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

DEFAULT_MAX_IMAGE_SIZE = 2048
JPEG_QUALITY = 85


class AzureVisionLLMError(VisionLLMError):
    """Raised when an Azure vision request fails."""


class AzureVisionLLM(BaseVisionLLM):
    """Ask a question about an image through an Azure OpenAI deployment."""

    DEFAULT_API_VERSION = "2024-02-01"
    DEFAULT_TIMEOUT = 120.0
    provider_name = "azure"

    def __init__(
        self,
        config: VisionLLMSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(config)
        if not (config.azure_endpoint or "").strip():
            raise AzureVisionLLMError(
                "[azure] vision_llm.azure_endpoint is not configured - "
                "set it in config/settings.yaml"
            )
        if not (config.deployment_name or config.model or "").strip():
            raise AzureVisionLLMError(
                "[azure] vision_llm.deployment_name is not configured - "
                "set it in config/settings.yaml"
            )
        self.timeout = float(timeout) if timeout is not None else self.DEFAULT_TIMEOUT
        self._http = JSONHttpClient(
            self.provider_name, timeout=self.timeout, transport=transport
        )

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._http.close()

    def __enter__(self) -> AzureVisionLLM:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: TraceContext | None = None,
    ) -> ChatResponse:
        """Answer `text` about the image given as a path, a base64 payload or bytes."""
        try:
            self.validate_text(text)
        except ValueError as exc:
            raise AzureVisionLLMError(f"[{self.provider_name}] invalid text: {exc}") from exc

        image = self.prepare_image(image_path)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": _data_url(image)}},
                    ],
                }
            ]
        }

        response = self._http.post_json(
            self._chat_completions_url(), payload, self._headers(), AzureVisionLLMError
        )
        return self._parse_response(response.json())

    def prepare_image(self, image_path: str | bytes) -> ImageInput:
        """Return the image, shrunk when it is larger than the configured limit."""
        image = super().prepare_image(image_path)
        max_size = self.config.max_image_size or DEFAULT_MAX_IMAGE_SIZE
        if max_size <= 0:
            return image
        return _compress(image, max_size)

    def _chat_completions_url(self) -> str:
        endpoint = (self.config.azure_endpoint or "").strip().rstrip("/")
        deployment = (self.config.deployment_name or self.config.model or "").strip()
        api_version = (self.config.api_version or "").strip() or self.DEFAULT_API_VERSION
        return (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = (self.config.api_key or "").strip()
        if api_key:
            headers["api-key"] = api_key
        return headers

    def _parse_response(self, payload: Any) -> ChatResponse:
        if not isinstance(payload, dict):
            raise AzureVisionLLMError(
                f"[{self.provider_name}] unexpected response type "
                f"{type(payload).__name__}"
            )

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AzureVisionLLMError(f"[{self.provider_name}] response carries no choices")

        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise AzureVisionLLMError(
                f"[{self.provider_name}] response carries no message content"
            )

        return ChatResponse(
            content=content,
            model=str(payload.get("model") or self.config.model or ""),
        )


def _data_url(image: ImageInput) -> str:
    """Render the image as an inline data URL."""
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


def _compress(image: ImageInput, max_size: int) -> ImageInput:
    """Shrink the image so neither side exceeds `max_size` pixels."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise AzureVisionLLMError(
            "[azure] shrinking an image needs the pillow package. "
            "Install it with 'pip install pillow'."
        ) from exc

    try:
        with Image.open(io.BytesIO(image.data)) as source:
            width, height = source.size
            if width <= max_size and height <= max_size:
                return image
            scale = max_size / max(width, height)
            resized = source.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.LANCZOS,
            )
            return _encode(resized, source.format)
    except AzureVisionLLMError:
        raise
    except Exception as exc:
        raise AzureVisionLLMError(
            f"[azure] could not read the image to shrink it ({type(exc).__name__})"
        ) from exc


def _encode(image: Any, source_format: str | None) -> ImageInput:
    """Write the resized image back to bytes."""
    buffer = io.BytesIO()
    if source_format == "JPEG":
        image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
        return ImageInput(data=buffer.getvalue(), mime_type="image/jpeg")
    image.save(buffer, format="PNG")
    return ImageInput(data=buffer.getvalue(), mime_type="image/png")
