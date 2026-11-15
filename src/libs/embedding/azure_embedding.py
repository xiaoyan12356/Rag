"""Azure OpenAI embedding client.

Azure serves the OpenAI embeddings format under a deployment-scoped URL and
authenticates with an `api-key` header, so the request shaping of
`OpenAIEmbedding` is reused unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.settings import EmbeddingSettings
from libs.embedding.openai_embedding import OpenAIEmbedding, OpenAIEmbeddingError


class AzureEmbedding(OpenAIEmbedding):
    """Embeddings against an Azure OpenAI deployment."""

    DEFAULT_API_VERSION = "2024-02-01"
    provider_name = "azure"

    def __init__(self, config: EmbeddingSettings, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        if not (config.azure_endpoint or "").strip():
            raise OpenAIEmbeddingError(
                "[azure] embedding.azure_endpoint is not configured - "
                "set it in config/settings.yaml"
            )
        if not (config.deployment_name or config.model or "").strip():
            raise OpenAIEmbeddingError(
                "[azure] embedding.deployment_name is not configured - "
                "set it in config/settings.yaml"
            )

    def _base_url(self) -> str:
        endpoint = (self.config.azure_endpoint or "").strip().rstrip("/")
        deployment = (self.config.deployment_name or self.config.model or "").strip()
        return f"{endpoint}/openai/deployments/{deployment}"

    def _embeddings_url(self) -> str:
        api_version = (self.config.api_version or "").strip() or self.DEFAULT_API_VERSION
        return f"{self._base_url()}/embeddings?api-version={api_version}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = (self.config.api_key or "").strip()
        if api_key:
            headers["api-key"] = api_key
        return headers

    def _request_body(self, texts: Sequence[str]) -> dict[str, Any]:
        # The deployment selects the model, so Azure needs no model in the body.
        body: dict[str, Any] = {"input": list(texts)}
        if (self.config.model or "").startswith("text-embedding-3"):
            body["dimensions"] = self.config.dimensions
        return body
