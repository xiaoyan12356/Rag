"""Azure OpenAI chat completions.

Azure serves the OpenAI chat completions format under a deployment-scoped URL
and authenticates with an `api-key` header instead of a bearer token.
"""

from __future__ import annotations

from typing import Any

from core.settings import LLMSettings
from libs.llm.openai_llm import OpenAILLM, OpenAILLMError


class AzureLLM(OpenAILLM):
    """Chat completions against an Azure OpenAI deployment."""

    DEFAULT_API_VERSION = "2024-02-01"
    provider_name = "azure"

    def __init__(self, config: LLMSettings, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        if not (config.azure_endpoint or "").strip():
            raise OpenAILLMError(
                "[azure] llm.azure_endpoint is not configured - set it in config/settings.yaml"
            )
        if not (config.deployment_name or config.model or "").strip():
            raise OpenAILLMError(
                "[azure] llm.deployment_name is not configured - set it in config/settings.yaml"
            )

    def _base_url(self) -> str:
        endpoint = (self.config.azure_endpoint or "").strip().rstrip("/")
        deployment = (self.config.deployment_name or self.config.model or "").strip()
        return f"{endpoint}/openai/deployments/{deployment}"

    def _chat_completions_url(self) -> str:
        api_version = (self.config.api_version or "").strip() or self.DEFAULT_API_VERSION
        return f"{self._base_url()}/chat/completions?api-version={api_version}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = (self.config.api_key or "").strip()
        if api_key:
            headers["api-key"] = api_key
        return headers
