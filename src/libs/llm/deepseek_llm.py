"""DeepSeek chat completions.

DeepSeek exposes the OpenAI chat completions format, so only the endpoint
differs from the OpenAI client.
"""

from __future__ import annotations

from libs.llm.openai_llm import OpenAILLM


class DeepSeekLLM(OpenAILLM):
    """Chat completions against the DeepSeek OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    provider_name = "deepseek"
