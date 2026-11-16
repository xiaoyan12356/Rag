"""Shared fixtures for the integration tests.

The tests that talk to a real provider need credentials. When the configured
provider cannot be reached, they skip instead of failing, so the default test
run stays free of network traffic.
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from core.settings import Settings, SettingsError, load_settings

# Providers that talk to a local server instead of a hosted API.
LOCAL_PROVIDERS = {"ollama"}


def local_server_is_up(base_url: str) -> bool:
    """Return whether the local server behind `base_url` accepts connections."""
    if not base_url:
        return False
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def why_unusable(settings: Settings) -> str:
    """Return why the configured provider cannot be used, or an empty string."""
    provider = (settings.llm.provider or "").strip().lower()
    if not provider:
        return "no llm provider is configured"

    if provider in LOCAL_PROVIDERS:
        if not local_server_is_up(settings.llm.base_url):
            return f"the {provider} server at {settings.llm.base_url!r} is not reachable"
        return ""

    if not (settings.llm.api_key or "").strip():
        return f"no api key is configured for {provider}"

    if provider == "azure" and not (settings.llm.azure_endpoint or "").strip():
        return "no azure endpoint is configured"

    return ""


@pytest.fixture
def llm_settings() -> Settings:
    """Return the settings of the repository, when a provider can be reached."""
    try:
        settings = load_settings()
    except SettingsError as exc:
        pytest.skip(f"No usable configuration: {exc}")

    reason = why_unusable(settings)
    if reason:
        pytest.skip(f"Skipping the LLM test: {reason}")
    return settings
