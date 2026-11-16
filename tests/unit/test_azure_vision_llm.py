"""Unit tests for the Azure vision LLM.

Every request is served by `httpx.MockTransport`, so the request shaping, the
image shrinking and the error handling are covered without touching the
network.
"""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest

from core.settings import Settings, VisionLLMSettings
from libs.llm.azure_vision_llm import AzureVisionLLM, AzureVisionLLMError
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory

API_KEY = "sk-test-key-do-not-log"
ENDPOINT = "https://my-resource.openai.azure.com"


def make_vision_settings(**overrides) -> VisionLLMSettings:
    values = {
        "enabled": True,
        "provider": "azure",
        "model": "gpt-4o",
        "deployment_name": "gpt-4o-vision",
        "azure_endpoint": ENDPOINT,
        "api_version": "2024-02-01",
        "api_key": API_KEY,
    }
    values.update(overrides)
    return VisionLLMSettings(**values)


def make_settings(**overrides) -> Settings:
    return Settings(vision_llm=make_vision_settings(**overrides))


def chat_payload(content: str = "a blue square") -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def recording_transport(payload=None, status_code: int = 200, error=None):
    """Return a transport that answers with `payload` and records the request."""
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if error is not None:
            raise error
        return httpx.Response(status_code, json=payload if payload is not None else chat_payload())

    return httpx.MockTransport(handler), recorded


def png_bytes(width: int = 4, height: int = 4) -> bytes:
    """Build a real PNG of the given size."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(width: int = 4, height: int = 4) -> bytes:
    """Build a real JPEG of the given size."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_vision(settings: Settings | None = None, **kwargs) -> AzureVisionLLM:
    transport, _ = recording_transport(**kwargs)
    config = settings.vision_llm if settings else make_vision_settings()
    return AzureVisionLLM(config, transport=transport)


def image_size(data: bytes) -> tuple[int, int]:
    """Return the pixel size of an encoded image."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return image.size


def sent_image_data(request: httpx.Request) -> bytes:
    """Return the image bytes carried by the recorded request."""
    body = json.loads(request.content)
    url = body["messages"][0]["content"][1]["image_url"]["url"]
    _, _, payload = url.partition(",")
    return base64.b64decode(payload)


class TestNormalCall:
    def test_the_reply_is_returned(self):
        vision = make_vision()

        response = vision.chat_with_image("what is in this picture", png_bytes())

        assert response.content == "a blue square"
        assert response.model == "gpt-4o"

    def test_the_request_targets_the_deployment_url(self):
        transport, recorded = recording_transport()

        AzureVisionLLM(make_vision_settings(), transport=transport).chat_with_image(
            "what is in this picture", png_bytes()
        )

        url = recorded[0].url
        assert url.host == "my-resource.openai.azure.com"
        assert url.path == "/openai/deployments/gpt-4o-vision/chat/completions"
        assert url.params["api-version"] == "2024-02-01"

    def test_the_request_authenticates_with_the_api_key_header(self):
        transport, recorded = recording_transport()

        AzureVisionLLM(make_vision_settings(), transport=transport).chat_with_image(
            "what is in this picture", png_bytes()
        )

        assert recorded[0].headers["api-key"] == API_KEY
        assert "Authorization" not in recorded[0].headers

    def test_the_request_carries_the_text_and_the_image(self):
        transport, recorded = recording_transport()

        AzureVisionLLM(make_vision_settings(), transport=transport).chat_with_image(
            "what is in this picture", png_bytes()
        )

        content = json.loads(recorded[0].content)["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "what is in this picture"}
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_an_image_on_disk_is_sent(self, tmp_path):
        image_file = tmp_path / "picture.png"
        image_file.write_bytes(png_bytes())
        transport, recorded = recording_transport()

        AzureVisionLLM(make_vision_settings(), transport=transport).chat_with_image(
            "what is in this picture", str(image_file)
        )

        assert sent_image_data(recorded[0]) == png_bytes()

    def test_a_base64_payload_is_sent(self):
        payload = base64.b64encode(png_bytes()).decode("ascii")
        transport, recorded = recording_transport()

        AzureVisionLLM(make_vision_settings(), transport=transport).chat_with_image(
            "what is in this picture", payload
        )

        assert sent_image_data(recorded[0]) == png_bytes()

    def test_the_mime_type_follows_the_image(self):
        transport, recorded = recording_transport()

        AzureVisionLLM(make_vision_settings(), transport=transport).chat_with_image(
            "what is in this picture", jpeg_bytes()
        )

        url = json.loads(recorded[0].content)["messages"][0]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_an_empty_text_is_rejected(self):
        vision = make_vision()

        with pytest.raises(AzureVisionLLMError):
            vision.chat_with_image("   ", png_bytes())


class TestImageCompression:
    def test_a_large_image_is_shrunk_to_the_configured_size(self):
        transport, recorded = recording_transport()

        AzureVisionLLM(
            make_vision_settings(max_image_size=64), transport=transport
        ).chat_with_image("what is in this picture", png_bytes(400, 200))

        width, height = image_size(sent_image_data(recorded[0]))
        assert max(width, height) == 64
        assert (width, height) == (64, 32)

    def test_a_small_image_is_left_alone(self):
        transport, recorded = recording_transport()

        AzureVisionLLM(
            make_vision_settings(max_image_size=2048), transport=transport
        ).chat_with_image("what is in this picture", png_bytes(40, 20))

        assert sent_image_data(recorded[0]) == png_bytes(40, 20)

    def test_the_default_limit_is_2048_pixels(self):
        assert make_vision_settings().max_image_size == 2048

    def test_a_large_jpeg_stays_a_jpeg(self):
        transport, recorded = recording_transport()

        AzureVisionLLM(
            make_vision_settings(max_image_size=32), transport=transport
        ).chat_with_image("what is in this picture", jpeg_bytes(200, 100))

        url = json.loads(recorded[0].content)["messages"][0]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_the_shrunk_image_is_still_readable(self):
        transport, recorded = recording_transport()

        AzureVisionLLM(
            make_vision_settings(max_image_size=64), transport=transport
        ).chat_with_image("what is in this picture", png_bytes(400, 200))

        assert image_size(sent_image_data(recorded[0])) == (64, 32)


class TestErrorHandling:
    def test_an_authentication_failure_reports_the_azure_error_code(self):
        transport, _ = recording_transport(
            payload={
                "error": {
                    "code": "401",
                    "message": "Access denied due to invalid subscription key",
                }
            },
            status_code=401,
        )

        vision = AzureVisionLLM(make_vision_settings(), transport=transport)

        with pytest.raises(AzureVisionLLMError) as exc_info:
            vision.chat_with_image("what is in this picture", png_bytes())

        message = str(exc_info.value)
        assert "401" in message
        assert "invalid subscription key" in message
        assert API_KEY not in message

    def test_a_timeout_is_reported_without_leaking_the_api_key(self):
        transport, _ = recording_transport(error=httpx.ReadTimeout("too slow"))

        vision = AzureVisionLLM(make_vision_settings(), transport=transport)

        with pytest.raises(AzureVisionLLMError) as exc_info:
            vision.chat_with_image("what is in this picture", png_bytes())

        message = str(exc_info.value)
        assert "timed out" in message
        assert API_KEY not in message

    def test_a_connection_failure_is_reported_without_leaking_the_api_key(self):
        transport, _ = recording_transport(error=httpx.ConnectError("refused"))

        vision = AzureVisionLLM(make_vision_settings(), transport=transport)

        with pytest.raises(AzureVisionLLMError) as exc_info:
            vision.chat_with_image("what is in this picture", png_bytes())

        message = str(exc_info.value)
        assert "could not reach" in message
        assert API_KEY not in message

    def test_a_response_without_choices_is_rejected(self):
        transport, _ = recording_transport(payload={"id": "chatcmpl-1"})

        vision = AzureVisionLLM(make_vision_settings(), transport=transport)

        with pytest.raises(AzureVisionLLMError, match="no choices"):
            vision.chat_with_image("what is in this picture", png_bytes())

    def test_a_response_without_content_is_rejected(self):
        transport, _ = recording_transport(payload={"choices": [{"message": {}}]})

        vision = AzureVisionLLM(make_vision_settings(), transport=transport)

        with pytest.raises(AzureVisionLLMError, match="no message content"):
            vision.chat_with_image("what is in this picture", png_bytes())

    def test_a_missing_azure_endpoint_is_rejected_at_creation(self):
        with pytest.raises(AzureVisionLLMError, match="azure_endpoint"):
            AzureVisionLLM(make_vision_settings(azure_endpoint=""))

    def test_a_missing_deployment_name_is_rejected_at_creation(self):
        with pytest.raises(AzureVisionLLMError, match="deployment_name"):
            AzureVisionLLM(make_vision_settings(deployment_name="", model=""))


class TestFactoryRouting:
    def test_the_factory_creates_the_azure_vision_llm(self):
        vision = LLMFactory.create_vision_llm(make_settings())

        assert isinstance(vision, AzureVisionLLM)
        assert isinstance(vision, BaseVisionLLM)

    def test_the_factory_passes_the_vision_configuration(self):
        vision = LLMFactory.create_vision_llm(
            make_settings(deployment_name="gpt-4o-vision", max_image_size=1024)
        )

        assert vision.config.deployment_name == "gpt-4o-vision"
        assert vision.config.max_image_size == 1024

    def test_the_provider_name_is_matched_case_insensitively(self):
        vision = LLMFactory.create_vision_llm(make_settings(provider="AZURE"))

        assert isinstance(vision, AzureVisionLLM)
