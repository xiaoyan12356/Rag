"""Unit tests for the vision LLM abstraction and the factory integration.

A fake vision provider stands in for a real one, so the factory routing and
the multimodal input contract are covered without any network access.
"""

from __future__ import annotations

import base64

import pytest

from core.settings import Settings, VisionLLMSettings
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM, ImageInput, VisionLLMError
from libs.llm.llm_factory import LLMFactory

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
JPEG_BYTES = b"\xff\xd8\xff" + b"fake-jpeg-body"


class FakeVisionLLM(BaseVisionLLM):
    """Stands in for a real vision provider."""

    def __init__(self, config: VisionLLMSettings) -> None:
        super().__init__(config)
        self.calls: list[tuple[str, str | bytes]] = []

    def chat_with_image(self, text, image_path, trace=None) -> ChatResponse:
        self.calls.append((text, image_path))
        return ChatResponse(content="fake vision reply", model=self.config.model)


class OtherFakeVisionLLM(FakeVisionLLM):
    """Second fake, used to prove a re-registration replaces the builder."""


def make_settings(provider: str = "fake", **overrides) -> Settings:
    values = {
        "provider": provider,
        "model": "fake-vision-model",
        "api_key": "fake-key",
    }
    values.update(overrides)
    return Settings(vision_llm=VisionLLMSettings(**values))


@pytest.fixture
def register_vision_provider():
    """Register vision providers and remove them again once the test finishes."""
    registered: list[str] = []

    def _register(name: str, builder) -> str:
        LLMFactory.register_vision(name, builder)
        registered.append(name)
        return name

    yield _register

    for name in registered:
        LLMFactory.unregister_vision(name)


class TestBaseVisionLLM:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseVisionLLM(VisionLLMSettings())

    def test_subclass_must_implement_chat_with_image(self):
        class IncompleteVisionLLM(BaseVisionLLM):
            pass

        with pytest.raises(TypeError):
            IncompleteVisionLLM(VisionLLMSettings())

    def test_the_provider_answers_about_the_image(self):
        vision = FakeVisionLLM(VisionLLMSettings(model="fake-vision-model"))

        response = vision.chat_with_image("what is in this picture", "picture.png")

        assert response.content == "fake vision reply"
        assert response.model == "fake-vision-model"
        assert vision.calls == [("what is in this picture", "picture.png")]

    def test_an_empty_text_is_rejected(self):
        vision = FakeVisionLLM(VisionLLMSettings())

        with pytest.raises(ValueError):
            vision.validate_text("   ")

    def test_an_empty_image_path_is_rejected(self):
        vision = FakeVisionLLM(VisionLLMSettings())

        with pytest.raises(ValueError):
            vision.validate_image("")

    def test_empty_image_bytes_are_rejected(self):
        vision = FakeVisionLLM(VisionLLMSettings())

        with pytest.raises(ValueError):
            vision.validate_image(b"")


class TestPrepareImage:
    def test_bytes_are_returned_unchanged(self):
        vision = FakeVisionLLM(VisionLLMSettings())

        prepared = vision.prepare_image(PNG_BYTES)

        assert prepared.data == PNG_BYTES
        assert prepared.mime_type == "image/png"

    def test_the_mime_type_is_detected_from_the_bytes(self):
        vision = FakeVisionLLM(VisionLLMSettings())

        assert vision.prepare_image(JPEG_BYTES).mime_type == "image/jpeg"

    def test_a_file_on_disk_is_read(self, tmp_path):
        image_file = tmp_path / "picture.png"
        image_file.write_bytes(PNG_BYTES)
        vision = FakeVisionLLM(VisionLLMSettings())

        prepared = vision.prepare_image(str(image_file))

        assert prepared.data == PNG_BYTES
        assert prepared.mime_type == "image/png"

    def test_a_base64_payload_is_decoded(self):
        vision = FakeVisionLLM(VisionLLMSettings())
        payload = base64.b64encode(PNG_BYTES).decode("ascii")

        prepared = vision.prepare_image(payload)

        assert prepared.data == PNG_BYTES

    def test_a_payload_that_is_neither_a_file_nor_base64_is_rejected(self):
        vision = FakeVisionLLM(VisionLLMSettings())

        with pytest.raises(VisionLLMError):
            vision.prepare_image("not-an-image-at-all")

    def test_prepare_image_is_the_preprocessing_extension_point(self, tmp_path):
        """A subclass may shrink the image before it is sent."""
        image_file = tmp_path / "picture.png"
        image_file.write_bytes(PNG_BYTES)

        class ShrinkingVisionLLM(FakeVisionLLM):
            def prepare_image(self, image_path) -> ImageInput:
                prepared = super().prepare_image(image_path)
                return ImageInput(data=prepared.data[:4], mime_type=prepared.mime_type)

        vision = ShrinkingVisionLLM(VisionLLMSettings())

        prepared = vision.prepare_image(str(image_file))

        assert prepared.data == PNG_BYTES[:4]


class TestLLMFactoryVisionRouting:
    def test_create_vision_llm_returns_the_registered_provider(
        self, register_vision_provider
    ):
        register_vision_provider("fake", lambda settings: FakeVisionLLM(settings.vision_llm))

        vision = LLMFactory.create_vision_llm(make_settings("fake"))

        assert isinstance(vision, FakeVisionLLM)

    def test_create_vision_llm_passes_the_vision_configuration(
        self, register_vision_provider
    ):
        register_vision_provider("fake", lambda settings: FakeVisionLLM(settings.vision_llm))

        vision = LLMFactory.create_vision_llm(
            make_settings("fake", model="gpt-4o", deployment_name="vision-deploy")
        )

        assert vision.config.model == "gpt-4o"
        assert vision.config.deployment_name == "vision-deploy"

    def test_the_provider_name_is_matched_case_insensitively(
        self, register_vision_provider
    ):
        register_vision_provider("fake", lambda settings: FakeVisionLLM(settings.vision_llm))

        vision = LLMFactory.create_vision_llm(make_settings("FAKE"))

        assert isinstance(vision, FakeVisionLLM)

    def test_different_providers_are_routed_to_different_implementations(
        self, register_vision_provider
    ):
        register_vision_provider("fake", lambda settings: FakeVisionLLM(settings.vision_llm))
        register_vision_provider(
            "other", lambda settings: OtherFakeVisionLLM(settings.vision_llm)
        )

        assert isinstance(LLMFactory.create_vision_llm(make_settings("fake")), FakeVisionLLM)
        assert isinstance(
            LLMFactory.create_vision_llm(make_settings("other")), OtherFakeVisionLLM
        )

    def test_unknown_provider_raises_a_readable_error(self):
        with pytest.raises(VisionLLMError) as exc_info:
            LLMFactory.create_vision_llm(make_settings("does-not-exist"))

        message = str(exc_info.value)
        assert "does-not-exist" in message
        assert "Registered vision providers" in message

    def test_missing_provider_raises_a_readable_error(self):
        with pytest.raises(VisionLLMError) as exc_info:
            LLMFactory.create_vision_llm(make_settings("   "))

        assert "vision_llm.provider" in str(exc_info.value)

    def test_builder_returning_a_non_vision_llm_is_rejected(
        self, register_vision_provider
    ):
        register_vision_provider("broken", lambda settings: "not a vision llm")

        with pytest.raises(VisionLLMError) as exc_info:
            LLMFactory.create_vision_llm(make_settings("broken"))

        assert "BaseVisionLLM" in str(exc_info.value)

    def test_registering_a_provider_twice_replaces_the_builder(
        self, register_vision_provider
    ):
        register_vision_provider("fake", lambda settings: FakeVisionLLM(settings.vision_llm))
        register_vision_provider(
            "fake", lambda settings: OtherFakeVisionLLM(settings.vision_llm)
        )

        vision = LLMFactory.create_vision_llm(make_settings("fake"))

        assert isinstance(vision, OtherFakeVisionLLM)

    def test_registered_vision_providers_lists_registered_names(
        self, register_vision_provider
    ):
        register_vision_provider("fake", lambda settings: FakeVisionLLM(settings.vision_llm))

        assert "fake" in LLMFactory.registered_vision_providers()

    def test_unregister_vision_removes_the_provider(
        self, register_vision_provider
    ):
        register_vision_provider("fake", lambda settings: FakeVisionLLM(settings.vision_llm))

        LLMFactory.unregister_vision("fake")

        assert "fake" not in LLMFactory.registered_vision_providers()

    def test_registering_an_empty_provider_name_is_rejected(self):
        with pytest.raises(ValueError):
            LLMFactory.register_vision("", lambda settings: FakeVisionLLM(settings.vision_llm))

    def test_the_vision_registry_is_separate_from_the_text_registry(
        self, register_vision_provider
    ):
        register_vision_provider("fake", lambda settings: FakeVisionLLM(settings.vision_llm))

        assert "fake" in LLMFactory.registered_vision_providers()
        assert "fake" not in LLMFactory.registered_providers()
