"""Unit tests for the image captioner.

The captioner describes the images of a chunk so that a text search can find
them. The tests pin two things: a caption lands in the chunk when the model
answers, and when the model is off or fails, the chunk keeps its images and
says that they were left alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import Settings, VisionLLMSettings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.image_captioner import CAPTION_TEMPLATE, ImageCaptioner
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM

# The smallest PNG that is still a PNG.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000050001"
)

CAPTION = "A flowchart of the ingestion pipeline."


class FakeVisionLLM(BaseVisionLLM):
    """Model that answers what the test tells it to."""

    def __init__(self, answer: str = CAPTION, error: Exception | None = None) -> None:
        super().__init__(VisionLLMSettings(provider="fake", model="fake"))
        self.answer = answer
        self.error = error
        self.calls: list[dict] = []

    def chat_with_image(self, text, image_path, trace=None) -> ChatResponse:
        self.calls.append({"text": text, "image_path": image_path})
        if self.error is not None:
            raise self.error
        return ChatResponse(content=self.answer, model="fake")


def make_settings(enabled: bool = True) -> Settings:
    settings = Settings()
    settings.vision_llm.enabled = enabled
    return settings


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    """A real image file on disk, so the captioner can find it."""
    path = tmp_path / "images" / "diagram.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG)
    return path


def make_chunk(
    image_id: str = "img_1",
    image_path: str | None = None,
    with_image: bool = True,
) -> Chunk:
    text = (
        "Some text about the pipeline."
        if not with_image
        else f"Some text about the pipeline.\n\n[IMAGE: {image_id}]"
    )
    metadata: dict = {"source_path": "data/documents/sample.pdf", "chunk_index": 0}
    if with_image:
        metadata["image_refs"] = [image_id]
        metadata["images"] = [
            {
                "id": image_id,
                "path": image_path or "data/images/doc/image.png",
                "page": 1,
                "text_offset": 0,
                "text_length": len(f"[IMAGE: {image_id}]"),
            }
        ]
    return Chunk(id="chunk_1", text=text, metadata=metadata, source_ref="doc_1")


class TestEnabledMode:
    def test_the_model_is_asked_about_the_image(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)

        captioner.transform([make_chunk(image_path=str(image_file))])

        assert len(llm.calls) == 1
        assert llm.calls[0]["image_path"] == str(image_file)

    def test_the_caption_is_written_to_the_metadata(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)

        captioned = captioner.transform([make_chunk(image_path=str(image_file))])

        assert captioned[0].metadata["image_captions"] == [
            {"id": "img_1", "caption": CAPTION}
        ]

    def test_the_caption_is_written_next_to_the_image(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)

        captioned = captioner.transform([make_chunk(image_path=str(image_file))])

        text = captioned[0].text
        description = CAPTION_TEMPLATE.format(caption=CAPTION)
        assert text.index("[IMAGE: img_1]") < text.index(CAPTION)
        assert f"[IMAGE: img_1]\n{description}" in text

    def test_the_chunk_is_not_marked_as_unprocessed(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)

        captioned = captioner.transform([make_chunk(image_path=str(image_file))])

        assert "has_unprocessed_images" not in captioned[0].metadata

    def test_the_model_is_given_the_context_of_the_image(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)

        captioner.transform([make_chunk(image_path=str(image_file))])

        assert "Some text about the pipeline." in llm.calls[0]["text"]

    def test_the_same_image_is_asked_about_only_once(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)
        chunks = [
            make_chunk(image_path=str(image_file)),
            make_chunk(image_path=str(image_file)),
        ]

        captioner.transform(chunks)

        assert len(llm.calls) == 1

    def test_the_chunk_keeps_its_images(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)

        captioned = captioner.transform([make_chunk(image_path=str(image_file))])

        assert captioned[0].metadata["image_refs"] == ["img_1"]
        assert captioned[0].metadata["images"][0]["id"] == "img_1"

    def test_the_chunk_keeps_its_identity(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)

        captioned = captioner.transform([make_chunk(image_path=str(image_file))])

        assert captioned[0].id == "chunk_1"
        assert captioned[0].source_ref == "doc_1"
        assert captioned[0].metadata["source_path"] == "data/documents/sample.pdf"

    def test_a_chunk_without_an_image_is_left_alone(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)
        chunk = make_chunk(with_image=False)

        captioned = captioner.transform([chunk])

        assert captioned[0] is chunk
        assert llm.calls == []

    def test_the_trace_records_what_happened(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)
        trace = TraceContext()

        captioner.transform([make_chunk(image_path=str(image_file))], trace)

        data = trace.get_stage_data("image_captioner")
        assert data["total_chunks"] == 1
        assert data["captioned_images"] == 1
        assert data["unprocessed_images"] == 0


class TestFallback:
    def test_a_disabled_model_leaves_the_images_alone(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(enabled=False), vision_llm=llm)

        captioned = captioner.transform([make_chunk(image_path=str(image_file))])

        assert "image_captions" not in captioned[0].metadata
        assert captioned[0].metadata["has_unprocessed_images"] is True
        assert llm.calls == []

    def test_a_failing_model_leaves_the_images_alone(self, image_file: Path):
        llm = FakeVisionLLM(error=RuntimeError("the model is down"))
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)

        captioned = captioner.transform([make_chunk(image_path=str(image_file))])

        assert "image_captions" not in captioned[0].metadata
        assert captioned[0].metadata["has_unprocessed_images"] is True

    def test_an_empty_answer_leaves_the_images_alone(self, image_file: Path):
        llm = FakeVisionLLM(answer="   ")
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)

        captioned = captioner.transform([make_chunk(image_path=str(image_file))])

        assert captioned[0].metadata["has_unprocessed_images"] is True

    def test_a_missing_image_file_leaves_the_images_alone(self, tmp_path: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)

        captioned = captioner.transform([make_chunk(image_path=str(tmp_path / "nowhere.png"))])

        assert captioned[0].metadata["has_unprocessed_images"] is True
        assert llm.calls == []

    def test_an_image_without_a_record_leaves_the_images_alone(self):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)
        chunk = Chunk(
            id="chunk_1",
            text="Text with an image in it.\n\n[IMAGE: img_1]",
            metadata={
                "source_path": "data/documents/sample.pdf",
                "image_refs": ["img_1"],
            },
            source_ref="doc_1",
        )

        captioned = captioner.transform([chunk])

        assert captioned[0].metadata["has_unprocessed_images"] is True
        assert llm.calls == []

    def test_the_images_stay_in_the_chunk(self, image_file: Path):
        llm = FakeVisionLLM(error=RuntimeError("the model is down"))
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)

        captioned = captioner.transform([make_chunk(image_path=str(image_file))])

        assert captioned[0].metadata["image_refs"] == ["img_1"]
        assert captioned[0].metadata["images"][0]["id"] == "img_1"
        assert "[IMAGE: img_1]" in captioned[0].text

    def test_one_broken_image_does_not_stop_the_others(self, image_file: Path, tmp_path: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)
        good = make_chunk("img_1", str(image_file))
        bad = make_chunk("img_2", str(tmp_path / "nowhere.png"))

        captioned = captioner.transform([good, bad])

        assert captioned[0].metadata["image_captions"] == [
            {"id": "img_1", "caption": CAPTION}
        ]
        assert captioned[1].metadata["has_unprocessed_images"] is True

    def test_the_trace_counts_the_images_left_alone(self, image_file: Path):
        llm = FakeVisionLLM(error=RuntimeError("the model is down"))
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)
        trace = TraceContext()

        captioner.transform([make_chunk(image_path=str(image_file))], trace)

        data = trace.get_stage_data("image_captioner")
        assert data["captioned_images"] == 0
        assert data["unprocessed_images"] == 1


class TestTransform:
    def test_every_chunk_comes_back(self, image_file: Path):
        captioner = ImageCaptioner(make_settings(), vision_llm=FakeVisionLLM())
        chunks = [make_chunk(image_path=str(image_file)), make_chunk(with_image=False)]

        captioned = captioner.transform(chunks)

        assert len(captioned) == 2

    def test_no_chunks_gives_no_chunks(self):
        captioner = ImageCaptioner(make_settings(), vision_llm=FakeVisionLLM())

        assert captioner.transform([]) == []

    def test_the_chunk_that_came_in_is_left_alone(self, image_file: Path):
        captioner = ImageCaptioner(make_settings(), vision_llm=FakeVisionLLM())
        chunk = make_chunk(image_path=str(image_file))

        captioner.transform([chunk])

        assert "image_captions" not in chunk.metadata

    def test_the_placeholder_is_found_in_the_text(self, image_file: Path):
        llm = FakeVisionLLM()
        captioner = ImageCaptioner(make_settings(), vision_llm=llm)
        chunk = Chunk(
            id="chunk_1",
            text="Text with an image in it.\n\n[IMAGE: img_9]",
            metadata={
                "source_path": "data/documents/sample.pdf",
                "images": [{"id": "img_9", "path": str(image_file)}],
            },
            source_ref="doc_1",
        )

        captioned = captioner.transform([chunk])

        assert captioned[0].metadata["image_captions"] == [
            {"id": "img_9", "caption": CAPTION}
        ]
        assert len(llm.calls) == 1


class TestPrompt:
    def test_the_prompt_of_the_repository_is_used(self):
        captioner = ImageCaptioner(make_settings())

        template = captioner._load_prompt()

        assert template is not None
        assert "{context}" in template

    def test_the_prompt_is_read_once(self):
        captioner = ImageCaptioner(make_settings())

        first = captioner._load_prompt()
        second = captioner._load_prompt()

        assert first is second

    def test_a_missing_prompt_is_reported(self, tmp_path: Path):
        captioner = ImageCaptioner(make_settings(), prompt_path=tmp_path / "nowhere.txt")

        assert captioner._load_prompt() is None

    def test_the_prompt_without_a_placeholder_is_used_as_it_is(self, tmp_path: Path):
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("Describe this image.", encoding="utf-8")
        captioner = ImageCaptioner(make_settings(), prompt_path=prompt)

        assert captioner._build_prompt("Some text.") == "Describe this image."


class TestBaseTransform:
    def test_the_base_class_cannot_be_used_on_its_own(self):
        with pytest.raises(TypeError, match="abstract"):
            BaseTransform()

    def test_a_captioner_is_a_transform(self):
        assert isinstance(ImageCaptioner(make_settings()), BaseTransform)
