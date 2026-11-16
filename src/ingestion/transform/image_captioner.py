"""Image captioning.

A document carries images that no text search can read. The captioner asks a
vision model what each image shows and writes the answer next to the image,
so that a question about the picture finds the page it is on.

The work only happens when the vision model is switched on and the chunk
really holds an image. When the model is off, unreachable or simply fails,
the chunk keeps its images and says that they were left alone.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.settings import Settings, resolve_path
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = "config/prompts/image_captioning.txt"

IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")
CAPTION_TEMPLATE = "[图片描述: {caption}]"
DEFAULT_PROMPT = "Describe this image for indexing purposes."
MAX_CONTEXT_LENGTH = 2000


class ImageCaptioner(BaseTransform):
    """Describe the images of a chunk with a vision model.

    The switch `vision_llm.enabled` decides whether the model is asked at
    all. When it is on but the model cannot be reached, the images stay in
    the chunk without a caption.
    """

    def __init__(
        self,
        settings: Settings,
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.enabled = settings.vision_llm.enabled
        self._vision_llm = vision_llm
        self._prompt_path = prompt_path or DEFAULT_PROMPT_PATH
        self._prompt_template: str | None = None
        self._captions: dict[str, str] = {}

    @property
    def vision_llm(self) -> BaseVisionLLM | None:
        """Return the model that describes the images, or None when there is none.

        The model is built the first time it is asked for, so a run that
        never needs it never pays for it.
        """
        if not self.enabled:
            return None
        if self._vision_llm is None:
            try:
                self._vision_llm = LLMFactory.create_vision_llm(self.settings)
            except Exception as exc:
                logger.warning(
                    "No vision LLM for image captioning, leaving the images alone: %s", exc
                )
                self.enabled = False
                return None
        return self._vision_llm

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Return the captioned chunks, in the order they came in."""
        captioned: list[Chunk] = []
        chunks_with_images = 0
        captioned_images = 0
        unprocessed_images = 0

        for chunk in chunks:
            try:
                captioned_chunk, captioned_count, unprocessed_count = self._caption_chunk(
                    chunk, trace
                )
            except Exception as exc:
                logger.error("Could not caption the images of chunk %s: %s", chunk.id, exc)
                captioned_chunk = self._chunk_with(chunk, dict(chunk.metadata), chunk.text)
                captioned_count = 0
                unprocessed_count = len(self._referenced_ids(chunk))
            if captioned_count or unprocessed_count:
                chunks_with_images += 1
            captioned_images += captioned_count
            unprocessed_images += unprocessed_count
            captioned.append(captioned_chunk)

        if trace is not None:
            trace.record_stage(
                "image_captioner",
                {
                    "total_chunks": len(chunks),
                    "chunks_with_images": chunks_with_images,
                    "captioned_images": captioned_images,
                    "unprocessed_images": unprocessed_images,
                    "enabled": self.enabled,
                },
            )

        return captioned

    def _caption_chunk(
        self, chunk: Chunk, trace: TraceContext | None
    ) -> tuple[Chunk, int, int]:
        """Return the chunk with its images described, and how many were done."""
        image_ids = self._referenced_ids(chunk)
        if not image_ids:
            return chunk, 0, 0

        images = self._image_records(chunk)
        metadata = dict(chunk.metadata)
        text = chunk.text
        captions: list[dict[str, str]] = []
        captioned_count = 0
        unprocessed_count = 0

        for image_id in image_ids:
            caption = self._caption_for(image_id, images.get(image_id), chunk.text, trace)
            if not caption:
                unprocessed_count += 1
                continue
            captioned_count += 1
            captions.append({"id": image_id, "caption": caption})
            text = _inject_caption(text, image_id, caption)

        if captions:
            metadata["image_captions"] = captions
        if unprocessed_count:
            metadata["has_unprocessed_images"] = True

        return self._chunk_with(chunk, metadata, text), captioned_count, unprocessed_count

    def _referenced_ids(self, chunk: Chunk) -> list[str]:
        """Return the ids of the images the chunk refers to."""
        refs = chunk.metadata.get("image_refs")
        if isinstance(refs, list) and refs:
            return [str(ref) for ref in refs]

        seen: list[str] = []
        for image_id in IMAGE_PLACEHOLDER_PATTERN.findall(chunk.text or ""):
            image_id = image_id.strip()
            if image_id and image_id not in seen:
                seen.append(image_id)
        return seen

    def _image_records(self, chunk: Chunk) -> dict[str, Mapping[str, Any]]:
        """Return the image references of the chunk, keyed by their id."""
        images = chunk.metadata.get("images")
        if not isinstance(images, list):
            return {}

        records: dict[str, Mapping[str, Any]] = {}
        for image in images:
            if isinstance(image, Mapping) and image.get("id"):
                records[str(image["id"])] = image
        return records

    def _caption_for(
        self,
        image_id: str,
        image: Mapping[str, Any] | None,
        context: str,
        trace: TraceContext | None,
    ) -> str | None:
        """Return the caption of the image, or None when there is none."""
        if image_id in self._captions:
            return self._captions[image_id]

        llm = self.vision_llm
        if llm is None:
            return None

        if image is None:
            logger.warning("The image %s is not described in the metadata", image_id)
            return None

        resolved = resolve_path(str(image["path"]))
        if not resolved.is_file():
            logger.warning("No image file at %s", resolved)
            return None

        try:
            response = llm.chat_with_image(self._build_prompt(context), str(resolved), trace)
        except Exception as exc:
            logger.warning("Could not caption the image %s: %s", image_id, exc)
            return None

        caption = (response.content or "").strip()
        if not caption:
            logger.warning("The model described the image %s with nothing", image_id)
            return None

        self._captions[image_id] = caption
        return caption

    def _build_prompt(self, context: str) -> str:
        """Return the prompt the model is asked with."""
        template = self._load_prompt()
        if not template:
            return DEFAULT_PROMPT
        if "{context}" not in template:
            return template
        return template.replace("{context}", (context or "")[:MAX_CONTEXT_LENGTH])

    def _load_prompt(self) -> str | None:
        """Return the prompt of the repository, or None when there is none."""
        if self._prompt_template is not None:
            return self._prompt_template

        path = resolve_path(self._prompt_path)
        if not path.exists():
            logger.warning("No prompt file at %s", path)
            return None

        try:
            self._prompt_template = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read the prompt at %s: %s", path, exc)
            return None
        return self._prompt_template

    def _chunk_with(self, chunk: Chunk, metadata: dict[str, Any], text: str) -> Chunk:
        """Return the chunk with the given metadata and text."""
        return Chunk(
            id=chunk.id,
            text=text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )


def _inject_caption(text: str, image_id: str, caption: str) -> str:
    """Return the text with the caption written under the image it belongs to."""
    placeholder = f"[IMAGE: {image_id}]"
    if placeholder not in text:
        return text
    description = CAPTION_TEMPLATE.format(caption=caption)
    pattern = re.compile(re.escape(placeholder) + r"(\s*)")
    return pattern.sub(
        lambda match: f"{placeholder}\n{description}\n{match.group(1)}", text, count=1
    )
