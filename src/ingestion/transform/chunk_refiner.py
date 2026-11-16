"""Chunk refinement.

A chunk that came out of a PDF carries the marks of the page it was taken
from: headers, footers, blank space, HTML left over from the conversion.
The refiner takes that noise away.

It works in two steps. The rules clean what the rules can recognise, which
is fast and never fails. When the LLM is switched on, the model rewrites the
cleaned text into a self-contained unit. If the model is unavailable or
misbehaves, the rule-based result stands and the chunk says why.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from core.settings import Settings, resolve_path
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = "config/prompts/chunk_refinement.txt"

# A fenced code block holds text that was written to be read as it is, so it
# is set aside before the rules run and put back afterwards.
_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# A line that holds nothing but a rule or a symbol is a leftover of the page
# layout, not content.
_SEPARATOR_LINE_PATTERN = re.compile(r"^[-─━═–—_=*~]{3,}$")
_NOISE_LINE_PATTERN = re.compile(r"^(?:[^\w\s]\s*){3,}$")

# The lines that repeat on every page of a document.
_PAGE_LINE_PATTERN = re.compile(r"^\s*page\s+\d+\b.*$", re.IGNORECASE)
_DECORATED_PAGE_PATTERN = re.compile(r"^\s*[-–—]\s*\d{1,4}\s*[-–—]\s*$")
_HEADER_FOOTER_PATTERN = re.compile(r"^\s*(header|footer)\b.*$", re.IGNORECASE)
_BOILERPLATE_PATTERN = re.compile(
    r"^\s*(©|\(c\)\s|confidential\b|all rights reserved\b).*$", re.IGNORECASE
)

# Whitespace inside a line, but not the indentation that gives a nested list
# its shape.
_MULTI_SPACE_PATTERN = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")
_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


class ChunkRefiner(BaseTransform):
    """Clean the chunks of a document, and optionally let the LLM rewrite them.

    The switch `ingestion.chunk_refiner.use_llm` decides whether the model
    is asked at all. When it is on but the model cannot be reached, the
    refiner keeps working with the rules alone.
    """

    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.use_llm = settings.ingestion.chunk_refiner.use_llm
        self._llm = llm
        self._prompt_path = prompt_path or DEFAULT_PROMPT_PATH
        self._prompt_template: str | None = None

    @property
    def llm(self) -> BaseLLM | None:
        """Return the model that rewrites the text, or None when there is none.

        The model is built the first time it is asked for, so a run that
        never needs it never pays for it.
        """
        if not self.use_llm:
            return None
        if self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
            except Exception as exc:
                logger.warning(
                    "No LLM for chunk refinement, keeping the rule-based result: %s", exc
                )
                self.use_llm = False
                return None
        return self._llm

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Return the refined chunks, in the order they came in."""
        refined: list[Chunk] = []
        counts = {"llm": 0, "rule": 0, "failed": 0}

        for chunk in chunks:
            try:
                refined_chunk, refined_by = self._refine_chunk(chunk, trace)
            except Exception as exc:
                logger.error("Could not refine chunk %s: %s", chunk.id, exc)
                refined_chunk = chunk
                refined_by = "failed"
            counts[refined_by] += 1
            refined.append(refined_chunk)

        if trace is not None:
            trace.record_stage(
                "chunk_refiner",
                {
                    "total_chunks": len(chunks),
                    "refined_by_llm": counts["llm"],
                    "refined_by_rule": counts["rule"],
                    "failed": counts["failed"],
                    "use_llm": self.use_llm,
                },
            )

        return refined

    def _refine_chunk(self, chunk: Chunk, trace: TraceContext | None) -> tuple[Chunk, str]:
        """Return the refined chunk and who refined it."""
        rule_text = self._rule_based_refine(chunk.text)
        metadata = dict(chunk.metadata)

        if not self.use_llm:
            metadata["refined_by"] = "rule"
            return (
                Chunk(
                    id=chunk.id,
                    text=rule_text,
                    metadata=metadata,
                    source_ref=chunk.source_ref,
                ),
                "rule",
            )

        llm_text = self._llm_refine(rule_text, trace)
        if llm_text is None:
            metadata["refined_by"] = "rule"
            metadata["refine_fallback_reason"] = "llm_failed"
            return (
                Chunk(
                    id=chunk.id,
                    text=rule_text,
                    metadata=metadata,
                    source_ref=chunk.source_ref,
                ),
                "rule",
            )

        metadata["refined_by"] = "llm"
        return (
            Chunk(
                id=chunk.id,
                text=llm_text,
                metadata=metadata,
                source_ref=chunk.source_ref,
            ),
            "llm",
        )

    def _rule_based_refine(self, text: str) -> str:
        """Return the text without the marks the page layout left on it.

        Page headers and footers, HTML comments and tags, rules that only
        separate, and runs of blank space all go. What is left is the text
        of the document, with its Markdown and its code blocks untouched.
        """
        if not text:
            return text
        if not text.strip():
            return ""

        code_blocks: list[str] = []

        def set_aside(match: re.Match[str]) -> str:
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        text = _CODE_BLOCK_PATTERN.sub(set_aside, text)
        text = _HTML_COMMENT_PATTERN.sub("", text)
        text = _HTML_TAG_PATTERN.sub("", text)

        kept = [line for line in text.split("\n") if not _is_noise_line(line)]
        text = "\n".join(kept)

        text = "\n".join(_MULTI_SPACE_PATTERN.sub(" ", line).rstrip() for line in text.split("\n"))
        text = _BLANK_LINES_PATTERN.sub("\n\n", text)

        for index, code_block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{index}__", code_block)

        return text.strip()

    def _llm_refine(self, text: str, trace: TraceContext | None = None) -> str | None:
        """Return the text as the model rewrote it, or None when it could not.

        None means the rules have to carry the chunk on their own, which is
        why nothing here raises.
        """
        if not text or not text.strip():
            return None

        llm = self.llm
        if llm is None:
            return None

        try:
            template = self._load_prompt()
            if not template:
                logger.warning("No prompt for chunk refinement, keeping the rule-based result")
                return None
            if "{text}" not in template:
                logger.warning(
                    "The prompt has no {text} placeholder, keeping the rule-based result"
                )
                return None

            prompt = template.replace("{text}", text)
            response = llm.chat([{"role": "user", "content": prompt}], trace)
            refined = response if isinstance(response, str) else response.content
            if refined and refined.strip():
                return refined.strip()
            logger.warning("The model returned nothing, keeping the rule-based result")
            return None
        except Exception as exc:
            logger.warning(
                "Could not refine with the model, keeping the rule-based result: %s", exc
            )
            return None

    def _load_prompt(self) -> str | None:
        """Return the prompt the model is asked with."""
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


def _is_noise_line(line: str) -> bool:
    """Return whether a line is a leftover of the page layout rather than content."""
    stripped = line.strip()
    if not stripped:
        return False
    if _SEPARATOR_LINE_PATTERN.match(stripped):
        return True
    if _NOISE_LINE_PATTERN.match(stripped):
        return True
    if _PAGE_LINE_PATTERN.match(stripped):
        return True
    if _DECORATED_PAGE_PATTERN.match(stripped):
        return True
    if _HEADER_FOOTER_PATTERN.match(stripped):
        return True
    if _BOILERPLATE_PATTERN.match(stripped):
        return True
    return False
