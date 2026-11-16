"""Metadata enrichment.

A chunk that was split out of a document carries the words of the document
but none of the words that describe them. The enricher adds them: a title, a
summary and the tags that make the chunk findable.

The rules always run, so every chunk comes back with metadata even when the
model is not there. When the model is asked, its answer replaces what the
rules found. If it cannot answer, the rules stand and the chunk says why.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from core.settings import Settings, resolve_path
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = "config/prompts/metadata_enrichment.txt"

TITLE_MAX_LENGTH = 150
SUMMARY_MAX_LENGTH = 500
SUMMARY_SENTENCES = 3
MAX_TAGS = 10
# Only the beginning of a chunk is worth sending to the model.
MAX_LLM_TEXT = 2000

_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+)$")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_IDENTIFIER_PATTERN = re.compile(r"\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+_[a-z_]+\b")
_EMPHASIS_PATTERN = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_ARTICLE_PATTERN = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_CODE_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")

_STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "not",
        "but",
        "you",
        "your",
        "its",
        "our",
        "their",
        "they",
        "them",
        "his",
        "her",
        "she",
        "him",
        "all",
        "any",
        "can",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "into",
        "than",
        "then",
        "there",
        "these",
        "those",
        "when",
        "where",
        "which",
        "while",
        "what",
        "who",
        "how",
        "why",
        "also",
        "more",
        "most",
        "some",
        "such",
        "only",
        "other",
        "over",
        "under",
        "about",
        "after",
        "before",
        "between",
        "because",
        "each",
        "very",
        "just",
        "like",
        "does",
        "done",
        "being",
        "both",
        "same",
        "still",
        "one",
        "two",
        "use",
        "used",
        "using",
    }
)


class MetadataEnricher(BaseTransform):
    """Give every chunk a title, a summary and tags.

    The switch `ingestion.metadata_enricher.use_llm` decides whether the
    model is asked at all. When it is on but the model cannot be reached, the
    rules carry on alone.
    """

    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.use_llm = settings.ingestion.metadata_enricher.use_llm
        self._llm = llm
        self._prompt_path = prompt_path or DEFAULT_PROMPT_PATH
        self._prompt_template: str | None = None

    @property
    def llm(self) -> BaseLLM | None:
        """Return the model that describes the chunks, or None when there is none.

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
                    "No LLM for metadata enrichment, keeping the rule-based result: %s", exc
                )
                self.use_llm = False
                return None
        return self._llm

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Return the enriched chunks, in the order they came in."""
        enriched: list[Chunk] = []
        counts = {"llm": 0, "rule": 0, "failed": 0}

        for chunk in chunks:
            try:
                enriched_chunk, enriched_by = self._enrich_chunk(chunk, trace)
            except Exception as exc:
                logger.error("Could not enrich chunk %s: %s", chunk.id, exc)
                metadata = dict(chunk.metadata)
                metadata.update(self._minimal_metadata(chunk.text))
                metadata["enriched_by"] = "failed"
                enriched_chunk = self._chunk_with(chunk, metadata)
                enriched_by = "failed"
            counts[enriched_by] += 1
            enriched.append(enriched_chunk)

        if trace is not None:
            trace.record_stage(
                "metadata_enricher",
                {
                    "total_chunks": len(chunks),
                    "enriched_by_llm": counts["llm"],
                    "enriched_by_rule": counts["rule"],
                    "failed": counts["failed"],
                    "use_llm": self.use_llm,
                },
            )

        return enriched

    def _enrich_chunk(self, chunk: Chunk, trace: TraceContext | None) -> tuple[Chunk, str]:
        """Return the enriched chunk and who enriched it."""
        rule_metadata = self._rule_based_enrich(chunk.text)

        if not self.use_llm:
            metadata = dict(chunk.metadata)
            metadata.update(rule_metadata)
            metadata["enriched_by"] = "rule"
            return self._chunk_with(chunk, metadata), "rule"

        llm_metadata = self._llm_enrich(chunk.text, trace)
        if llm_metadata is None:
            metadata = dict(chunk.metadata)
            metadata.update(rule_metadata)
            metadata["enriched_by"] = "rule"
            metadata["enrich_fallback_reason"] = "llm_failed"
            return self._chunk_with(chunk, metadata), "rule"

        metadata = dict(chunk.metadata)
        metadata.update(_merge(rule_metadata, llm_metadata))
        metadata["enriched_by"] = "llm"
        return self._chunk_with(chunk, metadata), "llm"

    def _rule_based_enrich(self, text: str) -> dict[str, Any]:
        """Return the title, the summary and the tags found in the text."""
        return {
            "title": self._extract_title(text),
            "summary": self._extract_summary(text),
            "tags": self._extract_tags(text),
        }

    def _extract_title(self, text: str) -> str:
        """Return a name for the chunk, taken from the text itself."""
        if not text or not text.strip():
            return "Untitled"

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:3]:
            heading = _HEADING_PATTERN.match(line)
            if heading:
                return _truncate(heading.group(1).strip(), TITLE_MAX_LENGTH)

        first_line = lines[0] if lines else ""
        if first_line and len(first_line) <= TITLE_MAX_LENGTH:
            if not first_line.endswith((".", ",", ";", ":")):
                return first_line

        sentences = [part for part in _SENTENCE_PATTERN.split(text.strip()) if part]
        if sentences:
            return _truncate(sentences[0].strip().rstrip(".!?"), TITLE_MAX_LENGTH)

        return _truncate(text.strip(), TITLE_MAX_LENGTH)

    def _extract_summary(self, text: str) -> str:
        """Return the first sentences of the chunk as its summary.

        A heading is a title rather than content, so it is left out of the
        summary.
        """
        if not text or not text.strip():
            return ""

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        body_lines = [line for line in lines if not _HEADING_PATTERN.match(line)]
        body = " ".join(body_lines or lines)

        sentences = [part.strip() for part in _SENTENCE_PATTERN.split(body) if part.strip()]
        summary = " ".join(sentences[:SUMMARY_SENTENCES]).strip()
        return _truncate(summary, SUMMARY_MAX_LENGTH)

    def _extract_tags(self, text: str) -> list[str]:
        """Return the words and phrases that describe the chunk."""
        if not text or not text.strip():
            return []

        tags: list[str] = []

        def add(value: str) -> None:
            value = value.strip().strip(".,;:!?")
            if value and value.lower() not in _STOP_WORDS and value not in tags:
                tags.append(value)

        for phrase in _PROPER_NOUN_PATTERN.findall(text)[:5]:
            add(_ARTICLE_PATTERN.sub("", phrase))
        for identifier in _IDENTIFIER_PATTERN.findall(text)[:5]:
            add(identifier)
        for match in _EMPHASIS_PATTERN.findall(text)[:5]:
            add(next((part for part in match if part), ""))

        counts = Counter(
            word.lower()
            for word in _WORD_PATTERN.findall(text)
            if word.lower() not in _STOP_WORDS
        )
        for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if len(tags) >= MAX_TAGS:
                break
            add(word)

        if not tags:
            for word in text.split():
                add(word)
                if len(tags) >= 3:
                    break

        return tags[:MAX_TAGS]

    def _llm_enrich(self, text: str, trace: TraceContext | None = None) -> dict[str, Any] | None:
        """Return the metadata as the model wrote it, or None when it could not.

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
                logger.warning("No prompt for metadata enrichment, keeping the rule-based result")
                return None
            if "{chunk_text}" not in template:
                logger.warning(
                    "The prompt has no {chunk_text} placeholder, keeping the rule-based result"
                )
                return None

            prompt = template.replace("{chunk_text}", text[:MAX_LLM_TEXT])
            response = llm.chat([{"role": "user", "content": prompt}], trace)
            answer = response if isinstance(response, str) else response.content
            metadata = self._parse_llm_response(answer)
            if metadata is None:
                logger.warning(
                    "The model did not answer with metadata, keeping the rule-based result"
                )
                return None
            return metadata
        except Exception as exc:
            logger.warning(
                "Could not enrich with the model, keeping the rule-based result: %s", exc
            )
            return None

    def _parse_llm_response(self, response: str) -> dict[str, Any] | None:
        """Return the metadata the model answered with, or None when there is none."""
        if not response or not response.strip():
            return None

        text = _CODE_FENCE_PATTERN.sub("", response.strip())
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None

        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        title = payload.get("title")
        summary = payload.get("summary")
        tags = payload.get("tags")

        metadata: dict[str, Any] = {}
        metadata["title"] = title.strip() if isinstance(title, str) else ""
        metadata["summary"] = summary.strip() if isinstance(summary, str) else ""
        if isinstance(tags, str):
            metadata["tags"] = [part.strip() for part in tags.split(",") if part.strip()]
        elif isinstance(tags, list):
            metadata["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]
        else:
            metadata["tags"] = []

        if not metadata["title"]:
            return None
        return metadata

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

    def _minimal_metadata(self, text: str) -> dict[str, Any]:
        """Return what is known about a chunk that could not be enriched."""
        return {
            "title": "Untitled",
            "summary": (text or "")[:SUMMARY_MAX_LENGTH],
            "tags": [],
        }

    def _chunk_with(self, chunk: Chunk, metadata: dict[str, Any]) -> Chunk:
        """Return the chunk with the given metadata."""
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )


def _merge(rule_metadata: dict[str, Any], llm_metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the metadata of the model, with the rules filling the gaps."""
    merged = dict(rule_metadata)
    for key, value in llm_metadata.items():
        if value:
            merged[key] = value
    return merged


def _truncate(text: str, limit: int) -> str:
    """Return the text, shortened to the limit."""
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
