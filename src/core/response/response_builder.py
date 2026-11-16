"""The response of a tool.

A tool answers in two ways at once: a Markdown text a reader can read, and
a structured part a program can use. The text carries the citation markers
`[1]`, `[2]`, ... that point at the structured citations, so the reader can
check where a passage came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.response.citation_generator import Citation, CitationGenerator
from core.types import RetrievalResult

DEFAULT_SNIPPET_LENGTH = 300


@dataclass
class MCPToolResult:
    """What a tool answered.

    Attributes:
        content: The content blocks, the first one being the text.
        structured_content: The part a program reads.
        is_error: Whether the tool ran into a problem the caller should know.
    """

    content: list[dict[str, Any]] = field(default_factory=list)
    structured_content: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the result in the shape the protocol expects."""
        return {
            "content": list(self.content),
            "structuredContent": dict(self.structured_content),
            "isError": self.is_error,
        }


class ResponseBuilder:
    """Build the answer of a tool out of the passages the search returned."""

    def __init__(
        self,
        citation_generator: CitationGenerator | None = None,
        snippet_max_length: int = DEFAULT_SNIPPET_LENGTH,
    ) -> None:
        """Set up the builder.

        Args:
            citation_generator: The one that numbers the citations; a plain
                one is used when this is left out.
            snippet_max_length: How long a passage may become in the text.
        """
        self.citation_generator = citation_generator or CitationGenerator()
        self.snippet_max_length = snippet_max_length

    def build(
        self,
        retrieval_results: list[RetrievalResult],
        query: str = "",
    ) -> MCPToolResult:
        """Return the answer of a tool for the passages that were found.

        Args:
            retrieval_results: The passages the search returned, best first.
            query: The question the passages answer.

        Returns:
            The result, with the Markdown text and the citations.
        """
        if not retrieval_results:
            return self._build_empty(query)

        citations = self.citation_generator.generate(retrieval_results)
        markdown = self._build_markdown(retrieval_results, citations, query)

        return MCPToolResult(
            content=[{"type": "text", "text": markdown}],
            structured_content={
                "query": query,
                "citations": [citation.to_dict() for citation in citations],
                "result_count": len(retrieval_results),
            },
        )

    def _build_empty(self, query: str) -> MCPToolResult:
        """Return the answer of a search that found nothing.

        An empty list would leave the caller guessing, so the text says what
        happened and what to do about it.
        """
        markdown = "\n".join(
            [
                "## No results",
                "",
                f"No document matches the query **{query}**.",
                "",
                "The documents may not have been ingested yet, or the query may",
                "use words the documents do not contain. Try another wording, or",
                "run the ingestion script to add more documents.",
            ]
        )
        return MCPToolResult(
            content=[{"type": "text", "text": markdown}],
            structured_content={
                "query": query,
                "citations": [],
                "result_count": 0,
            },
        )

    def _build_markdown(
        self,
        retrieval_results: list[RetrievalResult],
        citations: list[Citation],
        query: str,
    ) -> str:
        """Return the passages as Markdown, with the citation markers."""
        lines: list[str] = ["## Results", ""]
        if query:
            lines.append(f'Found {len(retrieval_results)} passage(s) for "{query}":')
            lines.append("")

        for result, citation in zip(retrieval_results, citations):
            marker = self.citation_generator.format_marker(citation.index)
            lines.append(f"### {marker} {citation.source}")
            details = [f"score: {citation.score:.4f}"]
            if citation.page is not None:
                details.append(f"page: {citation.page}")
            lines.append(f"_{' | '.join(details)}_")
            lines.append("")
            snippet = self._truncate(result.text)
            if snippet:
                lines.append(snippet)
                lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## Sources")
        lines.append("")
        for citation in citations:
            marker = self.citation_generator.format_marker(citation.index)
            source = f"`{citation.source}`"
            if citation.page is not None:
                source += f" (p.{citation.page})"
            lines.append(f"- {marker} {source} - chunk `{citation.chunk_id}`")

        return "\n".join(lines)

    def _truncate(self, text: str) -> str:
        """Return the text of a passage, shortened when it is too long."""
        if not text:
            return ""

        cleaned = " ".join(text.split())
        if len(cleaned) <= self.snippet_max_length:
            return cleaned

        shortened = cleaned[: self.snippet_max_length].rsplit(" ", 1)[0]
        return f"{shortened}..."
