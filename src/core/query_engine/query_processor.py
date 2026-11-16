"""Query preparation.

A question arrives as a sentence. The search needs two things out of it: the
words that carry the meaning, and the filters that narrow the search down.
This module cuts the sentence into words and reads the filters out of it.

The words are cut with the same rules the index was built with, so a term
that was indexed can be found again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.query_engine.tokenizer import DEFAULT_MIN_TERM_LENGTH, tokenize
from core.trace.trace_context import TraceContext

# The words that carry no meaning of their own. A question is mostly made of
# them, and they would only dilute the search.
DEFAULT_STOPWORDS = frozenset(
    {
        # English
        "a", "about", "after", "all", "also", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "can", "could", "did", "do", "does",
        "doing", "done", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here",
        "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it",
        "its", "itself", "just", "me", "more", "most", "my", "no", "nor",
        "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
        "ours", "out", "over", "own", "same", "she", "should", "so", "some",
        "such", "than", "that", "the", "their", "theirs", "them", "then",
        "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "use", "used", "using", "very", "was", "we",
        "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "will", "with", "would", "you", "your", "yours",
        # Chinese
        "什么", "怎么", "如何", "哪些", "哪个", "为什么", "怎样", "可以",
        "需要", "是否", "以及", "并且", "但是", "因为", "所以", "如果",
        "这个", "那个", "这些", "那些", "我们", "你们", "他们", "它们",
        "一个", "一些", "一下", "请问", "麻烦", "告诉", "介绍", "说明",
    }
)

# The metadata fields a question may ask for, and the syntax to ask for them:
# `doc_type:pdf`.
DEFAULT_FILTER_KEYS = ("collection", "doc_type", "language", "source_path", "tags", "title")

DEFAULT_MAX_KEYWORDS = 32


@dataclass
class ProcessedQuery:
    """A question that is ready to be searched with."""

    original_query: str
    keywords: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the processed query as a plain dictionary."""
        return {
            "original_query": self.original_query,
            "keywords": list(self.keywords),
            "filters": dict(self.filters),
        }


class QueryProcessor:
    """Turn a question into the words and filters a search needs."""

    def __init__(
        self,
        stopwords: frozenset[str] | set[str] | None = None,
        min_term_length: int = DEFAULT_MIN_TERM_LENGTH,
        max_keywords: int = DEFAULT_MAX_KEYWORDS,
        filter_keys: tuple[str, ...] | None = None,
    ) -> None:
        """Set up the processor.

        Args:
            stopwords: The words that are left out of the keywords.
            min_term_length: How short a keyword may be to be kept.
            max_keywords: How many keywords are kept at most.
            filter_keys: The metadata fields a question may ask for.
        """
        if min_term_length < 1:
            raise ValueError(f"min_term_length must be at least 1, got {min_term_length}")
        if max_keywords < 1:
            raise ValueError(f"max_keywords must be at least 1, got {max_keywords}")

        self.stopwords = frozenset(DEFAULT_STOPWORDS if stopwords is None else stopwords)
        self.min_term_length = min_term_length
        self.max_keywords = max_keywords
        self.filter_keys = tuple(DEFAULT_FILTER_KEYS if filter_keys is None else filter_keys)
        self._filter_pattern = self._build_filter_pattern(self.filter_keys)

    def process(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> ProcessedQuery:
        """Return the words and filters of `query`.

        The filters that are handed in are kept, and the ones the question
        names are added to them.
        """
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("query cannot be empty or whitespace-only")

        parsed_filters, text = self.parse_filters(query, filters)
        keywords = self.extract_keywords(text)

        processed = ProcessedQuery(
            original_query=query,
            keywords=keywords,
            filters=parsed_filters,
        )

        if trace is not None:
            trace.record_stage(
                "query_processing",
                {
                    "keywords": list(keywords),
                    "keyword_count": len(keywords),
                    "filters": dict(parsed_filters),
                },
            )
        return processed

    def extract_keywords(self, text: str) -> list[str]:
        """Return the words of the text that carry meaning.

        The words are cut the way the index was built, so a term that was
        indexed can be found again. A word that is only a stopword is left
        out; if that would leave nothing at all, the stopwords are kept, so
        a question always has something to search for.
        """
        terms = tokenize(text, min_term_length=self.min_term_length)
        meaningful = [term for term in terms if term not in self.stopwords]
        if not meaningful:
            meaningful = terms
        return self._unique(meaningful)[: self.max_keywords]

    def parse_filters(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Return the filters of the question and the text that is left.

        A question may name a metadata field, as in `doc_type:pdf`. The part
        that names it is taken out of the text, so it does not end up among
        the keywords.
        """
        merged = dict(filters or {})
        if self._filter_pattern is None:
            return merged, query

        found: dict[str, str] = {}
        for match in self._filter_pattern.finditer(query):
            key = match.group("key").lower()
            value = match.group("value").strip().strip("\"'")
            if value:
                found[key] = value

        for key, value in found.items():
            merged.setdefault(key, value)

        text = self._filter_pattern.sub(" ", query)
        return merged, text

    @staticmethod
    def _build_filter_pattern(filter_keys: tuple[str, ...]) -> re.Pattern[str] | None:
        """Return the pattern that finds the filters in a question."""
        keys = [key for key in filter_keys if key]
        if not keys:
            return None
        alternatives = "|".join(re.escape(key) for key in keys)
        return re.compile(
            rf"(?P<key>{alternatives})\s*:\s*(?P<value>\"[^\"]*\"|'[^']*'|\S+)",
            re.IGNORECASE,
        )

    @staticmethod
    def _unique(terms: list[str]) -> list[str]:
        """Return the terms without the repeats, keeping the order."""
        seen: set[str] = set()
        unique: list[str] = []
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            unique.append(term)
        return unique
