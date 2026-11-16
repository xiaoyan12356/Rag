"""Word cutting, shared by the index and the question.

A word only helps if the index and the question cut it the same way: the
ingestion side counts the words of a chunk with these rules, and the query
side counts the words of a question with them, so a term that was indexed can
still be looked up.
"""

from __future__ import annotations

import logging
import re

import jieba

# jieba reports the dictionary it builds on the first call, which is noise in
# the logs of a pipeline that is only counting words.
jieba.setLogLevel(logging.WARNING)

# A token that holds nothing but punctuation and spaces is not a term.
_PUNCTUATION_ONLY = re.compile(r"[\s\W]+", re.UNICODE)

DEFAULT_MIN_TERM_LENGTH = 2


def tokenize(
    text: str,
    *,
    min_term_length: int = DEFAULT_MIN_TERM_LENGTH,
    lowercase: bool = True,
) -> list[str]:
    """Return the terms of the text, in the order they occur.

    Args:
        text: The text to cut into terms.
        min_term_length: How short a term may be to be kept.
        lowercase: Whether the terms are lowercased.
    """
    if not text or not text.strip():
        return []

    terms: list[str] = []
    for token in jieba.lcut(text):
        token = token.strip()
        if not token or _PUNCTUATION_ONLY.fullmatch(token):
            continue
        if lowercase:
            token = token.lower()
        if len(token) >= min_term_length:
            terms.append(token)
    return terms
