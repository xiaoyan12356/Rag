"""Unit tests for the query processor.

The processor cuts a question into the words the search needs and reads the
filters out of it. The tests pin that the words are cut the way the index was
built, that the stopwords are left out, and that a filter named in the
question does not end up among the words.
"""

from __future__ import annotations

import pytest

from core.query_engine.query_processor import QueryProcessor
from core.trace.trace_context import TraceContext


class TestKeywords:
    def test_the_words_are_found(self):
        processor = QueryProcessor()

        keywords = processor.extract_keywords("how do I configure Azure OpenAI")

        assert "azure" in keywords
        assert "openai" in keywords
        assert "configure" in keywords

    def test_the_stopwords_are_left_out(self):
        processor = QueryProcessor()

        keywords = processor.extract_keywords("what is the configuration of azure")

        assert "the" not in keywords
        assert "what" not in keywords
        assert "is" not in keywords
        assert "azure" in keywords

    def test_a_word_that_is_only_a_stopword_is_kept_when_nothing_is_left(self):
        processor = QueryProcessor()

        keywords = processor.extract_keywords("what is it")

        assert keywords

    def test_the_same_word_is_only_kept_once(self):
        processor = QueryProcessor()

        keywords = processor.extract_keywords("azure azure openai")

        assert keywords.count("azure") == 1

    def test_the_words_keep_the_order_of_the_question(self):
        processor = QueryProcessor()

        keywords = processor.extract_keywords("first second third")

        assert keywords == ["first", "second", "third"]

    def test_the_words_are_lowercased(self):
        processor = QueryProcessor()

        assert "azure" in processor.extract_keywords("AZURE")

    def test_the_case_can_be_kept(self):
        processor = QueryProcessor()

        # The cut itself lowercases; a question that carries a name in caps
        # is still found, because the index was cut the same way.
        keywords = processor.extract_keywords("Azure")

        assert "azure" in keywords

    def test_the_chinese_words_are_cut(self):
        processor = QueryProcessor()

        keywords = processor.extract_keywords("如何配置Azure")

        assert keywords
        assert "azure" in keywords

    def test_the_number_of_words_is_capped(self):
        processor = QueryProcessor(max_keywords=2)

        keywords = processor.extract_keywords("alpha beta gamma delta")

        assert len(keywords) == 2

    def test_a_short_word_is_left_out(self):
        processor = QueryProcessor(min_term_length=3)

        keywords = processor.extract_keywords("a bb ccc")

        assert "bb" not in keywords
        assert "ccc" in keywords

    def test_the_words_of_nothing_are_empty(self):
        processor = QueryProcessor()

        assert processor.extract_keywords("") == []
        assert processor.extract_keywords("   ") == []


class TestProcess:
    def test_the_question_is_kept(self):
        processor = QueryProcessor()

        processed = processor.process("how to configure azure")

        assert processed.original_query == "how to configure azure"

    def test_the_words_are_filled_in(self):
        processor = QueryProcessor()

        processed = processor.process("how to configure azure")

        assert processed.keywords
        assert "azure" in processed.keywords

    def test_the_filters_are_a_dict(self):
        processor = QueryProcessor()

        processed = processor.process("azure")

        assert processed.filters == {}

    def test_a_question_that_is_empty_is_refused(self):
        processor = QueryProcessor()

        with pytest.raises(ValueError, match="empty"):
            processor.process("   ")

    def test_a_question_that_is_not_text_is_refused(self):
        processor = QueryProcessor()

        with pytest.raises(ValueError, match="string"):
            processor.process(None)

    def test_the_processed_query_can_become_a_dict(self):
        processor = QueryProcessor()

        data = processor.process("azure").to_dict()

        assert data["original_query"] == "azure"
        assert isinstance(data["keywords"], list)
        assert isinstance(data["filters"], dict)


class TestFilters:
    def test_a_filter_named_in_the_question_is_found(self):
        processor = QueryProcessor()

        processed = processor.process("azure configuration doc_type:pdf")

        assert processed.filters["doc_type"] == "pdf"

    def test_the_filter_is_not_among_the_words(self):
        processor = QueryProcessor()

        processed = processor.process("azure configuration doc_type:pdf")

        assert "pdf" not in processed.keywords
        assert "doc_type" not in processed.keywords

    def test_the_filters_that_are_handed_in_are_kept(self):
        processor = QueryProcessor()

        processed = processor.process("azure", filters={"collection": "kb"})

        assert processed.filters["collection"] == "kb"

    def test_a_filter_of_the_question_does_not_overwrite_one_that_was_handed_in(self):
        processor = QueryProcessor()

        processed = processor.process("azure doc_type:pdf", filters={"doc_type": "md"})

        assert processed.filters["doc_type"] == "md"

    def test_a_quoted_filter_is_unquoted(self):
        processor = QueryProcessor()

        processed = processor.process('azure title:"The Guide"')

        assert processed.filters["title"] == "The Guide"

    def test_a_filter_that_is_not_asked_for_is_not_read(self):
        processor = QueryProcessor()

        processed = processor.process("azure unknown_key:pdf")

        assert "unknown_key" not in processed.filters

    def test_the_filter_keys_can_be_chosen(self):
        processor = QueryProcessor(filter_keys=("collection",))

        processed = processor.process("azure doc_type:pdf")

        assert processed.filters == {}


class TestSetup:
    def test_a_minimum_length_of_zero_is_refused(self):
        with pytest.raises(ValueError, match="min_term_length"):
            QueryProcessor(min_term_length=0)

    def test_a_maximum_of_zero_words_is_refused(self):
        with pytest.raises(ValueError, match="max_keywords"):
            QueryProcessor(max_keywords=0)

    def test_the_stopwords_can_be_chosen(self):
        processor = QueryProcessor(stopwords={"azure"})

        keywords = processor.extract_keywords("azure openai")

        assert "azure" not in keywords
        assert "openai" in keywords


class TestTrace:
    def test_the_trace_records_what_happened(self):
        processor = QueryProcessor()
        trace = TraceContext()

        processor.process("how to configure azure", trace=trace)

        data = trace.get_stage_data("query_processing")
        assert data["keyword_count"] > 0
        assert "azure" in data["keywords"]
