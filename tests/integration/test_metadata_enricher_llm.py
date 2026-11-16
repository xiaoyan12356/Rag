"""Integration tests for the metadata enricher against a real LLM.

These tests make real API calls, so they only run when the configuration
names a provider that can be reached. Without credentials the whole module
skips, which keeps the default test run free of network traffic.

Run with:
    pytest tests/integration/test_metadata_enricher_llm.py -v -s -m llm
"""

from __future__ import annotations

import pytest

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher

pytestmark = [pytest.mark.integration, pytest.mark.llm]

CHUNK_TEXT = """\
# Vector upsert idempotency

Every chunk is written to the vector store under an id derived from its
content, so ingesting the same document twice does not create a second copy
of a chunk. The id changes when the content changes.
"""


@pytest.fixture
def enricher(llm_settings: Settings) -> MetadataEnricher:
    """Return an enricher that talks to the configured provider."""
    llm_settings.ingestion.metadata_enricher.use_llm = True
    return MetadataEnricher(llm_settings)


@pytest.fixture
def chunk() -> Chunk:
    return Chunk(
        id="chunk_test_001",
        text=CHUNK_TEXT,
        metadata={"source_path": "technical_doc.pdf", "chunk_index": 0},
        source_ref="doc_technical",
    )


class TestRealLlmEnrichment:
    def test_the_model_describes_the_chunk(self, enricher: MetadataEnricher, chunk: Chunk):
        enriched = enricher.transform([chunk])

        assert len(enriched) == 1
        assert enriched[0].metadata["title"]
        assert enriched[0].metadata["summary"]
        assert enriched[0].metadata["tags"]

    def test_the_metadata_says_the_model_did_it(self, enricher: MetadataEnricher, chunk: Chunk):
        enriched = enricher.transform([chunk])

        assert enriched[0].metadata["enriched_by"] == "llm"

    def test_the_chunk_keeps_its_identity(self, enricher: MetadataEnricher, chunk: Chunk):
        enriched = enricher.transform([chunk])

        assert enriched[0].id == chunk.id
        assert enriched[0].source_ref == chunk.source_ref
        assert enriched[0].text == chunk.text
        assert enriched[0].metadata["source_path"] == "technical_doc.pdf"

    def test_the_trace_records_the_model(self, enricher: MetadataEnricher, chunk: Chunk):
        trace = TraceContext()

        enricher.transform([chunk], trace)

        data = trace.get_stage_data("metadata_enricher")
        assert data is not None
        assert data["total_chunks"] == 1
        assert data["enriched_by_llm"] == 1
        assert data["failed"] == 0

    def test_every_chunk_of_a_batch_comes_back(self, enricher: MetadataEnricher, chunk: Chunk):
        chunks = [
            chunk,
            Chunk(
                id="chunk_test_002",
                text="The BM25 index is built from the term statistics of every chunk.",
                metadata={"source_path": "technical_doc.pdf", "chunk_index": 1},
                source_ref="doc_technical",
            ),
        ]

        enriched = enricher.transform(chunks)

        assert len(enriched) == 2
        assert all(item.metadata["enriched_by"] == "llm" for item in enriched)


class TestFallbackAgainstARealProvider:
    def test_an_unreachable_model_falls_back_to_the_rules(
        self, llm_settings: Settings, chunk: Chunk
    ):
        llm_settings.ingestion.metadata_enricher.use_llm = True
        llm_settings.llm.model = "a-model-that-does-not-exist"

        enriched = MetadataEnricher(llm_settings).transform([chunk])

        assert len(enriched) == 1
        assert enriched[0].metadata["enriched_by"] == "rule"
        assert enriched[0].metadata.get("enrich_fallback_reason") == "llm_failed"
        assert enriched[0].metadata["title"] == "Vector upsert idempotency"
        assert enriched[0].metadata["tags"]
