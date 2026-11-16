"""Integration tests for the chunk refiner against a real LLM.

These tests make real API calls, so they only run when the configuration
names a provider that can be reached. Without credentials the whole module
skips, which keeps the default test run free of network traffic.

Run with:
    pytest tests/integration/test_chunk_refiner_llm.py -v -s -m llm
"""

from __future__ import annotations

import pytest

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner

pytestmark = [pytest.mark.integration, pytest.mark.llm]

NOISY_PDF_CHUNK = """\
────────────────────────────
Page 42 | Technical Documentation
────────────────────────────

Chapter 5: System Architecture

The   microservices   architecture  consists  of  several  key  components.

<!-- Internal note: update the diagram -->

The main   components   are:
- Gateway   Service
- Authentication  Service
- Data   Processing   Service

────────────────────────────
Footer: Confidential
────────────────────────────
"""

EXPECTED_KEYWORDS = [
    "Chapter 5",
    "System Architecture",
    "microservices",
    "Gateway Service",
    "Authentication Service",
]


@pytest.fixture
def refiner(llm_settings: Settings) -> ChunkRefiner:
    """Return a refiner that talks to the configured provider."""
    llm_settings.ingestion.chunk_refiner.use_llm = True
    return ChunkRefiner(llm_settings)


@pytest.fixture
def chunk() -> Chunk:
    return Chunk(
        id="chunk_test_001",
        text=NOISY_PDF_CHUNK,
        metadata={"source_path": "technical_doc.pdf", "chunk_index": 0},
        source_ref="doc_technical",
    )


class TestRealLlmRefinement:
    def test_the_model_rewrites_the_chunk(self, refiner: ChunkRefiner, chunk: Chunk):
        refined = refiner.transform([chunk])

        assert len(refined) == 1
        assert refined[0].text.strip()
        assert refined[0].metadata["refined_by"] == "llm"

    def test_the_page_marks_are_gone(self, refiner: ChunkRefiner, chunk: Chunk):
        refined = refiner.transform([chunk])

        text = refined[0].text
        assert "────" not in text
        assert "<!--" not in text
        assert "Page 42" not in text
        assert "Confidential" not in text

    def test_the_content_survived(self, refiner: ChunkRefiner, chunk: Chunk):
        refined = refiner.transform([chunk])

        text = refined[0].text
        for keyword in EXPECTED_KEYWORDS:
            assert keyword in text, f"'{keyword}' is missing from the refined text"

    def test_the_chunk_keeps_its_identity(self, refiner: ChunkRefiner, chunk: Chunk):
        refined = refiner.transform([chunk])

        assert refined[0].id == chunk.id
        assert refined[0].source_ref == chunk.source_ref
        assert refined[0].metadata["source_path"] == "technical_doc.pdf"

    def test_the_trace_records_the_model(self, refiner: ChunkRefiner, chunk: Chunk):
        trace = TraceContext()

        refiner.transform([chunk], trace)

        data = trace.get_stage_data("chunk_refiner")
        assert data is not None
        assert data["total_chunks"] == 1
        assert data["refined_by_llm"] == 1
        assert data["failed"] == 0

    def test_every_chunk_of_a_batch_comes_back(self, refiner: ChunkRefiner, chunk: Chunk):
        chunks = [
            chunk,
            Chunk(
                id="chunk_test_002",
                text="Another  chunk   with   noise\n\n\n\nand issues.",
                metadata={"source_path": "other.pdf", "chunk_index": 1},
                source_ref="doc_technical",
            ),
        ]

        refined = refiner.transform(chunks)

        assert len(refined) == 2
        assert all(item.metadata["refined_by"] == "llm" for item in refined)


class TestFallbackAgainstARealProvider:
    def test_an_unreachable_model_falls_back_to_the_rules(
        self, llm_settings: Settings, chunk: Chunk
    ):
        llm_settings.ingestion.chunk_refiner.use_llm = True
        llm_settings.llm.model = "a-model-that-does-not-exist"

        refined = ChunkRefiner(llm_settings).transform([chunk])

        assert len(refined) == 1
        assert refined[0].metadata["refined_by"] == "rule"
        assert refined[0].metadata.get("refine_fallback_reason") == "llm_failed"
        assert "────" not in refined[0].text
        assert refined[0].text.strip()
