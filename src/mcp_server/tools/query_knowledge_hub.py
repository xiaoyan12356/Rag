"""The main search tool.

The tool runs the search the command line runs: the question is prepared,
both routes go looking, the two rankings are merged, the passages are
reranked, and everything that comes back carries the citation that says
where it came from.
"""

from __future__ import annotations

import logging
from typing import Any

from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.reranker import Reranker
from core.response.multimodal_assembler import MultimodalAssembler
from core.response.response_builder import MCPToolResult, ResponseBuilder
from core.settings import Settings
from core.trace.trace_collector import collect_trace
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from ingestion.storage.image_storage import ImageStorage
from mcp_server.tools import ToolArgumentError

logger = logging.getLogger(__name__)

TOOL_NAME = "query_knowledge_hub"

DESCRIPTION = (
    "Search the knowledge hub. The question is answered with hybrid retrieval "
    "(dense vectors and BM25, merged with RRF) and the passages are reranked. "
    "Every passage comes with the citation of the file, the page and the chunk "
    "it came from."
)

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The question to search for.",
        },
        "top_k": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "How many passages to return. The configured number is used when "
                "it is left out."
            ),
        },
        "collection": {
            "type": "string",
            "description": (
                "The collection to search. The configured one is used when it "
                "is left out."
            ),
        },
    },
    "required": ["query"],
}

# The search engine holds a client and a store, so it is built once per
# collection instead of once per question.
_ENGINES: dict[tuple[str, str], HybridSearch] = {}

# The assembler opens the image index, so it is built once as well.
_ASSEMBLERS: dict[tuple[str, str], MultimodalAssembler] = {}


def query_knowledge_hub(
    query: str,
    top_k: int | None = None,
    collection: str | None = None,
    *,
    settings: Settings | None = None,
    search: HybridSearch | None = None,
    reranker: Reranker | None = None,
    assembler: MultimodalAssembler | None = None,
) -> MCPToolResult:
    """Answer a question with the passages of the knowledge hub.

    Args:
        query: The question to search for.
        top_k: How many passages to return.
        collection: The collection to search.
        settings: The configuration of the providers.
        search: The search to run; one is built from the settings when it is
            left out.
        reranker: The reranker that orders the passages.
        assembler: The one that collects the images of the passages.

    Returns:
        The answer, with the Markdown text and the citations.

    Raises:
        ToolArgumentError: The arguments do not fit the schema.
    """
    _check_query(query)
    _check_top_k(top_k)
    _check_collection(collection)

    if search is None:
        if settings is None:
            raise RuntimeError("The tool needs a configuration to search with")
        search = _search_engine(settings, collection)

    trace = TraceContext(trace_type="query")
    trace.metadata["query"] = query
    if collection is not None:
        trace.metadata["collection"] = collection
    results = search.search(query, top_k=top_k, trace=trace)
    results = _rerank(settings, query, results, reranker, trace)

    if settings is not None:
        collect_trace(trace, settings)

    response = ResponseBuilder().build(results, query)
    images = (assembler or _assembler(settings)).assemble(results)
    if images:
        response.content.extend(images)
    return response


def handle(arguments: dict[str, Any], settings: Settings | None = None) -> MCPToolResult:
    """Run the tool with the arguments a client sent.

    Args:
        arguments: The arguments of the call.
        settings: The configuration of the providers.
    """
    return query_knowledge_hub(
        query=arguments.get("query"),
        top_k=arguments.get("top_k"),
        collection=arguments.get("collection"),
        settings=settings,
    )


def _check_query(query: Any) -> None:
    """Refuse a question that is not a text."""
    if not isinstance(query, str) or not query.strip():
        raise ToolArgumentError("The 'query' argument must be a non-empty string")


def _check_top_k(top_k: Any) -> None:
    """Refuse a number of passages that cannot be asked for."""
    if top_k is None:
        return
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ToolArgumentError("The 'top_k' argument must be a positive integer")


def _check_collection(collection: Any) -> None:
    """Refuse a collection that cannot be named."""
    if collection is None:
        return
    if not isinstance(collection, str) or not collection.strip():
        raise ToolArgumentError("The 'collection' argument must be a non-empty string")


def _search_engine(settings: Settings, collection: str | None) -> HybridSearch:
    """Return the search of a collection, building it the first time it is asked for."""
    name = collection or settings.vector_store.collection_name
    key = (settings.vector_store.persist_directory or "", name)
    engine = _ENGINES.get(key)
    if engine is None:
        engine = HybridSearch(settings, collection=name)
        _ENGINES[key] = engine
    return engine


def _assembler(settings: Settings | None) -> MultimodalAssembler:
    """Return the assembler that finds the images of the passages."""
    if settings is None:
        return MultimodalAssembler()

    key = (settings.ingestion.image_index_db, settings.ingestion.images_root)
    assembler = _ASSEMBLERS.get(key)
    if assembler is None:
        try:
            storage = ImageStorage(
                db_path=settings.ingestion.image_index_db,
                images_root=settings.ingestion.images_root,
            )
        except Exception as exc:
            logger.warning("The images cannot be looked up: %s", exc)
            return MultimodalAssembler()
        assembler = MultimodalAssembler(image_storage=storage)
        _ASSEMBLERS[key] = assembler
    return assembler


def _rerank(
    settings: Settings | None,
    query: str,
    results: list[RetrievalResult],
    reranker: Reranker | None,
    trace: TraceContext | None = None,
) -> list[RetrievalResult]:
    """Return the passages in the order the reranker put them in."""
    if not results:
        return results

    if reranker is None:
        if settings is None:
            return results
        try:
            reranker = Reranker(settings)
        except Exception as exc:
            logger.warning(
                "The reranker could not be built, the order of the search is kept: %s", exc
            )
            return results

    return reranker.rerank(query, results, trace=trace)
