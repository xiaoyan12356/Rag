"""BM25 index.

The sparse encoder counts the terms of every chunk. This module turns those
counts into something searchable: for every term it keeps the list of chunks
that hold it, together with how rare the term is across the corpus.

The index lives on disk as one JSON file per collection. The file is written
through a temporary file, so a run that stops halfway leaves the previous
index intact.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.settings import resolve_path
from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = "data/db/bm25"
DEFAULT_COLLECTION = "default"
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25Indexer:
    """Build, store and query a BM25 index.

    The indexer takes the term statistics of the sparse encoder and answers
    which chunks match a set of query terms best.

    The rarer a term is, the more a chunk that holds it is worth. That rarity
    is the IDF: `log((N - df + 0.5) / (df + 0.5))`, where `N` is the number
    of chunks in the corpus and `df` the number of them that hold the term.
    """

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        """Set up an indexer that writes to `index_dir`.

        Args:
            index_dir: Where the index files are kept.
            k1: How fast a repeated term stops paying off.
            b: How much a long chunk is held back against a short one.
        """
        if k1 <= 0:
            raise ValueError(f"k1 must be positive, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be between 0 and 1, got {b}")

        self.index_dir = resolve_path(index_dir)
        self.k1 = k1
        self.b = b

        self._index: dict[str, dict[str, Any]] = {}
        self._documents: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, Any] = {}

    def build(
        self,
        term_stats: list[dict[str, Any]],
        collection: str = DEFAULT_COLLECTION,
        trace: TraceContext | None = None,
    ) -> None:
        """Build the index from the term statistics of a corpus.

        The corpus is counted, every term gets its IDF, and the result is
        written to disk under `collection`.

        Args:
            term_stats: What the sparse encoder returned for every chunk.
            collection: The name the index is filed under.
            trace: Optional trace to record the work in.
        """
        if not term_stats:
            raise ValueError("Cannot build an index from an empty corpus")
        _validate_term_stats(term_stats)

        self._documents = {
            stat["chunk_id"]: {
                "term_frequencies": dict(stat["term_frequencies"]),
                "doc_length": stat["doc_length"],
            }
            for stat in term_stats
        }
        self._recalculate(collection)
        self._save(collection)

        if trace is not None:
            trace.record_stage(
                "bm25_indexer",
                {
                    "collection": collection,
                    "num_docs": self._metadata["num_docs"],
                    "total_terms": self._metadata["total_terms"],
                    "avg_doc_length": self._metadata["avg_doc_length"],
                },
            )

    def load(
        self,
        collection: str = DEFAULT_COLLECTION,
        trace: TraceContext | None = None,
    ) -> bool:
        """Read the index of `collection` from disk.

        Returns whether an index was found. A file that cannot be read is
        reported as an error, because a half-read index would answer
        questions wrongly.
        """
        index_path = self._get_index_path(collection)
        if not index_path.exists():
            return False

        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"The index at {index_path} cannot be read: {exc}") from exc

        if not isinstance(data, dict) or "metadata" not in data or "index" not in data:
            raise ValueError(f"The index at {index_path} does not have the expected shape")

        self._metadata = data["metadata"]
        self._index = data["index"]
        self._documents = data.get("documents", {})

        if trace is not None:
            trace.record_stage(
                "bm25_indexer",
                {
                    "collection": collection,
                    "num_docs": self._metadata.get("num_docs", 0),
                    "total_terms": self._metadata.get("total_terms", 0),
                    "loaded": True,
                },
            )
        return True

    def save(self, collection: str = DEFAULT_COLLECTION) -> None:
        """Write the index of `collection` to disk."""
        self._save(collection)

    def query(
        self,
        query_terms: list[str],
        top_k: int = 10,
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return the chunks that match `query_terms` best.

        Every chunk that holds at least one of the terms is scored with the
        BM25 formula and the best `top_k` are returned, highest score first.
        """
        if not self._index:
            raise ValueError("The index is empty - build or load it first")
        if not query_terms:
            raise ValueError("query_terms cannot be empty")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        scores: dict[str, float] = {}
        for term in query_terms:
            term_data = self._index.get(term.lower())
            if term_data is None:
                continue
            for posting in term_data["postings"]:
                chunk_id = posting["chunk_id"]
                score = self._score(
                    tf=posting["tf"],
                    doc_length=posting["doc_length"],
                    avg_doc_length=self._metadata.get("avg_doc_length", 0.0),
                    idf=term_data["idf"],
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        results = [{"chunk_id": chunk_id, "score": score} for chunk_id, score in scores.items()]
        results.sort(key=lambda item: (-item["score"], item["chunk_id"]))

        if trace is not None:
            trace.record_stage(
                "bm25_query",
                {
                    "query_terms": list(query_terms),
                    "matched_chunks": len(results),
                    "returned": min(top_k, len(results)),
                },
            )
        return results[:top_k]

    def rebuild(
        self,
        term_stats: list[dict[str, Any]],
        collection: str = DEFAULT_COLLECTION,
        trace: TraceContext | None = None,
    ) -> None:
        """Throw the index away and build it again from `term_stats`."""
        self._index = {}
        self._documents = {}
        self._metadata = {}
        self.build(term_stats, collection, trace)

    def add_documents(
        self,
        term_stats: list[dict[str, Any]],
        collection: str = DEFAULT_COLLECTION,
        doc_id: str | None = None,
        trace: TraceContext | None = None,
    ) -> None:
        """Add chunks to the index without throwing the old ones away.

        When `doc_id` is given, the chunks of that document are replaced by
        the new ones, so ingesting the same document twice does not leave
        two copies of every chunk behind.
        """
        if not term_stats:
            return
        _validate_term_stats(term_stats)

        if not self._index and not self.load(collection):
            self._documents = {}

        if doc_id:
            self._drop_document(doc_id)

        for stat in term_stats:
            self._documents[stat["chunk_id"]] = {
                "term_frequencies": dict(stat["term_frequencies"]),
                "doc_length": stat["doc_length"],
            }

        self._recalculate(collection)
        self._save(collection)

        if trace is not None:
            trace.record_stage(
                "bm25_indexer",
                {
                    "collection": collection,
                    "added_chunks": len(term_stats),
                    "num_docs": self._metadata["num_docs"],
                    "total_terms": self._metadata["total_terms"],
                    "incremental": True,
                },
            )

    def remove_document(
        self,
        doc_id: str,
        collection: str = DEFAULT_COLLECTION,
        chunk_ids: Sequence[str] | None = None,
    ) -> bool:
        """Remove the chunks of a document from the index.

        The chunks are found by their id prefixing `doc_id`, or taken from
        `chunk_ids` when the caller knows exactly which chunks to drop. The
        explicit list is what the document manager uses: the ids of a chunk
        carry a hash of its text, so they cannot be derived from the id of
        the document alone.

        Args:
            doc_id: The id of the document whose chunks are removed.
            collection: The collection the index is filed under.
            chunk_ids: The ids of the chunks to remove, when they are known.

        Returns:
            Whether anything was removed.
        """
        if not self._index and not self.load(collection):
            return False

        removed = False
        if chunk_ids is not None:
            removed = self._drop_chunks(chunk_ids)
        removed = self._drop_document(doc_id) or removed
        if not removed:
            return False

        self._recalculate(collection)
        self._save(collection)
        return True

    def get_metadata(self) -> dict[str, Any]:
        """Return what the index knows about the corpus."""
        return dict(self._metadata)

    def get_document_count(self) -> int:
        """Return the number of chunks in the index."""
        return len(self._documents)

    def _drop_document(self, doc_id: str) -> bool:
        """Remove the chunks whose id belongs to `doc_id`."""
        stale = [
            chunk_id
            for chunk_id in self._documents
            if chunk_id == doc_id or chunk_id.startswith(f"{doc_id}_")
        ]
        for chunk_id in stale:
            del self._documents[chunk_id]
        return bool(stale)

    def _drop_chunks(self, chunk_ids: Sequence[str]) -> bool:
        """Remove the chunks with exactly these ids."""
        removed = False
        for chunk_id in chunk_ids:
            if self._documents.pop(str(chunk_id), None) is not None:
                removed = True
        return removed

    def _recalculate(self, collection: str) -> None:
        """Recount the corpus and rebuild the inverted index from the documents."""
        num_docs = len(self._documents)
        total_length = sum(doc["doc_length"] for doc in self._documents.values())
        avg_doc_length = total_length / num_docs if num_docs else 0.0

        index: dict[str, dict[str, Any]] = {}
        for chunk_id, doc in self._documents.items():
            for term, tf in doc["term_frequencies"].items():
                if tf <= 0:
                    continue
                entry = index.setdefault(term, {"idf": 0.0, "df": 0, "postings": []})
                entry["postings"].append(
                    {
                        "chunk_id": chunk_id,
                        "tf": tf,
                        "doc_length": doc["doc_length"],
                    }
                )

        for term, entry in index.items():
            entry["df"] = len(entry["postings"])
            entry["idf"] = self._calculate_idf(num_docs, entry["df"])

        self._index = index
        self._metadata = {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "total_terms": len(index),
            "collection": collection,
        }

    def _calculate_idf(self, num_docs: int, df: int) -> float:
        """Return how rare a term is: `log((N - df + 0.5) / (df + 0.5))`."""
        return math.log((num_docs - df + 0.5) / (df + 0.5))

    def _score(
        self,
        tf: int,
        doc_length: int,
        avg_doc_length: float,
        idf: float,
    ) -> float:
        """Return what one term contributes to the score of a chunk."""
        if avg_doc_length <= 0:
            avg_doc_length = 1.0
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (
            1 - self.b + self.b * (doc_length / avg_doc_length)
        )
        if denominator <= 0:
            return 0.0
        return idf * (numerator / denominator)

    def _get_index_path(self, collection: str) -> Path:
        """Return where the index of `collection` is kept."""
        return self.index_dir / f"{collection}_bm25.json"

    def _save(self, collection: str) -> None:
        """Write the index to disk through a temporary file."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._get_index_path(collection)
        temp_path = index_path.with_suffix(".tmp")

        data = {
            "metadata": self._metadata,
            "index": self._index,
            "documents": self._documents,
        }
        try:
            temp_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temp_path.replace(index_path)
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            raise OSError(f"Could not write the index at {index_path}: {exc}") from exc


def _validate_term_stats(term_stats: list[dict[str, Any]]) -> None:
    """Raise `ValueError` unless every item carries what the index needs."""
    for position, stat in enumerate(term_stats):
        if not isinstance(stat, dict):
            raise ValueError(
                f"term_stats[{position}] must be a mapping, got {type(stat).__name__}"
            )
        for field in ("chunk_id", "term_frequencies", "doc_length"):
            if field not in stat:
                raise ValueError(f"term_stats[{position}] is missing '{field}'")
        if not isinstance(stat["term_frequencies"], dict):
            raise ValueError(
                f"term_stats[{position}]['term_frequencies'] must be a mapping, "
                f"got {type(stat['term_frequencies']).__name__}"
            )
        if not isinstance(stat["doc_length"], int) or stat["doc_length"] < 0:
            raise ValueError(
                f"term_stats[{position}]['doc_length'] must be a non-negative "
                f"integer, got {stat['doc_length']!r}"
            )
