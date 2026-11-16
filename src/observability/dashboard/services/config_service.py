"""Configuration service for the dashboard.

The overview page shows what the system is made of: which provider each
component uses and how it is tuned. This service reads the `Settings` and
turns them into the cards the page draws, so the page itself does not have
to know the shape of the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.settings import DEFAULT_SETTINGS_PATH, Settings, load_settings


@dataclass
class ComponentInfo:
    """One component of the system, as the overview page shows it."""

    name: str
    provider: str
    model: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_rows(self) -> list[tuple[str, str]]:
        """Return the details as label-value pairs the page can render."""
        return [(str(key), _format_value(value)) for key, value in self.details.items()]


class ConfigService:
    """Read the configuration and describe the components it names.

    Args:
        settings_path: The configuration file to read; the default one is
            used when this is left out.
    """

    def __init__(self, settings_path: str | Path | None = None) -> None:
        """Set up the service; the file is read when it is first needed."""
        self._settings_path = settings_path
        self._settings: Settings | None = None

    @property
    def settings(self) -> Settings:
        """Return the configuration, reading the file the first time."""
        if self._settings is None:
            self._settings = load_settings(self._settings_path or DEFAULT_SETTINGS_PATH)
        return self._settings

    def reload(self) -> None:
        """Read the configuration file again, so a change is picked up."""
        self._settings = None

    def get_component_cards(self) -> list[ComponentInfo]:
        """Return one card per configured component.

        The components that are switched off are left out, because a card
        for something that is not running says nothing useful.
        """
        settings = self.settings
        cards = [
            ComponentInfo(
                name="LLM",
                provider=settings.llm.provider or "not configured",
                model=settings.llm.model or "-",
                details={
                    "temperature": settings.llm.temperature,
                    "max_tokens": settings.llm.max_tokens,
                },
            ),
            ComponentInfo(
                name="Embedding",
                provider=settings.embedding.provider or "not configured",
                model=settings.embedding.model or "-",
                details={"dimensions": settings.embedding.dimensions},
            ),
            ComponentInfo(
                name="Vector Store",
                provider=settings.vector_store.provider or "not configured",
                model=settings.vector_store.collection_name or "-",
                details={
                    "persist_directory": settings.vector_store.persist_directory or "-",
                },
            ),
            ComponentInfo(
                name="Retrieval",
                provider="hybrid",
                model="dense + sparse + RRF",
                details={
                    "dense_top_k": settings.retrieval.dense_top_k,
                    "sparse_top_k": settings.retrieval.sparse_top_k,
                    "fusion_top_k": settings.retrieval.fusion_top_k,
                    "rrf_k": settings.retrieval.rrf_k,
                },
            ),
            ComponentInfo(
                name="Ingestion",
                provider=settings.ingestion.splitter or "recursive",
                model="-",
                details={
                    "chunk_size": settings.ingestion.chunk_size,
                    "chunk_overlap": settings.ingestion.chunk_overlap,
                    "batch_size": settings.ingestion.batch_size,
                },
            ),
            ComponentInfo(
                name="Observability",
                provider=settings.observability.log_level or "INFO",
                model=settings.observability.trace_file or "-",
                details={
                    "trace_enabled": settings.observability.trace_enabled,
                    "structured_logging": settings.observability.structured_logging,
                },
            ),
        ]

        if settings.rerank.enabled:
            cards.append(
                ComponentInfo(
                    name="Reranker",
                    provider=settings.rerank.provider or "not configured",
                    model=settings.rerank.model or "-",
                    details={"top_k": settings.rerank.top_k},
                )
            )

        if settings.vision_llm.enabled:
            cards.append(
                ComponentInfo(
                    name="Vision LLM",
                    provider=settings.vision_llm.provider or "not configured",
                    model=settings.vision_llm.model or "-",
                    details={"max_image_size": settings.vision_llm.max_image_size},
                )
            )

        if settings.evaluation.enabled:
            cards.append(
                ComponentInfo(
                    name="Evaluation",
                    provider=settings.evaluation.provider or "not configured",
                    model=", ".join(settings.evaluation.metrics) or "-",
                    details={"backends": ", ".join(settings.evaluation.backends) or "-"},
                )
            )

        return cards

    def get_summary_rows(self) -> list[tuple[str, str]]:
        """Return the settings that describe the dashboard itself."""
        dashboard = self.settings.dashboard
        return [
            ("enabled", _format_value(dashboard.enabled)),
            ("port", str(dashboard.port)),
            ("traces_dir", dashboard.traces_dir or "-"),
            ("auto_refresh", _format_value(dashboard.auto_refresh)),
            ("refresh_interval", str(dashboard.refresh_interval)),
        ]


def _format_value(value: Any) -> str:
    """Return the value as the short text the page shows."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "-"
    text = str(value)
    return text if text.strip() else "-"
