"""Strategy selection for the reranker layer.

The strategy name comes from `config/settings.yaml`; the factory turns it into
the matching `BaseReranker` implementation. Reranking off, or the `none`
strategy, falls back to `NoneReranker` so a pipeline never depends on a
reranker being installed.
"""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankerError

StrategyBuilder = Callable[[Settings], BaseReranker]


class RerankerFactory:
    """Registry mapping a configured strategy name to its implementation."""

    _builders: dict[str, StrategyBuilder] = {}

    @classmethod
    def register(cls, strategy: str, builder: StrategyBuilder) -> None:
        """Register `builder` as the implementation of `strategy`."""
        name = strategy.strip().lower()
        if not name:
            raise ValueError("Cannot register a reranker strategy with an empty name")
        cls._builders[name] = builder

    @classmethod
    def unregister(cls, strategy: str) -> None:
        """Remove a strategy registration; unknown names are ignored."""
        cls._builders.pop(strategy.strip().lower(), None)

    @classmethod
    def registered_strategies(cls) -> list[str]:
        """Return the registered strategy names, sorted."""
        return sorted(cls._builders)

    @classmethod
    def create(cls, settings: Settings) -> BaseReranker:
        """Build the reranker named by `settings.rerank.provider`."""
        provider = (settings.rerank.provider or "").strip().lower()
        if not provider:
            raise RerankerError(
                "rerank.provider is not configured - set it in config/settings.yaml"
            )

        if provider == "none":
            return NoneReranker(settings.rerank)

        builder = cls._builders.get(provider)
        if builder is None:
            available = ", ".join(cls.registered_strategies()) or "(none)"
            raise RerankerError(
                f"Unknown reranker strategy '{provider}'. Registered strategies: {available}"
            )

        if not settings.rerank.enabled:
            return NoneReranker(settings.rerank)

        reranker = builder(settings)
        if not isinstance(reranker, BaseReranker):
            raise RerankerError(
                f"Strategy '{provider}' returned {type(reranker).__name__}, "
                "which is not a BaseReranker"
            )
        return reranker
