"""Strategy selection for the splitter layer.

The strategy name comes from `config/settings.yaml`; the factory turns it into
the matching `BaseSplitter` implementation, so switching chunking behaviour is
a config edit rather than a code change.
"""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter, SplitterError

StrategyBuilder = Callable[[Settings], BaseSplitter]


class SplitterFactory:
    """Registry mapping a configured strategy name to its implementation."""

    _builders: dict[str, StrategyBuilder] = {}

    @classmethod
    def register(cls, strategy: str, builder: StrategyBuilder) -> None:
        """Register `builder` as the implementation of `strategy`."""
        name = strategy.strip().lower()
        if not name:
            raise ValueError("Cannot register a splitter strategy with an empty name")
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
    def create(cls, settings: Settings) -> BaseSplitter:
        """Build the splitter named by `settings.ingestion.splitter`."""
        strategy = (settings.ingestion.splitter or "").strip().lower()
        if not strategy:
            raise SplitterError(
                "ingestion.splitter is not configured - set it in config/settings.yaml"
            )

        builder = cls._builders.get(strategy)
        if builder is None:
            available = ", ".join(cls.registered_strategies()) or "(none)"
            raise SplitterError(
                f"Unknown splitter strategy '{strategy}'. Registered strategies: {available}"
            )

        splitter = builder(settings)
        if not isinstance(splitter, BaseSplitter):
            raise SplitterError(
                f"Strategy '{strategy}' returned {type(splitter).__name__}, "
                "which is not a BaseSplitter"
            )
        return splitter
