"""Provider selection for the evaluator layer.

The provider name comes from `config/settings.yaml`; the factory turns it into
the matching `BaseEvaluator` implementation. Evaluation off, or the `none`
provider, falls back to `NoneEvaluator` so a pipeline never depends on an
evaluation framework being installed.
"""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError, NoneEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator

ProviderBuilder = Callable[[Settings], BaseEvaluator]


class EvaluatorFactory:
    """Registry mapping a configured provider name to its implementation."""

    _builders: dict[str, ProviderBuilder] = {
        "custom": lambda settings: CustomEvaluator(settings.evaluation),
    }

    @classmethod
    def register(cls, provider: str, builder: ProviderBuilder) -> None:
        """Register `builder` as the implementation of `provider`."""
        name = provider.strip().lower()
        if not name:
            raise ValueError("Cannot register an evaluation provider with an empty name")
        cls._builders[name] = builder

    @classmethod
    def register_ragas(cls) -> None:
        """Register the ragas provider.

        The import happens here and not at the top of the module, so that a
        project without ragas installed can still use the other evaluators.
        """
        from observability.evaluation.ragas_evaluator import RagasEvaluator

        cls.register(
            "ragas",
            lambda settings: RagasEvaluator(settings.evaluation, settings=settings),
        )

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider registration; unknown names are ignored."""
        cls._builders.pop(provider.strip().lower(), None)

    @classmethod
    def registered_providers(cls) -> list[str]:
        """Return the registered provider names, sorted."""
        return sorted(cls._builders)

    @classmethod
    def create(cls, settings: Settings) -> BaseEvaluator:
        """Build the evaluator named by `settings.evaluation.provider`."""
        provider = (settings.evaluation.provider or "").strip().lower()
        if not provider:
            raise EvaluatorError(
                "evaluation.provider is not configured - set it in config/settings.yaml"
            )

        if provider == "none":
            return NoneEvaluator(settings.evaluation)

        builder = cls._builders.get(provider)
        if builder is None:
            available = ", ".join(cls.registered_providers()) or "(none)"
            raise EvaluatorError(
                f"Unknown evaluation provider '{provider}'. Registered providers: {available}"
            )

        if not settings.evaluation.enabled:
            return NoneEvaluator(settings.evaluation)

        evaluator = builder(settings)
        if not isinstance(evaluator, BaseEvaluator):
            raise EvaluatorError(
                f"Provider '{provider}' returned {type(evaluator).__name__}, "
                "which is not a BaseEvaluator"
            )
        return evaluator


EvaluatorFactory.register_ragas()
