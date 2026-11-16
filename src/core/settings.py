"""Configuration loading and validation.

The project is configuration driven: every pluggable component is selected in
`config/settings.yaml`. `load_settings()` turns that file into a typed
`Settings` object and `validate_settings()` rejects a configuration that is
missing a field the runtime cannot default, so a misconfigured deployment
fails at startup instead of at the first API call.

A required field that the YAML file does not carry stays `None` instead of
falling back to a default: the point of validation is to tell the operator
which key is missing, not to silently paper over it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(relative: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute path.

    An absolute path is returned unchanged; anything else is resolved against
    the repository root so a config file can name a data directory without
    depending on the working directory.
    """
    path = Path(relative)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()

# Provider -> environment variable holding the credential. An empty `api_key`
# in the config file falls back to this variable.
API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "together": "TOGETHER_API_KEY",
}

# DEV_SPEC and the setup skill template spell a few keys differently; accept
# both so either source can generate a working configuration.
_KEY_ALIASES = {
    ("vector_store", "provider"): "backend",
    ("vector_store", "persist_directory"): "persist_path",
    ("retrieval", "dense_top_k"): "top_k_dense",
    ("retrieval", "sparse_top_k"): "top_k_sparse",
    ("retrieval", "fusion_top_k"): "top_k_final",
    ("rerank", "provider"): "backend",
    ("rerank", "top_k"): "top_m",
}

# Section -> fields that must carry a value for the runtime to start.
_REQUIRED_FIELDS = {
    "llm": ("provider", "model"),
    "embedding": ("provider", "model"),
    "vector_store": ("provider", "persist_directory"),
    "retrieval": ("dense_top_k", "sparse_top_k", "fusion_top_k"),
    "rerank": ("provider",),
    "evaluation": ("provider",),
    "observability": ("log_level",),
}


class SettingsError(Exception):
    """Raised when the configuration file is unusable or incomplete."""


@dataclass
class LLMSettings:
    provider: str | None = None
    model: str | None = None
    api_key: str = ""
    deployment_name: str = ""
    azure_endpoint: str = ""
    api_version: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class EmbeddingSettings:
    provider: str | None = None
    model: str | None = None
    dimensions: int = 1536
    api_key: str = ""
    deployment_name: str = ""
    azure_endpoint: str = ""
    api_version: str = ""
    base_url: str = ""


@dataclass
class VisionLLMSettings:
    enabled: bool = False
    provider: str = ""
    model: str = ""
    api_key: str = ""
    deployment_name: str = ""
    azure_endpoint: str = ""
    api_version: str = ""
    base_url: str = ""
    max_image_size: int = 2048


@dataclass
class VectorStoreSettings:
    provider: str | None = None
    persist_directory: str | None = None
    collection_name: str = "knowledge_hub"


@dataclass
class RetrievalSettings:
    dense_top_k: int | None = None
    sparse_top_k: int | None = None
    fusion_top_k: int | None = None
    rrf_k: int = 60


@dataclass
class RerankSettings:
    enabled: bool = False
    provider: str | None = None
    model: str = ""
    top_k: int = 5


@dataclass
class EvaluationSettings:
    enabled: bool = False
    provider: str | None = None
    backends: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)


@dataclass
class ObservabilitySettings:
    log_level: str | None = None
    trace_enabled: bool = True
    trace_file: str = "./logs/traces.jsonl"
    structured_logging: bool = True


@dataclass
class ChunkRefinerSettings:
    use_llm: bool = False


@dataclass
class MetadataEnricherSettings:
    use_llm: bool = False


@dataclass
class IngestionSettings:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter: str = "recursive"
    batch_size: int = 100
    bm25_index_dir: str = "data/db/bm25"
    images_root: str = "data/images"
    image_index_db: str = "data/db/image_index.db"
    history_db: str = "data/db/ingestion_history.db"
    chunk_refiner: ChunkRefinerSettings = field(default_factory=ChunkRefinerSettings)
    metadata_enricher: MetadataEnricherSettings = field(default_factory=MetadataEnricherSettings)


@dataclass
class DashboardSettings:
    enabled: bool = True
    port: int = 8501
    traces_dir: str = "./logs"
    auto_refresh: bool = True
    refresh_interval: int = 5


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    vision_llm: VisionLLMSettings = field(default_factory=VisionLLMSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    return value if isinstance(value, dict) else {}


def _value(raw: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    data = _section(raw, section)
    if key in data:
        return data[key]
    alias = _KEY_ALIASES.get((section, key))
    if alias is not None and alias in data:
        return data[alias]
    return default


def _resolve_api_key(provider: str | None, configured: str) -> str:
    if configured and configured.strip():
        return configured
    env_var = API_KEY_ENV_VARS.get((provider or "").lower())
    if env_var:
        return os.environ.get(env_var, "")
    return ""


def _optional_str(raw: dict[str, Any], section: str, key: str, default: str = "") -> str:
    value = _value(raw, section, key, default)
    return "" if value is None else str(value)


def _optional_int(raw: dict[str, Any], section: str, key: str, default: int) -> int:
    value = _value(raw, section, key, default)
    return default if value is None else int(value)


def _required_str(raw: dict[str, Any], section: str, key: str) -> str | None:
    value = _value(raw, section, key)
    return None if value is None else str(value)


def _required_int(raw: dict[str, Any], section: str, key: str) -> int | None:
    value = _value(raw, section, key)
    return None if value is None else int(value)


def _build_settings(raw: dict[str, Any]) -> Settings:
    llm_provider = _required_str(raw, "llm", "provider")
    llm = LLMSettings(
        provider=llm_provider,
        model=_required_str(raw, "llm", "model"),
        api_key=_resolve_api_key(llm_provider, _optional_str(raw, "llm", "api_key")),
        deployment_name=_optional_str(raw, "llm", "deployment_name"),
        azure_endpoint=_optional_str(raw, "llm", "azure_endpoint"),
        api_version=_optional_str(raw, "llm", "api_version"),
        base_url=_optional_str(raw, "llm", "base_url"),
        temperature=float(_value(raw, "llm", "temperature", 0.0) or 0.0),
        max_tokens=_optional_int(raw, "llm", "max_tokens", 4096),
    )

    embedding_provider = _required_str(raw, "embedding", "provider")
    embedding = EmbeddingSettings(
        provider=embedding_provider,
        model=_required_str(raw, "embedding", "model"),
        dimensions=_optional_int(raw, "embedding", "dimensions", 1536),
        api_key=_resolve_api_key(
            embedding_provider, _optional_str(raw, "embedding", "api_key")
        ),
        deployment_name=_optional_str(raw, "embedding", "deployment_name"),
        azure_endpoint=_optional_str(raw, "embedding", "azure_endpoint"),
        api_version=_optional_str(raw, "embedding", "api_version"),
        base_url=_optional_str(raw, "embedding", "base_url"),
    )

    vision_provider = _optional_str(raw, "vision_llm", "provider")
    vision_llm = VisionLLMSettings(
        enabled=bool(_value(raw, "vision_llm", "enabled", False)),
        provider=vision_provider,
        model=_optional_str(raw, "vision_llm", "model"),
        api_key=_resolve_api_key(
            vision_provider, _optional_str(raw, "vision_llm", "api_key")
        ),
        deployment_name=_optional_str(raw, "vision_llm", "deployment_name"),
        azure_endpoint=_optional_str(raw, "vision_llm", "azure_endpoint"),
        api_version=_optional_str(raw, "vision_llm", "api_version"),
        base_url=_optional_str(raw, "vision_llm", "base_url"),
        max_image_size=_optional_int(raw, "vision_llm", "max_image_size", 2048),
    )

    vector_store = VectorStoreSettings(
        provider=_required_str(raw, "vector_store", "provider"),
        persist_directory=_required_str(raw, "vector_store", "persist_directory"),
        collection_name=_optional_str(raw, "vector_store", "collection_name", "knowledge_hub"),
    )

    retrieval = RetrievalSettings(
        dense_top_k=_required_int(raw, "retrieval", "dense_top_k"),
        sparse_top_k=_required_int(raw, "retrieval", "sparse_top_k"),
        fusion_top_k=_required_int(raw, "retrieval", "fusion_top_k"),
        rrf_k=_optional_int(raw, "retrieval", "rrf_k", 60),
    )

    rerank = RerankSettings(
        enabled=bool(_value(raw, "rerank", "enabled", False)),
        provider=_required_str(raw, "rerank", "provider"),
        model=_optional_str(raw, "rerank", "model"),
        top_k=_optional_int(raw, "rerank", "top_k", 5),
    )

    evaluation_backends = _value(raw, "evaluation", "backends")
    if not isinstance(evaluation_backends, list):
        evaluation_backends = []
    evaluation_provider = _required_str(raw, "evaluation", "provider")
    if not evaluation_provider and evaluation_backends:
        evaluation_provider = str(evaluation_backends[0])
    evaluation = EvaluationSettings(
        enabled=bool(_value(raw, "evaluation", "enabled", False)),
        provider=evaluation_provider,
        backends=[str(item) for item in evaluation_backends] or (
            [evaluation_provider] if evaluation_provider else []
        ),
        metrics=[str(item) for item in (_value(raw, "evaluation", "metrics") or [])],
    )

    observability = ObservabilitySettings(
        log_level=_required_str(raw, "observability", "log_level"),
        trace_enabled=bool(_value(raw, "observability", "trace_enabled", True)),
        trace_file=_optional_str(raw, "observability", "trace_file", "./logs/traces.jsonl"),
        structured_logging=bool(_value(raw, "observability", "structured_logging", True)),
    )

    ingestion = IngestionSettings(
        chunk_size=_optional_int(raw, "ingestion", "chunk_size", 1000),
        chunk_overlap=_optional_int(raw, "ingestion", "chunk_overlap", 200),
        splitter=_optional_str(raw, "ingestion", "splitter", "recursive"),
        batch_size=_optional_int(raw, "ingestion", "batch_size", 100),
        bm25_index_dir=_optional_str(raw, "ingestion", "bm25_index_dir", "data/db/bm25"),
        images_root=_optional_str(raw, "ingestion", "images_root", "data/images"),
        image_index_db=_optional_str(raw, "ingestion", "image_index_db", "data/db/image_index.db"),
        history_db=_optional_str(raw, "ingestion", "history_db", "data/db/ingestion_history.db"),
        chunk_refiner=ChunkRefinerSettings(
            use_llm=bool(_value(raw, "ingestion", "chunk_refiner", {}).get("use_llm", False))
        ),
        metadata_enricher=MetadataEnricherSettings(
            use_llm=bool(_value(raw, "ingestion", "metadata_enricher", {}).get("use_llm", False))
        ),
    )

    dashboard = DashboardSettings(
        enabled=bool(_value(raw, "dashboard", "enabled", True)),
        port=_optional_int(raw, "dashboard", "port", 8501),
        traces_dir=_optional_str(raw, "dashboard", "traces_dir", "./logs"),
        auto_refresh=bool(_value(raw, "dashboard", "auto_refresh", True)),
        refresh_interval=_optional_int(raw, "dashboard", "refresh_interval", 5),
    )

    return Settings(
        llm=llm,
        embedding=embedding,
        vision_llm=vision_llm,
        vector_store=vector_store,
        retrieval=retrieval,
        rerank=rerank,
        evaluation=evaluation,
        observability=observability,
        ingestion=ingestion,
        dashboard=dashboard,
    )


def validate_settings(settings: Settings) -> None:
    """Raise `SettingsError` listing every required field that has no value."""
    missing: list[str] = []
    for section_name, field_names in _REQUIRED_FIELDS.items():
        section = getattr(settings, section_name, None)
        if section is None:
            missing.append(section_name)
            continue
        for field_name in field_names:
            value = getattr(section, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(f"{section_name}.{field_name}")

    if missing:
        raise SettingsError(
            "Invalid configuration - missing required field(s): " + ", ".join(missing)
        )


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """Read the YAML configuration and return a validated `Settings` object.

    Raises:
        SettingsError: the file is missing, unparsable, or lacks required fields.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise SettingsError(f"Configuration file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SettingsError(f"Failed to parse {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SettingsError(f"{config_path} must contain a YAML mapping at the top level")

    settings = _build_settings(raw)
    validate_settings(settings)
    return settings
