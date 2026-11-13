"""Unit tests for the configuration loader and validator."""

from pathlib import Path

import pytest
import yaml

from core.settings import (
    Settings,
    SettingsError,
    load_settings,
    validate_settings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def base_config() -> dict:
    return {
        "llm": {"provider": "openai", "model": "gpt-4o"},
        "embedding": {"provider": "openai", "model": "text-embedding-ada-002"},
        "vector_store": {"provider": "chroma", "persist_directory": "./data/db/chroma"},
        "retrieval": {"dense_top_k": 20, "sparse_top_k": 20, "fusion_top_k": 10},
        "rerank": {"provider": "none"},
        "evaluation": {"provider": "custom"},
        "observability": {"log_level": "INFO"},
    }


class TestLoadSettings:
    def test_loads_project_configuration(self):
        settings = load_settings(CONFIG_PATH)

        assert isinstance(settings, Settings)
        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o"
        assert settings.embedding.provider == "openai"
        assert settings.vector_store.provider == "chroma"
        assert settings.vector_store.persist_directory == "./data/db/chroma"
        assert settings.retrieval.fusion_top_k == 10
        assert settings.rerank.provider == "none"
        assert settings.evaluation.provider == "custom"
        assert settings.observability.log_level == "INFO"

    def test_loads_nested_ingestion_options(self):
        settings = load_settings(CONFIG_PATH)

        assert settings.ingestion.chunk_size == 1000
        assert settings.ingestion.chunk_overlap == 200
        assert settings.ingestion.splitter == "recursive"
        assert settings.ingestion.chunk_refiner.use_llm is False
        assert settings.ingestion.metadata_enricher.use_llm is False

    def test_missing_file_raises_readable_error(self, tmp_path):
        missing = tmp_path / "nope.yaml"

        with pytest.raises(SettingsError, match="Configuration file not found"):
            load_settings(missing)

    def test_invalid_yaml_raises_readable_error(self, tmp_path):
        path = tmp_path / "settings.yaml"
        path.write_text("llm: [unclosed\n", encoding="utf-8")

        with pytest.raises(SettingsError, match="Failed to parse"):
            load_settings(path)

    def test_non_mapping_document_raises_readable_error(self, tmp_path):
        path = tmp_path / "settings.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")

        with pytest.raises(SettingsError, match="must contain a YAML mapping"):
            load_settings(path)

    def test_missing_required_field_names_the_field(self, tmp_path):
        data = base_config()
        del data["embedding"]["provider"]
        path = write_config(tmp_path, data)

        with pytest.raises(SettingsError, match=r"embedding\.provider"):
            load_settings(path)

    def test_missing_section_names_every_absent_field(self, tmp_path):
        data = base_config()
        del data["retrieval"]
        path = write_config(tmp_path, data)

        with pytest.raises(SettingsError) as exc_info:
            load_settings(path)

        message = str(exc_info.value)
        assert "retrieval.dense_top_k" in message
        assert "retrieval.sparse_top_k" in message
        assert "retrieval.fusion_top_k" in message

    def test_empty_required_field_is_rejected(self, tmp_path):
        data = base_config()
        data["llm"]["model"] = ""
        path = write_config(tmp_path, data)

        with pytest.raises(SettingsError, match=r"llm\.model"):
            load_settings(path)


class TestValidateSettings:
    def test_valid_settings_pass(self):
        settings = load_settings(CONFIG_PATH)

        assert validate_settings(settings) is None

    def test_invalid_settings_raise_with_field_path(self):
        settings = Settings()
        settings.embedding.provider = "openai"

        with pytest.raises(SettingsError, match=r"llm\.provider"):
            validate_settings(settings)


class TestApiKeyResolution:
    def test_empty_api_key_falls_back_to_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        data = base_config()
        data["llm"]["api_key"] = ""
        path = write_config(tmp_path, data)

        settings = load_settings(path)

        assert settings.llm.api_key == "sk-from-env"

    def test_configured_api_key_wins_over_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        data = base_config()
        data["llm"]["api_key"] = "sk-from-config"
        path = write_config(tmp_path, data)

        settings = load_settings(path)

        assert settings.llm.api_key == "sk-from-config"

    def test_provider_without_credentials_stays_empty(self, tmp_path):
        data = base_config()
        data["llm"]["provider"] = "ollama"
        path = write_config(tmp_path, data)

        settings = load_settings(path)

        assert settings.llm.api_key == ""


class TestKeyAliases:
    def test_accepts_dev_spec_key_names(self, tmp_path):
        data = base_config()
        data["vector_store"] = {"backend": "chroma", "persist_path": "./data/db/chroma"}
        data["retrieval"] = {
            "top_k_dense": 15,
            "top_k_sparse": 12,
            "top_k_final": 7,
        }
        data["rerank"] = {"backend": "none"}
        path = write_config(tmp_path, data)

        settings = load_settings(path)

        assert settings.vector_store.provider == "chroma"
        assert settings.vector_store.persist_directory == "./data/db/chroma"
        assert settings.retrieval.dense_top_k == 15
        assert settings.retrieval.sparse_top_k == 12
        assert settings.retrieval.fusion_top_k == 7
        assert settings.rerank.provider == "none"


class TestMainEntryPoint:
    def test_main_fails_fast_on_invalid_configuration(self, tmp_path, monkeypatch):
        data = base_config()
        del data["embedding"]["provider"]
        path = write_config(tmp_path, data)
        monkeypatch.setenv("SMART_KNOWLEDGE_HUB_CONFIG", str(path))

        import main

        assert main.main() == 1
