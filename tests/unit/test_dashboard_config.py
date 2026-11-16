"""Unit tests for the dashboard configuration service.

The service turns the settings into the cards the overview page draws, so the
tests pin that every component is described and that a component which is
switched off is left out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from observability.dashboard.services.config_service import ConfigService

SETTINGS_YAML = """
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.2
  max_tokens: 2048

embedding:
  provider: "openai"
  model: "text-embedding-ada-002"
  dimensions: 1536

vision_llm:
  enabled: false
  provider: "openai"
  model: "gpt-4o"

vector_store:
  provider: "chroma"
  persist_directory: "./data/db/chroma"
  collection_name: "knowledge_hub"

retrieval:
  dense_top_k: 20
  sparse_top_k: 20
  fusion_top_k: 10
  rrf_k: 60

rerank:
  enabled: false
  provider: "cross_encoder"
  model: "bge-reranker"
  top_k: 5

evaluation:
  enabled: false
  provider: "custom"

observability:
  log_level: "INFO"
  trace_enabled: true
  trace_file: "./logs/traces.jsonl"

ingestion:
  chunk_size: 1000
  chunk_overlap: 200
  splitter: "recursive"
  batch_size: 100

dashboard:
  enabled: true
  port: 8501
  traces_dir: "./logs"
  auto_refresh: true
  refresh_interval: 5
"""


@pytest.fixture
def settings_file(tmp_path: Path) -> Path:
    """Write a configuration file and return its path."""
    path = tmp_path / "settings.yaml"
    path.write_text(SETTINGS_YAML, encoding="utf-8")
    return path


@pytest.fixture
def service(settings_file: Path) -> ConfigService:
    """Return a service that reads the temporary configuration."""
    return ConfigService(settings_file)


class TestComponentCards:
    def test_the_llm_is_described(self, service: ConfigService):
        cards = {card.name: card for card in service.get_component_cards()}

        assert cards["LLM"].provider == "openai"
        assert cards["LLM"].model == "gpt-4o"

    def test_the_embedding_is_described(self, service: ConfigService):
        cards = {card.name: card for card in service.get_component_cards()}

        assert cards["Embedding"].provider == "openai"
        assert cards["Embedding"].details["dimensions"] == 1536

    def test_the_vector_store_is_described(self, service: ConfigService):
        cards = {card.name: card for card in service.get_component_cards()}

        assert cards["Vector Store"].provider == "chroma"
        assert cards["Vector Store"].model == "knowledge_hub"

    def test_the_retrieval_is_described(self, service: ConfigService):
        cards = {card.name: card for card in service.get_component_cards()}

        assert cards["Retrieval"].details["fusion_top_k"] == 10

    def test_the_ingestion_is_described(self, service: ConfigService):
        cards = {card.name: card for card in service.get_component_cards()}

        assert cards["Ingestion"].details["chunk_size"] == 1000

    def test_a_component_that_is_switched_off_is_left_out(self, service: ConfigService):
        cards = {card.name for card in service.get_component_cards()}

        assert "Reranker" not in cards
        assert "Vision LLM" not in cards

    def test_every_card_names_a_provider_and_a_model(self, service: ConfigService):
        for card in service.get_component_cards():
            assert card.provider
            assert card.model


class TestComponentDetails:
    def test_the_details_are_rows_of_label_and_value(self, service: ConfigService):
        card = next(card for card in service.get_component_cards() if card.name == "LLM")

        rows = card.as_rows()

        assert ("temperature", "0.2") in rows

    def test_a_value_that_is_nothing_is_shown_as_a_dash(self, service: ConfigService):
        card = next(card for card in service.get_component_cards() if card.name == "LLM")

        rows = dict(card.as_rows())

        assert rows["max_tokens"] == "2048"


class TestDashboardSettings:
    def test_the_dashboard_settings_are_listed(self, service: ConfigService):
        rows = dict(service.get_summary_rows())

        assert rows["port"] == "8501"
        assert rows["auto_refresh"] == "yes"

    def test_the_configuration_is_read_only_once(self, service: ConfigService):
        first = service.settings
        second = service.settings

        assert first is second

    def test_the_configuration_is_read_again_after_a_reload(self, service: ConfigService):
        first = service.settings

        service.reload()

        assert service.settings is not first
