"""Smoke tests: the project skeleton must stay importable."""

import importlib

import pytest

TOP_LEVEL_PACKAGES = ["core", "ingestion", "libs", "mcp_server", "observability"]

SUB_PACKAGES = [
    "core.query_engine",
    "core.response",
    "core.trace",
    "ingestion.chunking",
    "ingestion.embedding",
    "ingestion.storage",
    "ingestion.transform",
    "libs.embedding",
    "libs.evaluator",
    "libs.llm",
    "libs.loader",
    "libs.reranker",
    "libs.splitter",
    "libs.vector_store",
    "mcp_server.tools",
    "observability.dashboard",
    "observability.evaluation",
]


@pytest.mark.parametrize("package_name", TOP_LEVEL_PACKAGES)
def test_top_level_package_imports(package_name):
    module = importlib.import_module(package_name)
    assert module.__name__ == package_name


@pytest.mark.parametrize("package_name", SUB_PACKAGES)
def test_sub_package_imports(package_name):
    module = importlib.import_module(package_name)
    assert module.__name__ == package_name


def test_main_entry_point_runs():
    import main

    assert main.main() == 0
