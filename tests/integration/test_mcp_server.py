"""Integration tests for the MCP server over stdio.

The tests start the server as a subprocess and talk to it the way a client
does. They pin the rule that makes an MCP server usable: the stdout carries
nothing but the protocol messages, while the logs go to stderr. A log line
on the stdout would break every client, so the stdout is parsed as JSON and
nothing else is allowed to be there.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

if not PYTHON.exists():  # pragma: no cover - depends on the checkout
    PYTHON = Path(sys.executable)

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0"},
    },
}

TOOLS_LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


def server_environment() -> dict[str, str]:
    """Return the environment the server is started with."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing}" if existing else str(SRC_DIR)
    return env


def run_server(messages: list[dict[str, Any]], mode: str = "plain") -> subprocess.CompletedProcess[str]:
    """Start the server, send it the messages and wait for it to stop.

    Args:
        messages: The messages to send, one per line.
        mode: The mode the server runs in.

    Returns:
        The finished process, with its output captured.
    """
    payload = "".join(json.dumps(message) + "\n" for message in messages)
    return subprocess.run(
        [
            str(PYTHON),
            "-m",
            "mcp_server.server",
            "--mode",
            mode,
        ],
        input=payload,
        cwd=str(REPO_ROOT),
        env=server_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def stdout_messages(output: str) -> list[dict[str, Any]]:
    """Return the messages the server wrote to its stdout."""
    messages: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        messages.append(json.loads(line))
    return messages


class TestTheHandshake:
    def test_the_server_answers_the_initialize_request(self):
        result = run_server([INITIALIZE])

        messages = stdout_messages(result.stdout)
        assert messages[0]["id"] == 1
        assert "result" in messages[0]

    def test_the_server_says_who_it_is(self):
        result = run_server([INITIALIZE])

        server_info = stdout_messages(result.stdout)[0]["result"]["serverInfo"]
        assert server_info["name"] == "smart-knowledge-hub"
        assert server_info["version"]

    def test_the_server_declares_the_tools_capability(self):
        result = run_server([INITIALIZE])

        capabilities = stdout_messages(result.stdout)[0]["result"]["capabilities"]
        assert "tools" in capabilities

    def test_the_server_says_the_protocol_version_it_speaks(self):
        result = run_server([INITIALIZE])

        assert stdout_messages(result.stdout)[0]["result"]["protocolVersion"]

    def test_the_server_exits_cleanly(self):
        result = run_server([INITIALIZE])

        assert result.returncode == 0


class TestTheTools:
    def test_the_tools_are_listed(self):
        result = run_server([INITIALIZE, TOOLS_LIST])

        messages = stdout_messages(result.stdout)
        names = [tool["name"] for tool in messages[1]["result"]["tools"]]
        assert "query_knowledge_hub" in names
        assert "list_collections" in names
        assert "get_document_summary" in names

    def test_every_tool_carries_a_schema(self):
        result = run_server([INITIALIZE, TOOLS_LIST])

        messages = stdout_messages(result.stdout)
        for tool in messages[1]["result"]["tools"]:
            assert tool["inputSchema"]["type"] == "object"
            assert tool["description"]


class TestTheStreams:
    def test_the_stdout_carries_only_the_protocol_messages(self):
        result = run_server([INITIALIZE, TOOLS_LIST])

        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            json.loads(line)

    def test_the_stdout_carries_no_log_line(self):
        result = run_server([INITIALIZE, TOOLS_LIST])

        assert "INFO" not in result.stdout
        assert "mcp_server:" not in result.stdout

    def test_the_stderr_carries_the_logs(self):
        result = run_server([INITIALIZE])

        assert "mcp_server" in result.stderr

    def test_the_stdout_of_an_unknown_method_is_still_only_json(self):
        result = run_server([{"jsonrpc": "2.0", "id": 9, "method": "nope"}])

        messages = stdout_messages(result.stdout)
        assert messages[0]["error"]["code"] == -32601


class TestTheSdkMode:
    def test_a_client_can_complete_the_handshake(self):
        result = run_server([INITIALIZE], mode="sdk")

        messages = stdout_messages(result.stdout)
        assert messages[0]["id"] == 1
        assert "result" in messages[0]

    def test_the_stdout_of_the_sdk_mode_is_only_json(self):
        result = run_server([INITIALIZE], mode="sdk")

        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            json.loads(line)


@pytest.mark.parametrize("mode", ["plain", "sdk"])
def test_the_server_starts_with_either_mode(mode: str) -> None:
    result = run_server([INITIALIZE], mode=mode)

    assert result.returncode == 0
    assert stdout_messages(result.stdout)


class FakeSearch:
    """A search that answers with a fixed set of results."""

    def __init__(self, results: list[Any]) -> None:
        """Set up the search.

        Args:
            results: The results every query answers with.
        """
        self.results = results
        self.queries: list[str] = []

    def search(self, query, top_k=None, filters=None, trace=None):
        """Return the results that were set up."""
        self.queries.append(query)
        return list(self.results)


class TestQueryKnowledgeHub:
    """The search tool, driven the way the protocol handler drives it."""

    def test_the_answer_carries_the_citation_markers(self):
        from core.types import RetrievalResult
        from mcp_server.tools.query_knowledge_hub import query_knowledge_hub

        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                text="The first passage.",
                metadata={"source_path": "docs/guide.pdf", "page": 2},
            ),
            RetrievalResult(
                chunk_id="chunk-2",
                score=0.8,
                text="The second passage.",
                metadata={"source_path": "docs/other.md"},
            ),
        ]

        result = query_knowledge_hub("what is it?", search=FakeSearch(results))

        text = result.content[0]["text"]
        assert "[1]" in text
        assert "[2]" in text

    def test_the_citations_carry_the_source_the_page_the_chunk_and_the_score(self):
        from core.types import RetrievalResult
        from mcp_server.tools.query_knowledge_hub import query_knowledge_hub

        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                text="The first passage.",
                metadata={"source_path": "docs/guide.pdf", "page": 2},
            )
        ]

        result = query_knowledge_hub("what is it?", search=FakeSearch(results))

        citation = result.structured_content["citations"][0]
        assert citation["source"] == "guide.pdf"
        assert citation["page"] == 2
        assert citation["chunk_id"] == "chunk-1"
        assert citation["score"] == pytest.approx(0.9)

    def test_a_search_that_finds_nothing_gives_a_friendly_message(self):
        from mcp_server.tools.query_knowledge_hub import query_knowledge_hub

        result = query_knowledge_hub("what is it?", search=FakeSearch([]))

        assert result.content[0]["text"].strip() != ""
        assert result.structured_content["citations"] == []

    def test_the_question_is_given_to_the_search(self):
        from mcp_server.tools.query_knowledge_hub import query_knowledge_hub

        search = FakeSearch([])

        query_knowledge_hub("what is it?", search=search)

        assert search.queries == ["what is it?"]

    def test_the_question_is_written_into_the_trace(self, tmp_path):
        from core.settings import IngestionSettings, ObservabilitySettings, Settings
        from core.types import RetrievalResult
        from mcp_server.tools.query_knowledge_hub import query_knowledge_hub

        settings = Settings(
            observability=ObservabilitySettings(trace_file=str(tmp_path / "traces.jsonl")),
            ingestion=IngestionSettings(
                image_index_db=str(tmp_path / "image_index.db"),
                images_root=str(tmp_path / "images"),
            ),
        )
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                text="The passage.",
                metadata={"source_path": "docs/guide.pdf"},
            )
        ]

        query_knowledge_hub("what is it?", search=FakeSearch(results), settings=settings)

        lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        trace = json.loads(lines[0])
        assert trace["metadata"]["query"] == "what is it?"

    def test_the_reranker_is_written_into_the_trace(self, tmp_path):
        from core.settings import (
            IngestionSettings,
            ObservabilitySettings,
            RerankSettings,
            Settings,
        )
        from core.types import RetrievalResult
        from mcp_server.tools.query_knowledge_hub import query_knowledge_hub

        settings = Settings(
            rerank=RerankSettings(provider="none"),
            observability=ObservabilitySettings(trace_file=str(tmp_path / "traces.jsonl")),
            ingestion=IngestionSettings(
                image_index_db=str(tmp_path / "image_index.db"),
                images_root=str(tmp_path / "images"),
            ),
        )
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                text="The passage.",
                metadata={"source_path": "docs/guide.pdf"},
            )
        ]

        query_knowledge_hub("what is it?", search=FakeSearch(results), settings=settings)

        lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        trace = json.loads(lines[0])
        stages = [stage["stage"] for stage in trace["stages"]]
        assert "rerank" in stages


class TestTheImageContent:
    """The images a passage points at travel with the answer."""

    def test_an_image_of_a_passage_is_returned_as_base64(self, tmp_path):
        from core.response.multimodal_assembler import MultimodalAssembler
        from core.types import RetrievalResult
        from mcp_server.tools.query_knowledge_hub import query_knowledge_hub

        image = tmp_path / "figure.png"
        image.write_bytes(b"hello")
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                text="The passage about the figure.",
                metadata={
                    "source_path": "docs/guide.pdf",
                    "images": [{"id": "img-1", "path": str(image)}],
                },
            )
        ]

        result = query_knowledge_hub(
            "what is it?",
            search=FakeSearch(results),
            assembler=MultimodalAssembler(),
        )

        blocks = [block for block in result.content if block["type"] == "image"]
        assert len(blocks) == 1
        assert blocks[0]["mimeType"] == "image/png"
        assert blocks[0]["data"] == "aGVsbG8="

    def test_the_text_comes_first(self, tmp_path):
        from core.response.multimodal_assembler import MultimodalAssembler
        from core.types import RetrievalResult
        from mcp_server.tools.query_knowledge_hub import query_knowledge_hub

        image = tmp_path / "figure.png"
        image.write_bytes(b"hello")
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                text="The passage about the figure.",
                metadata={"images": [{"id": "img-1", "path": str(image)}]},
            )
        ]

        result = query_knowledge_hub(
            "what is it?",
            search=FakeSearch(results),
            assembler=MultimodalAssembler(),
        )

        assert result.content[0]["type"] == "text"

