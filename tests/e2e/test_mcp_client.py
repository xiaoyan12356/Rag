"""End-to-end test of the MCP server, driven by a client.

The server is started the way a client starts it: as a subprocess, over
stdin and stdout, with the messages of the protocol written to its stdin and
the answers read from its stdout. The documents are ingested before the
server is asked anything, so the query tool has something to find.

The only thing that is replaced is the embedding provider, which a small
local server stands in for, so the test needs no credentials and no network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from stub_embedding_server import DIMENSION, start_stub_server, stop_stub_server

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
INGEST_SCRIPT = REPO_ROOT / "scripts" / "ingest.py"
MAIN_SCRIPT = REPO_ROOT / "main.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "sample_documents"
COLLECTION = "mcp_client"

PROTOCOL_VERSION = "2025-06-18"


class MCPClient:
    """A client that talks to the server over its stdin and stdout."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        """Set up the client.

        Args:
            process: The server that was started.
        """
        self.process = process
        self._next_id = 0

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Ask the server something and return the answer.

        Args:
            method: The method of the protocol to call.
            params: The parameters of the call.

        Returns:
            The answer of the server.
        """
        self._next_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params or {},
        }
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

        line = self.process.stdout.readline()
        if not line:
            raise AssertionError("The server stopped answering")
        return json.loads(line)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Tell the server something, without waiting for an answer."""
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def close(self) -> str:
        """Stop the server and return what it wrote to its stderr."""
        self.process.stdin.close()
        self.process.wait(timeout=60)
        return self.process.stderr.read()


@pytest.fixture
def embedding_server():
    """A local server that answers the embedding requests."""
    address, server = start_stub_server()
    yield address
    stop_stub_server(server)


@pytest.fixture
def config_file(tmp_path: Path, embedding_server: str) -> Path:
    """A configuration that keeps every product in a temporary folder."""
    config = {
        "llm": {"provider": "openai", "model": "gpt-4o"},
        "embedding": {
            "provider": "ollama",
            "model": "stub",
            "dimensions": DIMENSION,
            "base_url": embedding_server,
        },
        "vision_llm": {"enabled": False},
        "vector_store": {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "chroma"),
            "collection_name": COLLECTION,
        },
        "retrieval": {"dense_top_k": 20, "sparse_top_k": 20, "fusion_top_k": 10},
        "rerank": {"provider": "none"},
        "evaluation": {"provider": "custom"},
        "observability": {"log_level": "INFO"},
        "ingestion": {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "splitter": "recursive",
            "batch_size": 16,
            "bm25_index_dir": str(tmp_path / "bm25"),
            "images_root": str(tmp_path / "images"),
            "image_index_db": str(tmp_path / "image_index.db"),
            "history_db": str(tmp_path / "history.db"),
        },
    }
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def ingested(config_file: Path) -> Path:
    """Take the sample documents in before the server is asked anything."""
    result = subprocess.run(
        [
            sys.executable,
            str(INGEST_SCRIPT),
            "--path",
            str(FIXTURES),
            "--collection",
            COLLECTION,
            "--config",
            str(config_file),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return config_file


@pytest.fixture
def client(ingested: Path):
    """Start the server and connect a client to it."""
    process = subprocess.Popen(
        [sys.executable, str(MAIN_SCRIPT), "--config", str(ingested)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    connection = MCPClient(process)
    yield connection
    try:
        connection.close()
    except Exception:
        process.kill()


@pytest.fixture
def connected(client: MCPClient) -> MCPClient:
    """Connect to the server and finish the handshake."""
    client.request(
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "e2e-client", "version": "1.0"},
        },
    )
    client.notify("notifications/initialized")
    return client


@pytest.fixture
def plain_client(ingested: Path):
    """Start the server in the plain mode, where it answers the protocol itself.

    The plain mode is the one that answers a bad request with a JSON-RPC
    error code, which is what the protocol asks for.
    """
    process = subprocess.Popen(
        [
            sys.executable,
            str(MAIN_SCRIPT),
            "--config",
            str(ingested),
            "--mode",
            "plain",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    connection = MCPClient(process)
    yield connection
    try:
        connection.close()
    except Exception:
        process.kill()


class TestTheHandshake:
    def test_the_server_answers_the_initialize_request(self, client: MCPClient):
        answer = client.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "e2e-client", "version": "1.0"},
            },
        )

        assert answer["id"] == 1
        assert answer["result"]["serverInfo"]["name"] == "smart-knowledge-hub"

    def test_the_server_declares_the_tools_capability(self, client: MCPClient):
        answer = client.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "e2e-client", "version": "1.0"},
            },
        )

        assert "tools" in answer["result"]["capabilities"]


class TestTheTools:
    def test_the_three_tools_are_listed(self, connected: MCPClient):
        answer = connected.request("tools/list")

        names = [tool["name"] for tool in answer["result"]["tools"]]
        assert names == [
            "query_knowledge_hub",
            "list_collections",
            "get_document_summary",
        ]

    def test_every_tool_carries_a_schema(self, connected: MCPClient):
        answer = connected.request("tools/list")

        for tool in answer["result"]["tools"]:
            assert tool["inputSchema"]["type"] == "object"
            assert tool["description"]

    def test_a_tool_that_does_not_exist_is_refused(self, connected: MCPClient):
        answer = connected.request("tools/call", {"name": "nope", "arguments": {}})

        assert answer["error"]["code"] == -32602


class TestTheSearch:
    def test_the_query_tool_answers_with_text(self, connected: MCPClient):
        answer = connected.request(
            "tools/call",
            {
                "name": "query_knowledge_hub",
                "arguments": {"query": "how is the document split into chunks"},
            },
        )

        content = answer["result"]["content"]
        assert content
        assert content[0]["type"] == "text"
        assert content[0]["text"].strip()

    def test_the_answer_carries_the_citations(self, connected: MCPClient):
        answer = connected.request(
            "tools/call",
            {
                "name": "query_knowledge_hub",
                "arguments": {"query": "how is the document split into chunks"},
            },
        )

        citations = answer["result"]["structuredContent"]["citations"]
        assert citations
        for citation in citations:
            assert citation["source"]
            assert citation["source_file"]
            assert citation["chunk_id"]
            assert 0.0 <= citation["score"] <= 1.0

    def test_the_answer_marks_the_passages_it_cites(self, connected: MCPClient):
        answer = connected.request(
            "tools/call",
            {
                "name": "query_knowledge_hub",
                "arguments": {"query": "how is the document split into chunks"},
            },
        )

        text = answer["result"]["content"][0]["text"]
        assert "[1]" in text

    def test_a_question_without_a_query_is_refused(self, connected: MCPClient):
        answer = connected.request(
            "tools/call",
            {"name": "query_knowledge_hub", "arguments": {}}
        )

        assert answer["error"]["code"] == -32602
        assert "query" in answer["error"]["message"]


class TestTheOtherTools:
    def test_the_collections_are_listed(self, connected: MCPClient):
        answer = connected.request("tools/call", {"name": "list_collections", "arguments": {}})

        content = answer["result"]["content"]
        assert content
        assert content[0]["text"].strip()

    def test_a_document_that_is_not_there_is_reported(self, connected: MCPClient):
        answer = connected.request(
            "tools/call",
            {
                "name": "get_document_summary",
                "arguments": {"doc_id": "nothing_like_this.pdf"},
            },
        )

        assert answer["result"]["isError"] is True


class TestTheErrorCodes:
    """The plain mode answers a bad request with the code of the protocol."""

    def test_a_method_that_does_not_exist_is_refused(self, plain_client: MCPClient):
        answer = plain_client.request("nope")

        assert answer["error"]["code"] == -32601

    def test_a_question_without_a_query_is_refused(self, plain_client: MCPClient):
        answer = plain_client.request(
            "tools/call",
            {"name": "query_knowledge_hub", "arguments": {}},
        )

        assert answer["error"]["code"] == -32602

    def test_a_tool_that_does_not_exist_is_refused(self, plain_client: MCPClient):
        answer = plain_client.request(
            "tools/call",
            {"name": "nope", "arguments": {}},
        )

        assert answer["error"]["code"] == -32602

    def test_a_message_that_is_not_json_is_refused(self, plain_client: MCPClient):
        plain_client.process.stdin.write("this is not json\n")
        plain_client.process.stdin.flush()

        answer = json.loads(plain_client.process.stdout.readline())

        assert answer["error"]["code"] == -32700
