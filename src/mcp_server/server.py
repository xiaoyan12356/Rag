"""The MCP server.

A client starts this program and talks to it over stdin and stdout. The
stdout carries nothing but the protocol messages, so every log line goes to
stderr.

The server runs in one of two modes. The plain mode answers the JSON-RPC
messages itself, which keeps the protocol layer testable and independent of
the SDK. The SDK mode hands the messages to the MCP SDK, which is what a
client that speaks the full protocol expects.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import IO, Any

from core.settings import Settings, SettingsError, load_settings, resolve_path
from mcp_server.protocol_handler import (
    DEFAULT_SERVER_NAME,
    DEFAULT_SERVER_VERSION,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    ProtocolHandler,
)
from mcp_server.tools import ToolArgumentError, build_tools
from observability.logger import get_logger

CONFIG_PATH_ENV_VAR = "SMART_KNOWLEDGE_HUB_CONFIG"
DEFAULT_CONFIG_PATH = "config/settings.yaml"

MODE_SDK = "sdk"
MODE_PLAIN = "plain"
DEFAULT_MODE = MODE_SDK

logger = logging.getLogger(__name__)


class MCPServer:
    """Answer the messages a client sends over stdin."""

    def __init__(
        self,
        handler: ProtocolHandler | None = None,
        stdin: IO[str] | None = None,
        stdout: IO[str] | None = None,
    ) -> None:
        """Set up the server.

        Args:
            handler: The one that answers the messages; an empty one is used
                when this is left out.
            stdin: Where the messages come from.
            stdout: Where the answers go.
        """
        self.handler = handler or ProtocolHandler()
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout
        self.logger = get_logger("mcp_server")

    def serve(self) -> int:
        """Answer the messages until the client stops talking."""
        self.logger.info("The MCP server is ready")

        try:
            while True:
                line = self.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                response = self.handler.parse(line)
                if response is not None:
                    self.write(response)
        except (OSError, ValueError) as exc:
            self.logger.info("The stream of the client ended: %s", exc)

        self.logger.info("The client stopped talking")
        return 0

    def write(self, message: dict[str, Any]) -> None:
        """Send a message to the client."""
        self.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.stdout.flush()


def build_handler(settings: Settings) -> ProtocolHandler:
    """Return the handler that offers the tools of the server."""
    return ProtocolHandler(
        tools=build_tools(settings),
        server_name=DEFAULT_SERVER_NAME,
        server_version=DEFAULT_SERVER_VERSION,
    )


def settings_path(override: str | None = None) -> Path:
    """Return the configuration file to read: the argument first, then the
    environment, then the one of the checkout.

    A client starts the server in whatever folder it likes, so a relative
    path is read from the checkout rather than from the working directory.
    """
    if override:
        return resolve_path(override)
    from_environment = os.environ.get(CONFIG_PATH_ENV_VAR)
    if from_environment:
        return resolve_path(from_environment)
    return resolve_path(DEFAULT_CONFIG_PATH)


def main(argv: list[str] | None = None) -> int:
    """Run the server. Returns the exit code of the process."""
    parser = argparse.ArgumentParser(
        prog="mcp-server",
        description="Run the Modular RAG MCP Server over stdio.",
    )
    parser.add_argument("--config", help="The configuration file to read.")
    parser.add_argument(
        "--mode",
        choices=(MODE_SDK, MODE_PLAIN),
        default=DEFAULT_MODE,
        help=(
            "How the protocol is spoken: 'sdk' uses the MCP SDK, 'plain' answers "
            f"the JSON-RPC messages directly (default: {DEFAULT_MODE})."
        ),
    )
    arguments = parser.parse_args(argv)

    _send_logs_to_stderr()

    server_logger = get_logger("mcp_server")
    try:
        settings = load_settings(settings_path(arguments.config))
    except SettingsError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    server_logger.info("Settings loaded from %s", settings_path(arguments.config))
    server_logger.info(
        "llm=%s/%s embedding=%s/%s vector_store=%s",
        settings.llm.provider,
        settings.llm.model,
        settings.embedding.provider,
        settings.embedding.model,
        settings.vector_store.provider,
    )

    if arguments.mode == MODE_PLAIN:
        return MCPServer(handler=build_handler(settings)).serve()

    _preload_heavy_imports()
    return run_sdk_server(build_handler(settings))


def _preload_heavy_imports() -> None:
    """Load the heavy modules before the SDK starts its own threads.

    The SDK runs a tool in a worker thread, and the first call has to build
    the vector store, which pulls in chromadb and its native libraries. An
    import that happens in a worker thread can deadlock against the thread
    that reads the client, so the import is done here, in the main thread.
    """
    for module in (
        "chromadb",
        "core.query_engine.hybrid_search",
        "core.query_engine.dense_retriever",
        "core.query_engine.sparse_retriever",
        "core.query_engine.reranker",
        "ingestion.storage.bm25_indexer",
        "libs.embedding.embedding_factory",
        "libs.vector_store.vector_store_factory",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:
            logger.info("The module '%s' was not loaded ahead: %s", module, exc)


def run_sdk_server(handler: ProtocolHandler) -> int:
    """Run the server with the MCP SDK, which speaks the whole protocol.

    Args:
        handler: The one that knows the tools of the server.

    Returns:
        The exit code of the process.
    """
    from mcp.server.lowlevel import NotificationOptions, Server
    from mcp.server.stdio import stdio_server

    server = Server(
        DEFAULT_SERVER_NAME,
        version=DEFAULT_SERVER_VERSION,
        on_list_tools=_make_list_tools(handler),
        on_call_tool=_make_call_tool(handler),
    )

    async def run() -> None:
        """Serve the client until it stops talking."""
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(
                    notification_options=NotificationOptions(tools_changed=False)
                ),
            )

    try:
        asyncio.run(run())
    except (KeyboardInterrupt, OSError) as exc:
        get_logger("mcp_server").info("The server stopped: %s", exc)
    except BaseException as exc:
        # The SDK reads the client in a task group, so a stream that cannot
        # be read arrives wrapped in an exception group.
        if _is_no_client(exc):
            get_logger("mcp_server").info("The client is not there: %s", exc)
            return 0
        logger.exception("The server stopped with an error")
        return 1
    return 0


def _is_no_client(exc: BaseException) -> bool:
    """Say whether the failure means the client is not there.

    A server that is started without a client on its stdin has nothing to
    do, which is not an error.
    """
    if isinstance(exc, io.UnsupportedOperation):
        return True
    nested = getattr(exc, "exceptions", None)
    if nested:
        return all(_is_no_client(inner) for inner in nested)
    return False


def _make_list_tools(handler: ProtocolHandler) -> Any:
    """Return the callback that answers `tools/list`."""

    async def list_tools(
        context: Any,
        params: Any = None,
    ) -> Any:
        """Return the schema of every tool the client may call."""
        import mcp.types as types

        tools = [
            types.Tool(
                name=tool["name"],
                description=tool["description"],
                input_schema=tool["inputSchema"],
            )
            for tool in handler.handle_tools_list()
        ]
        return types.ListToolsResult(tools=tools)

    return list_tools


def _make_call_tool(handler: ProtocolHandler) -> Any:
    """Return the callback that answers `tools/call`."""

    async def call_tool(context: Any, params: Any) -> Any:
        """Run the tool the client asked for."""
        import mcp.types as types

        name = getattr(params, "name", None)
        arguments = getattr(params, "arguments", None) or {}
        try:
            result = handler.handle_tools_call(name, arguments)
        except ToolArgumentError as exc:
            raise _as_protocol_error(exc, INVALID_PARAMS) from exc
        except Exception as exc:
            logger.exception("The tool '%s' failed", name)
            raise _as_protocol_error(exc, INTERNAL_ERROR) from exc

        content = [
            types.TextContent(type="text", text=str(block.get("text", "")))
            for block in result.get("content", [])
            if block.get("type") == "text"
        ]
        content.extend(
            types.ImageContent(
                type="image",
                data=str(block["data"]),
                mimeType=str(block["mimeType"]),
            )
            for block in result.get("content", [])
            if block.get("type") == "image"
        )

        return types.CallToolResult(
            content=content,
            structured_content=result.get("structuredContent"),
            is_error=bool(result.get("isError", False)),
        )

    return call_tool


def _as_protocol_error(exc: Exception, code: int) -> Exception:
    """Turn an exception into the error the protocol asks for.

    The message stays short: the traceback goes to the log, never to the
    client.
    """
    from mcp.shared.exceptions import MCPError

    return MCPError(code, str(exc))


def _send_logs_to_stderr() -> None:
    """Make sure no log record ever reaches the stdout."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    sys.exit(main())
