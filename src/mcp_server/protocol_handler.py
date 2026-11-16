"""The JSON-RPC 2.0 layer.

The protocol handler turns a message into an answer. It knows the methods a
client uses most (initialize, tools/list, tools/call), it knows which tools
are registered, and it knows how to say no in the way the protocol asks
for: a message that is not a request, a method that does not exist, arguments
that do not fit and a tool that breaks all have their own code.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp_server.tools import MCPTool, ToolArgumentError

logger = logging.getLogger(__name__)

JSONRPC_VERSION = "2.0"

# The protocol versions this server understands. A client that asks for one of
# them gets that one; a client that asks for something else gets the newest
# version this server speaks.
PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

DEFAULT_SERVER_NAME = "smart-knowledge-hub"
DEFAULT_SERVER_VERSION = "0.1.0"


class UnknownMethodError(Exception):
    """Raised when a client asks for a method the server does not have."""


class ProtocolHandler:
    """Answer the messages a client sends."""

    def __init__(
        self,
        tools: list[MCPTool] | None = None,
        server_name: str = DEFAULT_SERVER_NAME,
        server_version: str = DEFAULT_SERVER_VERSION,
    ) -> None:
        """Set up the handler.

        Args:
            tools: The tools the client may call.
            server_name: The name the server tells the client it has.
            server_version: The version the server tells the client it has.
        """
        self.tools: dict[str, MCPTool] = {tool.name: tool for tool in (tools or [])}
        self.server_name = server_name
        self.server_version = server_version

    def parse(self, raw: str) -> dict[str, Any] | None:
        """Turn a line of JSON into the answer to it.

        Args:
            raw: The line the client sent.

        Returns:
            The answer, or None when the client sent a notification.
        """
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("A message could not be read: %s", exc.msg)
            return self._error(None, PARSE_ERROR, f"The message is not valid JSON: {exc.msg}")
        return self.handle_message(message)

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        """Answer one message.

        Args:
            message: The message the client sent.

        Returns:
            The answer, or None when the client sent a notification, which
            asks for no answer at all.
        """
        if not isinstance(message, dict):
            return self._error(None, INVALID_REQUEST, "A request must be a JSON object")

        request_id = message.get("id")
        if message.get("jsonrpc") != JSONRPC_VERSION:
            return self._error(
                request_id, INVALID_REQUEST, "The 'jsonrpc' field must be '2.0'"
            )

        method = message.get("method")
        if not isinstance(method, str) or not method:
            return self._error(
                request_id, INVALID_REQUEST, "The 'method' field must be a string"
            )

        params = message.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return self._error(
                request_id, INVALID_PARAMS, "The 'params' field must be an object"
            )

        if "id" not in message:
            self._answer_notification(method)
            return None

        try:
            result = self._dispatch(method, params)
        except UnknownMethodError as exc:
            return self._error(request_id, METHOD_NOT_FOUND, str(exc))
        except ToolArgumentError as exc:
            return self._error(request_id, INVALID_PARAMS, str(exc))
        except Exception as exc:
            logger.exception("The method '%s' failed", method)
            return self._error(
                request_id, INTERNAL_ERROR, f"The method '{method}' failed: {exc}"
            )

        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return the capabilities of the server.

        Args:
            params: The parameters the client sent; the protocol version it
                asks for is read from them.
        """
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.server_name, "version": self.server_version},
        }

    def handle_tools_list(self) -> list[dict[str, Any]]:
        """Return the schema of every tool the client may call."""
        return [tool.to_dict() for tool in self.tools.values()]

    def handle_tools_call(
        self,
        name: Any,
        arguments: Any = None,
    ) -> dict[str, Any]:
        """Run a tool and return what it answered.

        Args:
            name: The name of the tool.
            arguments: The arguments the client sent.

        Raises:
            ToolArgumentError: The tool does not exist, or the arguments do
                not fit it.
        """
        if not isinstance(name, str) or not name:
            raise ToolArgumentError("A tool call needs the name of the tool")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ToolArgumentError("The arguments of a tool must be an object")

        tool = self.tools.get(name)
        if tool is None:
            raise ToolArgumentError(f"Unknown tool: {name}")

        return _as_result(tool.handler(arguments))

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        """Run the method the client asked for."""
        if method == "initialize":
            return self.handle_initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self.handle_tools_list()}
        if method == "tools/call":
            return self.handle_tools_call(params.get("name"), params.get("arguments"))
        raise UnknownMethodError(f"Unknown method: {method}")

    def _answer_notification(self, method: str) -> None:
        """Do the work of a notification, without answering it."""
        if method not in ("notifications/initialized", "initialized"):
            logger.info("The notification '%s' was ignored", method)
            return
        logger.info("The client says it is ready")

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        """Return the answer that says the request could not be answered."""
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {"code": code, "message": message},
        }


def _as_result(result: Any) -> dict[str, Any]:
    """Return what a tool answered in the shape MCP expects."""
    if isinstance(result, dict):
        return result
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {"content": [{"type": "text", "text": str(result)}]}
