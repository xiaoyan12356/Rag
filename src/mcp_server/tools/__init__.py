"""The tools the server offers.

A tool is a name, a description, the schema of its arguments and the
function that runs it. The protocol handler only knows this contract, so a
new tool can be added without touching the protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.response.response_builder import MCPToolResult


class ToolArgumentError(ValueError):
    """Raised when the arguments of a tool do not fit its schema."""


@dataclass(frozen=True)
class MCPTool:
    """One tool the client can call."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], MCPToolResult]

    def to_dict(self) -> dict[str, Any]:
        """Return the tool as the schema a client reads."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": dict(self.input_schema),
        }


def build_tools(settings: Any | None = None) -> list[MCPTool]:
    """Return the tools the server offers.

    Args:
        settings: The configuration the tools run with.
    """
    from mcp_server.tools import (
        get_document_summary,
        list_collections,
        query_knowledge_hub,
    )

    return [
        MCPTool(
            name=query_knowledge_hub.TOOL_NAME,
            description=query_knowledge_hub.DESCRIPTION,
            input_schema=query_knowledge_hub.INPUT_SCHEMA,
            handler=lambda arguments: query_knowledge_hub.handle(arguments, settings),
        ),
        MCPTool(
            name=list_collections.TOOL_NAME,
            description=list_collections.DESCRIPTION,
            input_schema=list_collections.INPUT_SCHEMA,
            handler=lambda arguments: list_collections.handle(arguments, settings),
        ),
        MCPTool(
            name=get_document_summary.TOOL_NAME,
            description=get_document_summary.DESCRIPTION,
            input_schema=get_document_summary.INPUT_SCHEMA,
            handler=lambda arguments: get_document_summary.handle(arguments, settings),
        ),
    ]
