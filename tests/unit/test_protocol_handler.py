"""Unit tests for the JSON-RPC layer of the MCP server.

The tests pin the three methods a client uses most - initialize, tools/list
and tools/call - and the four ways a request can be refused: a message that
is not a request, a method that does not exist, arguments that do not fit
and a tool that breaks. An internal error must never leak a stack trace.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from core.response.response_builder import MCPToolResult
from mcp_server.protocol_handler import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    ProtocolHandler,
)
from mcp_server.tools import MCPTool, ToolArgumentError

INPUT_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}


def echo_tool() -> MCPTool:
    """Return a tool that answers with the arguments it was given."""
    return MCPTool(
        name="echo",
        description="Answer with the query that was asked.",
        input_schema=INPUT_SCHEMA,
        handler=lambda arguments: MCPToolResult(
            content=[{"type": "text", "text": str(arguments.get("query", ""))}],
            structured_content={"query": arguments.get("query")},
        ),
    )


def broken_tool() -> MCPTool:
    """Return a tool that always fails."""
    return MCPTool(
        name="broken",
        description="A tool that fails.",
        input_schema=INPUT_SCHEMA,
        handler=_raise,
    )


def _raise(arguments: dict[str, Any]) -> MCPToolResult:
    raise RuntimeError("the tool is broken")


def make_handler() -> ProtocolHandler:
    """Return a handler that offers the echo tool."""
    return ProtocolHandler(tools=[echo_tool()])


class TestInitialize:
    def test_the_server_says_who_it_is(self):
        handler = make_handler()

        result = handler.handle_initialize({})

        assert result["serverInfo"]["name"] == "smart-knowledge-hub"
        assert result["serverInfo"]["version"] == "0.1.0"

    def test_the_tools_capability_is_declared(self):
        handler = make_handler()

        result = handler.handle_initialize({})

        assert "tools" in result["capabilities"]

    def test_the_asked_protocol_version_is_kept(self):
        handler = make_handler()

        result = handler.handle_initialize({"protocolVersion": "2024-11-05"})

        assert result["protocolVersion"] == "2024-11-05"

    def test_an_unknown_protocol_version_falls_back(self):
        handler = make_handler()

        result = handler.handle_initialize({"protocolVersion": "1999-01-01"})

        assert result["protocolVersion"] == "2025-06-18"

    def test_the_message_is_answered_as_a_result(self):
        handler = make_handler()

        response = handler.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "error" not in response


class TestToolsList:
    def test_the_registered_tools_are_listed(self):
        handler = make_handler()

        tools = handler.handle_tools_list()

        assert [tool["name"] for tool in tools] == ["echo"]

    def test_the_tool_carries_its_schema(self):
        handler = make_handler()

        tools = handler.handle_tools_list()

        assert tools[0]["inputSchema"] == INPUT_SCHEMA
        assert tools[0]["description"] == "Answer with the query that was asked."

    def test_the_message_is_answered_with_a_list(self):
        handler = make_handler()

        response = handler.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

        assert [tool["name"] for tool in response["result"]["tools"]] == ["echo"]


class TestToolsCall:
    def test_the_tool_is_routed_and_answers(self):
        handler = make_handler()

        response = handler.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"query": "hello"}},
            }
        )

        assert response["result"]["content"][0]["text"] == "hello"

    def test_the_structured_part_is_kept(self):
        handler = make_handler()

        response = handler.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"query": "hello"}},
            }
        )

        assert response["result"]["structuredContent"] == {"query": "hello"}

    def test_an_unknown_tool_is_refused(self):
        handler = make_handler()

        response = handler.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "nope"},
            }
        )

        assert response["error"]["code"] == INVALID_PARAMS

    def test_a_missing_tool_name_is_refused(self):
        handler = make_handler()

        response = handler.handle_message(
            {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {}}
        )

        assert response["error"]["code"] == INVALID_PARAMS

    def test_arguments_that_are_not_an_object_are_refused(self):
        handler = make_handler()

        response = handler.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": "hello"},
            }
        )

        assert response["error"]["code"] == INVALID_PARAMS

    def test_a_tool_that_raises_is_an_internal_error(self):
        handler = ProtocolHandler(tools=[broken_tool()])

        response = handler.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {"name": "broken", "arguments": {}},
            }
        )

        assert response["error"]["code"] == INTERNAL_ERROR

    def test_an_internal_error_does_not_leak_a_stack_trace(self):
        handler = ProtocolHandler(tools=[broken_tool()])

        response = handler.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "broken", "arguments": {}},
            }
        )

        message = response["error"]["message"]
        assert "Traceback" not in message
        assert "File \"" not in message

    def test_a_tool_that_raises_is_still_answered(self):
        handler = ProtocolHandler(tools=[broken_tool()])

        response = handler.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "broken", "arguments": {}},
            }
        )

        assert response["id"] == 10


class TestErrors:
    def test_an_unknown_method_is_refused(self):
        handler = make_handler()

        response = handler.handle_message({"jsonrpc": "2.0", "id": 11, "method": "nope"})

        assert response["error"]["code"] == METHOD_NOT_FOUND

    def test_a_message_that_is_not_json_is_refused(self):
        handler = make_handler()

        response = handler.parse("{not json")

        assert response["error"]["code"] == PARSE_ERROR

    def test_a_message_that_is_not_an_object_is_refused(self):
        handler = make_handler()

        response = handler.handle_message(["hello"])

        assert response["error"]["code"] == INVALID_REQUEST

    def test_a_wrong_jsonrpc_version_is_refused(self):
        handler = make_handler()

        response = handler.handle_message({"jsonrpc": "1.0", "id": 12, "method": "initialize"})

        assert response["error"]["code"] == INVALID_REQUEST

    def test_a_missing_method_is_refused(self):
        handler = make_handler()

        response = handler.handle_message({"jsonrpc": "2.0", "id": 13})

        assert response["error"]["code"] == INVALID_REQUEST

    def test_params_that_are_not_an_object_are_refused(self):
        handler = make_handler()

        response = handler.handle_message(
            {"jsonrpc": "2.0", "id": 14, "method": "initialize", "params": "hello"}
        )

        assert response["error"]["code"] == INVALID_PARAMS

    def test_the_id_of_the_request_is_kept_in_the_error(self):
        handler = make_handler()

        response = handler.handle_message({"jsonrpc": "2.0", "id": 15, "method": "nope"})

        assert response["id"] == 15

    def test_a_tool_argument_error_is_an_invalid_params(self):
        handler = ProtocolHandler(
            tools=[
                MCPTool(
                    name="strict",
                    description="A tool with strict arguments.",
                    input_schema=INPUT_SCHEMA,
                    handler=_refuse,
                )
            ]
        )

        response = handler.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 16,
                "method": "tools/call",
                "params": {"name": "strict", "arguments": {}},
            }
        )

        assert response["error"]["code"] == INVALID_PARAMS


def _refuse(arguments: dict[str, Any]) -> MCPToolResult:
    raise ToolArgumentError("The 'query' argument is missing")


class TestParse:
    def test_a_line_of_json_is_answered(self):
        handler = make_handler()

        response = handler.parse(json.dumps({"jsonrpc": "2.0", "id": 17, "method": "ping"}))

        assert response["result"] == {}

    def test_a_notification_is_not_answered(self):
        handler = make_handler()

        response = handler.parse(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        )

        assert response is None


class TestRegistration:
    def test_a_tool_without_a_handler_is_refused(self):
        handler = ProtocolHandler(tools=[echo_tool()])

        response = handler.handle_tools_call("echo", {"query": "hi"})

        assert response["content"][0]["text"] == "hi"

    def test_the_tools_are_kept_by_name(self):
        handler = ProtocolHandler(tools=[echo_tool(), echo_tool()])

        assert list(handler.tools) == ["echo"]


def test_a_message_without_an_id_is_not_answered() -> None:
    handler = make_handler()

    response = handler.handle_message({"jsonrpc": "2.0", "method": "initialize"})

    assert response is None


def test_the_ping_method_is_answered() -> None:
    handler = make_handler()

    response = handler.handle_message({"jsonrpc": "2.0", "id": 18, "method": "ping"})

    assert response["result"] == {}


def test_an_empty_handler_offers_no_tools() -> None:
    handler = ProtocolHandler()

    assert handler.handle_tools_list() == []


def test_a_tool_that_answers_with_a_plain_dictionary_is_accepted() -> None:
    handler = ProtocolHandler(
        tools=[
            MCPTool(
                name="plain",
                description="A tool that answers with a dictionary.",
                input_schema=INPUT_SCHEMA,
                handler=lambda arguments: {"content": []},
            )
        ]
    )

    assert handler.handle_tools_call("plain", {}) == {"content": []}


def test_a_tool_that_answers_with_nothing_is_wrapped() -> None:
    handler = ProtocolHandler(
        tools=[
            MCPTool(
                name="bare",
                description="A tool that answers with a text.",
                input_schema=INPUT_SCHEMA,
                handler=lambda arguments: "plain text",
            )
        ]
    )

    result = handler.handle_tools_call("bare", {})

    assert result["content"][0]["text"] == "plain text"


@pytest.mark.parametrize("code", [INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS])
def test_the_error_codes_are_the_ones_of_the_protocol(code: int) -> None:
    assert code in (-32600, -32601, -32602)
