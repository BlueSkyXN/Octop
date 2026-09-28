"""MCP wire content and in-process LangChain fallback keep images and errors."""

from __future__ import annotations

from typing import Any

import pytest
from mcp.types import CallToolResult

from octop.infra.connectors.catalog import get_catalog_entry
from octop.infra.connectors.gateway.langchain import build_gateway_langchain_tools
from octop.infra.connectors.gateway.protocol import handle_mcp_request


def test_wire_and_langchain_keep_all_content(monkeypatch: pytest.MonkeyPatch) -> None:
    content = [
        {"type": "text", "text": "first"},
        {"type": "image", "data": "aW1hZ2U=", "mimeType": "image/png"},
        {"type": "text", "text": "last"},
    ]
    received: list[dict[str, Any]] = []

    def call(kind: str, creds: dict[str, Any], name: str, args: dict[str, Any]) -> Any:
        received.append(args)
        return content

    monkeypatch.setattr("octop.infra.connectors.gateway.protocol.call_gateway_tool", call)
    response = handle_mcp_request(
        kind="feishu-cli",
        creds={},
        body={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "im"}},
    )
    CallToolResult.model_validate(response["result"])
    tools = build_gateway_langchain_tools(
        entry=get_catalog_entry("feishu-cli"),
        instance_id="test",
        mcp_server_name="feishu",
        creds={},
    )
    tool = next(item for item in tools if item.name.endswith("_im"))
    result = tool.invoke({"method": "+messages-resources-download", "args": {"message_id": "om_1"}})
    assert received[-1]["args"] == {"message_id": "om_1"}
    assert result == [
        content[0],
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
        content[2],
    ]


def test_error_is_a_failed_tool_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def call(*args: Any) -> str:
        raise ValueError("permission denied")

    monkeypatch.setattr("octop.infra.connectors.gateway.protocol.call_gateway_tool", call)
    tools = build_gateway_langchain_tools(
        entry=get_catalog_entry("feishu-cli"),
        instance_id="test",
        mcp_server_name="feishu",
        creds={},
    )
    tool = next(item for item in tools if item.name.endswith("_im"))
    result = tool.invoke(
        {"type": "tool_call", "id": "call_1", "name": tool.name, "args": {"method": "list"}}
    )
    assert result.status == "error"
    assert "permission denied" in result.content
