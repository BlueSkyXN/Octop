"""LangChain tool factory for in-process gateway MCP fallback."""

from __future__ import annotations

import json
import re
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, create_model, model_validator

from octop.infra.connectors.catalog import ConnectorCatalogEntry
from octop.infra.connectors.gateway.protocol import handle_mcp_request
from octop.infra.connectors.gateway.registry import GatewayToolResult, mcp_tools_for_kind


def _coerce_json_container(value: Any) -> Any:
    """Accept a JSON string where the tool schema asks for object/array.

    Harness ``mcp_args_model`` flattens object/array schemas to ``str``, so
    models trained against the MCP schema send ``"{\"k\": 1}"``; parse it back
    instead of dropping it in the adapter.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("{", "[")):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


class _GatewayArgsBase(BaseModel):
    """Drop explicit ``null`` args before validation (harness mcp parity)."""

    @model_validator(mode="before")
    @classmethod
    def _drop_null_arguments(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return {key: value for key, value in data.items() if value is not None}


def gateway_args_model(tool_name: str, input_schema: dict[str, Any]) -> type[Any]:
    """Build a Pydantic args model that keeps object/array schemas native.

    Unlike harness ``mcp_args_model`` (object/array → ``str``), object fields
    stay ``dict`` so the JSON schema the model sees matches the gateway
    tool contract, and JSON strings sent by the model still validate.
    """
    props = input_schema.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    required = set(input_schema.get("required") or [])

    fields: dict[str, Any] = {}
    for key, spec in props.items():
        spec_dict: dict[str, Any] = spec if isinstance(spec, dict) else {}
        desc = spec_dict.get("description")
        json_type = spec_dict.get("type")
        annotated: Any
        if json_type == "integer":
            annotated = int
        elif json_type == "number":
            annotated = float
        elif json_type == "boolean":
            annotated = bool
        elif json_type == "array":
            annotated = Annotated[  # noqa: UP037
                list[Any], BeforeValidator(_coerce_json_container)
            ]
        elif json_type == "object":
            annotated = Annotated[  # noqa: UP037
                dict[str, Any], BeforeValidator(_coerce_json_container)
            ]
        else:
            annotated = str
        if key in required:
            fields[str(key)] = (annotated, Field(description=desc))
        else:
            fields[str(key)] = (annotated | None, Field(default=None, description=desc))

    model_name = re.sub(r"[^A-Za-z0-9_]", "_", tool_name)
    if not model_name or model_name[0].isdigit():
        model_name = f"Gateway_{model_name}"
    return create_model(model_name, __base__=_GatewayArgsBase, **fields)


def _content_to_result(content: list[Any]) -> GatewayToolResult:
    """Unwrap single-text content; pass multimodal blocks through as a list."""
    if len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "text":
        return str(content[0].get("text") or "")
    return [block for block in content if isinstance(block, dict)]


def build_gateway_langchain_tools(
    *,
    entry: Any,
    instance_id: str,
    mcp_server_name: str,
    creds: dict[str, Any],
) -> list[Any]:
    """In-process LangChain tools when harness HTTP MCP load misses gateway servers."""
    from langchain_core.tools import StructuredTool

    del instance_id
    if not isinstance(entry, ConnectorCatalogEntry) or entry.mcp_mode != "gateway":
        return []
    out: list[Any] = []

    def _tool_fn(kind: str, tool_name: str) -> Any:
        def _run(**kwargs: Any) -> GatewayToolResult:
            from langchain_core.tools import ToolException

            cleaned = {k: v for k, v in kwargs.items() if v is not None}
            resp = handle_mcp_request(
                kind=kind,
                creds=creds,
                body={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": cleaned},
                },
            )
            if not isinstance(resp, dict):
                raise ToolException("gateway error")
            if resp.get("error"):
                err = resp.get("error") or {}
                raise ToolException(str(err.get("message") or err))
            result = resp.get("result") or {}
            if result.get("isError"):
                content = result.get("content") or []
                if content and isinstance(content[0], dict):
                    raise ToolException(str(content[0].get("text") or "tool error"))
                raise ToolException("tool error")
            content = result.get("content") or []
            if content:
                return _content_to_result(content)
            return json.dumps(result, ensure_ascii=False)

        return _run

    for tool_def in mcp_tools_for_kind(entry.kind):
        name = str(tool_def.get("name") or "").strip()
        if not name:
            continue
        input_schema = tool_def.get("inputSchema")
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}
        from harness_agent.mcp import sanitize_llm_tool_name

        lc_name = sanitize_llm_tool_name(f"{mcp_server_name}_{name}")
        out.append(
            StructuredTool.from_function(
                func=_tool_fn(entry.kind, name),
                name=lc_name,
                description=str(tool_def.get("description") or name),
                args_schema=gateway_args_model(lc_name, input_schema),
                handle_tool_error=True,
            )
        )
    return out


__all__ = ["build_gateway_langchain_tools", "gateway_args_model"]
