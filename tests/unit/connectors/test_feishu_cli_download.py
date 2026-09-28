"""Feishu resource downloads keep image bytes inside MCP content blocks."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult

from octop.infra.connectors.gateway.adapters import feishu_cli

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
_CREDS: dict[str, Any] = {"app_id": "cli_x", "app_secret": "s", "default_as": "bot"}


@pytest.fixture(autouse=True)
def fake_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path))
    monkeypatch.setenv("OCTOP_CONNECTOR_IMAGE_BLOCKS", "1")
    monkeypatch.setattr(feishu_cli, "resolve_binary", lambda _name: "fake-lark-cli")
    monkeypatch.setattr(feishu_cli, "_prepare_env", lambda creds, **_kw: {})
    state: dict[str, Any] = {"payload": _TINY_PNG}

    def run(argv: list[str], **kwargs: Any) -> str:
        state["argv"] = argv
        cwd = Path(kwargs["cwd"])
        state["cwd"] = cwd
        saved = state.get("saved_path", cwd / "resource.bin")
        if "saved_path" not in state:
            saved.write_bytes(state["payload"])
        return json.dumps({"saved_path": str(saved)})

    monkeypatch.setattr(feishu_cli, "run_cli", run)
    return state


def _call(args: Any = None) -> Any:
    return feishu_cli.call_tool(
        _CREDS,
        "im",
        {
            "method": "+messages-resources-download",
            "args": args or {"message_id": "om_1", "file_key": "img_v2_1"},
        },
    )


def test_download_returns_mcp_image_and_cleans_temp_dir(fake_cli: dict[str, Any]) -> None:
    result = _call()
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image"
    assert result[1]["mimeType"] == "image/png"
    assert base64.b64decode(result[1]["data"]) == _TINY_PNG
    CallToolResult.model_validate({"content": result, "isError": False})
    assert not fake_cli["cwd"].exists()


def test_download_switch_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTOP_CONNECTOR_IMAGE_BLOCKS", "0")
    result = _call()
    assert isinstance(result, str)
    assert "图片内联展示已由配置关闭" in result


def test_download_non_image(fake_cli: dict[str, Any]) -> None:
    fake_cli["payload"] = b"not-an-image"
    result = _call()
    assert isinstance(result, str)
    assert "非图片格式" in result


def test_download_path_outside_temp_dir_is_not_read(
    tmp_path: Path, fake_cli: dict[str, Any]
) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(_TINY_PNG)
    fake_cli["saved_path"] = outside
    result = _call()
    assert isinstance(result, str)
    assert outside.read_bytes() == _TINY_PNG


def test_download_oversized_image(fake_cli: dict[str, Any]) -> None:
    fake_cli["payload"] = _TINY_PNG + b"\x00" * (feishu_cli.VISION_MAX_BYTES + 1)
    result = _call()
    assert isinstance(result, str)
    assert "上限" in result


def test_string_args_from_model_are_parsed(fake_cli: dict[str, Any]) -> None:
    result = _call(json.dumps({"message_id": "om_1", "file_key": "img_v2_1"}))
    assert isinstance(result, list)
    assert "--message_id" in fake_cli["argv"]
    assert "om_1" in fake_cli["argv"]
