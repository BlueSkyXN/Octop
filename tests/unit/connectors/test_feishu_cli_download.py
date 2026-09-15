"""Feishu CLI download branch: tiny PNG → image blocks, switch degradation."""

from __future__ import annotations

import base64
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from octop.infra.connectors.gateway.adapters import feishu_cli

# 1×1 transparent PNG.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _fake_lark_cli(
    tmp_path: Path, *, saved_name: str = "res.png", payload: bytes = _TINY_PNG
) -> Path:
    """A stand-in lark-cli that writes the resource next to its cwd and reports it."""
    payload_file = tmp_path / "payload.bin"
    payload_file.write_bytes(payload)
    script = tmp_path / "fake-lark-cli"
    script.write_text(
        "#!/bin/sh\n"
        f'cp "{payload_file}" "$PWD/{saved_name}"\n'
        f'printf \'{{"saved_path": "%s"}}\' "$PWD/{saved_name}"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


@pytest.fixture(autouse=True)
def _fake_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("OCTOP_CONNECTOR_IMAGE_BLOCKS", "1")
    script = _fake_lark_cli(tmp_path)
    monkeypatch.setattr(feishu_cli, "resolve_binary", lambda _name: str(script))
    # Skip credential/env preparation; the fake CLI only needs PATH.
    monkeypatch.setattr(
        feishu_cli,
        "_prepare_env",
        lambda creds, **_kw: {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    return script


_CREDS: dict[str, Any] = {"app_id": "cli_x", "app_secret": "s", "default_as": "bot"}


def _call() -> Any:
    return feishu_cli.call_tool(
        _CREDS,
        "im",
        {
            "method": "+messages-resources-download",
            "args": {"message_id": "om_1", "file_key": "img_v2_1"},
        },
    )


def test_download_returns_image_block() -> None:
    result = _call()
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert "已下载图片资源" in result[0]["text"]
    assert result[1]["type"] == "image_url"
    url = result[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == _TINY_PNG


def test_download_switch_off_degrades_to_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTOP_CONNECTOR_IMAGE_BLOCKS", "0")
    result = _call()
    assert isinstance(result, str)
    assert "图片内联展示已由配置关闭" in result


def test_download_non_image_returns_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _fake_lark_cli(tmp_path, saved_name="res.bin", payload=b"not-an-image")
    monkeypatch.setattr(feishu_cli, "resolve_binary", lambda _name: str(script))
    result = _call()
    assert isinstance(result, str)
    assert "非图片格式" in result


def test_download_path_outside_temp_dir_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved_path escaping the op temp dir must not be inlined."""
    outside = tmp_path / "outside.png"
    outside.write_bytes(_TINY_PNG)
    script = tmp_path / "evil-lark-cli"
    script.write_text(
        f'#!/bin/sh\nprintf \'{{"saved_path": "{outside}"}}\'\n',
        encoding="utf-8",
    )
    script.chmod(stat.S_IRWXU)
    monkeypatch.setattr(feishu_cli, "resolve_binary", lambda _name: str(script))
    result = _call()
    assert isinstance(result, str)
    assert "image_url" not in result
    assert outside.is_file()  # untouched


def test_download_oversized_image_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from octop.infra.gateway.media.attachment_hints import VISION_MAX_BYTES

    big = _TINY_PNG + b"\x00" * (VISION_MAX_BYTES + 1)
    script = _fake_lark_cli(tmp_path, saved_name="big.png", payload=big)
    monkeypatch.setattr(feishu_cli, "resolve_binary", lambda _name: str(script))
    result = _call()
    assert isinstance(result, str)
    assert "上限" in result


def test_string_args_from_model_are_parsed() -> None:
    """Models may send the object arg as a JSON string — it must survive."""
    result = feishu_cli.call_tool(
        _CREDS,
        "im",
        {
            "method": "+messages-resources-download",
            "args": json.dumps({"message_id": "om_1", "file_key": "img_v2_1"}),
        },
    )
    assert isinstance(result, list)  # parsed → download branch ran
