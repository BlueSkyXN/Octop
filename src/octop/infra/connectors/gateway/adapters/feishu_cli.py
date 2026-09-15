"""Feishu / Lark official CLI gateway — category tools wrapping ``lark-cli``."""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

from octop.infra.connectors.gateway.cli_dirs import resolve_cli_config_key
from octop.infra.connectors.gateway.cli_runner import resolve_binary, run_cli
from octop.infra.connectors.gateway.feishu_creds import prepare_feishu_cli_env
from octop.infra.gateway.media.attachment_hints import (
    VISION_MAX_BYTES,
    sniff_image_media_type,
)
from octop.infra.utils.paths import PathLayout

logger = logging.getLogger(__name__)

_KIND = "feishu-cli"
# MCP tool name → lark-cli domain
_DOMAIN_BY_TOOL = {
    "doc": "docs",
    "base": "base",
    "calendar": "calendar",
    "im": "im",
}
# Shortcuts that only accept ``--as user`` (lark-cli rejects bot).
_USER_ONLY_SHORTCUTS = frozenset(
    {
        ("docs", "+search"),
    }
)

# Explicit message-resource download shortcut: the only path allowed to turn
# downloaded bytes into an image block for the model.
_MESSAGE_RESOURCE_DOWNLOAD = "+messages-resources-download"
_IMAGE_BLOCKS_ENV = "OCTOP_CONNECTOR_IMAGE_BLOCKS"
_DOWNLOAD_TIMEOUT_S = 120.0

_USER_AUTH_REQUIRED_MSG = (
    "文档搜索需要先完成飞书账号授权。"
    "请用户打开 Octop「连接器 → 飞书 CLI」，点击「登录授权」并在弹出页完成授权，然后点「我已授权」。"
    "禁止建议、生成或执行任何终端命令（包括任何 CLI）。"
)
_MISSING_SEARCH_SCOPE_MSG = (
    "文档搜索权限尚未授予当前登录用户。"
    "请用户打开 Octop「连接器 → 飞书 CLI」，点击「登录授权」重新授权；"
    "若仍失败，再到飞书开放平台确认已开通并发布「搜索云文档」权限。"
    "禁止建议、生成或执行任何终端命令（包括任何 CLI）。"
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "doc",
        "description": (
            "Feishu docs via Octop Connectors (gateway). "
            "method examples: '+search', '+fetch', '+create'. "
            "On auth/permission errors: tell the user to open Octop "
            "Connectors → 飞书 CLI → 登录授权. "
            "NEVER suggest or run shell/CLI commands. NEVER invent auth login commands."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "args": {"type": "object"},
            },
            "required": ["method"],
        },
    },
    {
        "name": "base",
        "description": "Feishu Base (多维表格) via lark-cli base <method…>",
        "inputSchema": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "args": {"type": "object"},
            },
            "required": ["method"],
        },
    },
    {
        "name": "calendar",
        "description": "Feishu calendar via lark-cli calendar <method…>",
        "inputSchema": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "args": {"type": "object"},
            },
            "required": ["method"],
        },
    },
    {
        "name": "im",
        "description": "Feishu messenger via lark-cli im <method…>",
        "inputSchema": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "args": {"type": "object"},
            },
            "required": ["method"],
        },
    },
    {
        "name": "help",
        "description": "Show lark-cli help for a domain (doc|base|calendar|im)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "doc | base | calendar | im",
                },
            },
            "required": ["category"],
        },
    },
]


def list_tools() -> list[dict[str, Any]]:
    return TOOLS


def call_tool(creds: dict[str, Any], name: str, args: dict[str, Any]) -> str | list[dict[str, Any]]:
    binary = resolve_binary("lark-cli")
    try:
        if name == "help":
            env = _prepare_env(creds)
            category = str(args.get("category") or "").strip()
            domain = _DOMAIN_BY_TOOL.get(category)
            if domain is None:
                raise ValueError("category must be one of: doc, base, calendar, im")
            return run_cli([binary, domain, "--help"], env=env)
        domain = _DOMAIN_BY_TOOL.get(name)
        if domain is None:
            raise ValueError(f"unknown Feishu CLI tool: {name}")
        method = str(args.get("method") or "").strip()
        if not method:
            raise ValueError("method is required")
        tokens = shlex.split(method)
        if not tokens:
            raise ValueError("method is required")
        identity = _resolve_identity(domain, tokens[0], creds)
        if (domain, tokens[0]) in _USER_ONLY_SHORTCUTS and str(
            creds.get("default_as") or ""
        ).strip().lower() != "user":
            # Octop has not recorded a completed user OAuth yet.
            raise ValueError(_USER_AUTH_REQUIRED_MSG)
        env = _prepare_env(creds, prefer_identity=identity)
        if (domain, tokens[0]) in _USER_ONLY_SHORTCUTS and not _has_user_scope(
            binary, env, "search:docs:read"
        ):
            raise ValueError(_MISSING_SEARCH_SCOPE_MSG)
        raw_args = args.get("args")
        if isinstance(raw_args, str):
            # Models trained against the MCP schema may send the object arg as
            # a JSON string; parse it back instead of dropping it.
            with contextlib.suppress(json.JSONDecodeError):
                raw_args = json.loads(raw_args)
        payload: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        argv = _build_argv(binary, domain, method, payload, identity=identity)
        if domain == "im" and tokens[0] == _MESSAGE_RESOURCE_DOWNLOAD:
            return _run_message_resource_download(argv, env)
        return run_cli(argv, env=env)
    except ValueError as exc:
        raise ValueError(_humanize_cli_error(str(exc))) from exc


def _image_blocks_enabled() -> bool:
    raw = os.environ.get(_IMAGE_BLOCKS_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _parse_json_output(out: str) -> dict[str, Any]:
    text = out.strip()
    if not text.startswith("{"):
        if "{" in text:
            text = text[text.find("{") :]
        else:
            return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _run_message_resource_download(
    argv: list[str], env: dict[str, str]
) -> str | list[dict[str, Any]]:
    """Run the download shortcut inside an op-owned temp dir and inline images.

    Only files inside this operation's temp directory are trusted; size comes
    from ``os.stat`` and the format from magic bytes — never from the JSON
    fields or the file name.
    """
    tmp = Path(tempfile.mkdtemp(prefix="octop-feishu-dl-"))
    try:
        out = run_cli(argv, env=env, timeout_s=_DOWNLOAD_TIMEOUT_S, cwd=str(tmp))
        payload = _parse_json_output(out)
        data = payload.get("data")
        saved = str(
            payload.get("saved_path")
            or (data.get("saved_path") if isinstance(data, dict) else "")
            or ""
        )
        if not saved:
            return out
        path = Path(saved)
        if not path.is_absolute():
            path = tmp / path
        try:
            path = path.resolve()
            path.relative_to(tmp.resolve())
        except (ValueError, OSError):
            logger.warning("feishu download reported a path outside its temp dir: %s", saved[:64])
            return out
        if not path.is_file():
            return out
        size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(16)
        mime = sniff_image_media_type(head)
        note = f"已下载消息资源 {path.name}（{size} 字节）"
        if mime is None:
            return f"{note}，非图片格式，未内联展示。"
        if not _image_blocks_enabled():
            return f"{note}，图片内联展示已由配置关闭。"
        if size > VISION_MAX_BYTES:
            return f"{note}，图片超过 {VISION_MAX_BYTES} 字节上限，未内联展示。"
        encoded = base64.b64encode(path.read_bytes()).decode()
        return [
            {"type": "text", "text": f"已下载图片资源（{size} 字节，{mime}），图片见下方输入块。"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _has_user_scope(binary: str, env: dict[str, str], scope: str) -> bool:
    try:
        run_cli([binary, "auth", "check", "--scope", scope], env=env, timeout_s=30.0)
        return True
    except ValueError:
        return False


def _resolve_identity(domain: str, shortcut: str, creds: dict[str, Any]) -> str | None:
    if (domain, shortcut) in _USER_ONLY_SHORTCUTS:
        return "user"
    return _identity_flag(creds)


def _identity_flag(creds: dict[str, Any]) -> str | None:
    default_as = str(creds.get("default_as") or "bot").strip().lower()
    if default_as == "user":
        return "user"
    if default_as == "bot":
        return "bot"
    return None


def _build_argv(
    binary: str,
    domain: str,
    method: str,
    args: dict[str, Any],
    *,
    identity: str | None = None,
) -> list[str]:
    tokens = shlex.split(method)
    if not tokens:
        raise ValueError("method is required")
    argv = [binary, domain, *tokens, "--format", "json"]
    if identity:
        argv.extend(["--as", identity])
    if not args:
        return argv
    if tokens[0].startswith("+"):
        for key, value in args.items():
            if value is None:
                continue
            argv.extend([f"--{key}", str(value)])
        return argv
    argv.extend(["--params", json.dumps(args, ensure_ascii=False)])
    return argv


def _humanize_cli_error(message: str) -> str:
    text = (message or "").strip()
    if text in {_MISSING_SEARCH_SCOPE_MSG, _USER_AUTH_REQUIRED_MSG}:
        return text
    if text.startswith("文档搜索权限尚未授予") or text.startswith("文档搜索缺少权限"):
        return _MISSING_SEARCH_SCOPE_MSG
    if text.startswith("文档搜索需要"):
        return _USER_AUTH_REQUIRED_MSG
    lower = text.lower()
    if "search:docs:read" in lower or "missing required scope" in lower:
        return _MISSING_SEARCH_SCOPE_MSG
    if (
        "only supports: user" in lower
        or "--as bot is not supported" in lower
        or "lark-cli auth login" in lower
        or "run `lark-cli auth login" in lower
    ):
        return _USER_AUTH_REQUIRED_MSG
    if "command not found" in lower or "未找到命令" in text:
        return (
            "主机上的飞书 CLI 未安装或不在 PATH。"
            "请在连接器抽屉中使用「安装 CLI」。"
            "禁止建议或执行任何终端命令。"
        )
    return text


def _prepare_env(
    creds: dict[str, Any],
    *,
    prefer_identity: str | None = None,
) -> dict[str, str]:
    app_id = str(creds.get("app_id") or creds.get("client_id") or "").strip()
    app_secret = str(creds.get("app_secret") or creds.get("api_key") or "").strip()
    instance_key = resolve_cli_config_key(creds)
    default_as = str(prefer_identity or creds.get("default_as") or "bot").strip().lower()
    if default_as != "user":
        default_as = "bot"
    config_dir = PathLayout.from_env().ensure_connector_cli_instance_dir(_KIND, instance_key)
    _binary, env = prepare_feishu_cli_env(
        config_dir,
        app_id=app_id,
        app_secret=app_secret,
        default_as=default_as,
    )
    return env


def probe_credentials(creds: dict[str, Any]) -> None:
    binary = resolve_binary("lark-cli")
    env = _prepare_env(creds)
    out = run_cli([binary, "auth", "status", "--json"], env=env, timeout_s=60.0)
    try:
        payload = json.loads(out) if out.lstrip().startswith("{") else {}
    except json.JSONDecodeError:
        payload = {}
    # auth status may print a leading OK line before JSON.
    if not payload and "{" in out:
        try:
            payload = json.loads(out[out.find("{") :])
        except json.JSONDecodeError:
            payload = {}
    identities = payload.get("identities") if isinstance(payload, dict) else None
    bot = identities.get("bot") if isinstance(identities, dict) else None
    user = identities.get("user") if isinstance(identities, dict) else None
    # App ID / App Secret must always mint a working bot identity.
    if not isinstance(bot, dict) or not bot.get("available"):
        raise ValueError(
            "飞书 App ID / App Secret 无效，或应用未启用机器人能力。请核对开放平台凭证后再试。"
        )
    default_as = str(creds.get("default_as") or "bot").strip().lower()
    if default_as == "user" and (not isinstance(user, dict) or not user.get("available")):
        raise ValueError(
            "飞书用户身份未就绪。请先在「连接器 → 飞书 CLI」完成登录授权，"
            "否则文档搜索等仅用户身份可用的能力无法使用。"
        )
