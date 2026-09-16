"""Streaming status card for the hardened Feishu channel (CardKit v1).

One agent turn is projected onto ONE interactive card that is created at
turn start and updated in place: thinking → tool calls → final reply.
This replaces the "one separate message per progress note" delivery.

Design ported from lark-coding-agent-bridge / larksuite/openclaw-lark
(both MIT): a reducer-shaped state, full-card replacement throttled to
~10 fps, monotonic ``sequence`` per card, and a hard degradation rule —
when CardKit is unavailable the channel falls back to its normal plain
message delivery, never worse than stock.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from octop.i18n import tr
from octop.infra.utils.locale import Locale

logger = logging.getLogger(__name__)

TERMINAL_RUNNING = "running"
TERMINAL_DONE = "done"
TERMINAL_ERROR = "error"

_TOOL_STATUS_RUNNING = "running"
_TOOL_STATUS_DONE = "done"
_TOOL_STATUS_ERROR = "error"

# Pacing / sizing, from the battle-tested defaults of the two references:
# flush at ~10 fps, and cap streamed text well below the ~30k-per-element
# server limit so the serialized card JSON stays safely under it.
_FLUSH_INTERVAL_SECONDS = 0.1
_MAX_CARD_TEXT_CHARS = 8000


class CardKitError(Exception):
    """A CardKit API call failed (permission, quota, malformed card…)."""


def _api_base() -> str:
    from harness_gateway.channels.feishu import _API_BASE

    return str(_API_BASE)


@dataclass
class CardTool:
    """One tool invocation shown as a line in the card's tool panel."""

    name: str
    status: str = _TOOL_STATUS_RUNNING


@dataclass
class StreamCardState:
    """Mutable projection of one turn; rendered wholesale on every flush."""

    tools: list[CardTool] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    progress_note: str = ""
    final_text: str | None = None
    error_text: str = ""
    terminal: str = TERMINAL_RUNNING

    def display_text(self) -> str:
        if self.final_text is not None:
            return self.final_text
        return "\n".join(t for t in (s.strip() for s in self.texts) if t)

    def summary_key(self) -> str:
        if self.terminal == TERMINAL_DONE:
            return "feishu.card.summary_done"
        if self.terminal == TERMINAL_ERROR:
            return "feishu.card.summary_error"
        if any(t.status == _TOOL_STATUS_RUNNING for t in self.tools):
            return "feishu.card.status_tools"
        if self.display_text():
            return "feishu.card.status_writing"
        return "feishu.card.status_thinking"


def _notation(text: str) -> dict[str, Any]:
    return {"tag": "markdown", "content": text, "text_size": "notation"}


def _markdown(text: str) -> dict[str, Any]:
    return {"tag": "markdown", "content": text}


def _tool_line_icons() -> dict[str, str]:
    return {
        _TOOL_STATUS_RUNNING: "⏳",
        _TOOL_STATUS_DONE: "✅",
        _TOOL_STATUS_ERROR: "❌",
    }


def render_card(state: StreamCardState, locale: Locale) -> dict[str, Any]:
    """Project ``state`` onto one CardKit 2.0 card payload."""
    elements: list[dict[str, Any]] = []

    if state.tools:
        running = state.terminal == TERMINAL_RUNNING
        title = (
            tr("feishu.card.status_tools", locale)
            if running
            else tr("feishu.card.tools_done", locale, count=len(state.tools))
        )
        icons = _tool_line_icons()
        lines = "\n".join(f"- {icons[t.status]} {t.name}" for t in state.tools)
        elements.append(
            {
                "tag": "collapsible_panel",
                "expanded": running,
                "header": {"title": {"tag": "markdown", "content": f"🧰 **{title}**"}},
                "border": {"color": "grey", "corner_radius": "5px"},
                "vertical_spacing": "8px",
                "padding": "8px 8px 8px 8px",
                "elements": [_notation(lines)],
            }
        )

    text = state.display_text()
    overflow = len(text) > _MAX_CARD_TEXT_CHARS
    if overflow:
        text = f"{text[:_MAX_CARD_TEXT_CHARS]}\n\n{tr('feishu.card.overflow_note', locale)}"
    if text:
        elements.append(_markdown(text))

    if state.terminal == TERMINAL_ERROR and state.error_text:
        elements.append(_notation(f"⚠️ {state.error_text}"))
    elif state.terminal == TERMINAL_DONE and not elements:
        elements.append(_notation(tr("feishu.card.empty_reply", locale)))

    if state.terminal == TERMINAL_RUNNING:
        if state.progress_note:
            elements.append(_notation(state.progress_note))
        elements.append(_notation(f"**{tr(state.summary_key(), locale)}**"))

    return {
        "schema": "2.0",
        "config": {
            "streaming_mode": state.terminal == TERMINAL_RUNNING,
            "summary": {"content": tr(state.summary_key(), locale)},
        },
        "body": {"elements": elements},
    }


def _requires_overflow_fallback(state: StreamCardState) -> bool:
    return len(state.display_text()) > _MAX_CARD_TEXT_CHARS


class CardKitClient:
    """Thin CardKit v1 HTTP wrapper; sequence-managed, fail-fast."""

    def __init__(
        self,
        http: Any,
        token_provider: Callable[[], Awaitable[str]],
    ) -> None:
        self._http = http
        self._token_provider = token_provider

    async def _headers(self) -> dict[str, str]:
        token = await self._token_provider()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def _request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{_api_base()}{path}"
        async with self._http.request(
            method, url, headers=await self._headers(), json=payload
        ) as resp:
            raw = await resp.text()
        try:
            data = json.loads(raw)
        except ValueError as exc:
            # CardKit endpoints have been observed to answer non-JSON bodies
            # (e.g. settings); that is a transport failure, not a turn failure.
            raise CardKitError(
                f"cardkit {method} {path} returned non-JSON body: {raw[:120]!r}"
            ) from exc
        if not isinstance(data, dict) or resp.status != 200 or data.get("code") != 0:
            body = json.dumps(data, ensure_ascii=False)[:200] if data is not None else ""
            raise CardKitError(f"cardkit {method} {path} failed: status={resp.status} body={body}")
        return data

    async def create_card(self, card_json: dict[str, Any]) -> str:
        data = await self._request(
            "POST",
            "/cardkit/v1/cards",
            {"type": "card_json", "data": json.dumps(card_json, ensure_ascii=False)},
        )
        card_id = str(((data.get("data") or {}).get("card_id")) or "")
        if not card_id:
            raise CardKitError("cardkit card.create returned no card_id")
        return card_id

    async def update_card(self, card_id: str, card_json: dict[str, Any], sequence: int) -> None:
        await self._request(
            "PUT",
            f"/cardkit/v1/cards/{card_id}",
            {
                "card": {
                    "type": "card_json",
                    "data": json.dumps(card_json, ensure_ascii=False),
                },
                "sequence": sequence,
                "uuid": f"u_{card_id}_{sequence}",
            },
        )

    async def close_card(self, card_id: str, sequence: int, summary: str) -> None:
        settings = json.dumps(
            {"config": {"streaming_mode": False, "summary": {"content": summary}}},
            ensure_ascii=False,
        )
        await self._request(
            "PUT",
            f"/cardkit/v1/cards/{card_id}/settings",
            {"settings": settings, "sequence": sequence, "uuid": f"s_{card_id}_{sequence}"},
        )


class StreamCardSession:
    """One turn's card lifecycle.

    Mutation methods only touch local state (cheap, safe from any await
    point on the channel's event loop); a flusher task pushes renders at
    ~10 fps while the turn runs. Any CardKit failure degrades the session:
    ``active`` turns False and the caller falls back to plain delivery.
    """

    def __init__(
        self,
        *,
        http_provider: Callable[[], Awaitable[Any]],
        token_provider: Callable[[], Awaitable[str]],
        deliver: Callable[[str], Awaitable[None]],
        locale: Locale,
        client_factory: Callable[
            [Any, Callable[[], Awaitable[str]]], CardKitClient
        ] = CardKitClient,
    ) -> None:
        self.state = StreamCardState()
        self.locale = locale
        self._http_provider = http_provider
        self._token_provider = token_provider
        self._deliver = deliver
        self._client_factory = client_factory
        self._card_id = ""
        self._sequence = 0
        self._degraded = False
        self._closed = False
        self._dirty = False
        self._flusher: asyncio.Task[None] | None = None
        self._client: CardKitClient | None = None

    @property
    def active(self) -> bool:
        """True while the card can still receive updates."""
        return not self._degraded and bool(self._card_id)

    async def start(self) -> None:
        """Create the card entity and deliver the carrying message."""
        client = self._client_factory(await self._http_provider(), self._token_provider)
        card_id = await client.create_card(render_card(self.state, self.locale))
        content = json.dumps({"type": "card", "data": {"card_id": card_id}}, ensure_ascii=False)
        await self._deliver(content)
        self._card_id = card_id
        self._client = client
        self._flusher = asyncio.create_task(self._flush_loop())

    # -- mutations (sync, cheap) ------------------------------------------------

    def on_tool_start(self, name: str) -> None:
        self.state.tools.append(CardTool(name=name))
        self._dirty = True

    def on_tool_end(self, name: str, *, error: bool = False) -> None:
        for tool in reversed(self.state.tools):
            if tool.name == name and tool.status == _TOOL_STATUS_RUNNING:
                tool.status = _TOOL_STATUS_ERROR if error else _TOOL_STATUS_DONE
                break
        self._dirty = True

    def on_text(self, text: str) -> None:
        if text:
            self.state.texts.append(text)
            self._dirty = True

    def on_progress(self, note: str) -> None:
        if note:
            self.state.progress_note = note
            self._dirty = True

    # -- lifecycle --------------------------------------------------------------

    async def close(self, terminal: str, *, error_text: str = "") -> list[str]:
        """Finalize the card; return texts that must be sent as plain messages.

        Idempotent: the first successful finalization wins, so an exception
        escaping after ``close(TERMINAL_DONE)`` cannot be "upgraded" to an
        error card by a second close from an exception handler.
        """
        fallback: list[str] = []
        if self._flusher is not None:
            self._flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flusher
            self._flusher = None
        if self._closed or not self.active:
            return fallback
        self._closed = True

        self.state.terminal = terminal
        self.state.error_text = error_text
        overflow = _requires_overflow_fallback(self.state)
        if terminal != TERMINAL_DONE:
            # Replies already streamed to the card stay there; only errors on a
            # card that never showed text need a plain-message fallback.
            if not self.state.display_text() and error_text:
                fallback.append(error_text)
        elif overflow:
            fallback.append(self.state.display_text())

        assert self._client is not None
        try:
            self._sequence += 1
            await self._client.update_card(
                self._card_id, render_card(self.state, self.locale), self._sequence
            )
        except Exception:
            logger.warning("Feishu stream card final update failed", exc_info=True)
            self._degraded = True
            if self.state.display_text():
                fallback.append(self.state.display_text())
            return fallback
        try:
            self._sequence += 1
            await self._client.close_card(
                self._card_id, self._sequence, tr(self.state.summary_key(), self.locale)
            )
        except Exception:
            # Cosmetic: the final card content is already live; Feishu also
            # auto-closes streaming cards after 10 minutes.
            logger.debug("Feishu stream card close-settings failed", exc_info=True)
        return fallback

    async def _flush_loop(self) -> None:
        assert self._client is not None
        try:
            while True:
                await asyncio.sleep(_FLUSH_INTERVAL_SECONDS)
                if not self._dirty or not self.active:
                    continue
                self._dirty = False
                self._sequence += 1
                await self._client.update_card(
                    self._card_id, render_card(self.state, self.locale), self._sequence
                )
        except asyncio.CancelledError:
            raise
        except CardKitError:
            logger.warning("Feishu stream card degraded; falling back to plain messages")
            self._degraded = True
        except Exception:
            logger.debug("Feishu stream card flush loop crashed", exc_info=True)
            self._degraded = True
