"""Streaming status card: state, renderer, CardKit client, session, wiring."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from harness_gateway.channel import BaseChannel as harness_channel
from harness_gateway.channels import feishu
from harness_gateway.models import MessageEvent, MessageEventType, TextContent

from octop.infra.gateway import feishu_card
from octop.infra.gateway import feishu_compat as compat
from octop.infra.gateway.gateway import _probe_processor

# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def test_render_initial_running_card() -> None:
    card = feishu_card.render_card(feishu_card.StreamCardState(), "zh")
    assert card["schema"] == "2.0"
    assert card["config"]["streaming_mode"] is True
    assert "思考中" in card["config"]["summary"]["content"]
    # No tools, no text yet: only the footer status line.
    assert len(card["body"]["elements"]) == 1


def test_render_tools_panel_and_terminal() -> None:
    state = feishu_card.StreamCardState()
    state.tools.append(feishu_card.CardTool(name="搜索文档", status="done"))
    state.tools.append(feishu_card.CardTool(name="发送消息", status="running"))
    running = feishu_card.render_card(state, "zh")
    panel = running["body"]["elements"][0]
    assert panel["tag"] == "collapsible_panel"
    assert panel["expanded"] is True
    body = panel["elements"][0]["content"]
    assert "✅ 搜索文档" in body
    assert "⏳ 发送消息" in body
    assert running["config"]["summary"]["content"] == "正在调用工具"

    state.terminal = feishu_card.TERMINAL_DONE
    done = feishu_card.render_card(state, "zh")
    assert done["config"]["streaming_mode"] is False
    assert done["config"]["summary"]["content"] == "已完成"
    assert done["body"]["elements"][0]["expanded"] is False


def test_render_terminal_notes() -> None:
    empty = feishu_card.StreamCardState()
    empty.terminal = feishu_card.TERMINAL_DONE
    card = feishu_card.render_card(empty, "zh")
    assert any("未返回内容" in e.get("content", "") for e in card["body"]["elements"])

    failed = feishu_card.StreamCardState()
    failed.terminal = feishu_card.TERMINAL_ERROR
    failed.error_text = "boom"
    card = feishu_card.render_card(failed, "zh")
    assert any("⚠️ boom" in e.get("content", "") for e in card["body"]["elements"])


def test_render_overflow_truncates_with_note() -> None:
    state = feishu_card.StreamCardState()
    state.texts.append("长" * (feishu_card._MAX_CARD_TEXT_CHARS + 500))
    card = feishu_card.render_card(state, "zh")
    text = next(e["content"] for e in card["body"]["elements"] if e.get("tag") == "markdown")
    assert len(text) < feishu_card._MAX_CARD_TEXT_CHARS + 200
    assert "内容过长" in text


# ---------------------------------------------------------------------------
# CardKit client (fake HTTP session)
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status = status

    async def json(self, content_type: str | None = None) -> dict[str, Any]:
        return self._payload

    async def text(self) -> str:
        return json.dumps(self._payload)

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class RawTextResponse:
    """Simulates CardKit endpoints answering a non-JSON body."""

    def __init__(self, body: str, status: int = 200) -> None:
        self._body = body
        self.status = status

    async def json(self, content_type: str | None = None) -> dict[str, Any]:
        raise ValueError("non-JSON")

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> RawTextResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeHttp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.responses: list[dict[str, Any]] = [{"code": 0, "data": {"card_id": "card_1"}}]

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs["json"]))
        return FakeResponse(self.responses[min(len(self.calls) - 1, len(self.responses) - 1)])


def _client(http: FakeHttp) -> feishu_card.CardKitClient:
    async def token() -> str:
        return "t"

    return feishu_card.CardKitClient(http, token)


async def test_client_create_update_close_payloads() -> None:
    http = FakeHttp()
    http.responses.extend(
        [{"code": 0, "data": {}}, {"code": 0, "data": {}}, {"code": 0, "data": {}}]
    )
    client = _client(http)
    card_id = await client.create_card({"schema": "2.0"})
    assert card_id == "card_1"

    await client.update_card(card_id, {"schema": "2.0"}, 1)
    await client.close_card(card_id, 2, "已完成")

    create = http.calls[0]
    assert create[0] == "POST" and create[1].endswith("/cardkit/v1/cards")
    assert create[2]["type"] == "card_json"

    update = http.calls[1]
    assert update[0] == "PUT" and update[1].endswith(f"/cardkit/v1/cards/{card_id}")
    assert update[2]["sequence"] == 1
    assert update[2]["uuid"] == f"u_{card_id}_1"

    close = http.calls[2]
    assert close[1].endswith(f"/cardkit/v1/cards/{card_id}/settings")
    settings = json.loads(close[2]["settings"])
    assert settings["config"]["streaming_mode"] is False


async def test_client_raises_on_api_error() -> None:
    http = FakeHttp()
    http.responses[0] = {"code": 230002, "msg": "no permission"}
    client = _client(http)
    with pytest.raises(feishu_card.CardKitError):
        await client.create_card({"schema": "2.0"})


async def test_client_wraps_non_json_body_as_cardkit_error() -> None:
    http = FakeHttp()
    raw = RawTextResponse("true{}")

    def raw_response(method: str, url: str, **kwargs: Any) -> RawTextResponse:
        return raw

    http.request = raw_response  # type: ignore[method-assign]
    client = _client(http)
    with pytest.raises(feishu_card.CardKitError, match="non-JSON"):
        await client.create_card({"schema": "2.0"})


async def test_session_close_is_idempotent_and_not_upgraded_to_error() -> None:
    class FailingCloseClient(FakeCardKitClient):
        async def close_card(self, card_id: str, sequence: int, summary: str) -> None:
            raise feishu_card.CardKitError("settings returned non-JSON")

    client = FailingCloseClient()
    session, _ = _session(client)
    await session.start()
    session.state.final_text = "回复内容"

    # close(DONE): final update lands; the settings failure is cosmetic and
    # must neither raise nor drop the delivered text.
    fallback = await session.close(feishu_card.TERMINAL_DONE)
    assert fallback == []
    assert session.state.terminal == feishu_card.TERMINAL_DONE

    # A second close from an exception handler must not flip it to ERROR.
    fallback = await session.close(feishu_card.TERMINAL_ERROR, error_text="boom")
    assert fallback == []
    assert session.state.terminal == feishu_card.TERMINAL_DONE
    assert session._client.updates  # the final card update did go out


# ---------------------------------------------------------------------------
# Session lifecycle (fake client)
# ---------------------------------------------------------------------------


class FakeCardKitClient:
    def __init__(self) -> None:
        self.updates: list[int] = []
        self.closed: list[str] = []
        self.fail_updates = False

    async def create_card(self, card_json: dict[str, Any]) -> str:
        return "card_live"

    async def update_card(self, card_id: str, card_json: dict[str, Any], sequence: int) -> None:
        if self.fail_updates:
            raise feishu_card.CardKitError("update rejected")
        self.updates.append(sequence)

    async def close_card(self, card_id: str, sequence: int, summary: str) -> None:
        self.closed.append(summary)


def _session(
    client: FakeCardKitClient, locale: str = "zh"
) -> tuple[feishu_card.StreamCardSession, list[str]]:
    delivered: list[str] = []

    async def http() -> Any:
        return None

    async def token() -> str:
        return "t"

    async def deliver(content: str) -> None:
        delivered.append(content)

    session = feishu_card.StreamCardSession(
        http_provider=http,
        token_provider=token,
        deliver=deliver,
        locale=locale,  # type: ignore[arg-type]
        client_factory=lambda _h, _t: client,  # type: ignore[arg-type,misc]
    )
    return session, delivered


async def test_session_start_deliver_and_close() -> None:
    client = FakeCardKitClient()
    session, delivered = _session(client)
    await session.start()
    assert session.active
    assert delivered and json.loads(delivered[0])["data"]["card_id"] == "card_live"

    session.on_tool_start("搜索")
    session.on_text("部分输出")
    await asyncio.sleep(0.25)  # let the flusher push at least once
    assert client.updates

    session.state.final_text = "最终回复"
    fallback = await session.close(feishu_card.TERMINAL_DONE)
    assert fallback == []
    assert client.updates and client.updates == sorted(client.updates)
    assert client.closed and client.closed[0] == "已完成"
    # Flusher stopped: no further pushes.
    count = len(client.updates)
    session.on_text("more")
    await asyncio.sleep(0.25)
    assert len(client.updates) == count


async def test_session_close_overflow_returns_fallback() -> None:
    client = FakeCardKitClient()
    session, _ = _session(client)
    await session.start()
    session.state.final_text = "x" * (feishu_card._MAX_CARD_TEXT_CHARS + 10)
    fallback = await session.close(feishu_card.TERMINAL_DONE)
    assert fallback == [session.state.final_text]
    assert client.closed


async def test_session_degrades_on_update_failure() -> None:
    client = FakeCardKitClient()
    session, _ = _session(client)
    await session.start()
    client.fail_updates = True
    session.on_text("触发推送")
    for _ in range(50):
        if not session.active:
            break
        await asyncio.sleep(0.05)
    assert not session.active
    # Degraded close is a quiet no-op with no fallback for streamed text.
    fallback = await session.close(feishu_card.TERMINAL_DONE)
    assert fallback == []


# ---------------------------------------------------------------------------
# Hardened-channel wiring
# ---------------------------------------------------------------------------


def _channel(stream_card: bool = True) -> Any:
    return compat.build_feishu_hardened_channel(
        _probe_processor,
        feishu.FeishuConfig(app_id="t", app_secret="t"),
        stream_card=stream_card,
    )


@pytest.fixture
def live_session() -> tuple[Any, FakeCardKitClient]:
    client = FakeCardKitClient()

    async def http() -> Any:
        return None

    async def token() -> str:
        return "t"

    async def deliver(content: str) -> None:
        return None

    session = feishu_card.StreamCardSession(
        http_provider=http,
        token_provider=token,
        deliver=deliver,
        locale="zh",
        client_factory=lambda _h, _t: client,  # type: ignore[arg-type,misc]
    )
    session._card_id = "card_wired"
    session._client = client  # type: ignore[assignment]
    return session, client


async def test_wiring_text_and_tools_route_into_card(
    live_session: tuple[Any, FakeCardKitClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _client = live_session
    channel = _channel()
    subject = type("Subject", (), {"subject_id": "ou_1", "metadata": {}})()

    delivered: list[tuple[str, str]] = []

    async def fake_deliver(self: Any, subject: Any, *, msg_type: str, content: str) -> None:
        delivered.append((msg_type, content))

    async def fake_send_text(self: Any, subject: Any, text: str) -> None:
        delivered.append(("post", text))

    monkeypatch.setattr(type(channel), "_deliver", fake_deliver)
    monkeypatch.setattr(type(channel).__mro__[1], "_send_text", fake_send_text, raising=False)

    channel._card_sessions[subject.subject_id] = session

    # Progress MESSAGE events are absorbed into the card instead of sent.
    progress = MessageEvent(
        type=MessageEventType.MESSAGE,
        content=[TextContent(text="仍在处理中…")],
        metadata={"progress": True},
    )
    await channel._deliver_event(subject, progress)
    assert session.state.progress_note == "仍在处理中…"

    await channel._on_tool_start(subject, MessageEvent.tool_start("搜索文档"))
    await channel._on_tool_end(subject, MessageEvent.tool_end("搜索文档"))
    assert [t.name for t in session.state.tools] == ["搜索文档"]
    assert session.state.tools[0].status == "done"

    await channel._send_text(subject, "中途文本")
    assert "中途文本" in session.state.texts
    assert delivered == []  # nothing hit the network

    # Without an active session the stock path is used.
    channel._card_sessions.clear()
    await channel._send_text(subject, "普通回复")
    assert delivered == [("post", "普通回复")]


async def test_maybe_start_card_session_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _channel(stream_card=False)
    subject = type("Subject", (), {"subject_id": "ou_1", "metadata": {}})()
    assert await channel._maybe_start_card_session(subject) is None


async def test_maybe_start_card_session_degrades_on_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    channel = _channel()
    subject = type("Subject", (), {"subject_id": "ou_1", "metadata": {}})()

    class FailingSession:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def start(self) -> None:
            raise feishu_card.CardKitError("no permission")

    monkeypatch.setattr(feishu_card, "StreamCardSession", FailingSession)
    with caplog.at_level("WARNING"):
        assert await channel._maybe_start_card_session(subject) is None
    assert any("stream card unavailable" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Typing-reaction removal bookkeeping
# ---------------------------------------------------------------------------


def test_pending_typing_ids_from_parsed_message() -> None:
    channel = _channel(stream_card=False)
    channel._typing_sessions["ou_sender"] = ["om_1", "om_2"]

    class Parsed:
        metadata = {
            "message_id": "om_2",
            "chat_id": "oc_1",
            "chat_type": "p2p",
            "to_handle": "ou_sender",
        }

    assert channel._pending_typing_ids(Parsed()) == ["om_2", "om_1"]
    assert "ou_sender" not in channel._typing_sessions


def test_pending_typing_ids_from_raw_dict_group() -> None:
    channel = _channel(stream_card=False)
    channel._typing_sessions["oc_group"] = ["om_a"]

    raw = {
        "message_id": "om_a",
        "chat_id": "oc_group",
        "chat_type": "group",
        "thread_id": "",
        "sender": {"sender_id": "ou_x", "sender_type": "user"},
    }
    assert channel._pending_typing_ids(raw) == ["om_a"]


def test_pending_typing_ids_thread_keyed() -> None:
    channel = _channel(stream_card=False)
    channel._typing_sessions["omt_topic"] = ["om_t1", "om_t2"]

    class Parsed:
        metadata = {
            "message_id": "om_t2",
            "chat_id": "oc_group",
            "chat_type": "group",
            "thread_id": "omt_topic",
            "to_handle": "oc_group",
        }

    ids = channel._pending_typing_ids(Parsed())
    assert ids == ["om_t2", "om_t1"]
    assert channel._typing_sessions == {}


async def test_remove_typing_reaction_called_for_every_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _channel(stream_card=False)
    removed: list[str] = []

    async def fake_remove(message_id: str) -> None:
        removed.append(message_id)

    async def noop_handle(self: Any, raw_payload: object) -> None:
        return None

    monkeypatch.setattr(channel, "_remove_typing_reaction", fake_remove)
    monkeypatch.setattr(harness_channel, "handle_inbound", noop_handle, raising=False)
    channel._typing_sessions["ou_sender"] = ["om_1"]

    class Parsed:
        metadata = {"message_id": "om_9", "chat_type": "p2p", "to_handle": "ou_sender"}

    await channel.handle_inbound(Parsed())
    assert removed == ["om_9", "om_1"]
