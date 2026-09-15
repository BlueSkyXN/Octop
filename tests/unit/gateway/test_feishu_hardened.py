"""FeishuHardenedChannel inbound contract: mention gate, topics, reactions."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from harness_gateway.channels.feishu import FeishuConfig
from harness_gateway.group_context import GroupContextConfig, GroupContextManager
from harness_gateway.models import ImageContent, InboundMessage

from octop.infra.gateway.feishu_compat import (
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_TURN_TIMEOUT_SECONDS,
    build_feishu_hardened_channel,
)

BOT_OPEN_ID = "ou_bot"


def _config(*, group_enabled: bool = True) -> FeishuConfig:
    return FeishuConfig(
        app_id="cli_x",
        app_secret="s",
        group_context=GroupContextConfig(enabled=group_enabled),
    )


def _channel(config: FeishuConfig | None = None) -> Any:
    async def processor(_msg: InboundMessage) -> Any:
        yield _msg  # pragma: no cover - not driven in these tests

    ch = build_feishu_hardened_channel(
        processor, config or _config(), channel_id="ch1", tenant_id="ag1"
    )
    ch._running = True
    ch._bot_open_id = BOT_OPEN_ID
    return ch


def _sdk_event(
    *,
    chat_type: str = "group",
    text: str = "@_user_1 你好",
    mentions: list[tuple[str, str, str]] | None = None,
    message_id: str = "om_1",
    thread_id: str = "",
    sender_type: str = "user",
) -> SimpleNamespace:
    if mentions is None:
        mentions = [("@_user_1", BOT_OPEN_ID, "OctopBot")]
    return SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                message_id=message_id,
                message_type="text",
                content=json.dumps({"text": text}),
                chat_id="oc_1",
                chat_type=chat_type,
                thread_id=thread_id,
                mentions=[
                    SimpleNamespace(
                        key=key,
                        id=SimpleNamespace(open_id=open_id),
                        name=name,
                    )
                    for key, open_id, name in mentions
                ],
            ),
            sender=SimpleNamespace(
                sender_type=sender_type,
                sender_id=SimpleNamespace(open_id="ou_user"),
            ),
        ),
        header=SimpleNamespace(create_time="1700000000000"),
    )


def _enqueue_channel(ch: Any) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    ch._enqueue_callback = payloads.append
    return payloads


# ---------------------------------------------------------------------------
# _on_message_event → raw_payload → parse_inbound → group gate
# ---------------------------------------------------------------------------


def test_group_mention_message_passes_gate() -> None:
    ch = _channel()
    payloads = _enqueue_channel(ch)
    ch._on_message_event(_sdk_event(message_id="om_a"))
    assert len(payloads) == 1
    message = ch.parse_inbound(payloads[0])
    assert message.metadata["bot_mentioned"] is True
    assert message.channel_subject.chat_type == "group"
    assert message.channel_subject.subject_id == "oc_1"  # group: chat handle
    prepared = GroupContextManager(_config().group_context).prepare(message)
    assert prepared is not None


def test_group_plain_message_is_buffered_not_processed() -> None:
    ch = _channel()
    payloads = _enqueue_channel(ch)
    ch._on_message_event(_sdk_event(text="路过", mentions=[], message_id="om_b"))
    message = ch.parse_inbound(payloads[0])
    assert message.metadata["bot_mentioned"] is False
    prepared = GroupContextManager(_config().group_context).prepare(message)
    assert prepared is None


def test_at_all_is_not_a_bot_mention() -> None:
    ch = _channel()
    payloads = _enqueue_channel(ch)
    ch._on_message_event(
        _sdk_event(
            text="@_all_ 开会了",
            mentions=[("@_all_", "all", "所有人")],
            message_id="om_c",
        )
    )
    message = ch.parse_inbound(payloads[0])
    assert message.metadata["bot_mentioned"] is False
    assert GroupContextManager(_config().group_context).prepare(message) is None


def test_dm_passes_without_mention() -> None:
    ch = _channel()
    payloads = _enqueue_channel(ch)
    ch._on_message_event(_sdk_event(chat_type="p2p", text="你好", mentions=[], message_id="om_d"))
    message = ch.parse_inbound(payloads[0])
    assert "bot_mentioned" not in message.metadata
    assert message.channel_subject.chat_type == "direct"
    assert message.channel_subject.subject_id == "ou_user"


def test_group_mention_without_bot_identity_is_refused() -> None:
    ch = _channel()
    ch._bot_open_id = ""  # identity not ready
    payloads = _enqueue_channel(ch)
    ch._on_message_event(_sdk_event(message_id="om_e"))
    assert payloads == []  # refused, not silently processed


def test_group_plain_message_without_identity_still_observed() -> None:
    ch = _channel()
    ch._bot_open_id = ""
    payloads = _enqueue_channel(ch)
    ch._on_message_event(_sdk_event(text="hi", mentions=[], message_id="om_f"))
    assert len(payloads) == 1  # cannot be a mention → safe to observe


def test_bot_mention_placeholder_stripped_others_named() -> None:
    ch = _channel()
    message = ch.parse_inbound(
        {
            "message_id": "om_g",
            "message_type": "text",
            "content": json.dumps({"text": "@_user_1 看 @_user_2 的图"}),
            "chat_id": "oc_1",
            "chat_type": "group",
            "thread_id": "",
            "sender": {"sender_id": "ou_user", "sender_type": "user"},
            "create_time": "",
            "mentions": [
                {"key": "@_user_1", "open_id": BOT_OPEN_ID, "name": "OctopBot"},
                {"key": "@_user_2", "open_id": "ou_zhang", "name": "张三"},
            ],
            "bot_mentioned": True,
            "bot_mention_keys": ["@_user_1"],
        }
    )
    assert message.content[0].text == "看 @张三 的图"


def test_turn_budget_metadata_stamped() -> None:
    ch = _channel()
    message = ch.parse_inbound(
        {
            "message_id": "om_h",
            "message_type": "text",
            "content": "{}",
            "chat_id": "oc_1",
            "chat_type": "p2p",
            "sender": {"sender_id": "ou_user", "sender_type": "user"},
        }
    )
    assert message.metadata["turn_budget"] == {
        "timeout_s": DEFAULT_TURN_TIMEOUT_SECONDS,
        "progress_interval_s": DEFAULT_PROGRESS_INTERVAL_SECONDS,
    }


def test_unmentioned_group_media_not_persisted() -> None:
    ch = _channel()
    message = ch.parse_inbound(
        {
            "message_id": "om_i",
            "message_type": "image",
            "content": json.dumps({"image_key": "img_v2_1"}),
            "chat_id": "oc_1",
            "chat_type": "group",
            "sender": {"sender_id": "ou_user", "sender_type": "user"},
            "mentions": [],
            "bot_mentioned": False,
        }
    )
    assert any(isinstance(p, ImageContent) for p in message.content)
    assert ch.group_context_manager.should_persist_media(message) is False


# ---------------------------------------------------------------------------
# Topic isolation: one chat, two topics → separate buffers
# ---------------------------------------------------------------------------


def test_topic_threads_isolate_buffers() -> None:
    ch = _channel()
    payloads = _enqueue_channel(ch)
    ch._on_message_event(
        _sdk_event(text="话题一闲聊", mentions=[], message_id="om_t1", thread_id="omt_1")
    )
    ch._on_message_event(
        _sdk_event(text="话题二闲聊", mentions=[], message_id="om_t2", thread_id="omt_2")
    )
    manager = GroupContextManager(_config().group_context)
    m1 = ch.parse_inbound(payloads[0])
    m2 = ch.parse_inbound(payloads[1])
    assert m1.channel_subject.subject_id == "omt_1"
    assert m2.channel_subject.subject_id == "omt_2"
    assert manager.prepare(m1) is None
    assert manager.prepare(m2) is None
    assert set(manager._buffers) == {"omt_1", "omt_2"}

    # A mention in topic 2 pulls only topic 2's buffer.
    ch._on_message_event(
        _sdk_event(
            text="@_user_1 回答",
            mentions=[("@_user_1", BOT_OPEN_ID, "OctopBot")],
            message_id="om_t2m",
            thread_id="omt_2",
        )
    )
    m3 = ch.parse_inbound(payloads[2])
    prepared = manager.prepare(m3)
    assert prepared is not None
    assert prepared.group_context is not None
    texts = [m.text for m in prepared.group_context.messages]
    assert any("话题二" in t for t in texts)
    assert not any("话题一" in t for t in texts)


# ---------------------------------------------------------------------------
# Typing reaction lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typing_reaction_only_for_turn_starting_messages() -> None:
    ch = _channel()
    ch._main_loop = asyncio.get_running_loop()
    added: list[str] = []

    async def fake_add(message_id: str) -> None:
        added.append(message_id)

    ch._add_typing_reaction = fake_add  # type: ignore[method-assign]
    _enqueue_channel(ch)

    ch._on_message_event(_sdk_event(message_id="om_r1"))  # group + mention
    ch._on_message_event(
        _sdk_event(text="hi", mentions=[], message_id="om_r2")  # group, no mention
    )
    ch._on_message_event(
        _sdk_event(chat_type="p2p", text="dm", mentions=[], message_id="om_r3")  # DM
    )
    await asyncio.sleep(0.05)  # run_coroutine_threadsafe dispatch
    assert added == ["om_r1", "om_r3"]


@pytest.mark.asyncio
async def test_typing_reaction_removed_after_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_gateway.channels.feishu import FeishuChannel

    ch = _channel()
    removed: list[str] = []

    async def fake_remove(message_id: str) -> None:
        # Mirrors the real method: record + drop the stored reaction id.
        removed.append(message_id)
        ch._typing_reactions.pop(message_id, None)

    ch._remove_typing_reaction = fake_remove  # type: ignore[method-assign]
    ch._typing_reactions["om_r4"] = "re_1"

    async def boom(_self: Any, _raw: Any) -> None:
        raise RuntimeError("turn failed")

    monkeypatch.setattr(FeishuChannel, "handle_inbound", boom)
    with pytest.raises(RuntimeError):
        await ch.handle_inbound({"message_id": "om_r4"})
    assert removed == ["om_r4"]  # finally semantics: removed even on failure
    assert "om_r4" not in ch._typing_reactions
