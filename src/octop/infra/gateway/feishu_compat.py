"""Feishu compatibility layer: connection teardown fix + hardened channel.

Two concerns live here, both pinned to harness-gateway 0.9.7 / lark-oapi 1.7.3:

1. ``ensure_feishu_ws_stop_fix`` — 0.9.7 calls ``stop()``, which lark-oapi
   1.7.3 does not provide. Cancel the SDK receiver, close the socket, then
   release worker tasks and its loop. Remove when harness-gateway handles the
   lifecycle itself.
2. :class:`FeishuHardenedChannel` — the stock FeishuChannel always adds a
   "Typing" reaction it never removes, drops mention metadata, and leaves
   ``channel_subject.chat_type`` empty, so GroupContextManager cannot apply
   the mention gate. The subclass fixes the inbound contract (mentions into
   ``raw_payload``, ``bot_mentioned`` flag with @all excluded, chat_type set,
   bot-mention placeholder stripped) and manages the typing reaction as a
   try/finally around the turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from harness_gateway.channel import ChannelCredentialsError, MessageProcessor
from harness_gateway.models import ContentPart, MessageEventType, TextContent

from octop.infra.gateway import feishu_card

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_octop_feishu_ws_stop_fix"
_STOP_TIMEOUT_SECONDS = 5.0

# Hardened channel defaults / knobs.
DEFAULT_TURN_TIMEOUT_SECONDS = 600.0
DEFAULT_PROGRESS_INTERVAL_SECONDS = 60.0
_BOT_INFO_RETRY_SECONDS = 300.0
_REACTION_ACK_TIMEOUT_SECONDS = 3.0
# Feishu represents @all with the literal open_id "all" — never a bot mention.
_ALL_MENTION_OPEN_ID = "all"


def _open_id_from_bot_info(data: Any) -> str:
    """Extract the bot's open_id from a ``GET /bot/v3/info`` payload.

    The API puts the payload under the top-level ``bot`` key (no ``data``
    wrapper); ``data`` is accepted as a defensive fallback.
    """
    if not isinstance(data, dict) or data.get("code") != 0:
        return ""
    info = data.get("bot") or data.get("data") or {}
    if not isinstance(info, dict):
        return ""
    return str(info.get("open_id") or "")


async def probe_feishu_credentials(config: dict[str, Any], processor: MessageProcessor) -> None:
    """Reuse token validation without registering a WebSocket event receiver."""
    from harness_gateway.channels.feishu import FeishuChannel, FeishuConfig

    channel_config = FeishuConfig.from_dict(config)
    missing = channel_config.missing_credentials()
    if missing:
        raise ChannelCredentialsError("feishu", missing)
    channel = FeishuChannel(processor, config=channel_config)
    try:
        await channel._refresh_token()
    finally:
        await channel._close_http()


def ensure_feishu_ws_stop_fix() -> bool:
    """Install the teardown patch once, unless the SDK provides stop()."""
    import lark_oapi as lark
    from harness_gateway.channels.feishu import FeishuChannel

    if getattr(FeishuChannel, _PATCH_MARKER, False) or hasattr(lark.ws.Client, "stop"):
        return False

    FeishuChannel._run_ws_thread = _fixed_run_ws_thread  # type: ignore[method-assign]
    FeishuChannel._stop_ws_client = _fixed_stop_ws_client  # type: ignore[method-assign]
    setattr(FeishuChannel, _PATCH_MARKER, True)
    return True


async def _run_ws_client(channel: Any) -> None:
    from lark_oapi.ws.exception import ClientException

    client = channel._ws_client
    receive = client._receive_message_loop

    async def receive_messages() -> None:
        channel._ws_receive_task = asyncio.current_task()
        await receive()

    client._receive_message_loop = receive_messages
    try:
        await client._connect()
    except ClientException:
        raise
    except Exception:
        await client._disconnect()
        if not client._auto_reconnect:
            raise
        await client._reconnect()
    await client._ping_loop()


async def _close_ws_client(client: Any, receive_task: asyncio.Task[Any] | None) -> None:
    if receive_task is not None:
        receive_task.cancel()
        await asyncio.gather(receive_task, return_exceptions=True)

    connection = client._conn
    try:
        async with asyncio.timeout(_STOP_TIMEOUT_SECONDS):
            await client._disconnect()
    except Exception:
        # A failed close handshake must not leave the socket alive after loop.close().
        if connection is not None:
            connection.transport.abort()
        logger.warning("Feishu WebSocket disconnect failed", exc_info=True)
    finally:
        # Legacy transports need their reader/close tasks alive during the handshake.
        tasks = asyncio.all_tasks() - {asyncio.current_task()}
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _fixed_run_ws_thread(self: Any) -> None:
    import lark_oapi.ws.client as ws_client_module

    client = self._ws_client
    try:
        with asyncio.Runner() as runner:
            loop = runner.get_loop()
            if client is None:
                return
            self._ws_receive_task = None
            task = loop.create_task(_run_ws_client(self))
            self._ws_task = task
            self._ws_loop = loop
            # stop() can run before the worker has published its loop.
            if not self._running:
                task.cancel()
            else:
                ws_client_module.loop = loop
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Feishu WebSocket thread failed")
            finally:
                client._auto_reconnect = False
                runner.run(_close_ws_client(client, self._ws_receive_task))
    finally:
        self._ws_client = None
        self._ws_loop = None
        self._ws_task = None
        self._ws_receive_task = None
        self._ws_thread = None
        self._ws_session_id = None


async def _fixed_stop_ws_client(self: Any) -> None:
    self._running = False
    client = self._ws_client
    if client is not None:
        client._auto_reconnect = False
    loop = getattr(self, "_ws_loop", None)
    task = getattr(self, "_ws_task", None)
    if loop is not None and task is not None:
        with contextlib.suppress(RuntimeError):  # worker may already have closed the loop
            loop.call_soon_threadsafe(task.cancel)

    thread = getattr(self, "_ws_thread", None)
    if thread is not None:
        await asyncio.get_running_loop().run_in_executor(
            None, thread.join, _STOP_TIMEOUT_SECONDS + 1.0
        )
        if thread.is_alive():
            # Keep references until the worker handles cancellation and finishes cleanup.
            logger.warning("Feishu WebSocket worker is still stopping")


# =============================================================================
# Hardened Feishu channel
# =============================================================================


def _extract_mentions(message: Any) -> list[dict[str, str]]:
    """Normalize lark-oapi ``MentionEvent`` entries into plain dicts."""
    out: list[dict[str, str]] = []
    for m in getattr(message, "mentions", None) or []:
        user_id = getattr(m, "id", None)
        out.append(
            {
                "key": str(getattr(m, "key", "") or ""),
                "open_id": str(getattr(user_id, "open_id", "") or ""),
                "name": str(getattr(m, "name", "") or ""),
            }
        )
    return out


def _replace_mention_placeholders(
    content: list[ContentPart],
    mentions: list[dict[str, str]],
    bot_mention_keys: list[str],
) -> list[ContentPart]:
    """Rewrite ``@_user_N`` placeholders: bot keys removed, others get names."""
    bot_keys = set(bot_mention_keys)
    replacements = {
        m["key"]: ("" if m["key"] in bot_keys else f"@{m['name'] or m['open_id'][:12]}")
        for m in mentions
        if m["key"]
    }
    if not replacements:
        return content
    new_parts: list[ContentPart] = []
    for part in content:
        if isinstance(part, TextContent):
            text = part.text
            for key, label in replacements.items():
                if key in text:
                    replacement = f"{label} " if label else ""
                    text = text.replace(f"{key} ", replacement)
                    text = text.replace(key, label)
                    text = text.replace("  ", " ").strip()
            new_parts.append(TextContent(text=text))
        else:
            new_parts.append(part)
    return new_parts


def _build_feishu_hardened_channel() -> type[Any]:
    from harness_gateway.channels.feishu import FeishuChannel, FeishuConfig

    class _FeishuHardenedChannel(FeishuChannel):
        """FeishuChannel with mention gating, typing-reaction lifecycle, turn budget.

        Register via ``ChannelManager.add_channel(instance)`` (see
        ``Gateway._register_channel``) — the instance path skips
        ``add_channel``'s kwargs plumbing, so all parameters are explicit here.
        """

        def __init__(
            self,
            processor: MessageProcessor,
            config: FeishuConfig,
            *,
            channel_id: str | None = None,
            tenant_id: str | None = None,
            debounce_seconds: float = 0.0,
            constraints: Any = None,
            turn_timeout_s: float = DEFAULT_TURN_TIMEOUT_SECONDS,
            progress_interval_s: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
            stream_card: bool = False,
        ) -> None:
            super().__init__(
                processor,
                config,
                channel_id=channel_id,
                tenant_id=tenant_id,
                debounce_seconds=debounce_seconds,
                constraints=constraints,
            )
            self._hardened_turn_timeout_s = float(turn_timeout_s)
            self._hardened_progress_interval_s = float(progress_interval_s)
            self._stream_card_enabled = bool(stream_card)
            self._card_sessions: dict[str, Any] = {}
            self._bot_open_id = ""
            self._bot_identity_task: asyncio.Task[None] | None = None
            self._typing_reactions: dict[str, str] = {}
            # message ids per session, swept when that session's turn ends —
            # batched turns merge to the first id, so removal must be by session.
            self._typing_sessions: dict[str, list[str]] = {}
            # Narrow the passive-media promise: an un-mentioned group message
            # never triggers a download, even under mention_recent visibility.
            inner_should_persist = self.group_context_manager.should_persist_media

            def _should_persist(message: Any) -> bool:
                subject = message.channel_subject
                if (
                    subject is not None
                    and getattr(subject, "chat_type", "") == "group"
                    and not message.metadata.get("bot_mentioned")
                ):
                    return False
                return inner_should_persist(message)

            self.group_context_manager.should_persist_media = _should_persist  # type: ignore[method-assign]

        # ------------------------------------------------------------------
        # Lifecycle
        # ------------------------------------------------------------------

        async def start(self) -> None:
            await super().start()
            self._bot_identity_task = asyncio.create_task(
                self._ensure_bot_identity(), name="feishu-bot-identity"
            )

        async def stop(self) -> None:
            task = self._bot_identity_task
            self._bot_identity_task = None
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await super().stop()

        async def _ensure_bot_identity(self) -> None:
            """Fetch the bot's own open_id, retrying until it succeeds."""
            from harness_gateway.channels.feishu import _API_BASE

            while self._running:
                try:
                    http = await self._ensure_http()
                    headers = await self._get_auth_headers()
                    async with http.get(f"{_API_BASE}/bot/v3/info", headers=headers) as resp:
                        data = await resp.json()
                    open_id = _open_id_from_bot_info(data)
                    if open_id:
                        self._bot_open_id = open_id
                        logger.info("Feishu bot identity ready (open_id=%s…)", open_id[:8])
                        return
                    logger.warning("Feishu bot info response unusable: code=%s", data.get("code"))
                except Exception:
                    logger.warning("Feishu bot identity fetch failed; retrying", exc_info=True)
                await asyncio.sleep(_BOT_INFO_RETRY_SECONDS)

        # ------------------------------------------------------------------
        # Inbound (lark-oapi thread)
        # ------------------------------------------------------------------

        def _on_message_event(self, data: Any) -> None:
            """Upstream handler + mention extraction, gate, gated typing reaction.

            Mirrors harness-gateway 0.9.7 ``FeishuChannel._on_message_event``
            (dedup, bot-sender skip, enqueue callback contract); changes:
            mentions/bot_mentioned/bot_mention_keys into ``raw_payload``,
            typing reaction only for messages that will start a turn, and
            group messages with unknown bot identity are refused (the identity
            task keeps retrying).
            """
            if not self._running:
                return

            try:
                event = data.event
                message = event.message
                sender = event.sender

                if not message or not sender:
                    return

                sender_type = getattr(sender, "sender_type", "") or ""
                if sender_type == "bot":
                    return

                message_id = message.message_id
                if self._is_duplicate(message_id):
                    logger.debug("Duplicate Feishu message ignored: %s", message_id)
                    return

                sender_id_obj = getattr(sender, "sender_id", None)
                sender_id = ""
                if sender_id_obj and getattr(sender_id_obj, "open_id", None):
                    sender_id = str(sender_id_obj.open_id).strip()

                mentions = _extract_mentions(message)
                chat_type = message.chat_type or ""
                is_group = chat_type == "group"
                bot_mentioned = False
                bot_mention_keys: list[str] = []
                if is_group:
                    needs_identity = any(
                        m["open_id"] and m["open_id"] != _ALL_MENTION_OPEN_ID for m in mentions
                    )
                    if needs_identity and not self._bot_open_id:
                        logger.warning(
                            "Feishu group message dropped: bot identity not ready (%s)",
                            str(message_id)[:8],
                        )
                        return
                    bot_mentioned = any(
                        m["open_id"] == self._bot_open_id
                        for m in mentions
                        if m["open_id"] != _ALL_MENTION_OPEN_ID
                    )
                    bot_mention_keys = [
                        m["key"] for m in mentions if m["key"] and m["open_id"] == self._bot_open_id
                    ]

                raw_payload = {
                    "message_id": message_id,
                    "message_type": message.message_type,
                    "content": message.content,
                    "chat_id": message.chat_id,
                    "chat_type": chat_type,
                    "thread_id": getattr(message, "thread_id", "") or "",
                    "sender": {
                        "sender_id": sender_id,
                        "sender_type": sender_type,
                    },
                    "create_time": data.header.create_time if data.header else "",
                    "mentions": mentions,
                    "bot_mentioned": bot_mentioned,
                    "bot_mention_keys": bot_mention_keys,
                }

                logger.info(
                    "Feishu received message: id=%s type=%s sender=%s group=%s mentioned=%s",
                    message_id,
                    message.message_type,
                    sender_id[:10],
                    is_group,
                    bot_mentioned if is_group else "-",
                )

                if not is_group or bot_mentioned:
                    self._add_typing_reaction_async(message_id)
                    session_key = self._typing_session_key(
                        {
                            "chat_id": message.chat_id or "",
                            "chat_type": chat_type,
                            "thread_id": getattr(message, "thread_id", "") or "",
                            "to_handle": (message.chat_id if is_group else sender_id) or "",
                        }
                    )
                    if session_key:
                        pending = self._typing_sessions.setdefault(session_key, [])
                        pending.append(message_id)
                        del pending[:-20]

                if self._enqueue_callback:
                    self._enqueue_callback(raw_payload)
                    logger.info("Feishu message enqueued: %s", message_id)
                else:
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.run_coroutine_threadsafe(self.handle_inbound(raw_payload), loop)
                        logger.info("Feishu message dispatched (fallback): %s", message_id)
                    except RuntimeError:
                        logger.error(
                            "Feishu: no running event loop, message DROPPED: %s", message_id
                        )

            except Exception:
                logger.exception("Error handling Feishu message event")

        # ------------------------------------------------------------------
        # Parsing
        # ------------------------------------------------------------------

        def parse_inbound(self, raw_payload: Any) -> Any:
            message = super().parse_inbound(raw_payload)
            is_group = False
            if isinstance(raw_payload, dict):
                is_group = str(raw_payload.get("chat_type", "") or "") == "group"
                if is_group:
                    message.metadata["bot_mentioned"] = bool(raw_payload.get("bot_mentioned"))
                mentions = raw_payload.get("mentions")
                if isinstance(mentions, list):
                    keys = raw_payload.get("bot_mention_keys")
                    message.content = _replace_mention_placeholders(
                        message.content,
                        [m for m in mentions if isinstance(m, dict)],
                        list(keys) if isinstance(keys, list) else [],
                    )
            if message.channel_subject is not None:
                message.channel_subject.chat_type = "group" if is_group else "direct"
            message.metadata["turn_budget"] = {
                "timeout_s": self._hardened_turn_timeout_s,
                "progress_interval_s": self._hardened_progress_interval_s,
            }
            return message

        # ------------------------------------------------------------------
        # Turn boundary: typing reaction add/remove
        # ------------------------------------------------------------------

        def _typing_session_key(self, meta: dict[str, Any]) -> str:
            """Session key shared by the receive path and the turn-end sweep."""
            thread_id = str(meta.get("thread_id") or "")
            if thread_id:
                return thread_id
            if str(meta.get("chat_type") or "") == "group":
                return str(meta.get("chat_id") or "")
            return str(meta.get("to_handle") or "")

        async def handle_inbound(self, raw_payload: object) -> None:
            message_ids = self._pending_typing_ids(raw_payload)
            try:
                await super().handle_inbound(raw_payload)
            finally:
                for pending_id in message_ids:
                    await self._remove_typing_reaction(pending_id)

        def _pending_typing_ids(self, raw_payload: object) -> list[str]:
            """Message ids whose typing reactions must be cleared at turn end.

            The manager worker passes a parsed (and batch-merged) InboundMessage,
            not the raw dict — the merged metadata keeps only the first message
            id, so sweep the whole session rather than the merged id alone.
            """
            meta: dict[str, Any] = {}
            if isinstance(raw_payload, dict):
                sender = raw_payload.get("sender")
                sender_id = ""
                if isinstance(sender, dict):
                    sender_id = str(sender.get("sender_id") or "")
                chat_type = str(raw_payload.get("chat_type") or "")
                chat_id = str(raw_payload.get("chat_id") or "")
                meta = {
                    "message_id": str(raw_payload.get("message_id") or ""),
                    "chat_id": chat_id,
                    "chat_type": chat_type,
                    "thread_id": str(raw_payload.get("thread_id") or ""),
                    "to_handle": (chat_id if chat_type == "group" else sender_id) or "",
                }
            else:
                candidate = getattr(raw_payload, "metadata", None)
                if isinstance(candidate, dict):
                    meta = candidate
            ids: list[str] = []
            merged_id = str(meta.get("message_id") or "")
            if merged_id:
                ids.append(merged_id)
            session_key = self._typing_session_key(meta)
            if session_key:
                ids.extend(self._typing_sessions.pop(session_key, []))
            unique: list[str] = []
            seen: set[str] = set()
            for pending_id in ids:
                if pending_id and pending_id not in seen:
                    seen.add(pending_id)
                    unique.append(pending_id)
            return unique

        def _add_typing_reaction_async(self, message_id: str) -> None:
            loop = self._main_loop
            if loop is not None and loop.is_running():
                asyncio.run_coroutine_threadsafe(self._add_typing_reaction(message_id), loop)
            else:
                logger.debug("Feishu: main loop not available, skipping reaction")

        async def _add_typing_reaction(self, message_id: str) -> None:
            """POST the Typing reaction and keep its id for later removal."""
            import aiohttp
            from harness_gateway.channels.feishu import _API_BASE

            try:
                async with asyncio.timeout(_REACTION_ACK_TIMEOUT_SECONDS):
                    token = await self._refresh_token()
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    }
                    url = f"{_API_BASE}/im/v1/messages/{message_id}/reactions"
                    payload = {"reaction_type": {"emoji_type": "Typing"}}
                    async with (
                        aiohttp.ClientSession() as session,
                        session.post(url, headers=headers, json=payload) as resp,
                    ):
                        data = await resp.json()
                        if resp.status == 200 and data.get("code") == 0:
                            reaction_id = str((data.get("data") or {}).get("reaction_id") or "")
                            if reaction_id:
                                self._typing_reactions[message_id] = reaction_id
                            else:
                                logger.warning(
                                    "Feishu typing reaction ack had no reaction_id (%s)",
                                    str(message_id)[:8],
                                )
                        else:
                            body = json.dumps(data, ensure_ascii=False)[:200]
                            logger.warning(
                                "Feishu typing reaction failed: status=%s body=%s",
                                resp.status,
                                body,
                            )
            except TimeoutError:
                logger.warning("Feishu typing reaction ack timed out (%s)", str(message_id)[:8])
            except Exception:
                logger.debug(
                    "Feishu _add_typing_reaction error for %s", str(message_id)[:8], exc_info=True
                )

        async def _remove_typing_reaction(self, message_id: str) -> None:
            """DELETE the Typing reaction once the turn finished.

            Failure is cosmetic: log redacted context and move on.
            """
            import aiohttp
            from harness_gateway.channels.feishu import _API_BASE

            reaction_id = self._typing_reactions.pop(message_id, None)
            if not reaction_id:
                return
            try:
                async with asyncio.timeout(_REACTION_ACK_TIMEOUT_SECONDS):
                    token = await self._refresh_token()
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    }
                    url = f"{_API_BASE}/im/v1/messages/{message_id}/reactions/{reaction_id}"
                    async with (
                        aiohttp.ClientSession() as session,
                        session.delete(url, headers=headers) as resp,
                    ):
                        if resp.status != 200:
                            logger.warning(
                                "Feishu typing reaction remove failed: status=%s msg=%s",
                                resp.status,
                                str(message_id)[:8],
                            )
            except TimeoutError:
                logger.warning("Feishu typing reaction remove timed out (%s)", str(message_id)[:8])
            except Exception:
                logger.debug(
                    "Feishu _remove_typing_reaction error for %s",
                    str(message_id)[:8],
                    exc_info=True,
                )

        # ------------------------------------------------------------------
        # Streaming status card (CardKit v1) — opt-in via channel config
        # ------------------------------------------------------------------

        async def _process_inbound(self, message: Any, subject: Any) -> bool:
            session = await self._maybe_start_card_session(subject)
            if session is None:
                return await super()._process_inbound(message, subject)
            key = str(subject.subject_id)
            self._card_sessions[key] = session
            try:
                succeeded = await super()._process_inbound(message, subject)
                fallback = await session.close(
                    feishu_card.TERMINAL_DONE if succeeded else feishu_card.TERMINAL_ERROR
                )
                await self._deliver_card_fallback(subject, fallback)
                return succeeded
            except Exception:
                fallback = await session.close(feishu_card.TERMINAL_ERROR)
                with contextlib.suppress(Exception):
                    await self._deliver_card_fallback(subject, fallback)
                raise
            finally:
                self._card_sessions.pop(key, None)

        async def _deliver_card_fallback(self, subject: Any, texts: list[str]) -> None:
            for text in texts:
                await super()._send_text(subject, text)

        async def _maybe_start_card_session(self, subject: Any) -> Any:
            """Create the turn's stream card; None ⇒ plain delivery (off/unavailable)."""
            if not self._stream_card_enabled:
                return None
            from octop.infra.utils.locale import resolve_user_locale

            async def deliver(content: str) -> None:
                await self._deliver(subject, msg_type="interactive", content=content)

            session = feishu_card.StreamCardSession(
                http_provider=self._ensure_http,
                token_provider=self._refresh_token,
                deliver=deliver,
                locale=resolve_user_locale(channel_type="feishu"),
            )
            try:
                await session.start()
            except Exception:
                logger.warning(
                    "Feishu stream card unavailable; using plain messages", exc_info=True
                )
                return None
            return session

        def _active_card_session(self, subject: Any) -> Any:
            session = self._card_sessions.get(str(getattr(subject, "subject_id", "") or ""))
            if session is not None and session.active:
                return session
            return None

        async def _send_text(self, subject: Any, text: str) -> None:
            session = self._active_card_session(subject)
            if session is not None:
                session.on_text(text)
                return
            await super()._send_text(subject, text)

        async def _on_tool_start(self, subject: Any, event: Any) -> None:
            session = self._active_card_session(subject)
            if session is not None:
                session.on_tool_start(str(event.metadata.get("tool_name") or "tool"))
                return
            await super()._on_tool_start(subject, event)

        async def _on_tool_end(self, subject: Any, event: Any) -> None:
            session = self._active_card_session(subject)
            if session is not None:
                session.on_tool_end(
                    str(event.metadata.get("tool_name") or "tool"),
                    error=bool(event.metadata.get("is_error")),
                )
                return
            await super()._on_tool_end(subject, event)

        async def _deliver_event(self, subject: Any, event: Any) -> None:
            session = self._active_card_session(subject)
            if (
                session is not None
                and event.type == MessageEventType.MESSAGE
                and event.metadata.get("progress")
            ):
                session.on_progress(
                    "\n".join(
                        part.text
                        for part in event.content
                        if isinstance(part, TextContent) and part.text
                    )
                )
                return
            await super()._deliver_event(subject, event)

    _FeishuHardenedChannel.__name__ = "FeishuHardenedChannel"
    _FeishuHardenedChannel.__qualname__ = "FeishuHardenedChannel"
    return _FeishuHardenedChannel


_HARDENED_CHANNEL_CLS: type[Any] | None = None


def feishu_hardened_channel_cls() -> type[Any]:
    """Resolve the hardened channel class (imports FeishuChannel lazily)."""
    global _HARDENED_CHANNEL_CLS
    if _HARDENED_CHANNEL_CLS is None:
        _HARDENED_CHANNEL_CLS = _build_feishu_hardened_channel()
    return _HARDENED_CHANNEL_CLS


def build_feishu_hardened_channel(
    processor: MessageProcessor,
    config: Any,
    *,
    channel_id: str | None = None,
    tenant_id: str | None = None,
    turn_timeout_s: float = DEFAULT_TURN_TIMEOUT_SECONDS,
    progress_interval_s: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
    stream_card: bool = False,
) -> Any:
    """Construct a :class:`FeishuHardenedChannel` with explicit parameters."""
    cls = feishu_hardened_channel_cls()
    return cls(
        processor,
        config,
        channel_id=channel_id,
        tenant_id=tenant_id,
        turn_timeout_s=turn_timeout_s,
        progress_interval_s=progress_interval_s,
        stream_card=stream_card,
    )


__all__ = [
    "DEFAULT_PROGRESS_INTERVAL_SECONDS",
    "DEFAULT_TURN_TIMEOUT_SECONDS",
    "build_feishu_hardened_channel",
    "ensure_feishu_ws_stop_fix",
    "feishu_hardened_channel_cls",
    "probe_feishu_credentials",
]
