"""Host-owned IM turn deadlines and progress, using Harness execution scopes."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import math
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from octop_gateway.channel import MessageProcessor
from octop_gateway.models import InboundMessage, MessageEvent, TextContent
from octop_harness import execution_scope

from octop.i18n import tr
from octop.infra.utils.locale import Locale

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnBudgetSpec:
    timeout_s: float
    progress_interval_s: float


@dataclass
class TurnBudgetOutcome:
    timed_out: bool = False


def _seconds(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, lower), upper) if math.isfinite(number) and number > 0 else default


def processor_with_feishu_budget(
    processor: MessageProcessor, config: dict[str, Any]
) -> MessageProcessor:
    timeout = _seconds(config.get("turn_timeout_seconds"), 600, 60, 3600)
    interval = _seconds(config.get("progress_interval_seconds"), 60, 30, 600)

    async def wrapped(message: InboundMessage) -> AsyncIterator[MessageEvent]:
        message.metadata["turn_budget"] = {
            "timeout_s": timeout,
            "progress_interval_s": min(interval, timeout / 2),
        }
        async for event in processor(message):
            yield event

    return wrapped


def turn_budget_from_metadata(metadata: dict[str, Any] | None) -> TurnBudgetSpec | None:
    raw = (metadata or {}).get("turn_budget")
    if not isinstance(raw, dict):
        return None
    timeout = _seconds(raw.get("timeout_s"), 0, 60, 3600)
    if not timeout:
        return None
    interval = _seconds(raw.get("progress_interval_s"), 60, 30, timeout / 2)
    return TurnBudgetSpec(timeout, interval)


async def consume_turn_stream(
    stream: AsyncIterator[MessageEvent],
    *,
    spec: TurnBudgetSpec,
    agent_id: str,
    thread_id: str,
    locale: Locale,
    cancel: Callable[[], Any],
    outcome: TurnBudgetOutcome,
) -> AsyncIterator[MessageEvent]:
    """Consume the source in one task so ContextVar scopes never cross contexts."""
    queue: asyncio.Queue[MessageEvent] = asyncio.Queue(maxsize=1)

    async def produce() -> None:
        async with execution_scope(timeout_s=spec.timeout_s):
            try:
                async for event in stream:
                    await queue.put(event)
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()

    producer = asyncio.create_task(produce(), context=contextvars.copy_context())
    getter: asyncio.Task[MessageEvent] | None = None
    started = time.monotonic()
    deadline = started + spec.timeout_s
    progress_at = started + spec.progress_interval_s
    try:
        while True:
            if producer.done() and queue.empty() and (getter is None or not getter.done()):
                producer.result()
                return
            now = time.monotonic()
            if now >= deadline:
                outcome.timed_out = True
                result = cancel()
                if asyncio.iscoroutine(result):
                    await result
                yield MessageEvent.error_event(
                    tr("feishu.turn_timeout", locale, minutes=max(1, round(spec.timeout_s / 60)))
                )
                return
            if now >= progress_at:
                yield MessageEvent(
                    content=[
                        TextContent(
                            text=tr(
                                "feishu.progress_running",
                                locale,
                                minutes=max(1, round((now - started) / 60)),
                            )
                        )
                    ],
                    metadata={"progress": True},
                )
                progress_at = now + spec.progress_interval_s
            if getter is None:
                getter = asyncio.create_task(queue.get())
            await asyncio.wait(
                {getter, producer},
                timeout=max(0, min(progress_at, deadline) - time.monotonic()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if getter.done():
                yield getter.result()
                getter = None
    finally:
        if getter is not None:
            getter.cancel()
            await asyncio.gather(getter, return_exceptions=True)
        if not producer.done():
            producer.cancel()
        done, _ = await asyncio.wait({producer}, timeout=2)
        if done:
            await asyncio.gather(producer, return_exceptions=True)
        else:
            logger.warning("Turn stream did not stop: agent=%s thread=%s", agent_id, thread_id)
            producer.cancel()
            producer.add_done_callback(_retrieve_failure)


def _retrieve_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()
