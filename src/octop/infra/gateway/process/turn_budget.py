"""Turn budget: heartbeat + deadline loop around an agent turn stream.

Feishu group turns that carry ``metadata["turn_budget"]`` are consumed through
:func:`consume_turn_stream`, which

* yields localized progress notes while the agent keeps working,
* cancels the harness stream and reaps turn execute process groups at the
  deadline (bounded drain collects whatever the stream flushes after cancel),
* steps the stream in one shared ``contextvars`` context — per-task context
  copies would reset harness ContextVar scopes (``session_header_scope``) in a
  different Context and fail the turn,
* tears the generator down in the verified order: cancel pending ``__anext__``
  task → wait (bounded) for it to exit and retrieve its result/exception →
  only then ``aclose()`` (calling ``aclose`` while ``__anext__`` is still
  running raises ``RuntimeError: asynchronous generator is already running``).

The loop shape and teardown order were validated standalone in
``local/verify/step0_a_heartbeat.py`` (busy stream, slow generator, budget
expiry, cancel-swallowing generators).

Channels without ``turn_budget`` metadata keep the plain ``async for`` path in
the processor — behavior unchanged.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from harness_gateway.models import MessageEvent, MessageEventType, TextContent

from octop.i18n import tr
from octop.infra.agents.execute_guard import (
    TurnExecContext,
    claim_turn_exec_context,
    release_turn_exec_context,
    reset_current_turn_exec_context,
    set_current_turn_exec_context,
)
from octop.infra.utils.locale import Locale

logger = logging.getLogger(__name__)

# Bounds mirror the standalone verification scripts; see module docstring.
_DRAIN_BUDGET_SECONDS = 2.0
_TEARDOWN_WAIT_SECONDS = 2.0
_MIN_WAIT_SECONDS = 0.05

_MIN_TIMEOUT_SECONDS = 60.0
_MAX_TIMEOUT_SECONDS = 3600.0
_MIN_PROGRESS_INTERVAL_SECONDS = 30.0

_METADATA_KEY = "turn_budget"


@dataclass(frozen=True)
class TurnBudgetSpec:
    timeout_s: float
    progress_interval_s: float


def turn_budget_from_metadata(metadata: dict[str, Any] | None) -> TurnBudgetSpec | None:
    """Read the channel-stamped budget spec from inbound metadata."""
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get(_METADATA_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        timeout_s = float(raw.get("timeout_s", 0) or 0)
        interval_s = float(raw.get("progress_interval_s", 0) or 0)
    except (TypeError, ValueError):
        return None
    if timeout_s <= 0:
        return None
    timeout_s = min(max(timeout_s, _MIN_TIMEOUT_SECONDS), _MAX_TIMEOUT_SECONDS)
    interval_s = min(
        max(interval_s, _MIN_PROGRESS_INTERVAL_SECONDS),
        max(timeout_s / 2.0, _MIN_PROGRESS_INTERVAL_SECONDS),
    )
    return TurnBudgetSpec(timeout_s=timeout_s, progress_interval_s=interval_s)


@dataclass
class TurnBudgetOutcome:
    """Observable result of one budgeted turn for the caller."""

    timed_out: bool = False


def _progress_event(locale: Locale, *, elapsed_s: float) -> MessageEvent:
    minutes = max(1, round(elapsed_s / 60.0))
    return MessageEvent(
        type=MessageEventType.MESSAGE,
        content=[TextContent(text=tr("feishu.progress_running", locale, minutes=minutes))],
        metadata={"progress": True},
    )


def _timeout_event(locale: Locale, *, timeout_s: float) -> MessageEvent:
    minutes = max(1, round(timeout_s / 60.0))
    return MessageEvent.error_event(tr("feishu.turn_timeout", locale, minutes=minutes))


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
    """Drive *stream* under a turn budget; yield progress/error plus its events."""
    ctx = claim_turn_exec_context(agent_id=agent_id, thread_id=thread_id, timeout_s=spec.timeout_s)
    token = set_current_turn_exec_context(ctx)
    # Every __anext__/aclose must run in this one context: harness streams bind
    # ContextVars (session_header_scope) around their whole body, and resetting
    # that token from a different task context raises ValueError.
    stream_ctx = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    it = stream.__aiter__()
    next_task: asyncio.Task[MessageEvent] | None = None
    progress_at = time.monotonic() + spec.progress_interval_s
    try:
        while True:
            now = time.monotonic()
            if now >= ctx.deadline:
                outcome.timed_out = True
                await _call_cancel(cancel, agent_id=agent_id, thread_id=thread_id)
                yield _timeout_event(locale, timeout_s=spec.timeout_s)
                # Bounded drain with the same join semantics: collect whatever
                # the stream flushes after cancel (≤ _DRAIN_BUDGET_SECONDS).
                drain_end = time.monotonic() + _DRAIN_BUDGET_SECONDS
                while True:
                    if next_task is None:
                        if time.monotonic() >= drain_end:
                            break
                        next_task = loop.create_task(_astep(it), context=stream_ctx)
                    wait_s = max(_MIN_WAIT_SECONDS, drain_end - time.monotonic())
                    done, _ = await asyncio.wait({next_task}, timeout=wait_s)
                    if not done:
                        break
                    try:
                        event = next_task.result()
                    except StopAsyncIteration:
                        next_task = None
                        break
                    except BaseException as exc:  # noqa: BLE001 — stream died mid-drain
                        logger.warning("turn stream raised during post-cancel drain: %r", exc)
                        next_task = None
                        break
                    next_task = None
                    yield event
                break
            if now >= progress_at:
                yield _progress_event(locale, elapsed_s=spec.timeout_s - (ctx.deadline - now))
                progress_at = now + spec.progress_interval_s
            if next_task is None:
                next_task = loop.create_task(_astep(it), context=stream_ctx)
            wait_s = max(_MIN_WAIT_SECONDS, min(ctx.deadline, progress_at) - time.monotonic())
            done, _ = await asyncio.wait({next_task}, timeout=wait_s)
            if not done:
                continue  # deadline/progress checks at loop top bound the wait
            try:
                event = next_task.result()
            except StopAsyncIteration:
                break
            next_task = None
            yield event
    finally:
        await _teardown(it, next_task, ctx, stream_ctx)
        reset_current_turn_exec_context(token)
        release_turn_exec_context(ctx)


async def _astep(it: AsyncIterator[MessageEvent]) -> MessageEvent:
    """Step the stream once as a coroutine so ``create_task`` can pin the context."""
    return await it.__anext__()


async def _call_cancel(cancel: Callable[[], Any], *, agent_id: str, thread_id: str) -> None:
    try:
        result = cancel()
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.exception("turn budget cancel failed: agent=%s thread=%s", agent_id, thread_id)


async def _aclose_iterator(it: AsyncIterator[MessageEvent]) -> None:
    """Close the stream generator when it exposes ``aclose`` (async generators do)."""
    aclose = getattr(it, "aclose", None)
    if aclose is not None:
        await aclose()


async def _teardown(
    it: AsyncIterator[MessageEvent],
    next_task: asyncio.Task[MessageEvent] | None,
    ctx: TurnExecContext,
    stream_ctx: contextvars.Context,
) -> None:
    """Cancel pending task → bounded wait for exit → aclose → kill processes."""
    if next_task is not None:
        if not next_task.done():
            next_task.cancel()
        done, _ = await asyncio.wait({next_task}, timeout=_TEARDOWN_WAIT_SECONDS)
        if done:
            try:
                next_task.result()
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            except BaseException as exc:  # noqa: BLE001 — recorded, never swallowed silently
                logger.warning("turn stream task ended with %r", exc)
        else:
            logger.warning(
                "turn stream task did not exit within %.0fs after cancel",
                _TEARDOWN_WAIT_SECONDS,
            )
    close_task = asyncio.get_running_loop().create_task(_aclose_iterator(it), context=stream_ctx)
    close_done, close_pending = await asyncio.wait({close_task}, timeout=_TEARDOWN_WAIT_SECONDS)
    if close_pending:
        logger.warning("turn stream generator did not close cleanly")
    else:
        try:
            close_task.result()
        except (asyncio.CancelledError, GeneratorExit):
            pass
        except BaseException as exc:  # noqa: BLE001 — recorded, never swallowed silently
            logger.warning("turn stream aclose raised %r", exc)
    await asyncio.to_thread(ctx.registry.kill_all)


__all__ = [
    "TurnBudgetOutcome",
    "TurnBudgetSpec",
    "consume_turn_stream",
    "turn_budget_from_metadata",
]
