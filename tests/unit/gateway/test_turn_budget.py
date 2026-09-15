"""Turn budget loop: heartbeats, deadline cancel, verified teardown order."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any

import pytest

from octop.infra.agents.execute_guard import (
    current_turn_exec_context,
    lookup_turn_exec_context,
)
from octop.infra.gateway.process.turn_budget import (
    TurnBudgetOutcome,
    consume_turn_stream,
    turn_budget_from_metadata,
)


def _spec(timeout_s: float, interval_s: float) -> Any:
    from octop.infra.gateway.process.turn_budget import TurnBudgetSpec

    return TurnBudgetSpec(timeout_s=timeout_s, progress_interval_s=interval_s)


async def _collect(gen: Any) -> tuple[list[Any], TurnBudgetOutcome, dict[str, Any]]:
    outcome = TurnBudgetOutcome()
    events = []
    cancels: list[str] = []
    async for ev in consume_turn_stream(
        gen,
        spec=_spec(0.3, 0.1),
        agent_id="ag1",
        thread_id="th1",
        locale="zh",
        cancel=lambda: cancels.append("stop"),
        outcome=outcome,
    ):
        events.append(ev)
    return events, outcome, {"cancels": cancels}


def _texts(events: list[Any]) -> list[str]:
    from harness_gateway.models import TextContent

    out = []
    for ev in events:
        text = "".join(p.text for p in ev.content if isinstance(p, TextContent))
        error = getattr(ev, "error", None)
        out.append(f"{ev.type.value}:{error or text}")
    return out


def test_budget_from_metadata_parses_and_clamps() -> None:
    spec = turn_budget_from_metadata(
        {"turn_budget": {"timeout_s": 600, "progress_interval_s": 120}}
    )
    assert spec is not None
    assert spec.timeout_s == 600
    assert spec.progress_interval_s == 120

    assert turn_budget_from_metadata(None) is None
    assert turn_budget_from_metadata({}) is None
    assert turn_budget_from_metadata({"turn_budget": {"timeout_s": 0}}) is None
    tiny = turn_budget_from_metadata({"turn_budget": {"timeout_s": 5, "progress_interval_s": 1}})
    assert tiny is not None
    assert tiny.timeout_s == 60  # clamped to the floor
    assert tiny.progress_interval_s >= 30


@pytest.mark.asyncio
async def test_busy_stream_delivers_all_events_with_heartbeats() -> None:
    async def gen():
        for i in range(10):
            await asyncio.sleep(0.02)
            yield _msg(f"ev{i}")

    events, outcome, extra = await _collect(gen())
    texts = _texts(events)
    assert [t for t in texts if t.startswith("message:ev")] == [f"message:ev{i}" for i in range(10)]
    assert sum(1 for t in texts if "仍在处理中" in t) >= 2
    assert outcome.timed_out is False
    assert extra["cancels"] == []
    assert lookup_turn_exec_context(agent_id="ag1", thread_id="th1") is None
    assert current_turn_exec_context() is None


@pytest.mark.asyncio
async def test_slow_generator_not_cancelled_multiple_heartbeats() -> None:
    state: list[str] = []

    async def gen():
        try:
            yield _msg("first")
            await asyncio.sleep(0.35)  # longer than 2× heartbeat interval
            yield _msg("last")
        finally:
            state.append("closed")

    events, outcome, _ = await _collect(gen())
    texts = _texts(events)
    assert "message:first" in texts and "message:last" in texts
    idx = texts.index("message:first")
    assert sum(1 for t in texts[idx : texts.index("message:last")] if "仍在处理中" in t) >= 2
    assert outcome.timed_out is False
    assert state == ["closed"]


@pytest.mark.asyncio
async def test_deadline_cancels_drains_flush_and_tears_down(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="octop.infra.gateway.process.turn_budget")
    cancel_event = asyncio.Event()

    async def gen():
        try:
            yield _msg("ack")
            sleep_t = asyncio.ensure_future(asyncio.sleep(30))
            cancel_t = asyncio.ensure_future(cancel_event.wait())
            done, pend = await asyncio.wait(
                {sleep_t, cancel_t}, timeout=31, return_when=asyncio.FIRST_COMPLETED
            )
            for p in pend:
                p.cancel()
            if cancel_t in done:
                yield _msg("partial_after_cancel")
                return
            yield _msg("never")
        finally:
            pass

    async def run():
        outcome = TurnBudgetOutcome()
        cancels: list[str] = []
        events = []
        async for ev in consume_turn_stream(
            gen(),
            spec=_spec(0.3, 0.1),
            agent_id="ag1",
            thread_id="th1",
            locale="zh",
            cancel=cancel_event.set,
            outcome=outcome,
        ):
            events.append(ev)
        return events, outcome, cancels

    events, outcome, _ = await run()
    texts = _texts(events)
    assert "message:ack" in texts
    assert any(t.startswith("error:") and "预算" in t for t in texts), texts
    assert "message:partial_after_cancel" in texts
    assert outcome.timed_out is True
    assert cancel_event.is_set()
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == [], (
        "clean teardown must not log warnings"
    )


@pytest.mark.asyncio
async def test_stubborn_generator_teardown_is_honest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Generator swallowing CancelledError: bounded wait + aclose failures logged."""
    caplog.set_level(logging.WARNING, logger="octop.infra.gateway.process.turn_budget")

    async def stubborn():
        swallows = 0
        yield _msg("ack")
        while True:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                swallows += 1
                if swallows >= 2:
                    raise
                await asyncio.sleep(30)

    async def run() -> tuple[list[Any], TurnBudgetOutcome, Any]:
        outcome = TurnBudgetOutcome()
        events: list[Any] = []
        gen = stubborn()
        async for ev in consume_turn_stream(
            gen,
            spec=_spec(0.15, 0.5),
            agent_id="ag2",
            thread_id="th2",
            locale="zh",
            cancel=lambda: None,
            outcome=outcome,
        ):
            events.append(ev)
        return events, outcome, gen

    events, outcome, gen = await run()
    assert outcome.timed_out is True
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("did not exit" in w for w in warnings), warnings
    assert any("aclose raised" in w for w in warnings), warnings
    # Cleanup: a second cancel makes the swallower re-raise and exit.
    leftover = [t for t in asyncio.all_tasks() if not t.done() and t is not asyncio.current_task()]
    for t in leftover:
        t.cancel()
    await asyncio.gather(*leftover, return_exceptions=True)
    await gen.aclose()


@pytest.mark.asyncio
async def test_stream_contextvar_scope_survives_across_steps() -> None:
    """Harness streams wrap their whole body in a ContextVar scope (session_header_scope).

    Stepping each __anext__ in a fresh task context resets that token in a
    different Context → ValueError → the finished turn is reported failed."""
    var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "t_session_scope", default=None
    )
    seen: list[str | None] = []

    async def gen():
        token = var.set("th1")
        try:
            seen.append(var.get())
            yield _msg("a")
            seen.append(var.get())
            yield _msg("b")
        finally:
            var.reset(token)

    events, outcome, _ = await _collect(gen())
    texts = _texts(events)
    assert [t for t in texts if t.startswith("message:")] == ["message:a", "message:b"]
    assert seen == ["th1", "th1"]
    assert outcome.timed_out is False


@pytest.mark.asyncio
async def test_timeout_teardown_resets_stream_contextvar_cleanly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="octop.infra.gateway.process.turn_budget")
    var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "t_session_scope_timeout", default=None
    )
    cancel_event = asyncio.Event()

    async def gen():
        token = var.set("th1")
        try:
            yield _msg("ack")
            sleep_t = asyncio.ensure_future(asyncio.sleep(30))
            cancel_t = asyncio.ensure_future(cancel_event.wait())
            done, pend = await asyncio.wait(
                {sleep_t, cancel_t}, timeout=31, return_when=asyncio.FIRST_COMPLETED
            )
            for p in pend:
                p.cancel()
            if cancel_t in done:
                return
            yield _msg("never")
        finally:
            var.reset(token)

    outcome = TurnBudgetOutcome()
    events: list[Any] = []
    async for ev in consume_turn_stream(
        gen(),
        spec=_spec(0.3, 0.1),
        agent_id="ag1",
        thread_id="th1",
        locale="zh",
        cancel=cancel_event.set,
        outcome=outcome,
    ):
        events.append(ev)
    texts = _texts(events)
    assert "message:ack" in texts
    assert any(t.startswith("error:") for t in texts), texts
    assert outcome.timed_out is True
    assert cancel_event.is_set()
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == [], (
        "context reset must stay in the stream's shared context"
    )


def _msg(text: str) -> Any:
    from harness_gateway.models import MessageEvent, TextContent

    return MessageEvent(
        content=[TextContent(text=text)],
    )
