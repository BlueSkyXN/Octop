"""Host turn budgets preserve context, progress, cancellation and final events."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator

import pytest
from octop_gateway.models import MessageEvent

from octop.infra.gateway.process.response_mode import collapse_to_invoke_response
from octop.infra.gateway.process.turn_budget import (
    TurnBudgetOutcome,
    TurnBudgetSpec,
    consume_turn_stream,
    turn_budget_from_metadata,
)


def test_budget_config_clamps_and_rejects_nonfinite() -> None:
    assert turn_budget_from_metadata(None) is None
    assert turn_budget_from_metadata({"turn_budget": {"timeout_s": "nan"}}) is None
    assert turn_budget_from_metadata({"turn_budget": {"timeout_s": 0}}) is None
    spec = turn_budget_from_metadata({"turn_budget": {"timeout_s": 5, "progress_interval_s": 1}})
    assert spec == TurnBudgetSpec(60, 30)


async def test_source_context_and_final_event_survive() -> None:
    scope: contextvars.ContextVar[str] = contextvars.ContextVar("source", default="outside")
    closed: list[str] = []

    async def source() -> AsyncIterator[MessageEvent]:
        token = scope.set("inside")
        try:
            yield MessageEvent.text(scope.get())
            await asyncio.sleep(0.08)
            yield MessageEvent.text(scope.get())
            yield MessageEvent.completed()
        finally:
            scope.reset(token)
            closed.append(scope.get())

    outcome = TurnBudgetOutcome()
    events = [
        event
        async for event in consume_turn_stream(
            source(),
            spec=TurnBudgetSpec(1, 0.02),
            agent_id="a",
            thread_id="t",
            locale="zh",
            cancel=lambda: None,
            outcome=outcome,
        )
    ]
    texts = ["".join(part.text for part in event.content) for event in events]
    assert [text for text in texts if text == "inside"] == ["inside", "inside"]
    assert events[-1].type.value == "completed"
    assert any(event.metadata.get("progress") for event in events)
    assert not outcome.timed_out
    assert closed == ["outside"]
    assert scope.get() == "outside"


async def test_deadline_cancels_and_closes_in_source_context() -> None:
    closed = asyncio.Event()
    cancelled: list[bool] = []
    scope: contextvars.ContextVar[str] = contextvars.ContextVar("source", default="outside")

    async def source() -> AsyncIterator[MessageEvent]:
        token = scope.set("inside")
        try:
            yield MessageEvent.text("first")
            await asyncio.Event().wait()
        finally:
            scope.reset(token)
            closed.set()

    outcome = TurnBudgetOutcome()
    events = [
        event
        async for event in consume_turn_stream(
            source(),
            spec=TurnBudgetSpec(0.05, 0.01),
            agent_id="a",
            thread_id="t",
            locale="zh",
            cancel=lambda: cancelled.append(True),
            outcome=outcome,
        )
    ]
    assert outcome.timed_out
    assert cancelled == [True]
    assert closed.is_set()
    assert any(event.error and "预算" in event.error for event in events)


async def test_source_error_is_not_swallowed() -> None:
    async def source() -> AsyncIterator[MessageEvent]:
        yield MessageEvent.text("first")
        raise ValueError("source failed")

    with pytest.raises(ValueError, match="source failed"):
        async for _event in consume_turn_stream(
            source(),
            spec=TurnBudgetSpec(1, 0.1),
            agent_id="a",
            thread_id="t",
            locale="en",
            cancel=lambda: None,
            outcome=TurnBudgetOutcome(),
        ):
            pass


async def test_invoke_keeps_heartbeat_separate_from_final_reply() -> None:
    async def source() -> AsyncIterator[MessageEvent]:
        yield MessageEvent.text("narration")
        yield MessageEvent(
            content=MessageEvent.text("progress").content, metadata={"progress": True}
        )
        yield MessageEvent.tool_start("execute")
        yield MessageEvent.text("answer")
        yield MessageEvent.completed()

    events = [event async for event in collapse_to_invoke_response(source())]
    assert events[0].metadata.get("progress")
    assert events[1].content[0].text == "answer"
