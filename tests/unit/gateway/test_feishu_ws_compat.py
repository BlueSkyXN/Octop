"""Tests for the Feishu WS teardown compat shim (connection leak fix)."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from octop.infra.gateway.feishu_ws_compat import (
    _PATCH_MARKER,
    _fixed_run_ws_thread,
    _fixed_stop_ws_client,
    ensure_feishu_ws_stop_fix,
)


@pytest.fixture()
def feishu_cls():
    """Expose FeishuChannel with the shim state restored after each test."""
    from harness_gateway.channels.feishu import FeishuChannel

    saved_run = FeishuChannel._run_ws_thread
    saved_stop = FeishuChannel._stop_ws_client
    saved_marker = getattr(FeishuChannel, _PATCH_MARKER, False)
    yield FeishuChannel
    FeishuChannel._run_ws_thread = saved_run  # type: ignore[assignment]
    FeishuChannel._stop_ws_client = saved_stop  # type: ignore[assignment]
    if saved_marker:
        setattr(FeishuChannel, _PATCH_MARKER, True)
    elif hasattr(FeishuChannel, _PATCH_MARKER):
        delattr(FeishuChannel, _PATCH_MARKER)


def test_apply_patch_then_idempotent(feishu_cls: Any) -> None:
    original_stop = feishu_cls._stop_ws_client
    assert original_stop is not _fixed_stop_ws_client

    assert ensure_feishu_ws_stop_fix() is True
    assert feishu_cls._stop_ws_client is _fixed_stop_ws_client
    assert feishu_cls._run_ws_thread is _fixed_run_ws_thread
    assert getattr(feishu_cls, _PATCH_MARKER) is True

    # Second call is a no-op, not an error.
    assert ensure_feishu_ws_stop_fix() is False


def test_guard_skips_when_sdk_has_public_stop(
    feishu_cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lark_oapi as lark

    monkeypatch.setattr(lark.ws.Client, "stop", lambda self: None, raising=False)

    assert ensure_feishu_ws_stop_fix() is False
    assert feishu_cls._stop_ws_client is not _fixed_stop_ws_client
    assert not hasattr(feishu_cls, _PATCH_MARKER)


def test_run_ws_thread_captures_loop() -> None:
    import lark_oapi.ws.client as ws_client_module

    seen: dict[str, Any] = {}

    class FakeClient:
        def start(self) -> None:
            seen["module_loop"] = ws_client_module.loop

    fake = SimpleNamespace(_ws_client=FakeClient(), _ws_loop=None)
    saved_module_loop = getattr(ws_client_module, "loop", None)
    try:
        _fixed_run_ws_thread(fake)
        assert isinstance(fake._ws_loop, asyncio.AbstractEventLoop)
        assert seen["module_loop"] is fake._ws_loop
    finally:
        ws_client_module.loop = saved_module_loop  # type: ignore[assignment]
        fake._ws_loop.close()


def _start_loop_thread() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True, name="ws-loop")
    thread.start()
    deadline = time.monotonic() + 5.0
    while not loop.is_running() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert loop.is_running()
    return loop, thread


async def test_stop_closes_socket_and_stops_worker_thread() -> None:
    """No public stop() → private _disconnect() runs on the client loop."""
    loop, thread = _start_loop_thread()
    calls: list[str] = []

    class LegacyClient:
        async def _disconnect(self) -> None:
            calls.append("disconnect")

    fake = SimpleNamespace(
        _ws_client=LegacyClient(),
        _ws_thread=thread,
        _ws_session_id="sess",
        _ws_loop=loop,
    )

    await _fixed_stop_ws_client(fake)

    assert calls == ["disconnect"]
    assert not thread.is_alive()  # loop.stop() ended the worker
    assert fake._ws_client is None
    assert fake._ws_thread is None
    assert fake._ws_session_id is None
    assert fake._ws_loop is None
    loop.close()


async def test_stop_prefers_public_stop_without_touching_loop() -> None:
    """Public stop() exists → it owns teardown; no private fallback, no loop stop."""
    loop, thread = _start_loop_thread()
    calls: list[str] = []

    class ModernClient:
        def stop(self) -> None:
            calls.append("stop")

        async def _disconnect(self) -> None:  # pragma: no cover - must not run
            calls.append("disconnect")

    fake = SimpleNamespace(
        _ws_client=ModernClient(),
        _ws_thread=thread,
        _ws_session_id="sess",
        _ws_loop=loop,
    )

    await _fixed_stop_ws_client(fake)

    assert calls == ["stop"]
    assert loop.is_running()  # we did not halt the SDK-owned loop
    assert fake._ws_client is None
    assert fake._ws_thread is None
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    loop.close()
