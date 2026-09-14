"""Compat shim: close leaked Feishu WebSocket connections on channel teardown.

harness-gateway 0.9.7 ``FeishuChannel._stop_ws_client()`` calls
``lark_oapi.ws.Client.stop()`` — a method that does not exist in any released
lark-oapi (only a private ``_disconnect()``). The ``AttributeError`` is
swallowed by a broad ``except``, so every channel re-registration (settings
save, connection probe) leaves the previous WebSocket thread and connection
alive. lark's msg-frontier load-balances inbound events across all live
connections of the same app, and the stale instances drop messages silently
(``_running`` is already False) — users see "bot shows connected but never
replies" that worsens with every settings save.

This shim replaces ``FeishuChannel._run_ws_thread`` / ``_stop_ws_client`` at
class level: the fixed teardown closes the socket on the client's own event
loop (``_disconnect()``) and halts that loop so the select/ping daemon thread
exits instead of leaking.

Removal: delete this module (and its call in ``gateway.py``) once
harness-gateway ships the fix; :func:`ensure_feishu_ws_stop_fix` also
self-disables when ``lark_oapi.ws.Client`` gains a public ``stop()``.

Scope: single-app connection leak only. The multi-app global-event-loop race
tracked in TencentCloud/Octop#620 is a different defect, not addressed here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_octop_feishu_ws_stop_fix"
_WS_LOOP_ATTR = "_ws_loop"
_JOIN_TIMEOUT_SECONDS = 5.0


def ensure_feishu_ws_stop_fix() -> bool:
    """Apply the Feishu WS teardown fix once; return True when (newly) patched.

    No-op when harness-gateway / lark-oapi are unavailable, when the SDK
    already exposes a public ``stop()`` (stock teardown works again), or when
    the patch is already applied.
    """
    try:
        import lark_oapi as lark
        from harness_gateway.channels.feishu import FeishuChannel
    except Exception:  # pragma: no cover - optional channel dependencies
        return False

    if getattr(FeishuChannel, _PATCH_MARKER, False):
        return False
    if hasattr(lark.ws.Client, "stop"):
        return False

    FeishuChannel._run_ws_thread = _fixed_run_ws_thread  # type: ignore[method-assign]
    FeishuChannel._stop_ws_client = _fixed_stop_ws_client  # type: ignore[method-assign]
    setattr(FeishuChannel, _PATCH_MARKER, True)
    logger.info("Applied Feishu WS teardown compat fix (harness-gateway connection leak)")
    return True


def _fixed_run_ws_thread(self: Any) -> None:
    """Stock thread body plus per-instance loop capture for teardown."""
    import asyncio as _asyncio

    # Create a new event loop for this thread (lark-oapi needs it)
    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)
    setattr(self, _WS_LOOP_ATTR, loop)
    try:
        import lark_oapi.ws.client as ws_client_module

        ws_client_module.loop = loop
    except (ImportError, AttributeError):
        pass
    try:
        if self._ws_client:
            self._ws_client.start()
    except Exception:
        logger.exception("Feishu WebSocket thread failed")


async def _fixed_stop_ws_client(self: Any) -> None:
    """Stop the WebSocket client for real; end its worker thread."""
    client = self._ws_client
    loop = getattr(self, _WS_LOOP_ATTR, None)

    used_fallback = False
    if client is not None:
        stopped = False
        if hasattr(client, "stop"):
            # Public stop() if a newer SDK provides one — it owns its own
            # loop/thread lifecycle.
            try:
                client.stop()
                stopped = True
            except Exception:
                logger.debug("Error stopping Feishu WebSocket client", exc_info=True)
        if not stopped and loop is not None and loop.is_running():
            # lark-oapi has no public stop(): close the socket on the client's
            # own loop so msg-frontier drops this device, then halt the loop to
            # end the select/ping tasks and the daemon thread.
            try:
                fut = asyncio.run_coroutine_threadsafe(client._disconnect(), loop)
                fut.result(timeout=_JOIN_TIMEOUT_SECONDS)
                stopped = True
                used_fallback = True
                logger.info("Feishu WebSocket connection closed")
            except Exception:
                logger.debug("Feishu WS disconnect failed", exc_info=True)
        self._ws_client = None
    if used_fallback and loop is not None and loop.is_running():
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(loop.stop)

    ws_thread = getattr(self, "_ws_thread", None)
    if ws_thread is not None:
        ws_thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
        self._ws_thread = None
    self._ws_session_id = None
    setattr(self, _WS_LOOP_ATTR, None)
