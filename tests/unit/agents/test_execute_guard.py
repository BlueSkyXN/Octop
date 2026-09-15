"""Execute process guard: registry, wrapped backends, /stop kill, clamp."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import Any

import pytest

from octop.infra.agents.execute_guard import (
    ExecuteBudgetClampMiddleware,
    TurnExecRegistry,
    claim_turn_exec_context,
    current_turn_exec_context,
    ensure_execute_process_guard,
    kill_turn_processes,
    lookup_turn_exec_context,
    release_turn_exec_context,
    reset_current_turn_exec_context,
    set_current_turn_exec_context,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")

# Unique sleep durations double as pgrep markers.
_MARK_TIMEOUT = "300.1097"
_MARK_STOP = "300.1098"


def _pgrep(pattern: str) -> list[int]:
    result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(x) for x in result.stdout.split()]


def _backend() -> Any:
    from harness_agent.backends.local_shell import HarnessLocalShellBackend

    backend = HarnessLocalShellBackend(
        root_dir=None,
        workspace_dir=None,
        virtual_mode=False,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    backend._refresh_execute_env()
    return backend


# ---------------------------------------------------------------------------
# Registry primitives
# ---------------------------------------------------------------------------


def test_registry_kill_all_without_processes_is_noop() -> None:
    registry = TurnExecRegistry()
    registry.kill_all()
    assert registry.killed is True


def test_claim_lookup_release_cycle() -> None:
    ctx = claim_turn_exec_context(agent_id="a1", thread_id="t1", timeout_s=60)
    assert lookup_turn_exec_context(agent_id="a1", thread_id="t1") is ctx
    release_turn_exec_context(ctx)
    assert lookup_turn_exec_context(agent_id="a1", thread_id="t1") is None
    # Releasing a stale context must not drop a newer claim.
    ctx2 = claim_turn_exec_context(agent_id="a1", thread_id="t1", timeout_s=60)
    release_turn_exec_context(ctx)
    assert lookup_turn_exec_context(agent_id="a1", thread_id="t1") is ctx2
    release_turn_exec_context(ctx2)


# ---------------------------------------------------------------------------
# Wrapped backends
# ---------------------------------------------------------------------------


def test_guard_install_is_idempotent() -> None:
    first = ensure_execute_process_guard()
    second = ensure_execute_process_guard()
    assert second is False
    if os.name == "posix":
        assert first is True


def _original_host_execute() -> Any:
    """The pristine upstream implementation captured behind the wrapper."""
    guard_marker = "_octop_execute_process_guard"
    from harness_agent.backends.local_shell import HarnessLocalShellBackend

    closure = HarnessLocalShellBackend._execute_on_host
    if getattr(HarnessLocalShellBackend, guard_marker, False):
        # The wrapper closes over the original; reach it through its cells.
        for cell in closure.__closure__ or ():
            value = cell.cell_contents
            if callable(value) and value is not closure:
                return value
    return closure


def test_without_context_output_matches_upstream_contract() -> None:
    """No turn context → the wrapped backend returns upstream-shaped output."""
    ensure_execute_process_guard()
    assert current_turn_exec_context() is None
    backend = _backend()
    response = backend.execute("echo hello-guard", timeout=10)
    assert response.exit_code == 0
    assert response.output == "hello-guard\n"

    # Byte-for-byte parity with the pristine implementation on a quick command.
    original = _original_host_execute()
    upstream = original(backend, "echo hello-guard", timeout=10)
    assert response.output == upstream.output
    assert response.exit_code == upstream.exit_code


@posix_only
def test_wrapped_backend_timeout_kills_grandchildren() -> None:
    ensure_execute_process_guard()
    ctx = claim_turn_exec_context(agent_id="a1", thread_id="t1", timeout_s=120)
    token = set_current_turn_exec_context(ctx)
    try:
        assert not _pgrep(_MARK_TIMEOUT)
        backend = _backend()
        started = time.monotonic()
        response = backend.execute(f"sleep {_MARK_TIMEOUT} & wait", timeout=1)
        elapsed = time.monotonic() - started
        assert response.exit_code == 124
        assert "timed out after 1 seconds (custom timeout)" in response.output
        assert elapsed < 30
        # The shell-spawned sleep (grandchild) must be gone too.
        assert _pgrep(_MARK_TIMEOUT) == []
    finally:
        reset_current_turn_exec_context(token)
        release_turn_exec_context(ctx)


@posix_only
@pytest.mark.asyncio
async def test_stop_path_kills_registered_turn_processes() -> None:
    ensure_execute_process_guard()
    ctx = claim_turn_exec_context(agent_id="a2", thread_id="t2", timeout_s=120)
    token = set_current_turn_exec_context(ctx)
    assert not _pgrep(_MARK_STOP)
    try:
        backend = _backend()
        task = asyncio.ensure_future(
            asyncio.to_thread(backend.execute, f"sleep {_MARK_STOP} & wait", timeout=60)
        )
        for _ in range(50):
            if _pgrep(_MARK_STOP):
                break
            await asyncio.sleep(0.1)
        assert _pgrep(_MARK_STOP), "command should be running"

        await asyncio.to_thread(kill_turn_processes, "a2", "t2")

        response = await asyncio.wait_for(task, 15)
        assert not _pgrep(_MARK_STOP), "grandchildren must be reaped by /stop path"
        assert response.exit_code != 0
    finally:
        reset_current_turn_exec_context(token)
        release_turn_exec_context(ctx)


# ---------------------------------------------------------------------------
# Clamp middleware
# ---------------------------------------------------------------------------


def _tool_request(name: str, args: dict[str, Any]) -> Any:
    from langgraph.prebuilt.tool_node import ToolCallRequest

    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "call_1"},
        tool=None,
        state={},
        runtime=None,
    )


@pytest.mark.asyncio
async def test_clampmiddleware_no_context_passthrough() -> None:
    assert current_turn_exec_context() is None
    seen: list[Any] = []

    async def handler(request: Any) -> Any:
        seen.append(request.tool_call["args"])
        return "ok"

    result = await ExecuteBudgetClampMiddleware().awrap_tool_call(
        _tool_request("execute", {"command": "sleep 5", "timeout": 9999}), handler
    )
    assert result == "ok"
    assert seen == [{"command": "sleep 5", "timeout": 9999}]


@pytest.mark.asyncio
async def test_clamp_middleware_clamps_execute_timeout_to_budget() -> None:
    ctx = claim_turn_exec_context(agent_id="a3", thread_id="t3", timeout_s=10)
    token = set_current_turn_exec_context(ctx)
    seen: list[Any] = []

    async def handler(request: Any) -> Any:
        seen.append(request.tool_call["args"])
        return "ok"

    try:
        await ExecuteBudgetClampMiddleware().awrap_tool_call(
            _tool_request("execute", {"command": "sleep 5", "timeout": 9999}), handler
        )
        assert seen[0]["timeout"] <= 11
        assert seen[0]["timeout"] >= 1
        # Non-execute tools and non-numeric timeouts pass through untouched.
        await ExecuteBudgetClampMiddleware().awrap_tool_call(
            _tool_request("read_file", {"path": "/tmp/x"}), handler
        )
        assert seen[1] == {"path": "/tmp/x"}
        await ExecuteBudgetClampMiddleware().awrap_tool_call(
            _tool_request("execute", {"command": "x"}), handler
        )
        assert seen[2] == {"command": "x"}
    finally:
        reset_current_turn_exec_context(token)
        release_turn_exec_context(ctx)
