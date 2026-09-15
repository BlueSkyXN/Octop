"""Per-turn subprocess control for budgeted IM turns.

harness-agent executes shell tools via ``subprocess.run`` with a ``timeout``
that only kills the direct child: shell-spawned grandchildren survive the
timeout and survive ``/stop`` (verified against 1.0.9 — see
``local/verify/step0_b_process.py``). For turns that carry a turn budget
(``TurnExecContext``) this module swaps that path for ``Popen`` with
``start_new_session=True`` plus a per-turn process-group registry, so budget
expiry and ``/stop`` can reap the whole process group (SIGTERM → SIGKILL).

Non-budgeted turns never register a context: the compat-wrapped backends then
call the original implementation unchanged, and the clamp middleware passes
tool calls through untouched.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.prebuilt.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)

_GUARD_MARKER = "_octop_execute_process_guard"
_KILL_GRACE_SECONDS = 2.0

# harness-agent versions whose backend internals this compat layer touches.
# Outside this range the patch is skipped and behavior stays upstream.
_SUPPORTED_HARNESS_AGENT_RANGE = ("1.0", "1.1")

_PROCESS_GROUPS_SUPPORTED = os.name == "posix" and hasattr(os, "killpg")


class TurnExecRegistry:
    """Thread-safe registry of live execute process groups for one turn.

    ``register``/``unregister`` run inside ``asyncio.to_thread`` worker
    threads; ``kill_all`` runs from the event-loop side (budget expiry) or the
    ``/stop`` preemption path.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pgids: set[int] = set()
        self._killed = False

    @property
    def killed(self) -> bool:
        return self._killed

    def register(self, pgid: int) -> None:
        with self._lock:
            self._pgids.add(pgid)

    def unregister(self, pgid: int) -> None:
        with self._lock:
            self._pgids.discard(pgid)

    def kill_all(self, *, grace_s: float = _KILL_GRACE_SECONDS) -> None:
        """SIGTERM every live group, then SIGKILL leftovers after *grace_s*.

        Blocking (sleeps up to ``grace_s``): call from a worker thread, e.g.
        ``asyncio.to_thread(registry.kill_all)``.
        """
        with self._lock:
            pgids = list(self._pgids)
            self._killed = True
        if not _PROCESS_GROUPS_SUPPORTED:
            if pgids:
                logger.warning(
                    "turn execute kill_all: process groups unsupported on this OS; "
                    "%d live pid(s) left to their own timeouts",
                    len(pgids),
                )
            return
        for pgid in pgids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGTERM)
        if pgids:
            time.sleep(min(grace_s, 0.5))
            for pgid in pgids:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pgid, signal.SIGKILL)


@dataclass(frozen=True)
class TurnExecContext:
    """Everything the turn loop, the clamp middleware, and the registry share.

    ``deadline`` is an absolute ``time.monotonic()`` timestamp; the same
    ContextVar value must reach tool calls so middleware and backend wrapper
    agree on it within one turn.
    """

    agent_id: str
    thread_id: str
    deadline: float
    registry: TurnExecRegistry

    def remaining_seconds(self) -> float:
        return self.deadline - time.monotonic()


_CURRENT_TURN: ContextVar[TurnExecContext | None] = ContextVar(
    "octop_turn_exec_context", default=None
)

_ACTIVE_TURNS: dict[tuple[str, str], TurnExecContext] = {}
_ACTIVE_TURNS_LOCK = threading.Lock()


def claim_turn_exec_context(*, agent_id: str, thread_id: str, timeout_s: float) -> TurnExecContext:
    """Create the per-turn context and publish it for ``/stop`` lookups."""
    ctx = TurnExecContext(
        agent_id=agent_id,
        thread_id=thread_id,
        deadline=time.monotonic() + max(timeout_s, 1.0),
        registry=TurnExecRegistry(),
    )
    with _ACTIVE_TURNS_LOCK:
        _ACTIVE_TURNS[(agent_id, thread_id)] = ctx
    return ctx


def release_turn_exec_context(ctx: TurnExecContext) -> None:
    """Unpublish the turn context; kills nothing (the turn loop owns teardown)."""
    with _ACTIVE_TURNS_LOCK:
        if _ACTIVE_TURNS.get((ctx.agent_id, ctx.thread_id)) is ctx:
            del _ACTIVE_TURNS[(ctx.agent_id, ctx.thread_id)]


def lookup_turn_exec_context(*, agent_id: str, thread_id: str) -> TurnExecContext | None:
    with _ACTIVE_TURNS_LOCK:
        return _ACTIVE_TURNS.get((agent_id, thread_id))


def current_turn_exec_context() -> TurnExecContext | None:
    return _CURRENT_TURN.get()


def set_current_turn_exec_context(
    ctx: TurnExecContext | None,
) -> Token[TurnExecContext | None]:
    return _CURRENT_TURN.set(ctx)


def reset_current_turn_exec_context(
    token: Token[TurnExecContext | None],
) -> None:
    _CURRENT_TURN.reset(token)


def kill_turn_processes(agent_id: str, thread_id: str) -> None:
    """Kill the live execute processes of one in-flight turn (``/stop`` path)."""
    ctx = lookup_turn_exec_context(agent_id=agent_id, thread_id=thread_id)
    if ctx is None:
        return
    logger.info("killing turn execute processes: agent=%s thread=%s", agent_id, thread_id)
    ctx.registry.kill_all()


# =============================================================================
# Backend compat wrapping (#683 pattern: probe + version lock + method patch)
# =============================================================================


def _harness_agent_version_ok() -> bool:
    from importlib.metadata import PackageNotFoundError, version

    try:
        raw = version("orcakit-harness-agent")
    except PackageNotFoundError:
        return False
    low, high = _SUPPORTED_HARNESS_AGENT_RANGE
    return low <= raw < high


def ensure_execute_process_guard() -> bool:
    """Wrap harness-agent shell backends so budgeted turns reap process groups.

    Idempotent; skips (returns False) when harness-agent is missing, an
    unsupported version, or the internal hooks this relies on moved. In every
    skip case behavior stays exactly upstream.
    """
    if not _PROCESS_GROUPS_SUPPORTED:
        return False
    try:
        from harness_agent.backends import bwrap_shell, local_shell
    except ImportError:
        logger.warning("execute process guard: harness-agent backends unavailable")
        return False
    if not _harness_agent_version_ok():
        logger.info("execute process guard: harness-agent version outside lock, skipping")
        return False

    local_cls = getattr(local_shell, "HarnessLocalShellBackend", None)
    bwrap_cls = getattr(bwrap_shell, "BubbledLocalShellBackend", None)
    if (
        local_cls is None
        or bwrap_cls is None
        or getattr(local_cls, "_execute_on_host", None) is None
        or getattr(bwrap_cls, "execute", None) is None
    ):
        logger.warning("execute process guard: backend hooks not found, skipping")
        return False
    if getattr(local_cls, _GUARD_MARKER, False):
        return False

    original_host_execute = local_cls._execute_on_host
    original_bwrap_execute = bwrap_cls.execute

    def guarded_execute_on_host(self: Any, command: str, *, timeout: int | None) -> Any:
        registry = _current_registry_or_none()
        if registry is None:
            return original_host_execute(self, command, timeout=timeout)
        try:
            effective_timeout = _effective_timeout(self, timeout)
        except ValueError:
            raise
        return _run_gauged_process_group(
            registry,
            command=command,
            shell=True,
            argv=None,
            env=getattr(self, "_env", None),
            cwd=str(self._host_execute_cwd()),
            effective_timeout=effective_timeout,
            timeout_response_builder=lambda t: _host_timeout_response(
                t, custom=timeout is not None
            ),
            output_formatter=lambda rc, out, err: local_shell.format_execute_result(
                subprocess.CompletedProcess(command, rc, stdout=out, stderr=err),
                max_output_bytes=self._max_output_bytes,
            ),
            on_spawn_error=None,
        )

    def guarded_bwrap_execute(self: Any, command: str, *, timeout: int | None) -> Any:
        if not command or not isinstance(command, str):
            return original_bwrap_execute(self, command, timeout=timeout)
        registry = _current_registry_or_none()
        bwrap = getattr(self, "_bwrap_path", None)
        if registry is None or bwrap is None:
            return original_bwrap_execute(self, command, timeout=timeout)
        self._refresh_execute_env()
        effective_timeout = _effective_timeout(self, timeout)
        argv = bwrap_shell.build_bwrap_argv(
            bwrap=bwrap,
            root_dir=self.cwd,
            command=command,
            work_dir=self._virtual_workspace_cwd(),
            extra_binds=self._skill_extra_binds(),
        )

        def _on_spawn_error(exc: Exception) -> Any:
            if isinstance(exc, FileNotFoundError):
                self._bwrap_path = None
                logger.warning(
                    "bwrap failed to start (%s); falling back to translated host execute",
                    exc,
                )
                return original_bwrap_execute(self, command, timeout=timeout)
            return _error_response(exc)

        return _run_gauged_process_group(
            registry,
            command=None,
            shell=False,
            argv=argv,
            env=getattr(self, "_env", None),
            cwd=None,
            effective_timeout=effective_timeout,
            timeout_response_builder=lambda t: _bwrap_timeout_response(
                t, custom=timeout is not None
            ),
            output_formatter=lambda rc, out, err: local_shell.format_execute_result(
                subprocess.CompletedProcess(argv, rc, stdout=out, stderr=err),
                max_output_bytes=self._max_output_bytes,
            ),
            on_spawn_error=_on_spawn_error,
        )

    local_cls._execute_on_host = guarded_execute_on_host
    bwrap_cls.execute = guarded_bwrap_execute
    setattr(local_cls, _GUARD_MARKER, True)
    setattr(bwrap_cls, _GUARD_MARKER, True)
    return True


def _current_registry_or_none() -> TurnExecRegistry | None:
    ctx = _CURRENT_TURN.get()
    return ctx.registry if ctx is not None else None


def _effective_timeout(self: Any, timeout: int | None) -> int:
    effective = timeout if timeout is not None else self._default_timeout
    if effective <= 0:
        msg = f"timeout must be positive, got {effective}"
        raise ValueError(msg)
    return effective


def _host_timeout_response(effective_timeout: int, *, custom: bool) -> Any:
    from deepagents.backends.protocol import ExecuteResponse

    detail = (
        f"{effective_timeout} seconds (custom timeout)"
        if custom
        else f"{effective_timeout} seconds"
    )
    return ExecuteResponse(
        output=(
            f"Error: Command timed out after {detail}. "
            "The command may be stuck or require more time."
        ),
        exit_code=124,
        truncated=False,
    )


def _bwrap_timeout_response(effective_timeout: int, *, custom: bool) -> Any:
    from deepagents.backends.protocol import ExecuteResponse

    if custom:
        msg = (
            f"Error: Command timed out after {effective_timeout} seconds "
            "(custom timeout). The command may be stuck or require more time."
        )
    else:
        msg = (
            f"Error: Command timed out after {effective_timeout} seconds. "
            "For long-running commands, re-run using the timeout parameter."
        )
    return ExecuteResponse(output=msg, exit_code=124, truncated=False)


def _run_gauged_process_group(
    registry: TurnExecRegistry,
    *,
    command: str | None,
    shell: bool,
    argv: list[str] | None,
    env: dict[str, str] | None,
    cwd: str | None,
    effective_timeout: int,
    timeout_response_builder: Any,
    output_formatter: Any,
    on_spawn_error: Any,
) -> Any:
    """Run one command in its own process group, registered for the turn.

    Output/exit-code strings replicate the wrapped backend verbatim — only the
    process lifecycle changes (killpg on timeout instead of killing the shell).
    """
    try:
        popen_args: Sequence[str] = tuple(argv) if argv is not None else str(command)
        proc = subprocess.Popen(
            popen_args,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
    except Exception as exc:
        if on_spawn_error is not None:
            return on_spawn_error(exc)
        return _error_response(exc)

    registry.register(proc.pid)
    try:
        try:
            out, err = proc.communicate(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
            try:
                out, err = proc.communicate(timeout=_KILL_GRACE_SECONDS + 1.0)
            except subprocess.TimeoutExpired:
                _terminate_process_group(proc, kill=True)
                out, err = proc.communicate()
            return timeout_response_builder(effective_timeout)
        except Exception as exc:
            return _error_response(exc)
        return output_formatter(proc.returncode, out, err)
    finally:
        registry.unregister(proc.pid)


def _terminate_process_group(proc: subprocess.Popen[Any], *, kill: bool = False) -> None:
    sig = signal.SIGKILL if kill else signal.SIGTERM
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_KILL_GRACE_SECONDS if not kill else 1.0)


def _error_response(exc: Exception) -> Any:
    from deepagents.backends.protocol import ExecuteResponse

    return ExecuteResponse(
        output=f"Error executing command ({type(exc).__name__}): {exc}",
        exit_code=1,
        truncated=False,
    )


# =============================================================================
# Clamp middleware (auxiliary budget protection)
# =============================================================================


class ExecuteBudgetClampMiddleware(AgentMiddleware[Any, Any]):
    """Clamp ``execute`` timeouts to the remaining turn budget.

    Zero mutable state: everything is read from the turn ContextVar. Without a
    context (dashboard, CLI, cron, non-budgeted channels) tool calls pass
    through untouched.
    """

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        ctx = _CURRENT_TURN.get()
        if ctx is None:
            return await handler(request)
        tool_call = request.tool_call
        if str(tool_call.get("name") or "") != "execute":
            return await handler(request)
        params = tool_call.get("args")
        if not isinstance(params, dict):
            return await handler(request)
        timeout = params.get("timeout")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            return await handler(request)
        remaining = max(1, int(ctx.remaining_seconds()) + 1)
        if timeout > remaining:
            patched = dict(params)
            patched["timeout"] = remaining
            request = request.override(tool_call={**tool_call, "args": patched})
        return await handler(request)


__all__ = [
    "ExecuteBudgetClampMiddleware",
    "TurnExecContext",
    "TurnExecRegistry",
    "claim_turn_exec_context",
    "current_turn_exec_context",
    "ensure_execute_process_guard",
    "kill_turn_processes",
    "lookup_turn_exec_context",
    "release_turn_exec_context",
    "reset_current_turn_exec_context",
    "set_current_turn_exec_context",
]
