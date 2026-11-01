"""Scoped, fail-closed batching for MLX Metal status checks.

The guard is deliberately independent of model parameters.  Metal operators
may register a device status array while a request scope is active; the model
fences it at a layer boundary, checks every status, and drops all references.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
from typing import Any, Iterator


class MetalStatusError(RuntimeError):
    """A deferred Metal status was non-zero or escaped its owning scope."""


@dataclass
class _State:
    pending: list[tuple[Any, tuple[Any, ...]]] = field(default_factory=list)
    closed: bool = False
    guard: "MetalStatusGuard | None" = None


_CURRENT: ContextVar[_State | None] = ContextVar("georefine_metal_status", default=None)


def active() -> bool:
    """Whether the current execution context has deferred Metal status on."""
    return _CURRENT.get() is not None


def _raise_if_poisoned(state: _State) -> None:
    if state.guard is not None:
        state.guard._raise_if_poisoned()
    if state.closed:
        raise MetalStatusError("Metal status scope is closed")


class MetalStatusGuard:
    """Persistent, single-owner status state for an engine's GPU work."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Keep only a diagnostic string: retaining the original traceback can
        # retain a frame containing device arrays after a failed fence.
        self._poisoned: str | None = None

    def _raise_if_poisoned(self) -> None:
        if self._poisoned is not None:
            raise MetalStatusError("Metal status guard is poisoned after a prior failure") from RuntimeError(self._poisoned)

    def _poison(self, exc: BaseException) -> None:
        if self._poisoned is None:
            self._poisoned = str(exc) or type(exc).__name__

    @contextmanager
    def scope(self) -> Iterator[None]:
        if _CURRENT.get() is not None:
            raise MetalStatusError("nested Metal status scopes are not allowed")
        self._raise_if_poisoned()
        if not self._lock.acquire(False):
            raise MetalStatusError("Metal status guard is already owned by another scope")
        try:
            # Another owner may have failed between the optimistic check and
            # acquiring the lock. Never admit work using its damaged caches.
            self._raise_if_poisoned()
        except BaseException:
            self._lock.release()
            raise
        state = _State(guard=self)
        token = _CURRENT.set(state)
        try:
            yield
            self._raise_if_poisoned()
            if state.pending:
                exc = MetalStatusError("pending Metal status escaped its model layer fence")
                self._poison(exc)
                state.pending.clear()
                raise exc
        except BaseException as exc:
            if state.pending:
                self._poison(exc)
            state.pending.clear()
            raise
        finally:
            state.closed = True
            state.pending.clear()
            _CURRENT.reset(token)
            self._lock.release()


def register(status: Any, *outputs: Any) -> bool:
    """Register a device status array; return True when the caller should defer."""
    state = _CURRENT.get()
    if state is None:
        return False
    _raise_if_poisoned(state)
    if status is None:
        raise MetalStatusError("cannot register an empty Metal status")
    state.pending.append((status, tuple(x for x in outputs if x is not None)))
    return True


def _nonzero(status: Any) -> list[int]:
    import numpy as np

    values = np.asarray(status).reshape(-1)
    return [int(x) for x in values if int(x) != 0]


def fence(*values: Any) -> None:
    """Evaluate a layer's values and all pending statuses, then clear them."""
    state = _CURRENT.get()
    if state is None:
        return
    _raise_if_poisoned(state)
    import mlx.core as mx

    pending = state.pending
    state.pending = []
    statuses = [status for status, _ in pending]
    try:
        mx.eval(*(tuple(values) + tuple(statuses)))
        for status in statuses:
            bad = _nonzero(status)
            if bad:
                raise MetalStatusError(f"Metal status failure: {len(bad)} nonzero values; first={bad[:8]}")
    except BaseException as exc:
        if state.guard is not None:
            state.guard._poison(exc)
        # Do not retain device arrays after a failed fence.
        pending.clear()
        raise
    finally:
        pending.clear()


def defer_result(value: Any) -> bool:
    """Return whether a caller should leave an output lazy in this scope."""
    state = _CURRENT.get()
    if state is None:
        return False
    _raise_if_poisoned(state)
    return True


@contextmanager
def scope(enabled: bool = True, *, owner: MetalStatusGuard | None = None) -> Iterator[None]:
    """Own one request's deferred state and fail if a status escapes."""
    if not enabled:
        yield
        return
    if _CURRENT.get() is not None:
        raise MetalStatusError("nested Metal status scopes are not allowed")
    guard = owner if owner is not None else MetalStatusGuard()
    if not isinstance(guard, MetalStatusGuard):
        raise TypeError("owner must be a MetalStatusGuard")
    with guard.scope():
        yield


# Descriptive aliases for callers and downstream qualification tests.
metal_status_scope = scope
register_status = register
fence_status = fence

__all__ = [
    "MetalStatusError", "MetalStatusGuard", "active", "defer_result", "fence", "fence_status",
    "metal_status_scope", "register", "register_status", "scope",
]
