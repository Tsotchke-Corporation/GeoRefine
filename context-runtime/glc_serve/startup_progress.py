"""Structured startup progress shared by loaders, kernel builds and servers.

Loaders call :func:`report` at phase boundaries and periodically inside long phases;
a server installs a sink with :func:`set_sink` so ``/health`` can show the current
phase and its progress instead of only "loading". With no sink installed, reports are
kept as the latest snapshot only. A sink must not raise.

Phases used by the Context runtime, in order: ``kernel_build``, ``resolver``,
``model_construct``, ``verify``, ``cache_load``, ``transcode``, ``cache_write``,
``processor``, ``fastdecoder``, ``tune``, ``capture``, ``batch_build``,
``batch_capture``, ``ready``; ``failed`` on error.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

_LOCK = threading.Lock()
_SINK: Optional[Callable[[Dict[str, Any]], None]] = None
_STARTED = time.time()
_LATEST: Dict[str, Any] = {"phase": "init", "t": _STARTED}
_HISTORY: List[Dict[str, Any]] = []


def set_sink(fn: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install (or clear with None) the callback that receives every report."""
    global _SINK
    with _LOCK:
        _SINK = fn


def has_sink() -> bool:
    """Whether a caller has requested live startup diagnostics."""
    with _LOCK:
        return _SINK is not None


def report(phase: str, *, done: Optional[float] = None, total: Optional[float] = None,
           unit: str = "", **detail: Any) -> Dict[str, Any]:
    """Record one progress event and forward it to the sink. Returns the event."""
    now = time.time()
    event: Dict[str, Any] = {"phase": str(phase), "t": now,
                             "since_start_s": round(now - _STARTED, 3)}
    if done is not None:
        event["done"] = done
    if total is not None:
        event["total"] = total
        if done is not None and total:
            event["fraction"] = round(float(done) / float(total), 4)
    if unit:
        event["unit"] = unit
    event.update(detail)
    with _LOCK:
        if not _HISTORY or _HISTORY[-1]["phase"] != event["phase"]:
            _HISTORY.append({"phase": event["phase"], "t": now})
        _LATEST.clear()
        _LATEST.update(event)
        sink = _SINK
    if sink is not None:
        sink(dict(event))
    return event


def snapshot() -> Dict[str, Any]:
    """Latest event plus the ordered phase-entry history (for /health and receipts)."""
    with _LOCK:
        return {"latest": dict(_LATEST), "phases": [dict(h) for h in _HISTORY]}


__all__ = ["set_sink", "has_sink", "report", "snapshot"]
