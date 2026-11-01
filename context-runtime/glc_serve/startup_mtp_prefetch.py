"""Explicit, bounded startup prefetch for the verified MTP head.

The controller is deliberately private and inert until ``start`` is called by
the verified BHE startup path.  It owns one worker and never owns the resolver:
the caller must join it before closing that resolver.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import sys
import time
from typing import Any, Callable

import torch

from .predictive_mtp import _mtp_source_copy_budget, load_package_mtp


class _MTPPrefetchController:
    """One-shot MTP loader with an explicit resolver lifetime boundary."""

    def __init__(self, *, loader: Callable[..., Any] = load_package_mtp):
        self._loader = loader
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        self._resolver = None
        self._aborted = False
        self.max_mtp_source_copy_bytes = 0
        self.elapsed_seconds = None
        self._prepared_skeleton = None

    @property
    def source_copy_budget_bytes(self) -> int:
        """Maximum temporary source-copy bytes reserved during overlap."""
        return self.max_mtp_source_copy_bytes

    @property
    def max_mtp_source_copy_byte_budget(self) -> int:
        """Spelled-out alias for startup accounting call sites."""
        return self.max_mtp_source_copy_bytes

    def start(self, package, config, device, resolver):
        if self._future is not None:
            raise RuntimeError("MTP prefetch controller can only start once")
        if resolver is None:
            raise ValueError("MTP prefetch requires a borrowed verified resolver")
        self.max_mtp_source_copy_bytes = _mtp_source_copy_budget(resolver)
        if self._loader is load_package_mtp:
            from .loader import build_mtp_skeleton
            # Skeleton construction temporarily patches Module registration.
            # Complete it on the caller before any conversion workers start.
            self._prepared_skeleton = build_mtp_skeleton(config)
        self._resolver = resolver
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="mtp-startup")
        try:
            self._future = self._executor.submit(
                self._run, package, config, device, resolver)
        except BaseException:
            self._shutdown()
            raise
        return self

    def _run(self, package, config, device, resolver):
        started = time.monotonic()
        try:
            return self._load(package, config, device, resolver)
        finally:
            self.elapsed_seconds = time.monotonic() - started

    def _load(self, package, config, device, resolver):
        target = torch.device(device)
        options = {'verified_resolver': resolver}
        if self._prepared_skeleton is not None:
            options['prepared_skeleton'] = self._prepared_skeleton
        if target.type == "cuda":
            with torch.cuda.device(target):
                result = self._loader(
                    package, config, device=device, **options)
                # A CUDA future is not complete until all work on the target
                # device is visible to the caller that joins this controller.
                torch.cuda.synchronize(target)
                return result
        return self._loader(package, config, device=device, **options)

    def _require_future(self) -> Future:
        if self._future is None:
            raise RuntimeError("MTP prefetch controller has not started")
        return self._future

    def result(self):
        """Wait for and return the MTP result, propagating worker failures."""
        future = self._require_future()
        return future.result()

    def join_before_resolver_close(self):
        """Join the worker before its borrowed resolver may be closed."""
        if self._future is None:
            self._shutdown()
            return None
        try:
            if self._aborted:
                return None
            return self._future.result()
        finally:
            self._shutdown()

    def abort(self):
        """Cancel pending work and join running work, suppressing its error.

        This is for an already-failing root conversion path.  The root error
        remains the one reported; the resolver is still safe to close.
        """
        if self._future is None:
            return None
        self._aborted = True
        self._future.cancel()
        try:
            self._future.result()
        except BaseException:
            pass
        finally:
            self._shutdown()
        return None

    def close(self):
        """Join and release the worker without masking an active root error."""
        active_error = sys.exc_info()[0] is not None
        try:
            return self.join_before_resolver_close()
        except BaseException:
            if active_error:
                return None
            raise

    def _shutdown(self):
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None


MTPPrefetchController = _MTPPrefetchController

__all__ = ["_MTPPrefetchController", "MTPPrefetchController"]
