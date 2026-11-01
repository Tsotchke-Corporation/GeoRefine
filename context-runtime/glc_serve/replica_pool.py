"""Unwired, process-local admission and request custody for pinned replicas.

No endpoint is contacted here. A dispatcher must authenticate heartbeat identity,
reserve worst-case KV/state bytes, and register synchronous runtime cleanup before
starting work. ``pool.run(..., adapter)`` supplies a stable Lease to that adapter;
``lease.publish(index, token, emit)`` is the only guarded publication API. Transport
acks, distributed coordinator fencing, replica startup, and autoscaling remain the
caller's responsibility. Publication attempts are at-most-once, including ambiguous
transport failures: this primitive never migrates or retries a request.

Admission, waiting, and default in-memory custody are bounded. Replay IDs are
never expired or evicted: MemoryCustody refuses new IDs when full. Production
dispatchers should supply SQLiteCustody at an explicit persistent local path and
namespace, or another durable Custody implementation, preserving claims across
restarts and coordinator replacement. One pool owns each custody namespace;
SQLite claim uniqueness does not provide distributed replica admission/fencing.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
import math
import os
from pathlib import Path
import sqlite3
import stat
from urllib.parse import quote
import inspect
import re
import threading
import time
from typing import Callable, Protocol, TypeVar

T = TypeVar("T")


class PoolError(RuntimeError):
    pass


class Backpressure(PoolError):
    pass


class DuplicateRequest(PoolError):
    pass


class ReplicaUnavailable(PoolError):
    pass


class IdentityMismatch(ReplicaUnavailable):
    pass


class RequestCancelled(PoolError):
    pass


class PublicationError(PoolError):
    pass


class CustodyError(PoolError):
    """Persistent custody is unavailable, invalid, closed, or out of space."""


class CleanupFailed(PoolError):
    """Reservation retained until repair_cleanup completes successfully."""


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 1024:
        raise ValueError(f"{label} must be a nonempty bounded string without outer whitespace")


def _integer(value: int, label: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _seconds(value: float, label: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if value < 0 or (not zero and value == 0):
        raise ValueError(f"{label} must be {'nonnegative' if zero else 'positive'}")
    return float(value)


@dataclass(frozen=True)
class Pins:
    model_sha256: str
    runtime_sha256: str
    profile_sha256: str

    def __post_init__(self):
        for value in (self.model_sha256, self.runtime_sha256, self.profile_sha256):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("model, runtime, and profile pins must be lowercase SHA256 digests")


@dataclass(frozen=True)
class ReplicaIdentity:
    replica_id: str
    endpoint: str
    generation: str
    pins: Pins

    def __post_init__(self):
        for label in ("replica_id", "endpoint", "generation"):
            _name(getattr(self, label), label)
        if not isinstance(self.pins, Pins):
            raise ValueError("pins must be Pins")


@dataclass(frozen=True)
class ReplicaSpec:
    identity: ReplicaIdentity
    max_active: int
    kv_capacity_bytes: int
    state_capacity_bytes: int

    def __post_init__(self):
        if not isinstance(self.identity, ReplicaIdentity):
            raise ValueError("identity must be ReplicaIdentity")
        _integer(self.max_active, "max_active", 1)
        _integer(self.kv_capacity_bytes, "kv_capacity_bytes")
        _integer(self.state_capacity_bytes, "state_capacity_bytes")


class Custody(Protocol):
    def claim(self, request_id: str) -> bool:
        """Atomically persist an ID forever; return False if previously claimed.

        Failures must raise, never return success before durable storage. Claims
        survive release, cancellation, publication failure, and process restarts.
        Calls are serialized under the admission lock: implementations must use
        bounded I/O and must not call back into the pool. Admission timeout bounds
        queue waiting, not backend I/O or an already-running transport callback.
        """
        ...


class MemoryCustody:
    """Bounded process-lifetime custody for local use; exhaustion fails closed.

    Terminal IDs have no TTL and cannot be evicted. Production requires a durable
    Custody implementation to scale total requests and survive process restarts.
    """
    def __init__(self, max_entries: int = 65536):
        _integer(max_entries, "max_entries", 1)
        self._max_entries = max_entries
        self._ids: set[str] = set()
        self._lock = threading.Lock()

    def claim(self, request_id: str) -> bool:
        with self._lock:
            if request_id in self._ids:
                return False
            if len(self._ids) >= self._max_entries:
                raise Backpressure("custody ledger is full; terminal IDs cannot be evicted")
            self._ids.add(request_id)
            return True


class SQLiteCustody:
    """Persistent, never-expiring request claims in a caller-selected local file.

    The parent directory must already exist and be trusted. No directory, cloud
    storage, temporary database, or global path is selected automatically. SQLite
    may create a private rollback journal beside this file. The caller must place
    the DB on persistent storage, retain it for the full request-ID lifetime, and
    arrange backups; choosing an explicit scratch path does not make it durable.

    New files are 0600. Existing files must be regular, owner-private custody DBs
    with this application ID, version, and exact schema; unrelated/empty/corrupt
    files are never initialized or repaired. Failed initialization leaves a file
    for operator inspection. Connections use DELETE journals, FULL synchronous
    commits, no mmap, and a bounded cache. Default 16,384 4-KiB pages cap the DB at
    64 MiB (the adjacent journal can temporarily add storage). ``max_pages=None``
    explicitly opts out. Exhaustion and busy/I/O failures raise, never evict IDs.

    Namespaces are explicit and independent; use the same namespace after restart.
    Multiple independent connections/processes may atomically claim IDs, but this
    does not coordinate their replica capacities. Close explicitly or use ``with``.
    """
    APPLICATION_ID = 0x474C4355
    SCHEMA_VERSION = 1
    PAGE_SIZE = 4096
    _SCHEMA = ("CREATE TABLE custody_claims (namespace TEXT NOT NULL, "
               "request_id TEXT NOT NULL, PRIMARY KEY (namespace, request_id)) WITHOUT ROWID")

    def __init__(self, path: str | os.PathLike[str], namespace: str, *,
                 max_pages: int | None = 16384, cache_kib: int = 1024,
                 busy_timeout: float = 5.0):
        _name(namespace, "namespace")
        _integer(cache_kib, "cache_kib", 1)
        if cache_kib > 2**31 - 1:
            raise ValueError("cache_kib exceeds SQLite signed integer range")
        if max_pages is not None:
            _integer(max_pages, "max_pages", 2)
        busy_timeout = _seconds(busy_timeout, "busy_timeout", zero=True)
        if busy_timeout > 3600:
            raise ValueError("busy_timeout must not exceed one hour")
        raw_path = os.fspath(path)
        if not isinstance(raw_path, str) or not raw_path or raw_path == ':memory:' or raw_path.startswith('file:'):
            raise ValueError("path must name an explicit local file, not a SQLite URI or memory DB")
        self._path = Path(os.path.abspath(raw_path))
        self._namespace = namespace
        self._max_pages = max_pages
        self._lock = threading.Lock()
        self._closed = True
        self._connection: sqlite3.Connection | None = None
        fd = None
        connection = None
        try:
            # O_EXCL ensures that only a newly created file may be initialized.
            flags = os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
            try:
                fd = os.open(self.path, flags | os.O_CREAT | os.O_EXCL, 0o600)
                created = True
                os.fchmod(fd, 0o600)
            except FileExistsError:
                if stat.S_ISLNK(os.lstat(self.path).st_mode):
                    raise CustodyError("custody path must not be a symbolic link")
                fd = os.open(self.path, flags)
                created = False
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
                raise CustodyError("custody file must be regular and owner-private")
            if not created:
                header = os.pread(fd, 100, 0)
                if (len(header) != 100 or header[:16] != b'SQLite format 3\x00'
                        or int.from_bytes(header[68:72], 'big') != self.APPLICATION_ID
                        or int.from_bytes(header[60:64], 'big') != self.SCHEMA_VERSION):
                    raise CustodyError("not a supported custody database")
            uri = 'file:' + quote(str(self.path), safe='/') + '?mode=rw'
            connection = sqlite3.connect(uri, uri=True, timeout=busy_timeout,
                                         isolation_level=None, check_same_thread=False)
            connection.execute('PRAGMA temp_store=MEMORY')
            current = os.stat(self.path, follow_symlinks=False)
            if (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino):
                raise CustodyError("custody path changed while opening")
            if not created:
                self._validate_schema(connection)
                if connection.execute('PRAGMA page_size').fetchone()[0] != self.PAGE_SIZE:
                    raise CustodyError("unsupported custody page size")
            connection.execute(f'PRAGMA busy_timeout={int(busy_timeout * 1000)}')
            connection.execute(f'PRAGMA cache_size={-cache_kib}')
            if connection.execute('PRAGMA cache_size').fetchone()[0] != -cache_kib:
                raise CustodyError("SQLite refused bounded custody cache")
            connection.execute('PRAGMA mmap_size=0')
            if created:
                connection.execute(f'PRAGMA page_size={self.PAGE_SIZE}')
            if connection.execute('PRAGMA journal_mode=DELETE').fetchone()[0].lower() != 'delete':
                raise CustodyError("custody requires DELETE journaling")
            connection.execute('PRAGMA synchronous=FULL')
            if connection.execute('PRAGMA synchronous').fetchone()[0] != 2:
                raise CustodyError("custody requires FULL synchronous commits")
            if max_pages is not None:
                if connection.execute('PRAGMA page_count').fetchone()[0] > max_pages:
                    raise CustodyError("existing custody DB exceeds configured page limit")
                if connection.execute(f'PRAGMA max_page_count={max_pages}').fetchone()[0] != max_pages:
                    raise CustodyError("SQLite refused custody page limit")
            if created:
                connection.execute('BEGIN IMMEDIATE')
                connection.execute(self._SCHEMA)
                connection.execute(f'PRAGMA application_id={self.APPLICATION_ID}')
                connection.execute(f'PRAGMA user_version={self.SCHEMA_VERSION}')
                connection.commit()
                # FULL sync covers DB/journal; persist the new directory entry too.
                parent_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            self._validate_schema(connection)
            if connection.execute('PRAGMA quick_check(1)').fetchall() != [('ok',)]:
                raise CustodyError("custody database integrity check failed")
            self._connection, self._closed = connection, False
        except (OSError, sqlite3.Error) as exc:
            raise CustodyError("could not open durable custody database") from exc
        finally:
            if fd is not None:
                os.close(fd)
            if self._closed and connection is not None:
                connection.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def namespace(self) -> str:
        return self._namespace

    @classmethod
    def _validate_schema(cls, connection: sqlite3.Connection) -> None:
        if (connection.execute('PRAGMA application_id').fetchone()[0] != cls.APPLICATION_ID
                or connection.execute('PRAGMA user_version').fetchone()[0] != cls.SCHEMA_VERSION):
            raise CustodyError("custody application ID or schema version changed")
        entries = connection.execute("SELECT type, name, tbl_name, sql FROM sqlite_master "
                                     "WHERE name NOT LIKE 'sqlite_%' ORDER BY name LIMIT 3").fetchall()
        if entries != [('table', 'custody_claims', 'custody_claims', cls._SCHEMA)]:
            raise CustodyError("custody schema does not match this implementation")

    def claim(self, request_id: str) -> bool:
        _name(request_id, "request_id")
        with self._lock:
            if self._closed:
                raise CustodyError("custody connection is closed")
            connection = self._connection
            try:
                connection.execute('BEGIN IMMEDIATE')
                self._validate_schema(connection)
                if self._max_pages is not None and connection.execute('PRAGMA page_count').fetchone()[0] > self._max_pages:
                    raise CustodyError("custody DB exceeds configured page limit")
                cursor = connection.execute("INSERT INTO custody_claims(namespace, request_id) VALUES (?, ?) "
                                            "ON CONFLICT(namespace, request_id) DO NOTHING",
                                            (self.namespace, request_id))
                claimed = cursor.rowcount == 1
                connection.commit()
                return claimed
            except BaseException as exc:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    connection.close()
                    self._closed = True
                if isinstance(exc, sqlite3.Error):
                    raise CustodyError("custody claim failed; no successful claim is reported") from exc
                raise

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._connection.close()
            self._closed = True
            return True

    def __enter__(self):
        if self.closed:
            raise CustodyError("custody connection is closed")
        return self

    def __exit__(self, *_):
        self.close()


@dataclass(frozen=True)
class ReplicaSnapshot:
    identity: ReplicaIdentity
    active: int
    kv_reserved_bytes: int
    state_reserved_bytes: int
    healthy: bool
    fresh: bool
    draining: bool
    quarantined: bool


@dataclass
class _Replica:
    spec: ReplicaSpec
    seen: float
    healthy: bool = True
    draining: bool = False
    quarantined: bool = False
    active: int = 0
    kv: int = 0
    state: int = 0
    last_assignment: int = 0


@dataclass(eq=False)
class _Waiter:
    request_id: str
    kv: int
    state: int
    cancelled: bool = False


class Lease:
    """One request, one immutable replica, one serialized publication sequence.

    A lease is a context manager. Cleanup runs once, before capacity is returned.
    If cleanup raises, the lease and reservation stay quarantined; explicitly use
    repair_cleanup with a verified teardown callback to retire the reservation.
    """
    def __init__(self, pool: ReplicaPool, request_id: str, replica: _Replica,
                 kv: int, state: int, cancel_event: threading.Event | None):
        self._pool, self._request_id, self._replica = pool, request_id, replica
        self._kv, self._state = kv, state
        self._cancel_event = cancel_event
        self._operation = threading.RLock()
        self._closed = self._cancelled = self._failed = self._closing = False
        self._cleanup: Callable[[], None] | None = None
        self._cleanup_failed = False
        self._next_token = 0
        self._publishing = False

    @property
    def request_id(self):
        return self._request_id

    @property
    def identity(self):
        return self._replica.spec.identity

    @property
    def published_attempts(self):
        with self._operation:
            return self._next_token

    @property
    def closed(self):
        with self._operation:
            return self._closed

    def set_cleanup(self, cleanup: Callable[[], None]) -> None:
        """Install once, before dispatch, a callback that synchronously retires runtime state."""
        if not callable(cleanup):
            raise TypeError("cleanup must be callable")
        with self._operation:
            self._check()
            if self._cleanup is not None or self._next_token:
                raise PoolError("cleanup must be installed once before publication")
            self._cleanup = cleanup

    def _check(self) -> None:
        if self._closed or self._closing or self._failed or self._cleanup_failed:
            raise PoolError("lease is closed or failed")
        if self._cancelled or (self._cancel_event is not None and self._cancel_event.is_set()):
            raise RequestCancelled(self.request_id)
        self._pool._check_replica(self)

    def check(self) -> None:
        with self._operation:
            self._check()

    def publish(self, index: int, token: T, emit: Callable[[T], object]) -> object:
        """Reserve index before calling emit; ambiguous/failed sends are never retried.

        Calls and cleanup on this lease serialize. Health is checked at the
        publication reservation point; an already-running transport send cannot
        be retracted by a subsequent heartbeat. Emit must not reenter lifecycle
        methods; reentrant publication or closure is rejected.
        """
        _integer(index, "index")
        if not callable(emit):
            raise TypeError("emit must be callable")
        with self._operation:
            self._check()
            if self._publishing or index != self._next_token:
                raise PublicationError("publication index must be the next unattempted token")
            self._next_token += 1
            self._publishing = True
            try:
                return emit(token)
            except BaseException:
                self._failed = True
                raise
            finally:
                self._publishing = False

    def _finish(self) -> None:
        self._pool._retire(self)
        self._closed = True

    def close(self) -> bool:
        with self._operation:
            if self._closed:
                return False
            if self._publishing or self._closing:
                raise PoolError("cannot close reentrantly during publication or cleanup")
            if self._cleanup_failed:
                raise CleanupFailed("cleanup failed; reservation remains held")
            self._closing = True
            try:
                if self._cleanup is not None:
                    self._cleanup()
                self._finish()
                return True
            except BaseException as exc:
                self._cleanup_failed = True
                raise CleanupFailed("cleanup failed; reservation remains held") from exc
            finally:
                self._closing = False

    def repair_cleanup(self, cleanup: Callable[[], None]) -> None:
        """Explicit failed-cleanup recovery, without restarting inference/publication."""
        if not callable(cleanup):
            raise TypeError("cleanup must be callable")
        with self._operation:
            if not self._cleanup_failed or self._closed or self._closing:
                raise PoolError("lease is not awaiting cleanup repair")
            self._closing = True
            try:
                cleanup()
                self._finish()
            finally:
                self._closing = False

    def cancel(self) -> bool:
        with self._operation:
            if self._closed:
                return False
            self._cancelled = True
            return self.close()

    def __enter__(self):
        try:
            self.check()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_):
        self.close()


class ReplicaPool:
    """Thread-safe local admission; strict FIFO waiting prevents starvation.

    FIFO deliberately permits head-of-line blocking. Least-loaded routing uses
    the maximum active/KV/state utilization, then oldest assignment for ties.
    There is no fixed global user cap: total capacity is the sum of registered
    replicas' finite capacities. Existing leases may finish during draining.
    """
    def __init__(self, pins: Pins, *, max_waiters: int = 1024,
                 max_wait_seconds: float = 30.0, heartbeat_ttl: float = 15.0,
                 custody: Custody | None = None):
        if not isinstance(pins, Pins):
            raise ValueError("pins must be Pins")
        _integer(max_waiters, "max_waiters")
        self._pins = pins
        self.max_waiters = max_waiters
        self.max_wait_seconds = _seconds(max_wait_seconds, "max_wait_seconds")
        self.heartbeat_ttl = _seconds(heartbeat_ttl, "heartbeat_ttl")
        self._custody = custody if custody is not None else MemoryCustody()
        self._cv = threading.Condition(threading.RLock())
        self._replicas: dict[str, _Replica] = {}
        self._leases: dict[str, Lease] = {}
        self._waiters: deque[_Waiter] = deque()
        self._waiting_ids: set[str] = set()
        self._assignment = 0

    @property
    def pins(self) -> Pins:
        return self._pins

    def register(self, spec: ReplicaSpec) -> None:
        if not isinstance(spec, ReplicaSpec):
            raise ValueError("spec must be ReplicaSpec")
        if spec.identity.pins != self.pins:
            raise IdentityMismatch("replica pins differ from pool pins")
        with self._cv:
            if spec.identity.replica_id in self._replicas:
                raise PoolError("replica ID already registered; drain and remove first")
            if any(r.spec.identity.endpoint == spec.identity.endpoint for r in self._replicas.values()):
                raise PoolError("endpoint already registered; capacity must not be double counted")
            self._replicas[spec.identity.replica_id] = _Replica(spec, time.monotonic())
            self._cv.notify_all()

    def heartbeat(self, identity: ReplicaIdentity, *, healthy: bool = True) -> None:
        if not isinstance(identity, ReplicaIdentity) or type(healthy) is not bool:
            raise ValueError("expected replica identity and boolean health")
        with self._cv:
            replica = self._replicas[identity.replica_id]
            if identity != replica.spec.identity:
                replica.quarantined = True
                replica.healthy = False
                self._cv.notify_all()
                raise IdentityMismatch("replica identity drift; drain and remove before replacement")
            if replica.quarantined:
                raise IdentityMismatch("replica remains quarantined after identity drift")
            replica.healthy, replica.seen = healthy, time.monotonic()
            self._cv.notify_all()

    def drain(self, replica_id: str) -> None:
        with self._cv:
            self._replicas[replica_id].draining = True
            self._cv.notify_all()

    def remove(self, replica_id: str) -> None:
        with self._cv:
            replica = self._replicas[replica_id]
            if not replica.draining or replica.active:
                raise PoolError("replica must be drained and all leases retired before removal")
            del self._replicas[replica_id]
            self._cv.notify_all()

    def _usable(self, replica: _Replica, now: float) -> bool:
        return replica.healthy and not replica.quarantined and now - replica.seen < self.heartbeat_ttl

    def _check_replica(self, lease: Lease) -> None:
        with self._cv:
            if self._leases.get(lease.request_id) is not lease or not self._usable(lease._replica, time.monotonic()):
                raise ReplicaUnavailable("lease owner is stale, unhealthy, quarantined, or retired")

    @staticmethod
    def _fits(replica: _Replica, kv: int, state: int) -> bool:
        spec = replica.spec
        return (replica.active < spec.max_active and replica.kv + kv <= spec.kv_capacity_bytes
                and replica.state + state <= spec.state_capacity_bytes)

    @staticmethod
    def _load(replica: _Replica):
        spec = replica.spec
        return (max(Fraction(replica.active, spec.max_active),
                    Fraction(replica.kv, max(1, spec.kv_capacity_bytes)),
                    Fraction(replica.state, max(1, spec.state_capacity_bytes))),
                replica.last_assignment, spec.identity.replica_id)

    def acquire(self, request_id: str, *, kv_bytes: int, state_bytes: int = 0,
                timeout: float = 0, cancel_event: threading.Event | None = None) -> Lease:
        _name(request_id, "request_id")
        _integer(kv_bytes, "kv_bytes")
        _integer(state_bytes, "state_bytes")
        timeout = _seconds(timeout, "timeout", zero=True)
        if timeout > self.max_wait_seconds:
            raise ValueError("timeout exceeds pool's bounded wait limit")
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise TypeError("cancel_event must be threading.Event")
        deadline = time.monotonic() + timeout
        waiter = _Waiter(request_id, kv_bytes, state_bytes)
        with self._cv:
            if request_id in self._leases or request_id in self._waiting_ids:
                raise DuplicateRequest(request_id)
            queued = False
            try:
                while True:
                    if waiter.cancelled or (cancel_event is not None and cancel_event.is_set()):
                        raise RequestCancelled(request_id)
                    now = time.monotonic()
                    if queued and now >= deadline:
                        raise Backpressure("admission deadline expired")
                    # A waiter may only bypass the queue when it is the queue head.
                    if not self._waiters or (queued and self._waiters[0] is waiter):
                        eligible = [r for r in self._replicas.values()
                                    if not r.draining and self._usable(r, now) and self._fits(r, kv_bytes, state_bytes)]
                        if eligible:
                            replica = min(eligible, key=self._load)
                            claimed = self._custody.claim(request_id)
                            if claimed is False:
                                raise DuplicateRequest(request_id)
                            if claimed is not True:
                                raise PoolError("custody claim must return a boolean")
                            # Durable claims may perform I/O. Never grant a stale or
                            # cancelled lease after that call; the spent ID remains
                            # retired even if the following checks reject admission.
                            if cancel_event is not None and cancel_event.is_set():
                                raise RequestCancelled(request_id)
                            now = time.monotonic()
                            if not self._usable(replica, now):
                                raise ReplicaUnavailable("replica became stale during custody claim")
                            if queued and now >= deadline:
                                raise Backpressure("admission deadline expired during custody claim")
                            lease = Lease(self, request_id, replica, kv_bytes, state_bytes, cancel_event)
                            replica.active += 1
                            replica.kv += kv_bytes
                            replica.state += state_bytes
                            self._assignment += 1
                            replica.last_assignment = self._assignment
                            self._leases[request_id] = lease
                            return lease
                    remaining = deadline - now
                    if remaining <= 0:
                        raise Backpressure("no healthy replica capacity before admission deadline")
                    if not queued:
                        if len(self._waiters) >= self.max_waiters:
                            raise Backpressure("bounded admission queue is full")
                        self._waiters.append(waiter)
                        self._waiting_ids.add(request_id)
                        queued = True
                    # Cancellation Events do not notify the condition themselves.
                    self._cv.wait(min(remaining, 0.05) if cancel_event is not None else remaining)
            finally:
                if queued:
                    self._waiters.remove(waiter)
                    self._waiting_ids.remove(request_id)
                    self._cv.notify_all()

    def _retire(self, lease: Lease) -> None:
        with self._cv:
            if self._leases.get(lease.request_id) is not lease:
                raise PoolError("lease ownership mismatch on retirement")
            replica = lease._replica
            replica.active -= 1
            replica.kv -= lease._kv
            replica.state -= lease._state
            del self._leases[lease.request_id]
            self._cv.notify_all()

    def cancel(self, request_id: str) -> bool:
        with self._cv:
            for waiter in self._waiters:
                if waiter.request_id == request_id:
                    waiter.cancelled = True
                    self._cv.notify_all()
                    return True
            lease = self._leases.get(request_id)
        # Never hold pool lock while waiting for a lease's publication/cleanup lock.
        return lease.cancel() if lease is not None else False

    def release(self, request_id: str) -> bool:
        """Retire the active request only after its synchronous cleanup succeeds."""
        with self._cv:
            lease = self._leases.get(request_id)
        return lease.close() if lease is not None else False

    def run(self, request_id: str, adapter: Callable[[Lease], T], **admission) -> T:
        """Call a synchronous dispatcher adapter with one lease; always clean up.

        Adapters must complete their work before returning (do not return a lazy
        generator or background task). Streaming adapters publish inside the call.
        """
        if not callable(adapter):
            raise TypeError("adapter must be callable")
        with self.acquire(request_id, **admission) as lease:
            result = adapter(lease)
            if isinstance(result, Iterator) or inspect.isawaitable(result) or inspect.isasyncgen(result):
                if inspect.isgenerator(result) or inspect.iscoroutine(result):
                    result.close()
                raise TypeError("adapter must finish synchronously; publish streaming output inside adapter")
            return result

    def snapshot(self) -> tuple[ReplicaSnapshot, ...]:
        with self._cv:
            now = time.monotonic()
            return tuple(ReplicaSnapshot(r.spec.identity, r.active, r.kv, r.state,
                                         r.healthy, now - r.seen < self.heartbeat_ttl,
                                         r.draining, r.quarantined) for r in self._replicas.values())

    @property
    def waiting(self) -> int:
        with self._cv:
            return len(self._waiters)
