"""Experimental BHE1 byte-sign/mantissa plus Huffman exponent N64 packets."""
from __future__ import annotations

import hashlib
import struct
import threading
import weakref
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

from . import context_tiled_packet as tiled
from . import context_exponent_tables as exponent_tables
import os

PacketError = tiled.PacketError
HEADER = struct.Struct('<4sBBBBIIIIIIII32s32s32s')
MAX_BYTES = 32 * 1048576
FORMAT = 'BHE1-experimental-N64-Huffman-exponent-v1'
_NATIVE_CPU_OVERRIDE=ContextVar('bhe_native_cpu_override',default=None)


# A witness is deliberately process-local and unpickleable.  It is a narrow
# opt-in proof that this exact immutable wire object has already gone through
# the CPU decoder.  Keeping the wire object alive prevents id reuse; no dense
# decoded array is retained.
_VALIDATION_WITNESS_TOKEN = object()


def _witness_metadata(info):
    return (int(info['rows']), int(info['cols']), int(info['row_classes']),
            int(info['col_classes']), int(info['split_k']), int(info['flags']),
            _hash(bytes(info['labels'])), _hash(bytes(info['cdf'])),
            str(info['source_digest']),
            str(info['source_cdf_digest']), str(info['map_digest']))


class _ValidationWitness:
    """Opaque proof of one completed exact CPU source validation."""

    __slots__ = ('_blob', '_blob_id', '_blob_digest', '_metadata',
                 '_source_digest', '_token', '_pid', '_sealed')

    def __init__(self, *args, **kwargs):
        raise TypeError('BHE1 validation witnesses are minted internally')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('BHE1 validation witnesses are immutable')
        object.__setattr__(self, name, value)

    def __reduce__(self):
        raise TypeError('BHE1 validation witnesses cannot be serialized')

    def _matches(self, blob, info):
        # Identity is intentional: an equal copied bytes object is validated
        # again, since the witness is scoped to the object that was decoded.
        try:
            return (type(self) is _ValidationWitness and self._token is _VALIDATION_WITNESS_TOKEN
                    and os.getpid() == self._pid
                    and type(blob) is bytes and blob is self._blob and id(blob) == self._blob_id
                    and self._blob_digest == _hash(blob)
                    and self._metadata == _witness_metadata(info)
                    and self._source_digest == str(info['source_digest']))
        except (AttributeError, TypeError):
            return False


@contextmanager
def native_cpu_backend(enabled=True):
    """Select the exact native CPU codec for this context without changing process flags."""
    if type(enabled) is not bool:raise ValueError('native CPU selection must be boolean')
    token=_NATIVE_CPU_OVERRIDE.set(enabled)
    try:yield
    finally:_NATIVE_CPU_OVERRIDE.reset(token)


def _hash(data):
    return hashlib.sha256(data).digest()


def _hash_parts(*parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.digest()


def _context_labels_valid(labels, nr, nc):
    row_max = int(np.frombuffer(labels, dtype=np.uint8, count=64).max(initial=0))
    column_max = int(np.frombuffer(labels, dtype=np.uint8, offset=64).max(initial=0))
    return row_max < nr and column_max < nc


def _labels(values, count, classes):
    try:
        a = np.frombuffer(values, dtype=np.uint8) if isinstance(values, (bytes, bytearray, memoryview)) else np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise PacketError('invalid supplied labels') from exc
    if a.shape != (count,) or a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a >= classes):
        raise PacketError('invalid supplied labels')
    return np.ascontiguousarray(a, dtype=np.uint8)


def _cdf(data, nr, nc):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise PacketError('sparse CDF must be bytes')
    return _checked_cdf_bytes(bytes(data), nr, nc)


_CDF_CACHE_MAX_ENTRIES = 32
_CDF_CACHE_MAX_BYTES = 8 * 1024 * 1024
# For small CDFs, the OrderedDict lookup is cheaper than the weakref probe;
# large serialized CDFs are the contention case where skipping the shared lock
# also avoids repeatedly hashing the payload.
_CDF_LOCAL_FASTPATH_MIN_BYTES = 4096
_CDF_MAX_LOCAL_STATES = 64
_CDFCacheInfo = namedtuple('CDFCacheInfo', 'hits misses maxsize currsize')


class _CDFCacheEntry:
    __slots__ = ('data', 'digest', 'validated', 'nr', 'nc', 'active', '__weakref__')

    def __init__(self, data, nr, nc, *, digest=None, validated=False):
        self.data = data
        self.digest = digest
        self.validated = validated
        self.nr = nr
        self.nc = nc
        self.active = True


class _CDFThreadState:
    __slots__ = ('refs', 'hits', 'digest_hits', 'generation', 'thread_ref')

    def __init__(self, generation, thread_ref):
        self.refs = []
        self.hits = 0
        self.digest_hits = 0
        self.generation = generation
        self.thread_ref = thread_ref


class _CDFValidationCache:
    """Bounded cache for exact CDF validation results and SHA-256 digests."""

    def __init__(self, maxsize, max_bytes):
        self.maxsize = maxsize
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._digest_hits = 0
        self._digest_misses = 0
        self._fast_hits = 0
        self._fast_digest_hits = 0
        self._generation = 0
        self._lock = threading.RLock()
        self._local = threading.local()
        self._thread_states = []

    def _thread_state(self):
        state = getattr(self._local, 'state', None)
        if state is False:
            return None
        if state is None:
            with self._lock:
                self._prune_thread_states_locked()
                if len(self._thread_states) >= _CDF_MAX_LOCAL_STATES:
                    self._local.state = False
                    return None
                state = _CDFThreadState(
                    self._generation,
                    weakref.ref(threading.current_thread()),
                )
                self._thread_states.append(state)
            self._local.state = state
        elif state.generation != self._generation:
            # cache_clear() only changes this thread's state when it next
            # calls the cache. Foreign thread-local lists remain owner-only.
            with self._lock:
                self._sync_state_generation_locked(state)
        return state

    def _sync_state_generation_locked(self, state):
        if state.generation == self._generation:
            return
        state.refs.clear()
        state.hits = 0
        state.digest_hits = 0
        state.generation = self._generation

    def _prune_thread_states_locked(self):
        live = []
        for state in self._thread_states:
            thread = state.thread_ref()
            if thread is not None and thread.is_alive():
                live.append(state)
                continue
            if state.generation == self._generation:
                self._fast_hits += state.hits
                self._fast_digest_hits += state.digest_hits
        self._thread_states = live

    @staticmethod
    def _remember(state, entry):
        refs = state.refs
        refs[:] = [ref for ref in refs
                   if (current := ref()) is not None and
                   current.active and current is not entry]
        refs.insert(0, weakref.ref(entry))
        del refs[2:]

    @staticmethod
    def _local_entry(state, data, nr, nc):
        for index, ref in enumerate(state.refs):
            entry = ref()
            if (entry is None or not entry.active or entry.nr != nr or
                    entry.nc != nc):
                continue
            if entry.data is data or entry.data == data:
                if index:
                    state.refs.insert(0, state.refs.pop(index))
                return entry
        return None

    def _flush_local_lru(self, state):
        # Warm hits defer LRU promotion; fold the thread's two recent weakrefs
        # into the shared order only when a caller already needs the lock.
        for ref in reversed(state.refs):
            entry = ref()
            if entry is None or not entry.active:
                continue
            key = (entry.data, entry.nr, entry.nc)
            if self._entries.get(key) is entry:
                self._entries.move_to_end(key)

    def _store(self, key, data, *, digest=None, validated=False):
        if len(data) > self.max_bytes:
            return None
        entry = _CDFCacheEntry(data, key[1], key[2], digest=digest,
                               validated=validated)
        self._entries[key] = entry
        self._bytes += len(data)
        self._entries.move_to_end(key)
        while (len(self._entries) > self.maxsize or
               self._bytes > self.max_bytes):
            _, evicted = self._entries.popitem(last=False)
            evicted.active = False
            self._bytes -= len(evicted.data)
        return entry

    def __call__(self, data, nr, nc):
        key = (data, nr, nc)
        state = (self._thread_state()
                 if len(data) >= _CDF_LOCAL_FASTPATH_MIN_BYTES else None)
        cached = (self._local_entry(state, data, nr, nc)
                  if state is not None else None)
        if cached is not None and cached.validated:
            state.hits += 1
            return cached.data
        with self._lock:
            if state is not None:
                self._sync_state_generation_locked(state)
                self._flush_local_lru(state)
            cached = self._entries.get(key)
            if cached is not None and cached.validated:
                self._entries.move_to_end(key)
                self._hits += 1
                if state is not None:
                    self._remember(state, cached)
                return cached.data
            self._misses += 1
            try:
                # Builder validates exact serialized sparse cumulatives without recounting.
                exponent_tables._sparse_cdf(data, nr, nc)
            except ValueError as exc:
                raise PacketError(str(exc)) from exc
            if cached is None:
                cached = self._store(key, data, validated=True)
            else:
                cached.validated = True
                self._entries.move_to_end(key)
            if cached is not None and state is not None:
                self._remember(state, cached)
            return data

    def digest(self, data, nr, nc):
        """Return the exact-CDF digest without implying sparse-CDF validity."""
        data = bytes(data)
        key = (data, nr, nc)
        state = (self._thread_state()
                 if len(data) >= _CDF_LOCAL_FASTPATH_MIN_BYTES else None)
        cached = (self._local_entry(state, data, nr, nc)
                  if state is not None else None)
        if cached is not None and cached.digest is not None:
            state.digest_hits += 1
            return cached.digest
        with self._lock:
            if state is not None:
                self._sync_state_generation_locked(state)
                self._flush_local_lru(state)
            cached = self._entries.get(key)
            if cached is not None and cached.digest is not None:
                self._entries.move_to_end(key)
                self._digest_hits += 1
                if state is not None:
                    self._remember(state, cached)
                return cached.digest
            self._digest_misses += 1
            digest = _hash(data)
            if cached is None:
                cached = self._store(key, data, digest=digest)
            else:
                cached.digest = digest
                self._entries.move_to_end(key)
            if cached is not None and state is not None:
                self._remember(state, cached)
            return digest

    def cache_clear(self):
        with self._lock:
            for entry in self._entries.values():
                entry.active = False
            self._entries.clear()
            self._bytes = 0
            self._hits = 0
            self._misses = 0
            self._digest_hits = 0
            self._digest_misses = 0
            self._fast_hits = 0
            self._fast_digest_hits = 0
            self._generation += 1

    def cache_info(self):
        with self._lock:
            self._prune_thread_states_locked()
            fast_hits = self._fast_hits + sum(
                state.hits for state in self._thread_states
                if state.generation == self._generation)
            return _CDFCacheInfo(self._hits + fast_hits, self._misses, self.maxsize,
                                 len(self._entries))

    def _digest_stats(self):
        with self._lock:
            self._prune_thread_states_locked()
            fast_hits = self._fast_digest_hits + sum(
                state.digest_hits for state in self._thread_states
                if state.generation == self._generation)
            return self._digest_hits + fast_hits, self._digest_misses

    def cache_parameters(self):
        return {'maxsize': self.maxsize, 'typed': False}


_checked_cdf_bytes = _CDFValidationCache(_CDF_CACHE_MAX_ENTRIES,
                                          _CDF_CACHE_MAX_BYTES)


def _cdf_digest(data, nr, nc):
    return _checked_cdf_bytes.digest(data, nr, nc)


def _pack_stream(symbols, mapping):
    acc = 0
    nbits = 0
    out = bytearray()
    for symbol in symbols:
        try:
            code, length = mapping[int(symbol)]
        except KeyError as exc:
            raise PacketError('exponent absent from exact source CDF') from exc
        acc = (acc << length) | code
        nbits += length
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 255)
            acc &= (1 << nbits) - 1 if nbits else 0
    if nbits:
        out.append((acc << (8 - nbits)) & 255)
    return bytes(out), nbits


def _unpack_stream(main, start, end, count, codes):
    if start == end:
        if count:
            # Singleton contexts have zero-length codes; empty contexts do not.
            return None
        return []
    if not codes:
        raise PacketError('nonempty stream has an empty exponent context')
    tree = {}
    for code, length, symbol in codes:
        node = tree
        for bit in range(length - 1, -1, -1):
            if 's' in node:
                raise PacketError('invalid canonical exponent tree')
            node = node.setdefault((code >> bit) & 1, {})
        if node:
            raise PacketError('invalid canonical exponent tree')
        node['s'] = symbol
    result = []
    cursor = start
    for _ in range(count):
        node = tree
        while 's' not in node:
            if cursor >= end:
                raise PacketError('truncated exponent stream')
            bit = (main[cursor >> 3] >> (7 - (cursor & 7))) & 1
            cursor += 1
            node = node.get(bit)
            if node is None:
                raise PacketError('invalid exponent codeword')
        result.append(node['s'])
    if cursor != end:
        raise PacketError('nonterminal exponent stream')
    return result


def _identities(source, labels, nr, nc, k, split, flags, cdf, map_digest):
    # Explicit geometry/labels bind the mapping used by native consumers.
    return _hash(struct.pack('<8I', 64, k, split, nr, nc, flags, len(labels), len(cdf)) +
                 labels + _cdf_digest(cdf, nr, nc) + map_digest + source)


def _native_cpu_enabled():
    override=_NATIVE_CPU_OVERRIDE.get()
    if override is not None:return override
    flag = os.environ.get('GEOREFINE_BHE1_NATIVE_CPU', '0')
    if flag not in ('0', '1'):
        raise PacketError('GEOREFINE_BHE1_NATIVE_CPU must be0or1')
    return flag == '1'


def _tables_for(cdf, nr, nc, cache):
    if cache is None:
        return exponent_tables.build(cdf, nr, nc)
    if type(cache) is not dict:
        raise PacketError('invalid private exponent table cache')
    cdf = bytes(cdf)
    key = (_cdf_digest(cdf, nr, nc), nr, nc, exponent_tables.BUILDER_VERSION)
    cached = cache.get(key)
    if cached is not None:
        if cached[0] != cdf:
            raise PacketError('cached exact CDF bytes mismatch')
        return cached[1]
    if len(cache) >= 2:
        raise PacketError('private exponent table cache entry limit exceeded')
    tables = exponent_tables.build(cdf, nr, nc)
    cache[key] = (cdf, tables)
    return tables


def _encode_main_python(exponents, rc, cc, tables, nc):
    offsets = [0]
    main = bytearray()
    # Offsets are bit offsets; append streams without internal byte padding.
    allbits = 0
    bitbuf = 0
    nbuf = 0
    mappings = []
    coding_possible = True
    for context_codes in tables.codes:
        mappings.append({s: (code, length) for code, length, s in context_codes})
    for row in range(64):
        for col_class in range(nc):
            cx = int(rc[row]) * nc + col_class
            cols = np.flatnonzero(cc == col_class)
            stream = []
            mapping = mappings[cx]
            singleton = tables.singletons[cx]
            for col in cols:
                symbol = int(exponents[row, col])
                if singleton == symbol and not mapping:
                    continue
                stream.append(symbol)
            if not mapping:
                if stream:
                    coding_possible = False
                    break
                offsets.append(allbits)
                continue
            for symbol in stream:
                if symbol not in mapping:
                    coding_possible = False
                    break
                code, length = mapping[symbol]
                bitbuf = (bitbuf << length) | code
                nbuf += length
                allbits += length
                while nbuf >= 8:
                    nbuf -= 8
                    main.append((bitbuf >> nbuf) & 255)
                    bitbuf &= (1 << nbuf) - 1 if nbuf else 0
            if not coding_possible:
                break
            offsets.append(allbits)
        if not coding_possible:
            break
    if nbuf:
        main.append((bitbuf << (8 - nbuf)) & 255)
    return main, offsets, coding_possible


def encode(words, *, split_k=1, row_classes=16, col_classes=4,
           row_labels=None, col_labels=None, sparse_cdf=None,
           experimental_large_geometry=False, shared_metadata=False, _table_cache=None):
    native_cpu = _native_cpu_enabled()
    if type(experimental_large_geometry) is not bool or type(shared_metadata) is not bool:
        raise PacketError('experimental geometry and shared metadata flags must be bool')
    a = np.asarray(words)
    valid_k = (a.ndim == 2 and a.shape[0] == 64 and
               (a.shape[1] in (512, 1024) or (experimental_large_geometry and
                128 <= a.shape[1] <= 32768 and a.shape[1] % 128 == 0)))
    if not valid_k or a.dtype.kind != 'u' or a.dtype.itemsize != 2:
        raise PacketError('BHE1 requires uint16 N64,K512/1024 or explicit bounded large geometry')
    if type(split_k) is not int or split_k != 1:
        raise PacketError('BHE1 requires split_k=1')
    nr, nc = row_classes, col_classes
    if (type(nr) is not int or not 1 <= nr <= 256 or type(nc) is not int or
            nc not in (1, 2, 4)):
        raise PacketError('invalid context classes')
    a = np.ascontiguousarray(a, dtype='<u2')
    k = int(a.shape[1])
    supplied = (row_labels is not None, col_labels is not None, sparse_cdf is not None)
    if any(supplied) and not all(supplied):
        raise PacketError('supply row labels, column labels and sparse CDF together')
    if all(supplied):
        rc = _labels(row_labels, 64, nr)
        cc = _labels(col_labels, k, nc)
        cdf = _cdf(sparse_cdf, nr, nc)
    else:
        if nr not in tiled.codec.ROWS or nc not in tiled.codec.COLS:
            raise PacketError('automatic labels support row classes 1/4/16/64 and column classes 1/4/16; supply explicit labels and CDF')
        rc, cc = tiled.codec._labels(a, nr, nc)
        _, cdf = tiled._global_tables(a, rc, cc, nr, nc)
        cdf = bytes(cdf)
    labels = rc.tobytes() + cc.tobytes()
    tables = _tables_for(cdf, nr, nc, _table_cache)
    stream_count = 64 * nc
    exponents = (a >> 7).astype(np.uint8)
    # Coded compact bytes preserve sign and the low seven mantissa bits.
    if native_cpu:
        from .context_exponent_native import encode_main_native
        main, native_offsets, coding_possible = encode_main_native(
            exponents, rc, cc, tables, col_classes=nc)
        offsets = native_offsets.tolist()
        allbits = offsets[-1]
        raw8 = ((a & 127) | ((a >> 8) & 128)).astype(np.uint8).tobytes()
    else:
        main, offsets, coding_possible = _encode_main_python(exponents, rc, cc, tables, nc)
        allbits = offsets[-1]
        raw8 = bytes((((int(w) & 127) | ((int(w) >> 8) & 128)) for w in a.ravel()))
    raw16 = a.tobytes()
    flags = int(shared_metadata)
    cdf_wire = cdf
    map_digest = tables.digest
    if coding_possible:
        if len(offsets) != stream_count + 1 or allbits > 0xffffffff:
            raise PacketError('exponent stream directory overflow')
        offset_bytes = struct.pack('<' + 'I' * len(offsets), *offsets)
        coded_size = len(raw8) + len(main) + len(offset_bytes)
        if not shared_metadata:
            coded_size += HEADER.size + len(labels) + len(cdf_wire) + 32
        raw_mode = coded_size >= len(raw16)
    else:
        offset_bytes = struct.pack('<' + 'I' * (stream_count + 1), *([0] * (stream_count + 1)))
        raw_mode = True
    if raw_mode:
        flags |= 2
        main = bytearray()
        offsets = [0] * (stream_count + 1)
        offset_bytes = struct.pack('<' + 'I' * len(offsets), *([0] * len(offsets)))
        raw_data = raw16
        if not shared_metadata:
            cdf_wire = b''
            map_digest = _hash(b'BHE1-empty-exponent-map-v1')
    else:
        raw_data = raw8
    source_digest = _hash(raw16)
    cdf_digest = _cdf_digest(cdf_wire, nr, nc)
    identity = _identities(source_digest, labels, nr, nc, k, split_k, flags, cdf_wire, map_digest)
    # The outer checksum includes the source/geometry/metadata identity.
    # Encode row-class count minus one so all 256 uint8 row labels are representable.
    body = b''.join((HEADER.pack(b'BHE1', 1, nr - 1, nc, 64, k, split_k, flags,
                        len(labels), len(cdf_wire), len(offsets), len(main), len(raw_data),
                        source_digest, cdf_digest, map_digest),
            labels, cdf_wire, offset_bytes, bytes(main), raw_data))
    return body + _hash_parts(body, identity)


def _parse(blob, *, _table_cache=None):
    if not isinstance(blob, (bytes, bytearray, memoryview)) or not HEADER.size + 32 <= memoryview(blob).nbytes <= MAX_BYTES:
        raise PacketError('invalid bounded BHE1 bundle size')
    blob = bytes(blob)
    (magic, ver, nr_field, nc, rows, k, split, flags, nl, ncdf, noff, nmain, nraw,
     source_digest, cdf_digest, map_digest) = HEADER.unpack_from(blob)
    nr = nr_field + 1
    if (magic != b'BHE1' or ver != 1 or rows != 64 or not 1 <= nr <= 256 or
            nc not in (1, 2, 4) or split != 1 or flags & ~3 or
            not (k in (512, 1024) or 128 <= k <= 32768 and k % 128 == 0) or nl != 64 + k or
            noff != 64 * nc + 1):
        raise PacketError('unsupported BHE1 geometry or flags')
    start = HEADER.size
    label_end = start + nl
    cdf_end = label_end + ncdf
    off_end = cdf_end + 4 * noff
    main_end = off_end + nmain
    end = main_end + nraw
    if end + 32 != len(blob) or _hash_parts(memoryview(blob)[:end], _identities(source_digest, blob[start:label_end], nr, nc, k, split, flags, blob[label_end:cdf_end], map_digest)) != blob[end:]:
        raise PacketError('BHE1 section lengths or checksum mismatch')
    labels = blob[start:label_end]
    if not _context_labels_valid(labels, nr, nc):
        raise PacketError('invalid BHE1 context labels')
    cdf = blob[label_end:cdf_end]
    if _cdf_digest(cdf, nr, nc) != cdf_digest:
        raise PacketError('source CDF digest mismatch')
    offsets = struct.unpack_from('<' + 'I' * noff, blob, cdf_end)
    main = blob[off_end:main_end]
    raw = blob[main_end:end]
    raw_mode = bool(flags & 2)
    shared = bool(flags & 1)
    if any(a > b for a, b in zip(offsets, offsets[1:])):
        raise PacketError('invalid exponent bit directory')
    if raw_mode:
        if nmain or any(offsets) or nraw != 128 * k or (ncdf == 0) == shared:
            raise PacketError('invalid raw fallback sections')
        if not shared and ncdf:
            raise PacketError('unexpected nonshared raw CDF')
        if not ncdf and (cdf_digest != _hash(b'') or map_digest != _hash(b'BHE1-empty-exponent-map-v1')):
            raise PacketError('invalid empty raw metadata digest')
        if ncdf:
            tables = _tables_for(cdf, nr, nc, _table_cache)
            if tables.digest != map_digest:
                raise PacketError('exponent map digest mismatch')
        else:
            tables = None
    else:
        if nraw != 64 * k or not ncdf or offsets[0] != 0 or offsets[-1] > 8 * nmain:
            raise PacketError('invalid coded section lengths')
        tables = _tables_for(cdf, nr, nc, _table_cache)
        if tables.digest != map_digest:
            raise PacketError('exponent map digest mismatch')
        bits = offsets[-1]
        if nmain != (bits + 7) // 8:
            raise PacketError('noncanonical exponent byte length')
        if bits & 7 and main and main[-1] & ((1 << (8 - (bits & 7))) - 1):
            raise PacketError('nonzero unused terminal bits')
    return blob, dict(rows=rows, cols=k, row_classes=nr, col_classes=nc, split_k=split,
        flags=flags, shared_metadata=shared, raw_mode=raw_mode, labels=labels, cdf=cdf,
        source_digest=source_digest.hex(), source_cdf_digest=cdf_digest.hex(),
        map_digest=map_digest.hex(), offsets=offsets, main_start=off_end, main_bytes=nmain,
        raw_start=main_end, raw_bytes=nraw, tables=tables, identity=_identities(
            source_digest, labels, nr, nc, k, split, flags, cdf, map_digest).hex())


def _decode(blob, info):
    native_cpu = _native_cpu_enabled()
    k, nc = info['cols'], info['col_classes']
    labels = info['labels']
    rc = np.frombuffer(labels[:64], dtype=np.uint8)
    cc = np.frombuffer(labels[64:], dtype=np.uint8)
    raw = blob[info['raw_start']:info['raw_start'] + info['raw_bytes']]
    if info['raw_mode']:
        out = np.frombuffer(raw, dtype='<u2').copy().reshape((64, k))
    elif native_cpu:
        from .context_exponent_native import decode_main_native
        main = blob[info['main_start']:info['main_start'] + info['main_bytes']]
        try:
            out = decode_main_native(main, np.array(info['offsets'], dtype='<u4'),
                np.frombuffer(raw, dtype=np.uint8).reshape((64,k)), rc, cc,
                info['tables'], col_classes=nc)
        except ValueError as exc:
            raise PacketError(str(exc)) from exc
    else:
        out = np.empty((64, k), dtype='<u2')
        main = blob[info['main_start']:info['main_start'] + info['main_bytes']]
        raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape((64, k))
        for row in range(64):
            for col_class in range(nc):
                stream_index = row * nc + col_class
                lo, hi = info['offsets'][stream_index:stream_index + 2]
                cols = np.flatnonzero(cc == col_class)
                cx = int(rc[row]) * nc + col_class
                singleton = info['tables'].singletons[cx]
                codes = info['tables'].codes[cx]
                if singleton == -2:
                    if len(cols):
                        raise PacketError('columns use an empty exponent context')
                    exps = _unpack_stream(main, lo, hi, len(cols), codes)
                    if exps:
                        raise PacketError('empty exponent context carries symbols')
                    continue
                if singleton >= 0:
                    if lo != hi:
                        raise PacketError('singleton exponent stream must be empty')
                    exps = [singleton] * len(cols)
                else:
                    exps = _unpack_stream(main, lo, hi, len(cols), codes)
                if exps is None or len(exps) != len(cols):
                    raise PacketError('exponent stream/context count mismatch')
                for col, exponent in zip(cols, exps):
                    b = int(raw_arr[row, col])
                    out[row, col] = (int(exponent) << 7) | (b & 127) | ((b & 128) << 8)
    if _hash(out.astype('<u2', copy=False).tobytes()).hex() != info['source_digest']:
        raise PacketError('decoded source digest mismatch')
    return out


def decode_cpu_with_witness(blob, *, source=None, _table_cache=None):
    """Decode one panel and return its source plus an opt-in validation proof."""
    if type(blob) is not bytes:
        raise PacketError('BHE1 validation witnesses require immutable bytes')
    blob, info = _parse(blob, _table_cache=_table_cache)
    decoded = _decode(blob, info)
    if not isinstance(decoded, np.ndarray) or decoded.dtype != np.dtype('<u2'):
        raise PacketError('BHE1 validation witness source has invalid dtype')
    if tuple(decoded.shape) != (64, int(info['cols'])):
        raise PacketError('BHE1 validation witness source has invalid geometry')
    if source is not None:
        expected = np.asarray(source)
        if expected.dtype != np.dtype('<u2') or expected.shape != decoded.shape:
            raise PacketError('BHE1 validation witness source roundtrip mismatch')
        if expected.tobytes() != decoded.tobytes():
            raise PacketError('BHE1 validation witness source roundtrip mismatch')
    witness = object.__new__(_ValidationWitness)
    object.__setattr__(witness, '_blob', blob)
    object.__setattr__(witness, '_blob_id', id(blob))
    object.__setattr__(witness, '_blob_digest', _hash(blob))
    object.__setattr__(witness, '_metadata', _witness_metadata(info))
    object.__setattr__(witness, '_source_digest', str(info['source_digest']))
    object.__setattr__(witness, '_token', _VALIDATION_WITNESS_TOKEN)
    object.__setattr__(witness, '_pid', os.getpid())
    object.__setattr__(witness, '_sealed', True)
    return decoded, witness


def mint_validation_witness(blob, *, source=None, _table_cache=None):
    """Mint a process-only proof after exact CPU decode and source roundtrip."""
    _, witness = decode_cpu_with_witness(blob, source=source, _table_cache=_table_cache)
    return witness


def decode_cpu(blob, *, _table_cache=None):
    blob, info = _parse(blob, _table_cache=_table_cache)
    return _decode(blob, info)


def inspect(blob):
    blob, info = _parse(blob)
    return dict(format=FORMAT, rows=info['rows'], cols=info['cols'],
        row_classes=info['row_classes'], col_classes=info['col_classes'],
        raw_fallback=info['raw_mode'], shared_metadata=info['shared_metadata'],
        total_bytes=len(blob), source_bytes=128 * info['cols'],
        source_digest=info['source_digest'], source_cdf_digest=info['source_cdf_digest'],
        map_digest=info['map_digest'])


def _resident_build(panel_items, *, matrix_mode, validation_witnesses=None):
    first_blob, first = panel_items[0]
    k, nr, nc = first['cols'], first['row_classes'], first['col_classes']
    binding = (k, nr, nc, first['cdf'], first['map_digest'], first['labels'][64:])
    decoded_bodies = {}
    if validation_witnesses is not None and len(validation_witnesses) != len(panel_items):
        raise PacketError('BHE1 validation witness count must match panel count')
    witness_reused = full_decodes = 0
    for index, (blob, info) in enumerate(panel_items):
        current = (info['cols'], info['row_classes'], info['col_classes'], info['cdf'],
                   info['map_digest'], info['labels'][64:])
        if current != binding:
            raise PacketError('matrix panels require identical source CDF, map, geometry and column metadata')
        body_key = _hash(blob)
        prior = decoded_bodies.get(body_key)
        if prior is not None:
            if prior != blob:
                raise PacketError('panel body digest collision')
        else:
            witness = (validation_witnesses[index]
                       if validation_witnesses is not None else None)
            if not (isinstance(witness, _ValidationWitness)
                    and witness._matches(blob, info)):
                _decode(blob, info)
                full_decodes += 1
            else:
                witness_reused += 1
            decoded_bodies[body_key] = blob
    panel_count = len(panel_items)
    total_rows = 64 * panel_count
    row_labels = np.concatenate([np.frombuffer(info['labels'][:64], dtype=np.uint8) for _, info in panel_items])
    # Main data is aligned to four bytes per panel; one final eight-byte guard is part of its allocation.
    main_buf = bytearray()
    local_main_bases = []
    raw_panel_bases = []
    panel_modes = []
    for blob, info in panel_items:
        while len(main_buf) & 3:
            main_buf.append(0)
        local_main_bases.append(len(main_buf) * 8)
        panel_modes.append(int(info['raw_mode']))
        if not info['raw_mode']:
            main_buf.extend(blob[info['main_start']:info['main_start'] + info['main_bytes']])
    main_buf.extend(bytes(8))
    raw_buf = bytearray()
    raw_row_bases = []
    modes = []
    for blob, info in panel_items:
        raw = blob[info['raw_start']:info['raw_start'] + info['raw_bytes']]
        raw_panel_bases.append(len(raw_buf))
        raw_buf.extend(raw)
        row_width = 2 * k if info['raw_mode'] else k
        for row in range(64):
            raw_row_bases.append(len(raw_buf) - len(raw) + row * row_width)
            modes.append(int(info['raw_mode']))
    # Stable row-class grouping; each tile carries exactly 64 row slots.
    class_rows = {cls: [] for cls in range(nr)}
    for output_row, cls in enumerate(row_labels):
        class_rows[int(cls)].append(output_row)
    tile_row_ids = []
    tile_rc = []
    for cls in range(nr):
        rows = class_rows[cls]
        for pos in range(0, len(rows), 64):
            tile_row_ids.append(rows[pos:pos + 64])
            tile_rc.append(cls)
    tiles = len(tile_row_ids)
    row_desc = np.full((tiles, 64, 3), -1, dtype=np.int64)
    row_desc[:, :, 1:] = 0
    streams = np.zeros((tiles, 64, nc, 2), dtype=np.int64)
    for tile_index, ids in enumerate(tile_row_ids):
        for slot, output_row in enumerate(ids):
            panel_index, source_row = divmod(output_row, 64)
            blob, info = panel_items[panel_index]
            row_desc[tile_index, slot] = (output_row, raw_row_bases[output_row], modes[output_row])
            if not info['raw_mode']:
                for cc in range(nc):
                    index = source_row * nc + cc
                    streams[tile_index, slot, cc] = (
                        local_main_bases[panel_index] + info['offsets'][index],
                        local_main_bases[panel_index] + info['offsets'][index + 1])
    stream_offsets = np.zeros((panel_count, 64 * nc + 1), dtype=np.uint32)
    for panel_index, (_, info) in enumerate(panel_items):
        if not info['raw_mode']:
            stream_offsets[panel_index] = np.asarray(info['offsets'], dtype=np.uint32)
    panel_desc = np.asarray([
        (local_main_bases[i] // 8, raw_panel_bases[i], panel_modes[i])
        for i in range(panel_count)], dtype=np.int64)
    row_map = row_desc[:, :, 0].astype(np.int32)
    # Group each 128-column chunk by class, preserving ascending source positions.
    cc = np.frombuffer(first['labels'][64:], dtype=np.uint8)
    positions = np.empty(k, dtype=np.uint8)
    counts = np.zeros((k // 128, nc + 1), dtype=np.int64)
    for chunk in range(k // 128):
        base = 128 * chunk
        cursor = 0
        for cls in range(nc):
            chunk_positions = np.flatnonzero(cc[base:base + 128] == cls)
            counts[chunk, cls] = cursor
            positions[base + cursor:base + cursor + len(chunk_positions)] = chunk_positions.astype(np.uint8)
            cursor += len(chunk_positions)
        counts[chunk, nc] = cursor
        if cursor != 128:
            raise PacketError('column grouping does not cover 128-position chunk')
    tables = first['tables']
    source_digests = [info['source_digest'] for _, info in panel_items]
    matrix_source_digest = _hash(b''.join(bytes.fromhex(q) for q in source_digests)).hex()
    row_permutation_digest = _hash(row_desc[:, :, 0].astype('<i8', copy=False).tobytes()).hex()
    expanded_diagnostic_bytes = streams.nbytes + row_desc.nbytes
    compact_native_bytes = stream_offsets.nbytes + panel_desc.nbytes + row_map.nbytes
    arrays = dict(main=np.frombuffer(main_buf, dtype=np.uint8), raw=np.frombuffer(raw_buf, dtype=np.uint8),
        tables=tables.words.copy() if tables is not None else np.zeros(0, dtype=np.uint32),
        context_dir=tables.context_dir.copy() if tables is not None else np.zeros((nr * nc, 4), dtype=np.int64),
        streams=streams, row_desc=row_desc, stream_offsets=stream_offsets,
        panel_desc=panel_desc, row_map=row_map, tile_rc=np.asarray(tile_rc, dtype=np.int64),
        positions=positions, counts=counts, N=total_rows, K=k, nr=nr, nc=nc, tiles=tiles,
        panels=panel_count, matrix_mode=matrix_mode, host_wire_bytes=sum(len(b) for b, _ in panel_items),
        source_digests=source_digests, matrix_source_digest=matrix_source_digest,
        row_permutation_digest=row_permutation_digest, source_cdf_digest=first['source_cdf_digest'],
        map_digest=first['map_digest'], host_peak_decoded_row_bytes=128 * k,
        host_payload_bytes=len(main_buf) + len(raw_buf),
        context_table_bytes=tables.words.nbytes if tables is not None else 0,
        expanded_diagnostic_bytes=expanded_diagnostic_bytes,
        compact_native_array_bytes=compact_native_bytes,
        resident_array_bytes=(len(main_buf) + len(raw_buf) +
            (tables.words.nbytes + tables.context_dir.nbytes if tables is not None else nr * nc * 4 * 8) +
            streams.nbytes + row_desc.nbytes + stream_offsets.nbytes + panel_desc.nbytes +
            row_map.nbytes + positions.nbytes + counts.nbytes +
            np.asarray(tile_rc, dtype=np.int64).nbytes))
    if validation_witnesses is not None:
        arrays['validation_witnesses_reused'] = witness_reused
        arrays['validation_full_decodes'] = full_decodes
    return arrays


def _resident_arrays(blob, panels=1):
    if type(panels) is not int or not 1 <= panels <= 4096:
        raise PacketError('invalid panel replication count')
    blob, info = _parse(blob)
    return _resident_build([(blob, info)] * panels, matrix_mode=False)


def _resident_matrix_arrays(blobs, _table_cache=None, validation_witnesses=None):
    """Build resident arrays, optionally reusing exact CPU source witnesses.

    Framing, checksum, geometry, CDF, map and panel-order checks always run.
    A witness can only suppress the repeated decoded-source reconstruction for
    the exact immutable bytes object it was minted from.
    """
    if not isinstance(blobs, (list, tuple)) or not 1 <= len(blobs) <= 4096:
        raise PacketError('matrix requires 1..4096 distinct panel blobs')
    if validation_witnesses is not None:
        if (not isinstance(validation_witnesses, (list, tuple)) or
                len(validation_witnesses) != len(blobs)):
            raise PacketError('BHE1 validation witness count must match panel count')
    cache = {} if _table_cache is None else _table_cache
    parsed = [_parse(blob, _table_cache=cache) for blob in blobs]
    infos = [q[1] for q in parsed]
    first = infos[0]
    signature = (first['cols'], first['row_classes'], first['col_classes'], first['cdf'], first['map_digest'], first['labels'][64:])
    for blob, info in parsed:
        if (info['cols'], info['row_classes'], info['col_classes'], info['cdf'], info['map_digest'], info['labels'][64:]) != signature:
            raise PacketError('matrix panels require identical CDF, map, geometry and column labels')
    return _resident_build(parsed, matrix_mode=True,
                           validation_witnesses=validation_witnesses)
