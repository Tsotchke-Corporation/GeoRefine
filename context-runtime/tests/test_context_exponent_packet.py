import hashlib
import gc
import struct
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from georefine_context_runtime import context_exponent_packet as p
from georefine_context_runtime import context_exponent_tables as t


def fixture(k=512, kind='coded'):
    rng = np.random.default_rng(819)
    if kind == 'raw':
        return rng.integers(0, 65536, (64, k), dtype=np.uint16)
    exponents = rng.choice(np.array([1, 2], dtype=np.uint16), (64, k), p=[.8, .2])
    raw = rng.integers(0, 256, (64, k), dtype=np.uint16)
    return (exponents << 7) | (raw & 127) | ((raw & 128) << 8)


def exact_cdf(nr=1, nc=1):
    offsets = [0]
    entries = []
    for _ in range(nr * nc):
        entries.extend([(4, 52429), (8, 65536)])
        offsets.append(len(entries))
    return struct.pack('<' + 'I' * len(offsets), *offsets) + b''.join(
        struct.pack('<HI', *entry) for entry in entries)


def cdf_for_frequencies(frequencies):
    entries=[];total=0
    for symbol,frequency in enumerate(frequencies):
        if frequency:
            total+=frequency;entries.append((4*symbol,total))
    assert total==65536
    return struct.pack('<2I',0,len(entries))+b''.join(struct.pack('<HI',*entry) for entry in entries)


def cdf_variants(count):
    result = []
    for variant in range(count):
        offsets = [0]
        entries = []
        for context in range(16 * 4):
            upper = 24000 + variant * 97 + (context % 7)
            entries.extend(((4, upper), (8, 65536)))
            offsets.append(len(entries))
        result.append(struct.pack('<' + 'I' * len(offsets), *offsets) +
                     b''.join(struct.pack('<HI', *entry) for entry in entries))
    return tuple(result)


def large_cdf_variants(count):
    result = []
    for variant in range(count):
        frequencies = [256] * 256
        frequencies[0] += variant
        frequencies[1] -= variant
        entries = []
        cumulative = 0
        for symbol, frequency in enumerate(frequencies):
            cumulative += frequency
            entries.append((4 * symbol, cumulative))
        offsets = [j * len(entries) for j in range(16 * 4 + 1)]
        result.append(struct.pack('<65I', *offsets) + b''.join(
            struct.pack('<HI', *entry)
            for _ in range(16 * 4) for entry in entries))
    return tuple(result)


@pytest.mark.parametrize('k', [512, 1024])
def test_coded_roundtrip_stream_tails_and_resident_arrays(k):
    x = fixture(k)
    blob = p.encode(x, row_classes=1, col_classes=1)
    parsed, info = p._parse(blob)
    assert parsed == blob and not info['raw_mode']
    assert np.array_equal(p.decode_cpu(blob), x)
    assert info['offsets'][0] == 0
    assert info['offsets'][-1] > 0
    assert p.inspect(blob)['source_digest'] == hashlib.sha256(x.astype('<u2').tobytes()).hexdigest()
    arrays = p._resident_arrays(blob)
    assert arrays['main'].nbytes == info['main_bytes'] + 8
    assert arrays['raw'].nbytes == 64 * k
    assert arrays['tables'].dtype == np.uint32


def test_raw_fallback_and_large_geometry_optin():
    x = fixture(kind='raw')
    blob = p.encode(x, row_classes=1, col_classes=1)
    _, info = p._parse(blob)
    assert info['raw_mode']
    assert np.array_equal(p.decode_cpu(blob), x)
    small = fixture(128)
    with pytest.raises(p.PacketError):
        p.encode(small, row_classes=1, col_classes=1)
    blob = p.encode(small, row_classes=1, col_classes=1, experimental_large_geometry=True)
    assert np.array_equal(p.decode_cpu(blob), small)


def test_exact_cdf_out_of_support_uses_raw_without_recounting():
    x = np.full((64, 512), 9 << 7, dtype=np.uint16)
    cdf = struct.pack('<2I', 0, 1) + struct.pack('<HI', 4, 65536)
    blob = p.encode(x, row_classes=1, col_classes=1, row_labels=bytes(64),
                    col_labels=bytes(512), sparse_cdf=cdf, shared_metadata=True)
    _, info = p._parse(blob)
    assert info['raw_mode'] and info['cdf'] == cdf
    assert info['source_cdf_digest'] == hashlib.sha256(cdf).hexdigest()
    assert np.array_equal(p.decode_cpu(blob), x)


def test_exact_cdf_is_preserved_and_never_recounted():
    x = fixture()
    cdf = exact_cdf()
    blob = p.encode(x, row_classes=1, col_classes=1, row_labels=bytes(64),
                     col_labels=bytes(512), sparse_cdf=cdf, shared_metadata=True)
    _, info = p._parse(blob)
    assert info['cdf'] == cdf
    assert info['source_cdf_digest'] == hashlib.sha256(cdf).hexdigest()
    assert np.array_equal(p.decode_cpu(blob), x)
    assert info['tables'].frequencies[0][1:3] == (52429, 13107)


def test_singleton_empty_and_unused_contexts():
    # One populated exponent context and three deliberately empty contexts.
    cdf = struct.pack('<5I', 0, 1, 1, 1, 1) + struct.pack('<HI', 28, 65536)
    tables = t.build(cdf, 2, 2)
    assert tables.singletons == (7, -2, -2, -2)
    assert tables.context_dir[0, 2] == 7
    assert np.all(tables.context_dir[1:, 1] == 0)
    assert tables.context_dir[0, 1] == 0 and tables.words.size == 0
    x = np.full((64, 512), 7 << 7, dtype=np.uint16)
    blob = p.encode(x, row_classes=1, col_classes=1, row_labels=bytes(64),
                    col_labels=bytes(512), sparse_cdf=struct.pack('<2I', 0, 1) + struct.pack('<HI', 28, 65536),
                    shared_metadata=True)
    assert np.array_equal(p.decode_cpu(blob), x)


def test_long_huffman_codes_use_secondary_tables():
    weights = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    total = sum(weights)
    frequencies = [w * 65536 // total for w in weights]
    for j in range(65536 - sum(frequencies)):
        frequencies[j % len(frequencies)] += 1
    cumulative = 0
    entries = []
    for exp, frequency in enumerate(frequencies):
        cumulative += frequency
        entries.append((4 * exp, cumulative))
    cdf = struct.pack('<2I', 0, len(entries)) + b''.join(
        struct.pack('<HI', *q) for q in entries)
    tables = t.build(cdf, 1, 1)
    assert tables.max_lengths[0] > 8
    assert len(tables.words) > 320
    assert any(int(word) >> 30 == 1 for word in tables.words[:256])
    short_code = next(q for q in tables.codes[0] if q[1] == 1)
    long_code = next(q for q in tables.codes[0] if q[1] > 8)
    prefix = long_code[0] >> (long_code[1] - 7)
    entry = int(tables.words[prefix])
    assert prefix >= 64
    assert (entry >> 28) & 3 == 1
    assert (entry >> 24) & 15 == 1
    assert entry & 255 == short_code[2]
    rng = np.random.default_rng(777)
    exps = rng.choice(np.arange(len(weights), dtype=np.uint16), (64, 512),
                      p=np.asarray(frequencies) / 65536)
    x = (exps << 7) | rng.integers(0, 128, (64, 512), dtype=np.uint16)
    blob = p.encode(x, row_classes=1, col_classes=1, row_labels=bytes(64),
                    col_labels=bytes(512), sparse_cdf=cdf, shared_metadata=True)
    assert np.array_equal(p.decode_cpu(blob), x)
    _, info = p._parse(blob)
    bits = info['offsets'][-1]
    if bits & 7:
        broken = bytearray(blob)
        broken[info['main_start'] + info['main_bytes'] - 1] |= 1
        body_end = len(broken) - 32
        broken[body_end:] = hashlib.sha256(
            broken[:body_end] + bytes.fromhex(info['identity'])).digest()
        with pytest.raises(p.PacketError, match='terminal bits'):
            p._parse(broken)


def test_corruption_and_geometry_refusal():
    x = fixture()
    blob = p.encode(x, row_classes=1, col_classes=1)
    for bad in (blob[:-1], blob[:64], blob + b'\0'):
        with pytest.raises(p.PacketError):
            p._parse(bad)
    broken = bytearray(blob)
    broken[0] ^= 1
    with pytest.raises(p.PacketError):
        p._parse(broken)
    # Recompute the outer checksum after changing only the declared map digest.
    # Parse must rebuild the canonical table from exact CDF bytes and catch it.
    _, info = p._parse(blob)
    broken = bytearray(blob)
    map_offset = p.HEADER.size - 32
    forged_map = bytes([info['tables'].digest[0] ^ 1]) + info['tables'].digest[1:]
    broken[map_offset:map_offset + 32] = forged_map
    body_end = len(broken) - 32
    identity = p._identities(bytes.fromhex(info['source_digest']), info['labels'],
        info['row_classes'], info['col_classes'], info['cols'], info['split_k'],
        info['flags'], info['cdf'], forged_map)
    broken[body_end:] = hashlib.sha256(broken[:body_end] + identity).digest()
    with pytest.raises(p.PacketError, match='map digest'):
        p._parse(broken)
    with pytest.raises(p.PacketError):
        p.encode(x, split_k=2, row_classes=1, col_classes=1)


def test_matrix_cdf_and_column_binding():
    x = fixture()
    cdf = exact_cdf()
    one = p.encode(x, row_classes=1, col_classes=1, row_labels=bytes(64),
                   col_labels=bytes(512), sparse_cdf=cdf, shared_metadata=True)
    two = p.encode(x, row_classes=1, col_classes=1, row_labels=bytes(64),
                   col_labels=bytes(512), sparse_cdf=cdf, shared_metadata=True)
    arrays = p._resident_matrix_arrays([one, two])
    assert arrays['matrix_mode'] and arrays['panels'] == 2
    assert arrays['stream_offsets'].shape == (2, 65)
    assert arrays['stream_offsets'].dtype == np.uint32
    assert arrays['panel_desc'].shape == (2, 3) and arrays['panel_desc'].dtype == np.int64
    assert arrays['row_map'].dtype == np.int32
    assert arrays['matrix_source_digest'] == hashlib.sha256(
        bytes.fromhex(p._parse(one)[1]['source_digest']) * 2).hexdigest()
    cdf4 = struct.pack('<5I', 0, 1, 2, 3, 4) + b''.join(
        struct.pack('<HI', 4, 65536) for _ in range(4))
    labelled = p.encode(x, row_classes=1, col_classes=4, row_labels=bytes(64),
        col_labels=bytes(512), sparse_cdf=cdf4, shared_metadata=True)
    different_columns = p.encode(x, row_classes=1, col_classes=4, row_labels=bytes(64),
        col_labels=bytes(np.arange(512, dtype=np.uint8) % 4), sparse_cdf=cdf4, shared_metadata=True)
    with pytest.raises(p.PacketError, match='column labels'):
        p._resident_matrix_arrays([labelled, different_columns])


def test_packed_context_cache_keys_exact_codes_but_preserves_cdf_identity(monkeypatch):
    with t._PACKED_CONTEXT_CACHE_LOCK:
        t._PACKED_CONTEXT_CACHE.clear()
    first_cdf=cdf_for_frequencies([0,52429,13107]+[0]*253)
    second_cdf=cdf_for_frequencies([0,40000,25536]+[0]*253)
    first_freq=[0,52429,13107]+[0]*253
    second_freq=[0,40000,25536]+[0]*253
    original_verify=t._verify_context
    verified=[]
    def counted(words,codes):
        verified.append(tuple(codes))
        return original_verify(words,codes)
    monkeypatch.setattr(t,'_verify_context',counted)
    words1,codes1,singleton1,max1=t._build_context(first_freq)
    words1[0]^=1
    words2,codes2,singleton2,max2=t._build_context(second_freq)
    assert codes1==codes2 and singleton1==singleton2==-1 and max1==max2
    assert words1!=words2  # the first return was caller-mutated; cache data stayed pristine
    assert len(verified)==1
    first=t.build(first_cdf,1,1)
    second=t.build(second_cdf,1,1)
    assert np.array_equal(first.words,second.words)
    assert first.digest!=second.digest
    assert first.frequencies!=second.frequencies
    malformed=bytearray(first_cdf)
    malformed[16:20]=struct.pack('<I',65535)
    with pytest.raises(ValueError,match='incomplete sparse CDF normalization'):
        t.build(malformed,1,1)


def test_packed_context_cache_is_bounded_and_returns_independent_lists(monkeypatch):
    with t._PACKED_CONTEXT_CACHE_LOCK:
        t._PACKED_CONTEXT_CACHE.clear()
    monkeypatch.setattr(t,'_PACKED_CONTEXT_CACHE_LIMIT',2)
    contexts=[[0,32768,32768]+[0]*253,
              [0,21845,21845,21846]+[0]*252,
              [0,16384,16384,16384,16384]+[0]*251]
    outputs=[t._build_context(freq) for freq in contexts]
    assert len(t._PACKED_CONTEXT_CACHE)<=2
    assert len({row[1] for row in outputs})==3
    words,_,_,_=t._build_context(contexts[-1])
    words[0]^=1
    again,_,_,_=t._build_context(contexts[-1])
    assert words[0]!=again[0]


def test_resident_factories_decode_each_distinct_wire_once(monkeypatch):
    x = fixture()
    blob = p.encode(x, row_classes=1, col_classes=1)
    original = p._decode
    calls = []

    def counted(data, info):
        calls.append(hashlib.sha256(data).digest())
        return original(data, info)

    monkeypatch.setattr(p, '_decode', counted)
    p._resident_arrays(blob, panels=4)
    assert calls == [hashlib.sha256(blob).digest()]
    calls.clear()
    p._resident_matrix_arrays([blob, blob])
    assert calls == [hashlib.sha256(blob).digest()]


def test_grouping_bijection_and_chunk_local_column_order():
    row_labels = np.arange(64, dtype=np.uint8) & 3
    col_labels = np.arange(512, dtype=np.uint8) & 3
    offsets = [0]
    entries = []
    for _ in range(16):
        entries.append((4, 65536))
        offsets.append(len(entries))
    cdf = struct.pack('<17I', *offsets) + b''.join(struct.pack('<HI', *q) for q in entries)
    x = np.full((64, 512), 1 << 7, dtype=np.uint16)
    blob = p.encode(x, row_classes=4, col_classes=4, row_labels=row_labels,
                    col_labels=col_labels, sparse_cdf=cdf, shared_metadata=True)
    arrays = p._resident_arrays(blob)
    assert arrays['tiles'] == 4
    assert arrays['stream_offsets'].shape == (1, 4 * 64 + 1)
    assert arrays['panel_desc'].tolist() == [[0, 0, 0]]
    assert arrays['tile_rc'].tolist() == [0, 1, 2, 3]
    assert arrays['row_desc'][0, :16, 0].tolist() == list(range(0, 64, 4))
    assert arrays['row_desc'][1, :16, 0].tolist() == list(range(1, 64, 4))
    assert np.all(arrays['row_desc'][:, 16:, 0] == -1)
    assert np.all(arrays['row_desc'][:, 16:, 1] == 0)
    assert arrays['counts'][0].tolist() == [0, 32, 64, 96, 128]
    assert arrays['positions'][:8].tolist() == [0, 4, 8, 12, 16, 20, 24, 28]


def test_explicit_256_row_classes_and_empty_context_use_rejection():
    offsets = [0]
    entries = []
    for _ in range(256):
        if len(offsets) <= 64:
            entries.append((4, 65536))
        offsets.append(len(entries))
    cdf256 = struct.pack('<257I', *offsets) + b''.join(
        struct.pack('<HI', *q) for q in entries)
    x = np.full((64, 512), 1 << 7, dtype=np.uint16)
    row_labels = np.arange(64, dtype=np.uint8)
    blob = p.encode(x, row_classes=256, col_classes=1, row_labels=row_labels,
        col_labels=bytes(512), sparse_cdf=cdf256, shared_metadata=True)
    assert np.array_equal(p.decode_cpu(blob), x)
    _, info = p._parse(blob)
    assert info['row_classes'] == 256

    cdf_empty = struct.pack('<3I', 0, 1, 1) + struct.pack('<HI', 4, 65536)
    blob = p.encode(x, row_classes=1, col_classes=2, row_labels=bytes(64),
        col_labels=bytes(512), sparse_cdf=cdf_empty, shared_metadata=True)
    _, info = p._parse(blob)
    broken = bytearray(blob)
    labels_start = p.HEADER.size
    broken[labels_start + 64] = 1
    body_end = len(broken) - 32
    new_labels = bytes(broken[labels_start:labels_start + 64 + 512])
    identity = p._identities(bytes.fromhex(info['source_digest']), new_labels,
        info['row_classes'], info['col_classes'], info['cols'], info['split_k'],
        info['flags'], info['cdf'], bytes.fromhex(info['map_digest']))
    broken[body_end:] = hashlib.sha256(broken[:body_end] + identity).digest()
    with pytest.raises(p.PacketError, match='empty exponent context'):
        p.decode_cpu(broken)


def test_explicit_native_scope_does_not_change_process_default(monkeypatch):
    monkeypatch.setenv('GEOREFINE_BHE1_NATIVE_CPU','0')
    assert not p._native_cpu_enabled()
    with pytest.raises(RuntimeError):
        with p.native_cpu_backend():
            assert p._native_cpu_enabled()
            with p.native_cpu_backend(False):assert not p._native_cpu_enabled()
            raise RuntimeError('synthetic failure')
    assert not p._native_cpu_enabled()


def test_cdf_validation_cache_binds_exact_bytes_and_geometry():
    import struct
    cdf = bytearray(struct.pack('<IIHIHI', 0, 2, 512, 32768, 513, 65536))
    p._checked_cdf_bytes.cache_clear()
    expected = bytes(cdf)
    assert p._cdf(cdf, 1, 1) == expected
    assert p._cdf(memoryview(cdf), 1, 1) == expected
    assert p._checked_cdf_bytes.cache_info().hits == 1
    with pytest.raises(p.PacketError):
        p._cdf(cdf, 2, 1)
    # Mutating the previously validated caller buffer must force validation again.
    cdf[-4:] = struct.pack('<I', 65535)
    with pytest.raises(p.PacketError, match='normalization'):
        p._cdf(cdf, 1, 1)
    assert p._cdf(expected, 1, 1) == expected
    for upper in range(30000, 30020):
        valid = struct.pack('<IIHIHI', 0, 2, 512, upper, 513, 65536)
        assert p._cdf(valid, 1, 1) == valid
    assert p._checked_cdf_bytes.cache_info().currsize <= p._CDF_CACHE_MAX_ENTRIES


def test_cdf_validation_cache_is_bounded_by_entries_and_serialized_bytes():
    p._checked_cdf_bytes.cache_clear()
    cdfs = cdf_variants(p._CDF_CACHE_MAX_ENTRIES + 8)
    for cdf in cdfs:
        assert p._cdf(cdf, 16, 4) == cdf
    info = p._checked_cdf_bytes.cache_info()
    assert info.currsize == p._CDF_CACHE_MAX_ENTRIES
    assert p._checked_cdf_bytes._bytes <= p._CDF_CACHE_MAX_BYTES


def test_cdf_validation_cache_evicts_by_byte_budget_in_lru_order():
    cdfs = cdf_variants(3)
    cache = p._CDFValidationCache(32, 2 * len(cdfs[0]))
    for cdf in cdfs[:2]:
        assert cache(cdf, 16, 4) == cdf
    assert cache(cdfs[0], 16, 4) == cdfs[0]
    assert cache(cdfs[2], 16, 4) == cdfs[2]
    assert cache.cache_info().currsize == 2
    assert cache._bytes == 2 * len(cdfs[0])
    assert cache(cdfs[0], 16, 4) == cdfs[0]
    assert cache.cache_info().hits == 2
    assert cache(cdfs[1], 16, 4) == cdfs[1]
    assert cache.cache_info().misses == 4


def test_cdf_validation_cache_valid_oversized_input_is_not_retained(monkeypatch):
    cdf = cdf_variants(1)[0]
    cache = p._CDFValidationCache(32, len(cdf) - 1)
    calls = []
    original = t._sparse_cdf

    def counted(data, nr, nc):
        calls.append((len(data), nr, nc))
        return original(data, nr, nc)

    monkeypatch.setattr(t, '_sparse_cdf', counted)
    for _ in range(2):
        assert cache(cdf, 16, 4) == cdf
    assert len(calls) == 2
    assert cache.cache_info().misses == 2
    assert cache.cache_info().hits == 0
    assert cache.cache_info().currsize == 0
    assert cache._bytes == 0


def test_cdf_validation_cache_oversized_input_is_validated_uncached(monkeypatch):
    p._checked_cdf_bytes.cache_clear()
    calls = []
    original = t._sparse_cdf

    def counted(data, nr, nc):
        calls.append((len(data), nr, nc))
        return original(data, nr, nc)

    monkeypatch.setattr(t, '_sparse_cdf', counted)
    oversized = b'\0' * (p._CDF_CACHE_MAX_BYTES + 1)
    for _ in range(2):
        with pytest.raises(p.PacketError, match='invalid sparse CDF directory'):
            p._cdf(oversized, 1, 1)
    info = p._checked_cdf_bytes.cache_info()
    assert len(calls) == 2
    assert info.hits == 0 and info.misses == 2 and info.currsize == 0


def test_cdf_digest_does_not_authorize_malformed_cdf_or_geometry():
    p._checked_cdf_bytes.cache_clear()
    malformed = struct.pack('<IIHIHI', 0, 2, 512, 32768, 513, 65535)
    digest = p._cdf_digest(malformed, 1, 1)
    assert digest == hashlib.sha256(malformed).digest()
    with pytest.raises(p.PacketError, match='normalization'):
        p._cdf(malformed, 1, 1)
    entry = p._checked_cdf_bytes._entries[(malformed, 1, 1)]
    assert entry.digest == digest and not entry.validated

    valid = exact_cdf()
    p._cdf_digest(valid, 1, 1)
    assert p._cdf(valid, 1, 1) == valid
    p._cdf_digest(valid, 2, 1)
    with pytest.raises(p.PacketError):
        p._cdf(valid, 2, 1)
    assert p._checked_cdf_bytes._entries[(valid, 1, 1)].validated
    assert not p._checked_cdf_bytes._entries[(valid, 2, 1)].validated


def test_cdf_digest_shares_byte_budget_and_eviction_with_validation_cache():
    cdfs = cdf_variants(3)
    cache = p._CDFValidationCache(32, 2 * len(cdfs[0]))
    first = cache.digest(cdfs[0], 16, 4)
    assert first == hashlib.sha256(cdfs[0]).digest()
    assert cache(cdfs[0], 16, 4) == cdfs[0]
    cache.digest(cdfs[1], 16, 4)
    cache.digest(cdfs[0], 16, 4)
    cache.digest(cdfs[2], 16, 4)
    assert cache._bytes == 2 * len(cdfs[0])
    assert (cdfs[0], 16, 4) in cache._entries
    assert (cdfs[1], 16, 4) not in cache._entries
    assert cache._entries[(cdfs[0], 16, 4)].validated


def test_cdf_digest_is_thread_safe_and_does_not_change_validation_counts(monkeypatch):
    p._checked_cdf_bytes.cache_clear()
    cdfs = cdf_variants(24)
    calls = []
    original = p._hash

    def counted(data):
        calls.append(len(data))
        return original(data)

    monkeypatch.setattr(p, '_hash', counted)

    def caller(cdf):
        for _ in range(8):
            assert p._cdf_digest(cdf, 16, 4) == hashlib.sha256(cdf).digest()

    with ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(caller, cdfs))
    assert len(calls) == len(cdfs)
    assert p._checked_cdf_bytes.cache_info().hits == 0
    assert p._checked_cdf_bytes.cache_info().misses == 0


def test_cdf_local_fastpath_keeps_digest_validation_and_geometry_independent():
    cache = p._CDFValidationCache(32, 8 * 1024 * 1024)
    cdf = large_cdf_variants(1)[0]
    digest = cache.digest(cdf, 16, 4)
    assert digest == hashlib.sha256(cdf).digest()
    with pytest.raises(p.PacketError):
        cache(cdf, 1, 4)
    entry = cache._entries[(cdf, 16, 4)]
    assert entry.digest == digest and not entry.validated
    assert cache(cdf, 16, 4) == cdf
    assert cache(cdf, 16, 4) == cdf
    assert cache.cache_info().misses == 2
    assert cache.cache_info().hits == 1


def test_cdf_local_fastpath_weakrefs_do_not_retain_evicted_payloads():
    cdfs = large_cdf_variants(2)
    cache = p._CDFValidationCache(1, 8 * 1024 * 1024)
    cache.digest(cdfs[0], 16, 4)
    entry_ref = weakref.ref(cache._entries[(cdfs[0], 16, 4)])
    cache.digest(cdfs[1], 16, 4)
    gc.collect()
    assert entry_ref() is None
    cache.digest(cdfs[0], 16, 4)
    assert cache._digest_misses == 3


def test_cdf_local_fastpath_clear_invalidates_refs_and_counters():
    cache = p._CDFValidationCache(32, 8 * 1024 * 1024)
    cdf = large_cdf_variants(1)[0]
    cache.digest(cdf, 16, 4)
    cache(cdf, 16, 4)
    cache.cache_clear()
    assert cache.cache_info().hits == 0
    assert cache.cache_info().misses == 0
    assert cache._digest_stats() == (0, 0)
    cache.digest(cdf, 16, 4)
    assert cache._digest_stats() == (0, 1)


def test_cdf_local_fastpath_is_thread_safe_for_mixed_warm_callers():
    cdfs = large_cdf_variants(24)
    cache = p._CDFValidationCache(32, 8 * 1024 * 1024)

    def caller(cdf):
        for _ in range(8):
            assert cache.digest(cdf, 16, 4) == hashlib.sha256(cdf).digest()
            assert cache(cdf, 16, 4) == cdf
            assert cache.digest(cdf, 16, 4) == hashlib.sha256(cdf).digest()
            assert cache(cdf, 16, 4) == cdf

    with ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(caller, cdfs))
    info = cache.cache_info()
    assert info.misses == 24
    assert info.hits == 24 * 15
    assert cache._digest_stats() == (24 * 15, 24)


def test_cdf_local_fastpath_concurrent_clear_eviction_and_warm_hits():
    cache = p._CDFValidationCache(2, 8 * 1024 * 1024)
    cdfs = large_cdf_variants(8)
    start = threading.Barrier(9)
    errors = []

    def worker(worker_id):
        try:
            start.wait()
            for step in range(100):
                cdf = cdfs[(worker_id + step) % len(cdfs)]
                cache.digest(cdf, 16, 4)
                cache(cdf, 16, 4)
                cache(cdf, 16, 4)
        except BaseException as exc:  # report every worker failure together
            errors.append(exc)

    def clearer():
        try:
            start.wait()
            for _ in range(100):
                cache.cache_clear()
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = [pool.submit(worker, i) for i in range(8)]
        futures.append(pool.submit(clearer))
        for future in futures:
            future.result()
    assert not errors
    assert cache.cache_info().currsize <= 2
    assert cache._bytes <= cache.max_bytes

    cache.cache_clear()
    assert cache.cache_info().hits == 0
    assert cache.cache_info().misses == 0
    assert cache._digest_stats() == (0, 0)
    cdf = cdfs[0]
    cache.digest(cdf, 16, 4)
    cache(cdf, 16, 4)
    cache(cdf, 16, 4)
    assert cache.cache_info().hits == 1
    assert cache.cache_info().misses == 1
    assert cache._digest_stats() == (0, 1)


def test_cdf_local_fastpath_bounds_short_lived_thread_metadata_and_folds_hits():
    cache = p._CDFValidationCache(32, 8 * 1024 * 1024)
    cdf = large_cdf_variants(1)[0]
    count = 256

    def caller():
        cache.digest(cdf, 16, 4)
        cache(cdf, 16, 4)
        cache(cdf, 16, 4)

    for _ in range(count):
        thread = threading.Thread(target=caller)
        thread.start()
        thread.join()

    info = cache.cache_info()
    digest_hits, digest_misses = cache._digest_stats()
    assert len(cache._thread_states) <= p._CDF_MAX_LOCAL_STATES
    assert info.hits >= 1 + 2 * (count - 1)
    assert digest_hits >= count - 1
    assert digest_misses == 1


def test_cdf_local_fastpath_falls_back_after_live_state_bound():
    cache = p._CDFValidationCache(32, 8 * 1024 * 1024)
    cdf = large_cdf_variants(1)[0]
    workers = p._CDF_MAX_LOCAL_STATES + 8
    ready = threading.Barrier(workers + 1)
    release = threading.Event()
    errors = []

    def caller():
        try:
            cache.digest(cdf, 16, 4)
            cache(cdf, 16, 4)
            ready.wait()
            release.wait()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=caller) for _ in range(workers)]
    for thread in threads:
        thread.start()
    ready.wait()
    assert len(cache._thread_states) == p._CDF_MAX_LOCAL_STATES
    release.set()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(cache._thread_states) == p._CDF_MAX_LOCAL_STATES
    assert cache.cache_info().hits >= workers - 1


def test_forged_source_digest_still_fails_after_cdf_digest_cache_hit():
    p._checked_cdf_bytes.cache_clear()
    x = fixture()
    blob = p.encode(x, row_classes=1, col_classes=1)
    _, info = p._parse(blob)
    broken = bytearray(blob)
    source_offset = struct.calcsize('<4sBBBBIIIIIIII')
    broken[source_offset] ^= 1
    body_end = len(broken) - 32
    source_digest = bytes(broken[source_offset:source_offset + 32])
    identity = p._identities(source_digest, info['labels'], info['row_classes'],
        info['col_classes'], info['cols'], info['split_k'], info['flags'],
        info['cdf'], bytes.fromhex(info['map_digest']))
    broken[body_end:] = hashlib.sha256(broken[:body_end] + identity).digest()
    with pytest.raises(p.PacketError, match='decoded source digest mismatch'):
        p.decode_cpu(broken)


def test_empty_raw_metadata_digest_cache_preserves_raw_roundtrip():
    p._checked_cdf_bytes.cache_clear()
    x = fixture(kind='raw')
    blob = p.encode(x, row_classes=1, col_classes=1)
    _, info = p._parse(blob)
    assert info['raw_mode'] and info['cdf'] == b''
    assert p._cdf_digest(b'', 1, 1) == hashlib.sha256(b'').digest()
    assert np.array_equal(p.decode_cpu(blob), x)


def test_cdf_validation_cache_is_thread_safe_for_interleaved_callers():
    p._checked_cdf_bytes.cache_clear()
    cdfs = cdf_variants(24)

    def caller(cdf):
        for _ in range(8):
            assert p._cdf(cdf, 16, 4) == cdf

    with ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(caller, cdfs))
    info = p._checked_cdf_bytes.cache_info()
    assert info.misses == len(cdfs)
    assert info.hits == len(cdfs) * 7
    assert info.currsize == len(cdfs)
