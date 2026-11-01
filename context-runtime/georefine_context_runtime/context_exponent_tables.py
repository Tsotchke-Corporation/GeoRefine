"""Deterministic Huffman exponent lookup tables for the BHE1 experiment."""
from __future__ import annotations

import hashlib
import heapq
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

BUILDER_VERSION = 1
PRIMARY_WORDS = 256
FALLBACK_TAG = 1 << 30
_PACKED_CONTEXT_CACHE_LIMIT = 128
_PACKED_CONTEXT_CACHE = OrderedDict()
_PACKED_CONTEXT_CACHE_LOCK = threading.RLock()


def _fail(message):
    raise ValueError(message)


def _sparse_cdf(data, nr, nc):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        _fail('sparse CDF must be bytes')
    data = bytes(data)
    count = nr * nc
    base = 4 * (count + 1)
    if len(data) < base:
        _fail('short sparse CDF')
    offsets = struct.unpack_from('<' + 'I' * (count + 1), data)
    if offsets[0] or offsets[-1] > count * 1024 or base + 6 * offsets[-1] != len(data):
        _fail('invalid sparse CDF directory')
    result = []
    for lo, hi in zip(offsets, offsets[1:]):
        if lo > hi or hi - lo > 1024:
            _fail('invalid sparse CDF span')
        entries = [struct.unpack_from('<HI', data, base + 6 * j) for j in range(lo, hi)]
        last_s = last_c = 0
        prev_s = -1
        for symbol, upper in entries:
            if not prev_s < symbol < 1024 or not last_c < upper <= 65536:
                _fail('invalid sparse CDF symbol or normalization')
            prev_s, last_c = symbol, upper
        if entries and last_c != 65536:
            _fail('incomplete sparse CDF normalization')
        result.append(entries)
    return result


def _codes(freq):
    heap = [(int(f), i) for i, f in enumerate(freq) if f]
    heapq.heapify(heap)
    if len(heap) <= 1:
        return (), (next((i for i, f in enumerate(freq) if f), -1))
    children = {}
    node = 256
    while len(heap) > 1:
        fa, a = heapq.heappop(heap)
        fb, b = heapq.heappop(heap)
        children[node] = (a, b)
        heapq.heappush(heap, (fa + fb, node))
        node += 1
    lengths = [0] * 256
    todo = [(heap[0][1], 0)]
    while todo:
        n, depth = todo.pop()
        if n < 256:
            lengths[n] = depth
        else:
            a, b = children[n]
            todo.extend(((a, depth + 1), (b, depth + 1)))
    if max(lengths) > 31:
        _fail('exponent Huffman code exceeds 31 bits')
    ordered = sorted((n, s) for s, n in enumerate(lengths) if n)
    top = ordered[-1][0]
    if sum(1 << (top - n) for n, _ in ordered) != 1 << top:
        _fail('incomplete exponent Huffman tree')
    code = previous = 0
    result = []
    for n, symbol in ordered:
        code <<= n - previous
        result.append((code, n, symbol))
        code += 1
        previous = n
    return tuple(result), -1


def _trie(codes):
    root = {}
    for code, length, symbol in codes:
        node = root
        for bit in range(length - 1, -1, -1):
            if 's' in node:
                _fail('prefix collision')
            node = node.setdefault((code >> bit) & 1, {})
        if node:
            _fail('duplicate or prefix collision')
        node['s'] = symbol
    return root


def _symbol(node, value, width):
    cur = node
    for used in range(width):
        cur = cur.get((value >> (width - used - 1)) & 1)
        if cur is None:
            _fail('invalid exponent prefix')
        if 's' in cur:
            return cur['s'], used + 1
    _fail('truncated exponent prefix')


def _pack_leaf(symbol, length):
    return int(symbol) | (int(length) << 8)


def _build_context(freq):
    codes, singleton = _codes(freq)
    if singleton >= 0:
        return [], (), singleton, 0
    key = (codes, singleton)
    with _PACKED_CONTEXT_CACHE_LOCK:
        cached = _PACKED_CONTEXT_CACHE.get(key)
        if cached is not None:
            _PACKED_CONTEXT_CACHE.move_to_end(key)
            packed_words, maximum = cached
            return list(packed_words), codes, singleton, maximum
    words, codes, singleton, maximum = _build_context_uncached(codes, singleton)
    # Store immutable words only after the independent canonical verifier passes.
    packed_words = tuple(words)
    with _PACKED_CONTEXT_CACHE_LOCK:
        existing = _PACKED_CONTEXT_CACHE.get(key)
        if existing is not None:
            _PACKED_CONTEXT_CACHE.move_to_end(key)
            packed_words, maximum = existing
        else:
            _PACKED_CONTEXT_CACHE[key] = (packed_words, maximum)
            while len(_PACKED_CONTEXT_CACHE) > _PACKED_CONTEXT_CACHE_LIMIT:
                _PACKED_CONTEXT_CACHE.popitem(last=False)
    return list(packed_words), codes, singleton, maximum


def _build_context_uncached(codes, singleton):
    if singleton >= 0:
        return [], (), singleton, 0
    root = _trie(codes)
    primary = [0] * 256
    secondary_roots = {}
    # The primary lookup emits up to three complete codewords in eight bits.
    for prefix in range(256):
        node = root
        used = 0
        syms = []
        while len(syms) < 3 and used < 8:
            node = root
            attempt = used
            while attempt < 8 and 's' not in node:
                node = node.get((prefix >> (7 - attempt)) & 1)
                attempt += 1
                if node is None:
                    _fail('primary lookup hole')
            if 's' not in node:
                break
            used = attempt
            syms.append(node['s'])
        if syms:
            packed = used << 24 | len(syms) << 28
            for j, symbol in enumerate(syms):
                packed |= symbol << (8 * j)
            primary[prefix] = packed
        else:
            # This prefix must lead to a codeword longer than eight bits.
            node = root
            for bit in range(7, -1, -1):
                node = node.get((prefix >> bit) & 1)
                if node is None or 's' in node:
                    _fail('primary lookup hole')
            secondary_roots[prefix] = node

    words = list(primary)

    def emit(node, depth):
        # A secondary level consumes one to four more bits.
        remaining = _max_depth(node)
        width = min(4, remaining)
        if not 1 <= width <= 4:
            _fail('invalid secondary depth')
        offset = len(words)
        words.extend([0] * (1 << width))
        for index in range(1 << width):
            cur = node
            used = 0
            while used < width and 's' not in cur:
                cur = cur.get((index >> (width - used - 1)) & 1)
                if cur is None:
                    _fail('secondary lookup hole')
                used += 1
            if 's' in cur:
                words[offset + index] = _pack_leaf(cur['s'], depth + used)
            else:
                child_width = min(4, _max_depth(cur))
                child_offset = emit(cur, depth + width)
                if child_offset >= 1 << 26:
                    _fail('secondary table offset overflow')
                words[offset + index] = FALLBACK_TAG | (child_width << 26) | child_offset
        return offset

    # Allocate roots in prefix order, recursively; references are table-relative.
    root_refs = {}
    for prefix, node in sorted(secondary_roots.items()):
        root_offset = emit(node, 8)
        if root_offset >= 1 << 26:
            _fail('secondary table offset overflow')
        root_refs[prefix] = root_offset
    for prefix, offset in root_refs.items():
        width = min(4, _max_depth(secondary_roots[prefix]))
        words[prefix] = FALLBACK_TAG | (width << 26) | offset
    maximum = max(n for _, n, _ in codes)
    lengths = [0] * 256
    for _, length, symbol in codes:
        lengths[symbol] = length
    words.extend(sum(lengths[4 * j + q] << (8 * q) for q in range(4)) for j in range(64))
    _verify_context(words, codes)
    return words, codes, -1, maximum


def _max_depth(node):
    todo = [(node, 0)]
    maximum = 0
    while todo:
        cur, depth = todo.pop()
        if 's' in cur:
            maximum = max(maximum, depth)
        else:
            for bit in (0, 1):
                child = cur.get(bit)
                if child is not None:
                    todo.append((child, depth + 1))
    return maximum


def _verify_context(words, codes):
    """Independent canonical-interval proof of every primary/secondary slot."""
    if len(words) < 320 or len(codes) < 2:
        _fail('invalid normal exponent table span')
    tail = len(words) - 64
    lengths = [0] * 256
    for _, length, symbol in codes:
        lengths[symbol] = length
    packed_lengths = [sum(lengths[4 * j + q] << (8 * q) for q in range(4))
                      for j in range(64)]
    if words[tail:] != packed_lengths:
        _fail('exponent length tail mismatch')

    def branch(word):
        if word >> 30 != 1:
            _fail('expected secondary branch')
        width = (word >> 26) & 15
        offset = word & ((1 << 26) - 1)
        if not 1 <= width <= 4:
            _fail('invalid secondary width')
        return width, offset

    spans = []

    def verify_level(offset, width, depth, prefix, candidates):
        expected_width = min(4, max(n - depth for _, n, _ in candidates))
        if width != expected_width:
            _fail('noncanonical secondary width')
        stop = offset + (1 << width)
        if offset < 256 or stop > tail:
            _fail('secondary span outside context table')
        spans.append((offset, stop))
        for index in range(1 << width):
            path = (prefix << width) | index
            path_depth = depth + width
            leaves = [(code, n, symbol) for code, n, symbol in candidates
                      if n <= path_depth and (path >> (path_depth - n)) == code]
            word = int(words[offset + index])
            if leaves:
                if len(leaves) != 1:
                    _fail('ambiguous secondary leaf')
                code, n, symbol = leaves[0]
                if word != _pack_leaf(symbol, n):
                    _fail('secondary leaf disagrees with canonical code')
                continue
            children = [(code, n, symbol) for code, n, symbol in candidates
                        if n > path_depth and (code >> (n - path_depth)) == path]
            if not children:
                _fail('secondary table hole')
            child_width, child_offset = branch(word)
            wanted = min(4, max(n - path_depth for _, n, _ in children))
            if child_width != wanted:
                _fail('noncanonical nested branch width')
            verify_level(child_offset, child_width, path_depth, path, children)

    # Independent codeword interval matching verifies every primary entry.
    for prefix in range(256):
        symbols = []
        consumed = 0
        while len(symbols) < 3 and consumed < 8:
            available = 8 - consumed
            fragment = (prefix >> (8 - consumed - available)) & ((1 << available) - 1)
            matches = [(code, n, symbol) for code, n, symbol in codes
                       if n <= available and (fragment >> (available - n)) == code]
            if not matches:
                break
            if len(matches) != 1:
                _fail('ambiguous canonical primary prefix')
            code, n, symbol = matches[0]
            symbols.append(symbol)
            consumed += n
        word = int(words[prefix])
        if symbols:
            expected = consumed << 24 | len(symbols) << 28
            for slot, symbol in enumerate(symbols):
                expected |= symbol << (8 * slot)
            if word != expected:
                _fail('primary entry disagrees with canonical intervals')
        else:
            candidates = [(code, n, symbol) for code, n, symbol in codes
                          if n > 8 and (code >> (n - 8)) == prefix]
            if not candidates:
                _fail('primary table hole')
            width, offset = branch(word)
            wanted = min(4, max(n - 8 for _, n, _ in candidates))
            if width != wanted:
                _fail('noncanonical primary fallback width')
            verify_level(offset, width, 8, prefix, candidates)

    # Secondary tables must be a disjoint, gap-free allocation before the length tail.
    ordered = sorted(spans)
    cursor = 256
    for lo, hi in ordered:
        if lo != cursor:
            _fail('secondary table hole or overlap')
        cursor = hi
    if cursor != tail:
        _fail('unreferenced words before exponent length tail')

    # Decode each canonical codeword with zero and one suffix through packed tables.
    maximum = max(n for _, n, _ in codes)
    trial_width = max(8, maximum)
    for code, length, symbol in codes:
        for suffix in (0, (1 << (trial_width - length)) - 1):
            bits = (code << (trial_width - length)) | suffix
            prefix = bits >> (trial_width - 8)
            entry = int(words[prefix])
            if entry >> 30 == 0:
                count = (entry >> 28) & 3
                emitted = [(entry >> (8 * j)) & 255 for j in range(count)]
                if not emitted or emitted[0] != symbol or lengths[symbol] != length:
                    _fail('packed primary codeword verification failed')
            else:
                width, offset = branch(entry)
                depth = 8
                for _ in range(32):
                    if depth + width > trial_width:
                        _fail('packed secondary codeword lookup truncated')
                    index = (bits >> (trial_width - depth - width)) & ((1 << width) - 1)
                    leaf = int(words[offset + index])
                    if leaf >> 30 == 0:
                        got_symbol = leaf & 255
                        got_length = (leaf >> 8) & 31
                        if (got_symbol, got_length) != (symbol, length):
                            _fail('packed secondary codeword verification failed')
                        break
                    depth += width
                    width, offset = branch(leaf)
                else:
                    _fail('packed secondary traversal did not terminate')


@dataclass(frozen=True)
class Tables:
    words: np.ndarray
    context_dir: np.ndarray
    codes: tuple
    singletons: tuple
    max_lengths: tuple
    digest: bytes
    frequencies: tuple


def build(sparse_cdf, nr, nc):
    """Build immutable per-context tables from exact sparse main CDF bytes."""
    source = _sparse_cdf(sparse_cdf, nr, nc)
    all_words = []
    directory = []
    all_codes = []
    singletons = []
    max_lengths = []
    frequencies = []
    for entries in source:
        freq = [0] * 256
        prior = 0
        for symbol, upper in entries:
            main_frequency = upper - prior
            freq[symbol >> 2] += main_frequency
            prior = upper
        frequencies.append(tuple(freq))
        if not entries:
            directory.append((len(all_words), 0, -2, 0))
            all_codes.append(())
            singletons.append(-2)
            max_lengths.append(0)
            continue
        words, codes, singleton, maximum = _build_context(freq)
        offset = len(all_words)
        all_words.extend(words)
        directory.append((offset, len(words), singleton, maximum))
        all_codes.append(codes)
        singletons.append(singleton)
        max_lengths.append(maximum)
    words = np.asarray(all_words, dtype=np.uint32)
    directory = np.asarray(directory, dtype=np.int64).reshape((-1, 4))
    # Bind exact CDF, serialized tables, directory and builder version.
    material = (b'BHE1-exponent-table-builder-v1' + bytes(sparse_cdf) +
                words.astype('<u4', copy=False).tobytes() +
                directory.astype('<i8', copy=False).tobytes())
    digest = hashlib.sha256(material).digest()
    words.flags.writeable = False
    directory.flags.writeable = False
    return Tables(words, directory, tuple(all_codes), tuple(singletons),
                  tuple(max_lengths), digest, tuple(frequencies))


def verify(tables, sparse_cdf, nr, nc):
    """Rebuild and compare all primary/secondary words and exact CDF binding."""
    expected = build(sparse_cdf, nr, nc)
    if (not np.array_equal(tables.words, expected.words) or
            not np.array_equal(tables.context_dir, expected.context_dir) or
            tables.digest != expected.digest):
        _fail('noncanonical exponent tables or digest')
    return expected
