"""CPU construction of resident exponent split checkpoints.

The walk mirrors context_exponent_packet.cuh's Cursor/group operations. It advances
one encoded group at a time per 128-column chunk, so preparation does not expand a
full decoded matrix or perform per-token work.
"""
from __future__ import annotations

import numpy as np


def _load_be32(data: np.ndarray, offset: int) -> int:
    return (int(data[offset]) << 24) | (int(data[offset + 1]) << 16) | (int(data[offset + 2]) << 8) | int(data[offset + 3])


class _Cursor:
    __slots__ = ("data", "bytes", "start", "end", "next_word", "buffer", "valid", "consumed", "error")

    def __init__(self, data, start, end):
        self.data = data; self.bytes = int(data.size); self.start = int(start); self.end = int(end)
        self.next_word = 0; self.buffer = 0; self.valid = 0; self.consumed = 0; self.error = 0
        if self.start < 0 or self.start > self.end or self.end > self.bytes * 8 or self.end - self.start > 32768 * 31:
            self.error = 1; return
        if self.start == self.end: return
        offset = (self.start >> 5) * 4
        if offset > self.bytes or self.bytes - offset < 4: self.error = 4; return
        self.valid = 32 - (self.start & 31)
        self.buffer = _load_be32(data, offset) & ((1 << self.valid) - 1)
        self.next_word = (self.start >> 5) + 1

    def peek(self, bits=8):
        if self.error: return None
        if self.valid < bits:
            offset = self.next_word * 4
            if offset > self.bytes or self.bytes - offset < 4:
                self.error = 4; return None
            self.buffer = ((self.buffer << 32) | _load_be32(self.data, offset)) & ((1 << 64) - 1)
            self.valid += 32; self.next_word += 1
        return (self.buffer >> (self.valid - bits)) & ((1 << bits) - 1)

    def consume(self, bits):
        if self.error or bits > self.valid or bits > self.end - self.start - self.consumed:
            self.error = self.error or 4; return False
        self.valid -= bits; self.consumed += bits; return True


def _group(table, singleton, maximum, cur, remaining):
    if cur.error or not remaining: cur.error = cur.error or 2; return 0
    if singleton >= 0:
        if singleton > 255: cur.error = 2; return 0
        return min(3, remaining)
    if singleton != -1 or table is None or table.size < 320 or not maximum or maximum > 31:
        cur.error = 2; return 0
    lookup_words = int(table.size) - 64
    prefix = cur.peek(8)
    if prefix is None: return 0
    entry = int(table[prefix])
    if entry >> 30 == 0:
        count = (entry >> 28) & 3; used = (entry >> 24) & 15
        if not count or not used or used > 8 or ((entry & 0xffffff) >> (count * 8)):
            cur.error = 2; return 0
        if count <= remaining:
            if not cur.consume(used): return 0
            return count
        symbol = entry & 255
        length = (int(table[lookup_words + symbol // 4]) >> (8 * (symbol & 3))) & 255
        if not length or length > maximum or length > used:
            cur.error = 2; return 0
        if not cur.consume(length): return 0
        return 1
    depth = 8
    for _ in range(8):
        if entry >> 30 != 1: cur.error = 2; return 0
        width = (entry >> 26) & 15; offset = entry & 0x03ffffff
        if not width or width > 4 or depth + width > 31 or offset < 256 or offset > lookup_words or (1 << width) > lookup_words - offset:
            cur.error = 2; return 0
        prefix = cur.peek(depth + width)
        if prefix is None: return 0
        entry = int(table[offset + (prefix & ((1 << width) - 1))]); depth += width
        if entry >> 30 == 0:
            length = (entry >> 8) & 31
            if entry >> 13 or not length or length > maximum or length > depth:
                cur.error = 2; return 0
            if not cur.consume(length): return 0
            return 1
    cur.error = 2; return 0


def add_split_checkpoints(arrays: dict, splits: int, *, native=True) -> dict:
    """Return arrays with int64 absolute stream checkpoints for split count."""
    if type(splits) is not int or not 1 <= splits <= 8 or splits > int(arrays["K"]) // 128:
        raise ValueError("splits must be 1..8 and no greater than K/128")
    if splits == 1:
        return arrays
    if native:
        from .context_exponent_native import split_checkpoints_native
        arrays = dict(arrays)
        arrays["split_checkpoints"] = split_checkpoints_native(arrays, splits).reshape(-1)
        return arrays
    n, k, nc = int(arrays["N"]), int(arrays["K"]), int(arrays["nc"])
    checkpoints = np.zeros((n, nc, splits - 1), dtype=np.int64)
    row_map = np.asarray(arrays["row_map"]); tile_rc = np.asarray(arrays["tile_rc"])
    streams = np.asarray(arrays["streams"]); context_dir = np.asarray(arrays["context_dir"])
    tables = np.asarray(arrays["tables"], dtype=np.uint32); counts = np.asarray(arrays["counts"], dtype=np.int64)
    main = np.asarray(arrays["main"], dtype=np.uint8); nch = k // 128
    loc = {}
    for tile in range(row_map.shape[0]):
        for slot, original in enumerate(row_map[tile]):
            if original >= 0: loc[int(original)] = (tile, slot, int(tile_rc[tile]))
    for original in range(n):
        if original not in loc: continue
        tile, slot, cls = loc[original]
        for cc in range(nc):
            start, end = (int(x) for x in streams[tile, slot, cc]);
            if splits > 1: checkpoints[original, cc, :] = start
            if start == end: continue
            d = context_dir[(cls * nc) + cc]; table = tables[int(d[0]):int(d[0] + d[1])]
            cur = _Cursor(main, start, end)
            targets = {((s * nch) // splits): s for s in range(1, splits)}
            for chunk in range(nch):
                if chunk in targets: checkpoints[original, cc, targets[chunk] - 1] = start + cur.consumed
                first, last = int(counts[chunk, cc]), int(counts[chunk, cc + 1]); produced = 0
                while produced < last - first and not cur.error:
                    got = _group(table, int(d[2]), int(d[3]), cur, last - first - produced)
                    if not got: break
                    produced += got
            if cur.error or cur.consumed != end - start:
                raise ValueError(f"cursor replay failed row={original} context={cc} error={cur.error} consumed={cur.consumed} length={end-start}")
    arrays = dict(arrays); arrays["split_checkpoints"] = checkpoints.reshape(-1)
    return arrays
