"""Validated, CPU-side coarse lookup tables for BCT2 sparse CDF data."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
import struct
from typing import Mapping

import numpy as np

TOTAL = 1 << 16
MAX_CONTEXTS = 1024
MAX_ENTRIES_PER_CONTEXT = 1024
_DIRECTORY = struct.Struct('<I')
_ENTRY = struct.Struct('<HI')


@dataclass(frozen=True)
class CDFLookup:
    """Immutable decoded arrays and the exact bytes occupied by their data."""

    coarse: np.ndarray
    records: np.ndarray
    offsets: np.ndarray
    source_sha256: str
    resident_byte_ledger: Mapping[str, int]
    resident_bytes: int

    def lookup(self, context: int, slot: int) -> tuple[int, int, int]:
        """Return ``(symbol, lower, frequency)`` for one 16-bit CDF slot.

        Empty contexts have no valid slot and raise ``ValueError``. Context and
        slot indices are checked before indexing the derived table.
        """
        if type(context) is not int or not 0 <= context < len(self.offsets) - 1:
            raise ValueError('context index out of range')
        if type(slot) is not int or not 0 <= slot < TOTAL:
            raise ValueError('CDF slot out of range')
        start, end = int(self.offsets[context]), int(self.offsets[context + 1])
        if start == end:
            raise ValueError('empty context has no lookupable CDF slots')

        packed = int(self.coarse[context, slot >> 8])
        lo, hi = packed & 0xffff, packed >> 16
        # Find the first record whose exclusive upper bound is above slot.
        while lo < hi:
            mid = lo + (hi - lo) // 2
            record = int(self.records[start + mid])
            lower = record & 0xffff
            frequency = (record >> 16) & 0x1ffff
            if lower + frequency <= slot:
                lo = mid + 1
            else:
                hi = mid
        if lo >= end - start:
            raise ValueError('slot is outside the context CDF')
        record = int(self.records[start + lo])
        lower = record & 0xffff
        frequency = (record >> 16) & 0x1ffff
        if not lower <= slot < lower + frequency:
            raise ValueError('slot is outside the context CDF')
        return (record >> 33) & 0x3ff, lower, frequency


def _readonly_array(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Copy through immutable bytes so ``setflags(write=True)`` also fails."""
    contiguous = np.ascontiguousarray(values, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(contiguous.shape)
    if not result.flags.aligned:
        raise ValueError('derived lookup arrays must have natural dtype alignment')
    return result


def _expected_digest(value: str | bytes | bytearray | memoryview) -> bytes:
    if isinstance(value, str):
        try:
            decoded = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError('expected_sha256 must be 32-byte SHA-256 hex') from exc
    elif isinstance(value, (bytes, bytearray, memoryview)):
        decoded = bytes(value)
    else:
        raise TypeError('expected_sha256 must be bytes or hexadecimal text')
    if len(decoded) != 32:
        raise ValueError('expected_sha256 must contain exactly 32 bytes')
    return decoded


def build_cdf_lookup(
    cdf_bytes: bytes | bytearray | memoryview,
    n_contexts: int,
    *,
    expected_sha256: str | bytes | bytearray | memoryview | None = None,
) -> CDFLookup:
    """Validate and derive lookup arrays from a BCT2 directory and raw6 table.

    The input is exactly ``uint32[n_contexts+1]`` little-endian entry offsets,
    followed by ``uint16 symbol, uint32 upper_cdf`` entries. No enclosing packet
    header is included. ``coarse[c, slot >> 8]`` packs half-open local record
    bounds as ``lo | (hi << 16)``. The resident ledger covers the derived
    arrays only; a GPU consumer must also count its unchanged packet buffer.
    Raw-only packets with no global CDF bypass this helper.
    """
    if type(n_contexts) is not int or not 1 <= n_contexts <= MAX_CONTEXTS:
        raise ValueError('n_contexts must be an integer in [1, 1024]')
    if not isinstance(cdf_bytes, (bytes, bytearray, memoryview)):
        raise TypeError('cdf_bytes must be bytes-like')
    directory_bytes = _DIRECTORY.size * (n_contexts + 1)
    input_bytes = cdf_bytes.nbytes if isinstance(cdf_bytes, memoryview) else len(cdf_bytes)
    if input_bytes < directory_bytes:
        raise ValueError('truncated CDF directory')
    if input_bytes > directory_bytes + _ENTRY.size * n_contexts * MAX_ENTRIES_PER_CONTEXT:
        raise ValueError('CDF input exceeds bounded capacity')
    try:
        raw = bytes(cdf_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError('cdf_bytes must be a contiguous byte buffer') from exc

    source_digest = hashlib.sha256(raw).digest()
    if expected_sha256 is not None and source_digest != _expected_digest(expected_sha256):
        raise ValueError('CDF SHA-256 does not match expected_sha256')

    offsets = np.frombuffer(raw, dtype='<u4', count=n_contexts + 1).astype(np.uint32)
    if int(offsets[0]) != 0:
        raise ValueError('CDF directory must start at entry zero')
    entry_count = int(offsets[-1])
    if entry_count > n_contexts * MAX_ENTRIES_PER_CONTEXT:
        raise ValueError('CDF entry count exceeds bounded capacity')
    expected_size = directory_bytes + _ENTRY.size * entry_count
    if len(raw) != expected_size:
        raise ValueError('CDF offsets, entry data, or trailing bytes disagree')
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError('CDF directory offsets must be monotonic')

    coarse = np.zeros((n_contexts, 256), dtype=np.uint32)
    records = np.empty(entry_count, dtype=np.uint64)
    bucket_starts = np.arange(256, dtype=np.uint32) * np.uint32(256)
    for context in range(n_contexts):
        begin, end = int(offsets[context]), int(offsets[context + 1])
        count = end - begin
        if count > MAX_ENTRIES_PER_CONTEXT:
            raise ValueError('context has more than 1024 CDF entries')
        if count == 0:
            continue

        uppers = []
        previous_symbol = -1
        previous_upper = 0
        for local_index, entry_index in enumerate(range(begin, end)):
            symbol, upper = _ENTRY.unpack_from(raw, directory_bytes + _ENTRY.size * entry_index)
            if not previous_symbol < symbol < 1024:
                raise ValueError('CDF symbols must be unique, increasing, and below 1024')
            if not previous_upper < upper <= TOTAL:
                raise ValueError('CDF upper bounds must increase within [1, 65536]')
            frequency = upper - previous_upper
            records[entry_index] = previous_upper | (frequency << 16) | (symbol << 33)
            previous_symbol, previous_upper = symbol, upper
            uppers.append(upper)
        if previous_upper != TOTAL:
            raise ValueError('non-empty context CDF must end at 65536')

        # Keep exactly the records intersecting each 256-slot interval.
        # Vectorized preprocessing avoids Python work for each bucket.
        upper_array = np.asarray(uppers, dtype=np.uint32)
        lows = np.searchsorted(upper_array, bucket_starts, side='right').astype(np.uint32)
        highs = (np.searchsorted(upper_array, bucket_starts + np.uint32(255),
                                 side='right').astype(np.uint32) + np.uint32(1))
        coarse[context] = lows | (highs << np.uint32(16))

    frozen_coarse = _readonly_array(coarse, np.dtype('<u4'))
    frozen_records = _readonly_array(records, np.dtype('<u8'))
    frozen_offsets = _readonly_array(offsets, np.dtype('<u4'))
    ledger = MappingProxyType({
        'coarse': int(frozen_coarse.nbytes),
        'records': int(frozen_records.nbytes),
        'offsets': int(frozen_offsets.nbytes),
    })
    resident_bytes = sum(ledger.values())
    return CDFLookup(
        coarse=frozen_coarse,
        records=frozen_records,
        offsets=frozen_offsets,
        source_sha256=source_digest.hex(),
        resident_byte_ledger=ledger,
        resident_bytes=resident_bytes,
    )
