"""Experimental integer alias tables and CPU alias-rANS prototype.

The decoder permutes the 16-bit slot space into equal power-of-two buckets.
This is a new mapping: source CDF intervals are used only to validate and build
frequencies, and never to decode a permuted slot.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .context_cdf_lookup import TOTAL, build_cdf_lookup

SCALE = 16
SLOT_TOTAL = 1 << SCALE
RANS_L = 1 << 23
MAX_SYMBOLS = 1 << 16
MAX_CONTEXTS = 1024
MAX_SYMBOL_ID = 1023
LANES = 32
MAGIC = b"AAN1"
VERSION = 1
_HEADER = struct.Struct("<4sBBHII32s32s")
_LANE_DIRECTORY = struct.Struct("<33I")
WORD_MAGIC = b"AAN2"
WORD_VERSION = 2
WORD_RANS_L = 1 << 16
WORD_RADIX = 1 << 16
WORD_HEADER = struct.Struct("<4sBBHII32s32s")
WORD_STATE_BYTES = LANES * 4
_FLAG_CUT_FULL = 1
_FLAG_PRIMARY_FREQ_FULL = 2
_FLAG_ALIAS_FREQ_FULL = 4
_KNOWN_FLAGS = _FLAG_CUT_FULL | _FLAG_PRIMARY_FREQ_FULL | _FLAG_ALIAS_FREQ_FULL


def _freeze_aligned(values: np.ndarray, dtype: np.dtype, alignment: int = 16) -> np.ndarray:
    """Return immutable contiguous storage, optionally aligned for decoder loads."""
    source = np.ascontiguousarray(values, dtype=dtype)
    raw = source.tobytes()
    storage = bytearray(len(raw) + alignment - 1)
    address = np.frombuffer(storage, dtype=np.uint8).__array_interface__["data"][0]
    start = (-address) % alignment
    storage[start:start + len(raw)] = raw
    view = memoryview(storage)[start:start + len(raw)].toreadonly()
    return np.frombuffer(view, dtype=source.dtype).reshape(source.shape)


@dataclass(frozen=True)
class AliasTables:
    """Immutable per-context decoder records and separately billed CPU inverse data."""

    bucket_records: np.ndarray
    bucket_offsets: np.ndarray
    context_shifts: np.ndarray
    active_symbols: np.ndarray
    active_frequencies: np.ndarray
    active_offsets: np.ndarray
    inverse_symbol_offsets: np.ndarray
    inverse_segments: np.ndarray
    source_cdf_digest: bytes
    decoder_byte_ledger: Mapping[str, int]
    inverse_byte_ledger: Mapping[str, int]
    decoder_bytes: int
    inverse_bytes: int

    @property
    def context_count(self) -> int:
        return len(self.bucket_offsets) - 1

    @property
    def resident_bytes(self) -> int:
        return self.decoder_bytes + self.inverse_bytes

    def _context(self, context: int) -> tuple[int, int, int]:
        if type(context) is not int or not 0 <= context < self.context_count:
            raise ValueError("context index out of range")
        first, end = int(self.bucket_offsets[context]), int(self.bucket_offsets[context + 1])
        if first == end:
            raise ValueError("empty context has no alias slots")
        if first > end or end > len(self.bucket_records):
            raise ValueError("alias bucket offsets are invalid")
        shift = int(self.context_shifts[context])
        if not 0 <= shift <= 10:
            raise ValueError("alias context shift is invalid")
        if end - first != 1 << shift:
            raise ValueError("alias bucket count disagrees with context shift")
        return first, end, shift

    def decode_slot(self, context: int, slot: int) -> tuple[int, int, int]:
        """Map one 16-bit slot to ``(original_symbol, frequency, symbol_rank)``."""
        if type(slot) is not int or not 0 <= slot < SLOT_TOTAL:
            raise ValueError("slot index out of range")
        first, _, shift = self._context(context)
        record = self.bucket_records[first + (slot >> (SCALE - shift))]
        primary, alias, cut16, primary_base, alias_base, primary16, alias16, flags = map(int, record)
        if flags & ~_KNOWN_FLAGS:
            raise ValueError("alias record has reserved flags")
        bucket_width = 1 << (SCALE - shift)
        cut = SLOT_TOTAL if flags & _FLAG_CUT_FULL else cut16
        primary_frequency = SLOT_TOTAL if flags & _FLAG_PRIMARY_FREQ_FULL else primary16
        alias_frequency = SLOT_TOTAL if flags & _FLAG_ALIAS_FREQ_FULL else alias16
        if not 0 <= cut <= bucket_width:
            raise ValueError("alias cut is outside its bucket")
        within = slot & (bucket_width - 1)
        if within < cut:
            symbol, frequency, rank = primary, primary_frequency, primary_base + within
        else:
            symbol, frequency, rank = alias, alias_frequency, alias_base + within - cut
        if symbol > MAX_SYMBOL_ID or not 1 <= frequency <= SLOT_TOTAL or rank >= frequency:
            raise ValueError("alias record maps outside symbol frequency")
        return symbol, frequency, rank

    def _active_index(self, context: int, symbol: int) -> int:
        if type(context) is not int or not 0 <= context < self.context_count:
            raise ValueError("context index out of range")
        if type(symbol) is not int or not 0 <= symbol <= MAX_SYMBOL_ID:
            raise ValueError("symbol index out of range")
        start, end = int(self.active_offsets[context]), int(self.active_offsets[context + 1])
        if start > end or end > len(self.active_symbols):
            raise ValueError("active-symbol offsets are invalid")
        if start == end:
            raise ValueError("empty context has no alias symbols")
        local = int(np.searchsorted(self.active_symbols[start:end], symbol, side="left"))
        index = start + local
        if index >= end or int(self.active_symbols[index]) != symbol:
            raise ValueError("symbol is absent from context")
        return index

    def frequency(self, context: int, symbol: int) -> int:
        """Return the validated source frequency for an active original symbol."""
        return int(self.active_frequencies[self._active_index(context, symbol)])

    def encode_slot(self, context: int, symbol: int, rank: int) -> int:
        """Invert ``decode_slot`` with CPU-side per-symbol segment search."""
        index = self._active_index(context, symbol)
        frequency = int(self.active_frequencies[index])
        if type(rank) is not int or not 0 <= rank < frequency:
            raise ValueError("symbol rank is outside its frequency")
        begin = int(self.inverse_symbol_offsets[index])
        end = int(self.inverse_symbol_offsets[index + 1])
        if begin >= end or end > len(self.inverse_segments):
            raise ValueError("symbol has no inverse slot segments")
        rows = self.inverse_segments[begin:end]
        local = int(np.searchsorted(rows[:, 0], rank, side="right")) - 1
        if local < 0:
            raise ValueError("symbol rank has no inverse slot")
        rank_base, length, slot_base = map(int, rows[local])
        if not rank_base <= rank < rank_base + length:
            raise ValueError("symbol rank has no inverse slot")
        slot = slot_base + rank - rank_base
        decoded_symbol, decoded_frequency, decoded_rank = self.decode_slot(context, slot)
        if (decoded_symbol, decoded_frequency, decoded_rank) != (symbol, frequency, rank):
            raise ValueError("inverse alias map disagrees with decoder table")
        return slot


def _frequencies(records: np.ndarray, begin: int, end: int) -> tuple[list[int], list[int]]:
    symbols, frequencies = [], []
    for packed in records[begin:end]:
        record = int(packed)
        symbols.append((record >> 33) & 0x3ff)
        frequencies.append((record >> 16) & 0x1ffff)
    return symbols, frequencies


def _vose(symbols: list[int], frequencies: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Build exact integer bucket cuts and aliases for one non-empty context."""
    active = len(symbols)
    if active == 0:
        return [], [], []
    buckets = 1 << (active - 1).bit_length()
    width = SLOT_TOTAL // buckets
    primary = [symbols[i] if i < active else symbols[0] for i in range(buckets)]
    aliases = primary.copy()
    scaled = frequencies.copy() + [0] * (buckets - active)
    cuts = [0] * buckets
    small = [i for i, value in enumerate(scaled) if value < width]
    large = [i for i, value in enumerate(scaled) if value >= width]
    assigned = [False] * buckets
    while small and large:
        low = small.pop()
        high = large.pop()
        cuts[low] = scaled[low]
        aliases[low] = primary[high]
        scaled[high] -= width - scaled[low]
        assigned[low] = True
        if scaled[high] < width:
            small.append(high)
        else:
            large.append(high)
    for index in small + large:
        if scaled[index] != width:
            raise ValueError("integer alias construction did not balance buckets")
        cuts[index] = width
        aliases[index] = primary[index]
        assigned[index] = True
    if not all(assigned):
        raise ValueError("integer alias construction left a bucket unset")
    return primary, aliases, cuts


def build_alias_tables(
    cdf_bytes: bytes | bytearray | memoryview,
    n_contexts: int,
    *,
    expected_sha256: str | bytes | bytearray | memoryview | None = None,
) -> AliasTables:
    """Validate exact sparse CDF bytes and derive compact alias-rANS tables."""
    cdf = build_cdf_lookup(cdf_bytes, n_contexts, expected_sha256=expected_sha256)
    bucket_records = []
    bucket_offsets = [0]
    shifts = []
    active_symbols = []
    active_frequencies = []
    active_offsets = [0]
    inverse_symbol_offsets = [0]
    inverse_segments = []

    for context in range(n_contexts):
        begin, end = int(cdf.offsets[context]), int(cdf.offsets[context + 1])
        symbols, frequencies = _frequencies(cdf.records, begin, end)
        if not symbols:
            shifts.append(0)
            bucket_offsets.append(len(bucket_records))
            active_offsets.append(len(active_symbols))
            continue

        primary, aliases, cuts = _vose(symbols, frequencies)
        bucket_count = len(primary)
        shift = bucket_count.bit_length() - 1
        width = SLOT_TOTAL // bucket_count
        shifts.append(shift)
        assigned = {symbol: 0 for symbol in symbols}
        by_symbol: dict[int, list[tuple[int, int, int]]] = {symbol: [] for symbol in symbols}
        symbol_frequencies = dict(zip(symbols, frequencies))
        for bucket, (primary_symbol, alias_symbol, cut) in enumerate(zip(primary, aliases, cuts)):
            p_count, a_count = cut, width - cut
            p_base = assigned[primary_symbol] if p_count else 0
            if p_count:
                by_symbol[primary_symbol].append((p_base, p_count, bucket * width))
                assigned[primary_symbol] += p_count
            a_base = assigned[alias_symbol] if a_count else 0
            if a_count:
                by_symbol[alias_symbol].append((a_base, a_count, bucket * width + cut))
                assigned[alias_symbol] += a_count

            p_frequency = symbol_frequencies[primary_symbol]
            a_frequency = symbol_frequencies[alias_symbol]
            flags = 0
            if cut == SLOT_TOTAL:
                flags |= _FLAG_CUT_FULL
            if p_frequency == SLOT_TOTAL:
                flags |= _FLAG_PRIMARY_FREQ_FULL
            if a_frequency == SLOT_TOTAL:
                flags |= _FLAG_ALIAS_FREQ_FULL
            bucket_records.append((primary_symbol, alias_symbol, cut & 0xffff,
                p_base, a_base, p_frequency & 0xffff, a_frequency & 0xffff, flags))

        for symbol, frequency in zip(symbols, frequencies):
            if assigned[symbol] != frequency:
                raise ValueError("alias table does not preserve symbol frequency")
            active_symbols.append(symbol)
            active_frequencies.append(frequency)
            for rank_base, length, slot_base in sorted(by_symbol[symbol]):
                inverse_segments.append((rank_base, length, slot_base))
            inverse_symbol_offsets.append(len(inverse_segments))
        active_offsets.append(len(active_symbols))
        bucket_offsets.append(len(bucket_records))

    records_array = _freeze_aligned(np.asarray(bucket_records, dtype=np.uint16).reshape(-1, 8), np.dtype("<u2"))
    bucket_offsets_array = _freeze_aligned(np.asarray(bucket_offsets, dtype=np.uint32), np.dtype("<u4"), alignment=4)
    shifts_array = _freeze_aligned(np.asarray(shifts, dtype=np.uint8), np.dtype("u1"), alignment=1)
    active_symbols_array = _freeze_aligned(np.asarray(active_symbols, dtype=np.uint16), np.dtype("<u2"), alignment=2)
    active_frequencies_array = _freeze_aligned(np.asarray(active_frequencies, dtype=np.uint32), np.dtype("<u4"), alignment=4)
    active_offsets_array = _freeze_aligned(np.asarray(active_offsets, dtype=np.uint32), np.dtype("<u4"), alignment=4)
    inverse_symbol_offsets_array = _freeze_aligned(np.asarray(inverse_symbol_offsets, dtype=np.uint32), np.dtype("<u4"), alignment=4)
    inverse_segments_array = _freeze_aligned(np.asarray(inverse_segments, dtype=np.uint32).reshape(-1, 3), np.dtype("<u4"), alignment=4)
    digest = bytes.fromhex(cdf.source_sha256)
    decoder_ledger = MappingProxyType({
        "bucket_records": int(records_array.nbytes),
        "bucket_offsets": int(bucket_offsets_array.nbytes),
        "context_shifts": int(shifts_array.nbytes),
        "source_cdf_digest": len(digest),
    })
    inverse_ledger = MappingProxyType({
        "active_symbols": int(active_symbols_array.nbytes),
        "active_frequencies": int(active_frequencies_array.nbytes),
        "active_context_offsets": int(active_offsets_array.nbytes),
        "symbol_segment_offsets": int(inverse_symbol_offsets_array.nbytes),
        "inverse_segments": int(inverse_segments_array.nbytes),
    })
    decoder_bytes = sum(decoder_ledger.values())
    inverse_bytes = sum(inverse_ledger.values())
    if records_array.nbytes and records_array.ctypes.data % 16:
        raise ValueError("decoder records are not 16-byte aligned")
    return AliasTables(
        bucket_records=records_array,
        bucket_offsets=bucket_offsets_array,
        context_shifts=shifts_array,
        active_symbols=active_symbols_array,
        active_frequencies=active_frequencies_array,
        active_offsets=active_offsets_array,
        inverse_symbol_offsets=inverse_symbol_offsets_array,
        inverse_segments=inverse_segments_array,
        source_cdf_digest=digest,
        decoder_byte_ledger=decoder_ledger,
        inverse_byte_ledger=inverse_ledger,
        decoder_bytes=decoder_bytes,
        inverse_bytes=inverse_bytes,
    )


def _vector(values, name: str, *, maximum: int, max_value: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or (result.size and result.dtype.kind not in "ui"):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    if result.size > maximum or (result.size and (int(result.min()) < 0 or int(result.max()) > max_value)):
        raise ValueError(f"{name} exceeds bounded range")
    return np.ascontiguousarray(result, dtype=np.uint32)


def _validate_max_symbols(max_symbols: int) -> int:
    if type(max_symbols) is not int or not 1 <= max_symbols <= 524288:
        raise ValueError("max_symbols must be an integer in [1, 524288]")
    return max_symbols


def _sequence_count(values, name: str) -> int:
    """Read a bounded input length without converting or allocating its array."""
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional integer sequence")
        return int(values.size)
    if isinstance(values, memoryview):
        if values.ndim != 1 or not values.c_contiguous:
            raise ValueError(f"{name} must be a one-dimensional integer sequence")
    try:
        count = len(values)
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"{name} must be a sized sequence") from exc
    if type(count) is not int or count < 0:
        raise ValueError(f"{name} has an invalid sequence length")
    return count


def rans_encode(symbols, contexts, tables: AliasTables) -> bytes:
    """Encode one CPU rANS stream using the alias table's inverse slot map."""
    values = _vector(symbols, "symbols", maximum=MAX_SYMBOLS, max_value=MAX_SYMBOL_ID)
    context_ids = _vector(contexts, "contexts", maximum=MAX_SYMBOLS, max_value=MAX_CONTEXTS - 1)
    if len(values) != len(context_ids):
        raise ValueError("symbols and contexts have different lengths")
    if tables.context_count > MAX_CONTEXTS:
        raise ValueError("too many alias contexts")
    state = RANS_L
    renorm = bytearray()
    for index in range(len(values) - 1, -1, -1):
        symbol, context = int(values[index]), int(context_ids[index])
        frequency = tables.frequency(context, symbol)
        maximum_state = ((RANS_L >> SCALE) << 8) * frequency
        while state >= maximum_state:
            renorm.append(state & 0xff)
            state >>= 8
        rank = state % frequency
        slot = tables.encode_slot(context, symbol, rank)
        state = ((state // frequency) << SCALE) + slot
    if not RANS_L <= state < RANS_L * 256:
        raise ValueError("encoded rANS state is outside the normalized range")
    return state.to_bytes(4, "little") + bytes(reversed(renorm))


def rans_decode(stream: bytes | bytearray | memoryview, contexts, tables: AliasTables) -> np.ndarray:
    """Decode one CPU alias-rANS stream and require exact terminal consumption."""
    if not isinstance(stream, (bytes, bytearray, memoryview)):
        raise TypeError("stream must be bytes-like")
    data = bytes(stream)
    context_ids = _vector(contexts, "contexts", maximum=MAX_SYMBOLS, max_value=MAX_CONTEXTS - 1)
    if len(data) < 4 or len(data) > 4 + 3 * len(context_ids):
        raise ValueError("rANS stream length is outside bounded size")
    state = int.from_bytes(data[:4], "little")
    if not RANS_L <= state < RANS_L * 256:
        raise ValueError("invalid initial rANS state")
    output = np.empty(len(context_ids), dtype="<u2")
    position = 4
    for index, context_value in enumerate(context_ids):
        context = int(context_value)
        slot = state & (SLOT_TOTAL - 1)
        symbol, frequency, rank = tables.decode_slot(context, slot)
        output[index] = symbol
        state = frequency * (state >> SCALE) + rank
        while state < RANS_L:
            if position >= len(data):
                raise ValueError("rANS stream ended during renormalization")
            state = (state << 8) | data[position]
            position += 1
    if position != len(data) or state != RANS_L:
        raise ValueError("rANS stream failed terminal state/consumption check")
    return output


def encode_32lane(symbols, contexts, tables: AliasTables) -> bytes:
    """Encode a new AAN1 frame as 32 independent rANS streams."""
    values = _vector(symbols, "symbols", maximum=MAX_SYMBOLS, max_value=MAX_SYMBOL_ID)
    context_ids = _vector(contexts, "contexts", maximum=MAX_SYMBOLS, max_value=MAX_CONTEXTS - 1)
    if len(values) != len(context_ids):
        raise ValueError("symbols and contexts have different lengths")
    lanes = [rans_encode(values[lane::LANES], context_ids[lane::LANES], tables) for lane in range(LANES)]
    offsets = [0]
    payload = bytearray()
    for lane in lanes:
        payload.extend(lane)
        offsets.append(len(payload))
    directory = _LANE_DIRECTORY.pack(*offsets)
    body = directory + payload
    header = _HEADER.pack(MAGIC, VERSION, LANES, 0, len(values), len(payload),
                          tables.source_cdf_digest, hashlib.sha256(body).digest())
    return header + body


def decode_32lane(frame: bytes | bytearray | memoryview, contexts, tables: AliasTables) -> np.ndarray:
    """Decode AAN1 after validating geometry, table binding, checksum, and lane ends."""
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise TypeError("frame must be bytes-like")
    data = bytes(frame)
    context_ids = _vector(contexts, "contexts", maximum=MAX_SYMBOLS, max_value=MAX_CONTEXTS - 1)
    if len(data) < _HEADER.size + _LANE_DIRECTORY.size:
        raise ValueError("short AAN1 frame")
    magic, version, lane_count, flags, count, payload_bytes, cdf_digest, body_digest = _HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or lane_count != LANES or flags:
        raise ValueError("unsupported AAN1 header")
    if count != len(context_ids) or count > MAX_SYMBOLS or payload_bytes > 4 * LANES + 3 * count:
        raise ValueError("AAN1 count or payload bound is invalid")
    if cdf_digest != tables.source_cdf_digest:
        raise ValueError("AAN1 CDF digest mismatch")
    expected_size = _HEADER.size + _LANE_DIRECTORY.size + payload_bytes
    if len(data) != expected_size:
        raise ValueError("AAN1 length mismatch")
    body = data[_HEADER.size:]
    if hashlib.sha256(body).digest() != body_digest:
        raise ValueError("AAN1 body checksum mismatch")
    offsets = _LANE_DIRECTORY.unpack_from(data, _HEADER.size)
    if offsets[0] != 0 or offsets[-1] != payload_bytes or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("AAN1 lane directory is invalid")
    payload_start = _HEADER.size + _LANE_DIRECTORY.size
    output = np.empty(count, dtype="<u2")
    for lane in range(LANES):
        lane_contexts = context_ids[lane::LANES]
        stream = data[payload_start + offsets[lane]:payload_start + offsets[lane + 1]]
        lane_values = rans_decode(stream, lane_contexts, tables)
        expected_lane_count = max(0, (count + LANES - 1 - lane) // LANES)
        if len(lane_values) != expected_lane_count:
            raise ValueError("AAN1 lane symbol count mismatch")
        output[lane::LANES] = lane_values
    return output


def encode32_word(symbols, contexts, tables: AliasTables, *,
                  max_symbols: int = MAX_SYMBOLS) -> bytes:
    """Encode AAN2 with 32 multiplexed uint32 states and a shared u16 word stream.

    Blocks are visited backwards and lanes within each block descend. Every
    symbol emits at most one low word before its alias-rANS inverse transition.
    A final partial block uses its active lanes 0 through ``tail-1``.
    """
    cap = _validate_max_symbols(max_symbols)
    symbol_count = _sequence_count(symbols, "symbols")
    context_count = _sequence_count(contexts, "contexts")
    if symbol_count > cap or context_count > cap:
        raise ValueError("symbol/context count exceeds max_symbols")
    if symbol_count != context_count:
        raise ValueError("symbols and contexts have different lengths")
    values = _vector(symbols, "symbols", maximum=cap, max_value=MAX_SYMBOL_ID)
    context_ids = _vector(contexts, "contexts", maximum=cap, max_value=MAX_CONTEXTS - 1)
    states = [WORD_RANS_L] * LANES
    emitted_words = []
    count = symbol_count
    if count:
        last_start = ((count - 1) // LANES) * LANES
        for block_start in range(last_start, -1, -LANES):
            active = min(LANES, count - block_start)
            for lane in range(active - 1, -1, -1):
                index = block_start + lane
                symbol, context = int(values[index]), int(context_ids[index])
                frequency = tables.frequency(context, symbol)
                renorm_limit = WORD_RADIX * frequency
                state = states[lane]
                if state >= renorm_limit:
                    emitted_words.append(state & 0xffff)
                    state >>= 16
                if state >= renorm_limit:
                    raise ValueError("word rANS requires more than one renormalization word")
                rank = state % frequency
                slot = tables.encode_slot(context, symbol, rank)
                state = (state // frequency) * WORD_RADIX + slot
                if not WORD_RANS_L <= state < (1 << 32):
                    raise ValueError("encoded word rANS state is outside uint32 range")
                states[lane] = state
    if any(state != WORD_RANS_L for state in states[min(count, LANES):]):
        raise AssertionError("inactive tail lanes changed state")
    payload = b"".join(word.to_bytes(2, "little") for word in reversed(emitted_words))
    if len(payload) > 2 * count:
        raise AssertionError("word rANS emitted more than one word per symbol")
    states_bytes = struct.pack("<32I", *states)
    body = states_bytes + payload
    header = WORD_HEADER.pack(WORD_MAGIC, WORD_VERSION, LANES, 0, count, len(payload),
                              tables.source_cdf_digest, hashlib.sha256(body).digest())
    return header + body


def decode32_word(frame: bytes | bytearray | memoryview, contexts,
                  tables: AliasTables, *, max_symbols: int = MAX_SYMBOLS) -> np.ndarray:
    """Decode AAN2 using ballot-order refill from one common little-endian u16 stream."""
    cap = _validate_max_symbols(max_symbols)
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise TypeError("frame must be bytes-like")
    view = memoryview(frame)
    if view.ndim != 1 or view.itemsize != 1 or not view.c_contiguous:
        raise ValueError("frame must be a contiguous byte buffer")
    context_count = _sequence_count(contexts, "contexts")
    if context_count > cap:
        raise ValueError("context count exceeds max_symbols")
    if view.nbytes < WORD_HEADER.size + WORD_STATE_BYTES:
        raise ValueError("short AAN2 frame")
    magic, version, lane_count, flags, count, payload_bytes, cdf_digest, body_digest = WORD_HEADER.unpack_from(view)
    if magic != WORD_MAGIC or version != WORD_VERSION or lane_count != LANES or flags:
        raise ValueError("unsupported AAN2 header")
    if count != context_count or count > cap:
        raise ValueError("AAN2 symbol count mismatch")
    if payload_bytes > 2 * count or payload_bytes & 1:
        raise ValueError("AAN2 word payload length is noncanonical")
    if cdf_digest != tables.source_cdf_digest:
        raise ValueError("AAN2 CDF digest mismatch")
    expected_size = WORD_HEADER.size + WORD_STATE_BYTES + payload_bytes
    if view.nbytes != expected_size:
        raise ValueError("AAN2 frame length mismatch")
    body = view[WORD_HEADER.size:]
    if hashlib.sha256(body).digest() != body_digest:
        raise ValueError("AAN2 body checksum mismatch")

    context_ids = _vector(contexts, "contexts", maximum=cap, max_value=MAX_CONTEXTS - 1)
    states = list(struct.unpack_from("<32I", body, 0))
    active_lanes = min(count, LANES)
    for lane, state in enumerate(states):
        if lane < active_lanes:
            if not WORD_RANS_L <= state < (1 << 32):
                raise ValueError("invalid AAN2 initial state")
        elif state != WORD_RANS_L:
            raise ValueError("inactive AAN2 tail lane state is noncanonical")

    payload_start = WORD_STATE_BYTES
    cursor = 0
    output = np.empty(count, dtype="<u2")
    for block_start in range(0, count, LANES):
        active = min(LANES, count - block_start)
        refill_mask = 0
        next_states = [0] * active
        for lane in range(active):
            state = states[lane]
            if not WORD_RANS_L <= state < (1 << 32):
                raise ValueError("AAN2 lane state left uint32 normalized range")
            context = int(context_ids[block_start + lane])
            slot = state & (WORD_RADIX - 1)
            symbol, frequency, rank = tables.decode_slot(context, slot)
            output[block_start + lane] = symbol
            next_state = frequency * (state >> 16) + rank
            if not 0 <= next_state < (1 << 32):
                raise ValueError("AAN2 state transition overflow")
            next_states[lane] = next_state
            if next_state < WORD_RANS_L:
                if next_state == 0:
                    raise ValueError("AAN2 renormalization state is zero")
                refill_mask |= 1 << lane
            else:
                states[lane] = next_state

        refill_count = refill_mask.bit_count()
        if cursor + 2 * refill_count > payload_bytes:
            raise ValueError("truncated AAN2 common word stream")
        # The mask plays the role of a ballot; words are consumed in ascending
        # active-lane order, matching the forward block decoder order.
        for lane in range(active):
            if refill_mask & (1 << lane):
                word = struct.unpack_from("<H", body, payload_start + cursor)[0]
                cursor += 2
                state = (next_states[lane] << 16) | word
                if not WORD_RANS_L <= state < (1 << 32):
                    raise ValueError("AAN2 refill state is outside uint32 normalized range")
                states[lane] = state

    if cursor != payload_bytes or any(state != WORD_RANS_L for state in states):
        raise ValueError("AAN2 terminal state or common-stream consumption mismatch")
    return output
