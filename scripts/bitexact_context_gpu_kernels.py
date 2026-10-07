"""CUDA kernels for independently indexed segments of unchanged BCTX streams."""
import triton
import triton.language as tl


@triton.jit
def decode_segments(stream, states, offsets, freq, cum, rows, columns, block_ids,
                    main_context, output, error, n, cols, nc, stream_bytes, count,
                    VOCAB: tl.constexpr, LOG_VOCAB: tl.constexpr,
                    KIND: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active_lane = lane < count
    block = tl.load(block_ids + lane, active_lane, 0).to(tl.int64)
    state = tl.load(states + block, active_lane, 1 << 23).to(tl.uint32)
    ip = tl.load(offsets + block, active_lane, 4).to(tl.int64)
    for step in range(STRIDE):
        i = block * STRIDE + step
        local = lane * STRIDE + step
        active = active_lane & (i < n)
        if KIND == 0:
            r = tl.load(rows + i // cols, active, 0).to(tl.int32)
            c = tl.load(columns + i % cols, active, 0).to(tl.int32)
            context = r * nc + c
        else:
            context = tl.load(main_context + local, active, 0).to(tl.int32)
        slot = (state & 65535).to(tl.int32)
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = tl.full((BLOCK,), VOCAB, tl.int32)
        for _ in range(LOG_VOCAB + 1):
            mid = (lo + hi) // 2
            boundary = tl.load(cum + context * (VOCAB + 1) + mid, active, 0)
            searching = lo < hi
            greater = boundary <= slot
            lo = tl.where(searching & greater, mid + 1, lo)
            hi = tl.where(searching & ~greater, mid, hi)
        symbol = lo - 1
        valid_symbol = (symbol >= 0) & (symbol < VOCAB)
        fr = tl.load(freq + context * VOCAB + symbol, active & valid_symbol, 1).to(tl.uint32)
        cu = tl.load(cum + context * (VOCAB + 1) + symbol, active & valid_symbol, 0).to(tl.uint32)
        bad = active & (~valid_symbol | (fr == 0))
        if tl.sum(bad.to(tl.int32), 0) > 0:
            tl.atomic_or(error, 1)
        tl.store(output + local, symbol.to(tl.uint16), active)
        state = tl.where(active, fr * (state >> 16) + slot.to(tl.uint32) - cu, state)
        for _ in range(3):
            need = active & (state < (1 << 23))
            in_bounds = ip < stream_bytes
            byte = tl.load(stream + ip, need & in_bounds, 0).to(tl.uint32)
            if tl.sum((need & ~in_bounds).to(tl.int32), 0) > 0:
                tl.atomic_or(error, 2)
            state = tl.where(need, (state << 8) | byte, state)
            ip += need.to(tl.int64)
        if tl.sum((active & (state < (1 << 23))).to(tl.int32), 0) > 0:
            tl.atomic_or(error, 4)
    has_next = (block + 1) * STRIDE < n
    expected_state = tl.load(states + block + 1, active_lane & has_next, 1 << 23).to(tl.uint32)
    expected_ip = tl.load(offsets + block + 1, active_lane & has_next, stream_bytes).to(tl.int64)
    if tl.sum((active_lane & ((state != expected_state) | (ip != expected_ip))).to(tl.int32), 0) > 0:
        tl.atomic_or(error, 8)


@triton.jit
def combine(main, residual, packed, block_ids, out, n, count, packed_bytes,
            RAW6: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = i // STRIDE
    active = lane < count
    block = tl.load(block_ids + lane, active, 0).to(tl.int64)
    source = block * STRIDE + i % STRIDE
    active = active & (source < n)
    symbol = tl.load(main + i, active, 0).to(tl.uint32)
    if RAW6:
        bit = source * 6
        byte = bit // 8
        hi = tl.load(packed + byte, active & (byte < packed_bytes), 0).to(tl.uint32)
        low = tl.load(packed + byte + 1, active & (byte + 1 < packed_bytes), 0).to(tl.uint32)
        r = (((hi << 8) | low) >> (10 - bit % 8)) & 63
    else:
        r = tl.load(residual + i, active, 0).to(tl.uint32)
    word = ((symbol >> 2) << 7) | ((symbol & 3) << 5) | (r & 31) | ((r >> 5) << 15)
    tl.store(out + i, word.to(tl.uint16), active)


@triton.jit
def _read_symbol(state, ip, stream, freq, cum, context, active, stream_bytes,
                 VOCAB: tl.constexpr, LOG_VOCAB: tl.constexpr, BLOCK: tl.constexpr):
    slot = (state & 65535).to(tl.int32)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), VOCAB, tl.int32)
    for _ in range(LOG_VOCAB + 1):
        mid = (lo + hi) // 2
        searching = lo < hi
        boundary = tl.load(cum + context * (VOCAB + 1) + mid, active & searching, 0)
        greater = boundary <= slot
        lo = tl.where(searching & greater, mid + 1, lo)
        hi = tl.where(searching & ~greater, mid, hi)
    symbol = lo - 1
    valid = (symbol >= 0) & (symbol < VOCAB)
    fr = tl.load(freq + context * VOCAB + symbol, active & valid, 1).to(tl.uint32)
    cu = tl.load(cum + context * (VOCAB + 1) + symbol, active & valid, 0).to(tl.uint32)
    bad = active & (~valid | (fr == 0))
    state = tl.where(active, fr * (state >> 16) + slot.to(tl.uint32) - cu, state)
    for _ in range(3):
        need = active & (state < (1 << 23))
        available = ip < stream_bytes
        byte = tl.load(stream + ip, need & available, 0).to(tl.uint32)
        bad = bad | (need & ~available)
        state = tl.where(need, (state << 8) | byte, state)
        ip += need.to(tl.int64)
    bad = bad | (active & (state < (1 << 23)))
    return symbol, state, ip, bad


@triton.jit
def decode_words(main_stream, main_states, main_offsets, main_freq, main_cum,
                 res_stream, res_states, res_offsets, res_freq, res_cum,
                 rows, columns, block_ids, out, error,
                 n, cols, nc, main_bytes, res_bytes, count,
                 RAW6: tl.constexpr, DIRECT: tl.constexpr,
                 STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid_lane = lane < count
    if DIRECT:
        block = lane.to(tl.int64)
    else:
        block = tl.load(block_ids + lane, valid_lane, 0).to(tl.int64)
    start = block * STRIDE
    row = start // cols
    col = start % cols
    state = tl.load(main_states + block, valid_lane, 1 << 23).to(tl.uint32)
    ip = tl.load(main_offsets + block, valid_lane, 4).to(tl.int64)
    if not RAW6:
        rstate = tl.load(res_states + block, valid_lane, 1 << 23).to(tl.uint32)
        rip = tl.load(res_offsets + block, valid_lane, 4).to(tl.int64)
    any_bad = tl.full((BLOCK,), False, tl.int1)
    for step in range(STRIDE):
        source = start + step
        active = valid_lane & (source < n)
        rc = tl.load(rows + row, active, 0).to(tl.int32)
        cc = tl.load(columns + col, active, 0).to(tl.int32)
        symbol, state, ip, bad = _read_symbol(state, ip, main_stream, main_freq, main_cum,
            rc * nc + cc, active, main_bytes, 1024, 10, BLOCK)
        any_bad = any_bad | bad
        if RAW6:
            bit = source * 6
            byte = bit // 8
            hi = tl.load(res_stream + byte, active & (byte < res_bytes), 0).to(tl.uint32)
            low = tl.load(res_stream + byte + 1, active & (byte + 1 < res_bytes), 0).to(tl.uint32)
            residual = (((hi << 8) | low) >> (10 - bit % 8)) & 63
        else:
            residual, rstate, rip, bad = _read_symbol(rstate, rip, res_stream, res_freq, res_cum,
                symbol, active, res_bytes, 64, 6, BLOCK)
            any_bad = any_bad | bad
        word = ((symbol.to(tl.uint32) >> 2) << 7) | ((symbol.to(tl.uint32) & 3) << 5) | (residual.to(tl.uint32) & 31) | ((residual.to(tl.uint32) >> 5) << 15)
        tl.store(out + lane * STRIDE + step, word.to(tl.uint16), active)
        col += 1
        crossed = col >= cols
        row += crossed.to(tl.int64)
        col = tl.where(crossed, 0, col)
    has_next = (block + 1) * STRIDE < n
    next_state = tl.load(main_states + block + 1, valid_lane & has_next, 1 << 23).to(tl.uint32)
    next_ip = tl.load(main_offsets + block + 1, valid_lane & has_next, main_bytes).to(tl.int64)
    any_bad = any_bad | (valid_lane & ((state != next_state) | (ip != next_ip)))
    if not RAW6:
        next_state = tl.load(res_states + block + 1, valid_lane & has_next, 1 << 23).to(tl.uint32)
        next_ip = tl.load(res_offsets + block + 1, valid_lane & has_next, res_bytes).to(tl.int64)
        any_bad = any_bad | (valid_lane & ((rstate != next_state) | (rip != next_ip)))
    if tl.sum(any_bad.to(tl.int32), 0) > 0:
        tl.atomic_or(error, 1)
