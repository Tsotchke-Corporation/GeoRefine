#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

extern "C" {
static constexpr uint32_t kScale = 16;
static constexpr uint32_t kTotal = 1u << kScale;
static constexpr uint32_t kRansLowerBound = 1u << 23;

static uint32_t bcx_symbol(uint16_t word) {
  return (((word >> 7) & 255u) << 2) | ((word & 127u) >> 5);
}

static uint32_t bcx_residual(uint16_t word) {
  return ((word >> 15) << 5) | (word & 31u);
}

int bcx_checkpoint(const uint8_t *stream, size_t stream_bytes, uint64_t n,
                   uint32_t rows, uint32_t cols, const uint8_t *rc,
                   const uint8_t *cc, uint32_t nr, uint32_t nc,
                   uint32_t kind, const uint16_t *resctx,
                   const uint32_t *freq, uint32_t stride, uint32_t *states,
                   uint32_t *offsets, uint16_t *out) {
  try {
    if (!stream || !rc || !cc || !freq || !states || !offsets || !out ||
        !rows || !cols || !nr || !nc || nr > 64 || nc > 16 || !stride ||
        kind > 1 || (kind == 1 && !resctx) ||
        n != uint64_t(rows) * uint64_t(cols) || !n ||
        stream_bytes < 4 || stream_bytes > UINT32_MAX ||
        n > std::numeric_limits<size_t>::max())
      return 1;

    const uint32_t vocab = kind == 0 ? 1024u : 64u;
    const uint32_t nctx = kind == 0 ? nr * nc : 1024u;
    const uint64_t checkpoint_count = (n - 1) / stride + 1;
    if (checkpoint_count > std::numeric_limits<size_t>::max())
      return 1;

    for (uint32_t row = 0; row < rows; ++row)
      if (rc[row] >= nr)
        return 1;
    for (uint32_t col = 0; col < cols; ++col)
      if (cc[col] >= nc)
        return 1;
    if (kind == 1)
      for (uint64_t i = 0; i < n; ++i)
        if (resctx[i] >= nctx)
          return 1;

    // Validate each fixed-point table row and build cumulative starts.
    std::vector<uint32_t> cumulative(static_cast<size_t>(nctx) * vocab);
    std::vector<uint16_t> lookup(static_cast<size_t>(nctx) * kTotal);
    std::vector<uint8_t> active(nctx, 0);
    for (uint32_t cx = 0; cx < nctx; ++cx) {
      uint64_t total = 0;
      for (uint32_t s = 0; s < vocab; ++s) {
        const uint32_t f = freq[static_cast<size_t>(cx) * vocab + s];
        if (f > kTotal || total + f > kTotal)
          return 2;
        cumulative[static_cast<size_t>(cx) * vocab + s] =
            static_cast<uint32_t>(total);
        std::fill(lookup.begin() + static_cast<size_t>(cx) * kTotal + total,
                  lookup.begin() + static_cast<size_t>(cx) * kTotal + total + f,
                  static_cast<uint16_t>(s));
        total += f;
      }
      if (total != 0 && total != kTotal)
        return 2;
      active[cx] = total == kTotal;
    }

    uint32_t state = uint32_t(stream[0]) | (uint32_t(stream[1]) << 8) |
                     (uint32_t(stream[2]) << 16) |
                     (uint32_t(stream[3]) << 24);
    if (state < kRansLowerBound)
      return 3;
    size_t ip = 4;
    std::vector<uint32_t> saved_states(static_cast<size_t>(checkpoint_count));
    std::vector<uint32_t> saved_offsets(static_cast<size_t>(checkpoint_count));
    std::vector<uint16_t> decoded(static_cast<size_t>(n));
    uint64_t next_checkpoint = 0;
    size_t checkpoint_slot = 0;
    uint32_t row_index = 0, col_index = 0;
    for (uint64_t i = 0; i < n; ++i) {
      if (i == next_checkpoint) {
        if (ip > UINT32_MAX)
          return 4;
        saved_states[checkpoint_slot] = state;
        saved_offsets[checkpoint_slot] = static_cast<uint32_t>(ip);
        ++checkpoint_slot;
        next_checkpoint += stride;
      }
      const uint32_t cx = kind == 0
          ? uint32_t(rc[row_index]) * nc + cc[col_index]
          : resctx[i];
      if (cx >= nctx || !active[cx])
        return 2;
      const size_t base = static_cast<size_t>(cx) * vocab;
      const uint32_t slot = state & (kTotal - 1);
      const uint32_t *cum = cumulative.data() + base;
      const uint32_t lo = lookup[static_cast<size_t>(cx) * kTotal + slot];
      if (lo >= vocab)
        return 2;
      const uint32_t fr = freq[base + lo];
      if (!fr || slot < cum[lo] || slot - cum[lo] >= fr)
        return 2;
      decoded[static_cast<size_t>(i)] = static_cast<uint16_t>(lo);
      const uint64_t next = uint64_t(fr) * (state >> kScale) + (slot - cum[lo]);
      if (next > UINT32_MAX)
        return 3;
      state = static_cast<uint32_t>(next);
      while (state < kRansLowerBound) {
        if (ip >= stream_bytes)
          return 3;
        state = (state << 8) | stream[ip++];
      }
      if (++col_index == cols) {
        col_index = 0;
        ++row_index;
      }
    }
    if (ip != stream_bytes || state != kRansLowerBound)
      return 3;

    std::copy(saved_states.begin(), saved_states.end(), states);
    std::copy(saved_offsets.begin(), saved_offsets.end(), offsets);
    std::copy(decoded.begin(), decoded.end(), out);
    return 0;
  } catch (const std::bad_alloc &) {
    return 5;
  } catch (...) {
    return 6;
  }
}
}
