#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
constexpr uint32_t kRansL = 1u << 16;
constexpr uint32_t kTotal = 1u << 16;
constexpr uint32_t kKnownFlags = 7;

struct Tables {
  const uint16_t *records;
  size_t record_rows;
  const uint32_t *bucket_offsets;
  const uint8_t *shifts;
  const uint16_t *active_symbols;
  size_t active_count;
  const uint32_t *active_frequencies;
  const uint32_t *active_offsets;
  const uint32_t *segment_offsets;
  const uint32_t *segments;
  size_t segment_count;
  size_t contexts;
};

bool decode_slot(const Tables &t, uint32_t context, uint32_t slot,
                 uint32_t *symbol, uint32_t *frequency, uint32_t *rank) {
  if (context >= t.contexts || slot >= kTotal) return false;
  const uint32_t first = t.bucket_offsets[context];
  const uint32_t end = t.bucket_offsets[context + 1];
  const uint8_t shift = t.shifts[context];
  if (first >= end || end > t.record_rows || shift > 10 || end - first != (1u << shift)) return false;
  const uint16_t *r = t.records + (first + (slot >> (16 - shift))) * 8;
  const uint32_t p = r[0], a = r[1], flags = r[7];
  if (p > 1023 || a > 1023 || (flags & ~kKnownFlags)) return false;
  const uint32_t width = 1u << (16 - shift);
  const uint32_t cut = (flags & 1) ? kTotal : r[2];
  const uint32_t pf = (flags & 2) ? kTotal : r[5];
  const uint32_t af = (flags & 4) ? kTotal : r[6];
  if (cut > width || pf == 0 || af == 0 || pf > kTotal || af > kTotal) return false;
  const uint32_t within = slot & (width - 1);
  if (within < cut) {
    *symbol = p; *frequency = pf; *rank = uint32_t(r[3]) + within;
  } else {
    *symbol = a; *frequency = af; *rank = uint32_t(r[4]) + within - cut;
  }
  return *rank < *frequency;
}

bool find_frequency(const Tables &t, uint32_t context, uint32_t symbol, uint32_t *freq,
                    uint32_t *active_index) {
  if (context >= t.contexts || symbol > 1023) return false;
  const uint32_t begin = t.active_offsets[context], end = t.active_offsets[context + 1];
  if (begin >= end || end > t.active_count) return false;
  uint32_t lo = begin, hi = end;
  while (lo < hi) { uint32_t mid = lo + (hi - lo) / 2; if (t.active_symbols[mid] < symbol) lo = mid + 1; else hi = mid; }
  if (lo == end || t.active_symbols[lo] != symbol) return false;
  *freq = t.active_frequencies[lo];
  *active_index = lo;
  return *freq > 0 && *freq <= kTotal;
}

bool inverse_slot(const Tables &t, uint32_t context, uint32_t symbol, uint32_t rank,
                  uint32_t frequency, uint32_t active_index, uint32_t *slot) {
  uint32_t begin = t.segment_offsets[active_index], end = t.segment_offsets[active_index + 1];
  if (begin >= end || end > t.segment_count) return false;
  uint32_t lo = begin, hi = end;
  while (lo < hi) { uint32_t mid = lo + (hi - lo) / 2; if (t.segments[mid * 3] <= rank) lo = mid + 1; else hi = mid; }
  if (lo == begin) return false;
  const uint32_t row = lo - 1;
  const uint32_t rb = t.segments[row * 3], len = t.segments[row * 3 + 1], sb = t.segments[row * 3 + 2];
  if (rank < rb || rank - rb >= len || sb > 65535 || len > 65536 - sb) return false;
  const uint32_t candidate = sb + rank - rb;
  uint32_t ds, df, dr;
  if (!decode_slot(t, context, candidate, &ds, &df, &dr) || ds != symbol || df != frequency || dr != rank) return false;
  *slot = candidate;
  return true;
}

bool sane(const Tables &t) {
  if ((t.record_rows && !t.records) || !t.bucket_offsets || !t.shifts ||
      (t.active_count && (!t.active_symbols || !t.active_frequencies)) || !t.active_offsets ||
      !t.segment_offsets || (t.segment_count && !t.segments) ||
      t.contexts == 0 || t.contexts > 1024) return false;
  if (t.bucket_offsets[0] != 0 || t.bucket_offsets[t.contexts] != t.record_rows ||
      t.active_offsets[0] != 0 || t.active_offsets[t.contexts] != t.active_count ||
      t.segment_offsets[0] != 0 || t.segment_offsets[t.active_count] != t.segment_count) return false;
  for (size_t c = 0; c < t.contexts; ++c) {
    if (t.bucket_offsets[c] > t.bucket_offsets[c + 1] || t.active_offsets[c] > t.active_offsets[c + 1]) return false;
    if (t.bucket_offsets[c] != t.bucket_offsets[c + 1] &&
        (t.shifts[c] > 10 || t.bucket_offsets[c + 1] - t.bucket_offsets[c] != (1u << t.shifts[c]))) return false;
  }
  for (size_t i = 0; i < t.active_count; ++i) {
    if (t.active_frequencies[i] == 0 || t.active_frequencies[i] > kTotal || t.segment_offsets[i] > t.segment_offsets[i + 1]) return false;
    if (i && t.active_offsets[0] == 0) { /* context ordering checked per operation */ }
  }
  for (size_t i = 0; i < t.segment_count; ++i) {
    if (t.segments[i * 3 + 1] == 0 || t.segments[i * 3] > 65536 || t.segments[i * 3 + 1] > 65536 - t.segments[i * 3] ||
        t.segments[i * 3 + 2] > 65535 || t.segments[i * 3 + 1] > 65536 - t.segments[i * 3 + 2]) return false;
  }
  return true;
}
}  // namespace

extern "C" int aan2_encode(const uint32_t *symbols, const uint32_t *contexts, uint32_t count,
                            const uint16_t *records, uint32_t record_rows,
                            const uint32_t *bucket_offsets, const uint8_t *shifts,
                            const uint16_t *active_symbols, uint32_t active_count,
                            const uint32_t *active_frequencies, const uint32_t *active_offsets,
                            const uint32_t *segment_offsets, const uint32_t *segments,
                            uint32_t segment_count, uint32_t context_count,
                            uint8_t *body, uint32_t *payload_bytes) {
  Tables t{records, record_rows, bucket_offsets, shifts, active_symbols, active_count,
           active_frequencies, active_offsets, segment_offsets, segments, segment_count, context_count};
  if (!body || !payload_bytes || !sane(t)) return 1;
  uint32_t states[32]; std::fill(states, states + 32, kRansL);
  std::vector<uint16_t> emitted; emitted.reserve(count);
  if (count) {
    const int64_t last = (int64_t(count - 1) / 32) * 32;
    for (int64_t start = last; start >= 0; start -= 32) {
      const uint32_t active = std::min<uint32_t>(32, count - uint32_t(start));
      for (int lane = int(active) - 1; lane >= 0; --lane) {
        uint32_t freq, ai;
        if (!find_frequency(t, contexts[start + lane], symbols[start + lane], &freq, &ai)) return 2;
        uint32_t state = states[lane];
        const uint64_t limit = uint64_t(kTotal) * freq;
        if (state >= limit) { emitted.push_back(uint16_t(state)); state >>= 16; }
        if (state >= limit) return 3;
        const uint32_t rank = state % freq;
        uint32_t slot;
        if (!inverse_slot(t, contexts[start + lane], symbols[start + lane], rank, freq, ai, &slot)) return 4;
        const uint64_t next = uint64_t(state / freq) * kTotal + slot;
        if (next < kRansL || next >= (uint64_t(1) << 32)) return 5;
        states[lane] = uint32_t(next);
      }
    }
  }
  for (int lane = int(std::min<uint32_t>(count, 32)); lane < 32; ++lane) if (states[lane] != kRansL) return 6;
  const uint32_t bytes = uint32_t(emitted.size() * 2);
  if (bytes > count * 2) return 7;
  std::memcpy(body, states, sizeof(states));
  for (size_t i = 0; i < emitted.size(); ++i) {
    const uint16_t word = emitted[emitted.size() - 1 - i];
    body[128 + 2 * i] = uint8_t(word); body[129 + 2 * i] = uint8_t(word >> 8);
  }
  *payload_bytes = bytes;
  return 0;
}

extern "C" int aan2_decode(const uint8_t *body, uint32_t count, uint32_t payload_bytes,
                            const uint32_t *contexts,
                            const uint16_t *records, uint32_t record_rows,
                            const uint32_t *bucket_offsets, const uint8_t *shifts,
                            const uint16_t *active_symbols, uint32_t active_count,
                            const uint32_t *active_frequencies, const uint32_t *active_offsets,
                            const uint32_t *segment_offsets, const uint32_t *segments,
                            uint32_t segment_count, uint32_t context_count,
                            uint16_t *output) {
  Tables t{records, record_rows, bucket_offsets, shifts, active_symbols, active_count,
           active_frequencies, active_offsets, segment_offsets, segments, segment_count, context_count};
  if (!body || !output || !sane(t)) return 1;
  uint32_t states[32]; std::memcpy(states, body, sizeof(states));
  const uint32_t active_lanes = std::min<uint32_t>(count, 32);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    if (lane < active_lanes) { if (states[lane] < kRansL) return 2; }
    else if (states[lane] != kRansL) return 3;
  }
  uint32_t cursor = 0;
  for (uint32_t start = 0; start < count; start += 32) {
    const uint32_t active = std::min<uint32_t>(32, count - start);
    uint32_t next[32]{}; uint32_t refill_mask = 0;
    for (uint32_t lane = 0; lane < active; ++lane) {
      const uint32_t state = states[lane];
      if (state < kRansL) return 4;
      uint32_t symbol, freq, rank;
      if (!decode_slot(t, contexts[start + lane], state & 65535, &symbol, &freq, &rank)) return 5;
      output[start + lane] = uint16_t(symbol);
      const uint64_t ns = uint64_t(freq) * (state >> 16) + rank;
      if (ns >= (uint64_t(1) << 32)) return 6;
      next[lane] = uint32_t(ns);
      if (next[lane] < kRansL) { if (next[lane] == 0) return 7; refill_mask |= 1u << lane; }
      else states[lane] = next[lane];
    }
    for (uint32_t lane = 0; lane < active; ++lane) {
      if (refill_mask & (1u << lane)) {
        if (cursor + 2 > payload_bytes) return 8;
        const uint32_t word = uint32_t(body[128 + cursor]) | (uint32_t(body[129 + cursor]) << 8);
        cursor += 2;
        const uint64_t refilled = (uint64_t(next[lane]) << 16) | word;
        if (refilled < kRansL || refilled >= (uint64_t(1) << 32)) return 9;
        states[lane] = uint32_t(refilled);
      }
    }
  }
  if (cursor != payload_bytes) return 10;
  for (uint32_t state : states) if (state != kRansL) return 11;
  return 0;
}
