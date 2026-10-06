#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>
extern "C" {
static constexpr uint32_t SCALE = 16, TOT = 1u << SCALE, RANS_L = 1u << 23;
static uint32_t symbol(uint16_t w) {
  return (((w >> 7) & 255u) << 2) | ((w & 127u) >> 5);
}
static uint32_t residual(uint16_t w) { return ((w >> 15) << 5) | (w & 31u); }
static uint32_t context_at(uint64_t i, uint32_t cols, const uint8_t *r,
                           const uint8_t *c, uint32_t nc) {
  return uint32_t(r[i / cols]) * nc + c[i % cols];
}
static bool normalize(const uint64_t *cnt, uint32_t vocab, uint32_t *out) {
  uint64_t n = 0;
  uint32_t k = 0;
  for (uint32_t s = 0; s < vocab; s++) {
    n += cnt[s];
    k += cnt[s] != 0;
  }
  if (!n)
    return false;
  struct Rem {
    uint32_t s;
    uint64_t r;
  };
  std::vector<Rem> rem;
  rem.reserve(k);
  int64_t sum = 0;
  for (uint32_t s = 0; s < vocab; s++) {
    if (!cnt[s]) {
      out[s] = 0;
      continue;
    }
    uint64_t prod = cnt[s] * uint64_t(TOT);
    uint32_t f = uint32_t(prod / n);
    uint64_t r = prod % n;
    if (!f)
      f = 1;
    out[s] = f;
    sum += f;
    rem.push_back({s, r});
  }
  std::sort(rem.begin(), rem.end(),
            [](auto a, auto b) { return a.r != b.r ? a.r > b.r : a.s < b.s; });
  if (sum < TOT) {
    uint32_t d = TOT - (uint32_t)sum;
    for (uint32_t j = 0; j < d; j++)
      out[rem[j % rem.size()].s]++;
  } else if (sum > TOT) {
    std::sort(rem.begin(), rem.end(), [](auto a, auto b) {
      return a.r != b.r ? a.r < b.r : a.s > b.s;
    });
    uint32_t d = (uint32_t)sum - TOT;
    for (uint32_t j = 0; d; j = (j + 1) % rem.size()) {
      auto s = rem[j].s;
      if (out[s] > 1) {
        out[s]--;
        d--;
      }
    }
  }
  return true;
}
static void build_cum(const uint32_t *f, uint32_t v, uint32_t *cum) {
  uint32_t s = 0;
  for (uint32_t i = 0; i < v; i++) {
    cum[i] = s;
    s += f[i];
  }
}
static void rans_encode(const uint16_t *vals, uint64_t n, uint32_t vocab,
                        const uint32_t *freq, const uint32_t *nctx,
                        uint32_t kind, uint32_t cols, const uint8_t *rows,
                        const uint8_t *col, uint32_t nc,
                        std::vector<uint8_t> &out) {
  std::vector<uint32_t> cum((size_t)nctx[0] * vocab);
  for (uint32_t cx = 0; cx < nctx[0]; cx++)
    build_cum(freq + (size_t)cx * vocab, vocab,
              cum.data() + (size_t)cx * vocab);
  uint32_t state = RANS_L;
  std::vector<uint8_t> ren;
  ren.reserve((size_t)n / 2);
  for (uint64_t z = n; z-- > 0;) {
    uint32_t s = kind == 0 ? symbol(vals[z]) : residual(vals[z]);
    uint32_t cx =
        kind == 0 ? context_at(z, cols, rows, col, nc) : symbol(vals[z]);
    const uint32_t *f = freq + (size_t)cx * vocab;
    uint32_t fr = f[s], cu = cum[(size_t)cx * vocab + s];
    if (!fr)
      throw 1;
    uint64_t xmax = ((uint64_t)(RANS_L >> SCALE) << 8) * fr;
    while (state >= xmax) {
      ren.push_back(state & 255);
      state >>= 8;
    }
    state = ((state / fr) << SCALE) + (state % fr) + cu;
  }
  out.resize(4 + ren.size());
  out[0] = state;
  out[1] = state >> 8;
  out[2] = state >> 16;
  out[3] = state >> 24;
  for (size_t i = 0; i < ren.size(); i++)
    out[4 + i] = ren[ren.size() - 1 - i];
}
static bool rans_decode(const uint8_t *stream, size_t bytes, uint64_t n,
                        uint32_t vocab, const uint32_t *freq, uint32_t nctx,
                        uint32_t kind, uint32_t cols, const uint8_t *rows,
                        const uint8_t *col, uint32_t nc, const uint16_t *resctx,
                        uint16_t *outv) {
  if (bytes < 4)
    return false;
  uint32_t state = stream[0] | uint32_t(stream[1]) << 8 |
                   uint32_t(stream[2]) << 16 | uint32_t(stream[3]) << 24;
  if (state < RANS_L)
    return false;
  size_t ip = 4;
  std::vector<uint32_t> cum((size_t)nctx * vocab);
  std::vector<uint16_t> lut((size_t)nctx * TOT);
  for (uint32_t cx = 0; cx < nctx; cx++) {
    const uint32_t *f = freq + (size_t)cx * vocab;
    uint32_t s = 0;
    for (uint32_t j = 0; j < vocab; j++) {
      cum[(size_t)cx * vocab + j] = s;
      uint32_t fr = f[j];
      if (fr) {
        if (fr > TOT - s)
          return false;
        std::fill(lut.begin() + (size_t)cx * TOT + s,
                  lut.begin() + (size_t)cx * TOT + s + fr, (uint16_t)j);
        s += fr;
      }
    }
    if (s && s != TOT)
      return false;
  }
  for (uint64_t i = 0; i < n; i++) {
    uint32_t cx = kind == 0 ? context_at(i, cols, rows, col, nc) : resctx[i];
    if (cx >= nctx)
      return false;
    uint32_t slot = state & (TOT - 1);
    uint32_t s = lut[(size_t)cx * TOT + slot];
    uint32_t fr = freq[(size_t)cx * vocab + s],
             cu = cum[(size_t)cx * vocab + s];
    if (!fr)
      return false;
    outv[i] = (uint16_t)s;
    state = fr * (state >> SCALE) + (slot - cu);
    while (state < RANS_L) {
      if (ip >= bytes)
        return false;
      state = (state << 8) | stream[ip++];
    }
  }
  return ip == bytes && state == RANS_L;
}
int bcx_encode(const uint16_t *w, uint64_t n, uint32_t rows_n, uint32_t cols_n,
               const uint8_t *rc, const uint8_t *cc, uint32_t nr, uint32_t nc,
               uint32_t kind, uint32_t *freq, uint8_t **stream,
               size_t *stream_n) {
  try {
    uint32_t vocab = kind ? 64 : 1024, nctx = kind ? 1024 : nr * nc;
    if (!w || !rc || !cc || !freq || !stream || !stream_n ||
        n != uint64_t(rows_n) * cols_n)
      return 1;
    std::vector<uint64_t> counts((size_t)nctx * vocab);
    for (uint64_t i = 0; i < n; i++) {
      uint32_t s = kind ? residual(w[i]) : symbol(w[i]);
      uint32_t cx = kind ? symbol(w[i]) : context_at(i, cols_n, rc, cc, nc);
      counts[(size_t)cx * vocab + s]++;
    }
    for (uint32_t cx = 0; cx < nctx; cx++)
      normalize(counts.data() + (size_t)cx * vocab, vocab,
                freq + (size_t)cx * vocab);
    std::vector<uint8_t> out;
    uint32_t tmp = nctx;
    rans_encode(w, n, vocab, freq, &tmp, kind, cols_n, rc, cc, nc, out);
    auto *p = (uint8_t *)std::malloc(out.size());
    if (!p)
      return 2;
    std::memcpy(p, out.data(), out.size());
    *stream = p;
    *stream_n = out.size();
    return 0;
  } catch (...) {
    return 3;
  }
}
int bcx_decode(const uint8_t *stream, size_t stream_n, uint64_t n,
               uint32_t rows_n, uint32_t cols_n, const uint8_t *rc,
               const uint8_t *cc, uint32_t nr, uint32_t nc, uint32_t kind,
               const uint16_t *resctx, const uint32_t *freq, uint16_t *out) {
  try {
    uint32_t vocab = kind ? 64 : 1024, nctx = kind ? 1024 : nr * nc;
    if (!stream || !rc || !cc || !freq || !out || !cols_n || !rows_n ||
        kind > 1 || !nr || !nc || nr > 64 || nc > 16 || (kind && !resctx) ||
        n != uint64_t(rows_n) * cols_n)
      return 1;
    return rans_decode(stream, stream_n, n, vocab, freq, nctx, kind, cols_n, rc,
                       cc, nc, resctx, out)
               ? 0
               : 2;
  } catch (...) {
    return 3;
  }
}
void bcx_free(void *p) { std::free(p); }
int bcx_pack6(const uint16_t *w, uint64_t n, uint8_t **out, size_t *bytes) {
  if (!w || !out || !bytes)
    return 1;
  size_t z = (n * 6 + 7) / 8;
  auto *p = (uint8_t *)std::calloc(z, 1);
  if (!p)
    return 2;
  uint64_t bit = 0;
  for (uint64_t i = 0; i < n; i++) {
    uint32_t v = residual(w[i]);
    for (int j = 5; j >= 0; j--, bit++)
      if ((v >> j) & 1)
        p[bit >> 3] |= uint8_t(1u << (7 - (bit & 7)));
  }
  *out = p;
  *bytes = z;
  return 0;
}
int bcx_unpack6(const uint8_t *in, size_t bytes, uint64_t n, uint16_t *out) {
  if (!in || !out || bytes != (n * 6 + 7) / 8)
    return 1;
  if (n * 6 % 8 && (in[bytes - 1] & ((1u << (8 - n * 6 % 8)) - 1)))
    return 2;
  uint64_t bit = 0;
  for (uint64_t i = 0; i < n; i++) {
    uint16_t v = 0;
    for (int j = 0; j < 6; j++, bit++)
      v = (v << 1) | ((in[bit >> 3] >> (7 - (bit & 7))) & 1);
    out[i] = v;
  }
  return 0;
}
}
