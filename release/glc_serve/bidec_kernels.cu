// bidec_kernels.cu -- the per-row / per-sequence kernels of the continuous-batching decode
// engine (glc_serve/bidec.py).  Every kernel computes each row with EXACTLY the instruction
// sequence fastdec_kernels.cu uses for that row (copied, only the indexing changed), and a
// row reads and writes only its own sequence's state:
//
//   row r      -> sequence slot row_slot[r], absolute position row_pos[r]
//   sequence b -> slot seq_slot[b], rows seq_row0[b] .. seq_row0[b] + seq_len[b] - 1
//                 (consecutive positions; the GDN conv + recurrence run them sequentially)
//   KV cache   -> paged, page = 256 positions = the attention split: the key block of
//                 split s of a row is page pagetab[slot][s], so the fixed split schedule of
//                 the single-stream engine is kept bit for bit
//   GDN state  -> per slot, an R-deep ring indexed by absolute position (as fastdec)
//
// So a row's bits depend on its own sequence only: batch-invariant by construction (and
// gated end to end by bi_gate_engine.py).  All metadata lives in device int32 arrays the host
// refreshes before each CUDA-graph replay; counts (n_seq, n_items) are device scalars and
// grids are sized for the capacity with early exit / grid-stride loops.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float b2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f2b(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float rb(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

template <int NT>
__device__ __forceinline__ float block_sum(float v, float* sm) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sm[w] = v;
  __syncthreads();
  float t = sm[0];
#pragma unroll
  for (int i = 1; i < NT / 32; ++i) t = __fadd_rn(t, sm[i]);
  return t;
}

constexpr int PAGE = 256;   // == attention SPLIT of fastdec

// ---------------------------------------------------------------- GDN causal conv (k=4) + silu
// hist: [slots][16][C] ring of PRE-conv inputs by absolute position & 15.  grid (C/256, maxseq)
__global__ void b_gdn_conv_kernel(const __nv_bfloat16* __restrict__ in, int64_t ldin,
                                  __nv_bfloat16* __restrict__ hist, const __nv_bfloat16* __restrict__ cw,
                                  const int32_t* __restrict__ seq_slot, const int32_t* __restrict__ seq_row0,
                                  const int32_t* __restrict__ seq_len, const int32_t* __restrict__ n_seq,
                                  const int32_t* __restrict__ row_pos, __nv_bfloat16* __restrict__ out, int C) {
  const int b = blockIdx.y;
  if (b >= *n_seq) return;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int r0 = seq_row0[b], M = seq_len[b];
  const int pos = row_pos[r0];
  __nv_bfloat16* hs = hist + (int64_t)seq_slot[b] * 16 * C;
  const float w0 = b2f(cw[c * 4 + 0]), w1 = b2f(cw[c * 4 + 1]);
  const float w2 = b2f(cw[c * 4 + 2]), w3 = b2f(cw[c * 4 + 3]);
  for (int t = 0; t < M; ++t) {
    const int p = pos + t;
    const __nv_bfloat16 xb = in[(int64_t)(r0 + t) * ldin + c];
    hs[(int64_t)(p & 15) * C + c] = xb;
    float acc = 0.f;
    acc = __fmaf_rn(w0, b2f(hs[(int64_t)((p - 3) & 15) * C + c]), acc);
    acc = __fmaf_rn(w1, b2f(hs[(int64_t)((p - 2) & 15) * C + c]), acc);
    acc = __fmaf_rn(w2, b2f(hs[(int64_t)((p - 1) & 15) * C + c]), acc);
    acc = __fmaf_rn(w3, b2f(xb), acc);
    acc = __fdiv_rn(acc, __fadd_rn(1.0f, expf(-acc)));
    out[(int64_t)(r0 + t) * C + c] = f2b(acc);
  }
}

// ---------------------------------------------------------------- GDN recurrence + gated RMSNorm
// grid (48, maxseq), 512 threads.  state: [slots][R][48][128][128] fp32.  Verbatim fastdec
// gdn_recur_kernel per sequence.
constexpr int GDN_NT = 512;
__global__ void __launch_bounds__(GDN_NT)
b_gdn_recur_kernel(const __nv_bfloat16* __restrict__ conv, const __nv_bfloat16* __restrict__ zba,
                   int64_t ldz, int z_off, int b_off, int a_off,
                   const __nv_bfloat16* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
                   const __nv_bfloat16* __restrict__ nw, float* __restrict__ state_all, int R,
                   const int32_t* __restrict__ seq_slot, const int32_t* __restrict__ seq_row0,
                   const int32_t* __restrict__ seq_len, const int32_t* __restrict__ n_seq,
                   const int32_t* __restrict__ row_pos, __nv_bfloat16* __restrict__ out,
                   float eps, int nkh_rep) {
  constexpr int DK = 128, DV = 128, HK = 16, HV = 48;
  constexpr int CONVD = 2 * HK * DK + HV * DV;           // 10240
  const int bq = blockIdx.y;
  if (bq >= *n_seq) return;
  __shared__ float qs[DK], ks[DK], os[DV], scal[4], sm[GDN_NT / 32];
  const int h = blockIdx.x, tid = threadIdx.x, j = tid >> 2, part = tid & 3;
  const int kh = h / nkh_rep;
  const int r0 = seq_row0[bq], M = seq_len[bq];
  const int pos = row_pos[r0];
  const int64_t hs = (int64_t)DK * DV;
  float* state = state_all + (int64_t)seq_slot[bq] * R * HV * hs;
  int slot = ((pos - 1) % R + R) % R;
  const float* sin_ = state + ((int64_t)slot * HV + h) * hs;
  float S[32];
#pragma unroll
  for (int r = 0; r < 32; ++r) S[r] = sin_[(int64_t)(part * 32 + r) * DV + j];
  const float scale = rsqrtf((float)DK);
  __syncthreads();                                       // every read of the old slot is done
  for (int t = 0; t < M; ++t) {
    const int row = r0 + t;
    const __nv_bfloat16* cv = conv + (int64_t)row * CONVD;
    if (tid < DK) qs[tid] = b2f(cv[kh * DK + tid]);
    else if (tid < 2 * DK) ks[tid - DK] = b2f(cv[HK * DK + kh * DK + tid - DK]);
    __syncthreads();
    const int w = tid >> 5, l = tid & 31;
    if (w < 2) {
      const float* src = w == 0 ? qs : ks;
      float s = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float v = src[l * 4 + e]; s = __fmaf_rn(v, v, s); }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
      if (l == 0) scal[w] = s;
    } else if (w == 2 && l == 0) {
      const float bb = b2f(zba[row * ldz + b_off + h]);
      scal[2] = rb(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-bb))));
      const float ab = __fadd_rn(b2f(zba[row * ldz + a_off + h]), b2f(dt_bias[h]));
      const float sp = ab > 20.f ? ab : log1pf(expf(ab));
      scal[3] = __fmul_rn(-expf(b2f(A_log[h])), sp);
    }
    __syncthreads();
    if (tid < DK) qs[tid] = __fmul_rn(__fdiv_rn(qs[tid], sqrtf(__fadd_rn(scal[0], 1e-6f))), scale);
    else if (tid < 2 * DK) ks[tid - DK] = __fdiv_rn(ks[tid - DK], sqrtf(__fadd_rn(scal[1], 1e-6f)));
    __syncthreads();
    const float beta = scal[2], decay = expf(scal[3]);
    float kv = 0.f;
#pragma unroll
    for (int r = 0; r < 32; ++r) {
      S[r] = __fmul_rn(S[r], decay);
      kv = __fmaf_rn(S[r], ks[part * 32 + r], kv);
    }
    kv = __fadd_rn(kv, __shfl_xor_sync(0xffffffffu, kv, 1));
    kv = __fadd_rn(kv, __shfl_xor_sync(0xffffffffu, kv, 2));
    const float vj = b2f(cv[2 * HK * DK + h * DV + j]);
    const float dl = __fmul_rn(__fsub_rn(vj, kv), beta);
    float o = 0.f;
#pragma unroll
    for (int r = 0; r < 32; ++r) {
      S[r] = __fmaf_rn(ks[part * 32 + r], dl, S[r]);
      o = __fmaf_rn(S[r], qs[part * 32 + r], o);
    }
    o = __fadd_rn(o, __shfl_xor_sync(0xffffffffu, o, 1));
    o = __fadd_rn(o, __shfl_xor_sync(0xffffffffu, o, 2));
    if (t >= M - R) {                    // only the last R states of a sequence can ever be read
      slot = ((pos + t) % R + R) % R;
      float* sout = state + ((int64_t)slot * HV + h) * hs;
#pragma unroll
      for (int r = 0; r < 32; ++r) sout[(int64_t)(part * 32 + r) * DV + j] = S[r];
    }
    const float ob = rb(o);
    if (part == 0) os[j] = ob;
    __syncthreads();
    if (w == 0) {
      float s = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float v = os[l * 4 + e]; s = __fmaf_rn(v, v, s); }
#pragma unroll
      for (int of = 16; of > 0; of >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, of));
      if (l == 0) scal[0] = s;
    }
    __syncthreads();
    if (part == 0) {
      const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(scal[0], (float)DV), eps));
      const float z = b2f(zba[row * ldz + z_off + h * DV + j]);
      const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-z)));
      float y = __fmul_rn(__fmul_rn(ob, rstd), b2f(nw[j]));
      y = __fmul_rn(__fmul_rn(y, z), sg);
      out[(int64_t)row * (HV * DV) + h * DV + j] = f2b(y);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------- attention: q/k norm + RoPE + paged KV append
// grid (maxrows, NQ + 2*NKV), 256 threads.  kc/vc: [pages][256][NKV][256]
__global__ void __launch_bounds__(256)
b_attn_prep_kernel(const __nv_bfloat16* __restrict__ qkv, int64_t ldq,
                   const __nv_bfloat16* __restrict__ qnw, const __nv_bfloat16* __restrict__ knw,
                   const __nv_bfloat16* __restrict__ cosT, const __nv_bfloat16* __restrict__ sinT,
                   int rope_dim, const int32_t* __restrict__ row_slot, const int32_t* __restrict__ row_pos,
                   const int32_t* __restrict__ n_rows, const int32_t* __restrict__ ropeoff,
                   const int32_t* __restrict__ pagetab, int maxpages,
                   __nv_bfloat16* __restrict__ qout, __nv_bfloat16* __restrict__ kc,
                   __nv_bfloat16* __restrict__ vc, int NQ, int NKV, float eps) {
  constexpr int D = 256;
  const int t = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
  if (t >= *n_rows) return;
  __shared__ float ys[D], sm[8];
  const int slot = row_slot[t];
  const int p = row_pos[t];
  const int64_t kvrow = (int64_t)pagetab[slot * maxpages + p / PAGE] * PAGE + (p % PAGE);
  const __nv_bfloat16* row = qkv + t * ldq;
  if (b >= NQ + NKV) {
    const int vh = b - NQ - NKV;
    vc[(kvrow * NKV + vh) * D + d] = row[NQ * 2 * D + NKV * D + vh * D + d];
    return;
  }
  const bool isq = b < NQ;
  const float x = b2f(isq ? row[b * 2 * D + d] : row[NQ * 2 * D + (b - NQ) * D + d]);
  const float ss = block_sum<256>(__fmul_rn(x, x), sm);
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(ss, (float)D), eps));
  const float wf = b2f(isq ? qnw[d] : knw[d]);
  const float y = rb(__fmul_rn(__fmul_rn(x, r), __fadd_rn(1.0f, wf)));
  ys[d] = y;
  __syncthreads();
  float o = y;
  if (d < rope_dim) {
    const int ri = p + ropeoff[slot];
    const float c = b2f(cosT[(int64_t)ri * rope_dim + d]), s = b2f(sinT[(int64_t)ri * rope_dim + d]);
    const int hf = rope_dim / 2;
    const float rot = d < hf ? -ys[d + hf] : ys[d - hf];
    o = rb(__fadd_rn(rb(__fmul_rn(y, c)), rb(__fmul_rn(rot, s))));
  }
  if (isq) qout[((int64_t)t * NQ + b) * D + d] = f2b(o);
  else kc[(kvrow * NKV + (b - NQ)) * D + d] = f2b(o);
}

// ---------------------------------------------------------------- attention: split partials (work list)
// item i -> (row item_row[i], split item_split[i]).  grid (G, NQ) grid-stride over items.
// Per (item, head) the arithmetic is fastdec attn_split_kernel's, verbatim.
__global__ void __launch_bounds__(256)
b_attn_split_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                    const __nv_bfloat16* __restrict__ vc, const int32_t* __restrict__ row_slot,
                    const int32_t* __restrict__ row_pos, const int32_t* __restrict__ item_row,
                    const int32_t* __restrict__ item_split, const int32_t* __restrict__ n_items,
                    const int32_t* __restrict__ pagetab, int maxpages,
                    float* __restrict__ pacc, float* __restrict__ pml, int NQ, int NKV, float scaling) {
  constexpr int D = 256;
  __shared__ float sc[PAGE], sm[8];
  const int hq = blockIdx.y;
  const int kh = hq / (NQ / NKV);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  const int ni = *n_items;
  for (int it = blockIdx.x; it < ni; it += gridDim.x) {
    const int t = item_row[it], s = item_split[it];
    const int slot = row_slot[t];
    const int nk = row_pos[t] + 1;
    const int start = s * PAGE;
    const int n = min(PAGE, nk - start);
    const int64_t kbase = (int64_t)pagetab[slot * maxpages + s] * PAGE;
    float qv[8];
    {
      const uint4 u = *reinterpret_cast<const uint4*>(q + ((int64_t)t * NQ + hq) * D + l * 8);
      const uint32_t a[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        qv[2 * e] = __uint_as_float(a[e] << 16);
        qv[2 * e + 1] = __uint_as_float(a[e] & 0xffff0000u);
      }
    }
    for (int i = w; i < n; i += 8) {
      const uint4 u = *reinterpret_cast<const uint4*>(kc + ((kbase + i) * NKV + kh) * D + l * 8);
      const uint32_t a[4] = {u.x, u.y, u.z, u.w};
      float dot = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        dot = __fmaf_rn(qv[2 * e], __uint_as_float(a[e] << 16), dot);
        dot = __fmaf_rn(qv[2 * e + 1], __uint_as_float(a[e] & 0xffff0000u), dot);
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, o));
      if (l == 0) sc[i] = __fmul_rn(dot, scaling);
    }
    __syncthreads();
    const int d = threadIdx.x;
    float mv = d < n ? sc[d] : -INFINITY;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, o));
    if (l == 0) sm[w] = mv;
    __syncthreads();
    float m = sm[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) m = fmaxf(m, sm[i]);
    __syncthreads();
    const float pv = d < n ? expf(__fsub_rn(sc[d], m)) : 0.f;
    const float lsum = block_sum<256>(pv, sm);
    sc[d] = pv;
    __syncthreads();
    float acc = 0.f;
    const __nv_bfloat16* vp = vc + (kbase * NKV + kh) * D + d;
    for (int i = 0; i < n; ++i) acc = __fmaf_rn(sc[i], b2f(vp[(int64_t)i * NKV * D]), acc);
    const int64_t base = (int64_t)it * NQ + hq;
    pacc[base * D + d] = acc;
    if (d == 0) { pml[base * 2] = m; pml[base * 2 + 1] = lsum; }
    __syncthreads();                               // sc/sm reuse by the next item
  }
}

// combine a row's splits in split order (its items are row_item0[t] .. + ns), * sigmoid(gate).
// grid (maxrows, NQ), 256 threads.  Verbatim fastdec attn_combine_kernel arithmetic.
__global__ void __launch_bounds__(256)
b_attn_combine_kernel(const float* __restrict__ pacc, const float* __restrict__ pml,
                      const __nv_bfloat16* __restrict__ qkv, int64_t ldq,
                      const int32_t* __restrict__ row_pos, const int32_t* __restrict__ row_item0,
                      const int32_t* __restrict__ n_rows, __nv_bfloat16* __restrict__ out, int NQ) {
  constexpr int D = 256;
  const int t = blockIdx.x, hq = blockIdx.y, d = threadIdx.x;
  if (t >= *n_rows) return;
  const int nk = row_pos[t] + 1;
  const int ns = (nk + PAGE - 1) / PAGE;
  const int64_t i0 = row_item0[t];
  float mx = pml[(i0 * NQ + hq) * 2];
  for (int s = 1; s < ns; ++s) mx = fmaxf(mx, pml[((i0 + s) * NQ + hq) * 2]);
  float den = 0.f, num = 0.f;
  for (int s = 0; s < ns; ++s) {
    const int64_t b = (i0 + s) * NQ + hq;
    const float e = expf(__fsub_rn(pml[b * 2], mx));
    den = __fmaf_rn(e, pml[b * 2 + 1], den);
    num = __fmaf_rn(e, pacc[b * D + d], num);
  }
  const float ob = rb(__fdiv_rn(num, den));
  const float g = b2f(qkv[t * ldq + hq * 2 * D + D + d]);
  const float sg = rb(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g))));
  out[(int64_t)t * NQ * D + hq * D + d] = f2b(__fmul_rn(ob, sg));
}

// KV import from a prefill cache: rows [0, n) of k [n][NKV][256] into this slot's pages.
__global__ void b_kv_import_kernel(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                                   const int32_t* __restrict__ pages, int n, int NKV,
                                   __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc) {
  const int p = blockIdx.x;
  if (p >= n) return;
  const int64_t dst = (int64_t)pages[p / PAGE] * PAGE + (p % PAGE);
  const int E = NKV * 256;
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    kc[dst * E + i] = k[(int64_t)p * E + i];
    vc[dst * E + i] = v[(int64_t)p * E + i];
  }
}

// dst[idx[r]] = src[r] for rows r < n_rows with flag[r] != 0 (the MTP hidden store).  grid (maxrows)
__global__ void b_scatter_rows_kernel(const __nv_bfloat16* __restrict__ src, const int32_t* __restrict__ flag,
                                      const int32_t* __restrict__ idx, const int32_t* __restrict__ n_rows,
                                      __nv_bfloat16* __restrict__ dst, int H) {
  const int r = blockIdx.x;
  if (r >= *n_rows || flag[r] == 0) return;
  const int64_t d = idx[r];
  for (int i = threadIdx.x; i < H; i += blockDim.x) dst[d * H + i] = src[(int64_t)r * H + i];
}

inline cudaStream_t cs() { return at::cuda::getCurrentCUDAStream(); }
template <typename T> T* P(const at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

}  // namespace

void b_gdn_conv(at::Tensor in, at::Tensor hist, at::Tensor cw, at::Tensor seq_slot, at::Tensor seq_row0,
                at::Tensor seq_len, at::Tensor n_seq, at::Tensor row_pos, at::Tensor out, int64_t maxseq) {
  const int C = (int)cw.size(0);
  dim3 grid((C + 255) / 256, (int)maxseq);
  b_gdn_conv_kernel<<<grid, 256, 0, cs()>>>(P<__nv_bfloat16>(in), in.stride(0), P<__nv_bfloat16>(hist),
      P<__nv_bfloat16>(cw), P<int32_t>(seq_slot), P<int32_t>(seq_row0), P<int32_t>(seq_len), P<int32_t>(n_seq),
      P<int32_t>(row_pos), P<__nv_bfloat16>(out), C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void b_gdn_recur(at::Tensor conv, at::Tensor zba, int64_t z_off, int64_t b_off, int64_t a_off, at::Tensor A_log,
                 at::Tensor dt_bias, at::Tensor nw, at::Tensor state, int64_t R, at::Tensor seq_slot,
                 at::Tensor seq_row0, at::Tensor seq_len, at::Tensor n_seq, at::Tensor row_pos, at::Tensor out,
                 int64_t maxseq, double eps, int64_t nkh_rep) {
  dim3 grid(48, (int)maxseq);
  b_gdn_recur_kernel<<<grid, GDN_NT, 0, cs()>>>(P<__nv_bfloat16>(conv), P<__nv_bfloat16>(zba), zba.stride(0),
      (int)z_off, (int)b_off, (int)a_off, P<__nv_bfloat16>(A_log), P<__nv_bfloat16>(dt_bias),
      P<__nv_bfloat16>(nw), P<float>(state), (int)R, P<int32_t>(seq_slot), P<int32_t>(seq_row0),
      P<int32_t>(seq_len), P<int32_t>(n_seq), P<int32_t>(row_pos), P<__nv_bfloat16>(out), (float)eps,
      (int)nkh_rep);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void b_attn_prep(at::Tensor qkv, at::Tensor qnw, at::Tensor knw, at::Tensor cosT, at::Tensor sinT,
                 at::Tensor row_slot, at::Tensor row_pos, at::Tensor n_rows, at::Tensor ropeoff, at::Tensor pagetab,
                 int64_t maxpages, at::Tensor qout, at::Tensor kc, at::Tensor vc, int64_t maxrows, int64_t NQ,
                 int64_t NKV, double eps) {
  dim3 grid((int)maxrows, (int)(NQ + 2 * NKV));
  b_attn_prep_kernel<<<grid, 256, 0, cs()>>>(P<__nv_bfloat16>(qkv), qkv.stride(0), P<__nv_bfloat16>(qnw),
      P<__nv_bfloat16>(knw), P<__nv_bfloat16>(cosT), P<__nv_bfloat16>(sinT), (int)cosT.size(1),
      P<int32_t>(row_slot), P<int32_t>(row_pos), P<int32_t>(n_rows), P<int32_t>(ropeoff), P<int32_t>(pagetab),
      (int)maxpages, P<__nv_bfloat16>(qout), P<__nv_bfloat16>(kc), P<__nv_bfloat16>(vc), (int)NQ, (int)NKV,
      (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void b_attn_split(at::Tensor q, at::Tensor kc, at::Tensor vc, at::Tensor row_slot, at::Tensor row_pos,
                  at::Tensor item_row, at::Tensor item_split, at::Tensor n_items, at::Tensor pagetab,
                  int64_t maxpages, at::Tensor pacc, at::Tensor pml, int64_t grid_items, int64_t NQ, int64_t NKV,
                  double scaling) {
  dim3 grid((int)grid_items, (int)NQ);
  b_attn_split_kernel<<<grid, 256, 0, cs()>>>(P<__nv_bfloat16>(q), P<__nv_bfloat16>(kc), P<__nv_bfloat16>(vc),
      P<int32_t>(row_slot), P<int32_t>(row_pos), P<int32_t>(item_row), P<int32_t>(item_split),
      P<int32_t>(n_items), P<int32_t>(pagetab), (int)maxpages, P<float>(pacc), P<float>(pml), (int)NQ,
      (int)NKV, (float)scaling);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void b_attn_combine(at::Tensor pacc, at::Tensor pml, at::Tensor qkv, at::Tensor row_pos, at::Tensor row_item0,
                    at::Tensor n_rows, at::Tensor out, int64_t maxrows, int64_t NQ) {
  dim3 grid((int)maxrows, (int)NQ);
  b_attn_combine_kernel<<<grid, 256, 0, cs()>>>(P<float>(pacc), P<float>(pml), P<__nv_bfloat16>(qkv),
      qkv.stride(0), P<int32_t>(row_pos), P<int32_t>(row_item0), P<int32_t>(n_rows), P<__nv_bfloat16>(out),
      (int)NQ);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void b_kv_import(at::Tensor k, at::Tensor v, at::Tensor pages, int64_t n, at::Tensor kc, at::Tensor vc) {
  const int NKV = (int)k.size(1);
  b_kv_import_kernel<<<(int)n, 256, 0, cs()>>>(P<__nv_bfloat16>(k), P<__nv_bfloat16>(v), P<int32_t>(pages),
      (int)n, NKV, P<__nv_bfloat16>(kc), P<__nv_bfloat16>(vc));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void b_scatter_rows(at::Tensor src, at::Tensor flag, at::Tensor idx, at::Tensor n_rows, at::Tensor dst, int64_t maxrows) {
  const int H = (int)src.size(1);
  b_scatter_rows_kernel<<<(int)maxrows, 256, 0, cs()>>>(P<__nv_bfloat16>(src), P<int32_t>(flag), P<int32_t>(idx),
      P<int32_t>(n_rows), P<__nv_bfloat16>(dst), H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
