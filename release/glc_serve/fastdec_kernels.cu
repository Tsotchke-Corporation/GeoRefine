// fastdec_kernels.cu -- the small fused kernels of the static-buffer decode
// engine (glc_serve/fastdec.py) for the Qwen3.5/3.8 hybrid (Gated-DeltaNet +
// gated full attention).  Every kernel is M-INVARIANT: row t of an M-row call
// runs exactly the instruction sequence of the M=1 call for that row (fixed
// per-thread orders, fixed butterflies, no M-dependent split), and the
// Gated-DeltaNet recurrence runs t = 0..M-1 SEQUENTIALLY with the FP32 op
// order of the single-step update.  All position-dependent state is indexed by
// ABSOLUTE position (KV rows, conv-input ring, recurrent-state ring), so a
// speculative rollback is "set the position": nothing is copied back.
//
// Positions come from device ints (so one captured CUDA graph serves every
// step): kvpos = cache row of row 0, ropeoff = (rope position - cache row).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float b2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f2b(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float rb(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// fixed-order block sum: per-warp xor butterfly, then warp partials summed in
// warp order by every thread (same bits everywhere).
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

// ---------------------------------------------------------------- embedding
__global__ void embed_kernel(const int32_t* __restrict__ ids, const __nv_bfloat16* __restrict__ tab,
                             __nv_bfloat16* __restrict__ out, int H) {
  const int t = blockIdx.x;
  const int64_t id = ids[t];
  const uint4* src = reinterpret_cast<const uint4*>(tab + id * H);
  uint4* dst = reinterpret_cast<uint4*>(out + (int64_t)t * H);
  for (int i = threadIdx.x; i < H / 8; i += blockDim.x) dst[i] = src[i];
}

// Q8_0 embedding rows: bf16_rn(d * q), SoA (qs int8 [V][H], sc fp16 [V][H/32])
__global__ void embed_q8_kernel(const int32_t* __restrict__ ids, const int8_t* __restrict__ qs,
                                const __half* __restrict__ sc, __nv_bfloat16* __restrict__ out, int H) {
  const int t = blockIdx.x;
  const int64_t id = ids[t];
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    const float d = __half2float(sc[id * (H / 32) + i / 32]);
    out[(int64_t)t * H + i] = __float2bfloat16_rn(__fmul_rn(d, (float)qs[id * H + i]));
  }
}

// ---------------------------------------------------------------- RMSNorm (+ residual add)
// if delta: h = bf16(h + delta) (written back).  out = bf16((h * rsqrt(mean(h^2) + eps)) * (1 + w))
// Qwen3.5 RMSNorm semantics (weight stored as offset from 1).  unit_offset=0 -> * w.
constexpr int RMS_NT = 512;
constexpr int RMS_MAXE = 16;
__global__ void __launch_bounds__(RMS_NT)
rmsnorm_kernel(__nv_bfloat16* __restrict__ h, int64_t ldh, const __nv_bfloat16* __restrict__ delta,
               int64_t ldd, const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
               int64_t ldo, int H, float eps, int unit_offset) {
  __shared__ float sm[RMS_NT / 32];
  const int t = blockIdx.x;
  __nv_bfloat16* hr = h + t * ldh;
  float xv[RMS_MAXE];
  float ss = 0.f;
  int n = 0;
  for (int i = threadIdx.x; i < H; i += RMS_NT, ++n) {
    float x = b2f(hr[i]);
    if (delta) {
      x = rb(__fadd_rn(x, b2f(delta[t * ldd + i])));
      hr[i] = f2b(x);
    }
    xv[n] = x;
    ss = __fmaf_rn(x, x, ss);
  }
  const float tot = block_sum<RMS_NT>(ss, sm);
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)H), eps));
  n = 0;
  for (int i = threadIdx.x; i < H; i += RMS_NT, ++n) {
    const float wf = b2f(w[i]);
    const float sc = unit_offset ? __fadd_rn(1.0f, wf) : wf;
    out[t * ldo + i] = f2b(__fmul_rn(__fmul_rn(xv[n], r), sc));
  }
}

// ---------------------------------------------------------------- GDN causal conv (k=4) + silu
// hist: [16][C] ring of PRE-conv inputs indexed by absolute position & 15.
__global__ void gdn_conv_kernel(const __nv_bfloat16* __restrict__ in, int64_t ldin,
                                __nv_bfloat16* __restrict__ hist, const __nv_bfloat16* __restrict__ cw,
                                const int32_t* __restrict__ pos_p, __nv_bfloat16* __restrict__ out,
                                int C, int M) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int pos = *pos_p;
  const float w0 = b2f(cw[c * 4 + 0]), w1 = b2f(cw[c * 4 + 1]);
  const float w2 = b2f(cw[c * 4 + 2]), w3 = b2f(cw[c * 4 + 3]);
  for (int t = 0; t < M; ++t) {
    const int p = pos + t;
    const __nv_bfloat16 xb = in[t * ldin + c];
    hist[(int64_t)(p & 15) * C + c] = xb;
    float acc = 0.f;
    acc = __fmaf_rn(w0, b2f(hist[(int64_t)((p - 3) & 15) * C + c]), acc);
    acc = __fmaf_rn(w1, b2f(hist[(int64_t)((p - 2) & 15) * C + c]), acc);
    acc = __fmaf_rn(w2, b2f(hist[(int64_t)((p - 1) & 15) * C + c]), acc);
    acc = __fmaf_rn(w3, b2f(xb), acc);
    acc = __fdiv_rn(acc, __fadd_rn(1.0f, expf(-acc)));
    out[t * C + c] = f2b(acc);
  }
}

// ---------------------------------------------------------------- GDN recurrent step(s) + gated RMSNorm
// One CTA per value head h (48), 512 threads: column j = tid >> 2 (value dim),
// part = tid & 3 owns key rows part*32 .. part*32+31 of S[h] (fp32 [128][128]).
// Runs t = 0..M-1 sequentially with S in registers; the state after token at
// absolute position p is written to ring slot p % R, the initial state is read
// from slot (pos-1) % R.  Single-step op order (FLA fused_recurrent, qk l2norm
// in kernel):  q,k /= sqrt(sum sq + 1e-6); q *= 1/sqrt(dk); S *= exp(g);
// kv = S^T k; d = (v - kv) * beta; S += k d^T; o = S^T q.
constexpr int GDN_NT = 512;
__global__ void __launch_bounds__(GDN_NT)
gdn_recur_kernel(const __nv_bfloat16* __restrict__ conv, const __nv_bfloat16* __restrict__ zba,
                 int64_t ldz, int z_off, int b_off, int a_off,
                 const __nv_bfloat16* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
                 const __nv_bfloat16* __restrict__ nw, float* __restrict__ state, int R,
                 const int32_t* __restrict__ pos_p, __nv_bfloat16* __restrict__ out, int M,
                 float eps, int nkh_rep) {
  constexpr int DK = 128, DV = 128, HK = 16, HV = 48;
  constexpr int CONVD = 2 * HK * DK + HV * DV;           // 10240
  __shared__ float qs[DK], ks[DK], os[DV], scal[4], sm[GDN_NT / 32];
  const int h = blockIdx.x, tid = threadIdx.x, j = tid >> 2, part = tid & 3;
  const int kh = h / nkh_rep;
  const int pos = *pos_p;
  const int64_t hs = (int64_t)DK * DV;
  int slot = ((pos - 1) % R + R) % R;
  const float* sin_ = state + ((int64_t)slot * HV + h) * hs;
  float S[32];
#pragma unroll
  for (int r = 0; r < 32; ++r) S[r] = sin_[(int64_t)(part * 32 + r) * DV + j];
  const float scale = rsqrtf((float)DK);
  for (int t = 0; t < M; ++t) {
    const __nv_bfloat16* cv = conv + (int64_t)t * CONVD;
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
      const float bb = b2f(zba[t * ldz + b_off + h]);
      scal[2] = rb(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-bb))));               // bf16 sigmoid
      const float ab = __fadd_rn(b2f(zba[t * ldz + a_off + h]), b2f(dt_bias[h]));
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
    slot = ((pos + t) % R + R) % R;
    float* sout = state + ((int64_t)slot * HV + h) * hs;
#pragma unroll
    for (int r = 0; r < 32; ++r) sout[(int64_t)(part * 32 + r) * DV + j] = S[r];
    const float ob = rb(o);                                   // recurrent output is bf16
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
      const float z = b2f(zba[t * ldz + z_off + h * DV + j]);
      const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-z)));
      float y = __fmul_rn(__fmul_rn(ob, rstd), b2f(nw[j]));
      y = __fmul_rn(__fmul_rn(y, z), sg);
      out[(int64_t)t * (HV * DV) + h * DV + j] = f2b(y);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------- attention: q/k norm + RoPE + KV append
// qkv row: [24 x (q256 | gate256) | k 4x256 | v 4x256].  grid (M, NQ + 2*NKV), 256 threads.
__global__ void __launch_bounds__(256)
attn_prep_kernel(const __nv_bfloat16* __restrict__ qkv, int64_t ldq,
                 const __nv_bfloat16* __restrict__ qnw, const __nv_bfloat16* __restrict__ knw,
                 const __nv_bfloat16* __restrict__ cosT, const __nv_bfloat16* __restrict__ sinT,
                 int rope_dim, const int32_t* __restrict__ kvpos_p, const int32_t* __restrict__ ropeoff_p,
                 __nv_bfloat16* __restrict__ qout, __nv_bfloat16* __restrict__ kc,
                 __nv_bfloat16* __restrict__ vc, int NQ, int NKV, float eps) {
  constexpr int D = 256;
  __shared__ float ys[D], sm[8];
  const int t = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
  const int p = *kvpos_p + t;
  const __nv_bfloat16* row = qkv + t * ldq;
  if (b >= NQ + NKV) {                                        // v head: append
    const int vh = b - NQ - NKV;
    vc[((int64_t)p * NKV + vh) * D + d] = row[NQ * 2 * D + NKV * D + vh * D + d];
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
    const int ri = p + *ropeoff_p;
    const float c = b2f(cosT[(int64_t)ri * rope_dim + d]), s = b2f(sinT[(int64_t)ri * rope_dim + d]);
    const int hf = rope_dim / 2;
    const float rot = d < hf ? -ys[d + hf] : ys[d - hf];
    o = rb(__fadd_rn(rb(__fmul_rn(y, c)), rb(__fmul_rn(rot, s))));
  }
  if (isq) qout[((int64_t)t * NQ + b) * D + d] = f2b(o);
  else kc[((int64_t)p * NKV + (b - NQ)) * D + d] = f2b(o);
}

// ---------------------------------------------------------------- attention: fixed split-size partials
// grid (NS, NQ, M), 256 threads.  Split s covers key rows [s*SPLIT, s*SPLIT+SPLIT) cut at the
// row's causal end p = kvpos + t.  Partials: m, l, acc[256] (fp32).
constexpr int SPLIT = 256;
__global__ void __launch_bounds__(256)
attn_split_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                  const __nv_bfloat16* __restrict__ vc, const int32_t* __restrict__ kvpos_p,
                  float* __restrict__ pacc, float* __restrict__ pml, int NQ, int NKV, int NS,
                  float scaling) {
  constexpr int D = 256;
  __shared__ float sc[SPLIT], sm[8];
  const int s = blockIdx.x, hq = blockIdx.y, t = blockIdx.z;
  const int nk = *kvpos_p + t + 1;
  const int start = s * SPLIT;
  if (start >= nk) return;
  const int n = min(SPLIT, nk - start);
  const int kh = hq / (NQ / NKV);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
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
    const uint4 u = *reinterpret_cast<const uint4*>(kc + ((int64_t)(start + i) * NKV + kh) * D + l * 8);
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
  sc[d] = pv;                    // every read of sc[] (the max) finished before block_sum's barriers
  __syncthreads();
  float acc = 0.f;
  const __nv_bfloat16* vp = vc + ((int64_t)start * NKV + kh) * D + d;
  for (int i = 0; i < n; ++i) acc = __fmaf_rn(sc[i], b2f(vp[(int64_t)i * NKV * D]), acc);
  const int64_t base = ((int64_t)t * NQ + hq) * NS + s;
  pacc[base * D + d] = acc;
  if (d == 0) { pml[base * 2] = m; pml[base * 2 + 1] = lsum; }
}

// combine splits in split order, then * sigmoid(gate) (bf16 ops).  grid (M, NQ), 256 threads.
__global__ void __launch_bounds__(256)
attn_combine_kernel(const float* __restrict__ pacc, const float* __restrict__ pml,
                    const __nv_bfloat16* __restrict__ qkv, int64_t ldq,
                    const int32_t* __restrict__ kvpos_p, __nv_bfloat16* __restrict__ out, int NQ,
                    int NS) {
  constexpr int D = 256;
  const int t = blockIdx.x, hq = blockIdx.y, d = threadIdx.x;
  const int nk = *kvpos_p + t + 1;
  const int ns = (nk + SPLIT - 1) / SPLIT;
  const int64_t base = ((int64_t)t * NQ + hq) * NS;
  float mx = pml[base * 2];
  for (int s = 1; s < ns; ++s) mx = fmaxf(mx, pml[(base + s) * 2]);
  float den = 0.f, num = 0.f;
  for (int s = 0; s < ns; ++s) {
    const float e = expf(__fsub_rn(pml[(base + s) * 2], mx));
    den = __fmaf_rn(e, pml[(base + s) * 2 + 1], den);
    num = __fmaf_rn(e, pacc[(base + s) * D + d], num);
  }
  const float ob = rb(__fdiv_rn(num, den));
  const float g = b2f(qkv[t * ldq + hq * 2 * D + D + d]);
  const float sg = rb(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g))));
  out[(int64_t)t * NQ * D + hq * D + d] = f2b(__fmul_rn(ob, sg));
}

// ---------------------------------------------------------------- SwiGLU: bf16(bf16(silu(g)) * u)
__global__ void silu_mul_kernel(const __nv_bfloat16* __restrict__ gu, int64_t ldg,
                                __nv_bfloat16* __restrict__ out, int F) {
  const int t = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= F) return;
  const float g = b2f(gu[t * ldg + i]), u = b2f(gu[t * ldg + F + i]);
  const float a = rb(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))));
  out[(int64_t)t * F + i] = f2b(__fmul_rn(a, u));
}

// ---------------------------------------------------------------- argmax (first max)
__global__ void __launch_bounds__(1024)
argmax_kernel(const __nv_bfloat16* __restrict__ lg, int64_t ld, int V, int32_t* __restrict__ out) {
  __shared__ float sv[32];
  __shared__ int si[32];
  const int t = blockIdx.x;
  const __nv_bfloat16* r = lg + t * ld;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < V; i += 1024) {
    const float v = b2f(r[i]);
    if (v > bv) { bv = v; bi = i; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) { sv[w] = bv; si[w] = bi; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < 32; ++i)
      if (sv[i] > bv || (sv[i] == bv && si[i] < bi)) { bv = sv[i]; bi = si[i]; }
    out[t] = bi;
  }
}

// ---------------------------------------------------------------- device-side step bookkeeping
// AR: vtok[0] = am[0]; log[pos + 1] = am[0]; pos += 1.
__global__ void ar_advance_kernel(const int32_t* am, int32_t* vtok, int32_t* log, int32_t* pos,
                                  int cap) {
  const int p = *pos;
  vtok[0] = am[0];
  if (p + 1 < cap) log[p + 1] = am[0];
  *pos = p + 1;
}
// MTP draft: vtok[*slot] = am[0]; mtok[0] = am[0]; *slot += 1; mkv += 1
__global__ void draft_advance_kernel(const int32_t* am, int32_t* vtok, int32_t* slot, int32_t* mtok,
                                     int32_t* mkv, int M) {
  const int s = *slot;
  if (s < 8) vtok[s] = am[0];
  mtok[0] = am[0];
  *slot = s + 1;
  *mkv = *mkv + M;
}

inline cudaStream_t cs() { return at::cuda::getCurrentCUDAStream(); }
template <typename T> T* P(const at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
template <typename T> const T* CP(const c10::optional<at::Tensor>& t) {
  return (t.has_value() && t->defined()) ? reinterpret_cast<const T*>(t->data_ptr()) : nullptr;
}

}  // namespace

void embed(at::Tensor ids, at::Tensor tab, at::Tensor out, int64_t M) {
  const int H = (int)tab.size(1);
  TORCH_CHECK(H % 8 == 0 && out.stride(0) == H);
  embed_kernel<<<(int)M, 256, 0, cs()>>>(P<int32_t>(ids), P<__nv_bfloat16>(tab), P<__nv_bfloat16>(out), H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void embed_q8(at::Tensor ids, at::Tensor qs, at::Tensor sc, at::Tensor out, int64_t M) {
  const int H = (int)qs.size(1);
  embed_q8_kernel<<<(int)M, 256, 0, cs()>>>(P<int32_t>(ids), P<int8_t>(qs), P<__half>(sc),
                                            P<__nv_bfloat16>(out), H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rmsnorm(at::Tensor h, c10::optional<at::Tensor> delta, at::Tensor w, at::Tensor out, int64_t M,
             double eps, int64_t unit_offset) {
  const int H = (int)w.numel();
  TORCH_CHECK(H <= RMS_NT * RMS_MAXE);
  const int64_t ldd = (delta.has_value() && delta->defined()) ? delta->stride(0) : 0;
  rmsnorm_kernel<<<(int)M, RMS_NT, 0, cs()>>>(P<__nv_bfloat16>(h), h.stride(0),
      CP<__nv_bfloat16>(delta), ldd, P<__nv_bfloat16>(w), P<__nv_bfloat16>(out), out.stride(0), H,
      (float)eps, (int)unit_offset);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_conv(at::Tensor in, at::Tensor hist, at::Tensor cw, at::Tensor pos, at::Tensor out, int64_t M) {
  const int C = (int)cw.size(0);
  gdn_conv_kernel<<<(C + 255) / 256, 256, 0, cs()>>>(P<__nv_bfloat16>(in), in.stride(0),
      P<__nv_bfloat16>(hist), P<__nv_bfloat16>(cw), P<int32_t>(pos), P<__nv_bfloat16>(out), C, (int)M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_recur(at::Tensor conv, at::Tensor zba, int64_t z_off, int64_t b_off, int64_t a_off,
               at::Tensor A_log, at::Tensor dt_bias, at::Tensor nw, at::Tensor state, int64_t R,
               at::Tensor pos, at::Tensor out, int64_t M, double eps, int64_t nkh_rep) {
  gdn_recur_kernel<<<48, GDN_NT, 0, cs()>>>(P<__nv_bfloat16>(conv), P<__nv_bfloat16>(zba),
      zba.stride(0), (int)z_off, (int)b_off, (int)a_off, P<__nv_bfloat16>(A_log),
      P<__nv_bfloat16>(dt_bias), P<__nv_bfloat16>(nw), P<float>(state), (int)R, P<int32_t>(pos),
      P<__nv_bfloat16>(out), (int)M, (float)eps, (int)nkh_rep);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attn_prep(at::Tensor qkv, at::Tensor qnw, at::Tensor knw, at::Tensor cosT, at::Tensor sinT,
               at::Tensor kvpos, at::Tensor ropeoff, at::Tensor qout, at::Tensor kc, at::Tensor vc,
               int64_t M, int64_t NQ, int64_t NKV, double eps) {
  dim3 grid((int)M, (int)(NQ + 2 * NKV));
  attn_prep_kernel<<<grid, 256, 0, cs()>>>(P<__nv_bfloat16>(qkv), qkv.stride(0), P<__nv_bfloat16>(qnw),
      P<__nv_bfloat16>(knw), P<__nv_bfloat16>(cosT), P<__nv_bfloat16>(sinT), (int)cosT.size(1),
      P<int32_t>(kvpos), P<int32_t>(ropeoff), P<__nv_bfloat16>(qout), P<__nv_bfloat16>(kc),
      P<__nv_bfloat16>(vc), (int)NQ, (int)NKV, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attn_split(at::Tensor q, at::Tensor kc, at::Tensor vc, at::Tensor kvpos, at::Tensor pacc,
                at::Tensor pml, int64_t M, int64_t NQ, int64_t NKV, int64_t NS, double scaling) {
  dim3 grid((int)NS, (int)NQ, (int)M);
  attn_split_kernel<<<grid, 256, 0, cs()>>>(P<__nv_bfloat16>(q), P<__nv_bfloat16>(kc),
      P<__nv_bfloat16>(vc), P<int32_t>(kvpos), P<float>(pacc), P<float>(pml), (int)NQ, (int)NKV,
      (int)NS, (float)scaling);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attn_combine(at::Tensor pacc, at::Tensor pml, at::Tensor qkv, at::Tensor kvpos, at::Tensor out,
                  int64_t M, int64_t NQ, int64_t NS) {
  dim3 grid((int)M, (int)NQ);
  attn_combine_kernel<<<grid, 256, 0, cs()>>>(P<float>(pacc), P<float>(pml), P<__nv_bfloat16>(qkv),
      qkv.stride(0), P<int32_t>(kvpos), P<__nv_bfloat16>(out), (int)NQ, (int)NS);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void silu_mul(at::Tensor gu, at::Tensor out, int64_t M) {
  const int F = (int)out.size(1);
  dim3 grid((F + 255) / 256, (int)M);
  silu_mul_kernel<<<grid, 256, 0, cs()>>>(P<__nv_bfloat16>(gu), gu.stride(0), P<__nv_bfloat16>(out), F);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void argmax_rows(at::Tensor lg, at::Tensor out, int64_t M) {
  argmax_kernel<<<(int)M, 1024, 0, cs()>>>(P<__nv_bfloat16>(lg), lg.stride(0), (int)lg.size(1),
                                            P<int32_t>(out));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ar_advance(at::Tensor am, at::Tensor vtok, at::Tensor log, at::Tensor pos) {
  ar_advance_kernel<<<1, 1, 0, cs()>>>(P<int32_t>(am), P<int32_t>(vtok), P<int32_t>(log),
                                       P<int32_t>(pos), (int)log.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void draft_advance(at::Tensor am, at::Tensor vtok, at::Tensor slot, at::Tensor mtok, at::Tensor mkv,
                   int64_t M) {
  draft_advance_kernel<<<1, 1, 0, cs()>>>(P<int32_t>(am), P<int32_t>(vtok), P<int32_t>(slot),
                                          P<int32_t>(mtok), P<int32_t>(mkv), (int)M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
