#pragma once
#include <stdint.h>

struct TpfPredictedCursor { uint32_t ow, ob; };

// Preconditions: monotone counts<=128, ob<32, and the valid stream's next
// absolute word offset fits uint32_t, exactly as the original carried cursor.
__host__ __device__ __forceinline__ TpfPredictedCursor tpf_predict_cursor(
    uint32_t ow, uint32_t ob, uint32_t header) {
  const uint32_t n1=header&255u, n2=(header>>8)&255u;
  const uint32_t n3=(header>>16)&255u, n4=header>>24;
  const uint32_t bits=ob+2u*n1+2u*n2+3u*n3+8u*n4;
  return {ow+(bits>>5), bits&31u};
}
