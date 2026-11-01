# TBE21 kernel-v2 source provenance

Kernel source origin is `newcodec-gpu-20261018` HEAD `90e83dd8fe83674c77b5ea7ccdce534a6fa28041`.
`bi_gemm_v2.cu`, `bi_gemm.cuh`, `bi_api.h`, `miv_tbe_kernels.cuh`, and `miv_tbe.h` derive from that revision. Their original SHA256 values are embedded in `glc_serve.tbe21_v2_backend._PINNED_SOURCE_SHA256`; current built-source hashes are returned by `source_bindings()` and included in backend identity.

Only these source adaptations were made:

- `bi_api.h`: added `GEOR_REFINED_CONTEXT_ONLY` guards around KQ enum/member fields, matching the public Context ABI convention. The private extension still carries pinned TBE21 fields and does not add a format to public `bigemm.FMT`/`WEIGHT_FORMATS`.
- `bi_gemm.cuh`: added Context-only guards around the KQ include, enum/member, and KQ loader specialization, matching the public Context source convention. TBE21 decoder and GEMM math are unchanged.
- `bi_gemm_v2.cu`: renamed its three exported v2 entrypoints with the `bi_tbe21_v2_` prefix and macro-renames namespace `bi` to `bi_tbe21_v2_private` before including `bi_gemm.cuh` to isolate template/static symbols. Kernel bodies and dispatch math remain unchanged.
- `bi_torch.cpp` and `bi_tbe21_v2_api.h` are a new narrow binding for fmt=5 only. They follow the pinned `fill_tbe21` layout/checks and invoke v2 directly with tile=64/spec=0. Its C++ binding function is uniquely named while the Python-visible method stays `bi_gemm_out` for BILin compatibility.

The source closure intentionally excludes `bi_gemm_dense.cu`, KQ kernels/headers, and the legacy extension binding. Build uses a unique module/library name and `BI_TBE21_V2_BUILD_DIR` cache override.

CUDA 12.9 requires inline-PTX LOP3 lookup tables to be template constants. The helper and its constant call sites use a template immediate; every operand, LUT value, and operation order is retained. Context-only guards also exclude the quantized fields in `to_wdesc`.
