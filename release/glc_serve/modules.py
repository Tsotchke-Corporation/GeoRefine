"""Serving modules: coded linears, a host-resident embedding, layer streaming.

``TBEServeLinear``  an ``nn.Linear`` whose weight is a TBE container uploaded
    AS STORED.  Three execution modes, all reading bit-identical weights:

      ``fused``  M <= 16: the fragment kernel (``tbe_mma_gemm``) consumes the
                 container straight into ``mma.sync`` -- no dense weight ever
                 exists.  16 < M <= ``fused_max_m``: the same kernel over
                 16-row chunks (reads the compressed weight ceil(M/16) times,
                 still fewer bytes than one dense read up to ~M=32).
                 M > ``fused_max_m`` (prefill): decode ONCE into a shared
                 transient buffer, then ``F.linear`` -- cuBLAS on the decoded
                 weight, i.e. exactly the dense computation.
      ``exact``  always decode-then-``F.linear``.  The weight bytes AND the
                 GEMM call are the dense model's, so logits are expected to be
                 bitwise equal to the uncompressed parent's (to be measured, not
                 assumed).  A proof mode: it pays a full decode per call.

    ``prefetch=True`` double-buffers the decode path: while linear ``i``'s
    GEMM runs, linear ``i+1`` (learned online from call order) is decoded on a
    side stream.  A misprediction costs a synchronous decode, never a wrong
    weight -- every buffer is tagged with the linear it holds.

``HostEmbedding``  the embedding table in pinned host RAM; a lookup gathers
    the requested rows on the CPU and copies them to the GPU.  Bit-exact (a
    copy), and the table (2.54 GB for the 27B) never occupies VRAM.

``LayerStreamer``  keeps whole decoder layers' containers in pinned host
    memory and streams each onto one of two GPU slots just before the layer
    runs, prefetching the next offloaded layer on a copy stream.  The bytes
    the kernel reads are the stored bytes; only their residence changes.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from glc_loader.tbe_container import TBETensor
from glc_loader.tbe_mma import (
    ESC_PAD,
    MAX_M,
    TBEDevice,
    exponent_table,
    tbe_mma_decode,
    tbe_mma_gemm,
    upload_tbe,
)

EXEC_MODES = ("fused", "exact")
DEFAULT_FUSED_MAX_M = 48
_UID = itertools.count()


class ServeError(RuntimeError):
    """A serving module was asked to do something it cannot prove."""


# The GEMM used after a full decode (``exact`` mode and the fused path's
# decode-then-GEMM branch).  ``None`` means ``F.linear`` -- the default, and
# byte-identical to the behaviour before this hook existed.  Set it only to
# an ``F.linear``-compatible callable (``glc_serve.miv_gemv.linear``).
_EXACT_LINEAR = None


def set_exact_linear(fn) -> None:
    global _EXACT_LINEAR
    _EXACT_LINEAR = fn


def _exact_linear(x, w, bias):
    fn = _EXACT_LINEAR
    return F.linear(x, w, bias) if fn is None else fn(x, w, bias)


# ---------------------------------------------------------------------------
# host-side container arrays (mirror of glc_loader.tbe_mma.upload_tbe)
# ---------------------------------------------------------------------------
def _int32_words(values: torch.Tensor) -> torch.Tensor:
    v = values.reshape(-1).to(torch.int64)
    v = torch.where(v >= 0x80000000, v - 0x100000000, v)
    return v.to(torch.int32).contiguous()


def host_arrays(c: TBETensor) -> Dict[str, torch.Tensor]:
    """The four arrays exactly as ``upload_tbe`` lays them out, on the CPU."""
    return {
        "planes": _int32_words(c.planes),
        "smb": c.smb.reshape(-1).contiguous().to(torch.uint8),
        "esc": torch.cat([
            c.esc.reshape(-1).contiguous().to(torch.uint8),
            torch.zeros(ESC_PAD, dtype=torch.uint8),
        ]),
        "sbbase": _int32_words(c.sbbase),
    }


def device_from_arrays(arrays: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> TBEDevice:
    e01, e23 = exponent_table(int(meta["mode"]), int(meta["base"]))
    return TBEDevice(
        planes=arrays["planes"], smb=arrays["smb"], esc=arrays["esc"],
        sbbase=arrays["sbbase"], shape=tuple(meta["shape"]), mode=int(meta["mode"]),
        base=int(meta["base"]), superblock=int(meta["superblock"]),
        tiles=int(meta["tiles"]), escapes=int(meta["escapes"]),
        e01=e01, e23=e23, esc_pad=ESC_PAD,
    )


# ---------------------------------------------------------------------------
# decode buffers
# ---------------------------------------------------------------------------
class DecodePool:
    """Per-device bf16 decode buffers for the decode-then-GEMM path.

    ``slots=1``: one buffer, synchronous decode.  ``slots=2``: double buffer
    with a side-stream prefetch of the predicted next linear.  Weights larger
    than ``capacity_numel`` (a coded ``lm_head`` in ``exact`` mode) use a
    separate lazily allocated buffer and are never prefetched.
    """

    def __init__(self, device: torch.device, capacity_numel: int, *, slots: int = 1):
        self.device = torch.device(device)
        self.capacity = int(capacity_numel)
        self.slots = 2 if int(slots) >= 2 else 1
        self._bufs = [
            torch.empty(self.capacity, dtype=torch.bfloat16, device=self.device)
            for _ in range(self.slots)
        ] if self.capacity > 0 else []
        self._tags: List[Optional[int]] = [None] * self.slots
        self._ready: List[Any] = [None] * self.slots
        self._free: List[Any] = [None] * self.slots
        self._side = (
            torch.cuda.Stream(self.device)
            if self.slots == 2 and self.device.type == "cuda" else None
        )
        self._big: Optional[torch.Tensor] = None
        self._rr = 0
        self._last_uid: Optional[int] = None
        self.successor: Dict[int, int] = {}
        self.modules: Dict[int, "TBEServeLinear"] = {}
        self.stats = {"sync_decodes": 0, "prefetch_hits": 0, "prefetch_issued": 0,
                      "big_decodes": 0}

    @property
    def capacity_bytes(self) -> int:
        return self.capacity * 2 * len(self._bufs) + (
            int(self._big.numel()) * 2 if self._big is not None else 0)

    def register(self, module: "TBEServeLinear") -> None:
        self.modules[module.uid] = module

    def _view(self, slot: int, n: int, k: int) -> torch.Tensor:
        return self._bufs[slot][: n * k].view(n, k)

    def acquire(self, module: "TBEServeLinear") -> Tuple[torch.Tensor, int]:
        n, k = module.out_features, module.in_features
        if self._last_uid is not None and self._last_uid != module.uid:
            self.successor[self._last_uid] = module.uid
        self._last_uid = module.uid
        if n * k > self.capacity:
            if self._big is None or self._big.numel() < n * k:
                self._big = torch.empty(n * k, dtype=torch.bfloat16, device=self.device)
            w = self._big[: n * k].view(n, k)
            tbe_mma_decode(module.device_container(), out=w)
            self.stats["big_decodes"] += 1
            return w, -1
        for s in range(self.slots):
            if self._tags[s] == module.uid and self._ready[s] is not None:
                torch.cuda.current_stream(self.device).wait_event(self._ready[s])
                self._ready[s] = None
                self.stats["prefetch_hits"] += 1
                return self._view(s, n, k), s
        s = self._rr
        self._rr = (self._rr + 1) % self.slots
        w = self._view(s, n, k)
        tbe_mma_decode(module.device_container(), out=w)
        self._tags[s] = module.uid
        self.stats["sync_decodes"] += 1
        return w, s

    def release(self, slot: int) -> None:
        if slot < 0 or self.device.type != "cuda":
            return
        self._free[slot] = torch.cuda.current_stream(self.device).record_event()

    def prefetch_after(self, module: "TBEServeLinear", slot: int) -> None:
        if self._side is None or slot < 0:
            return
        nxt = self.successor.get(module.uid)
        target = self.modules.get(nxt) if nxt is not None else None
        if target is None or target._dev is None:
            return
        n, k = target.out_features, target.in_features
        if n * k > self.capacity:
            return
        other = 1 - slot
        if self._free[other] is not None:
            self._side.wait_event(self._free[other])
        # the side stream must also see everything issued so far on the
        # compute stream that could touch ``other`` (its last consumer)
        self._side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._side):
            tbe_mma_decode(target._dev, out=self._view(other, n, k))
            self._ready[other] = self._side.record_event()
        self._tags[other] = target.uid
        self.stats["prefetch_issued"] += 1


class PoolRegistry:
    def __init__(self):
        self.pools: Dict[str, DecodePool] = {}

    def get(self, device, capacity_numel: int, slots: int) -> DecodePool:
        key = str(torch.device(device))
        pool = self.pools.get(key)
        if pool is None:
            pool = DecodePool(device, capacity_numel, slots=slots)
            self.pools[key] = pool
        return pool

    @property
    def capacity_bytes(self) -> int:
        return sum(p.capacity_bytes for p in self.pools.values())

    def stats(self) -> Dict[str, Any]:
        return {k: dict(p.stats, capacity_bytes=p.capacity_bytes) for k, p in self.pools.items()}


# ---------------------------------------------------------------------------
# the coded linear
# ---------------------------------------------------------------------------
class TBEServeLinear(nn.Module):
    """A TBE-coded linear built from the STORED container (no re-encode)."""

    def __init__(
        self,
        container: TBETensor,
        bias: Optional[torch.Tensor],
        device: torch.device,
        *,
        pool: Optional[DecodePool],
        name: str = "",
        exec_mode: str = "fused",
        fused_max_m: int = DEFAULT_FUSED_MAX_M,
        offload: bool = False,
    ) -> None:
        super().__init__()
        if exec_mode not in EXEC_MODES:
            raise ServeError(f"exec_mode must be one of {EXEC_MODES}, got {exec_mode!r}")
        dev = torch.device(device)
        if dev.type != "cuda":
            raise ServeError(
                f"{name}: TBE fragment kernels are CUDA-only (got {dev}); there is "
                "no silent CPU fallback"
            )
        self.uid = next(_UID)
        self.name = name
        self.exec_mode = exec_mode
        self.fused_max_m = int(fused_max_m)
        self.out_features, self.in_features = int(container.shape[0]), int(container.shape[1])
        self.target_device = dev
        self.meta = {
            "shape": (self.out_features, self.in_features), "mode": int(container.mode),
            "base": int(container.base), "superblock": int(container.superblock),
            "tiles": int(container.tiles), "escapes": int(container.escapes),
        }
        self._stored_bytes = int(container.byte_size()["total"])
        self.offloaded = bool(offload)
        self._host: Optional[Dict[str, torch.Tensor]] = None
        self._dev: Optional[TBEDevice] = None
        if self.offloaded:
            self._host = host_arrays(container)
        else:
            self._dev = upload_tbe(container, dev)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().to(dev), requires_grad=False)
        self.pool = pool
        if pool is not None:
            pool.register(self)

    # -- accounting -------------------------------------------------------
    @property
    def resident_bytes(self) -> int:
        """Bytes this module keeps on the GPU permanently (0 when offloaded)."""
        return 0 if self.offloaded else int(self._dev.resident_bytes)

    @property
    def stored_bytes(self) -> int:
        return self._stored_bytes

    @property
    def dense_bytes(self) -> int:
        return self.out_features * self.in_features * 2

    def device_container(self) -> TBEDevice:
        if self._dev is None:
            raise ServeError(
                f"{self.name}: container is host-resident and not bound; the layer "
                "streamer binds it for the duration of its layer's forward"
            )
        return self._dev

    def temporary_device_container(self) -> TBEDevice:
        """A GPU copy for one-off use (the startup gate on an offloaded layer)."""
        if self._dev is not None:
            return self._dev
        arrays = {k: v.to(self.target_device) for k, v in self._host.items()}
        return device_from_arrays(arrays, self.meta)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, backend=tbe, exec={self.exec_mode}, "
                f"offloaded={self.offloaded}")

    # -- compute ----------------------------------------------------------
    def _decode_matmul(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is None:
            w = tbe_mma_decode(self.device_container())
            return _exact_linear(x.to(torch.bfloat16), w, self.bias)
        w, slot = self.pool.acquire(self)
        self.pool.prefetch_after(self, slot)
        y = _exact_linear(x.to(torch.bfloat16), w, self.bias)
        self.pool.release(slot)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = int(x.numel() // max(1, x.shape[-1]))
        if self.exec_mode == "exact":
            return self._decode_matmul(x).to(x.dtype)
        big = self.pool is not None and self.dense_bytes // 2 > self.pool.capacity
        if m > self.fused_max_m and not big:
            return self._decode_matmul(x).to(x.dtype)
        flat = x.reshape(-1, x.shape[-1])
        dev = self.device_container()
        if m <= MAX_M:
            y = tbe_mma_gemm(dev, flat, self.bias)
        else:
            y = torch.cat([
                tbe_mma_gemm(dev, flat[i:i + MAX_M], self.bias)
                for i in range(0, m, MAX_M)
            ], dim=0)
        return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


# ---------------------------------------------------------------------------
# host-resident embedding
# ---------------------------------------------------------------------------
class HostEmbedding(nn.Module):
    """``nn.Embedding`` whose table lives in (pinned) host RAM.

    The table is deliberately NOT a registered parameter or buffer: ``.to()``
    must never drag it onto the GPU, and ``model.device`` must keep reporting
    the accelerator.
    """

    def __init__(self, weight: torch.Tensor, device: torch.device, *,
                 padding_idx: Optional[int] = None, pin: bool = True):
        super().__init__()
        w = weight.detach().contiguous()
        if w.device.type != "cpu":
            w = w.cpu()
        if pin and torch.cuda.is_available():
            w = w.pin_memory()
        self._table = w
        self.num_embeddings, self.embedding_dim = int(w.shape[0]), int(w.shape[1])
        self.padding_idx = padding_idx
        self.target_device = torch.device(device)

    @property
    def weight(self) -> torch.Tensor:
        return self._table

    @property
    def host_bytes(self) -> int:
        return int(self._table.numel()) * int(self._table.element_size())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        idx = ids.reshape(-1).to("cpu", torch.long)
        rows = self._table.index_select(0, idx)
        return rows.reshape(*ids.shape, self.embedding_dim).to(self.target_device)

    def extra_repr(self) -> str:
        return (f"{self.num_embeddings}, {self.embedding_dim}, host_resident=True, "
                f"pinned={self._table.is_pinned() if torch.cuda.is_available() else False}")


# ---------------------------------------------------------------------------
# layer streaming
# ---------------------------------------------------------------------------
_ALIGN = 256


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class LayerStreamer:
    """Stream offloaded decoder layers' containers over PCIe, two slots deep.

    Each offloaded layer's coded arrays are concatenated (256-byte aligned)
    into ONE pinned host blob, so a layer is one ``cudaMemcpyAsync``.  Two GPU
    slots alternate; the next offloaded layer is copied on a dedicated copy
    stream while the current one computes.  Slot reuse waits on an event
    recorded after the previous occupant's forward, so a copy can never
    overwrite bytes a kernel is still reading.
    """

    def __init__(self, device: torch.device, layers: List[Tuple[nn.Module, List[TBEServeLinear]]]):
        self.device = torch.device(device)
        self.layers = []
        max_bytes = 0
        for layer_module, linears in layers:
            offsets = []
            off = 0
            for lin in linears:
                spec = {}
                for key in ("planes", "smb", "esc", "sbbase"):
                    arr = lin._host[key]
                    nbytes = int(arr.numel()) * int(arr.element_size())
                    spec[key] = (off, nbytes, arr.dtype, int(arr.numel()))
                    off = _align(off + nbytes)
                offsets.append(spec)
            blob = torch.empty(off, dtype=torch.uint8)
            for lin, spec in zip(linears, offsets):
                for key, (o, nb, _dt, _n) in spec.items():
                    blob[o:o + nb].copy_(lin._host[key].contiguous().view(torch.uint8).reshape(-1))
                lin._host = None  # the blob is now the only host copy
            if torch.cuda.is_available():
                blob = blob.pin_memory()
            self.layers.append({"module": layer_module, "linears": linears,
                                "offsets": offsets, "blob": blob, "bytes": off})
            max_bytes = max(max_bytes, off)
        self.slot_bytes = max_bytes
        self.slots = [torch.empty(max_bytes, dtype=torch.uint8, device=self.device)
                      for _ in range(2)] if self.layers else []
        self.copy_stream = torch.cuda.Stream(self.device) if self.layers else None
        self._free: List[Any] = [None, None]
        self._next_slot = 0
        self._pending: Dict[int, Tuple[int, Any]] = {}
        self._active: Dict[int, int] = {}
        self._handles = []
        self.stats = {"copies": 0, "bytes_copied": 0, "cold_waits": 0}
        for i, rec in enumerate(self.layers):
            self._handles.append(rec["module"].register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(rec["module"].register_forward_hook(self._make_post(i)))
        for i, rec in enumerate(self.layers):
            for lin, spec in zip(rec["linears"], rec["offsets"]):
                lin._stream_spec = (i, spec)
                lin.temporary_device_container = self._temp_container_fn(lin)

    @property
    def host_bytes(self) -> int:
        return sum(r["bytes"] for r in self.layers)

    @property
    def gpu_slot_bytes(self) -> int:
        return 2 * self.slot_bytes if self.layers else 0

    def _issue(self, i: int) -> None:
        slot = self._next_slot
        self._next_slot ^= 1
        rec = self.layers[i]
        if self._free[slot] is not None:
            self.copy_stream.wait_event(self._free[slot])
        with torch.cuda.stream(self.copy_stream):
            self.slots[slot][: rec["bytes"]].copy_(rec["blob"], non_blocking=True)
            ev = self.copy_stream.record_event()
        self._pending[i] = (slot, ev)
        self.stats["copies"] += 1
        self.stats["bytes_copied"] += rec["bytes"]

    def _bind(self, i: int, slot: int) -> None:
        rec = self.layers[i]
        buf = self.slots[slot]
        for lin, spec in zip(rec["linears"], rec["offsets"]):
            arrays = {}
            for key, (o, nb, dt, n) in spec.items():
                arrays[key] = buf[o:o + nb].view(dt)[:n]
            lin._dev = device_from_arrays(arrays, lin.meta)

    def _make_pre(self, i: int):
        def pre(_module, _args):
            if i not in self._pending:
                self.stats["cold_waits"] += 1
                self._issue(i)
            slot, ev = self._pending.pop(i)
            torch.cuda.current_stream(self.device).wait_event(ev)
            self._bind(i, slot)
            self._active[i] = slot
            nxt = (i + 1) % len(self.layers)
            if nxt != i and nxt not in self._pending and nxt not in self._active:
                self._issue(nxt)
        return pre

    def _make_post(self, i: int):
        def post(_module, _args, _out):
            slot = self._active.pop(i)
            self._free[slot] = torch.cuda.current_stream(self.device).record_event()
            for lin in self.layers[i]["linears"]:
                lin._dev = None
        return post

    def _temp_container_fn(self, lin: TBEServeLinear):
        def temp():
            i, spec = lin._stream_spec
            blob = self.layers[i]["blob"]
            arrays = {}
            for key, (o, nb, dt, n) in spec.items():
                arrays[key] = blob[o:o + nb].view(dt)[:n].to(self.device)
            return device_from_arrays(arrays, lin.meta)
        return temp

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


__all__ = [
    "DEFAULT_FUSED_MAX_M",
    "DecodePool",
    "EXEC_MODES",
    "HostEmbedding",
    "LayerStreamer",
    "PoolRegistry",
    "ServeError",
    "TBEServeLinear",
    "device_from_arrays",
    "host_arrays",
]
