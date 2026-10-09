"""Read a GLC codec v2 / v2.1 artifact and upload its per-tensor streams for the BI-GEMM kernels.

Both format ids are readable: ``glc-codec-v2`` (format_version 1, ``codec_v2``, kernel TBE2) and
``glc-codec-v2.1`` (format_version 2, ``codec_v21``, kernel TBE21; spec
``docs/research/CODEC_V21_FORMAT_20261004.md``).  The manifest's ``format`` picks the codec.

Artifact layout (written by ``scripts/codec_v2_pack.py``; format frozen in
``docs/research/CODEC_V2_FORMAT_20261004.md``)::

    <artifact>/manifest.json                 {"format": "glc-codec-v2", "format_version": 1,
                                              "tensors": [<header> + {"file", "bytes", ...}
                                                          | {"name", "alias_of", ...}]}
    <artifact>/tensors/<NNNNN>.safetensors   "<name>::smb|planes|ovf|len|grp|codebook" or "::raw"

Device layout (what ``bi_csrc/bi_gemm.cuh`` ``Ld<F_TBE2>`` reads; the 16-word ``ovf`` tail and
the 16-byte ``len`` tail are the format's "the loader appends ..." contract, section 10, and are
what makes the kernel's window load and masked ``len`` read fault-free)::

    planes  int32 [N * nch * 8]          smb   uint8 [N * K]
    ovf     int32 [W + 16]               len   uint8 [N * nchp + 16]
    grp     int32 [N * ngrp]             cb16  uint8 [16]   (cb[0..12], three zero bytes)

Integrity: every stream's crc32 is checked against the header before upload; ``decode_host``
re-derives the source words on the CPU and checks the per-tensor sha256 (the census identity).
Only numpy is needed to read; ``torch`` only for ``upload``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import numpy as np

from . import codec_v2 as cv
from . import codec_v21 as cv21

#: format id -> (codec module, format_version)
CODECS = {cv.FORMAT: (cv, cv.FORMAT_VERSION), cv21.FORMAT: (cv21, cv21.FORMAT_VERSION)}

OVF_TAIL_WORDS = 16
LEN_TAIL_BYTES = 16
CB_BYTES = 16


class V2ArtifactError(RuntimeError):
    pass


_NP = {"BF16": np.uint16, "F16": np.uint16, "F32": np.float32, "F64": np.float64,
       "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8,
       "BOOL": np.bool_}


class V2Artifact:
    """A codec-v2 artifact directory: manifest + one safetensors file per stored tensor."""

    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        mpath = os.path.join(self.path, "manifest.json")
        if not os.path.isfile(mpath):
            raise V2ArtifactError(f"{self.path}: no manifest.json (not a codec v2 artifact)")
        with open(mpath) as fh:
            self.manifest = json.load(fh)
        fmt = self.manifest.get("format")
        if fmt not in CODECS:
            raise V2ArtifactError(f"{self.path}: format {fmt!r} is not one of {sorted(CODECS)}")
        self.codec, want_ver = CODECS[fmt]
        self.format = fmt
        if int(self.manifest.get("format_version", -1)) != want_ver:
            raise V2ArtifactError(f"{self.path}: format_version {self.manifest.get('format_version')} "
                                  f"!= {want_ver} for {fmt}")
        self.records: Dict[str, dict] = {r["name"]: r for r in self.manifest["tensors"]}

    # -- names -------------------------------------------------------------
    def names(self) -> List[str]:
        return [r["name"] for r in self.manifest["tensors"]]

    def stored_record(self, name: str) -> dict:
        """The record that owns the bytes (follows ``alias_of``)."""
        r = self.records[name]
        seen = set()
        while r.get("alias_of"):
            if r["name"] in seen:
                raise V2ArtifactError(f"alias cycle at {name}")
            seen.add(r["name"])
            r = self.records[r["alias_of"]]
        return r

    def is_alias(self, name: str) -> bool:
        return bool(self.records[name].get("alias_of"))

    def has_config(self) -> bool:
        return os.path.isfile(os.path.join(self.path, "config.json"))

    # -- tensors -----------------------------------------------------------
    def read(self, name: str, *, check_crc: bool = True):
        """The stored tensor for ``name`` (``codec_v2.V2Tensor`` or ``codec_v21.V21Tensor``, by the
        artifact's format; an alias returns its source's streams, renamed)."""
        from safetensors.numpy import load_file

        r = self.stored_record(name)
        arrays = load_file(os.path.join(self.path, r["file"]))
        t = self.codec.from_named_arrays(r, arrays)
        if check_crc:
            self.codec.verify_crc(t)
        if name != r["name"]:
            t.name = name
        return t

    def decode_host(self, name: str) -> np.ndarray:
        """Exact source words of ``name`` on the CPU (crc32 + sha256 checked).  BF16 / F16 come
        back as uint16 bit patterns in the source shape; other dtypes as typed arrays."""
        t = self.read(name)
        out = self.codec.decode_tensor(t)                 # raises on crc / sha mismatch
        if t.geom.mode == cv.MODE_RAW and t.dtype not in ("BF16", "F16"):
            dt = _NP.get(t.dtype)
            if dt is None:
                raise V2ArtifactError(f"{name}: unsupported raw dtype {t.dtype}")
            return np.frombuffer(out.tobytes(), dtype=dt).reshape(t.geom.shape)
        return out

    def items(self) -> Iterator[dict]:
        yield from self.manifest["tensors"]


def gemm_ready(t: cv.V2Tensor) -> bool:
    """Can this tensor be a BI-GEMM TBE2 weight?  Rows mode, no column padding (padding words are
    ``cb[0] << 7``, not zeros, so a padded tensor may not enter a dot product), and the
    BI-GEMM binding's K % 256 rule."""
    g = t.geom
    return g.mode == cv.MODE_ROWS and g.K == g.Kp and g.K % 256 == 0


@dataclass
class V2Device:
    """One tensor's TBE2 streams on a device, laid out for ``Ld<F_TBE2>``."""
    name: str
    N: int
    K: int
    len_unit: int
    sha256: str
    planes: object        # torch.int32 [N*nch*8]
    smb: object           # torch.uint8 [N*K]
    ovf: object           # torch.int32 [W + 16]
    len: object           # torch.uint8 [N*nchp + 16]
    grp: object           # torch.int32 [N*ngrp]
    cb16: object          # torch.uint8 [16]
    meta: Dict[str, int] = field(default_factory=dict)

    @property
    def nch(self) -> int:
        return self.K // cv.CH

    @property
    def ovf_words(self) -> int:
        """Overflow words WITHOUT the loader tail."""
        return int(self.ovf.numel()) - OVF_TAIL_WORDS

    @property
    def bytes(self) -> int:
        return int(sum(int(a.numel()) * int(a.element_size())
                       for a in (self.planes, self.smb, self.ovf, self.len, self.grp, self.cb16)))


def device_arrays_host(t: cv.V2Tensor) -> Dict[str, np.ndarray]:
    """The device layout as numpy (what ``upload`` copies; also what the CPU proofs consume)."""
    if not gemm_ready(t):
        raise V2ArtifactError(f"{t.name}: not a TBE2 GEMM weight (mode {t.geom.mode}, K {t.geom.K}, "
                              f"Kp {t.geom.Kp})")
    cb16 = np.zeros(CB_BYTES, np.uint8)
    cb16[:cv.NSYM] = np.asarray(t.cb, np.uint8)
    return {
        "planes": np.ascontiguousarray(t.planes, np.uint32).reshape(-1).view(np.int32),
        "smb": np.ascontiguousarray(t.smb, np.uint8).reshape(-1),
        "ovf": np.concatenate([np.asarray(t.ovf, np.uint32).reshape(-1),
                               np.zeros(OVF_TAIL_WORDS, np.uint32)]).view(np.int32),
        "len": np.concatenate([np.asarray(t.len_, np.uint8).reshape(-1),
                               np.zeros(LEN_TAIL_BYTES, np.uint8)]),
        "grp": np.ascontiguousarray(t.grp, np.uint32).reshape(-1).view(np.int32),
        "cb16": cb16,
    }


def upload(t: cv.V2Tensor, device) -> V2Device:
    """Copy one coded tensor's streams to ``device`` in the kernel's layout (no decode)."""
    import torch

    h = device_arrays_host(t)
    dev = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in h.items()}
    return V2Device(t.name, int(t.geom.N), int(t.geom.K), int(t.len_unit), t.sha256,
                    dev["planes"], dev["smb"], dev["ovf"], dev["len"], dev["grp"], dev["cb16"],
                    meta={"slow_path_chunks": int(t.counts.get("slow_path_chunks", 0))})


# ----------------------------------------------------------------------------- v2.1 (TBE21)
@dataclass
class V21Device:
    """One tensor's TBE21 streams on a device, laid out for ``Ld<F_TBE21>``, with the RESIDENT
    split checkpoints for the launch split count ``S`` (empty when S == 1)."""
    name: str
    N: int
    K: int
    S: int
    ncb: int
    sha256: str
    l1: object            # torch.int16 [N*nch*16]  (lane-interleaved level-1 codes)
    smb: object           # torch.uint8 [N*K]
    ovf: object           # torch.int32 [W + 16]
    rowoff: object        # torch.int32 [N]
    cb16: object          # torch.uint8 [ncb*16]
    rowcb: object         # torch.int32 [N]  codebook of each row (0..ncb-1)
    ckpt: object          # torch.int32 [N*(S-1)]

    @property
    def ovf_words(self) -> int:
        return int(self.ovf.numel()) - OVF_TAIL_WORDS

    @property
    def stored_bytes(self) -> int:
        return int(sum(int(a.numel()) * int(a.element_size())
                       for a in (self.l1, self.smb, self.ovf, self.rowoff, self.cb16))) - 4 * OVF_TAIL_WORDS

    @property
    def resident_bytes(self) -> int:
        return int(sum(int(a.numel()) * int(a.element_size())
                       for a in (self.l1, self.smb, self.ovf, self.rowoff, self.cb16, self.rowcb, self.ckpt)))


def gemm_ready21(t) -> bool:
    g = t.geom
    return g.mode == cv21.MODE_ROWS and g.K == g.Kp and g.K % 256 == 0


def device_arrays_host21(t, S: int, offsets: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """The TBE21 device layout as numpy (what ``upload21`` copies) for launch split count S."""
    if not gemm_ready21(t):
        raise V2ArtifactError(f"{t.name}: not a TBE21 GEMM weight (mode {t.geom.mode}, K {t.geom.K})")
    ck = cv21.split_checkpoints(t, S, offsets)
    return {
        "l1": np.ascontiguousarray(t.l1, np.uint16).reshape(-1).view(np.int16),
        "smb": np.ascontiguousarray(t.smb, np.uint8).reshape(-1),
        "ovf": np.concatenate([np.asarray(t.ovf, np.uint32).reshape(-1),
                               np.zeros(OVF_TAIL_WORDS, np.uint32)]).view(np.int32),
        "rowoff": np.ascontiguousarray(t.rowoff, np.uint32).view(np.int32),
        "cb16": np.ascontiguousarray(t.cb, np.uint8).reshape(-1),
        "rowcb": t.row_codebook().astype(np.int32),
        "ckpt": np.ascontiguousarray(ck, np.uint32).reshape(-1).view(np.int32),
    }


def upload21(t, device, S: int, offsets: Optional[np.ndarray] = None) -> V21Device:
    """Copy one v2.1 tensor's streams to ``device`` (no decode) + its checkpoints for S."""
    import torch

    h = device_arrays_host21(t, S, offsets)
    dev = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in h.items()}
    return V21Device(t.name, int(t.geom.N), int(t.geom.K), int(S), int(t.ncb), t.sha256,
                     dev["l1"], dev["smb"], dev["ovf"], dev["rowoff"], dev["cb16"], dev["rowcb"],
                     dev["ckpt"])


__all__ = ["CB_BYTES", "CODECS", "LEN_TAIL_BYTES", "OVF_TAIL_WORDS", "V21Device", "V2Artifact",
           "V2ArtifactError", "V2Device", "device_arrays_host", "device_arrays_host21", "gemm_ready",
           "gemm_ready21", "upload", "upload21"]
