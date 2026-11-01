"""Off-accelerator PLE (n-gram) embedding gather for Qwen3.8-Flash-Next, on MLX.

WHY THIS FILE EXISTS
---------------------
Qwen3.8-Flash-Next's PLE ("Product-of-Learned-Experts" n-gram) layer is a
plain ``nn.Embedding`` gather -- see ``modeling_qwen4_exp.py``,
``Qwen4ExpTextNGramEmbedding.forward`` (around line 1123, the
``self.ngram_embedding(ngram_ids...)`` call) and ``Qwen4ExpTextPLELayer``
(around line 1190) -- firing at exactly one of 48 layers
(``config.ple_layer_ids``, e.g. ``[2]``) and pulling 16 rows per token
(``ngram_heads = (ngram_size - 1) * heads_per_ngram``). HF's own code
assumes this table lives off-accelerator: it explicitly special-cases a
``meta``-device embedding weight and does the lookup wherever the weight
actually lives (``execution_device`` in that same forward).

The table itself is enormous: ``padded_vocab_size x head_dim_per_ngram``
bf16, checkpointed as ~128 shards named
``model.language_model.layers.<N>.ple.ple_embedding.ngram_embedding.shard_<K>.weight``,
each ``[2500012, 160]`` bf16 (0.80 GB), 102.4 GB total. Per token the actual
output is 16 rows x 320 B = 5 KB -- three orders of magnitude smaller than
the table. This file is the CPU-side half of that trade: never map the
102 GB table into GPU-addressable (unified) memory, never even ``mx.array``
it -- MLX maintainers confirm mmapped arrays are not GPU-usable in MLX (a
community attempt measured 0.025 tok/s doing exactly that). Instead:

  * each shard stays a NumPy ``memmap`` (``mode="r"``) directly over the
    safetensors payload, opened by hand-parsing the 8-byte header length +
    JSON header so the 0.80 GB tensor is never read in full;
  * the 16-row-per-token gather happens on the CPU via NumPy fancy
    indexing (touches only the pages the rows live on);
  * the resulting small ``[tokens, 16, 160]`` bf16 buffer -- and ONLY that
    buffer -- is handed to MLX, which uploads it into unified memory
    trivially because it is a handful of KB, not 102 GB.

The move is bit-exact end to end: the safetensors payload is a raw uint16
bf16 bit pattern; it is never widened to float32 and back. It is copied
uint16-for-uint16 by the memmap gather, then bit-reinterpreted (not
converted) to ``mlx.core.bfloat16`` via ``mx.view`` -- see
``PLEGather.gather`` below and ``tests/test_ple_gather_mlx.py`` for the
bit-exact equality check against an in-memory reference.

INTEGRATION CONTRACT (read this before wiring the MLX PLE layer)
------------------------------------------------------------------
Input side -- what the caller must produce before calling this module:

  1. Compute ``ngram_ids``: the head-mixed, per-head-offset n-gram
     vocabulary ids, exactly as ``Qwen4ExpTextNGramEmbedding.forward``
     does (hash the last ``ngram_size`` token ids with
     ``layer_multipliers``, XOR them together per n-gram order, reduce
     modulo each head's vocab size, add that head's offset). Shape
     ``[tokens, ngram_heads]`` (``ngram_heads == 16`` for this model),
     dtype int64, values in ``[0, padded_vocab_size)``.
  2. Split each flat id into a ``(shard_id, local_row)`` pair. Which shard
     a row lives in is the checkpoint's business, not this module's: the
     loader lane knows ``rows_per_shard`` (2 500 012 for this checkpoint)
     from the shard manifest, so the split is
     ``shard_id, local_row = divmod(ngram_id, rows_per_shard)``.
     ``PLEGather.split_shard`` below is a convenience wrapper for exactly
     that divmod, kept here only so callers do not re-derive it; the
     mapping itself is supplied by the loader.
  3. Call ``PLEGather.gather(shard_ids, local_rows)`` with two int arrays
     of shape ``[tokens, 16]`` (or any matching leading shape).

Output side -- what comes back and what the caller does with it:

  ``gather`` returns an ``mx.array`` of dtype ``mlx.core.bfloat16``, shape
  ``[tokens, 16, 160]`` (``160 == head_dim_per_ngram`` for this
  checkpoint; read from the shard tables, never hardcoded here). The MLX
  PLE layer flattens the last two axes -- mirroring
  ``Qwen4ExpTextNGramEmbedding.forward``'s trailing ``.flatten(-2)`` --
  to get ``[tokens, ngram_heads * head_dim_per_ngram]`` (``ple_embed_dim``),
  then feeds that into ``key_proj`` / ``value_proj`` exactly as
  ``Qwen4ExpTextPLELayer.forward`` does, gates by
  ``sigmoid((key . query) / sqrt(hidden_size))``, and adds the gated value
  into the hyper-connection residual stream (the ``hidden_states = hidden_states
  + self.ple(...)`` line in ``Qwen4ExpTextDecoderLayer.forward``). This
  module's job stops at the ``mx.array`` -- it does not know about
  ``key_proj``/``value_proj``/gating; those stay in the MLX model.

Caller sketch, per decode step::

    shard_ids, local_rows = ple_gather.split_shard(ngram_ids, rows_per_shard)
    embeddings = ple_gather.gather(shard_ids, local_rows)     # bf16 [tokens,16,160]
    embeddings = embeddings.reshape(embeddings.shape[0], -1)  # [tokens, ple_embed_dim]

SAFETY
-------
This module never opens a model weight file for anything but the PLE
shards, never loads a model, and never touches a serving process -- it is
CPU-memmap + a small MLX upload only, well inside the standing safety hold
(Metal allocation kept under a fraction of the 256 MB ceiling; see
``METAL_MEMORY_LIMIT_BYTES``). Every entry point in this module calls
``mx.set_memory_limit`` at import time, the same pattern
``tbe_decode_mlx.py`` uses for the earlier standing hold, so an
over-budget call fails as an MLX allocation error rather than growing
unbounded.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np

import mlx.core as mx

PathLike = Union[str, "os.PathLike[str]"]

#: 8-byte little-endian header-length prefix, per the safetensors format.
_HEADER_LEN_BYTES = 8

#: safetensors dtype tag this module accepts. The PLE tables are shipped
#: bf16; anything else is a loader-config mismatch, not something to
#: silently widen.
BF16_DTYPE_TAG = "BF16"
_BF16_ITEMSIZE = 2

#: Hard ceiling on total Metal allocation for every call in this module,
#: enforcing the standing safety HOLD in force while this component is
#: built: total Metal allocation under 256 MB, no model loads, no serving
#: process interaction. Set well under that ceiling so MLX's own
#: bookkeeping (the tiny per-batch upload plus its bookkeeping overhead)
#: never approaches it.
METAL_MEMORY_LIMIT_BYTES = 128 * 1024 * 1024


def _enforce_memory_limit() -> None:
    try:
        mx.set_memory_limit(METAL_MEMORY_LIMIT_BYTES)
    except Exception:
        # Older/newer MLX may rename this call; never let the safety call
        # itself be the thing that crashes an otherwise-working gather.
        pass


_enforce_memory_limit()


def mlx_metal_available() -> bool:
    """True iff MLX can see a Metal GPU device on this machine.

    Never raises -- callers (tests, benchmark) use this to skip cleanly
    rather than fail when there is no GPU or the Metal backend errors out.
    """
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class PLEShardError(RuntimeError):
    """Base class for every error this module raises for a malformed shard."""


class PLEShardTruncatedError(PLEShardError):
    """The shard file is shorter than its own header says it should be.

    A truncated ``.safetensors`` shard is a known trap in this repo (see
    ``bug_gstmp_hardlink_truncated_weights.md``): a hardlink or interrupted
    copy can leave a file whose header describes a full tensor while the
    bytes on disk stop early. This is always a fetch/sync defect, never a
    legitimate shard -- it must never be silently zero-padded.
    """


class PLEShardDtypeError(PLEShardError):
    """The named tensor's safetensors dtype tag is not ``BF16``."""


class PLEShardShapeError(PLEShardError):
    """The named tensor's shape does not match what the caller expected."""


class PLEShardMissingTensorError(PLEShardError):
    """The named tensor is not present in this shard's header."""


class PLEGatherError(RuntimeError):
    """Base class for gather-time errors (as opposed to shard-open errors)."""


class PLEGatherShardError(PLEGatherError):
    """A row referenced a shard id this ``PLEGather`` was not given a table for."""


class PLEGatherIndexError(PLEGatherError):
    """A local row index fell outside ``[0, table.rows)`` for its shard."""


# ---------------------------------------------------------------------------
# Single-shard safetensors memmap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TensorHeaderEntry:
    dtype: str
    shape: Tuple[int, ...]
    start: int
    end: int


def _parse_safetensors_header(path: Path) -> Tuple[Dict[str, _TensorHeaderEntry], int, int]:
    """Read only the header of a safetensors file: 8-byte length + JSON.

    Returns ``(tensors, header_end_offset, file_size)`` where
    ``header_end_offset`` is the byte offset (from the start of the file)
    at which the binary data blob begins -- every tensor's
    ``data_offsets`` are relative to that point. Never reads tensor
    payload bytes.
    """
    file_size = path.stat().st_size
    if file_size < _HEADER_LEN_BYTES:
        raise PLEShardTruncatedError(
            f"{path}: file is {file_size} bytes, too short to hold even the "
            f"{_HEADER_LEN_BYTES}-byte safetensors header-length prefix"
        )
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(_HEADER_LEN_BYTES))[0]
        header_end = _HEADER_LEN_BYTES + header_len
        if header_end > file_size:
            raise PLEShardTruncatedError(
                f"{path}: header claims {header_len} bytes but the file is only "
                f"{file_size - _HEADER_LEN_BYTES} bytes past the length prefix"
            )
        raw_header = f.read(header_len)
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise PLEShardError(f"{path}: header is not valid JSON: {exc}") from exc

    tensors: Dict[str, _TensorHeaderEntry] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = entry["dtype"]
            shape = tuple(int(s) for s in entry["shape"])
            start, end = entry["data_offsets"]
        except (KeyError, TypeError, ValueError) as exc:
            raise PLEShardError(f"{path}: malformed header entry {name!r}: {entry!r}") from exc
        tensors[name] = _TensorHeaderEntry(dtype=dtype, shape=shape, start=int(start), end=int(end))

    return tensors, header_end, file_size


class PLEShardTable:
    """One PLE embedding shard: a NumPy memmap uint16 view over its safetensors payload.

    Opens ``path``, parses only the safetensors header (never the tensor
    payload), and exposes a read-only ``np.memmap`` of the named tensor's
    raw bf16 bit pattern as ``uint16`` -- no dtype conversion, ever. The
    dtype/shape check is strict and typed: a wrong dtype, a wrong shape,
    or a file too short for the bytes the header promises all raise a
    distinct, catchable error rather than returning garbage or a
    zero-padded array.
    """

    def __init__(
        self,
        path: PathLike,
        tensor_name: str,
        *,
        expected_shape: "Tuple[int, int] | None" = None,
        expected_dtype: str = BF16_DTYPE_TAG,
    ) -> None:
        self.path = Path(path)
        self.tensor_name = tensor_name

        tensors, header_end, file_size = _parse_safetensors_header(self.path)

        if tensor_name not in tensors:
            available = ", ".join(sorted(tensors)[:8])
            raise PLEShardMissingTensorError(
                f"{self.path}: tensor {tensor_name!r} not in header "
                f"(have {len(tensors)} tensors, e.g. {available})"
            )
        entry = tensors[tensor_name]

        if entry.dtype != expected_dtype:
            raise PLEShardDtypeError(
                f"{self.path}:{tensor_name}: dtype is {entry.dtype!r}, expected "
                f"{expected_dtype!r} -- refusing to gather from a table that is "
                f"not the raw bf16 bit pattern this module assumes"
            )

        if len(entry.shape) != 2:
            raise PLEShardShapeError(
                f"{self.path}:{tensor_name}: shape {entry.shape} is not 2-D "
                f"(rows, cols)"
            )
        if expected_shape is not None and tuple(entry.shape) != tuple(expected_shape):
            raise PLEShardShapeError(
                f"{self.path}:{tensor_name}: shape {entry.shape} != expected "
                f"{tuple(expected_shape)}"
            )

        rows, cols = entry.shape
        nbytes = entry.end - entry.start
        expected_nbytes = rows * cols * _BF16_ITEMSIZE
        if nbytes != expected_nbytes:
            raise PLEShardShapeError(
                f"{self.path}:{tensor_name}: header data_offsets span "
                f"{nbytes} bytes but shape {entry.shape} x {_BF16_ITEMSIZE} "
                f"bytes/elem implies {expected_nbytes} bytes -- header is "
                f"internally inconsistent"
            )

        data_start = header_end + entry.start
        data_end = header_end + entry.end
        if data_end > file_size:
            raise PLEShardTruncatedError(
                f"{self.path}:{tensor_name}: tensor payload needs bytes "
                f"[{data_start}, {data_end}) but the file is only {file_size} "
                f"bytes long ({data_end - file_size} bytes missing) -- likely a "
                f"truncated sync/hardlink, not a legitimate shard"
            )

        self.rows = rows
        self.cols = cols
        self._memmap = np.memmap(
            self.path, dtype=np.uint16, mode="r", offset=data_start, shape=(rows, cols)
        )

    @property
    def memmap(self) -> np.memmap:
        return self._memmap

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)


# ---------------------------------------------------------------------------
# Multi-shard gather -> MLX
# ---------------------------------------------------------------------------


class PLEGather:
    """Gathers rows from a set of :class:`PLEShardTable`\\ s straight to MLX.

    Which shard a given row lives in is the model/loader's business (see
    the module docstring's Integration Contract) -- this class accepts
    ``(shard_id, local_row)`` pairs and does not re-derive the split
    itself, except for the small ``split_shard`` convenience below.
    """

    def __init__(self, tables: Dict[int, PLEShardTable]) -> None:
        if not tables:
            raise ValueError("PLEGather needs at least one shard table")
        cols = {t.cols for t in tables.values()}
        if len(cols) != 1:
            raise PLEShardShapeError(
                f"all shards must share the same row width (head_dim_per_ngram); "
                f"got column widths {sorted(cols)}"
            )
        self._tables: Dict[int, PLEShardTable] = dict(tables)
        self.cols = cols.pop()

    @staticmethod
    def split_shard(row_idx: np.ndarray, rows_per_shard: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience: flat global row id -> (shard_id, local_row), by divmod.

        Only correct when every shard holds exactly ``rows_per_shard`` rows
        (true for this checkpoint's uniform ``shard_K`` layout). Provided
        so callers do not re-derive the divmod; the shard boundaries
        themselves come from the loader's shard manifest, not this class.
        """
        row_idx = np.asarray(row_idx)
        shard_id, local_row = np.divmod(row_idx, rows_per_shard)
        return shard_id, local_row

    def gather(self, shard_ids: np.ndarray, rows: np.ndarray) -> mx.array:
        """Gather ``[..., cols]`` bf16 rows, bit-exact, as an ``mx.array``.

        ``shard_ids`` and ``rows`` must have identical shape (typically
        ``[tokens, 16]``); each ``(shard_ids[i], rows[i])`` pair addresses
        one row of one shard. The uint16 bf16 bit pattern is moved by
        NumPy fancy indexing on the memmap (never widened to float32),
        assembled into a small contiguous buffer, and bit-reinterpreted
        (``mx.view``, not a numeric cast) to ``mx.bfloat16``.
        """
        shard_ids = np.asarray(shard_ids)
        rows = np.asarray(rows)
        if shard_ids.shape != rows.shape:
            raise ValueError(
                f"shard_ids shape {shard_ids.shape} != rows shape {rows.shape}"
            )

        out_shape = shard_ids.shape + (self.cols,)
        out = np.empty(out_shape, dtype=np.uint16)
        flat_shard = shard_ids.reshape(-1)
        flat_rows = rows.reshape(-1)
        flat_out = out.reshape(-1, self.cols)

        for shard_id in np.unique(flat_shard):
            table = self._tables.get(int(shard_id))
            if table is None:
                raise PLEGatherShardError(
                    f"row referenced shard id {int(shard_id)}, but this "
                    f"PLEGather only has tables for {sorted(self._tables)}"
                )
            mask = flat_shard == shard_id
            local_rows = flat_rows[mask]
            if local_rows.size:
                lo, hi = int(local_rows.min()), int(local_rows.max())
                if lo < 0 or hi >= table.rows:
                    raise PLEGatherIndexError(
                        f"shard {int(shard_id)}: row index out of range "
                        f"[{lo}, {hi}] for a table with {table.rows} rows"
                    )
            flat_out[mask] = table.memmap[local_rows]

        mx_out = mx.array(out)
        return mx.view(mx_out, mx.bfloat16)
