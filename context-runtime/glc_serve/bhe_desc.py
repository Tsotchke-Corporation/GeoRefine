"""Explicit BHE1 serving descriptors with the parent's split reduction contract."""
from __future__ import annotations

import hashlib
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch

from georefine_context_runtime import context_exponent_gemm as gemm
from georefine_context_runtime import context_exponent_packet as packet
from georefine_context_runtime.context_exponent_native import native_cpu_session


@packet.native_cpu_backend()
@native_cpu_session()
def encode_matrix(weight, *, row_classes=16, validation_witnesses=False):
    """Encode verified BF16 source, including zero padding for small projections.

    ``validation_witnesses`` is an explicit startup-only opt-in.  When set,
    the existing exact panel roundtrip also returns opaque process-local proofs
    that may be supplied to resident preparation for the same immutable blobs.
    """
    from georefine_context_runtime import bitexact_context_codec as codec

    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError('BHE1 requires a BF16 matrix')
    n, k = map(int, weight.shape)
    if not 1 <= n <= 262144 or not 128 <= k <= 32768 or k % 128:
        raise ValueError('unsupported BHE1 matrix geometry')
    if type(row_classes) is not int or not 1 <= row_classes <= 256:
        raise ValueError('row_classes must be 1..256')
    if type(validation_witnesses) is not bool:
        raise ValueError('validation_witnesses must be boolean')
    source = weight.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
    source_sha = hashlib.sha256(memoryview(source)).hexdigest()
    words = np.zeros(((n + 63) // 64 * 64, k), dtype='<u2')
    words[:n] = source
    rc, cc = codec._labels(words, row_classes, 4)
    frequencies = codec._frequencies_kind(codec._lib(), words, rc, cc, row_classes, 4, 0)
    freq = np.frombuffer(frequencies, dtype='<u4').reshape(row_classes * 4, 1024)
    offsets, entries = [0], bytearray()
    for row in freq:
        cumulative = np.cumsum(row, dtype=np.uint64)
        if int(cumulative[-1]) not in (0, 65536):
            raise ValueError('invalid normalized BHE1 frequencies')
        symbols = np.flatnonzero(row)
        for symbol in symbols:
            entries.extend(struct.pack('<HI', int(symbol), int(cumulative[symbol])))
        offsets.append(offsets[-1] + len(symbols))
    cdf = struct.pack('<' + 'I' * len(offsets), *offsets) + entries
    blobs, witnesses, cache = [], [], {}
    for start in range(0, len(words), 64):
        panel = words[start:start + 64]
        blob = packet.encode(panel, row_classes=row_classes, col_classes=4,
                             row_labels=rc[start:start + 64], col_labels=cc,
                             sparse_cdf=bytes(cdf), shared_metadata=True,
                             experimental_large_geometry=True, _table_cache=cache)
        if validation_witnesses:
            decoded, witness = packet.decode_cpu_with_witness(
                blob, source=panel, _table_cache=cache)
        else:
            decoded = packet.decode_cpu(blob, _table_cache=cache)
        if not np.array_equal(decoded.view(np.uint16), panel):
            raise ValueError('BHE1 source round trip mismatch')
        blobs.append(blob)
        if validation_witnesses:
            witnesses.append(witness)
    receipt = dict(shape=[n, k], padded_shape=list(words.shape), source_sha256=source_sha,
                   source_bf16_bytes=n * k * 2, wire_bytes=sum(map(len, blobs)))
    return ((blobs, receipt, tuple(witnesses)) if validation_witnesses
            else (blobs, receipt))


class BHELinear(torch.nn.Module):
    """Own one compressed representation shared by prefill and batch decode."""

    def __init__(self, blobs, receipt, *, device, validation_witnesses=None):
        super().__init__()
        self.in_features = int(receipt['shape'][1])
        self.out_features = int(receipt['shape'][0])
        self.blobs, self.receipt = tuple(blobs), dict(receipt)
        if validation_witnesses is not None:
            if (not isinstance(validation_witnesses, (list, tuple)) or
                    len(validation_witnesses) != len(self.blobs)):
                raise ValueError('BHE1 validation witness count must match panel count')
            self.validation_witnesses = tuple(validation_witnesses)
        else:
            self.validation_witnesses = None
        self.register_buffer('_device_anchor', torch.empty(0, device=device), persistent=False)
        self.bias = None
        self._descriptor = None
        self._prefill_mode = 'fused'
        self._prefill_dense_requirements = None

    def fast_desc(self, *, splits=None):
        if self._descriptor is None:
            self._descriptor = BHELin(self, splits=splits)
        elif splits is not None and self._descriptor.S != splits:
            raise ValueError('BHE1 descriptor already bound to a different reduction order')
        return self._descriptor

    @property
    def resident_bytes(self):
        return self.fast_desc().bytes

    def configure_prefill(self, mode='fused', *, max_dense_bytes=None):
        if mode not in ('fused', 'dense_temporary'):
            raise ValueError("prefill mode must be 'fused' or 'dense_temporary'")
        if mode == 'fused':
            self._prefill_mode = mode
            self._prefill_dense_requirements = None
            return None
        if type(max_dense_bytes) is not int or max_dense_bytes <= 0:
            raise ValueError('dense_temporary requires a positive max_prefill_dense_bytes')
        desc = self.fast_desc()
        dense_bytes = int(self.receipt['padded_shape'][0]) * self.in_features * 2
        tiles = desc.provenance.get('tiles')
        if type(tiles) is not int or tiles <= 0:
            raise ValueError('BHE1 descriptor lacks valid decode status tile metadata')
        status_bytes = tiles * 256 * 4
        total = dense_bytes + status_bytes
        if total > max_dense_bytes:
            raise MemoryError(
                f'prefill dense temporary budget exceeded: required {total} > {max_dense_bytes}')
        self._prefill_mode = mode
        self._prefill_dense_requirements = {
            'dense_bytes': dense_bytes, 'status_bytes': status_bytes, 'total_bytes': total,
            'padded_shape': list(self.receipt['padded_shape']),
        }
        return dict(self._prefill_dense_requirements)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError('BHE1 input must be a Torch tensor')
        if x.dtype != torch.bfloat16:
            raise TypeError('BHE1 input must be BF16')
        if x.device != self._device_anchor.device:
            raise RuntimeError('BHE1 input device does not match the descriptor device')
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError('BHE1 input trailing dimension does not match K')
        rows = x.reshape(-1, self.in_features).contiguous()
        if self._prefill_mode == 'dense_temporary':
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('dense_temporary prefill is not graph-capturable')
            import torch.nn.functional as F
            desc = self.fast_desc()
            dense, status = desc._backend.decode(desc.plan)
            try:
                if not isinstance(dense, torch.Tensor) or dense.dtype != torch.bfloat16:
                    raise RuntimeError('BHE dense prefill decode returned invalid weight dtype')
                padded_n = int(self.receipt['padded_shape'][0])
                if dense.device != rows.device or dense.ndim != 2:
                    raise RuntimeError('BHE dense prefill decode returned invalid weight device/shape')
                if tuple(dense.shape) != (padded_n, self.in_features):
                    raise RuntimeError('BHE dense prefill decode returned invalid weight shape')
                if not isinstance(status, torch.Tensor) or status.dtype != torch.int32:
                    raise RuntimeError('BHE dense prefill decode returned invalid status dtype')
                tiles = desc.provenance.get('tiles')
                expected_status = tiles * 256 if type(tiles) is int and tiles > 0 else None
                if (status.device != rows.device or status.numel() != expected_status):
                    raise RuntimeError('BHE dense prefill decode returned invalid status buffer')
                if bool(torch.any(status != 0).item()):
                    raise RuntimeError('BHE dense prefill decode status failure')
                logical = dense[:self.out_features, :self.in_features]
                result = F.linear(rows, logical, self.bias)
            finally:
                del dense, status
            return result.reshape(*x.shape[:-1], self.out_features)
        y = torch.empty((len(rows), self.out_features), device=x.device, dtype=torch.bfloat16)
        desc = self.fast_desc()
        for start in range(0, len(rows), 512):
            desc(rows[start:start + 512], y[start:start + 512])
        return y.reshape(*x.shape[:-1], self.out_features)


def activation_tile(rows, cap, supported, forced=None):
    if cap not in (64,128) or supported not in (64,128):
        raise ValueError('BHE1 tile capacity must be 64 or 128')
    if forced is not None:
        if forced not in (64,128) or forced > supported:
            raise ValueError('requested BHE1 tile exceeds prepared shared-memory capacity')
        return forced
    return min(cap,supported) if rows > 64 else 64


class BHEBuffers:
    def __init__(self, device):
        self.device = device
        self.output = torch.empty(0, dtype=torch.bfloat16, device=device)
        self.workspace = torch.empty(0, dtype=torch.float32, device=device)
        self.status = torch.empty(0, dtype=torch.int32, device=device)

    def reserve(self, output, workspace, status):
        for name, size in (('output', output), ('workspace', workspace), ('status', status)):
            old = getattr(self, name)
            if old.numel() < size:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('reserve BHE1 workspace before graph capture')
                setattr(self, name, torch.empty(size, dtype=old.dtype, device=self.device))


class BHELin:
    kind = 'bhe1'

    @packet.native_cpu_backend()
    @native_cpu_session()
    def __init__(self, module, *, splits=None):
        # Resident preparation rechecks every panel's decoded source digest.
        # Use the exact native validator here as during encoding; otherwise
        # descriptor creation traverses full-model Huffman streams in Python.
        from .bigemm import split_for

        self.N, self.K = module.out_features, module.in_features
        self.device = module._device_anchor.device
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.S = split_for(self.N, self.K, sms) if splits is None else int(splits)
        prepare_options = dict(device=self.device, splits=self.S)
        if module.validation_witnesses is not None:
            prepare_options['validation_witnesses'] = module.validation_witnesses
        self.plan, self._backend = gemm._prepare_matrix_cuda_bound(
            module.blobs, **prepare_options)
        self.provenance = dict(self._backend.plan_provenance(self.plan))
        self.padded_N = int(module.receipt['padded_shape'][0])
        self._bytes = int(self.provenance['gpu_plan_bytes'])
        self.tile_cap = 64
        self.max_tile = int(self.provenance['max_activation_tile'])
        self.buffers = BHEBuffers(self.device)
        self._capacity = 0
        self._calls = 0
        self.reserve(16)

    @property
    def bytes(self):
        return self._bytes

    @property
    def coded_bytes(self):
        return self._bytes

    def reserve(self, max_m):
        if type(max_m) is not int or not 1 <= max_m <= 512:
            raise ValueError('BHE1 batch capacity must be 1..512')
        if max_m <= self._capacity:
            return
        self._capacity = max_m
        self.buffers.reserve(*self.buffer_sizes(max_m))

    def buffer_sizes(self, rows):
        return (rows * self.padded_N,
                self.S * rows * self.padded_N,
                self.S * int(self.provenance['tiles']) * ((rows + 63) // 64) * 256)

    def share_workspace(self, ws, tile=None):
        if tile is not None:
            if tile not in (64,128):
                raise ValueError("BHE1 tile capacity must be 64 or 128")
            self.tile_cap = tile
        if not hasattr(ws, 'bhe_buffers'):
            ws.bhe_buffers = BHEBuffers(self.device)
        self.buffers = ws.bhe_buffers
        self._capacity = 0


    def __call__(self, x, y, tile=None):
        m = int(x.shape[0])
        if tuple(y.shape) != (m, self.N) or y.dtype != torch.bfloat16 or y.device != self.device:
            raise ValueError('BHE1 output must have matching shape, BF16 dtype and device')
        if m > self._capacity:
            self.reserve(m)
        tiles = int(self.provenance['tiles'])
        selected_tile = activation_tile(m, self.tile_cap, self.max_tile, tile)
        blocks = (m + selected_tile - 1) // selected_tile
        count = self.S * tiles * blocks * 256
        shape = (tiles, blocks, 256)
        if self.S > 1:
            shape = (self.S, *shape)
        status = self.buffers.status[:count].view(shape)
        output = self.buffers.output[:m * self.padded_N].view(m, self.padded_N)
        self._backend.gemm_out(self.plan, x, output, self.buffers.workspace, status, selected_tile)
        y.copy_(output[:, :self.N])
        self._calls += 1

    def backend_identity(self):
        return dict(backend='bhe1', format='bhe1', N=self.N, K=self.K, S=self.S,
                    tile_cap=self.tile_cap, max_activation_tile=self.max_tile, dispatch_calls=self._calls, graph_replays_counted=False,
                    plan=dict(self.provenance))


def group_descriptor(modules):
    """Use the fused parent's split count, even when panels are stored separately."""
    from .bigemm import split_for
    from .q8serve import GroupLin

    if not modules or any(not isinstance(m, BHELinear) for m in modules):
        raise ValueError('BHE1 projection group must contain BHE1 modules only')
    k = modules[0].in_features
    if any(m.in_features != k for m in modules):
        raise ValueError('BHE1 group members must share K')
    sms = torch.cuda.get_device_properties(modules[0]._device_anchor.device).multi_processor_count
    splits = split_for(sum(m.out_features for m in modules), k, sms)
    members = [(m.fast_desc(splits=splits), m.out_features) for m in modules]
    return members[0][0] if len(members) == 1 else GroupLin(members)


def _fast_projection_groups(model):
    """The text projection groups consumed by FastDecoder, before replacement."""
    from .fastdec import _text

    groups = []
    for layer in _text(model).layers:
        mlp = layer.mlp
        groups.extend(((mlp.gate_proj, mlp.up_proj), (mlp.down_proj,)))
        if (getattr(layer, 'linear_attn', None) is not None
                and getattr(layer, 'block_type', 'linear_attention') == 'linear_attention'):
            attn = layer.linear_attn
            groups.extend(((attn.in_proj_qkv, attn.in_proj_z, attn.in_proj_b, attn.in_proj_a),
                           (attn.out_proj,)))
        else:
            attn = layer.self_attn
            groups.extend(((attn.q_proj, attn.k_proj, attn.v_proj), (attn.o_proj,)))
    groups.append((model.get_output_embeddings(),))
    return groups


class _DescriptorPipeline:
    """Prepare complete, verified groups while later tensors are still encoded.

    Replacements stay ordered on the converter's thread. A group is submitted
    once every member is installed; group_descriptor retains the parent's fused
    split count and the existing native parser, decoded-source checks and upload.
    """

    def __init__(self, groups, *, selected_ids, device, workers, log=print):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError('descriptor preparation workers must be 1..8')
        self.groups = [tuple(id(module) for module in group) for group in groups]
        self.by_module = {}
        for index, group in enumerate(self.groups):
            if not group:
                raise ValueError('empty descriptor projection group')
            for ident in group:
                if ident in self.by_module:
                    raise ValueError('shared projection cannot enter multiple descriptor groups')
                self.by_module[ident] = index
        if set(self.by_module) != set(selected_ids):
            raise ValueError('descriptor groups do not cover exactly the selected text projections')
        self.device, self.workers, self.log = torch.device(device), workers, log
        self.installed = {}
        self.submitted = set()
        self.futures = []
        self.started = time.perf_counter()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='bhe-descriptor')

    def check(self):
        for future in self.futures:
            if future.done() and not future.cancelled():
                future.result()  # Propagate validation/upload failures before more conversion.

    def _prepare(self, modules):
        context = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()
        with context:
            return group_descriptor(modules)

    def install(self, original_id, replacement):
        self.check()
        index = self.by_module.get(original_id)
        if index is None:
            return  # Input embeddings are installed by the same conversion loop.
        if original_id in self.installed:
            raise ValueError('descriptor projection installed twice')
        if not isinstance(replacement, BHELinear):
            raise ValueError('descriptor preparation requires verified BHE1 replacements')
        self.installed[original_id] = replacement
        group = self.groups[index]
        if index not in self.submitted and all(ident in self.installed for ident in group):
            modules = tuple(self.installed[ident] for ident in group)
            self.futures.append(self.pool.submit(self._prepare, modules))
            self.submitted.add(index)

    def finish(self):
        if len(self.submitted) != len(self.groups):
            raise ValueError('descriptor preparation has incomplete projection groups')
        for future in self.futures:
            future.result()
        self.log(f'[bhe-descriptors] ready groups={len(self.groups)} '
                 f'projections={len(self.by_module)} workers={self.workers}')
        return dict(status='ready', workers=self.workers, groups=len(self.groups),
                    projections=len(self.by_module),
                    elapsed_seconds=time.perf_counter() - self.started)

    def close(self):
        for future in self.futures:
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)


def extra_workspace_bytes(fd, rows):
    """Conservative admission charge for the shared BHE1 buffers, before capture."""
    maxima = [0, 0, 0]
    def visit(desc):
        if isinstance(desc, BHELin):
            for i, value in enumerate(desc.buffer_sizes(rows)):
                maxima[i] = max(maxima[i], value)
        elif hasattr(desc, 'members'):
            for member, _ in desc.members:
                visit(member)
        elif hasattr(desc, 'd'):
            visit(desc.d)
    for desc in fd._descs():
        visit(desc)
    return sum(n * size for n, size in zip(maxima, (2, 4, 4)))
