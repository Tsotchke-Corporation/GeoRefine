"""Verified, bounded startup conversion into the explicit BHE1 serving backend."""
from __future__ import annotations

import math
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time

import torch

from .bhe_desc import BHELinear, encode_matrix
from ._startup_limits import MAX_CPU_ENCODE_WORKERS


def convert_predictive_modules_to_bhe(model, *, device, log=print,
                                      source_inflight_bytes=8 * 1024**3,
                                      verified_cpu_stream=False, cpu_encode_workers=1,
                                      cpu_encode_pipeline=False, cpu_source_resolver=None,
                                      ready_install=False,
                                      validation_witnesses=False, **options):
    """Bound decoded source tensors; resolver cache and encoder workspace are separate.

    Encoding runs in workers; the explicit CPU pipeline also resolves verified
    sources there. Replacements stay ordered on the caller by default. The
    opt-in ``ready_install`` mode may install completed futures out of order
    while retaining the source-byte and queue bounds. On CUDA, complete
    projection groups prepare validated resident descriptors in bounded workers.
    """
    from . import predictive_tbe as bridge

    if options.get('transcode_cache') is not None:
        raise ValueError('BHE1 cannot use a TBE transcode cache')
    if options.get('cpu_vision_dense'):
        raise ValueError('BHE1 dense vision conversion is not supported')
    if type(cpu_encode_workers) is not int or not 1 <= cpu_encode_workers <= MAX_CPU_ENCODE_WORKERS:
        raise ValueError(f'cpu_encode_workers must be an integer between 1 and {MAX_CPU_ENCODE_WORKERS}')
    if cpu_encode_workers > 1 and not verified_cpu_stream:
        raise ValueError('parallel BHE1 encoding requires verified_cpu_stream')
    if type(cpu_encode_pipeline) is not bool:
        raise ValueError('cpu_encode_pipeline must be boolean')
    if type(ready_install) is not bool:
        raise ValueError('ready_install must be boolean')
    if type(validation_witnesses) is not bool:
        raise ValueError('validation_witnesses must be boolean')
    if cpu_encode_pipeline and (cpu_encode_workers <= 1 or cpu_source_resolver is None):
        raise ValueError('BHE1 CPU pipeline requires multiple workers and a verified resolver')
    if ready_install and (not cpu_encode_pipeline or cpu_encode_workers <= 1
                          or cpu_source_resolver is None):
        raise ValueError(
            'BHE1 ready installation requires the CPU pipeline, multiple workers, and a verified resolver')
    if type(source_inflight_bytes) is not int or source_inflight_bytes <= 0:
        raise ValueError('source_inflight_bytes must be a positive integer')
    runtime = bridge._predictive_runtime()
    linear, embedding = runtime.BctxLinear, runtime.BctxEmbedding
    rows, _ = bridge._predictive_module_names(model, linear, embedding)
    text = bridge._text_model(model)
    selected = {id(m) for m in text.modules() if isinstance(m, (linear, embedding))}
    selected.add(id(model.get_output_embeddings()))
    chosen = [r for r in rows if r[2] in selected]
    context_names = {cid: name for name, _, _, cid in chosen}
    names = {name: weight for name, weight, _, _ in chosen}
    order = bridge._dependency_order(model, chosen, context_names)
    if verified_cpu_stream:
        order.reverse()
    started = time.perf_counter()
    receipts = []
    pending = deque()
    # Keep workers fed when an early dependency finishes after later tensors.
    # Reservations still obey the source-byte cap; active workers stay unchanged.
    queue_limit = cpu_encode_workers * (4 if cpu_encode_pipeline else 1)
    inflight = 0
    peak_inflight = 0
    descriptor_pipeline = None
    descriptor_receipt = None
    if cpu_encode_pipeline and torch.device(device).type == 'cuda':
        from .bhe_desc import _DescriptorPipeline, _fast_projection_groups
        projection_ids = {mid for _, _, mid, _ in chosen if mid != id(model.get_input_embeddings())}
        descriptor_pipeline = _DescriptorPipeline(
            _fast_projection_groups(model), selected_ids=projection_ids,
            device=device, workers=min(8, cpu_encode_workers), log=log)
    original_ids = {name: mid for name, _, mid, _ in chosen}
    installed_names = set()

    def install(name, parent, attr, replacement, row, timing):
        if name in installed_names:
            raise ValueError(f'{names[name]}: module installed twice')
        row['startup_timing_seconds'] = dict(timing, elapsed=time.perf_counter() - started)
        setattr(parent, attr, replacement)
        if descriptor_pipeline is not None:
            descriptor_pipeline.install(original_ids[name], replacement)
        installed_names.add(name)
        receipts.append(row)
        log(f"[context-bhe1] converted {names[name]} source_bytes={row['source_bf16_bytes']} "
            f"wire_bytes={row['wire_bytes']} decode_s={timing['decode']:.3f} "
            f"verify_s={timing['verify']:.3f} convert_s={timing['convert']:.3f} "
            f"elapsed_s={row['startup_timing_seconds']['elapsed']:.3f}")

    def encode_cpu(decoded):
        begin = time.perf_counter()
        if validation_witnesses:
            blobs, row, witnesses = encode_matrix(decoded, validation_witnesses=True)
        else:
            blobs, row = encode_matrix(decoded)
            witnesses = None
        return blobs, row, time.perf_counter() - begin, witnesses

    def decode_verified(context, shape, name):
        begin = time.perf_counter()
        decoded = context.decode(check=True)
        decoded_at = time.perf_counter()
        if not isinstance(decoded, torch.Tensor) or decoded.dtype != torch.bfloat16 or tuple(decoded.shape) != shape:
            raise ValueError(f'{names[name]}: invalid decoded BF16 source')
        if verified_cpu_stream and decoded.device.type != 'cpu':
            raise ValueError('verified CPU BHE1 source must remain on CPU')
        actual = bridge._sha256_bf16(decoded)
        if actual != getattr(context, 'source_sha256', None):
            raise ValueError(f'{names[name]}: source SHA mismatch')
        return decoded, actual, dict(decode=decoded_at - begin, verify=time.perf_counter() - decoded_at)

    def decode_encode_cpu(source, shape, name, timing):
        decoded, _, phases = decode_verified(source, shape, name)
        timing.update(phases)
        return encode_cpu(decoded)

    def decode_only_cpu(source, shape, name):
        decoded, actual, timing = decode_verified(source, shape, name)
        return decoded, actual, timing

    def resolver_source(name, context, shape):
        source = cpu_source_resolver.tensor(names[name])
        if (source.device != 'cpu' or source.resident_bytes != 0 or tuple(source.shape) != shape
                or source.source_sha256 != context.source_sha256):
            raise ValueError(f'{names[name]}: CPU resolver source identity mismatch')
        return source

    def finish_one():
        nonlocal inflight
        pending_index = 0
        if ready_install:
            def ready_index():
                return next((index for index, item in enumerate(pending)
                             if item[0].done() and not item[0].cancelled()), None)

            def completed_index():
                return next((index for index, item in enumerate(pending)
                             if item[0].done()), None)

            pending_index = ready_index()
            if pending_index is None:
                # A completion can race the scan above. Wait for any pending
                # future, then select the completed result instead of waiting
                # on the queue head and recreating the FIFO convoy.
                wait([item[0] for item in pending], return_when=FIRST_COMPLETED)
                pending_index = ready_index()
                if pending_index is None:
                    # A cancelled future is an error on the worker path; let
                    # result() surface it while cleanup joins every worker.
                    pending_index = completed_index()
                    if pending_index is None:
                        pending_index = 0
        future, name, parent, attr, size, timing, original, kind = pending[pending_index]
        if kind == 'linear':
            blobs, row, elapsed, witnesses = future.result()
            linear_options = dict(device=device)
            if witnesses is not None:
                linear_options['validation_witnesses'] = witnesses
            replacement = BHELinear(blobs, row, **linear_options)
            row.update(runtime_kind='bhe1_linear', name=names[name])
            timing['convert'] = elapsed
        else:
            decoded, actual, phases = future.result()
            timing.update(phases)
            convert_started = time.perf_counter()
            replacement = bridge._dense_embedding(decoded, original, device)
            row = dict(name=names[name], shape=list(tuple(int(v) for v in original.context_weight.shape)),
                       source_sha256=actual, source_bf16_bytes=size,
                       runtime_dense_bytes=size, runtime_kind='bf16_input_embedding', wire_bytes=0)
            timing['convert'] = time.perf_counter() - convert_started
            del decoded
        install(name, parent, attr, replacement, row, timing)
        if pending_index:
            pending.rotate(-pending_index)
        pending.popleft()
        if pending_index:
            pending.rotate(pending_index)
        inflight -= size

    prefetched_head = None
    prefetched_head_name = None
    head_prefetch_enabled = False
    head_module = model.get_output_embeddings()
    head_name = next((module_name for module_name, _, module_id, _ in chosen
                      if module_id == id(head_module)), None)

    pool = ThreadPoolExecutor(max_workers=cpu_encode_workers) if cpu_encode_workers > 1 else None
    try:
        # Start the late output projection early only when it can coexist with
        # the largest other decoded source under the declared raw-source cap.
        if cpu_encode_pipeline and pool is not None and head_name is not None:
            head_parent, head_attr = bridge._parent_and_attr(model, head_name)
            head_original = getattr(head_parent, head_attr)
            head_context = head_original.context_weight
            head_shape = tuple(int(v) for v in head_context.shape)
            head_size = math.prod(head_shape) * 2
            other_max = max((math.prod(tuple(int(v) for v in getattr(parent, attr).context_weight.shape)) * 2
                             for other_name, _, module_id, _ in chosen if other_name != head_name
                             for parent, attr in [bridge._parent_and_attr(model, other_name)]), default=0)
            if (isinstance(head_original, linear) and head_original.bias is None
                    and head_shape == (head_original.out_features, head_original.in_features)
                    and head_size + other_max <= source_inflight_bytes):
                head_source = resolver_source(head_name, head_context, head_shape)
                head_timing = {}
                future = pool.submit(decode_encode_cpu, head_source, head_shape, head_name, head_timing)
                prefetched_head = (future, head_name, head_parent, head_attr, head_size,
                                   head_timing, head_original, 'linear')
                prefetched_head_name = head_name
                head_prefetch_enabled = True
                inflight = peak_inflight = head_size
                log(f"[context-bhe1] prefetched {names[head_name]} source_bytes={head_size}")
                del head_source, head_context, head_original

        for name in order:
            if descriptor_pipeline is not None:
                descriptor_pipeline.check()
            if ready_install and name in installed_names:
                continue
            parent, attr = bridge._parent_and_attr(model, name)
            original = getattr(parent, attr)
            context = original.context_weight
            shape = tuple(int(v) for v in context.shape)
            size = math.prod(shape) * 2
            if size > source_inflight_bytes:
                raise ValueError(f'{names[name]} exceeds CPU source in-flight budget')
            if isinstance(original, linear) and original.bias is not None:
                raise ValueError(f'{names[name]}: biased projection requires a qualified bias path')
            is_prefetched_head = prefetched_head is not None and name == prefetched_head_name
            while (not is_prefetched_head and pending
                   and (len(pending) + int(prefetched_head is not None) >= queue_limit
                        or inflight + size > source_inflight_bytes)):
                finish_one()
            if not is_prefetched_head:
                peak_inflight = max(peak_inflight, inflight + size)
            if is_prefetched_head:
                pending.append(prefetched_head)
                prefetched_head = None
                prefetched_head_name = None
                del original, context
                continue
            if cpu_encode_pipeline and isinstance(original, linear):
                if shape != (original.out_features, original.in_features):
                    raise ValueError(f'{names[name]}: module geometry differs from verified source')
                source = resolver_source(name, context, shape)
                timing = {}
                future = pool.submit(decode_encode_cpu, source, shape, name, timing)
                pending.append((future, name, parent, attr, size, timing, original, 'linear'))
                inflight += size
                del original, context, source
                continue
            if cpu_encode_pipeline and isinstance(original, embedding):
                if shape != (original.num_embeddings, original.embedding_dim):
                    raise ValueError(f'{names[name]}: embedding geometry differs from verified source')
                source = resolver_source(name, context, shape)
                future = pool.submit(decode_only_cpu, source, shape, name)
                pending.append((future, name, parent, attr, size, {}, original, 'embedding'))
                inflight += size
                del context, source
                continue
            decoded, actual, timing = decode_verified(context, shape, name)
            verified_at = time.perf_counter()
            if isinstance(original, linear):
                if shape != (original.out_features, original.in_features):
                    raise ValueError(f'{names[name]}: module geometry differs from verified source')
                if pool is None:
                    blobs, row, elapsed, witnesses = encode_cpu(decoded)
                    linear_options = dict(device=device)
                    if witnesses is not None:
                        linear_options['validation_witnesses'] = witnesses
                    replacement = BHELinear(blobs, row, **linear_options)
                    row.update(runtime_kind='bhe1_linear', name=names[name])
                    timing['convert'] = elapsed
                    install(name, parent, attr, replacement, row, timing)
                    del replacement
                else:
                    pending.append((pool.submit(encode_cpu, decoded), name, parent, attr, size,
                                    timing, original, 'linear'))
                    inflight += size
            else:
                while pending:
                    finish_one()
                if shape != (original.num_embeddings, original.embedding_dim):
                    raise ValueError(f'{names[name]}: embedding geometry differs from verified source')
                replacement = bridge._dense_embedding(decoded, original, device)
                row = dict(name=names[name], shape=list(shape), source_sha256=actual,
                           source_bf16_bytes=size, runtime_dense_bytes=size,
                           runtime_kind='bf16_input_embedding', wire_bytes=0)
                timing['convert'] = time.perf_counter() - verified_at
                install(name, parent, attr, replacement, row, timing)
                del replacement
            del decoded, original, context
        while pending:
            finish_one()
        if prefetched_head is not None:
            raise RuntimeError('prefetched BHE1 output head was not encountered in dependency order')
        if descriptor_pipeline is not None:
            descriptor_receipt = descriptor_pipeline.finish()
    finally:
        if pool is not None:
            for item in pending:
                item[0].cancel()
            if prefetched_head is not None:
                prefetched_head[0].cancel()
            pool.shutdown(wait=True, cancel_futures=True)
        if descriptor_pipeline is not None:
            descriptor_pipeline.close()
    preserved = []
    for name, _, mid, _ in rows:
        if mid not in selected:
            parent, attr = bridge._parent_and_attr(model, name)
            preserved.append(getattr(parent, attr).context_weight)
    result = dict(cpu_encode_workers=cpu_encode_workers, cpu_encode_pipeline=cpu_encode_pipeline,
                output_head_prefetched=head_prefetch_enabled,
                peak_source_inflight_bytes=peak_inflight,
                tensor_count=len(receipts),
                linear_count=sum(r['runtime_kind'] == 'bhe1_linear' for r in receipts),
                source_bf16_bytes=sum(r['source_bf16_bytes'] for r in receipts),
                bhe1_wire_bytes=sum(r['wire_bytes'] for r in receipts),
                preserved_context_resident_bytes=bridge._context_resident_bytes(preserved),
                conversion_seconds=time.perf_counter() - started, tensors=receipts,
                runtime_format='bhe1', source='verified_context_transcode')
    if ready_install:
        result['ready_install'] = True
    if validation_witnesses:
        result['validation_witnesses'] = True
    if descriptor_receipt is not None:
        result['descriptor_preparation'] = descriptor_receipt
    return result
