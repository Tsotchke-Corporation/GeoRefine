"""Bind speculative planner costs to the loaded engine's captured CUDA graphs."""
from __future__ import annotations

import math
import statistics

import torch


def cost_from_receipt(receipt, *, row_limit, row_aware=False):
    from .bidec_spec import CostModel

    if type(row_aware) is not bool:
        raise ValueError('row_aware must be a boolean')
    table = {int(k): float(v) for k, v in receipt['trunk_ms'].items()}
    draft = {int(k): float(v) for k, v in receipt['mtp_ms'].items()}
    if not table or min(table) != 1 or max(table) < row_limit:
        raise ValueError('calibration must cover row1 and the planner row limit')
    if any(k < 1 or not math.isfinite(v) or v <= 0 for k,v in table.items()):
        raise ValueError('trunk calibration must contain positive finite timings')
    if any(k < 1 or not math.isfinite(v) or v <= 0 for k,v in draft.items()):
        raise ValueError('MTP calibration must contain positive finite timings')
    if row_aware and not draft:
        raise ValueError('row-aware MTP costs require a nonempty calibrated head table')
    ceiling = min(k for k in table if k >= row_limit)
    if draft and any(k not in draft for k in table if k <= ceiling):
        raise ValueError('MTP calibration must cover every usable trunk bucket')
    base = table[1]
    return CostModel(step_ms=base, mtp_ms=max((v for k,v in draft.items() if k<=ceiling), default=0.0),
                     shape=tuple((k,v/base) for k,v in sorted(table.items())),
                     bucket_rows=tuple(sorted(table)), flat_rows=row_limit,
                     mtp_shape=tuple((k,v) for k,v in sorted(draft.items()) if k<=ceiling)
                     if row_aware and draft else None)


def calibrate_engine(decoder, *, repeats=5, include_uncaptured=False):
    """Startup only: padding graphs, no live request state or performance claim."""
    if type(repeats) is not int or not 3 <= repeats <= 20:
        raise ValueError('calibration repeats must be 3..20')
    if type(include_uncaptured) is not bool:
        raise ValueError('include_uncaptured must be a bool')
    if not decoder.graphs:
        raise ValueError('capture the engine before calibration')
    trunk, draft = {}, {}
    if not include_uncaptured:
        with torch.cuda.device(decoder.dev):
            for bucket, graph in sorted(decoder.graphs.items()):
                decoder._set_padding_only(bucket)
                for target, current in ((trunk, graph), (draft, decoder.graphs_m.get(bucket))):
                    if current is None:
                        continue
                    current.replay()
                    samples = []
                    for _ in range(repeats):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        current.replay()
                        end.record()
                        end.synchronize()
                        samples.append(float(begin.elapsed_time(end)))
                    target[bucket] = statistics.median(samples)
        return dict(schema='georefine.spec_cost.v1', scope='loaded model, all-padding graph CUDA-event medians',
                    context_latency_qualified=False, mtp_cost='conservative maximum usable batch cost per pass',
                    samples_per_bucket=repeats, trunk_ms=trunk, mtp_ms=draft)

    trunk_modes, draft_modes = {}, {}
    buckets = (tuple(sorted(decoder.buckets)) if include_uncaptured
               else tuple(sorted(decoder.graphs)))
    mtp_graphs = getattr(decoder, 'graphs_m', {}) or {}
    has_mtp = bool(mtp_graphs) or bool(getattr(decoder, 'spec', False))

    def measure(target, graph, eager, bucket, modes):
        mode = 'graph' if graph is not None else 'eager'
        if graph is None and not include_uncaptured:
            return
        decoder._set_padding_only(bucket)
        (graph.replay() if graph is not None else eager())
        samples = []
        for _ in range(repeats):
            decoder._set_padding_only(bucket)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            (graph.replay() if graph is not None else eager())
            end.record()
            end.synchronize()
            samples.append(float(begin.elapsed_time(end)))
        target[bucket] = statistics.median(samples)
        modes[str(bucket)] = mode

    with torch.cuda.device(decoder.dev):
        for bucket in buckets:
            graph = decoder.graphs.get(bucket)
            measure(trunk, graph, lambda bucket=bucket: decoder._step(bucket), bucket, trunk_modes)
            if has_mtp:
                measure(draft, mtp_graphs.get(bucket),
                        lambda bucket=bucket: decoder._mtp_step(bucket), bucket, draft_modes)
    receipt = dict(schema='georefine.spec_cost.v1',
                scope='loaded model, all-padding graph/eager CUDA-event medians',
                context_latency_qualified=False, mtp_cost='conservative maximum usable batch cost per pass',
                samples_per_bucket=repeats, trunk_ms=trunk, mtp_ms=draft)
    if include_uncaptured:
        receipt.update(trunk_modes=trunk_modes, mtp_modes=draft_modes,
                       calibration_include_uncaptured=True)
    return receipt
