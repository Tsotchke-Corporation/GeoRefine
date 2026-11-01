"""Greedy speculative verification for MLX, including recurrent-state rollback.

Unlike KV-only trimming, snapshot and replay also restores GDN/Mamba state.
This is an explicit adapter building block, not the CUDA scheduler or an MTP
loader. Throughput and real-model state parity require separate qualification.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Sequence


def cache_array_bytes(caches) -> int:
    """Conservative array payload count, deduplicating repeated object references."""
    seen = set()
    def visit(value):
        if value is None or isinstance(value,(str,bytes,int,float,bool)) or id(value) in seen:
            return 0
        seen.add(id(value))
        nbytes = getattr(value,'nbytes',None)
        if isinstance(nbytes,int):
            return nbytes
        if isinstance(value,dict):
            return sum(visit(v) for v in value.values())
        if isinstance(value,(list,tuple)):
            return sum(visit(v) for v in value)
        if hasattr(value,'__dict__'):
            return visit(vars(value))
        raise TypeError(f'unsupported cache state for memory admission: {type(value).__name__}')
    return visit(caches)


@dataclass(frozen=True)
class Verification:
    emitted: tuple[int,...]
    accepted: int
    proposed: int
    replayed_rows: int
    snapshot_array_bytes: int


def verify_greedy(caches: list, current_token: int, drafts: Sequence[int],
                  forward_tokens: Callable[[list[int]],Sequence[int]], *,
                  snapshot_budget_bytes: int, output_limit: int | None = None,
                  stop_tokens=()) -> Verification:
    """Verify once; on rejection restore every cache and replay the accepted prefix.

    ``forward_tokens`` mutates the supplied cache list and returns one argmax per
    input row. Its output must be fully evaluated before it returns. No sampling
    is implied. The final emitted token remains unconsumed, as in ordinary AR.
    MLX implements deepcopy of arrays and uses cache deepcopy in its own server.
    The snapshot budget counts array payloads, not allocator/Python overhead.
    """
    if not isinstance(caches,list):
        raise TypeError('a mutable cache list is required')
    if len(drafts)>7:
        raise ValueError('at most seven speculative draft tokens are supported')
    stop_tokens=set(stop_tokens)
    if any(type(t) is not int or t<0 for t in stop_tokens):
        raise ValueError('stop tokens must be nonnegative integers')
    tokens=[current_token,*drafts]
    if any(type(t) is not int or t<0 for t in tokens):
        raise ValueError('tokens must be nonnegative integers')
    limit=len(tokens) if output_limit is None else output_limit
    if type(limit) is not int or limit<1:
        raise ValueError('output_limit must be a positive integer')
    if type(snapshot_budget_bytes) is not int or snapshot_budget_bytes<0:
        raise ValueError('snapshot budget must be a nonnegative integer')
    needed=cache_array_bytes(caches)
    if needed>snapshot_budget_bytes:
        raise MemoryError(f'speculative cache snapshot needs {needed} bytes, budget {snapshot_budget_bytes}')
    snapshot=copy.deepcopy(caches)
    if cache_array_bytes(snapshot)>snapshot_budget_bytes:
        raise MemoryError('cloned cache exceeds the snapshot budget')
    def forward(inputs):
        predictions=list(forward_tokens(inputs))
        if len(predictions)!=len(inputs) or any(type(t) is not int or t<0 for t in predictions):
            raise RuntimeError('target must return one nonnegative integer argmax per input row')
        return predictions
    try:
        predicted=forward(tokens)
        accepted=0
        while accepted<len(drafts) and drafts[accepted]==predicted[accepted]:
            accepted+=1
        emitted=predicted[:min(accepted+1,limit)]
        for i,token in enumerate(emitted):
            if token in stop_tokens:
                emitted=emitted[:i+1]
                break
        replayed=0
        if len(emitted)!=len(tokens):
            caches[:]=copy.deepcopy(snapshot)
            replayed=len(emitted)
            if forward(tokens[:replayed])!=emitted:
                raise RuntimeError('target predictions changed after rejection replay')
        return Verification(tuple(emitted),min(accepted,len(emitted)),len(drafts),replayed,needed)
    except BaseException:
        caches[:]=snapshot
        raise


def verify_mlx(model, caches, current_token, drafts, *, snapshot_budget_bytes,
               output_limit=None,stop_tokens=()):
    """Explicit single-sequence MLX adapter; importing this module is CPU-safe."""
    import mlx.core as mx
    def forward(tokens):
        logits=model(mx.array(tokens,dtype=mx.int32)[None,:],cache=caches)
        if logits.ndim!=3 or logits.shape[0]!=1 or logits.shape[1]!=len(tokens):
            raise RuntimeError('MLX target returned an incompatible logits shape')
        predictions=mx.argmax(logits[0],axis=-1)
        mx.eval(predictions)
        return predictions.tolist()
    return verify_greedy(caches,current_token,drafts,forward,
                         snapshot_budget_bytes=snapshot_budget_bytes,
                         output_limit=output_limit,stop_tokens=stop_tokens)
