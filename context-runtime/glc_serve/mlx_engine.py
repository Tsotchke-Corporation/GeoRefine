"""Explicit MLX greedy serving with request-local target and MTP state.

Default requests run serially. The explicit fused_batching option groups target
and MTP calls across requests, with bounded rows and transactional cache merges.
The byte budget covers cache arrays, including speculative copies, not allocator
reserve. Full-model memory and performance require separate qualification.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from .mlx_speculative import cache_array_bytes, verify_greedy
from glc_loader.metal.status_guard import MetalStatusGuard, scope as metal_status_scope


@dataclass
class Request:
    cache: list
    mtp_cache: list
    hidden: object
    current: int
    remaining: int
    stop_tokens: frozenset
    reserved_bytes: int
    position: int
    rope_delta: int = 0
    current_digest: dict | None = None
    tokens: list = field(default_factory=list)
    done: bool = False


class Engine:
    def __init__(self, model, mtp=None, *, cache_budget_bytes, max_requests=1, spec_k=0, fused_batching=False, max_batch_rows=128, vision_working_budget_bytes=256*1024**2, prefill_chunk_rows=None, exact_digest=False, defer_metal_status=False):
        if type(spec_k) is not int or not 0 <= spec_k <= 7:
            raise ValueError('spec_k must be 0..7')
        if spec_k and mtp is None:
            raise ValueError('speculation requires a verified MTP head')
        if type(cache_budget_bytes) is not int or cache_budget_bytes <= 0:
            raise ValueError('cache budget must be positive')
        if type(max_requests) is not int or max_requests <= 0:
            raise ValueError('max_requests must be positive')
        if type(fused_batching) is not bool or type(max_batch_rows) is not int or max_batch_rows < spec_k+1:
            raise ValueError('invalid fused batch row capacity')
        if type(vision_working_budget_bytes) is not int or vision_working_budget_bytes<=0:
            raise ValueError('vision working budget must be positive')
        if prefill_chunk_rows is not None and (type(prefill_chunk_rows) is not int or not 1<=prefill_chunk_rows<=max_batch_rows):
            raise ValueError('prefill chunk rows must fit the operator row capacity')
        if type(exact_digest) is not bool:raise ValueError('exact_digest must be boolean')
        self.exact_digest=exact_digest
        if type(defer_metal_status) is not bool:
            raise ValueError('defer_metal_status must be boolean')
        self.defer_metal_status = defer_metal_status
        self._metal_status_guard = MetalStatusGuard() if defer_metal_status else None
        self.prefill_chunk_rows=prefill_chunk_rows
        self.vision_working_budget_bytes=vision_working_budget_bytes
        self.fused_batching, self.max_batch_rows = fused_batching, max_batch_rows
        self.model, self.mtp = model, mtp
        self.cache_budget_bytes = cache_budget_bytes
        self.max_requests, self.spec_k = max_requests, spec_k
        self.requests = {}
        self._lock = threading.RLock()

    def _bytes(self, extra=()):
        return cache_array_bytes([[(r.cache,r.mtp_cache,r.hidden) for r in self.requests.values()],extra])

    def _admit(self, extra=()):
        if self._bytes(extra) > self.cache_budget_bytes:
            raise MemoryError('request caches exceed admitted array byte budget')

    def submit(self, request_id, prompt, *, max_tokens, stop_tokens=(), temperature=0, visual_inputs=None):
        with self._metal_scope():
            return self._submit(request_id, prompt, max_tokens=max_tokens,
                                stop_tokens=stop_tokens, temperature=temperature,
                                visual_inputs=visual_inputs)

    def _submit(self, request_id, prompt, *, max_tokens, stop_tokens=(), temperature=0, visual_inputs=None):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        with self._lock:
            if not isinstance(request_id,str) or not request_id or request_id in self.requests:
                raise ValueError('request_id must be unique and nonempty')
            if len(self.requests) >= self.max_requests:
                raise MemoryError('request capacity exhausted')
            if temperature != 0:
                raise ValueError('only greedy temperature=0 is supported')
            if not prompt or any(type(t) is not int or not 0 <= t < self.model.args.vocab_size for t in prompt):
                raise ValueError('prompt contains invalid token IDs')
            if type(max_tokens) is not int or max_tokens < 1:
                raise ValueError('max_tokens must be positive')
            if len(prompt)+max_tokens > self.model.args.max_position_embeddings:
                raise ValueError('request exceeds configured context length')
            stop=frozenset(stop_tokens)
            if any(type(t) is not int or not 0 <= t < self.model.args.vocab_size for t in stop):
                raise ValueError('invalid stop token')
            reservation=3*self.model.cache_reservation_bytes(len(prompt)+max_tokens, mtp=self.mtp is not None)
            if sum(r.reserved_bytes for r in self.requests.values())+reservation>self.cache_budget_bytes:
                raise MemoryError('request cache reservation exceeds budget before prefill')
            # Fresh request remains unpublished until prefill and memory checks succeed.
            cache=self.model.make_cache()
            x=mx.array(prompt,dtype=mx.int32)[None,:]
            embeds=None;position_ids=None;delta=0
            if visual_inputs is not None:
                working=self.model.vision_working_bytes(visual_inputs)+len(prompt)*self.model.args.hidden_size*4
                if working>self.vision_working_budget_bytes:
                    raise MemoryError('vision working arrays exceed admitted budget')
                embeds,position_ids,delta=self.model.prepare_visual_inputs(x,visual_inputs)
            mtp_cache=[KVCache()] if self.mtp is not None else []
            last=None
            chunk_rows=self.prefill_chunk_rows or (self.max_batch_rows if self.fused_batching else len(prompt))
            for start in range(0,len(prompt),chunk_rows):
                end=min(start+chunk_rows,len(prompt))
                chunk=x[:,start:end]
                kwargs={}
                if embeds is not None:
                    kwargs=dict(inputs_embeds=embeds[:,start:end],position_ids=position_ids[:,:,start:end])
                hidden=self.model.forward_hidden(chunk,cache,**kwargs)
                if self.mtp is not None:
                    head_start=start if start else 1
                    if end>head_start:
                        previous=mx.concatenate([last,hidden[:,:-1]],axis=1) if start else hidden[:,:-1]
                        next_embeds=self.model.model.embed_tokens(x[:,head_start:end]) if embeds is None else embeds[:,head_start:end]
                        head_kwargs={} if position_ids is None else dict(position_ids=position_ids[:,:,head_start:end])
                        warm=self.mtp(previous,next_embeds,mtp_cache,**head_kwargs)
                        mx.eval(warm)
                last=hidden[:,-1:]
                mx.eval(last)
                self._admit((cache,mtp_cache,last))
            from .mlx_digest import predict
            predicted,initial_digest=predict(self.model,last,exact_digest=self.exact_digest)
            current=predicted[0][0]
            self._admit((cache,mtp_cache,last))
            self.requests[request_id]=Request(cache,mtp_cache,last,current,max_tokens,stop,reservation,len(prompt),delta,
                current_digest=initial_digest[0][0] if initial_digest else None)

    def cancel(self, request_id):
        with self._lock:
            self.requests.pop(request_id,None)

    def step(self):
        with self._metal_scope():
            return self._step()

    @contextmanager
    def _metal_scope(self):
        if self._metal_status_guard is None:
            yield
        else:
            with metal_status_scope(owner=self._metal_status_guard):
                yield

    def _step(self):
        import mlx.core as mx
        events=[]
        with self._lock:
            if self.fused_batching:
                from .mlx_batch import step
                return step(self)
            for request_id in list(self.requests):
                r=self.requests[request_id]
                # The first prediction already exists after prefill. Emit it before
                # consuming it; this also handles EOS and one-token requests.
                if not r.tokens:
                    r.tokens.append(r.current);r.remaining-=1
                    finished=r.current in r.stop_tokens or r.remaining==0
                    event=dict(id=request_id,tokens=[r.current],finished=finished)
                    if self.exact_digest:event['digests']=[r.current_digest]
                    events.append(event)
                    if finished:
                        del self.requests[request_id]
                    continue
                before=None
                try:
                    if self._bytes()+2*cache_array_bytes((r.cache,r.mtp_cache,r.hidden))>self.cache_budget_bytes:
                        raise MemoryError('speculative copies exceed cache budget')
                    before=copy.deepcopy((r.cache,r.mtp_cache,r.hidden))
                    draft_cache=copy.deepcopy(r.mtp_cache)
                    drafts=[]; token=r.current; h=r.hidden
                    for _ in range(min(self.spec_k,r.remaining-1)):
                        embeds=self.model.model.embed_tokens(mx.array([[token]],dtype=mx.int32))
                        kwargs={}
                        if r.rope_delta:
                            from .mlx_qwen import positions
                            kwargs['position_ids']=positions([r.position+r.rope_delta+len(drafts)],1)
                        h=self.mtp(h,embeds,draft_cache,**kwargs)
                        token=int(mx.argmax(self.model.project(h),axis=-1).item())
                        drafts.append(token)
                        if token in r.stop_tokens:break
                    del draft_cache
                    rows=[];deciding_digests=[]
                    def forward(tokens):
                        kwargs={}
                        if r.rope_delta:
                            from .mlx_qwen import positions
                            kwargs['position_ids']=positions([r.position+r.rope_delta],len(tokens))
                        h=self.model.forward_hidden(mx.array([tokens],dtype=mx.int32),r.cache,**kwargs)
                        from .mlx_digest import predict
                        predictions,digests=predict(self.model,h,exact_digest=self.exact_digest)
                        if self.exact_digest and not deciding_digests:deciding_digests.extend(digests[0])
                        rows[:]=[h]
                        return predictions[0]
                    result=verify_greedy(r.cache,r.current,drafts,forward,
                        snapshot_budget_bytes=self.cache_budget_bytes,
                        output_limit=r.remaining,stop_tokens=r.stop_tokens)
                    committed=rows[0]
                    if self.mtp is not None:
                        inputs=[r.current,*result.emitted[:-1]]
                        previous=mx.concatenate([r.hidden,committed[:,:-1]],axis=1)
                        kwargs={}
                        if r.rope_delta:
                            from .mlx_qwen import positions
                            kwargs['position_ids']=positions([r.position+r.rope_delta],len(inputs))
                        warm=self.mtp(previous,self.model.model.embed_tokens(mx.array([inputs],dtype=mx.int32)),r.mtp_cache,**kwargs)
                        mx.eval(warm)
                    r.hidden=committed[:,-1:]
                    self._admit()
                    r.tokens.extend(result.emitted);r.remaining-=len(result.emitted)
                    r.current=result.emitted[-1]
                    r.position+=len(result.emitted)
                    finished=r.current in r.stop_tokens or r.remaining==0
                    event=dict(id=request_id,tokens=list(result.emitted),finished=finished,
                               accepted=result.accepted,proposed=result.proposed)
                    if self.exact_digest:event['digests']=deciding_digests[:len(result.emitted)]
                    events.append(event)
                    if finished:del self.requests[request_id]
                except BaseException as exc:
                    if before is not None:r.cache,r.mtp_cache,r.hidden=before
                    if not isinstance(exc, Exception):
                        raise
                    del self.requests[request_id]
                    events.append(dict(id=request_id,tokens=[],finished=True,error=str(exc)))
        return events


def load_engine(package, *, model_budget_bytes, cache_budget_bytes, spec_k=0,
                max_requests=1, allowed_unused=(), compressed_linears=None,
                source_working_budget_bytes=512*1024**2, allow_large_model=False,
                fused_batching=False, max_batch_rows=128, multimodal=False,
                vision_working_budget_bytes=256*1024**2, prefill_chunk_rows=None, exact_digest=False, metal_operator="decode",
                package_format="predictive", defer_metal_status=False):
    from .predictive_mlx import load_verified_mlx, load_verified_mlx_mtp
    from .mlx_qwen import Model, MTP, bindings
    trunk_unused=set(allowed_unused)
    if spec_k:
        if package_format == 'typed':
            from georefine_context_runtime.typed_context_package import TypedContextResolver
            resolver_context = TypedContextResolver(package, max_source_bytes=source_working_budget_bytes)
        elif package_format == 'safetensors':
            from georefine_context_runtime.safetensors_source import SafetensorsResolver
            resolver_context = SafetensorsResolver(package, max_source_bytes=source_working_budget_bytes)
        else:
            from georefine_context_runtime.bitexact_predictive_package import VerifiedTensorResolver
            resolver_context = VerifiedTensorResolver(package,cache_bytes=0)
        with resolver_context as resolver:
            head_sources={name for name in resolver.tensor_names if name.startswith('mtp.')}
        if not head_sources:raise ValueError('speculation requires original MTP tensors in the verified package')
        # These tensors are consumed by the independently verified head below,
        # not silently discarded from the complete engine.
        trunk_unused.update(head_sources)
    model,receipt=load_verified_mlx(package,model_factory=lambda config:Model(config,with_vision=multimodal),bindings=bindings,
        allowed_unused=tuple(sorted(trunk_unused)),compressed_linears=compressed_linears,
        memory_budget_bytes=model_budget_bytes,source_working_budget_bytes=source_working_budget_bytes,
        allow_large_model=allow_large_model,max_rows=max_batch_rows,metal_operator=metal_operator,
        package_format=package_format)
    head=None;head_receipt=None
    if spec_k:
        used=receipt['resident_dense_bytes']+receipt['resident_compressed_bytes']+receipt['largest_known_transient_bytes']
        remaining=model_budget_bytes-used
        if remaining<=0:raise MemoryError('no admitted memory remains for MTP')
        head,head_receipt=load_verified_mlx_mtp(package,head_factory=MTP,
            bindings=lambda model,config:bindings(model,config,mtp=True),
            memory_budget_bytes=remaining,source_working_budget_bytes=source_working_budget_bytes,
            allow_large_model=allow_large_model,package_format=package_format)
        if (receipt['manifest_sha256'],receipt['config_sha256']) != (head_receipt['manifest_sha256'],head_receipt['config_sha256']):
            raise ValueError('model and MTP package identities differ')
    model.eval()
    if head is not None:head.eval()
    return Engine(model,head,cache_budget_bytes=cache_budget_bytes,max_requests=max_requests,spec_k=spec_k,
                  fused_batching=fused_batching,max_batch_rows=max_batch_rows,
                  vision_working_budget_bytes=vision_working_budget_bytes,
                  prefill_chunk_rows=prefill_chunk_rows or (max_batch_rows if compressed_linears or metal_operator=='fused_auto' else None),
                  exact_digest=exact_digest, defer_metal_status=defer_metal_status),dict(model=receipt,mtp=head_receipt)
