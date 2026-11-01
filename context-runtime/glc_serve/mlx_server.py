"""Opt-in verified MLX text serving through the existing chat/SSE HTTP interface.

Run ``python -m glc_serve.mlx_server --help``. Defaults remain serialized text.
Explicit --multimodal enables images; --fused-batching enables concurrent request
execution. Neither option advertises GPU qualification.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
import uuid
import numpy as np


class Session:
    def __init__(self, runtime, tokenizer, eos_ids, receipt, *, device, processor=None):
        self.memory_monitor=None
        self.qualification_identity={}
        self._qualification_stats_lock=threading.Lock()
        self.qualification_counters={'events':0,'drafts_proposed':0,'drafts_accepted':0}
        self.runtime=runtime
        self.exact_digest=runtime.exact_digest
        self.service=None
        if runtime.fused_batching:
            from .mlx_service import Service
            self.service=Service(runtime,device=device)
        self.tokenizer=tokenizer
        self.processor=processor
        self.supports_prepared_inputs=True
        self.supports_ignore_eos=True
        self._processor_lock=threading.Lock()
        self.eos_ids=set(eos_ids)
        self.meta=dict(receipt,backend='mlx',dynamic_batching=runtime.fused_batching,serving_qualified=False)
        if self.exact_digest:self.meta['exact_digest']=True
        self.engine=self
        self.device=device

    def prepare(self, requests):
        from .context_fast_server import FastServerError
        if len(requests)!=1:
            raise FastServerError('MLX HTTP adapter accepts one request at a time')
        req=requests[0]
        with self._processor_lock:
            if req.images:
                if self.processor is None:
                    raise FastServerError('vision was not enabled when loading this model')
                text=self.processor.apply_chat_template(req.messages,tokenize=False,
                    add_generation_prompt=True,**(req.chat_template_kwargs or {}))
                encoded=dict(self.processor(text=[text],images=req.images,return_tensors='np',
                                            return_mm_token_type_ids=True))
                required={'input_ids','pixel_values','image_grid_thw','mm_token_type_ids'}
                if not required<=set(encoded):raise FastServerError('processor did not return complete image inputs')
                ids=encoded['input_ids'][0].tolist()
                visual={name:encoded[name] for name in required-{'input_ids'}}
            else:
                ids=self.tokenizer.apply_chat_template(req.messages,tokenize=True,
                    add_generation_prompt=True,**(req.chat_template_kwargs or {}))
                visual=None
        if not isinstance(ids,list) or not ids:
            raise FastServerError('chat template returned no token IDs')
        req.prompt_tokens=len(ids)
        return {'input_ids':np.asarray([ids],dtype=np.int64),'visual_inputs':visual}

    def prefill(self, *, token_ids=None, prepared_inputs=None):
        # Admission needs max_tokens, supplied by generate. No cache is allocated here.
        if prepared_inputs is not None:
            return {'token_ids':prepared_inputs['input_ids'][0].tolist(),
                    'visual_inputs':prepared_inputs.get('visual_inputs')}
        return tuple(token_ids)

    def generate(self, state, *, max_tokens, stop_ids, temperature, top_p, spec_k, with_digests=False):
        if temperature!=0 or top_p!=1 or spec_k not in (None,self.runtime.spec_k):
            raise ValueError('unsupported MLX generation settings')
        if with_digests and not self.exact_digest:raise ValueError('exact digest mode was not enabled')
        def output(event):
            if self.qualification_identity:
                with self._qualification_stats_lock:
                    self.qualification_counters['events']+=1
                    self.qualification_counters['drafts_proposed']+=event.get('proposed',0)
                    self.qualification_counters['drafts_accepted']+=event.get('accepted',0)
            if not with_digests:return iter(event['tokens'])
            digests=event.get('digests')
            if not isinstance(digests,list) or len(digests)!=len(event['tokens']):
                raise RuntimeError('target event lacks complete logit digests')
            return iter(zip(event['tokens'],digests))
        prompt=state['token_ids'] if isinstance(state,dict) else list(state)
        visual=state.get('visual_inputs') if isinstance(state,dict) else None
        request_id=uuid.uuid4().hex
        if self.service is not None:
            try:
                ticket=self.service.submit(request_id,prompt,max_tokens=max_tokens,
                                           stop_tokens=stop_ids,temperature=temperature,visual_inputs=visual)
            except (ValueError,MemoryError) as exc:
                from .context_fast_server import FastServerError
                raise FastServerError(str(exc),429 if isinstance(exc,MemoryError) else 400) from exc
            yield_events=self.service.events(request_id,ticket)
            try:
                for event in yield_events:yield from output(event)
            finally:yield_events.close()
            return
        import mlx.core as mx
        with mx.stream(mx.gpu if self.device=='metal' else mx.cpu):
            try:
                self.runtime.submit(request_id,prompt,max_tokens=max_tokens,
                                    stop_tokens=stop_ids,temperature=temperature,visual_inputs=visual)
            except (ValueError,MemoryError) as exc:
                from .context_fast_server import FastServerError
                raise FastServerError(str(exc),429 if isinstance(exc,MemoryError) else 400) from exc
            try:
                while request_id in self.runtime.requests:
                    for event in self.runtime.step():
                        if event['id']!=request_id:
                            raise RuntimeError('serialized HTTP session contains another request')
                        if 'error' in event:
                            raise RuntimeError(event['error'])
                        yield from output(event)
            finally:
                self.runtime.cancel(request_id)


    def memory_receipt(self):
        from .context_fast_server import FastServerError
        if self.memory_monitor is None:raise FastServerError('memory instrumentation is disabled',409)
        receipt=self.memory_monitor.snapshot()
        if self.qualification_identity:
            with self._qualification_stats_lock:
                receipt['serving_counters']=dict(self.qualification_counters)
        for key in ('frame_id','protocol_sha256','qualification_role'):
            if key in self.qualification_identity:receipt[key]=self.qualification_identity[key]
        return receipt

    def close(self):
        try:
            if self.service is not None:self.service.close()
        finally:
            if self.memory_monitor is not None:self.memory_monitor.stop()


class ConcurrentLock:
    """FastServerState's outer serialization is replaced by the model worker."""
    def acquire(self):return True
    def release(self):pass


def read_panels(manifest_path):
    if manifest_path is None:return None
    path=Path(manifest_path).resolve()
    data=json.loads(path.read_text())
    if not isinstance(data,dict):raise ValueError('panel inventory must map module names to pinned files')
    result={}
    for module,files in data.items():
        if not isinstance(module,str) or not module or not isinstance(files,list) or not files:
            raise ValueError('invalid module panel inventory')
        panels=[]
        for entry in files:
            if not isinstance(entry,dict) or set(entry)!={'path','sha256'}:
                raise ValueError('each panel needs path and sha256')
            source=(path.parent/entry['path']).resolve()
            if not source.is_relative_to(path.parent):raise ValueError('panel path escapes inventory directory')
            blob=source.read_bytes()
            if hashlib.sha256(blob).hexdigest()!=entry['sha256']:raise ValueError('panel SHA mismatch')
            panels.append(blob)
        result[module]=panels
    return result


def prepare_typed_metadata(package, cache_dir):
    """Verify and materialize typed package assets outside its immutable root.

    Typed packages store tokenizer/config assets under ``assets/`` rather than
    the historical BCTX archive.  The MLX HTTP adapter still needs a normal
    Hugging Face directory, so copy only the manifest-pinned assets to a
    content-addressed cache.  The package itself is never modified.
    """
    from georefine_context_runtime.typed_context_package import (
        TypedContextResolver, _safe_rel,
    )

    def reject_lexical_symlinks(path, *, label):
        """Reject symlink components before any path resolution or creation."""
        lexical = Path(path).absolute()
        current = Path(lexical.anchor)
        for component in lexical.parts[1:]:
            current /= component
            if current.is_symlink():
                raise ValueError(f'{label} contains a symlink: {current}')

    reject_lexical_symlinks(package, label='typed metadata package')
    reject_lexical_symlinks(cache_dir, label='typed metadata cache')
    package = Path(package).resolve()
    cache_dir = Path(cache_dir).resolve()
    if cache_dir == package or package in cache_dir.parents:
        raise ValueError('typed metadata cache must be outside the immutable package')
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise ValueError('typed metadata cache must be a regular directory')
    with TypedContextResolver(package) as resolver:
        manifest_sha = resolver.manifest_sha256
        assets = resolver.manifest.get('assets')
        if not isinstance(assets, (tuple, list)) or not assets:
            raise ValueError('typed package has no metadata assets')
        destination = cache_dir / manifest_sha
        if destination.is_symlink():
            raise ValueError('typed metadata cache destination is invalid')
        expected = {}
        for asset in assets:
            if not isinstance(asset, Mapping):
                raise ValueError('invalid typed metadata asset')
            source_rel = _safe_rel(asset.get('source_path'), label='asset source path')
            _safe_rel(asset.get('package_path'), label='asset package path')
            if source_rel.as_posix() in expected:
                raise ValueError(f'duplicate typed metadata asset: {source_rel}')
            raw = resolver.read_asset(source_rel.as_posix())
            if (len(raw) != asset.get('bytes') or
                    hashlib.sha256(raw).hexdigest() != asset.get('sha256')):
                raise ValueError(f"typed metadata asset integrity mismatch: {source_rel}")
            expected[source_rel.as_posix()] = (raw, asset['bytes'], asset['sha256'])
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError('typed metadata cache destination is invalid')
            if any(path.is_symlink() for path in destination.rglob('*')):
                raise ValueError('typed metadata cache contains a symlink')
            present = {p.relative_to(destination).as_posix(): p
                       for p in destination.rglob('*') if p.is_file()}
            if set(present) != set(expected):
                raise ValueError('typed metadata cache inventory mismatch')
            for rel, (_, size, digest) in expected.items():
                payload = present[rel].read_bytes()
                if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError(f'typed metadata cache integrity mismatch: {rel}')
            return destination
        staging = Path(tempfile.mkdtemp(prefix=manifest_sha + '.partial-', dir=cache_dir))
        try:
            for rel, (raw, _, _) in expected.items():
                target = staging.joinpath(*Path(rel).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination


def _verify_qualification_runtime(runtime,args):
    actual=(runtime.spec_k,runtime.fused_batching,runtime.max_batch_rows)
    configured=(args.spec_k,args.fused_batching,args.max_batch_rows)
    if any(type(value) is not type(expected) or value!=expected
           for value,expected in zip(actual,configured)):
        raise ValueError('qualification runtime identity mismatch')


def load_session(args):
    identity={}
    if args.qualification_protocol:
        from .qualification_identity import bind_protocol
        identity=bind_protocol(args.qualification_protocol,role=args.qualification_role,
            package=args.package,wheel=args.qualification_wheel,exact_digest=args.exact_digest,
            max_requests=args.max_requests,device=args.device,metal_operator=args.metal_operator,
            backend='mlx',spec_k=args.spec_k,fused_batching=args.fused_batching,
            max_batch_rows=args.max_batch_rows,panels=args.panels)
    import mlx.core as mx
    from transformers import AutoTokenizer
    from .mlx_engine import load_engine
    if args.device=='metal':
        if not mx.metal.is_available():raise RuntimeError('Metal is unavailable')
        mx.set_default_device(mx.gpu)
    else:mx.set_default_device(mx.cpu)
    if args.panels and args.device!='metal':raise ValueError('BHE Metal panels require --device metal')
    package=Path(args.package)
    package_format = getattr(args, 'package_format', 'predictive')
    from .qualification_identity import package_manifest_path
    manifest_path = package_manifest_path(package)
    manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if package_format == 'typed':
        metadata=prepare_typed_metadata(package,package.parent/'.scratch/mlx-http-metadata')
    elif package_format == 'safetensors':
        metadata=package.resolve()
    else:
        from georefine_context_runtime.bitexact_predictive_serving import prepare_metadata
        metadata=prepare_metadata(package,package.parent/'.scratch/mlx-http-metadata')
    tokenizer=AutoTokenizer.from_pretrained(metadata,local_files_only=True,trust_remote_code=False)
    exclusions=json.loads(Path(args.exclusions).read_text()) if args.exclusions else []
    if not isinstance(exclusions,list) or any(not isinstance(n,str) for n in exclusions):
        raise ValueError('exclusions must be an explicit list of source tensor names')
    runtime,receipt=load_engine(package,model_budget_bytes=args.model_budget_bytes,
        cache_budget_bytes=args.cache_budget_bytes,spec_k=args.spec_k,
        allowed_unused=exclusions,compressed_linears=read_panels(args.panels),
        source_working_budget_bytes=args.source_working_budget_bytes,
        allow_large_model=args.allow_large_model,
        fused_batching=args.fused_batching,max_requests=args.max_requests,max_batch_rows=args.max_batch_rows,
        multimodal=args.multimodal,vision_working_budget_bytes=args.vision_working_budget_bytes,
        prefill_chunk_rows=args.prefill_chunk_rows,exact_digest=args.exact_digest,metal_operator=args.metal_operator,
        package_format=package_format)
    if receipt['model']['manifest_sha256']!=manifest_sha:
        raise ValueError('tokenizer metadata and model package identities differ')
    if args.max_context>runtime.model.args.max_position_embeddings:
        raise ValueError('max_context exceeds model configuration')
    processor=None
    if args.multimodal:
        from transformers import Qwen3VLProcessor
        processor=Qwen3VLProcessor.from_pretrained(metadata,tokenizer=tokenizer,
            local_files_only=True,trust_remote_code=False)
    generation=metadata/'generation_config.json'
    eos=json.loads(generation.read_text()).get('eos_token_id',tokenizer.eos_token_id) if generation.exists() else tokenizer.eos_token_id
    eos=[] if eos is None else eos if isinstance(eos,list) else [eos]
    if any(type(t) is not int or not 0<=t<runtime.model.args.vocab_size for t in eos):
        raise ValueError('invalid configured EOS token IDs')
    session=Session(runtime,tokenizer,eos,receipt,device=args.device,processor=processor)
    if args.qualification_protocol:
        try:
            from .qualification_identity import bind_protocol
            after=bind_protocol(args.qualification_protocol,role=args.qualification_role,
                package=package,wheel=args.qualification_wheel,exact_digest=args.exact_digest,
                max_requests=args.max_requests,device=args.device,metal_operator=args.metal_operator,
                backend='mlx',spec_k=args.spec_k,fused_batching=args.fused_batching,
                max_batch_rows=args.max_batch_rows,panels=args.panels)
            _verify_qualification_runtime(runtime,args)
            if after!=identity:raise ValueError('qualification identity changed during startup')
            identity['compressed_descriptor_count']=sum(
                row.get('kind','').startswith('bhe1') for row in receipt['model']['tensors'])
            identity['served_parameter_count']=sum(math.prod(row['shape'])
                for section in (receipt['model'],receipt.get('mtp')) if section
                for row in section['tensors'])
            if args.metal_operator in ('fused','fused_auto') and identity['compressed_descriptor_count']==0:
                raise ValueError('fused qualification runtime has no compressed linear operators')
            session.qualification_identity=identity
        except BaseException:
            session.close();raise
    return session


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True)
    p.add_argument('--package-format',choices=['predictive','typed','safetensors'],default='predictive')
    p.add_argument('--exclusions',help='JSON array explicitly naming every omitted vision/MTP source tensor')
    p.add_argument('--panels',help='JSON module -> ordered path/SHA256 BHE panel inventory')
    p.add_argument('--memory-metrics',action='store_true')
    p.add_argument('--memory-interval-s',type=float,default=.1)
    p.add_argument('--qualification-protocol')
    p.add_argument('--qualification-role',choices=['candidate','reference'],default='candidate')
    p.add_argument('--qualification-wheel')
    p.add_argument('--metal-operator',choices=['decode','fused','fused_auto'],default='decode')
    p.add_argument('--device',choices=['metal','cpu'],default='metal')
    p.add_argument('--model-budget-bytes',type=int,required=True)
    p.add_argument('--cache-budget-bytes',type=int,required=True)
    p.add_argument('--source-working-budget-bytes',type=int,default=512*1024**2)
    p.add_argument('--allow-large-model',action='store_true')
    p.add_argument('--exact-digest',action='store_true',help='diagnostic target-logit hashes; excludes performance qualification')
    p.add_argument('--multimodal',action='store_true')
    p.add_argument('--vision-working-budget-bytes',type=int,default=256*1024**2)
    p.add_argument('--fused-batching',action='store_true')
    p.add_argument('--max-requests',type=int,default=1)
    p.add_argument('--prefill-chunk-rows',type=int)
    p.add_argument('--max-batch-rows',type=int,default=128)
    p.add_argument('--spec-k',type=int,choices=range(8),default=0)
    p.add_argument('--max-context',type=int,default=32768)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--model-id',default='context-mlx')
    args=p.parse_args(argv)
    if min(args.model_budget_bytes,args.cache_budget_bytes,args.source_working_budget_bytes,args.vision_working_budget_bytes,args.max_context)<=0:
        p.error('budgets and context must be positive')
    if args.max_requests<1 or not args.spec_k+1<=args.max_batch_rows<=512:
        p.error('invalid request or batch row capacity')
    if args.prefill_chunk_rows is not None and not 1<=args.prefill_chunk_rows<=args.max_batch_rows:
        p.error('prefill chunk rows must fit max-batch-rows')
    if args.max_requests>1 and not args.fused_batching:
        p.error('multiple HTTP requests require --fused-batching')
    if not math.isfinite(args.memory_interval_s) or args.memory_interval_s<=0:
        p.error('memory sampling interval must be finite and positive')
    if args.qualification_protocol and not args.memory_metrics:
        p.error('qualification protocol requires --memory-metrics')
    if args.metal_operator=='fused' and not args.panels:
        p.error('--metal-operator fused requires --panels')
    if args.metal_operator=='fused_auto' and args.panels:
        p.error('--metal-operator fused_auto derives panels from the verified package; omit --panels')
    if args.metal_operator in ('fused','fused_auto') and args.device!='metal':
        p.error('fused Metal operators require --device metal')
    return args


def main(argv=None):
    from .context_fast_server import FastServerState,serve
    args=parse(argv)
    class MLXServerState(FastServerState):
        def health(self):
            code,body=super().health()
            body['dynamic_batching']=args.fused_batching
            return code,body
        def receipt(self):
            value=super().receipt()
            value.update(dynamic_batching=args.fused_batching,batch_size=args.max_requests,
                         multimodal=args.multimodal,max_batch_rows=args.max_batch_rows)
            if self.session is not None:value.update(self.session.qualification_identity)
            return value
    state=MLXServerState(model_id=args.model_id,max_context=args.max_context,
                          source_kind='verified_mlx',source=args.package)
    server=serve(state,args.host,args.port)
    listener=threading.Thread(target=server.serve_forever,daemon=True);listener.start()
    monitor=None
    try:
        try:
            if args.memory_metrics:
                from .runtime_memory import MemoryMonitor
                monitor=MemoryMonitor('mlx' if args.device=='metal' else 'cpu',args.memory_interval_s).start()
            session=load_session(args)
            session.memory_monitor=monitor
            state.attach(session)
            if args.fused_batching:state.request_lock=ConcurrentLock()
        except Exception as exc:
            state.fail_startup('verified_mlx_load',str(exc))
            print(json.dumps({'status':'startup_failed','error':str(exc)}),flush=True)
            return 2
        print(json.dumps({'status':'ready','receipt':state.receipt()}),flush=True)
        listener.join()
    except KeyboardInterrupt:pass
    finally:
        server.shutdown();server.server_close()
        if state.session is not None:state.session.close()
        elif monitor is not None:monitor.stop()
    return 0


if __name__=='__main__':raise SystemExit(main())
