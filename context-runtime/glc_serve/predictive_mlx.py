"""Bounded verified-package loading into an explicitly supplied MLX model factory.

Architecture conversion is explicit: a binding names its source tensors and any
concatenation/transpose. No unverified checkpoint or implicit name guessing is
used. The caller supplies a fresh model whose parameterization matches these
bindings; this loader does not claim support for an unported architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import struct
import functools
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TensorBinding:
    sources: tuple[str,...]
    concatenate_axis: int | None = None
    transpose: tuple[int,...] | None = None
    reshape: tuple[int,...] | None = None


def _flatten(tree,prefix=''):
    if isinstance(tree,dict):
        for key,value in tree.items():
            yield from _flatten(value,f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(tree,(list,tuple)):
        for index,value in enumerate(tree):
            yield from _flatten(value,f'{prefix}.{index}')
    elif hasattr(tree,'shape'):
        yield prefix,tree
    else:
        raise ValueError(f'unsupported model parameter leaf: {prefix}')


def _shape(binding,entries):
    if not isinstance(binding,TensorBinding) or not binding.sources:
        raise ValueError('each binding needs at least one named source tensor')
    shapes=[tuple(entries[name]['shape']) for name in binding.sources]
    shape=shapes[0]
    axis=binding.concatenate_axis
    if len(shapes)>1:
        if type(axis) is not int or not 0<=axis<len(shape):
            raise ValueError('multiple sources require an explicit concatenation axis')
        if any(len(s)!=len(shape) or any(s[i]!=shape[i] for i in range(len(s)) if i!=axis)
               for s in shapes):
            raise ValueError('source concatenation shapes differ')
        shape=tuple(sum(s[i] for s in shapes) if i==axis else shape[i] for i in range(len(shape)))
    elif axis is not None:
        raise ValueError('single-source bindings do not concatenate')
    if binding.transpose is not None:
        if (not isinstance(binding.transpose,tuple) or any(type(i) is not int for i in binding.transpose)
                or sorted(binding.transpose)!=list(range(len(shape)))):
            raise ValueError('invalid transpose permutation')
        shape=tuple(shape[i] for i in binding.transpose)
    if binding.reshape is not None:
        if (not isinstance(binding.reshape,tuple) or not binding.reshape or
                any(type(d) is not int or d<=0 for d in binding.reshape) or
                math.prod(shape)!=math.prod(binding.reshape)):
            raise ValueError('invalid source reshape')
        shape=binding.reshape
    return shape


def _module_at(model,path):
    value=model
    for part in path.split('.'):
        if isinstance(value,(list,tuple)):value=value[int(part)]
        elif isinstance(value,dict):value=value[part]
        else:value=getattr(value,part)
    return value


def _is_mlx_linear(module):
    import mlx.nn as nn
    # A subclass may override the operation; only the known linear contract is accepted.
    return type(module) is nn.Linear


def _replace(model,path,value):
    parts=path.split('.')
    parent=model
    for part in parts[:-1]:
        if isinstance(parent,(list,tuple)):parent=parent[int(part)]
        elif isinstance(parent,dict):parent=parent[part]
        else:parent=getattr(parent,part)
    key=parts[-1]
    if isinstance(parent,list):parent[int(key)]=value
    elif isinstance(parent,dict):parent[key]=value
    else:setattr(parent,key,value)


def _mlx_array(words):
    import mlx.core as mx
    value=mx.view(mx.array(words),mx.bfloat16)
    mx.eval(value)
    return value


def _mlx_typed_array(value, dtype):
    """Materialize original storage bits in an explicitly supported MLX dtype."""
    import mlx.core as mx
    native = {'F32': mx.float32, 'F16': mx.float16, 'BF16': mx.bfloat16,
              'I8': mx.int8, 'U8': mx.uint8, 'I16': mx.int16, 'U16': mx.uint16,
              'I32': mx.int32, 'U32': mx.uint32, 'I64': mx.int64, 'U64': mx.uint64,
              'BOOL': mx.bool_}
    if dtype not in native:
        raise ValueError(f'MLX cannot materialize source dtype {dtype!r} without conversion')
    # Start with unsigned storage words so float payloads (including NaNs) are
    # never interpreted by a numerical conversion on the way to the device.
    bits = value.dtype.itemsize * 8
    storage = np.ascontiguousarray(value).reshape(value.shape).view(np.dtype(f'<u{bits // 8}'))
    result = mx.view(mx.array(storage), native[dtype])
    mx.eval(result)
    return result


def _read_typed_config(resolver, metadata_cache_dir):
    pin = next((row for row in resolver.manifest['assets']
                if row['source_path'] == 'config.json'), None)
    if pin is None:
        raise ValueError('typed package has no original model config')
    from georefine_context_runtime.typed_context_package import read_asset
    raw = read_asset(resolver, 'config.json')
    def unique(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate model config key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique), hashlib.sha256(raw).hexdigest()


def _read_safetensors_config(resolver, metadata_cache_dir):
    del metadata_cache_dir
    raw = resolver.config_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != resolver.config_sha256:
        raise ValueError('original model config differs from pinned source')
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate model config key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique), resolver.config_sha256


def _read_config(resolver,metadata_cache_dir):
    from georefine_context_runtime.bitexact_predictive_serving import prepare_metadata
    directory=prepare_metadata(resolver.package,metadata_cache_dir)
    raw=(directory/'config.json').read_bytes()
    pin=resolver.manifest['metadata_assets']['sidecars/config.json']
    if len(raw)!=pin['bytes'] or hashlib.sha256(raw).hexdigest()!=pin['sha256']:
        raise ValueError('model config differs from verified package metadata')
    def unique(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate model config key')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique),hashlib.sha256(raw).hexdigest()


def _resolve_binding_words(resolver,binding,entries,expected_shape, *, typed=False):
    """Resolve, verify, and assemble one binding while retaining only this tensor."""
    pieces=[];hashes={}
    for source in binding.sources:
        words=resolver.numpy_view(source) if typed else resolver.resolve(source)
        entry=entries[source]
        if ((not typed and words.dtype!=np.dtype('<u2'))
                or tuple(words.shape)!=tuple(entry['shape'])):
            raise ValueError(f'verified source shape/dtype mismatch: {source}')
        words=(np.ascontiguousarray(words).reshape(words.shape) if typed
               else np.ascontiguousarray(words,dtype='<u2'))
        digest=hashlib.sha256(memoryview(words).cast('B')).hexdigest()
        if digest!=entry['source_sha256']:raise ValueError(f'source SHA mismatch: {source}')
        hashes[source]=digest;pieces.append(words)
    if binding.transpose is not None:
        pieces=[piece.transpose(binding.transpose) for piece in pieces]
    if len(pieces)>1:
        axis=binding.concatenate_axis
        if binding.transpose is not None:axis=binding.transpose.index(axis)
        words=np.empty(expected_shape,dtype=pieces[0].dtype,order='C')
        np.concatenate(pieces,axis=axis,out=words)
    else:
        words=np.ascontiguousarray(pieces[0]).reshape(pieces[0].shape)
    del pieces
    if binding.reshape is not None:words=words.reshape(binding.reshape)
    digest=hashlib.sha256(memoryview(words).cast('B')).hexdigest()
    return words,hashes,digest


def _encode_auto_panels(words, *, packet_module, fused_operator, max_rows,
                        source_budget_bytes, source_piece_bytes=0, retain_packets=False):
    """Encode one tensor in 64-row panels, retaining packet hashes only."""
    hashes=[];panel_bytes=[];largest_peak=0;resident_sizes={};packets=[];retained_bytes=0
    padded_n=((words.shape[0]+63)//64)*64
    workspace_bytes=4*64*words.shape[1]*2
    if source_piece_bytes+words.nbytes+workspace_bytes+64*1024**2>source_budget_bytes:
        raise MemoryError('automatic source working buffers and census exceed budget')
    row_labels,col_labels=packet_module.tiled.codec._labels(words,16,4)
    # Bounded vectorized census; never iterate over every matrix word in Python.
    frequencies=np.zeros((64,1024),dtype=np.uint64)
    step=max(1,1_000_000//words.shape[1])
    for lo in range(0,len(words),step):
        symbols=(words[lo:lo+step].astype(np.uint32)>>5)&1023
        contexts=row_labels[lo:lo+step,None].astype(np.uint32)*4+col_labels[None,:]
        keys=(contexts*1024+symbols).ravel()
        frequencies+=np.bincount(keys,minlength=65536).reshape(64,1024).astype(np.uint64)
    padding=padded_n-len(words)
    if padding:
        for cls in range(4):frequencies[int(row_labels[0])*4+cls,0]+=padding*int(np.count_nonzero(col_labels==cls))
    offsets=[0];entries=bytearray()
    for counts in frequencies:
        values={int(i):int(counts[i]) for i in np.flatnonzero(counts)}
        table=packet_module.tiled._normalise(values) if values else {}
        for symbol,(cum,freq) in table.items():entries.extend(struct.pack('<HI',symbol,cum+freq))
        offsets.append(len(entries)//6)
    cdf=struct.pack('<'+'I'*len(offsets),*offsets)+entries
    table_cache={}
    for start in range(0,padded_n,64):
        panel=np.zeros((64,words.shape[1]),dtype='<u2')
        count=max(0,min(64,words.shape[0]-start))
        if count:panel[:count]=words[start:start+count]
        panel_rows=np.empty(64,dtype=np.uint8)
        if count:panel_rows[:count]=row_labels[start:start+count]
        if count<64:panel_rows[count:]=row_labels[0]
        # packet.encode and fused host conversion both have bounded panel-local workspaces.
        blob=packet_module.encode(panel,row_classes=16,col_classes=4,row_labels=panel_rows,
            col_labels=col_labels,sparse_cdf=cdf,experimental_large_geometry=True,
            shared_metadata=True,_table_cache=table_cache)
        packet_bytes=len(blob)
        arrays,_=fused_operator.host_inputs([blob],_table_cache=table_cache)
        one_panel_arrays=sum(value.nbytes for value in arrays.values() if isinstance(value,np.ndarray))
        workspace=workspace_bytes+one_panel_arrays
        peak=source_piece_bytes+words.nbytes+packet_bytes+retained_bytes+workspace
        if peak>source_budget_bytes:
            raise MemoryError('automatic fused panel and source working buffers exceed budget')
        largest_peak=max(largest_peak,peak)
        hashes.append(hashlib.sha256(blob).hexdigest());panel_bytes.append(packet_bytes)
        if retain_packets:
            packets.append(blob);retained_bytes+=packet_bytes
        for _,_,key in fused_operator.metal._INPUTS[:-1]:
            if key in ('tables','context_dir','positions','counts'):
                resident_sizes[key]=max(resident_sizes.get(key,0),arrays[key].nbytes)
            else:resident_sizes[key]=resident_sizes.get(key,0)+arrays[key].nbytes
        if not retain_packets:del blob
        del arrays,panel
    return hashes,panel_bytes,largest_peak,resident_sizes,packets


def _auto_fused_ledger(resident_sizes,*,shape,max_rows,fused_operator):
    """Conservative fused resident ledger from one-at-a-time panel metadata."""
    class Sized:
        def __init__(self,nbytes):self.nbytes=nbytes
    arrays={key:Sized(size) for key,size in resident_sizes.items()}
    arrays.update(N=((shape[0]+63)//64)*64,K=shape[1],tiles=(shape[0]+63)//64)
    return fused_operator.allocation_ledger(arrays,max_rows=max_rows)


def _auto_native_codec(function):
    @functools.wraps(function)
    def call(*args,**kwargs):
        if kwargs.get('metal_operator')=='fused_auto':
            from georefine_context_runtime.context_exponent_packet import native_cpu_backend
            with native_cpu_backend():return function(*args,**kwargs)
        return function(*args,**kwargs)
    return call


@_auto_native_codec
def load_verified_mlx(package, *, model_factory, bindings=None, compressed_linears=None,
                      allowed_unused=(), memory_budget_bytes=512*1024**2,
                      source_working_budget_bytes=512*1024**2, max_rows=128,
                      metadata_cache_dir=None, resolver_factory=None,
                      config_reader=None, array_factory=None, linear_factory=None, linear_validator=None,
                      source_scope="model", allow_large_model=False, metal_operator="decode",
                      package_format="predictive"):
    """Build a fresh model, reconcile its entire tensor set, then load one binding at a time.

    ``compressed_linears`` maps a module path to its pre-encoded BHE panels.
    Other parameters remain dense BF16. Unused source names must be enumerated,
    so text-only or MTP-off operation cannot silently discard unknown tensors.
    The memory bound covers known tensor payloads; allocator/kernel overhead is
    separately reported. No dense checkpoint is written.
    """
    from georefine_context_runtime.bitexact_predictive_package import VerifiedTensorResolver
    from georefine_context_runtime import context_exponent_metal as metal

    if package_format not in ('predictive', 'typed', 'safetensors'):
        raise ValueError('package_format must be predictive, typed, or safetensors')
    typed = package_format in ('typed', 'safetensors')
    custom_array_factory = array_factory is not None

    for name,value in [('memory_budget_bytes',memory_budget_bytes),
                       ('source_working_budget_bytes',source_working_budget_bytes)]:
        if type(value) is not int or value<=0:raise ValueError(f'{name} must be positive')
    if memory_budget_bytes>900*1024**2 and not allow_large_model:
        raise ValueError('experimental Metal loader budget is capped at 900 MiB')
    # The aggregate model allowance may exceed a single native plan's ceiling.
    linear_allocation_budget = min(memory_budget_bytes, 900 * 1024**2)
    if type(max_rows) is not int or not 1<=max_rows<=512:
        raise ValueError('max_rows must be 1..512')
    if source_scope not in ('model','mtp'):
        raise ValueError('source_scope must be model or mtp')
    if source_scope=='mtp' and allowed_unused:
        raise ValueError('MTP scope cannot exclude MTP tensors')
    compressed=dict(compressed_linears or {})
    if metal_operator not in ('decode','fused','fused_auto','exact'):raise ValueError('unknown Metal operator')
    if metal_operator=='exact' and (not compressed or max_rows>2):
        raise ValueError('exact Metal requires explicit compressed linears and max_rows<=2')
    automatic=metal_operator=='fused_auto'
    operator=metal
    if metal_operator in ('fused','fused_auto'):
        if metal_operator=='fused' and not compressed:
            raise ValueError('fused Metal requires explicit compressed linears')
        from georefine_context_runtime import context_exponent_metal_fused as operator
    elif metal_operator=='exact':
        from georefine_context_runtime import context_exponent_metal_gemv_exact as operator
    if package_format == 'typed':
        from georefine_context_runtime.typed_context_package import TypedContextResolver
        resolver_factory=resolver_factory or TypedContextResolver
    elif package_format == 'safetensors':
        from georefine_context_runtime.safetensors_source import SafetensorsResolver
        resolver_factory=resolver_factory or SafetensorsResolver
    else:
        resolver_factory=resolver_factory or VerifiedTensorResolver
    config_reader=config_reader or (_read_typed_config if package_format == 'typed' else
                                    _read_safetensors_config if package_format == 'safetensors' else _read_config)
    array_factory=array_factory or _mlx_array
    linear_factory=linear_factory or operator.prepare_verified_linear
    linear_validator=linear_validator or _is_mlx_linear
    package=Path(package)
    cache=Path(metadata_cache_dir) if metadata_cache_dir is not None else package.parent/'.scratch/mlx-metadata'
    resolver_kwargs = ({'max_source_bytes': source_working_budget_bytes} if typed else {'cache_bytes': 0})
    with resolver_factory(package,**resolver_kwargs) as resolver:
        config,config_sha=config_reader(resolver,cache)
        model=model_factory(copy.deepcopy(config))
        expected={name:tuple(parameter.shape) for name,parameter in _flatten(model.parameters())}
        if not expected:raise ValueError('model factory returned no parameters')
        if callable(bindings):
            bindings = bindings(model, copy.deepcopy(config))
        mapping=(dict(bindings) if bindings is not None
                 else {name:TensorBinding((('mtp.' if source_scope=='mtp' else '')+name,)) for name in expected})
        if set(mapping)!=set(expected):
            raise ValueError('binding set differs from model parameter set')
        if any(not isinstance(b,TensorBinding) or not isinstance(b.sources,tuple) or not b.sources
               or any(not isinstance(n,str) or not n for n in b.sources) for b in mapping.values()):
            raise ValueError('bindings require nonempty tuples of source names')
        present=set(resolver.tensor_names)
        used={name for binding in mapping.values() for name in binding.sources}
        if source_scope=='mtp':
            if any(not name.startswith('mtp.') for name in used):
                raise ValueError('MTP scope may only bind original MTP tensors')
            unused={name for name in present if not name.startswith('mtp.')}
        else:
            unused=set(allowed_unused)
        if not used<=present or used&unused or present-used!=unused:
            raise ValueError('source tensor coverage differs from explicit bindings and exclusions')
        if any(name.startswith('mtp.') for name in used&set(resolver.manifest.get('selected_candidates',{}))):
            raise ValueError('MTP sources must be unchanged baseline frames')
        entries={name:resolver.entry(name) for name in used}
        binding_dtypes = {}
        for name, binding in mapping.items():
            kinds = {entries[source].get('dtype', 'BF16') for source in binding.sources}
            if len(kinds) != 1:
                raise ValueError(f'binding sources have different dtypes: {name}')
            binding_dtypes[name] = next(iter(kinds))
        def source_bytes(source):
            return (entries[source]['actual_source_bytes'] if typed
                    else math.prod(entries[source]['shape'])*2)
        def assembled_bytes(name, shape):
            # Concatenation/transpose/reshape never changes storage width.
            return sum(source_bytes(source) for source in mapping[name].sources)
        auto_modules={}
        if automatic:
            for name,shape in expected.items():
                if binding_dtypes[name]!='BF16' or not name.endswith('.weight') or len(shape)!=2 or shape[1]%128:
                    continue
                module=name.removesuffix('.weight')
                members={key for key in expected if key.startswith(module+'.')}
                if members=={name} and linear_validator(_module_at(model,module)):
                    auto_modules[module]=name
            if not auto_modules:raise ValueError('automatic fused loader found no eligible bias-free linear')
        dense_bytes=0
        compressed_bytes=0
        transient_bytes=0
        plans={}
        for name,expected_shape in expected.items():
            shape=_shape(mapping[name],entries)
            if shape!=expected_shape:raise ValueError(f'parameter shape mismatch: {name}')
            working=sum(source_bytes(n) for n in mapping[name].sources)+assembled_bytes(name,shape)
            if working>source_working_budget_bytes:
                raise MemoryError(f'{name}: source working buffers exceed budget')
            module=name.removesuffix('.weight') if name.endswith('.weight') else None
            if module in compressed or module in auto_modules:
                if binding_dtypes[name] != 'BF16':
                    raise ValueError('compressed MLX linears require original BF16 source weights')
                members={key for key in expected if key.startswith(module+'.')}
                if (len(shape)!=2 or members!={name}
                        or not linear_validator(_module_at(model,module))):
                    raise ValueError('compressed MLX linears require a known bias-free linear with only weight')
                if module in compressed:
                    arrays,_=operator.host_inputs(compressed[module])
                    if (arrays['N'],arrays['K'])!=((shape[0]+63)//64*64,shape[1]):
                        raise ValueError(f'compressed panel geometry mismatch: {name}')
                    ledger=operator.allocation_ledger(arrays,max_rows=max_rows)
                    plans[module]=ledger
                    compressed_bytes+=ledger['resident_bytes']
                    transient_bytes=max(transient_bytes,ledger['known_transient_bytes'])
                    del arrays
            else:
                dense_bytes+=assembled_bytes(name,shape)
        if set(plans)!=set(compressed):raise ValueError('compressed module is absent from model parameters')

        auto_evidence={};auto_source_peak=0
        if automatic:
            from georefine_context_runtime import context_exponent_packet as packet
            for name,shape in expected.items():
                binding=mapping[name]
                piece_bytes=sum(source_bytes(n) for n in binding.sources)
                working=piece_bytes+assembled_bytes(name,shape)
                if working>source_working_budget_bytes:
                    raise MemoryError(f'{name}: source working buffers exceed budget')
                words,hashes,digest=_resolve_binding_words(resolver,binding,entries,shape,typed=typed)
                if name.removesuffix('.weight') in auto_modules:
                    module=name.removesuffix('.weight')
                    packet_hashes,packet_sizes,peak,resident_sizes,_=_encode_auto_panels(
                        words,packet_module=packet,fused_operator=operator,max_rows=max_rows,
                        source_budget_bytes=source_working_budget_bytes,
                        source_piece_bytes=piece_bytes)
                    auto_source_peak=max(auto_source_peak,peak)
                    ledger=_auto_fused_ledger(resident_sizes,shape=shape,max_rows=max_rows,
                                              fused_operator=operator)
                    host_peak=working+sum(packet_sizes)+3*ledger['resident_bytes']+64*1024**2
                    if host_peak>source_working_budget_bytes:
                        raise MemoryError('automatic host packet assembly exceeds source working budget')
                    auto_source_peak=max(auto_source_peak,host_peak)
                    plans[module]=ledger
                    compressed_bytes+=ledger['resident_bytes']
                    transient_bytes=max(transient_bytes,ledger['known_transient_bytes'])
                    auto_evidence[name]=dict(source_hashes=hashes,source_digest=digest,
                                             packet_hashes=packet_hashes,packet_sizes=packet_sizes)
                    del packet_hashes,packet_sizes,resident_sizes
                else:
                    auto_evidence[name]=dict(source_hashes=hashes,source_digest=digest)
                del words
        if dense_bytes+compressed_bytes+transient_bytes>memory_budget_bytes:
            raise MemoryError('known model tensors and decode temporary exceed Metal budget')
        rows=[]
        for name,expected_shape in expected.items():
            binding=mapping[name]
            words,hashes,digest=_resolve_binding_words(resolver,binding,entries,expected_shape,typed=typed)
            if automatic:
                recorded=auto_evidence[name]
                if hashes!=recorded['source_hashes'] or digest!=recorded['source_digest']:
                    raise ValueError(f'automatic source changed between passes: {name}')
            module=name.removesuffix('.weight') if name.endswith('.weight') else None
            if module in auto_modules:
                packet_hashes,_,peak,_,packets=_encode_auto_panels(
                    words,packet_module=packet,fused_operator=operator,max_rows=max_rows,
                    source_budget_bytes=source_working_budget_bytes,
                    source_piece_bytes=sum(source_bytes(n) for n in binding.sources),
                    retain_packets=True)
                auto_source_peak=max(auto_source_peak,peak)
                if packet_hashes!=auto_evidence[name]['packet_hashes']:
                    raise ValueError(f'automatic packet encoding changed between passes: {name}')
                replacement=linear_factory(packets,shape=expected_shape,source_sha256=digest,
                    max_rows=max_rows,allocation_budget_bytes=linear_allocation_budget)
                _replace(model,module,replacement)
                kind='bhe1-metal-fused-auto'
                del packets
            elif module in compressed:
                replacement=linear_factory(compressed[module],shape=expected_shape,source_sha256=digest,
                    max_rows=max_rows,allocation_budget_bytes=linear_allocation_budget)
                _replace(model,module,replacement)
                kind={'fused':'bhe1-metal-fused','exact':'bhe1-metal-exact'}.get(metal_operator,'bhe1-metal')
            else:
                value=(_mlx_typed_array(words,binding_dtypes[name])
                       if typed and not custom_array_factory else array_factory(words))
                if tuple(value.shape)!=expected_shape:raise ValueError('array conversion changed parameter shape')
                model.load_weights([(name,value)],strict=False)
                del value
                kind=binding_dtypes[name].lower() if typed else 'bf16'
            rows.append(dict(name=name,shape=list(expected_shape),sources=hashes,
                runtime_source_sha256=digest,kind=kind,concatenate_axis=binding.concatenate_axis,
                transpose=binding.transpose,reshape=binding.reshape))
            del words
        if typed:
            # Bind lifecycle checks to both passes before exposing a usable model.
            resolver.verify_integrity()
        receipt=dict(schema='georefine.predictive_mlx_load.v1',scope=source_scope,
            **({"metal_operator":metal_operator} if metal_operator!="decode" else {}),
            manifest_sha256=resolver.manifest_sha256,config_sha256=config_sha,
            tensor_count=len(rows),tensors=rows,unused_sources=sorted(unused),
            resident_dense_bytes=dense_bytes,resident_compressed_bytes=compressed_bytes,
            largest_known_transient_bytes=transient_bytes,memory_budget_bytes=memory_budget_bytes,
            source_working_budget_bytes=source_working_budget_bytes,
            memory_exclusions=['allocator reserve','MLX matmul workspace','cache and scheduler state',
                               'CPU codec tables and caller-owned packet bytes'],
            materialized_checkpoint=False,serving_qualified=False)
        if automatic:receipt['automatic_source_working_peak_bytes']=auto_source_peak
        if typed:receipt['package_format']=package_format
    return model,receipt


def load_verified_mlx_mtp(package, *, head_factory, **kwargs):
    """Load every original MTP tensor into a matching fresh MLX head skeleton.

    Other package tensors are excluded by this explicit head-only scope. Missing
    or extra MTP tensors, modified candidates and non-MTP bindings are refused.
    """
    return load_verified_mlx(package,model_factory=head_factory,source_scope='mtp',**kwargs)
