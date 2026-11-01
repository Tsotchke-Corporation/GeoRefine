"""Transactional batched MLX verification over independent request caches.

All work uses merged copies; individual request state changes only after every
verification/replay in the group succeeds. Uniform draft-length groups avoid
padded token rows. KV histories may have different lengths and are left-padded
by MLX's cache merge, with an explicit attention mask.
"""
from __future__ import annotations

from collections import defaultdict
from .mlx_speculative import cache_array_bytes


def merge_caches(rows):
    from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache
    if not rows or any(len(row)!=len(rows[0]) for row in rows):
        raise ValueError('incompatible cache layer counts')
    merged=[]
    for caches in zip(*rows):
        if all(isinstance(c,KVCache) for c in caches):
            # A one-token prompt leaves MTP caches empty; create a real empty
            # batch cache instead of inferring dimensions from absent arrays.
            if all(c.keys is None for c in caches):
                merged.append(BatchKVCache([0]*len(caches)))
            else:merged.append(BatchKVCache.merge(caches))
        elif all(isinstance(c,ArraysCache) for c in caches):
            merged.append(ArraysCache.merge(caches))
        else:raise TypeError('unsupported mixed cache types')
    return merged


def extract_caches(caches,index):
    return [cache.extract(index) for cache in caches]


def _forward(model,tokens,caches,requests,*,exact_digest=False):
    import mlx.core as mx
    kwargs={}
    if any(r.rope_delta for r in requests):
        from .mlx_qwen import positions
        kwargs['position_ids']=positions([r.position+r.rope_delta for r in requests],len(tokens[0]))
    hidden=model.forward_hidden(mx.array(tokens,dtype=mx.int32),caches,**kwargs)
    from .mlx_digest import predict
    predictions,digests=predict(model,hidden,exact_digest=exact_digest)
    return hidden,predictions,digests


def verify_group(engine,ids,depth):
    import mlx.core as mx
    requests=[engine.requests[i] for i in ids]
    originals=[r.cache for r in requests]
    current=[r.current for r in requests]
    drafts=[[] for _ in requests]
    if depth:
        draft_caches=merge_caches([r.mtp_cache for r in requests])
        hidden=mx.concatenate([r.hidden for r in requests],axis=0)
        token=current
        for draft_index in range(depth):
            embeds=engine.model.model.embed_tokens(mx.array(token,dtype=mx.int32)[:,None])
            kwargs={}
            if any(r.rope_delta for r in requests):
                from .mlx_qwen import positions
                kwargs['position_ids']=positions([r.position+r.rope_delta+draft_index for r in requests],1)
            hidden=engine.mtp(hidden,embeds,draft_caches,**kwargs)
            predicted=mx.argmax(engine.model.project(hidden),axis=-1)
            mx.eval(predicted,hidden)
            token=predicted[:,0].tolist()
            for row,t in zip(drafts,token):row.append(t)
        del draft_caches,hidden
    inputs=[[token,*row] for token,row in zip(current,drafts)]
    merged=merge_caches(originals)
    hidden,predicted,deciding_digests=_forward(engine.model,inputs,merged,requests,exact_digest=engine.exact_digest)
    results=[];replay=defaultdict(list)
    for i,(r,proposal,prediction) in enumerate(zip(requests,drafts,predicted)):
        accepted=0
        while accepted<len(proposal) and proposal[accepted]==prediction[accepted]:accepted+=1
        emitted=prediction[:min(accepted+1,r.remaining)]
        for j,t in enumerate(emitted):
            if t in r.stop_tokens:
                emitted=emitted[:j+1];break
        result=dict(emitted=emitted,accepted=min(accepted,len(emitted)),proposed=depth)
        if engine.exact_digest:result['digests']=deciding_digests[i][:len(emitted)]
        if len(emitted)==depth+1:
            result.update(cache=extract_caches(merged,i),hidden=hidden[i:i+1])
        else:replay[len(emitted)].append(i)
        results.append(result)
    # Replaying only accepted prefixes restores GDN and attention consistently.
    # Group equal prefix lengths so rejection does not force one model call/user.
    for count,indices in replay.items():
        replay_cache=merge_caches([originals[i] for i in indices])
        replay_hidden,replay_predicted,_=_forward(engine.model,[inputs[i][:count] for i in indices],replay_cache,[requests[i] for i in indices])
        for row,i in enumerate(indices):
            if replay_predicted[row]!=results[i]['emitted']:
                raise RuntimeError('batched target predictions changed after rejection replay')
            results[i].update(cache=extract_caches(replay_cache,row),hidden=replay_hidden[row:row+1])
    del merged,hidden
    if engine.mtp is not None:
        grouped=defaultdict(list)
        for i,result in enumerate(results):grouped[len(result['emitted'])].append(i)
        for count,indices in grouped.items():
            head_cache=merge_caches([requests[i].mtp_cache for i in indices])
            previous=mx.concatenate([mx.concatenate([requests[i].hidden,results[i]['hidden'][:,:-1]],axis=1) for i in indices],axis=0)
            tokens=[[requests[i].current,*results[i]['emitted'][:-1]] for i in indices]
            kwargs={}
            if any(requests[i].rope_delta for i in indices):
                from .mlx_qwen import positions
                kwargs['position_ids']=positions([requests[i].position+requests[i].rope_delta for i in indices],count)
            warm=engine.mtp(previous,engine.model.model.embed_tokens(mx.array(tokens,dtype=mx.int32)),head_cache,**kwargs)
            mx.eval(warm)
            for row,i in enumerate(indices):results[i]['mtp_cache']=extract_caches(head_cache,row)
    for result in results:result['hidden']=result['hidden'][:,-1:]
    # Enforce the live payload bound before committing any request in this group.
    replacements={request_id:result for request_id,result in zip(ids,results)}
    states=[]
    for request_id,r in engine.requests.items():
        result=replacements.get(request_id)
        states.append((result['cache'],result.get('mtp_cache',r.mtp_cache),result['hidden']) if result else (r.cache,r.mtp_cache,r.hidden))
    if cache_array_bytes(states)>engine.cache_budget_bytes:
        raise MemoryError('batched result exceeds cache budget')
    events=[]
    for request_id,r,result in zip(ids,requests,results):
        r.cache=result['cache'];r.mtp_cache=result.get('mtp_cache',r.mtp_cache)
        r.hidden=result['hidden'];r.tokens.extend(result['emitted'])
        r.remaining-=len(result['emitted']);r.current=result['emitted'][-1]
        finished=r.remaining==0 or r.current in r.stop_tokens
        event=dict(id=request_id,tokens=result['emitted'],finished=finished,
                   accepted=result['accepted'],proposed=depth,batch_size=len(ids))
        if engine.exact_digest:event['digests']=result['digests']
        events.append(event)
        if finished:del engine.requests[request_id]
    return events


def step(engine):
    events=[];groups=defaultdict(list)
    for request_id,r in list(engine.requests.items()):
        if not r.tokens:
            r.tokens.append(r.current);r.remaining-=1
            finished=r.current in r.stop_tokens or r.remaining==0
            event=dict(id=request_id,tokens=[r.current],finished=finished)
            if engine.exact_digest:event['digests']=[r.current_digest]
            events.append(event)
            if finished:del engine.requests[request_id]
        else:groups[min(engine.spec_k,r.remaining-1)].append(request_id)
    for depth,ids in groups.items():
        row_capacity=engine.max_batch_rows//(depth+1)
        while ids:
            count=min(len(ids),row_capacity)
            # Cache merging pads to the longest history. Reserve work/replay/head
            # copies before allocation, including MLX's 256-token KV growth step.
            while count:
                chosen=ids[:count]
                longest=max(engine.requests[i].position+depth+1 for i in chosen)
                work=4*count*engine.model.cache_reservation_bytes(longest,mtp=engine.mtp is not None)
                if engine._bytes()+work<=engine.cache_budget_bytes:break
                count//=2
            if not count:
                request_id=ids.pop(0)
                del engine.requests[request_id]
                events.append(dict(id=request_id,tokens=[],finished=True,error='batched work buffers exceed cache budget'))
                continue
            chosen=ids[:count];del ids[:count]
            try:
                output=verify_group(engine,chosen,depth)
                for event in output:
                    if event['id'] in engine.requests:
                        engine.requests[event['id']].position+=len(event['tokens'])
                events.extend(output)
            except Exception as exc:
                # Originals remain unchanged; failure must not lose successful
                # events for unrelated groups already processed in this step.
                for request_id in chosen:
                    engine.requests.pop(request_id,None)
                    events.append(dict(id=request_id,tokens=[],finished=True,error=str(exc)))
    return events
