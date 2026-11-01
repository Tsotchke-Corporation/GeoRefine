"""BCA4 bounded alias-rANS word streams with exact source-CDF binding.

Four wide16-row packets;32 lanes share one uint16 stream per packet.
No old loader or codec selects this experimental wire.
"""
from __future__ import annotations
import hashlib
import os
import struct
import numpy as np
from . import context_tiled_packet as tiled
from . import context_alias_ans as alias

PacketError = tiled.PacketError
HEADER = struct.Struct('<4sBBBBIIIIII32s32s32sIIII')
PACKET = tiled.PACKET
ENTRY = tiled.ENTRY
DESCRIPTOR_KEYS = tiled.DESCRIPTOR_KEYS
FORMAT = 'BCA4-experimental-raw6-wide16row-chunkmajor-32lane-S1-v1'
LANES, L, TOTAL = 32, 1 << 16, 1 << 16
MAX_BYTES = 32*1048576


class _AliasMetadataCache:
    """Private, caller-scoped store of immutable metadata derived from exact CDF bytes."""
    _MAX_ENTRIES = 2
    __slots__ = ("__entries",)

    def __init__(self):
        self.__entries = {}

    def _lookup(self, key):
        return self.__entries.get(key)

    def _remember(self, key, entry):
        previous = self.__entries.get(key)
        if previous is None:
            if len(self.__entries) >= self._MAX_ENTRIES:
                raise PacketError('scoped alias metadata cache entry limit exceeded')
            self.__entries[key] = entry
        elif (previous[0] != entry[0] or previous[1] is not entry[1] or
              previous[2] != entry[2]):
            raise PacketError('conflicting scoped alias metadata cache entry')


def _new_alias_metadata_cache():
    """Make a private cache for one explicitly bounded multi-panel operation."""
    return _AliasMetadataCache()


def _word_function(name):
    flag=os.environ.get('GEOREFINE_BCA4_NATIVE_CPU','0')
    if flag not in ('0','1'):raise PacketError('GEOREFINE_BCA4_NATIVE_CPU must be0or1')
    if flag=='1':
        from . import context_alias_ans_native
        return getattr(context_alias_ans_native,name+'_native')
    return getattr(alias,name)


def _hash(data):
    return hashlib.sha256(data).digest()


def _cdf(data, nr, nc):
    """Validate exact serialized sparse cumulatives; never renormalize/recount."""
    if not isinstance(data,(bytes,bytearray,memoryview)):
        raise PacketError('sparse CDF must be bytes')
    if memoryview(data).nbytes>MAX_BYTES:
        raise PacketError('oversized sparse CDF')
    data=bytes(data)
    nb=4*(nr*nc+1)
    if len(data)<nb:
        raise PacketError('short sparse CDF')
    offsets=struct.unpack_from('<'+'I'*(nr*nc+1),data)
    if offsets[0] or nb+6*offsets[-1]!=len(data) or offsets[-1]>nr*nc*1024:
        raise PacketError('invalid sparse CDF directory')
    tables=[]
    for a,b in zip(offsets,offsets[1:]):
        if a>b or b-a>1024:
            raise PacketError('invalid sparse CDF span')
        entries=[ENTRY.unpack_from(data,nb+6*j) for j in range(a,b)]
        last_s,last_c=-1,0
        for symbol,upper in entries:
            if not last_s<symbol<1024 or not last_c<upper<=TOTAL:
                raise PacketError('invalid sparse CDF symbol/normalization')
            last_s,last_c=symbol,upper
        if entries and last_c!=TOTAL:
            raise PacketError('incomplete sparse CDF normalization')
        tables.append(entries)
    return tables


def _labels(values, count, classes):
    a=np.frombuffer(values,dtype=np.uint8) if isinstance(values,(bytes,bytearray,memoryview)) else np.asarray(values)
    if a.shape!=(count,) or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=classes):
        raise PacketError('invalid supplied labels')
    return np.ascontiguousarray(a,dtype=np.uint8)


def _chunk_words(a):
    return np.ascontiguousarray(a.reshape(16,a.shape[1]//128,128).transpose(1,0,2)).ravel()


def _row_words(flat, k):
    return np.ascontiguousarray(flat.reshape(k//128,16,128).transpose(1,0,2).reshape(16,k))


def _validate_alias(tables,cdf,nr,nc):
    """Independent interval proof: every bucket slot and symbol rank exactly once.

    No 65536-slot enumeration per context; canonical CDF frequencies and all
    flags/branches are checked, then rank intervals sorted for exact coverage.
    """
    source=_cdf(cdf,nr,nc)
    offsets=tables.bucket_offsets
    if len(offsets)!=nr*nc+1 or int(offsets[0]) or int(offsets[-1])!=len(tables.bucket_records):
        raise PacketError('alias directory mismatch')
    if len(tables.context_shifts)!=nr*nc:raise PacketError('alias shifts length')
    for cx,entries in enumerate(source):
        a,b=int(offsets[cx]),int(offsets[cx+1]);h=int(tables.context_shifts[cx])
        if a>b or b>len(tables.bucket_records):raise PacketError('alias offset span')
        if not entries:
            if a!=b or h:raise PacketError('noncanonical empty alias context')
            continue
        expected=1<<(len(entries)-1).bit_length()
        if not 0<=h<=10 or b-a!=expected or 1<<h!=expected:raise PacketError('alias bucket geometry')
        freqs={};last=0
        for sym,upper in entries:freqs[sym]=upper-last;last=upper
        segments={sym:[] for sym in freqs};width=65536>>h
        for rec in tables.bucket_records[a:b]:
            primary,other,cut16,pbase,abase,pf16,af16,flags=map(int,rec)
            if flags&~7 or (flags&1 and cut16) or (flags&2 and pf16) or (flags&4 and af16):
                raise PacketError('noncanonical alias flags')
            cut=65536 if flags&1 else cut16
            pf=65536 if flags&2 else pf16;af=65536 if flags&4 else af16
            if not 0<=cut<=width or freqs.get(primary)!=pf or freqs.get(other)!=af:
                raise PacketError('alias CDF frequency mismatch')
            for symbol,base,length in ((primary,pbase,cut),(other,abase,width-cut)):
                if not length:
                    if base:raise PacketError('noncanonical empty alias rank')
                else:
                    if base+length>freqs[symbol]:raise PacketError('alias rank outside frequency')
                    segments[symbol].append((base,base+length))
        for symbol,spans in segments.items():
            cursor=0
            for begin,end in sorted(spans):
                if begin!=cursor:raise PacketError('alias rank gap/overlap')
                cursor=end
            if cursor!=freqs[symbol]:raise PacketError('incomplete alias ranks')


def _map_digest(tables):
    if tables is None:return _hash(b'BCA4-empty-alias-v1')
    return _hash(b'BCA4-alias-builder1-record1-word1'+tables.bucket_records.tobytes()+
                 tables.bucket_offsets.tobytes()+tables.context_shifts.tobytes())


def _alias_metadata(table,nr,nc,cache,*,present=True):
    """Reuse only immutable metadata derived from the exact serialized CDF."""
    if type(cache) is not _AliasMetadataCache:
        raise PacketError('invalid private alias metadata cache')
    table=bytes(table)
    digest=_hash(table)
    key=(digest,table,nr,nc,1,1,1,bool(present))
    cached=cache._lookup(key)
    if cached is not None:
        if cached[0]!=table:
            raise PacketError('cached CDF bytes mismatch')
        return cached[1],cached[2]
    tables=alias.build_alias_tables(table,nr*nc) if present else None
    if present:_validate_alias(tables,table,nr,nc)
    map_digest=_map_digest(tables)
    cache._remember(key,(table,tables,map_digest))
    return tables,map_digest


def _identity(k,nr,nc,labels,table,map_digest,flags=0):
    return _hash(struct.pack('<9I',64,k,1,nr,nc,1,1,1,flags)+labels+table+map_digest)


def _contexts(labels,nc,row0,k):
    rc=np.frombuffer(labels[:64],dtype=np.uint8)
    cc=np.frombuffer(labels[64:],dtype=np.uint8)
    return _chunk_words((rc[row0:row0+16,None].astype(np.uint16)*nc+cc[None,:]).astype('<u2'))



def encode(words, *, split_k=1, row_classes=16, col_classes=4,
           row_labels=None, col_labels=None, sparse_cdf=None, experimental_large_geometry=False,
           shared_metadata=False, _alias_cache=None):
    """Encode bounded fixtures with derived or exact supplied global metadata.

    Supply all three metadata arguments together. Large geometry requires explicit
    opt-in and accepts K multiples128 through32768. Cumulatives are imported
    verbatim, including frequencies not proportional to this fixture's counts.
    Default coded/raw selection includes the global table cost. Explicit
    shared_metadata preserves the exact CDF and uses packet-local selection;
    header flag bit0 identifies this matrix-oriented form.
    """
    if type(shared_metadata) is not bool:raise PacketError('shared metadata opt-in must be bool')
    a=np.asarray(words)
    if type(experimental_large_geometry) is not bool:raise PacketError('large geometry opt-in must be bool')
    valid_shape=(a.ndim==2 and a.shape[0]==64 and (a.shape[1] in (512,1024) or
        (experimental_large_geometry and 128<=a.shape[1]<=32768 and a.shape[1]%128==0)))
    if not valid_shape or a.dtype.kind!='u' or a.dtype.itemsize!=2:
        raise PacketError('BCA4 requires uint16 N64,K512/1024 or explicit bounded large geometry; tails forbidden')
    if type(split_k) is not int or split_k!=1:
        raise PacketError('BCA4 prototype requires split_k=1')
    nr,nc=row_classes,col_classes
    if type(nr) is not int or type(nc) is not int or nr not in tiled.codec.ROWS or nc not in tiled.codec.COLS:
        raise PacketError('invalid context classes')
    a=np.ascontiguousarray(a,dtype='<u2'); k=a.shape[1]
    supplied=[q is not None for q in (row_labels,col_labels,sparse_cdf)]
    if any(supplied) and not all(supplied):
        raise PacketError('supply both labels and sparse CDF together')
    if all(supplied):
        rc,cc=_labels(row_labels,64,nr),_labels(col_labels,k,nc)
        tables=_cdf(sparse_cdf,nr,nc)
        table=bytes(sparse_cdf)
    else:
        rc,cc=tiled.codec._labels(a,nr,nc)
        _,table=tiled._global_tables(a,rc,cc,nr,nc)
        tables=_cdf(table,nr,nc)
    if _alias_cache is None:
        tables=alias.build_alias_tables(table,nr*nc)
    else:
        tables,_=_alias_metadata(table,nr,nc,_alias_cache,present=True)
    word_encoder=_word_function("encode32_word")
    candidates=[]
    labels=rc.tobytes()+cc.tobytes()
    for row0 in range(0,64,16):
        original=a[row0:row0+16]
        flat=_chunk_words(original)
        contexts=_contexts(labels,nc,row0,k)
        try:
            # BCA4 carries only AAN2 states/common words, not its nested header.
            main=word_encoder((flat>>5)&1023,contexts,tables,max_symbols=16*k)[80:]
        except ValueError as exc:
            raise PacketError(str(exc)) from exc
        residual=tiled._pack6(flat)
        def frame(mode,main,residual):
            body=PACKET.pack(b'BAP4',4,mode,0,row0,0,16,k,len(main),len(residual),
                             _hash(original.tobytes()))+main+residual
            return body+_hash(body)
        coded,raw=frame(0,main,residual),frame(1,flat.tobytes(),b'')
        candidates.append((coded if len(coded)<len(raw) else raw,raw))
    if shared_metadata or len(table)+sum(len(q[0]) for q in candidates)<sum(len(q[1]) for q in candidates):
        packets=[q[0] for q in candidates]
    else:
        packets=[q[1] for q in candidates]; table=b''; tables=None
    labels=rc.tobytes()+cc.tobytes(); offsets=[0]
    for q in packets:offsets.append(offsets[-1]+len(q))
    body=(HEADER.pack(b'BCA4',4,nr,nc,32,64,k,1,4,len(labels),len(table),
                      _identity(k,nr,nc,labels,table,_map_digest(tables),int(shared_metadata)),_hash(a.tobytes()),
                      _hash(table),1,1,1,int(shared_metadata))+
          labels+table+struct.pack('<5Q',*offsets)+b''.join(packets))
    return body+_hash(body)


def _parse(blob,*,_alias_cache=None):
    if not isinstance(blob,(bytes,bytearray,memoryview)):
        raise PacketError('bundle must be bytes')
    if not HEADER.size+32<=memoryview(blob).nbytes<=MAX_BYTES:
        raise PacketError('invalid bounded bundle size')
    blob=bytes(blob)
    magic,ver,nr,nc,lanes,rows,k,split,count,nl,nt,identity,digest,cdf_digest,builder,record,traversal,flags=HEADER.unpack_from(blob)
    if (magic!=b'BCA4' or ver!=4 or lanes!=32 or nr not in tiled.codec.ROWS or nc not in tiled.codec.COLS or
            (builder,record,traversal)!=(1,1,1) or flags&~1 or rows!=64 or not (128<=k<=32768 and k%128==0) or split!=1 or count!=4 or nl!=64+k):
        raise PacketError('unsupported BCA4 geometry/split; tails forbidden')
    ts=HEADER.size+nl; ds=ts+nt; start=ds+40
    if start>len(blob)-32 or _hash(blob[:-32])!=blob[-32:]:
        raise PacketError('bundle checksum/length mismatch')
    labels=blob[HEADER.size:ts]
    if any(r>=nr for r in labels[:64]) or any(c>=nc for c in labels[64:]):
        raise PacketError('invalid global labels')
    table=blob[ts:ds]
    if _hash(table)!=cdf_digest:raise PacketError('source CDF digest mismatch')
    # Reuse is confined to an explicit caller-owned scope and exact CDF bytes.
    if _alias_cache is None:
        tables=alias.build_alias_tables(table,nr*nc) if nt else None
        if nt:_validate_alias(tables,table,nr,nc)
        map_digest=_map_digest(tables)
    else:
        tables,map_digest=_alias_metadata(table,nr,nc,_alias_cache,present=bool(nt))
    if _identity(k,nr,nc,labels,table,map_digest,flags)!=identity:
        raise PacketError('metadata identity mismatch')
    offsets=struct.unpack_from('<5Q',blob,ds)
    if offsets[0] or offsets[-1]!=len(blob)-32-start or any(a>=b for a,b in zip(offsets,offsets[1:])):
        raise PacketError('invalid packet directory')
    packets=[]
    for index,(lo,hi) in enumerate(zip(offsets,offsets[1:])):
        ps,end=start+lo,start+hi
        if end-ps<PACKET.size+32:
            raise PacketError('short wide packet')
        pm,pv,mode,reserved,row0,col0,pr,pc,nm,nd,pdigest=PACKET.unpack_from(blob,ps)
        if (pm!=b'BAP4' or pv!=4 or mode not in (0,1) or reserved or row0!=16*index or col0 or pr!=16 or pc!=k or
                end-ps!=PACKET.size+nm+nd+32):
            raise PacketError('invalid wide packet geometry/framing')
        if _hash(blob[ps:end-32])!=blob[end-32:end]:
            raise PacketError('packet checksum mismatch')
        ms=ps+PACKET.size; rs=ms+nm
        if mode:
            if nd or nm!=32*k:
                raise PacketError('invalid raw length')
            raw=_row_words(np.frombuffer(blob[ms:rs],dtype='<u2'),k)
            if _hash(raw.tobytes())!=pdigest:
                raise PacketError('raw word checksum mismatch')
        else:
            if not nt or nm<128 or nm%2 or nm>128+32*k or nd!=12*k:
                raise PacketError('invalid coded length/table')
            for lane in range(32):
                state=struct.unpack_from('<I',blob,ms+4*lane)[0]
                if state<L:raise PacketError('invalid initial state')
        packets.append(dict(mode=mode,row0=row0,col0=0,packet_rows=16,packet_cols=k,
            main_start=ms,main_bytes=nm,residual_start=rs,residual_bytes=nd,
            digest=pdigest,packet_start=ps,packet_bytes=end-ps))
    if nt and all(q['mode'] for q in packets) and not flags&1:
        raise PacketError('unused CDF in raw bundle')
    return blob,dict(shared_metadata=bool(flags&1),rows=64,cols=k,split_k=1,row_classes=nr,col_classes=nc,labels=labels,
        table_start=ts,table_bytes=nt,tables=tables,packets=packets,
        context_identity=identity.hex(),source_digest=digest.hex(),
        source_cdf_digest=cdf_digest.hex(),alias_map_digest=map_digest.hex())


def inspect(blob,*,_alias_cache=None):
    blob,info=_parse(blob,_alias_cache=_alias_cache); q=info['packets']
    components=dict(bundle_header=HEADER.size,global_labels=len(info['labels']),
        global_sparse_cdf=info['table_bytes'],directory=40,packet_headers=4*PACKET.size,
        checksums=5*32,main_streams_including_states=sum(p['main_bytes'] for p in q if not p['mode']),
        residual_streams=sum(p['residual_bytes'] for p in q),raw_payloads=sum(p['main_bytes'] for p in q if p['mode']))
    if sum(components.values())!=len(blob):raise PacketError('accounting mismatch')
    return dict(format=FORMAT,rows=64,cols=info['cols'],split_k=1,lanes=32,packets=[{k:v for k,v in p.items() if k!='digest'} for p in q],
        total_bytes=len(blob),source_bytes=128*info['cols'],components=components,
        context_identity=info['context_identity'],source_digest=info['source_digest'],source_cdf_digest=info['source_cdf_digest'],
        alias_map_digest=info['alias_map_digest'],
        descriptors_bytes=4*9*8,final_lane_status_bytes=4*32*4,
        shared_slab_bytes=64*136*2,decoded_weight_hbm_bytes=0,
        scope='experimental alias raw6; no archive ratio qualification')


def decode_cpu(blob,*,_alias_cache=None):
    blob,info=_parse(blob,_alias_cache=_alias_cache)
    return _decode(blob,info)


def _decode(blob,info):
    word_decoder=_word_function("decode32_word")
    k=info['cols']; out=np.empty((64,k),dtype='<u2')
    for q in info['packets']:
        ms,rs=q['main_start'],q['residual_start']
        if q['mode']:
            flat=np.frombuffer(blob[ms:rs],dtype='<u2')
        else:
            body=blob[ms:rs]
            frame=struct.pack('<4sBBHII32s32s',b'AAN2',2,32,0,16*k,len(body)-128,
                              info['tables'].source_cdf_digest,_hash(body))+body
            try:
                main=word_decoder(frame,_contexts(info['labels'],info['col_classes'],q['row0'],k),info['tables'],max_symbols=16*k)
            except ValueError as exc:raise PacketError(str(exc)) from exc
            flat=np.empty(16*k,dtype='<u2')
            for z in range(16*k):
                bit=6*z;residual=blob[rs+bit//8]>>(bit%8)
                if bit%8>2:residual|=blob[rs+bit//8+1]<<(8-bit%8)
                residual&=63
                flat[z]=(int(main[z])<<5)|(residual&31)|((residual>>5)<<15)
        decoded=_row_words(flat,k)
        if _hash(decoded.tobytes())!=q['digest']:raise PacketError('decoded packet checksum mismatch')
        out[q['row0']:q['row0']+16]=decoded
    if _hash(out.tobytes()).hex()!=info['source_digest']:raise PacketError('decoded source checksum mismatch')
    return out


def _resident_arrays(blob,panels=1,*,_alias_cache=None):
    """Full host validation before sealing; only compact payload/metadata on GPU.

    Derived records are canonical deterministic builder output. Builder interval
    coverage checks prove a bijection; payload decode verifies all states, stream
    consumption and original word digests. No host decoded weights are retained.
    """
    if type(panels) is not int or not 1<=panels<=1024:raise PacketError('invalid panels')
    blob,info=_parse(blob,_alias_cache=_alias_cache)
    _decode(blob,info)
    tables=info['tables'];payload=bytearray();descriptors=[]
    for p in info['packets']:
        q=dict(p);q['main_start']=len(payload)
        payload.extend(blob[p['main_start']:p['main_start']+p['main_bytes']])
        q['residual_start']=len(payload)
        payload.extend(blob[p['residual_start']:p['residual_start']+p['residual_bytes']])
        descriptors.append([q[key] for key in DESCRIPTOR_KEYS])
    c=info['row_classes']*info['col_classes']
    records=tables.bucket_records.tobytes() if tables else b''
    offsets=tables.bucket_offsets.tobytes() if tables else bytes(4*(c+1))
    shifts=tables.context_shifts.tobytes() if tables else bytes(c)
    # Records start at aligned allocation base. Metadata is shared across panels.
    metadata=(records+offsets+shifts+info['labels']+bytes.fromhex(info['source_cdf_digest'])+
              bytes.fromhex(info['alias_map_digest'])+struct.pack('<8I',1,info['cols'],panels,c,len(records)//16,1,1,1))
    return dict(data=np.tile(np.frombuffer(payload,dtype=np.uint8),(panels,1)),
        desc=np.tile(np.asarray(descriptors,dtype=np.int64),(panels,1,1)),
        metadata=np.frombuffer(metadata,dtype=np.uint8),K=info['cols'],nr=info['row_classes'],
        nc=info['col_classes'],buckets=len(records)//16,host_wire_bytes=len(blob),
        panels=panels,label_rows=64,matrix_mode=False,
        host_validation_decoded_weight_transient_bytes=128*info['cols'],
        host_inverse_bytes=tables.inverse_bytes if tables else 0,
        source_cdf_digest=info['source_cdf_digest'],alias_map_digest=info['alias_map_digest'])



def _matrix_panels(blobs,*,decode=False,_alias_cache=None):
    """Validate a transitional list of distinct row-panel wires as one matrix.

    Repeated CDF bytes are billed on the host. Derived alias/inverse tables are
    built/validated once; original row labels remain distinct. Payload validation
    is panel-at-a-time and never retains a decoded full matrix.
    """
    if not isinstance(blobs,(list,tuple)) or not 1<=len(blobs)<=4096:
        raise PacketError('matrix requires 1..4096 row-panel blobs, N multiple64')
    cache=_new_alias_metadata_cache() if _alias_cache is None else _alias_cache
    result=[];binding=None
    for blob in blobs:
        blob,info=_parse(blob,_alias_cache=cache)
        current=(info['cols'],info['row_classes'],info['col_classes'],
                 blob[info['table_start']:info['table_start']+info['table_bytes']],
                 info['alias_map_digest'],info['labels'][64:])
        if binding is None:binding=current
        elif current!=binding:raise PacketError('matrix panels require identical source CDF/map/column metadata')
        if decode:_decode(blob,info)
        result.append((blob,info))
    return result


def _resident_matrix_arrays(blobs,*,_alias_cache=None):
    panels=_matrix_panels(blobs,decode=True,_alias_cache=_alias_cache);first=panels[0][1]
    count=len(panels);n=64*count;k=first['cols'];tables=first['tables']
    size=sum(q['main_bytes']+q['residual_bytes'] for _,info in panels for q in info['packets'])
    # One final host payload allocation, then one private CUDA copy. No padded
    # equal-length panels and no GPU clone of temporary per-panel allocations.
    data=np.empty(size,dtype=np.uint8);descriptors=np.empty((count,4,9),dtype=np.int64)
    cursor=0
    for panel,(blob,info) in enumerate(panels):
        for packet_index,p in enumerate(info['packets']):
            q=dict(p);q['main_start']=cursor
            stop=cursor+p['main_bytes'];data[cursor:stop]=np.frombuffer(blob,dtype=np.uint8,count=p['main_bytes'],offset=p['main_start']);cursor=stop
            q['residual_start']=cursor
            stop=cursor+p['residual_bytes'];data[cursor:stop]=np.frombuffer(blob,dtype=np.uint8,count=p['residual_bytes'],offset=p['residual_start']);cursor=stop
            descriptors[panel,packet_index]=[q[key] for key in DESCRIPTOR_KEYS]
    c=first['row_classes']*first['col_classes']
    records=tables.bucket_records.tobytes() if tables else b''
    offsets=tables.bucket_offsets.tobytes() if tables else bytes(4*(c+1))
    shifts=tables.context_shifts.tobytes() if tables else bytes(c)
    labels=b''.join(info['labels'][:64] for _,info in panels)+first['labels'][64:]
    metadata=(records+offsets+shifts+labels+bytes.fromhex(first['source_cdf_digest'])+
              bytes.fromhex(first['alias_map_digest'])+struct.pack('<8I',2,k,count,c,len(records)//16,1,1,1))
    return dict(data=data,desc=descriptors,metadata=np.frombuffer(metadata,dtype=np.uint8),
        K=k,nr=first['row_classes'],nc=first['col_classes'],buckets=len(records)//16,
        panels=count,label_rows=n,matrix_mode=True,host_wire_bytes=sum(len(blob) for blob,_ in panels),
        host_validation_decoded_weight_transient_bytes=128*k,
        host_inverse_bytes=tables.inverse_bytes if tables else 0,
        source_cdf_digest=first['source_cdf_digest'],alias_map_digest=first['alias_map_digest'],
        matrix_source_digest=_hash(b''.join(bytes.fromhex(info['source_digest']) for _,info in panels)).hex())
