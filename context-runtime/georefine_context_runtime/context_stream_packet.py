"""BCS3 bounded experimental raw6 streams; no BCTX mode128/1.53x bridge claim.

N64,K512/1024,S1 only. Four wide16-row packets,32 rANS lanes per packet,
chunk-major traversal; every CH128 slab carries the same lane state/cursor.
Old BCI1/BCT2 encoders and wire formats are untouched. No CUDA entry exists.
"""
from __future__ import annotations
import bisect
import hashlib
import struct
import numpy as np
from . import context_tiled_packet as tiled

PacketError = tiled.PacketError
HEADER = struct.Struct('<4sBBBBIIIIII32s32s')
PACKET = tiled.PACKET
ENTRY = tiled.ENTRY
DESCRIPTOR_KEYS = tiled.DESCRIPTOR_KEYS
FORMAT = 'BCS3-experimental-raw6-wide16row-chunkmajor-32lane-S1-v1'
LANES, L, TOTAL = 32, 1 << 23, 1 << 16
MAX_BYTES = 1048576


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


def _identity(k,nr,nc,labels,table):
    return _hash(struct.pack('<IIIII',64,k,1,nr,nc)+labels+table)


def encode(words, *, split_k=1, row_classes=16, col_classes=4,
           row_labels=None, col_labels=None, sparse_cdf=None):
    """Encode bounded fixtures with derived or exact supplied global metadata.

    Supply all three metadata arguments together. Cumulatives are imported
    verbatim, including frequencies not proportional to this fixture's counts.
    Coded/raw selection includes the global table cost. Raw fallback needs no CDF.
    """
    a=np.asarray(words)
    if a.shape not in ((64,512),(64,1024)) or a.dtype.kind!='u' or a.dtype.itemsize!=2:
        raise PacketError('BCS3 requires uint16 N64,K512/1024; tails forbidden')
    if type(split_k) is not int or split_k!=1:
        raise PacketError('BCS3 prototype requires split_k=1')
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
    models=[]
    for entries in tables:
        lower=0; model={}
        for symbol,upper in entries:
            model[symbol]=(lower,upper-lower); lower=upper
        models.append(model)
    candidates=[]
    for row0 in range(0,64,16):
        original=a[row0:row0+16]
        flat=_chunk_words(original); streams=[]
        for lane in range(32):
            state=L; ren=bytearray()
            for t in reversed(range(len(flat)//32)):
                u,r,g=t>>6,(t>>2)&15,t&3
                cx=int(rc[row0+r])*nc+int(cc[128*u+32*g+lane])
                symbol=(int(flat[32*t+lane])>>5)&1023
                if symbol not in models[cx]:
                    raise PacketError('supplied CDF lacks a source symbol/context')
                lower,freq=models[cx][symbol]
                while state>=((L>>16)<<8)*freq:
                    ren.append(state&255); state>>=8
                state=((state//freq)<<16)+(state%freq)+lower
            streams.append(struct.pack('<I',state)+ren[::-1])
        offsets=[0]
        for stream in streams:offsets.append(offsets[-1]+len(stream))
        main=struct.pack('<33I',*offsets)+b''.join(streams)
        residual=tiled._pack6(flat)
        def frame(mode,main,residual):
            body=PACKET.pack(b'BSP3',3,mode,0,row0,0,16,k,len(main),len(residual),
                             _hash(original.tobytes()))+main+residual
            return body+_hash(body)
        coded,raw=frame(0,main,residual),frame(1,flat.tobytes(),b'')
        candidates.append((coded if len(coded)<len(raw) else raw,raw))
    if len(table)+sum(len(q[0]) for q in candidates)<sum(len(q[1]) for q in candidates):
        packets=[q[0] for q in candidates]
    else:
        packets=[q[1] for q in candidates]; table=b''
    labels=rc.tobytes()+cc.tobytes(); offsets=[0]
    for q in packets:offsets.append(offsets[-1]+len(q))
    body=(HEADER.pack(b'BCS3',3,nr,nc,32,64,k,1,4,len(labels),len(table),
                      _identity(k,nr,nc,labels,table),_hash(a.tobytes()))+
          labels+table+struct.pack('<5Q',*offsets)+b''.join(packets))
    return body+_hash(body)


def _parse(blob):
    if not isinstance(blob,(bytes,bytearray,memoryview)):
        raise PacketError('bundle must be bytes')
    if not HEADER.size+32<=memoryview(blob).nbytes<=MAX_BYTES:
        raise PacketError('invalid bounded bundle size')
    blob=bytes(blob)
    magic,ver,nr,nc,lanes,rows,k,split,count,nl,nt,identity,digest=HEADER.unpack_from(blob)
    if (magic!=b'BCS3' or ver!=3 or lanes!=32 or nr not in tiled.codec.ROWS or nc not in tiled.codec.COLS or
            rows!=64 or k not in (512,1024) or split!=1 or count!=4 or nl!=64+k):
        raise PacketError('unsupported BCS3 geometry/split; tails forbidden')
    ts=HEADER.size+nl; ds=ts+nt; start=ds+40
    if start>len(blob)-32 or _hash(blob[:-32])!=blob[-32:]:
        raise PacketError('bundle checksum/length mismatch')
    labels=blob[HEADER.size:ts]
    if any(r>=nr for r in labels[:64]) or any(c>=nc for c in labels[64:]):
        raise PacketError('invalid global labels')
    table=blob[ts:ds]; tables=_cdf(table,nr,nc) if nt else []
    if _identity(k,nr,nc,labels,table)!=identity:
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
        if (pm!=b'BSP3' or pv!=3 or mode not in (0,1) or reserved or row0!=16*index or col0 or pr!=16 or pc!=k or
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
            if not nt or nm<260 or nd!=12*k:
                raise PacketError('invalid coded length/table')
            ld=struct.unpack_from('<33I',blob,ms)
            if ld[0] or ld[-1]!=nm-132 or any(b-a<4 for a,b in zip(ld,ld[1:])):
                raise PacketError('invalid lane directory')
            for lane in range(32):
                state=struct.unpack_from('<I',blob,ms+132+ld[lane])[0]
                if not L<=state<L*256:
                    raise PacketError('invalid initial state')
        packets.append(dict(mode=mode,row0=row0,col0=0,packet_rows=16,packet_cols=k,
            main_start=ms,main_bytes=nm,residual_start=rs,residual_bytes=nd,
            digest=pdigest,packet_start=ps,packet_bytes=end-ps))
    if nt and all(q['mode'] for q in packets):
        raise PacketError('unused CDF in raw bundle')
    return blob,dict(rows=64,cols=k,split_k=1,row_classes=nr,col_classes=nc,labels=labels,
        table_start=ts,table_bytes=nt,tables=tables,packets=packets,
        context_identity=identity.hex(),source_digest=digest.hex())


def inspect(blob):
    blob,info=_parse(blob); q=info['packets']
    components=dict(bundle_header=HEADER.size,global_labels=len(info['labels']),
        global_sparse_cdf=info['table_bytes'],directory=40,packet_headers=4*PACKET.size,
        checksums=5*32,main_streams_including_states=sum(p['main_bytes'] for p in q if not p['mode']),
        residual_streams=sum(p['residual_bytes'] for p in q),raw_payloads=sum(p['main_bytes'] for p in q if p['mode']))
    if sum(components.values())!=len(blob):raise PacketError('accounting mismatch')
    return dict(format=FORMAT,rows=64,cols=info['cols'],split_k=1,lanes=32,packets=[{k:v for k,v in p.items() if k!='digest'} for p in q],
        total_bytes=len(blob),source_bytes=128*info['cols'],components=components,
        context_identity=info['context_identity'],source_digest=info['source_digest'],
        descriptors_bytes=4*9*8,final_lane_status_bytes=4*32*4,
        shared_slab_bytes=64*136*2,decoded_weight_hbm_bytes=0,
        scope='bounded experimental raw6; no CUDA or archive ratio qualification')


def decode_cpu(blob):
    blob,info=_parse(blob); k=info['cols']; out=np.empty((64,k),dtype='<u2')
    uppers=[[v[1] for v in table] for table in info['tables']]
    for q in info['packets']:
        ms,rs=q['main_start'],q['residual_start']
        if q['mode']:
            flat=np.frombuffer(blob[ms:rs],dtype='<u2')
        else:
            flat=np.empty(16*k,dtype='<u2'); ld=struct.unpack_from('<33I',blob,ms)
            for lane in range(32):
                pos,stop=ms+132+ld[lane],ms+132+ld[lane+1]
                state=struct.unpack_from('<I',blob,pos)[0]; pos+=4
                for t in range(k//2):
                    col=128*(t>>6)+32*(t&3)+lane; row=q['row0']+((t>>2)&15)
                    cx=info['labels'][row]*info['col_classes']+info['labels'][64+col]
                    slot=state&65535; index=bisect.bisect_right(uppers[cx],slot)
                    if index==len(uppers[cx]):raise PacketError('missing context symbol')
                    symbol,upper=info['tables'][cx][index]
                    lower=uppers[cx][index-1] if index else 0
                    state=(upper-lower)*(state>>16)+slot-lower
                    if state>=L*256:raise PacketError('invalid next state')
                    while state<L:
                        if pos>=stop:raise PacketError('truncated rANS stream')
                        state=(state<<8)|blob[pos];pos+=1
                    z=32*t+lane; bit=6*z; residual=blob[rs+bit//8]>>(bit%8)
                    if bit%8>2:residual|=blob[rs+bit//8+1]<<(8-bit%8)
                    residual&=63
                    flat[z]=(symbol<<5)|(residual&31)|((residual>>5)<<15)
                if state!=L or pos!=stop:raise PacketError('noncanonical termination')
        decoded=_row_words(flat,k)
        if _hash(decoded.tobytes())!=q['digest']:raise PacketError('decoded packet checksum mismatch')
        out[q['row0']:q['row0']+16]=decoded
    if _hash(out.tobytes()).hex()!=info['source_digest']:raise PacketError('decoded source checksum mismatch')
    return out
