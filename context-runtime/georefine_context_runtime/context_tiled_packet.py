"""BCT2 experimental exact10bitmain+raw6 Context tiles.

Global quantile labels and one shared sparse CDF serve rectangular row/K tiles.
Each tile has independent 32-lane rANS restart streams, so a selected K tile
requires no prefix entropy replay. This standalone bounded correctness primitive
is opt-in and is not selected by production loaders or fused with GEMM.
"""
from __future__ import annotations

import bisect
import hashlib
import os
from pathlib import Path
import struct

import numpy as np

from . import bitexact_context_codec as codec

MAX_WORDS = 65536
MAX_PACKETS = 4096
LANES = 32
RANS_L = 1 << 23
TOTAL = 1 << 16
HEADER = struct.Struct('<4sBBBBIIIII32s32s')
PACKET = struct.Struct('<4sBBHIIIIII32s')
ENTRY = struct.Struct('<HI')
FORMAT = 'BCT2-exact10bitmain-raw6-global-sparsecdf-32lane-v2'
DESCRIPTOR_KEYS = ('mode','row0','col0','packet_rows','packet_cols',
                   'main_start','main_bytes','residual_start','residual_bytes')


class PacketError(ValueError):
    """Unsupported, out-of-bounds or corrupt experimental tile bundle."""


def _hash(x):
    return hashlib.sha256(x).digest()


def _normalise(counts):
    n = sum(counts.values())
    freq = {s: max(1, c*TOTAL//n) for s, c in counts.items()}
    delta = TOTAL-sum(freq.values())
    order = sorted(counts, key=lambda s: (-(counts[s]*TOTAL % n), s))
    if delta > 0:
        for j in range(delta):
            freq[order[j % len(order)]] += 1
    elif delta < 0:
        order.reverse()
        j = 0
        while delta:
            s = order[j % len(order)]
            if freq[s] > 1:
                freq[s] -= 1
                delta += 1
            j += 1
    cumulative = 0
    result = {}
    for s in sorted(freq):
        result[s] = (cumulative, freq[s])
        cumulative += freq[s]
    return result


def _pack6(words):
    out = bytearray((6*len(words)+7)//8)
    for i, w in enumerate(words):
        v = ((int(w) >> 15) << 5) | (int(w) & 31)
        bit = i*6
        out[bit//8] |= (v << (bit % 8)) & 255
        if bit % 8 > 2:
            out[bit//8+1] |= v >> (8-bit % 8)
    return bytes(out)


def _global_tables(a, rc, cc, nr, nc):
    counts = [{} for _ in range(nr*nc)]
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            symbol = (int(a[row,col]) >> 5) & 1023
            cx = int(rc[row])*nc+int(cc[col])
            counts[cx][symbol] = counts[cx].get(symbol,0)+1
    tables = [_normalise(c) if c else {} for c in counts]
    entries = bytearray()
    offsets = [0]
    for table in tables:
        for symbol,(cum,freq) in table.items():
            entries += ENTRY.pack(symbol,cum+freq)
        offsets.append(len(entries)//ENTRY.size)
    return tables, struct.pack('<'+'I'*len(offsets),*offsets)+entries


def _tile_candidates(a, row0, col0, rc, cc, nc, tables):
    a = np.ascontiguousarray(a,dtype='<u2')
    flat, width = a.ravel(), a.shape[1]
    streams = []
    for lane in range(LANES):
        state, ren = RANS_L, bytearray()
        for i in reversed(range(lane,len(flat),LANES)):
            symbol = (int(flat[i]) >> 5) & 1023
            cx = int(rc[row0+i//width])*nc+int(cc[col0+i % width])
            cum,freq = tables[cx][symbol]
            while state >= ((RANS_L >> 16) << 8)*freq:
                ren.append(state & 255)
                state >>= 8
            state = ((state//freq) << 16)+(state % freq)+cum
        streams.append(struct.pack('<I',state)+ren[::-1])
    offsets = [0]
    for stream in streams:
        offsets.append(offsets[-1]+len(stream))
    main = struct.pack('<33I',*offsets)+b''.join(streams)
    residual = _pack6(flat)
    raw = a.tobytes()
    digest = _hash(raw)

    def frame(mode,main,residual):
        body = PACKET.pack(b'BTP2',2,mode,0,row0,col0,*a.shape,len(main),len(residual),digest)+main+residual
        return body+_hash(body)

    coded, fallback = frame(0,main,residual), frame(1,raw,b'')
    return (coded if len(coded)<len(fallback) else fallback), fallback


def encode(words, *, rows_per_packet=8, cols_per_packet=512, row_classes=16, col_classes=4):
    """Encode a bounded matrix; each rectangular tile restarts all 32 rANS lanes.

    Global labels and sparse CDF are serialized once. The all-raw candidate wins
    unless aggregate tile payload savings pay for the shared CDF itself.
    """
    a = np.asarray(words)
    if a.ndim != 2 or a.dtype.kind != 'u' or a.dtype.itemsize != 2 or not all(a.shape) or a.size>MAX_WORDS:
        raise PacketError('expected nonempty 2D uint16 array of at most 65536 words')
    if (type(rows_per_packet) is not int or rows_per_packet<1 or
            type(cols_per_packet) is not int or cols_per_packet<1 or
            type(row_classes) is not int or type(col_classes) is not int or
            row_classes not in codec.ROWS or col_classes not in codec.COLS or row_classes*col_classes>1024):
        raise PacketError('invalid tile extents or context classes')
    a = np.ascontiguousarray(a,dtype='<u2')
    count = ((a.shape[0]+rows_per_packet-1)//rows_per_packet)*((a.shape[1]+cols_per_packet-1)//cols_per_packet)
    if count>MAX_PACKETS:
        raise PacketError('too many tiles')
    rc,cc = codec._labels(a,row_classes,col_classes)
    labels = rc.tobytes()+cc.tobytes()
    identity = _hash(struct.pack('<IIII',*a.shape,row_classes,col_classes)+labels)
    tables,table = _global_tables(a,rc,cc,row_classes,col_classes)
    candidates = [_tile_candidates(a[r:r+rows_per_packet,c:c+cols_per_packet],r,c,rc,cc,col_classes,tables)
                  for r in range(0,a.shape[0],rows_per_packet) for c in range(0,a.shape[1],cols_per_packet)]
    if len(table)+sum(len(pair[0]) for pair in candidates) < sum(len(pair[1]) for pair in candidates):
        packets = [pair[0] for pair in candidates]
    else:
        packets,table = [pair[1] for pair in candidates],b''
    offsets = [0]
    for packet in packets:
        offsets.append(offsets[-1]+len(packet))
    body = (HEADER.pack(b'BCT2',2,row_classes,col_classes,LANES,*a.shape,count,len(labels),len(table),
                        identity,_hash(a.tobytes()))+labels+table+
            struct.pack('<'+'Q'*len(offsets),*offsets)+b''.join(packets))
    return body+_hash(body)


def _parse(blob):
    if not isinstance(blob,(bytes,bytearray,memoryview)):
        raise PacketError('bundle must be bytes')
    size = blob.nbytes if isinstance(blob,memoryview) else len(blob)
    if not HEADER.size+32<=size<=8*MAX_WORDS+128*MAX_PACKETS:
        raise PacketError('invalid bounded bundle size')
    blob = bytes(blob)
    magic,ver,nr,nc,lanes,rows,cols,count,nl,nt,identity,digest = HEADER.unpack_from(blob)
    if magic!=b'BCT2' or ver!=2 or lanes!=LANES or nr not in codec.ROWS or nc not in codec.COLS or nr*nc>1024:
        raise PacketError('unsupported tiled format')
    if not rows or not cols or rows*cols>MAX_WORDS or not 1<=count<=min(MAX_PACKETS,rows*cols) or nl!=rows+cols:
        raise PacketError('invalid geometry')
    ts = HEADER.size+nl
    ds = ts+nt
    start = ds+8*(count+1)
    if start>len(blob)-32 or _hash(blob[:-32])!=blob[-32:]:
        raise PacketError('bundle checksum/length mismatch')
    labels = blob[HEADER.size:ts]
    if any(x>=nr for x in labels[:rows]) or any(x>=nc for x in labels[rows:]):
        raise PacketError('invalid global label')
    if _hash(struct.pack('<IIII',rows,cols,nr,nc)+labels)!=identity:
        raise PacketError('global context identity mismatch')
    tables = []
    if nt:
        if nt<4*(nr*nc+1):
            raise PacketError('short global sparse directory')
        td = struct.unpack_from('<'+'I'*(nr*nc+1),blob,ts)
        es = ts+4*(nr*nc+1)
        if td[0] or 6*td[-1]+es!=ds or td[-1]>rows*cols or any(a>b or b-a>1024 for a,b in zip(td,td[1:])):
            raise PacketError('invalid global sparse directory')
        for a,b in zip(td,td[1:]):
            entries = [ENTRY.unpack_from(blob,es+6*j) for j in range(a,b)]
            last_s,last_c = -1,0
            for symbol,cum in entries:
                if not last_s<symbol<1024 or not last_c<cum<=TOTAL:
                    raise PacketError('invalid global sparse CDF')
                last_s,last_c = symbol,cum
            if entries and last_c!=TOTAL:
                raise PacketError('incomplete global sparse CDF')
            tables.append(entries)
    offsets = struct.unpack_from('<'+'Q'*(count+1),blob,ds)
    if offsets[0] or offsets[-1]!=len(blob)-32-start or any(a>=b for a,b in zip(offsets,offsets[1:])):
        raise PacketError('invalid tile directory')
    coverage = np.zeros((rows,cols),dtype=np.bool_)
    packets = []
    for lo,hi in zip(offsets,offsets[1:]):
        ps,end = start+lo,start+hi
        if end-ps<PACKET.size+32:
            raise PacketError('short tile')
        magic,ver,mode,reserved,row0,col0,pr,pc,nm,nd,pdigest = PACKET.unpack_from(blob,ps)
        if (magic!=b'BTP2' or ver!=2 or mode not in (0,1) or reserved or not pr or not pc or
                row0+pr>rows or col0+pc>cols or PACKET.size+nm+nd+32!=end-ps):
            raise PacketError('invalid tile header/geometry')
        tile = coverage[row0:row0+pr,col0:col0+pc]
        if np.any(tile):
            raise PacketError('overlapping tile coverage')
        tile[:] = True
        if _hash(blob[ps:end-32])!=blob[end-32:end]:
            raise PacketError('tile checksum mismatch')
        ms,rs,n = ps+PACKET.size,ps+PACKET.size+nm,pr*pc
        if mode:
            if nd or nm!=2*n or _hash(blob[ms:rs])!=pdigest:
                raise PacketError('invalid raw tile')
        else:
            if not nt or nm<132+128 or nd!=(6*n+7)//8:
                raise PacketError('invalid coded lengths or missing global CDF')
            ld = struct.unpack_from('<33I',blob,ms)
            if ld[0] or ld[-1]!=nm-132 or any(b-a<4 for a,b in zip(ld,ld[1:])):
                raise PacketError('invalid lane directory')
            if (6*n)%8 and blob[rs+nd-1] >> ((6*n)%8):
                raise PacketError('nonzero residual padding')
            for lane in range(LANES):
                state = struct.unpack_from('<I',blob,ms+132+ld[lane])[0]
                if not RANS_L<=state<RANS_L*256:
                    raise PacketError('invalid initial rANS state')
        packets.append(dict(mode=mode,row0=row0,col0=col0,packet_rows=pr,packet_cols=pc,
                            main_start=ms,main_bytes=nm,residual_start=rs,residual_bytes=nd,
                            digest=pdigest,packet_bytes=end-ps,packet_start=ps))
    if not np.all(coverage):
        raise PacketError('incomplete tile coverage')
    if nt and all(p['mode'] for p in packets):
        raise PacketError('unused global CDF in all-raw bundle')
    return blob,dict(rows=rows,cols=cols,row_classes=nr,col_classes=nc,context_identity=identity.hex(),
                     source_digest=digest.hex(),labels=labels,table_start=ts,table_bytes=nt,
                     tables=tables,packets=packets)


def inspect(blob):
    """Validate complete coverage/framing and bill every byte, shared CDF once."""
    blob,info = _parse(blob)
    packets = info['packets']
    components = dict(bundle_header=HEADER.size,global_labels=len(info['labels']),
                      global_sparse_cdf=info['table_bytes'],directory=8*(len(packets)+1),
                      packet_headers=PACKET.size*len(packets),checksums=32*(len(packets)+1),
                      main_streams_including_states=sum(p['main_bytes'] for p in packets if not p['mode']),
                      residual_streams=sum(p['residual_bytes'] for p in packets),
                      raw_payloads=sum(p['main_bytes'] for p in packets if p['mode']))
    if sum(components.values())!=len(blob):
        raise PacketError('accounting mismatch')
    return dict(format=FORMAT,rows=info['rows'],cols=info['cols'],context_identity=info['context_identity'],
                source_digest=info['source_digest'],source_bytes=2*info['rows']*info['cols'],total_bytes=len(blob),
                components=components,lanes=LANES,table_start=info['table_start'],table_bytes=info['table_bytes'],
                shared_cdf_entries=sum(len(t) for t in info['tables']),
                packets=[{k:v for k,v in q.items() if k!='digest'} for q in packets])


def _selected(info,packet_index):
    if packet_index is None:
        return info['packets'],(0,0),(info['rows'],info['cols'])
    if type(packet_index) is not int or not 0<=packet_index<len(info['packets']):
        raise PacketError('invalid tile index')
    q = info['packets'][packet_index]
    return [q],(q['row0'],q['col0']),(q['packet_rows'],q['packet_cols'])


def decode_cpu(blob, *, packet_index=None):
    """Decode whole matrix or one tile; other tiles' ANS streams are not decoded."""
    blob,info = _parse(blob)
    packets,origin,shape = _selected(info,packet_index)
    result = np.empty(shape,dtype='<u2')
    for q in packets:
        n,width = q['packet_rows']*q['packet_cols'],q['packet_cols']
        ms,rs = q['main_start'],q['residual_start']
        if q['mode']:
            out = np.frombuffer(blob[ms:rs],dtype='<u2').copy()
        else:
            out = np.empty(n,dtype='<u2')
            ld = struct.unpack_from('<33I',blob,ms)
            for lane in range(LANES):
                pos,end = ms+132+ld[lane],ms+132+ld[lane+1]
                state = struct.unpack_from('<I',blob,pos)[0]
                pos += 4
                for i in range(lane,n,LANES):
                    cx = (info['labels'][q['row0']+i//width]*info['col_classes']+
                          info['labels'][info['rows']+q['col0']+i%width])
                    table = info['tables'][cx]
                    slot = state & 65535
                    ix = bisect.bisect_right([entry[1] for entry in table],slot)
                    if ix==len(table):
                        raise PacketError('missing symbol context')
                    symbol,upper = table[ix]
                    lower = table[ix-1][1] if ix else 0
                    state = (upper-lower)*(state >> 16)+slot-lower
                    while state<RANS_L:
                        if pos>=end:
                            raise PacketError('truncated rANS stream')
                        state = (state << 8)|blob[pos]
                        pos += 1
                    bit = 6*i
                    residual = blob[rs+bit//8] >> (bit%8)
                    if bit%8>2:
                        residual |= blob[rs+bit//8+1] << (8-bit%8)
                    residual &= 63
                    out[i] = (symbol << 5)|(residual & 31)|((residual >> 5) << 15)
                if state!=RANS_L or pos!=end:
                    raise PacketError('noncanonical rANS termination')
        if _hash(out.tobytes())!=q['digest']:
            raise PacketError('decoded tile checksum mismatch')
        rr,cc = q['row0']-origin[0],q['col0']-origin[1]
        result[rr:rr+q['packet_rows'],cc:cc+q['packet_cols']] = out.reshape(q['packet_rows'],q['packet_cols'])
    if packet_index is None and _hash(result.tobytes()).hex()!=info['source_digest']:
        raise PacketError('decoded bundle checksum mismatch')
    return result


def kernel_build_plan():
    from glc_serve import kernel_cache
    arch = os.environ.get('TORCH_CUDA_ARCH_LIST')
    if not arch:
        import torch
        major, minor = torch.cuda.get_device_capability()
        arch = f'{major}.{minor}'
        # Match the actual torch loader target with the versioned cache key.
        # Explicit operator architecture lists are preserved.
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch
    base = Path(__file__).with_suffix('')
    name = 'context_tiled_packet_v2'
    sources = tuple(str(base.with_suffix(s)) for s in ('.cpp','.cu'))
    spec = kernel_cache.ExtensionSpec(name=name,sources=sources,inline_sources=(Path(__file__).read_text(),),flags=('-O3','-O3','-lineinfo'),arch=arch,
                                     extra=(('wire_format',FORMAT),))
    return name, kernel_cache.extension_dir(spec), spec


def extension():
    from glc_serve import kernel_cache
    from torch.utils.cpp_extension import load
    name, directory, spec = kernel_build_plan()
    return kernel_cache.build(name,directory,lambda: load(name=name,sources=list(spec.sources),
                              build_directory=str(directory),extra_cflags=['-O3'],
                              extra_cuda_cflags=['-O3','-lineinfo'],verbose=False),spec=spec)


def decode_cuda(blob, *, device, packet_index=None):
    """Decode on an explicit CUDA device; fail closed on every nonzero device status.

    Returns words (CUDA uint16), status (CUDA int32 [packets,32]), and provenance.
    Parsing/hashing is host-side; entropy decoding and reconstruction are CUDA.
    A selected tile produces only its local rectangle, with no ANS prefix replay.
    The current wrapper validates/transfers the whole bundle even for one tile.
    No CPU decoding fallback. This initial primitive synchronizes to check status.
    """
    blob, info = _parse(blob)
    packets, origin, shape = _selected(info,packet_index)
    import torch
    dev = torch.device(device)
    if dev.type != 'cuda' or dev.index is None:
        raise PacketError('explicit CUDA device index required, e.g. cuda:0')
    with torch.cuda.device(dev):
        ext = extension()
        data = torch.from_numpy(np.frombuffer(blob,dtype=np.uint8).copy()).to(dev)
        descriptors = torch.tensor([[p[k] for k in DESCRIPTOR_KEYS] for p in packets],dtype=torch.int64,device=dev)
        words, status = ext.decode(data,descriptors,info['rows'],info['cols'],info['row_classes'],info['col_classes'],
                                   HEADER.size,info['table_start'],info['table_bytes'],*origin,*shape)
        codes = status.cpu().reshape(-1).tolist()
        if any(codes):
            raise PacketError(f'CUDA packet status failure: {codes}')
        from glc_serve import kernel_cache
        name, directory, spec = kernel_build_plan()
        base = Path(__file__).with_suffix('')
        bindings = {base.with_suffix(s).name: hashlib.sha256(base.with_suffix(s).read_bytes()).hexdigest()
                    for s in ('.py','.cpp','.cu','.cuh')}
        binary = Path(ext.__file__).resolve()
        provenance = dict(source_bindings=bindings,build_key=kernel_cache.spec_digest(spec),
                          extension_path=str(binary),extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                          format=FORMAT,device=str(dev),source_digest=info['source_digest'],
                          context_identity=info['context_identity'],kernel_cache_key=kernel_cache.spec_digest(spec),
                          extension=name,build_directory=str(directory),torch_version=torch.__version__,
                          cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name(dev),
                          gpu_allocation_bytes=dict(serialized_blob=data.numel(),
                              descriptors=descriptors.numel()*descriptors.element_size(),
                              decoded_output=words.numel()*words.element_size(),
                              lane_status=status.numel()*status.element_size()),
                          decoded_on='cuda',output_digest_verified=False,packet_index=packet_index,
                          output_origin=origin,output_shape=shape,
                          output_expected_sha256=packets[0]['digest'].hex() if packet_index is not None else info['source_digest'])
        return dict(words=words.reshape(shape),status=status,provenance=provenance)
