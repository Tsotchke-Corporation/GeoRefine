"""Experimental exact10bitmain+raw6, 32-state Context packets (BCI1).

Opt-in standalone correctness primitive; no production loader selects this format.
Quantile labels are tensor-global. Packet models/restarts are independent. Each
warp lane owns every 32nd word and a separately bounded rANS byte stream. Compact
sparse cumulative tables contain at most 1024 entries per context, never a 65536
slot GPU LUT. Headers, labels, tables, states and checksums are all billed.
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
HEADER = struct.Struct('<4sBBBBIIII32s32s')
PACKET = struct.Struct('<4sBBHIIIII32s')
ENTRY = struct.Struct('<HI')
FORMAT = 'BCI1-exact10bitmain-raw6-sparsecdf-32lane-v1'


class PacketError(ValueError):
    """Unsupported, out-of-bounds or corrupt experimental packet."""


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


def _encode_packet(a, row0, rc, cc, nr, nc):
    flat = a.ravel()
    cols = a.shape[1]
    symbols = [((int(w) >> 5) & 1023) for w in flat]
    contexts = [int(rc[row0+i//cols])*nc+int(cc[i % cols]) for i in range(a.size)]
    counts = [{} for _ in range(nr*nc)]
    for s, cx in zip(symbols, contexts):
        counts[cx][s] = counts[cx].get(s, 0)+1
    tables = [_normalise(c) if c else {} for c in counts]
    entries = bytearray()
    offsets = [0]
    for table in tables:
        for s, (cum, freq) in table.items():
            entries += ENTRY.pack(s, cum+freq)
        offsets.append(len(entries)//ENTRY.size)
    table = struct.pack('<'+'I'*len(offsets), *offsets)+entries
    streams = []
    for lane in range(LANES):
        state, ren = RANS_L, bytearray()
        for i in reversed(range(lane, len(flat), LANES)):
            cum, freq = tables[contexts[i]][symbols[i]]
            while state >= ((RANS_L >> 16) << 8)*freq:
                ren.append(state & 255)
                state >>= 8
            state = ((state//freq) << 16)+(state % freq)+cum
        streams.append(struct.pack('<I', state)+ren[::-1])
    offsets = [0]
    for stream in streams:
        offsets.append(offsets[-1]+len(stream))
    main = struct.pack('<33I', *offsets)+b''.join(streams)
    residual = _pack6(flat)
    raw = a.tobytes()
    mode = 0
    if len(table)+len(main)+len(residual) >= len(raw):
        mode, table, main, residual = 1, b'', raw, b''
    body = PACKET.pack(b'BIP1', 1, mode, 0, row0, len(a), len(table), len(main),
                       len(residual), _hash(raw))+table+main+residual
    return body+_hash(body)


def encode(words, *, rows_per_packet=8, row_classes=16, col_classes=4):
    a = np.asarray(words)
    if a.ndim != 2 or a.dtype.kind != 'u' or a.dtype.itemsize != 2 or not all(a.shape) or a.size > MAX_WORDS:
        raise PacketError('expected bounded nonempty 2D uint16 array (at most 65536 words)')
    if (type(rows_per_packet) is not int or rows_per_packet < 1 or
            type(row_classes) is not int or type(col_classes) is not int or
            row_classes not in codec.ROWS or col_classes not in codec.COLS or row_classes*col_classes > 1024):
        raise PacketError('invalid packet rows or context classes')
    a = np.ascontiguousarray(a, dtype='<u2')
    count = (len(a)+rows_per_packet-1)//rows_per_packet
    if count > MAX_PACKETS:
        raise PacketError('too many packets')
    rc, cc = codec._labels(a, row_classes, col_classes)
    labels = rc.tobytes()+cc.tobytes()
    context = _hash(struct.pack('<IIII', *a.shape, row_classes, col_classes)+labels)
    packets = [_encode_packet(a[r:r+rows_per_packet], r, rc, cc, row_classes, col_classes)
               for r in range(0, len(a), rows_per_packet)]
    offsets = [0]
    for packet in packets:
        offsets.append(offsets[-1]+len(packet))
    body = (HEADER.pack(b'BCI1', 1, row_classes, col_classes, LANES, *a.shape, count,
                        len(labels), context, _hash(a.tobytes()))+labels+
            struct.pack('<'+'Q'*len(offsets), *offsets)+b''.join(packets))
    return body+_hash(body)


def _parse(blob):
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise PacketError('bundle must be bytes')
    size = blob.nbytes if isinstance(blob, memoryview) else len(blob)
    if not HEADER.size+32 <= size <= 8*MAX_WORDS+128*MAX_PACKETS:
        raise PacketError('invalid bounded bundle size')
    blob = bytes(blob)
    magic, ver, nr, nc, lanes, rows, cols, count, nl, context, digest = HEADER.unpack_from(blob)
    if magic != b'BCI1' or ver != 1 or lanes != LANES or nr not in codec.ROWS or nc not in codec.COLS or nr*nc > 1024:
        raise PacketError('unsupported format')
    if not rows or not cols or rows*cols > MAX_WORDS or not 1 <= count <= min(MAX_PACKETS, rows) or nl != rows+cols:
        raise PacketError('invalid geometry')
    ds = HEADER.size+nl
    start = ds+8*(count+1)
    if start > len(blob)-32 or _hash(blob[:-32]) != blob[-32:]:
        raise PacketError('bundle checksum/length mismatch')
    labels = blob[HEADER.size:ds]
    if any(x >= nr for x in labels[:rows]) or any(x >= nc for x in labels[rows:]):
        raise PacketError('invalid global label')
    if _hash(struct.pack('<IIII', rows, cols, nr, nc)+labels) != context:
        raise PacketError('global context identity mismatch')
    offsets = struct.unpack_from('<'+'Q'*(count+1), blob, ds)
    if offsets[0] or offsets[-1] != len(blob)-32-start or any(a >= b for a, b in zip(offsets, offsets[1:])):
        raise PacketError('invalid packet directory')
    packets, next_row = [], 0
    for lo, hi in zip(offsets, offsets[1:]):
        ps, end = start+lo, start+hi
        if end-ps < PACKET.size+32:
            raise PacketError('short packet')
        magic, ver, mode, reserved, row0, nrows, nt, nm, nd, pdigest = PACKET.unpack_from(blob, ps)
        if (magic != b'BIP1' or ver != 1 or mode not in (0, 1) or reserved or row0 != next_row or
                not nrows or row0+nrows > rows or PACKET.size+nt+nm+nd+32 != end-ps):
            raise PacketError('invalid packet header/coverage')
        if _hash(blob[ps:end-32]) != blob[end-32:end]:
            raise PacketError('packet checksum mismatch')
        ts, ms, rs = ps+PACKET.size, ps+PACKET.size+nt, ps+PACKET.size+nt+nm
        n = nrows*cols
        tables = []
        if mode == 1:
            if nt or nd or nm != 2*n or _hash(blob[ms:rs]) != pdigest:
                raise PacketError('invalid raw packet')
        else:
            if nt < 4*(nr*nc+1) or nm < 132+128 or nd != (6*n+7)//8:
                raise PacketError('invalid coded lengths')
            td = struct.unpack_from('<'+'I'*(nr*nc+1), blob, ts)
            es = ts+4*(nr*nc+1)
            if td[0] or 6*td[-1]+es != ms or any(a > b or b-a > 1024 for a,b in zip(td,td[1:])):
                raise PacketError('invalid sparse directory')
            for a,b in zip(td, td[1:]):
                entries = [ENTRY.unpack_from(blob, es+6*j) for j in range(a,b)]
                last_s, last_c = -1, 0
                for s,c in entries:
                    if not last_s < s < 1024 or not last_c < c <= TOTAL:
                        raise PacketError('invalid sparse CDF')
                    last_s, last_c = s,c
                if entries and last_c != TOTAL:
                    raise PacketError('incomplete sparse CDF')
                tables.append(entries)
            ld = struct.unpack_from('<33I', blob, ms)
            if ld[0] or ld[-1] != nm-132 or any(b-a < 4 for a,b in zip(ld,ld[1:])):
                raise PacketError('invalid lane directory')
            if (6*n) % 8 and blob[rs+nd-1] >> ((6*n) % 8):
                raise PacketError('nonzero residual padding')
            for j in range(32):
                state = struct.unpack_from('<I', blob, ms+132+ld[j])[0]
                if not RANS_L <= state < RANS_L*256:
                    raise PacketError('invalid initial rANS state')
        packets.append(dict(mode=mode,row0=row0,packet_rows=nrows,table_start=ts,table_bytes=nt,
                            main_start=ms,main_bytes=nm,residual_start=rs,residual_bytes=nd,
                            digest=pdigest,tables=tables,packet_bytes=end-ps))
        next_row += nrows
    if next_row != rows:
        raise PacketError('incomplete row coverage')
    return blob, dict(rows=rows,cols=cols,row_classes=nr,col_classes=nc,context_identity=context.hex(),
                      source_digest=digest.hex(),labels=labels,packets=packets)


def inspect(blob):
    blob, info = _parse(blob)
    packets = info['packets']
    components = dict(bundle_header=HEADER.size,global_labels=len(info['labels']),
                      directory=8*(len(packets)+1),packet_headers=PACKET.size*len(packets),
                      checksums=32*(len(packets)+1),tables=sum(p['table_bytes'] for p in packets),
                      main_streams_including_states=sum(p['main_bytes'] for p in packets if not p['mode']),
                      residual_streams=sum(p['residual_bytes'] for p in packets),
                      raw_payloads=sum(p['main_bytes'] for p in packets if p['mode']))
    if sum(components.values()) != len(blob):
        raise PacketError('accounting mismatch')
    return dict(format=FORMAT,rows=info['rows'],cols=info['cols'],context_identity=info['context_identity'],
                source_digest=info['source_digest'],source_bytes=2*info['rows']*info['cols'],
                total_bytes=len(blob),components=components,lanes=LANES,
                packets=[{k:v for k,v in p.items() if k not in ('digest','tables')} for p in packets])


def decode_cpu(blob, *, packet_index=None):
    blob, info = _parse(blob)
    if packet_index is not None and (type(packet_index) is not int or not 0 <= packet_index < len(info['packets'])):
        raise PacketError('invalid packet index')
    packets = info['packets'] if packet_index is None else [info['packets'][packet_index]]
    result = []
    for p in packets:
        n, cols = p['packet_rows']*info['cols'], info['cols']
        ms, rs = p['main_start'], p['residual_start']
        if p['mode']:
            out = np.frombuffer(blob[ms:rs], dtype='<u2').copy()
        else:
            out = np.empty(n, dtype='<u2')
            ld = struct.unpack_from('<33I', blob, ms)
            for lane in range(LANES):
                pos, end = ms+132+ld[lane], ms+132+ld[lane+1]
                state = struct.unpack_from('<I', blob, pos)[0]
                pos += 4
                for i in range(lane,n,LANES):
                    cx = info['labels'][p['row0']+i//cols]*info['col_classes']+info['labels'][info['rows']+i % cols]
                    table = p['tables'][cx]
                    slot = state & 65535
                    ix = bisect.bisect_right([entry[1] for entry in table], slot)
                    if ix == len(table):
                        raise PacketError('missing symbol context')
                    sym, upper = table[ix]
                    lower = table[ix-1][1] if ix else 0
                    state = (upper-lower)*(state >> 16)+slot-lower
                    while state < RANS_L:
                        if pos >= end:
                            raise PacketError('truncated rANS stream')
                        state = (state << 8)|blob[pos]
                        pos += 1
                    bit = 6*i
                    residual = blob[rs+bit//8] >> (bit % 8)
                    if bit % 8 > 2:
                        residual |= blob[rs+bit//8+1] << (8-bit % 8)
                    residual &= 63
                    out[i] = (sym << 5)|(residual & 31)|((residual >> 5) << 15)
                if state != RANS_L or pos != end:
                    raise PacketError('noncanonical rANS termination')
        if _hash(out.tobytes()) != p['digest']:
            raise PacketError('decoded packet checksum mismatch')
        result.append(out.reshape(p['packet_rows'], cols))
    out = np.concatenate(result)
    if packet_index is None and _hash(out.tobytes()).hex() != info['source_digest']:
        raise PacketError('decoded bundle checksum mismatch')
    return out


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
    name = 'context_interleaved_packet_v1'
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


def decode_cuda(blob, *, device):
    """Decode on an explicit CUDA device; fail closed on every nonzero device status.

    Returns words (CUDA uint16), status (CUDA int32 [packets,32]), and provenance.
    Parsing/hashing is host-side; entropy decoding and reconstruction are CUDA.
    No CPU decoding fallback. This initial primitive synchronizes to check status.
    """
    blob, info = _parse(blob)
    import torch
    dev = torch.device(device)
    if dev.type != 'cuda' or dev.index is None:
        raise PacketError('explicit CUDA device index required, e.g. cuda:0')
    with torch.cuda.device(dev):
        ext = extension()
        data = torch.from_numpy(np.frombuffer(blob,dtype=np.uint8).copy()).to(dev)
        keys = ('mode','row0','packet_rows','table_start','table_bytes','main_start','main_bytes','residual_start','residual_bytes')
        descriptors = torch.tensor([[p[k] for k in keys] for p in info['packets']],dtype=torch.int64,device=dev)
        words, status = ext.decode(data,descriptors,info['rows'],info['cols'],info['row_classes'],info['col_classes'],HEADER.size)
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
                          decoded_on='cuda',output_digest_verified=False)
        return dict(words=words.reshape(info['rows'],info['cols']),status=status,provenance=provenance)
