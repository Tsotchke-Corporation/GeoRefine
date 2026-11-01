"""Bounded CPU packet reference for the existing exact Context codec.

This module is not selected by any production loader. It preserves the tensor-wide
row/column quantile labels, but estimates each packet's main-symbol frequencies
locally. Each packet restarts the existing single-state integer rANS coder; the
6-bit sign/low-mantissa residual is packed verbatim. No interleaved-state, CUDA,
performance or model-compression claim is made.

The existing BCTX primitive accepts explicit row/column labels. Reusing that
primitive avoids recomputing quantiles per packet or changing its 10/6 split.
All headers, labels, frequency tables, ANS initial states, residuals, hashes and
the random-access directory are included in the returned bytes and accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np

from . import bitexact_context_codec as codec

MAX_WORDS = 1 << 20
MAX_PACKET_WORDS = 1 << 16
MAX_PACKETS = 4096
MAX_CONTEXTS = 64
BUNDLE = struct.Struct('<4sB3xI')
PACKET = struct.Struct('<4sBBHIIIIQ32s')
U64 = struct.Struct('<Q')


class PacketError(ValueError):
    """Invalid or unsupported experimental packet input."""


@dataclass(frozen=True)
class GlobalContext:
    rows: int
    cols: int
    row_classes: int
    col_classes: int
    row_labels: bytes
    col_labels: bytes
    identity: bytes


def _words(words):
    a = np.asarray(words)
    if a.ndim != 2 or a.dtype.kind != 'u' or a.dtype.itemsize != 2:
        raise PacketError('expected a two-dimensional uint16 array')
    if not all(a.shape) or a.size > MAX_WORDS or a.shape[1] > MAX_PACKET_WORDS:
        raise PacketError('reference input exceeds bounded geometry')
    return np.ascontiguousarray(a, dtype='<u2')


def global_context(words, *, row_classes=16, col_classes=4):
    """Compute the original codec's quantile labels once over the complete input."""
    a = _words(words)
    if (type(row_classes) is not int or type(col_classes) is not int or
            row_classes not in codec.ROWS or col_classes not in codec.COLS or
            row_classes * col_classes > MAX_CONTEXTS):
        raise PacketError('unsupported bounded class count')
    rc, cc = codec._labels(a, row_classes, col_classes)
    shape = struct.pack('<IIII', *a.shape, row_classes, col_classes)
    identity = hashlib.sha256(shape + rc.tobytes() + cc.tobytes()).digest()
    return GlobalContext(*a.shape, row_classes, col_classes, rc.tobytes(), cc.tobytes(), identity)


def _encode_packet(a, row0, context, lib):
    r, c = a.shape
    # The existing ctypes wrapper requires writable arrays, even though C++ reads only.
    rc = np.frombuffer(context.row_labels[row0:row0+r], dtype=np.uint8).copy()
    cc = np.frombuffer(context.col_labels, dtype=np.uint8).copy()
    nr, nc = context.row_classes, context.col_classes
    raw = a.tobytes()
    digest = hashlib.sha256(raw).digest()
    fallback = codec._frame(1, nr, nc, r, c, b'', b'', b'', b'', raw, digest)
    freq, main = codec._encode_kind(lib, a, rc, cc, nr, nc, 0)
    labels = codec.zlib.compress(rc.tobytes() + cc.tobytes(), 6)
    table = codec.zlib.compress(freq, 6)
    residual = codec._pack6(lib, a)
    coded = codec._frame(0, nr, nc, r, c, labels, table, main, residual, b'', digest)
    frame = coded if len(coded) < len(fallback) else fallback
    header = PACKET.pack(b'BCPK', 1, 0, 0, context.rows, context.cols,
                         row0, r, len(frame), context.identity)
    body = header + frame
    return body + hashlib.sha256(body).digest()


def encode(words, *, rows_per_packet=8, row_classes=16, col_classes=4):
    """Return a self-contained bundle of independently restartable full-row packets."""
    a = _words(words)
    if (type(rows_per_packet) is not int or rows_per_packet < 1 or
            min(rows_per_packet, len(a))*a.shape[1] > MAX_PACKET_WORDS):
        raise PacketError('invalid rows_per_packet or packet word bound')
    count = (len(a) + rows_per_packet - 1)//rows_per_packet
    if count > MAX_PACKETS:
        raise PacketError('too many reference packets')
    context = global_context(a, row_classes=row_classes, col_classes=col_classes)
    lib = codec._lib()
    packets = [_encode_packet(a[row0:row0+rows_per_packet], row0, context, lib)
               for row0 in range(0, len(a), rows_per_packet)]
    offsets = [0]
    for packet in packets:
        offsets.append(offsets[-1] + len(packet))
    return (BUNDLE.pack(b'BCPB', 1, count) +
            b''.join(U64.pack(x) for x in offsets) + b''.join(packets))


def _directory(blob):
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise PacketError('bundle must be bytes')
    size = blob.nbytes if isinstance(blob, memoryview) else len(blob)
    if size < BUNDLE.size or size > 4*MAX_WORDS + 256*MAX_PACKETS:
        raise PacketError('invalid bounded bundle length')
    blob = bytes(blob)
    magic, version, count = BUNDLE.unpack_from(blob)
    if magic != b'BCPB' or version != 1 or not 1 <= count <= MAX_PACKETS:
        raise PacketError('unsupported bundle header')
    if blob[5:8] != b'\0\0\0':
        raise PacketError('nonzero reserved bundle bytes')
    start = BUNDLE.size + (count+1)*U64.size
    if start > len(blob):
        raise PacketError('truncated packet directory')
    offsets = [U64.unpack_from(blob, BUNDLE.size+i*U64.size)[0] for i in range(count+1)]
    if (offsets[0] != 0 or offsets[-1] != len(blob)-start or
            any(lo >= hi for lo, hi in zip(offsets, offsets[1:]))):
        raise PacketError('invalid packet offsets')
    return blob, start, offsets


def _packet(packet):
    if len(packet) < PACKET.size + codec.HEADER.size + 64:
        raise PacketError('short packet')
    magic, ver, flags, reserved, rows, cols, row0, nrows, length, identity = PACKET.unpack_from(packet)
    if magic != b'BCPK' or ver != 1 or flags or reserved:
        raise PacketError('unsupported packet header')
    if (not rows or not cols or rows*cols > MAX_WORDS or not nrows or
            nrows*cols > MAX_PACKET_WORDS or row0+nrows > rows):
        raise PacketError('invalid bounded packet geometry')
    if length != len(packet)-PACKET.size-32:
        raise PacketError('packet length mismatch')
    if hashlib.sha256(packet[:-32]).digest() != packet[-32:]:
        raise PacketError('packet checksum mismatch')
    frame = packet[PACKET.size:-32]
    fm, fv, mode, nr, nc, r, c, ml, tl, sl, dl, _ = codec.HEADER.unpack_from(frame)
    if (fm != codec.MAGIC or fv != codec.VERSION or mode not in (0, 1) or
            nr not in codec.ROWS or nc not in codec.COLS or nr*nc > MAX_CONTEXTS or
            (r, c) != (nrows, cols)):
        raise PacketError('unsupported inner frame or geometry')
    raw_bytes = 2*r*c if mode == 1 else 0
    if mode == 1 and (ml or tl or sl or dl):
        raise PacketError('raw packet carries coded streams')
    if codec.HEADER.size + ml+tl+sl+dl+raw_bytes+32 != len(frame):
        raise PacketError('inner frame length mismatch')
    if mode == 0 and (sl < 4 or dl != (6*r*c+7)//8):
        raise PacketError('invalid main state or packed residual length')
    if hashlib.sha256(frame[:-32]).digest() != frame[-32:]:
        raise PacketError('inner frame checksum mismatch')
    return dict(rows=rows, cols=cols, row0=row0, packet_rows=nrows, mode=mode,
                row_classes=nr, col_classes=nc, context_identity=identity.hex(),
                label_bytes=ml, table_bytes=tl, main_bytes=sl, residual_bytes=dl,
                raw_bytes=raw_bytes, frame=frame, packet_bytes=len(packet))


def inspect(blob):
    """Validate framing and report every byte; frequency validity is checked on decode."""
    blob, start, offsets = _directory(blob)
    packets = [_packet(blob[start+lo:start+hi]) for lo, hi in zip(offsets, offsets[1:])]
    first = packets[0]
    next_row = 0
    for p in packets:
        if (p['rows'], p['cols'], p['context_identity'], p['row_classes'], p['col_classes']) != (
                first['rows'], first['cols'], first['context_identity'], first['row_classes'], first['col_classes']):
            raise PacketError('mixed packet contexts or geometry')
        if p['row0'] != next_row:
            raise PacketError('packet row coverage is not contiguous')
        next_row += p['packet_rows']
    if next_row != first['rows']:
        raise PacketError('packet row coverage is incomplete')
    totals = dict(bundle_header=BUNDLE.size, directory=(len(packets)+1)*8,
                  packet_headers=len(packets)*PACKET.size,
                  frame_headers=len(packets)*codec.HEADER.size,
                  checksums=len(packets)*64,
                  labels=sum(p['label_bytes'] for p in packets),
                  tables=sum(p['table_bytes'] for p in packets),
                  main_streams_including_rans_states=sum(p['main_bytes'] for p in packets),
                  residual_streams=sum(p['residual_bytes'] for p in packets),
                  raw_payloads=sum(p['raw_bytes'] for p in packets))
    if sum(totals.values()) != len(blob):
        raise PacketError('inclusive byte accounting mismatch')
    return dict(format='BCPB/BCPK-v1-reference', rows=first['rows'], cols=first['cols'],
                context_identity=first['context_identity'], source_bytes=2*first['rows']*first['cols'],
                total_bytes=len(blob), components=totals,
                packets=[{k:v for k,v in p.items() if k != 'frame'} for p in packets])


def decode_packet(blob, index, *, expected_context=None):
    """Decode only the selected packet, with no state from earlier packets."""
    blob, start, offsets = _directory(blob)
    if type(index) is not int or not 0 <= index < len(offsets)-1:
        raise PacketError('packet index outside directory')
    p = _packet(blob[start+offsets[index]:start+offsets[index+1]])
    if expected_context is not None and p['context_identity'] != expected_context:
        raise PacketError('unexpected global context identity')
    try:
        return codec.decode(p['frame'])
    except codec.CodecError as exc:
        raise PacketError(str(exc)) from exc


def decode(blob):
    info = inspect(blob)
    return np.concatenate([decode_packet(blob, i, expected_context=info['context_identity'])
                           for i in range(len(info['packets']))], axis=0)
