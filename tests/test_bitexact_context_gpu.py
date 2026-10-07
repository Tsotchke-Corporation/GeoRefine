import hashlib
import numpy as np
import pytest
from scripts import bitexact_context_codec as base
from scripts.bitexact_context_gpu import index_frame


@pytest.mark.parametrize('stride', [1, 7, 128, 1024])
def test_index_frame_binds_original_bits(stride):
    words=np.resize(np.array([0,0x8000,0x7f80,0xff80,0x7fc1,0xffe5,1,0x8001],dtype=np.uint16),(64,67))
    frame=base.encode(words,row_classes=16,col_classes=4)
    indexed=index_frame(frame,stride)
    assert indexed['source_sha256']==hashlib.sha256(words.tobytes()).hexdigest()
    assert indexed['frame_sha256']==hashlib.sha256(frame).hexdigest()
    assert indexed['shape']==words.shape
    if indexed['mode']!=1:
        assert len(indexed['main_states'])==(words.size+stride-1)//stride


def test_index_frame_raw_and_checksum_rejection():
    words=np.random.default_rng(9).integers(0,65536,(13,17),dtype=np.uint16)
    frame=base.encode(words,row_classes=16,col_classes=4)
    indexed=index_frame(frame)
    assert indexed['mode']==1
    assert np.array_equal(indexed['raw'].reshape(words.shape),words)
    with pytest.raises(base.CodecError,match='checksum'):
        index_frame(frame[:-1]+bytes([frame[-1]^1]))
    with pytest.raises(base.CodecError,match='stride'):
        index_frame(frame,0)
