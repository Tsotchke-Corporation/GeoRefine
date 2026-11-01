"""Synthetic byte-for-byte native packing proof; no GPU or trained tensors."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from glc_loader import codec_v21 as cv, codec_v21_native as native


@pytest.fixture(autouse=True)
def selected_environment(monkeypatch):
    cache=Path(__file__).resolve().parents[2]/'.scratch/native-pack-tests'
    monkeypatch.setenv('GEOREFINE_TBE21_NATIVE_CACHE',str(cache))
    monkeypatch.delenv('GEOREFINE_TBE21_NATIVE_PACK',raising=False)
    monkeypatch.setattr(native,'_CONFIG_CACHE',None)


def fixtures():
    rng=np.random.default_rng(21021)
    specials=np.array([0,0x8000,0x7f80,0xff80,0x7fc0,0xffc0,0x7f81,0xffff,1,0x8001,0x7f,0x807f,0x80,0x7f7f],np.uint16)
    normal=(rng.standard_normal((24,5120)).astype(np.float32).view(np.uint32)>>16).astype(np.uint16)
    exps=rng.integers(0,16,(32,256),dtype=np.uint16)
    return {'every_word':np.arange(65536,dtype=np.uint16).reshape(256,256),
        'specials':np.resize(specials,(17,513)),'single':np.full((32,256),0x3f80,np.uint16),
        'ties':(exps<<7)|rng.integers(0,128,exps.shape,dtype=np.uint16),
        'raw':rng.integers(0,65536,(17,512),dtype=np.uint16),'normal_5120':normal,
        'flat':rng.integers(0,65536,4101,dtype=np.uint16),
        'two_clusters':np.vstack((np.tile(np.arange(16,dtype=np.uint16),(16,16))<<7,
                                  np.tile(np.arange(128,144,dtype=np.uint16),(16,16))<<7))}


@pytest.mark.parametrize('name',list(fixtures()))
@pytest.mark.parametrize('target,max_cb',[(1<<23,2),(129,2),(1<<23,1)])
def test_native_serialization_is_byte_identical(monkeypatch,name,target,max_cb):
    words=fixtures()[name]
    monkeypatch.setenv('GEOREFINE_TBE21_NATIVE_PACK','0');reference=cv.encode_tensor(name,words,target_elems=target,max_cb=max_cb)
    monkeypatch.setenv('GEOREFINE_TBE21_NATIVE_PACK','1');actual=cv.encode_tensor(name,words,target_elems=target,max_cb=max_cb)
    assert actual.header()==reference.header()
    for key,value in reference.streams().items():
        assert actual.streams()[key].dtype==value.dtype and actual.streams()[key].shape==value.shape
        assert actual.streams()[key].tobytes()==value.tobytes(),key
    assert cv.serialize(actual)==cv.serialize(reference)
    if name=='two_clusters':assert actual.counts['ncb']==max_cb
    assert np.array_equal(cv.decode_tensor(actual,check_crc=True,check_sha=True),words)
    assert '_native_packer_receipt' not in json.dumps(actual.header())
    receipt=actual._native_packer_receipt
    assert receipt['recipe']['abi']==receipt['recipe']['threads']==1
    assert receipt['binary_sha256']==hashlib.sha256(Path(receipt['binary_path']).read_bytes()).hexdigest()
    assert set(receipt['recipe']['source_sha256'])=={'codec_v21.py','codec_v21_native.py','codec_v21_native.cpp'}
    assert receipt['loaded_binary_sha256']==receipt['binary_sha256']
    assert receipt['compiled_cpp_sha256']==receipt['recipe']['source_sha256']['codec_v21_native.cpp']


def test_default_off_never_builds_and_raw_dtypes_unchanged(monkeypatch):
    def no_build(*a,**k):raise AssertionError('default off unexpectedly built native library')
    monkeypatch.setattr(native,'library',no_build);assert native.enabled() is False
    cv.encode_tensor('rows',np.full((32,128),0x3f80,np.uint16))
    for dtype,words in [('F32',np.arange(16,dtype=np.float32)),('F16',np.arange(16,dtype=np.uint16)),('BF16',np.arange(16,dtype=np.uint16))]:
        monkeypatch.setenv('GEOREFINE_TBE21_NATIVE_PACK','0');a=cv.encode_tensor('raw',words,dtype=dtype)
        monkeypatch.setenv('GEOREFINE_TBE21_NATIVE_PACK','1');b=cv.encode_tensor('raw',words,dtype=dtype)
        assert cv.serialize(a)==cv.serialize(b)


@pytest.mark.parametrize('value',['','true','01','2',' 1'])
def test_malformed_selector_fails_closed(monkeypatch,value):
    monkeypatch.setenv('GEOREFINE_TBE21_NATIVE_PACK',value)
    with pytest.raises(native.NativePackError,match='exactly 0'):cv.encode_tensor('x',np.zeros(10,np.uint16))


def prepared():return np.arange(128,dtype=np.uint16).reshape(1,128),np.full((1,256),16,np.uint8),np.zeros(1,np.uint8)


def test_wrapper_validation_precedes_build(monkeypatch):
    rows,lut,sel=prepared();monkeypatch.setattr(native,'library',lambda *a:pytest.fail('invalid input reached compiler'))
    for r,l,s in [(rows.astype(np.int16),lut,sel),(rows[:,:127],lut,sel),
                  (np.zeros((2,128),np.uint16)[:,::-1],lut,np.zeros(2,np.uint8)),
                  (rows,lut.astype(np.uint16),sel),(rows,lut,np.zeros(2,np.uint8)),(rows,lut,sel.astype(bool))]:
        with pytest.raises(native.NativePackError):native.pack(r,l,s)


def test_native_capacity_symbols_offsets_and_overflow():
    rows,lut,sel=prepared();lib,_=native.library();p=native._pointer
    U8,U16,U32,U64=ctypes.c_uint8,ctypes.c_uint16,ctypes.c_uint32,ctypes.c_uint64
    rowoff=np.full(1,123,np.uint32);stats=np.zeros(7,np.uint64)
    def args():return [p(rows,U16),rows.size,1,128,p(lut,U8),lut.size,1,p(sel,U8),sel.size,p(rowoff,U32),rowoff.size]
    bad=args();bad[1]-=1
    assert lib.t21_plan(*bad,p(stats,U64),7)==1 and rowoff[0]==123
    sel[0]=1;assert lib.t21_plan(*args(),p(stats,U64),7)==3;sel[0]=0
    lut[0,0]=17;assert lib.t21_plan(*args(),p(stats,U64),7)==3;lut[0,0]=16
    assert lib.t21_plan(*args(),p(stats,U64),6)==2
    assert lib.t21_plan(*args(),p(stats,U64),7)==0
    off,nxt=U32(),U64()
    assert lib.t21_advance_row(2**32-2,32,ctypes.byref(off),ctypes.byref(nxt))==0 and nxt.value==2**32-1
    for base,bits in [(2**32-2,33),(2**32,0),(0,2**64-1)]:
        assert lib.t21_advance_row(base,bits,ctypes.byref(off),ctypes.byref(nxt))==4
    smb=np.full(128,0xa5,np.uint8);l1=np.full(16,0xa5a5,np.uint16)
    ovf=np.full(int(stats[6])+2,0xdeadbeef,np.uint32);short=ovf[1:-2]
    assert lib.t21_fill(*args(),p(smb,U8),128,p(l1,U16),16,p(short,U32),short.size)==2
    assert ovf[0]==ovf[-1]==ovf[-2]==0xdeadbeef
    rowoff[0]=1
    assert lib.t21_fill(*args(),p(smb,U8),128,p(l1,U16),16,p(short,U32),short.size)==5


def test_selected_native_error_never_falls_back(monkeypatch):
    lib,receipt=native.library()
    monkeypatch.setattr(native,'library',lambda:(SimpleNamespace(t21_plan=lib.t21_plan,t21_fill=lambda *args:2),receipt))
    monkeypatch.setenv('GEOREFINE_TBE21_NATIVE_PACK','1')
    monkeypatch.setattr(cv,'_levels',lambda *a:pytest.fail('native failure fell back'))
    with pytest.raises(native.NativePackError,match='fill failed'):cv.encode_tensor('x',np.zeros((32,128),np.uint16))


def test_cache_hash_and_identity_timeout_fail_closed(monkeypatch,tmp_path):
    lib,receipt=native.library(tmp_path/'.scratch/cache');binary=Path(receipt['binary_path'])
    # Never truncate a loaded inode: Linux may still execute its mapped ELF
    # finalizers at exit. Replace the path while retaining the loaded mapping.
    replacement=binary.with_name(binary.name+'.tampered')
    replacement.write_bytes(binary.read_bytes()+b'tampered');replacement.replace(binary)
    with pytest.raises(native.NativePackError,match='identity mismatch'):native.library(tmp_path/'.scratch/cache')
    native._compiler_version.cache_clear()
    def timeout(*args,**kw):raise subprocess.TimeoutExpired(args[0],kw['timeout'])
    monkeypatch.setattr(native.subprocess,'run',timeout)
    with pytest.raises(native.NativePackError,match='identity exceeded 10s'):native.library(tmp_path/'.scratch/timeout')


def test_compile_timeout_typed_and_partial_removed(monkeypatch,tmp_path):
    recipe=native._recipe();monkeypatch.setattr(native,'_recipe',lambda:recipe)
    def timeout(*args,**kw):
        assert kw['timeout']==60
        raise subprocess.TimeoutExpired(args[0],kw['timeout'])
    monkeypatch.setattr(native.subprocess,'run',timeout)
    with pytest.raises(native.NativePackError,match='compile exceeded 60s'):native.library(tmp_path/'.scratch/compile-timeout')
    assert not list(tmp_path.rglob('*.partial')) and not list(tmp_path.rglob('build.json'))


def test_coherent_cache_replacement_rejects_loaded_identity(tmp_path):
    cache=tmp_path/'.scratch/replaced';_,receipt=native.library(cache)
    binary=Path(receipt['binary_path'])
    # A new inode preserves Linux's live library mapping through finalization.
    replacement=binary.with_name(binary.name+'.changed')
    replacement.write_bytes(binary.read_bytes()+b'changed');replacement.replace(binary)
    receipt['binary_sha256']=native._digest(binary)
    record={key:receipt[key] for key in ('recipe','key','binary_path','binary_sha256')}
    (binary.parent/'build.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    with pytest.raises(native.NativePackError,match='loaded binary identity changed'):
        native.library(cache)


def test_compile_rechecks_source_and_compiler_before_publish(monkeypatch,tmp_path):
    recipe=native._recipe();recipes=iter([recipe,{**recipe,'compiler_version':'changed'}])
    monkeypatch.setattr(native,'_recipe',lambda:next(recipes))
    def built(command,**kw):
        Path(command[-1]).write_bytes(b'never published')
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(native.subprocess,'run',built)
    with pytest.raises(native.NativePackError,match='identity changed during build'):
        native.library(tmp_path/'.scratch/changed')
    assert not list(tmp_path.rglob('*.partial')) and not list(tmp_path.rglob('build.json'))
    assert not list(tmp_path.rglob('tbe21_pack.*'))


def test_profile_change_and_corrupt_receipt_fail_closed(monkeypatch,tmp_path):
    _,receipt=native.library(tmp_path/'.scratch/corrupt')
    (Path(receipt['binary_path']).parent/'build.json').write_text('{broken')
    with pytest.raises(native.NativePackError,match='receipt is unreadable'):
        native.library(tmp_path/'.scratch/corrupt')
    monkeypatch.setenv('GEOREFINE_TBE21_NATIVE_PACK','1')
    monkeypatch.setattr(cv,'PROFILE',(2,2,2,4))
    with pytest.raises(native.NativePackError,match='codec profile'):
        cv.encode_tensor('profile',np.zeros((32,128),np.uint16))


def test_compiler_snapshot_change_cannot_publish(monkeypatch,tmp_path):
    recipe=native._recipe();monkeypatch.setattr(native,'_recipe',lambda:recipe)
    def built(command,**kw):
        source=Path(command[-3]);assert source!=native._SOURCE
        assert native._digest(source)==recipe['source_sha256']['codec_v21_native.cpp']
        source.write_text('changed while compiling')
        Path(command[-1]).write_bytes(b'never published')
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(native.subprocess,'run',built)
    with pytest.raises(native.NativePackError,match='identity changed during build'):
        native.library(tmp_path/'.scratch/snapshot')
    assert not list(tmp_path.rglob('*.partial')) and not list(tmp_path.rglob('.packer-source-*.cpp'))
