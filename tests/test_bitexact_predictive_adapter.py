import hashlib, importlib.util, json, lzma, struct, sys, types
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('predictor_serving',ROOT.parent/'scripts'/'bitexact_predictive_serving.py')
serving=importlib.util.module_from_spec(spec);sys.modules[spec.name]=serving;spec.loader.exec_module(serving)

def sha(b): return hashlib.sha256(b).hexdigest()

def test_packaged_module_load_does_not_change_inventory(tmp_path):
    source=tmp_path/'decoder.py';source.write_text('VALUE = 7\n')
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    assert serving._load_module(source,'immutable_decoder_probe').VALUE==7
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()}==before

def test_default_native_adapter_uses_package_backend_and_external_cache(tmp_path,monkeypatch):
    package=tmp_path/'package';(package/'decoder').mkdir(parents=True)
    codec=types.SimpleNamespace(CACHE_ROOT=None,BUILD=None)
    backend=types.ModuleType('bitexact_predictive_gpu');backend.CudaPredictiveTensor=object
    backend.codec=codec;backend.CACHE_ROOT=None
    monkeypatch.setitem(sys.modules,'bitexact_predictive_gpu',backend)
    monkeypatch.setattr(serving.sys,'path',list(serving.sys.path))
    monkeypatch.delenv('BITEXACT_PREDICTIVE_CACHE_DIR',raising=False)
    assert serving._default_predictive_factories(package) is object
    assert backend.CACHE_ROOT==package.parent/'.scratch'/'bitexact-predictive-native-cache'
    assert codec.CACHE_ROOT==backend.CACHE_ROOT and codec.BUILD==backend.CACHE_ROOT/'build'
    assert not (package/'.scratch').exists()

def frame(a,magic=b'FRM!'):
    a=np.ascontiguousarray(a,dtype='<u2');a=a.reshape(1,-1) if a.ndim==1 else a;return magic+struct.pack('<II',*a.shape)+a.tobytes()
def unpack(b):
    r,c=struct.unpack_from('<II',b,4);return np.frombuffer(b,dtype='<u2',offset=12).copy().reshape(r,c)
def bf(words): return torch.from_numpy(np.ascontiguousarray(words,dtype='<u2')).view(torch.bfloat16)

class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__();self.up=nn.Linear(2,3);self.gate=nn.Linear(2,3);self.down=nn.Linear(3,2)

class Native:
    def __init__(self,payload,ref=None):
        self.value=unpack(payload);self.shape=self.value.shape;self.frame_sha256=sha(payload);self.source_sha256=sha(self.value.tobytes());self.resident_bytes=17;self.ref=ref;self.calls=0
    def decode(self,*args,**kwargs): self.calls+=1;return bf(self.value)
    def gather_rows(self,ids,**kwargs): return self.decode()[ids]

def fixture(tmp_path,monkeypatch,mode='single'):
    pkg=tmp_path/'pkg';(pkg/'decoder').mkdir(parents=True);(pkg/'frames').mkdir();(pkg/'predictive-frames').mkdir()
    side=pkg/'sidecars';side.mkdir();(side/'config.json').write_text('{}')
    up=np.array([[0x3f80,0x3f81],[0x3f82,0x3f83],[0x3f84,0x3f85]],dtype='<u2')
    gate=np.array([[0x3f86,0x3f87],[0x3f88,0x3f89],[0x3f8a,0x3f8b]],dtype='<u2')
    down=np.array([[0x3f90,0x3f91,0x3f92],[0x3f93,0x3f94,0x3f95]],dtype='<u2')
    arrays={'up.weight':up,'gate.weight':gate,'down.weight':down,
            'up.bias':np.array([0x0000,0x0000,0x0000],dtype='<u2'),
            'gate.bias':np.array([0x0000,0x0000,0x0000],dtype='<u2'),
            'down.bias':np.array([0x0000,0x0000],dtype='<u2')}
    rows=[];frame_map={}
    for name,a in arrays.items():
        original=np.ascontiguousarray(a); raw=original.tobytes(); payload=frame(original)
        rel='frames/'+name.replace('.','_')+'.bctx';(pkg/rel).write_bytes(payload);frame_map[payload]=original
        rows.append({'name':name,'dtype':'BF16','shape':list(a.shape),'source_bytes':len(raw),'source_sha256':sha(raw),'frame':rel,'frame_bytes':len(payload),'frame_sha256':sha(payload),'source_shard':'model.safetensors'})
    coeff_words=None; derived=None
    refs=[{'name':'up.weight','sha256':sha(arrays['up.weight'].tobytes())}]
    if mode=='mixbf16':
        refs.append({'name':'gate.weight','sha256':sha(arrays['gate.weight'].tobytes())})
        coeff_words=np.zeros((3,3),dtype='<u2');coeff_words[:,0]=0x3f80;coeff_words[:,2]=0x3e00
        # Test mix predictor implements exact BF16 words for the fixture.
        derived=arrays['up.weight'].copy()
        coeff_path='predictive-coefficients/down.bf16';(pkg/'predictive-coefficients').mkdir();(pkg/coeff_path).write_bytes(coeff_words.tobytes())
    candidate_frame=frame(down.T,magic=b'PPCX'); rel='predictive-frames/down.ppcx';(pkg/rel).write_bytes(candidate_frame);frame_map[candidate_frame]=down.T.copy()
    selected={'down.weight':{'schema':'ppcx-reference-conditioned-bf16-v1','mode':mode,'name':'down.weight','source_sha256':sha(down.tobytes()),'source_bytes':down.nbytes,'shape':list(down.shape),'frame':rel,'frame_bytes':len(candidate_frame),'frame_sha256':sha(candidate_frame),'saved_bytes':10,'references':refs,'transform':'target transpose'}}
    if mode=='mixbf16':
        selected['down.weight']['coefficients']={'path':coeff_path,'bytes':coeff_words.nbytes,'sha256':sha(coeff_words.tobytes()),'shape':[3,3]}
        selected['down.weight']['derived_reference_sha256']=sha(derived.tobytes())
    (pkg/'baseline-manifest.json').write_text(json.dumps({'tensors':rows}))
    (pkg/'manifest.json').write_text(json.dumps({'schema':serving.PREDICTIVE_SCHEMA,'complete':True,'verify_on_load':True,'experimental_codec':'ppcx-reference-conditioned-bf16-v1','selected_candidates':selected,'metadata_assets':{}}))
    (pkg/'decoder'/'predictor_package.py').write_text('''\nimport numpy as np, json, hashlib, math\ndef read_baseline_manifest(package):return json.loads((package/'baseline-manifest.json').read_text())\ndef read_coefficients(package,candidate):\n c=candidate['coefficients'];raw=(package/c['path']).read_bytes()\n if len(raw)!=c['bytes'] or hashlib.sha256(raw).hexdigest()!=c['sha256']:raise ValueError('mixed coefficient checksum mismatch')\n return np.frombuffer(raw,dtype='<u2').reshape(c['shape']).copy()\ndef unpack(b):\n import struct\n r,c=struct.unpack_from('<II',b,4);return np.frombuffer(b,dtype='<u2',offset=12).copy().reshape(r,c)\nclass BaseCodec:\n def decode(self,b):return unpack(b)\nclass PairCodec:\n def decode(self,b,ref):return unpack(b)\nclass Base:\n def decode_codec(self,package,cache_dir=None):return BaseCodec()\ndef baseline_module(package=None):return Base()\ndef load_pair_codec(package,cache):return PairCodec()\ndef load_mix_helper(package):return type('Mix',(),{'predict':staticmethod(lambda a,b,c:a.copy())})\ndef verify(*a,**k):return {}\n''')
    monkeypatch.setattr(serving,'prepare_metadata',lambda package,cache:side)
    def host_decode(payload): return frame_map[payload].copy()
    def context_factory(payload,**kw): return Native(payload)
    def predictive_factory(payload,reference,**kw): return Native(payload,reference)
    def mix_predictor(a,b,c):
        if isinstance(a,torch.Tensor):
            assert c.shape==(3,3) and int(c.view(torch.uint16)[0,0])==0x3f80
            au=a.contiguous().view(torch.uint16).cpu().numpy();cu=c.contiguous().view(torch.uint16).cpu().numpy()
            result=au.copy()
            # Coefficients in this fixture are alpha=1, beta=0, intercept=0.25.
            result[:,0]=0x3f80
            return bf(result)
        return np.ascontiguousarray(a,dtype='<u2')
    model,receipt=serving.load_predictive_model(pkg,device='cpu',dtype=torch.bfloat16,tensor_factory=None,context_factory=context_factory,predictive_factory=predictive_factory,model_factory=lambda c,a:FixtureModel(),config_loader=lambda p:{},empty_weights_factory=lambda:serving._NullContext(),host_decoder=host_decode,mix_predictor=mix_predictor,package_verifier=lambda *a,**k:None,log=lambda _:None,loader_workers=4)
    return model,receipt,arrays,selected['down.weight'],frame_map

def test_orientation_reference_reuse_and_no_dense_weight_fallback(tmp_path,monkeypatch):
    model,receipt,arrays,candidate,_=fixture(tmp_path,monkeypatch,'single')
    assert isinstance(model.down,serving.BctxLinear)
    assert model.down.context_weight.shape==(2,3)
    assert model.down.context_weight.references[0] is model.up.context_weight
    assert 'down.weight' not in dict(model.named_parameters())
    decoded=model.down.context_weight.decode().view(torch.uint16).cpu().numpy()
    assert np.array_equal(decoded,arrays['down.weight'])
    assert model.up.context_weight.native.calls==1
    assert receipt['resident_context_bytes']==sum(x.context_weight.resident_bytes for x in (model.up,model.gate,model.down))
    assert receipt['context_count']==3

def test_mixed_coefficients_are_packaged_and_used_at_runtime(tmp_path,monkeypatch):
    model,receipt,arrays,candidate,_=fixture(tmp_path,monkeypatch,'mixbf16')
    tensor=model.down.context_weight
    assert len(tensor.references)==2 and tensor.coefficients.shape==(3,3)
    decoded=tensor.decode().view(torch.uint16).cpu().numpy()
    assert np.array_equal(decoded,arrays['down.weight'])
    assert receipt['candidate_count']==1
    assert tensor.resident_bytes==17+tensor.coefficients.numel()*2

def test_rejects_candidate_source_and_coefficient_sha_errors(tmp_path,monkeypatch):
    pkg=tmp_path/'badsource';model,receipt,arrays,candidate,_=fixture(tmp_path/'source',monkeypatch,'single')
    # Source pins are checked before the validator/factories can serve a selected frame.
    path=tmp_path/'source'/'pkg'/'manifest.json';doc=json.loads(path.read_text());doc['selected_candidates']['down.weight']['source_sha256']='0'*64;path.write_text(json.dumps(doc))
    with pytest.raises(serving.ContextServingError,match='source identity'):
        serving.load_predictive_model(tmp_path/'source'/'pkg',device='cpu',model_factory=lambda c,a:FixtureModel(),config_loader=lambda p:{},empty_weights_factory=lambda:serving._NullContext(),package_verifier=lambda *a,**k:None)
    # A separate mixed package catches coefficient pin tampering before native construction.
    (tmp_path/'coeff').mkdir(); model2,_,_,_,_=fixture(tmp_path/'coeff',monkeypatch,'mixbf16')
    p=tmp_path/'coeff'/'pkg'/'manifest.json';m=json.loads(p.read_text());m['selected_candidates']['down.weight']['coefficients']['sha256']='0'*64;p.write_text(json.dumps(m))
    with pytest.raises(serving.ContextServingError,match='coefficient identity'):
        serving.load_predictive_model(tmp_path/'coeff'/'pkg',device='cpu',context_factory=lambda *a,**k:Native(a[0]),predictive_factory=lambda *a,**k:Native(a[0]),model_factory=lambda c,a:FixtureModel(),config_loader=lambda p:{},empty_weights_factory=lambda:serving._NullContext(),host_decoder=lambda b:unpack(b),mix_predictor=lambda a,b,c:a,package_verifier=lambda *a,**k:None)

def test_predictive_mtp_uses_hash_checked_external_baseline_manifest(tmp_path,monkeypatch):
    pkg=tmp_path/'pkg';(pkg/'decoder').mkdir(parents=True);(pkg/'frames').mkdir()
    names=[f'mtp.tensor.{i}' for i in range(15)];rows=[]
    for i,name in enumerate(names):
        raw=f'frame-{i}'.encode();frame_path=pkg/'frames'/f'{i}.bctx';frame_path.write_bytes(raw)
        rows.append({'name':name,'shape':[1],'frame':f'frames/{i}.bctx','frame_bytes':len(raw),'frame_sha256':sha(raw)})
    baseline_raw=json.dumps({'tensors':rows},separators=(',',':')).encode()
    (pkg/'baseline-manifest.json.xz').write_bytes(lzma.compress(baseline_raw))
    (pkg/'decoder'/'predictor_package.py').write_text('''
import json, lzma
def verify(*a, **k): return {}
def read_baseline_manifest(package): return json.loads(lzma.decompress((package/'baseline-manifest.json.xz').read_bytes()))
''')
    doc={'schema':serving.PREDICTIVE_SCHEMA,'complete':True,'verify_on_load':True,
     'baseline_manifest_sha256':sha(baseline_raw),'baseline_manifest_bytes':len(baseline_raw),
     'selected_candidates':{}}
    (pkg/'manifest.json').write_text(json.dumps(doc))
    fake=types.ModuleType('scripts.bitexact_context_mtp')
    fake.MTP_SHAPES={name:(1,) for name in names}
    class Speculator:
        def __init__(self,model,mtp):self.model=model;self.mtp=mtp
    class Head: _bctx_receipt={'loaded_from_bctx_only':True}
    def load_bctx_mtp(path,frame_root,text_config,**kwargs):
        assert path.parent==tmp_path/'external-cache'
        assert path.read_bytes()==baseline_raw and frame_root==pkg
        assert text_config=='text-config' and kwargs['checkpoint_stride']==256
        return Head()
    fake.MTPSpeculator=Speculator;fake.load_bctx_mtp=load_bctx_mtp
    monkeypatch.setitem(sys.modules,'scripts.bitexact_context_mtp',fake)
    model=types.SimpleNamespace(config=types.SimpleNamespace(text_config='text-config'))
    head,speculator,receipt=serving.load_predictive_mtp(pkg,model,device='cpu',checkpoint_stride=256,cache_dir=tmp_path/'external-cache')
    assert isinstance(head,Head) and isinstance(speculator,Speculator)
    assert receipt=={'loaded_from_bctx_only':True}
    assert not (pkg/'.scratch').exists()
    doc['selected_candidates'][names[0]]={}
    (pkg/'manifest.json').write_text(json.dumps(doc))
    with pytest.raises(serving.ContextServingError,match='not an unchanged BCTX frame'):
        serving.load_predictive_mtp(pkg,model,cache_dir=tmp_path/'external-cache')

def test_predictive_http_wrapper_reuses_bctx_backend(tmp_path,monkeypatch):
    wrapper_spec=importlib.util.spec_from_file_location('serve_predictive',ROOT.parent/'scripts'/'serve_bitexact_predictive.py')
    wrapper=importlib.util.module_from_spec(wrapper_spec);wrapper_spec.loader.exec_module(wrapper)
    metadata=tmp_path/'metadata';metadata.mkdir()
    model=object();calls={}
    loader=types.ModuleType('scripts.bitexact_predictive_serving')
    loader.load_predictive_model=lambda *a,**k:(model,{'metadata_dir':str(metadata)})
    backend=types.ModuleType('scripts.serve_bitexact_context')
    class Engine:
        def __init__(self,*a):calls['engine']=a
    backend.BctxModelEngine=Engine
    backend.create_app=lambda **kwargs:calls.update(app=kwargs) or 'app'
    transformers=types.ModuleType('transformers')
    class Processor:
        tokenizer='tokenizer'
    transformers.AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a,**k:Processor())
    transformers.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:'fallback-tokenizer')
    monkeypatch.setitem(sys.modules,'scripts.bitexact_predictive_serving',loader)
    monkeypatch.setitem(sys.modules,'scripts.serve_bitexact_context',backend)
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    assert wrapper.create_predictive_app(package=tmp_path/'candidate',device='cpu',max_pending_requests=2)=='app'
    assert calls['engine'][0] is model and calls['engine'][2]=='tokenizer'
    assert calls['app']['engine'] is not None and calls['app']['max_pending_requests']==2

def test_predictive_http_cli_forwards_serving_options(tmp_path,monkeypatch):
    wrapper_spec=importlib.util.spec_from_file_location('serve_predictive_cli',ROOT.parent/'scripts'/'serve_bitexact_predictive.py')
    wrapper=importlib.util.module_from_spec(wrapper_spec);wrapper_spec.loader.exec_module(wrapper)
    calls={};wrapper.create_predictive_app=lambda **kwargs:calls.update(app=kwargs) or 'app'
    uvicorn=types.ModuleType('uvicorn');uvicorn.run=lambda app,**kwargs:calls.update(run=(app,kwargs))
    monkeypatch.setitem(sys.modules,'uvicorn',uvicorn)
    wrapper.main(['--package',str(tmp_path/'pkg'),'--device','cpu','--host','0.0.0.0','--port','9123',
                  '--loader-workers','3','--max-pending-requests','5','--allow-remote-host','media.example'])
    assert calls['app']['device']=='cpu' and calls['app']['loader_workers']==3
    assert calls['app']['max_pending_requests']==5 and calls['app']['remote_url_allowlist']==['media.example']
    assert calls['run']==('app',{'host':'0.0.0.0','port':9123,'workers':1})
