import hashlib, importlib.util, json, lzma, os, shutil, struct, tarfile, types
import errno
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('predictor_package',HERE.parent/'scripts'/'bitexact_predictive_package.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
probe_spec=importlib.util.spec_from_file_location('bitexact_predictive_codec',HERE.parent/'scripts'/'bitexact_predictive_codec.py')
codec=importlib.util.module_from_spec(probe_spec);probe_spec.loader.exec_module(codec)
mix_spec=importlib.util.spec_from_file_location('bitexact_predictive_mix',HERE.parent/'scripts'/'bitexact_predictive_mix.py')
mix=importlib.util.module_from_spec(mix_spec);mix_spec.loader.exec_module(mix)

def sha(b):return hashlib.sha256(b).hexdigest()
class FakeBaseCodec:
 def __init__(self,package):
  doc=mod.read_baseline_manifest(Path(package))
  self.by_bytes={}
  for row in doc['tensors']:
   path=Path(package)/row['frame']
   if path.is_file():
    raw=path.read_bytes();shape=row['shape'];nr=shape[0] if len(shape)>1 else 1;nc=len(raw)//2//nr
    self.by_bytes[raw]=np.frombuffer(raw,dtype='<u2').copy().reshape(nr,nc)
 def decode(self,b):return self.by_bytes[b].copy()
class FakeBase:
 key_for=staticmethod(lambda n:sha(n.encode()))
 decode_codec=staticmethod(lambda package,*a,**k:FakeBaseCodec(package))
 @staticmethod
 def verify(package,**kwargs):return json.loads((Path(package)/'manifest.json').read_text())

def make_package(root,monkeypatch):
 monkeypatch.setattr(mod,'baseline_module',lambda **kwargs:FakeBase)
 p=root/'pkg';(p/'decoder').mkdir(parents=True);(p/'frames').mkdir()
 (p/'decoder'/'bitexact_context_codec.py').write_text('')
 (p/'decoder'/'bitexact_context_ans.cpp').write_text('')
 (p/'decoder'/'bitexact_context_package.py').write_text('')
 (p/'decoder'/'predictor_package.py').write_text('')
 arrays={'A':np.array([0x3f80,0x3f81,0x3f82,0x3f83],dtype='<u2'),
         'B':np.array([0x3f81,0x3f82,0x3f83,0x3f84],dtype='<u2'),
         'C':np.array([0x3f82,0x3f83,0x3f84,0x3f85],dtype='<u2')}
 names={k:'tensor.'+k for k in arrays};rows=[];sel={}
 for k,a in arrays.items():
  name=names[k];raw=a.tobytes();fr=f'frames/{k}.raw';(p/fr).write_bytes(raw)
  rows.append({'name':name,'source_sha256':sha(raw),'source_bytes':len(raw),'source_shard':'model-00001-of-00001.safetensors','shape':[4],'dtype':'BF16','offsets':[list(arrays).index(k)*len(raw),(list(arrays).index(k)+1)*len(raw)],'frame':fr,'frame_bytes':len(raw)})
 # Candidate frames use named, recursively resolved references.
 for k,ref in [('B','A'),('C','B')]:
  a=arrays[k].reshape(1,-1);b=arrays[ref].reshape(1,-1);frame=codec.encode(a,b);rel=f'predictive-frames/{k}.ppcx';(p/rel).parent.mkdir(exist_ok=True);(p/rel).write_bytes(frame)
  row=next(x for x in rows if x['name']==names[k]);refrow=next(x for x in rows if x['name']==names[ref])
  row['frame_bytes']=len(frame)+1
  sel[names[k]]={'schema':mod.EXPERIMENT,'mode':'single','name':names[k],'source_sha256':row['source_sha256'],'source_bytes':row['source_bytes'],'source_shard':row['source_shard'],'shape':[4],'frame':rel,'frame_bytes':len(frame),'frame_sha256':sha(frame),'saved_bytes':1,'references':[{'name':names[ref],'sha256':refrow['source_sha256']}],'transform':'identity'}
 # Small exact safetensors header and shard used to prove original shard restoration.
 header={}
 raw=b''
 for k,a in arrays.items():
  start=len(raw);raw+=a.tobytes();header[names[k]]={'dtype':'BF16','shape':[4],'data_offsets':[start,len(raw)]}
 h=json.dumps(header,separators=(',',':')).encode();header_bytes=struct.pack('<Q',len(h))+h
 shard_bytes=header_bytes+raw;shard='model-00001-of-00001.safetensors';(p/'headers').mkdir();(p/'headers'/(shard+'.header')).write_bytes(header_bytes)
 # Header receipt and original asset archive.
 archive=p/'model-assets.tar.xz'
 with tarfile.open(archive,'w:xz') as tf:
  info=tarfile.TarInfo('sidecars/config.json');payload=b'{"ok":true}\n';info.size=len(payload)
  import io;tf.addfile(info,io.BytesIO(payload))
 assets={'sidecars/config.json':{'bytes':len(payload),'sha256':sha(payload)}}
 (p/'baseline-manifest.json').write_text(json.dumps({'tensors':rows}))
 (p/'decoder'/'bitexact_predictive_codec.py').write_bytes((HERE.parent/'scripts'/'bitexact_predictive_codec.py').read_bytes())
 (p/'decoder'/'bitexact_predictive_ans.cpp').write_bytes((HERE.parent/'scripts'/'bitexact_predictive_ans.cpp').read_bytes())
 (p/'decoder'/'bitexact_predictive_mix.py').write_bytes((HERE.parent/'scripts'/'bitexact_predictive_mix.py').read_bytes())
 for decoder_name in ('bitexact_predictive_gpu.py','bitexact_predictive_gpu_index.cpp','bitexact_predictive_gpu_kernels.py'):
  (p/'decoder'/decoder_name).write_bytes((HERE.parent/'scripts'/decoder_name).read_bytes())
 manifest={'schema':mod.SCHEMA,'complete':True,'verify_on_load':True,'experimental_codec':mod.EXPERIMENT,'baseline_manifest_sha256':sha(json.dumps({'tensors':rows}).encode()),'baseline_tool_sha256':sha((p/'decoder'/'bitexact_context_package.py').read_bytes()),'decoder_files':['decoder/bitexact_context_package.py','decoder/bitexact_context_codec.py','decoder/bitexact_context_ans.cpp','decoder/bitexact_predictive_codec.py','decoder/bitexact_predictive_ans.cpp','decoder/bitexact_predictive_mix.py','decoder/bitexact_predictive_gpu.py','decoder/bitexact_predictive_gpu_index.cpp','decoder/bitexact_predictive_gpu_kernels.py','decoder/predictor_package.py'],'selected_candidates':sel,
  'source':{'shards':{shard:{'bytes':len(shard_bytes),'sha256':sha(shard_bytes)}}},
  'headers':{shard:{'path':f'headers/{shard}.header','bytes':len(header_bytes),'sha256':sha(header_bytes)}},'metadata_assets':assets}
 mod.finalize(p,manifest)
 return p,shard_bytes,arrays,names

def test_recursive_reference_restore_exact_shard(tmp_path,monkeypatch):
 p,expected,arrays,names=make_package(tmp_path,monkeypatch)
 assert mod.verify(p)['package_bytes']==sum(r['bytes'] for r in json.loads((p/'manifest.json').read_text())['files'])
 result=mod.restore(p,tmp_path/'restored')
 assert result['status']=='RESTORE_EXACT' and result['tensors']==3 and len(result['shards'])==1 and result['candidates']==2 and result['assets']==1
 assert (tmp_path/'restored'/'model-00001-of-00001.safetensors').read_bytes()==expected

def rewrite_manifest(p,change):
 m=json.loads((p/'manifest.json').read_text());change(m);mod.finalize(p,m)

def test_corrupt_reference_sha_rejected(tmp_path,monkeypatch):
 p,_,_,names=make_package(tmp_path,monkeypatch)
 rewrite_manifest(p,lambda m:m['selected_candidates'][names['B']]['references'][0].__setitem__('sha256','0'*64))
 with pytest.raises(ValueError,match='reference identity SHA'):
  mod.verify(p)

def test_cycle_rejected(tmp_path,monkeypatch):
 p,_,_,names=make_package(tmp_path,monkeypatch)
 def cycle(m):
  b=m['selected_candidates'][names['B']];c=m['selected_candidates'][names['C']]
  b['references']=[{'name':names['C'],'sha256':c['source_sha256']}]
  c['references']=[{'name':names['B'],'sha256':b['source_sha256']}]
 rewrite_manifest(p,cycle)
 with pytest.raises(ValueError,match='cycle'):
  mod.verify(p)

def test_path_traversal_rejected(tmp_path,monkeypatch):
 p,_,_,names=make_package(tmp_path,monkeypatch)
 rewrite_manifest(p,lambda m:m['selected_candidates'][names['B']].__setitem__('frame','../outside.ppcx'))
 with pytest.raises(ValueError,match='unsafe relative path'):
  mod.verify(p)

def test_shard_restore_then_aggregate(tmp_path,monkeypatch):
 p,expected,_,_=make_package(tmp_path,monkeypatch);dest=tmp_path/'parallel-restored'
 part=mod.restore(p,dest,'model-00001-of-00001.safetensors')
 assert part['status']=='SHARD_RESTORED' and part['assets']==0
 final=mod.aggregate(p,dest)
 assert final['status']=='RESTORE_EXACT' and final['shard_count']==1
 assert (dest/'model-00001-of-00001.safetensors').read_bytes()==expected

def test_baseline_helper_is_explicit_or_packaged(tmp_path,monkeypatch):
 p,_,_,_=make_package(tmp_path,monkeypatch)
 monkeypatch.undo()
 tool=tmp_path/'baseline_tool.py';tool.write_text('MARKER = "explicit"\n')
 external=mod.baseline_module(tool_path=tool)
 packaged=mod.baseline_module(package=p)
 assert external.MARKER=='explicit'
 assert Path(packaged.__file__)==p/'decoder'/'bitexact_context_package.py'

def test_legacy_schema_and_decoder_filenames_remain_readable(tmp_path,monkeypatch):
 p,_,_,_=make_package(tmp_path,monkeypatch)
 m=json.loads((p/'manifest.json').read_text());m['schema']=mod.LEGACY_SCHEMA;m.pop('verify_on_load',None)
 renames={'bitexact_predictive_codec.py':'predictive_pair_codec.py','bitexact_predictive_ans.cpp':'predictive_ans.cpp','bitexact_predictive_mix.py':'predictive_mix.py'}
 updated=[]
 for name in m['decoder_files']:
  for new,old in renames.items():
   if name.endswith(new):
    target=p/'decoder'/old;(p/'decoder'/new).rename(target);name='decoder/'+old
  updated.append(name)
 m['decoder_files']=updated;mod.finalize(p,m)
 assert mod.verify(p,decode_frames=False)['schema']==mod.LEGACY_SCHEMA

def make_build_inputs(root):
 base=root/'baseline';(base/'frames').mkdir(parents=True);(base/'decoder').mkdir()
 sweep=root/'sweep';sweep.mkdir()
 names={k:'tensor.'+k for k in ('A','B','C')}
 arrays={'A':np.array([0x3f80,0x3f81,0x3f82,0x3f83],dtype='<u2'),
         'B':np.array([0x3f81,0x3f82,0x3f83,0x3f84],dtype='<u2'),
         'C':np.array([0x3f82,0x3f83,0x3f84,0x3f85],dtype='<u2')}
 rows=[];raw_shard=b''
 for k,a in arrays.items():
  raw=a.tobytes();start=len(raw_shard);raw_shard+=raw;fr=f'frames/{k}.bctx';(base/fr).write_bytes(raw)
  r={'name':names[k],'source_sha256':sha(raw),'source_bytes':len(raw),'source_shard':'model-00001-of-00001.safetensors','shape':[4],'dtype':'BF16','offsets':[start,len(raw_shard)],'frame':fr,'frame_bytes':len(raw),'frame_sha256':sha(raw)};rows.append(r)
 # Give B room for a synthetic positive delta; C is nonbeneficial and has no saved frame.
 up=codec.encode(arrays['B'].reshape(1,-1),arrays['A'].reshape(1,-1)); rows[1]['frame_bytes']=len(up)+10
 down=codec.encode(arrays['C'].reshape(1,-1),arrays['A'].reshape(1,-1))
 rows[2]['frame_bytes']=len(down)
 (base/'decoder'/'bitexact_context_package.py').write_text('# supplied baseline helper')
 (base/'decoder'/'bitexact_context_codec.py').write_text('# baseline codec')
 (base/'decoder'/'bitexact_context_ans.cpp').write_text('// baseline ans')
 hdr={}
 for k,a in arrays.items():
  i=list(arrays).index(k);hdr[names[k]]={'dtype':'BF16','shape':[4],'data_offsets':[i*8,(i+1)*8]}
 hb=json.dumps(hdr,separators=(',',':')).encode();header=struct.pack('<Q',len(hb))+hb
 shard='model-00001-of-00001.safetensors';actual_shard=header+raw_shard
 (base/'headers').mkdir();(base/'headers'/(shard+'.header')).write_bytes(header)
 import io
 with tarfile.open(base/'model-assets.tar.xz','w:xz') as tf:
  payload=b'{}\n';info=tarfile.TarInfo('sidecars/config.json');info.size=len(payload);tf.addfile(info,io.BytesIO(payload))
 source={'shards':{shard:{'bytes':len(actual_shard),'sha256':sha(actual_shard)}}}
 headers={shard:{'path':f'headers/{shard}.header','bytes':len(header),'sha256':sha(header)}}
 assets={'sidecars/config.json':{'bytes':3,'sha256':sha(b'{}\n')}}
 doc={'schema':'baseline-fixture','package_bytes':123,'tensors':rows,'source':source,'headers':headers,'metadata_assets':assets}
 (base/'manifest.json').write_text(json.dumps(doc))
 def receipt(target,ref,frame,size,saved,layer):
  tr={'status':'PASS','exact':True,'target':names[target],'reference':names[ref],
      'source_sha256':next(x for x in rows if x['name']==names[target])['source_sha256'],
      'reference_sha256':next(x for x in rows if x['name']==names[ref])['source_sha256'],
      'published_frame_bytes':next(x for x in rows if x['name']==names[target])['frame_bytes'],
      'candidate_frame_bytes':size,'saved_bytes':saved,'transform':'identity','frame_sha256':sha(frame)}
  if saved>0:(sweep/f'layer{layer}-up.ppcx').write_bytes(frame)
  (sweep/f'layer{layer}-up.json').write_text(json.dumps(tr))
 receipt('B','A',up,len(up),10,1);receipt('C','A',down,len(down),0,2)
 return base,sweep,rows,arrays,shard

def test_builder_replaces_only_selected_frames_and_accounts_bytes(tmp_path,monkeypatch):
 base,sweep,rows,_,shard=make_build_inputs(tmp_path);monkeypatch.setattr(mod,'baseline_module',lambda **kwargs:FakeBase)
 receipt_payloads={'receipts/a.json':b'{"receipt":1}\n','receipts/nested/b.json':b'{"receipt":2}\n'}
 for rel,payload in receipt_payloads.items():
  path=base/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
 baseline_manifest_raw=(base/'manifest.json').read_bytes()
 archive_a,_=mod._receipt_archive(base);archive_b,_=mod._receipt_archive(base);assert archive_a==archive_b
 out=tmp_path/'candidate';res=mod.build(base,sweep,out,base/'decoder'/'bitexact_context_package.py')
 assert res['schema']==mod.SCHEMA
 assert not (out/'frames'/'B.bctx').exists()
 assert (base/'frames'/'B.bctx').is_file() and (base/'frames'/'B.bctx').read_bytes()==np.array([0x3f81,0x3f82,0x3f83,0x3f84],dtype='<u2').tobytes()
 assert (out/'frames'/'A.bctx').is_file() and (out/'frames'/'C.bctx').is_file()
 assert os.stat(out/'frames'/'A.bctx').st_ino==os.stat(base/'frames'/'A.bctx').st_ino
 assert len(list((out/'predictive-frames').glob('*.ppcx')))==1
 assert not (sweep/'layer2-up.ppcx').exists()
 m=json.loads((out/'manifest.json').read_text())
 assert m['schema']=='bitexact-predictive-context-package-v1' and m['verify_on_load'] is True
 assert not ({'qualification_status','build_progress','targets','measurements','source_paths'} & set(m))
 assert m['package_bytes']==sum(x['bytes'] for x in m['files'])
 assert m['package_bytes']==sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
 assert (out/'decoder'/'predictor_package.py').is_file()
 assert lzma.decompress((out/'baseline-manifest.json.xz').read_bytes())==baseline_manifest_raw
 assert not (out/'baseline-manifest.json').exists() and not (out/'receipts').exists()
 with tarfile.open(out/'baseline-receipts.tar.xz','r:xz') as tf:
  assert {m.name:tf.extractfile(m).read() for m in tf.getmembers()}==receipt_payloads

def test_builder_rejects_mtp_candidates_to_preserve_bctx_frames(tmp_path,monkeypatch):
 base,sweep,_,_,_=make_build_inputs(tmp_path);monkeypatch.setattr(mod,'baseline_module',lambda **kwargs:FakeBase)
 receipt=sweep/'layer1-up.json';doc=json.loads(receipt.read_text());doc['target']='mtp.hidden.weight';receipt.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match='MTP tensors must remain unchanged'):
  mod.build(base,sweep,tmp_path/'mtp-package',base/'decoder'/'bitexact_context_package.py')

def test_candidate_generator_up_profile_search_and_hash_checked_resume(tmp_path,monkeypatch):
 base=tmp_path/'baseline';(base/'frames').mkdir(parents=True);(base/'decoder').mkdir()
 tool=base/'decoder'/'bitexact_context_package.py';tool.write_text('# helper')
 up=np.arange(12,dtype='<u2').reshape(4,3)+0x3f80;gate=up+1;down=up.T.copy();mtp=np.array([0x3f80],dtype='<u2')
 arrays={'model.layers.0.mlp.up_proj.weight':up,'model.layers.0.mlp.gate_proj.weight':gate,
         'model.layers.0.mlp.down_proj.weight':down,'mtp.embed.weight':mtp}
 rows=[]
 for name,array in arrays.items():
  raw=np.ascontiguousarray(array,dtype='<u2').tobytes();frame=f'frames/{sha(name.encode())}.bctx';(base/frame).write_bytes(raw)
  rows.append({'name':name,'shape':list(array.shape),'dtype':'BF16','source_bytes':len(raw),'source_sha256':sha(raw),
               'frame':frame,'frame_bytes':len(raw)+2048,'frame_sha256':sha(raw),
               'source_shard':'model-00001-of-00001.safetensors'})
 (base/'manifest.json').write_text(json.dumps({'tensors':rows}))
 (base/'baseline-manifest.json').write_text(json.dumps({'tensors':rows}))
 class CandidateCodec:
  def __init__(self):self.calls=[];self.decoded={}
  def encode(self,y,ref,*,row_profile='quartile',bin_profile='quantile'):
   y=np.ascontiguousarray(y,dtype='<u2');ref=np.ascontiguousarray(ref,dtype='<u2')
   self.calls.append((sha(y.tobytes()),row_profile,bin_profile))
   prefix=(row_profile+'-'+bin_profile+'-').encode()+bytes.fromhex(sha(y.tobytes()))[:6]+bytes.fromhex(sha(ref.tobytes()))[:6]
   size={('quartile','quantile'):90,('residual','quantile'):75,('tail','quantile'):85,('residual','lloyd'):50}[(row_profile,bin_profile)]
   frame=prefix.ljust(size,b'x');self.decoded[frame]=y.copy();return frame
  def decode(self,frame,ref):return self.decoded[frame].copy()
 class CandidateMix:
  def __init__(self):self.fit_calls=0
  def fit_mixed_coefficients(self,target,up_words,gate_words):
   self.fit_calls+=1;return np.zeros((target.shape[0],3),dtype='<u2')
  def predict(self,up_words,gate_words,coeff):return np.ascontiguousarray(up_words,dtype='<u2')
 codec=CandidateCodec();mix=CandidateMix();monkeypatch.setattr(mod,'baseline_module',lambda **kwargs:FakeBase)
 out=tmp_path/'candidate-receipts'
 result=mod.generate_candidates(base,out,tool,codec_module=codec,mix_module=mix)
 singles=out/'single';mixed=out/'mixed';up_receipt=json.loads(next(singles.glob('*-up.json')).read_text())
 mixed_receipt=json.loads(next(mixed.glob('*-down.json')).read_text())
 assert result['single_receipts']==2 and result['mixed_receipts']==1
 assert up_receipt['target']=='model.layers.0.mlp.up_proj.weight' and up_receipt['reference']=='model.layers.0.mlp.gate_proj.weight'
 assert up_receipt['transform']=='identity'
 assert (mixed_receipt['row_profile'],mixed_receipt['bin_profile'])==('residual','lloyd')
 assert mixed_receipt['reference_sha256']==[next(r['source_sha256'] for r in rows if r['name']==n) for n in mixed_receipt['references']]
 target_digest=sha(np.ascontiguousarray(down.T).tobytes())
 down_profiles=[(row,bin_) for digest,row,bin_ in codec.calls if digest==target_digest]
 assert len(down_profiles)==7 and down_profiles.count(('quartile','quantile'))==4
 assert set(down_profiles)=={('quartile','quantile'),('residual','quantile'),('tail','quantile'),('residual','lloyd')}
 assert all(not json.loads(p.read_text()).get('target','').startswith('mtp.') for p in (*singles.glob('*.json'),*mixed.glob('*.json')))
 calls_before=len(codec.calls);fits_before=mix.fit_calls
 mod.generate_candidates(base,out,tool,codec_module=codec,mix_module=mix)
 assert len(codec.calls)==calls_before and mix.fit_calls==fits_before
 coeff=next(mixed.glob('*-coeff.bf16'));coeff.write_bytes(b'tampered')
 with pytest.raises(ValueError,match='resume coefficient SHA/size mismatch'):
  mod.generate_candidates(base,out,tool,codec_module=codec,mix_module=mix)

def make_mix_build_inputs(root,single_ref='gate',raw_savings_override=None):
 base=root/'mix-baseline';(base/'frames').mkdir(parents=True);(base/'decoder').mkdir();single=root/'single-sweep';single.mkdir();mixed=root/'mix-sweep';mixed.mkdir()
 rng=np.random.default_rng(33);up=rng.integers(0x3e00,0x4000,(128,128),dtype=np.uint16);gate=rng.integers(0x3e00,0x4000,(128,128),dtype=np.uint16)
 coeff=np.zeros((128,3),dtype=np.uint16);coeff[:,0]=0x3f80
 derived=mix.predict(up,gate,coeff);down=np.ascontiguousarray(derived.T)
 names={'up':'model.language_model.layers.31.mlp.up_proj.weight','gate':'model.language_model.layers.31.mlp.gate_proj.weight','down':'model.language_model.layers.31.mlp.down_proj.weight'}
 arrays={'up':up,'gate':gate,'down':down};rows=[];offset=0
 for k,a in arrays.items():
  raw=np.ascontiguousarray(a).tobytes();f=f'frames/{k}.bctx';(base/f).write_bytes(raw)
  rows.append({'name':names[k],'source_sha256':sha(raw),'source_bytes':len(raw),'source_shard':'model-00001-of-00001.safetensors','shape':list(a.shape),'dtype':'BF16','offsets':[offset,offset+len(raw)],'frame':f,'frame_bytes':len(raw)});offset+=len(raw)
 frame=codec.encode(np.ascontiguousarray(down.T),derived);coeff_bytes=coeff.astype('<u2').tobytes();single_frame=codec.encode(np.ascontiguousarray(down.T),arrays[single_ref])
 baseline_frame_bytes=40000 if raw_savings_override is None else len(frame)+len(coeff_bytes)+raw_savings_override
 rows[2]['frame_bytes']=baseline_frame_bytes
 (mixed/'layer31-down.ppcx').write_bytes(frame);(mixed/'layer31-coeff.bf16').write_bytes(coeff_bytes)
 mix_row={'status':'PASS','exact':True,'target':names['down'],'references':[names['up'],names['gate']],'source_sha256':rows[2]['source_sha256'],'candidate_frame_bytes':len(frame),'coefficient_bytes':len(coeff_bytes),'candidate_total_bytes':len(frame)+len(coeff_bytes),'published_frame_bytes':rows[2]['frame_bytes'],'saved_bytes':baseline_frame_bytes-len(frame)-len(coeff_bytes),'frame_sha256':sha(frame),'coefficients_sha256':sha(coeff_bytes),'derived_reference_sha256':sha(derived.tobytes())}
 (mixed/'layer31-down.json').write_text(json.dumps(mix_row))
 (single/'layer31-down.ppcx').write_bytes(single_frame)
 single_index={'up':0,'gate':1}[single_ref]
 single_row={'status':'PASS','exact':True,'target':names['down'],'reference':names[single_ref],'source_sha256':rows[2]['source_sha256'],'reference_sha256':rows[single_index]['source_sha256'],'published_frame_bytes':rows[2]['frame_bytes'],'candidate_frame_bytes':len(single_frame),'saved_bytes':baseline_frame_bytes-len(single_frame),'transform':'target transpose','frame_sha256':sha(single_frame)}
 (single/'layer31-down.json').write_text(json.dumps(single_row))
 header_doc={};source=b''
 for k,a in arrays.items():
  raw=np.ascontiguousarray(a).tobytes();start=len(source);source+=raw;header_doc[names[k]]={'dtype':'BF16','shape':list(a.shape),'data_offsets':[start,len(source)]}
 hb=json.dumps(header_doc,separators=(',',':')).encode();header=struct.pack('<Q',len(hb))+hb;shard='model-00001-of-00001.safetensors';shard_bytes=header+source
 (base/'headers').mkdir();(base/'headers'/(shard+'.header')).write_bytes(header)
 import io
 with tarfile.open(base/'model-assets.tar.xz','w:xz') as tf:
  p=b'{}\n';info=tarfile.TarInfo('sidecars/config.json');info.size=len(p);tf.addfile(info,io.BytesIO(p))
 (base/'decoder'/'bitexact_context_package.py').write_text('# helper stub');(base/'decoder'/'bitexact_context_codec.py').write_text('# codec stub');(base/'decoder'/'bitexact_context_ans.cpp').write_text('// ans stub')
 assets={'sidecars/config.json':{'bytes':3,'sha256':sha(b'{}\n')}};doc={'schema':'baseline','package_bytes':100,'tensors':rows,'source':{'shards':{shard:{'bytes':len(shard_bytes),'sha256':sha(shard_bytes)}}},'headers':{shard:{'path':f'headers/{shard}.header','bytes':len(header),'sha256':sha(header)}},'metadata_assets':assets}
 (base/'manifest.json').write_text(json.dumps(doc))
 return base,single,mixed,rows,names,arrays,shard_bytes

def test_mix_builder_selects_best_savings_and_restores(tmp_path,monkeypatch):
 base,single,mixed,rows,names,arrays,expected=make_mix_build_inputs(tmp_path);monkeypatch.setattr(mod,'baseline_module',lambda **kwargs:FakeBase)
 out=tmp_path/'mix-candidate';res=mod.build(base,single,out,base/'decoder'/'bitexact_context_package.py',mixed)
 candidate=res['selected_candidates'][names['down']]
 assert candidate['mode']=='mixbf16' and candidate['saved_bytes']>json.loads((single/'layer31-down.json').read_text())['saved_bytes']
 coeffmeta=candidate['coefficients'];assert coeffmeta['encoding']=='zlib' and coeffmeta['bytes']<coeffmeta['decoded_bytes']
 assert candidate['physical_saved_bytes']>candidate['saved_bytes']
 coeffwords=mod.read_coefficients(out,candidate);assert coeffwords.dtype==np.dtype('<u2') and coeffwords.shape==tuple(coeffmeta['shape'])
 assert candidate['references']==[{'name':names['up'],'sha256':next(r['source_sha256'] for r in rows if r['name']==names['up'])},{'name':names['gate'],'sha256':next(r['source_sha256'] for r in rows if r['name']==names['gate'])}]
 assert not (out/'frames'/'down.bctx').exists()
 assert len(list((out/'predictive-coefficients').glob('*.bf16')))==1
 package_doc=json.loads((out/'manifest.json').read_text())
 assert package_doc['package_bytes']==sum(x['bytes'] for x in package_doc['files'])
 assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())==package_doc['package_bytes']
 assert mod.verify(out)['schema']==mod.SCHEMA
 restored=mod.restore(out,tmp_path/'mix-restored')
 assert restored['status']=='RESTORE_EXACT'
 assert (tmp_path/'mix-restored'/'model-00001-of-00001.safetensors').read_bytes()==expected

def test_unselected_mix_files_are_not_packaged(tmp_path,monkeypatch):
 base,single,mixed,_,names,_,_=make_mix_build_inputs(tmp_path,single_ref='up');monkeypatch.setattr(mod,'baseline_module',lambda **kwargs:FakeBase)
 out=tmp_path/'single-wins';res=mod.build(base,single,out,base/'decoder'/'bitexact_context_package.py',mixed)
 assert res['selected_candidates'][names['down']]['mode']=='single'
 assert not (out/'predictive-coefficients').exists()
 assert not list((out/'predictive-frames').glob('*.bf16'))

def test_physical_savings_can_be_positive_when_raw_savings_are_negative(tmp_path,monkeypatch):
 base,single,mixed,_,names,_,expected=make_mix_build_inputs(tmp_path,raw_savings_override=-4);monkeypatch.setattr(mod,'baseline_module',lambda **kwargs:FakeBase)
 out=tmp_path/'compressed-wins';res=mod.build(base,single,out,base/'decoder'/'bitexact_context_package.py',mixed)
 c=res['selected_candidates'][names['down']]
 assert c['mode']=='mixbf16' and c['saved_bytes']==-4 and c['physical_saved_bytes']>0
 assert mod.verify(out)['selected_candidates'][names['down']]['saved_bytes']==-4
 restored=mod.restore(out,tmp_path/'compressed-restored')
 assert restored['status']=='RESTORE_EXACT' and (tmp_path/'compressed-restored'/'model-00001-of-00001.safetensors').read_bytes()==expected

def built_mix(tmp_path,monkeypatch):
 base,single,mixed,rows,names,arrays,expected=make_mix_build_inputs(tmp_path)
 monkeypatch.setattr(mod,'baseline_module',lambda **kwargs:FakeBase)
 out=tmp_path/'mix-candidate';mod.build(base,single,out,base/'decoder'/'bitexact_context_package.py',mixed)
 return out,rows,names

def test_mix_corrupt_coefficient_rejected(tmp_path,monkeypatch):
 p,_,_=built_mix(tmp_path,monkeypatch);m=json.loads((p/'manifest.json').read_text());c=next(iter(m['selected_candidates'].values()))
 coeff=p/c['coefficients']['path'];raw=bytearray(coeff.read_bytes());raw[0]^=1;coeff.write_bytes(raw);mod.finalize(p,m)
 with pytest.raises(ValueError,match='mixed coefficient checksum'):
  mod.verify(p)

def test_mix_bad_reference_rejected(tmp_path,monkeypatch):
 p,_,_=built_mix(tmp_path,monkeypatch)
 def badref(m):next(iter(m['selected_candidates'].values()))['references'][1]['sha256']='0'*64
 rewrite_manifest(p,badref)
 with pytest.raises(ValueError,match='reference identity SHA'):
  mod.verify(p,decode_frames=False)

def test_mix_reference_cycle_rejected(tmp_path,monkeypatch):
 p,rows,names=built_mix(tmp_path,monkeypatch);m=json.loads((p/'manifest.json').read_text())
 down=m['selected_candidates'][names['down']];up_row=next(r for r in rows if r['name']==names['up']);gate_row=next(r for r in rows if r['name']==names['gate'])
 down['references'][0]={'name':names['up'],'sha256':up_row['source_sha256']}
 frame_src=p/down['frame'];coeff_src=p/down['coefficients']['path'];up_frame=f'predictive-frames/{sha(names["up"].encode())}.ppcx';up_coeff=f'predictive-coefficients/{sha(names["up"].encode())}.bf16'
 (p/up_frame).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(frame_src,p/up_frame);(p/up_coeff).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(coeff_src,p/up_coeff)
 physical_total=down['frame_bytes']+down['coefficients']['bytes'];raw_total=down['frame_bytes']+down['coefficients']['decoded_bytes']
 candidate={**down,'name':names['up'],'source_sha256':up_row['source_sha256'],'source_bytes':up_row['source_bytes'],'source_shard':up_row['source_shard'],'shape':up_row['shape'],'frame':up_frame,'references':[{'name':names['down'],'sha256':next(r['source_sha256'] for r in rows if r['name']==names['down'])},{'name':names['gate'],'sha256':gate_row['source_sha256']}],'coefficients':{**down['coefficients'],'path':up_coeff},'saved_bytes':up_row['frame_bytes']-raw_total,'physical_saved_bytes':up_row['frame_bytes']-physical_total}
 m['selected_candidates'][names['up']]=candidate;mod.finalize(p,m)
 with pytest.raises(ValueError,match='cycle'):
  mod.verify(p,decode_frames=False)

def test_read_coefficients_accepts_raw_identity_encoding(tmp_path):
 words=np.array([[0x3f80,0xbf80,0x3f00],[0x3e80,0xbe80,0x4000]],dtype='<u2');raw=words.tobytes();path=tmp_path/'c.bf16';path.write_bytes(raw)
 candidate={'coefficients':{'path':'c.bf16','encoding':'identity','bytes':len(raw),'sha256':sha(raw),'decoded_bytes':len(raw),'decoded_sha256':sha(raw),'shape':[2,3]}}
 assert np.array_equal(mod.read_coefficients(tmp_path,candidate),words)

def test_baseline_frame_link_falls_back_to_copy_on_exdev(tmp_path,monkeypatch):
 src=tmp_path/'source.bctx';dst=tmp_path/'target.bctx';src.write_bytes(b'frame bytes')
 def cross_device(*args):raise OSError(errno.EXDEV,'cross-device link')
 monkeypatch.setattr(mod.os,'link',cross_device)
 mod._link_baseline_frame(src,dst)
 assert dst.read_bytes()==src.read_bytes() and os.stat(src).st_ino!=os.stat(dst).st_ino
