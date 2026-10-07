#!/usr/bin/env python3
"""Private package experiment for exact reference-conditioned BF16 frames."""
from __future__ import annotations
import argparse, errno, hashlib, io, json, lzma, math, os, shutil, struct, sys, tarfile, types, zlib
from collections import OrderedDict
from pathlib import Path, PurePosixPath
import numpy as np

ROOT=Path(__file__).resolve().parent
SCHEMA='bitexact-predictive-context-package-v1'
LEGACY_SCHEMA='private-predictive-context-package-v1'
EXPERIMENT='ppcx-reference-conditioned-bf16-v1'
DECODE_CACHE_BYTES=512*1024*1024

class TensorCache:
 def __init__(self,limit=DECODE_CACHE_BYTES):self.limit=limit;self.size=0;self.data=OrderedDict()
 def get(self,key):
  value=self.data.get(key)
  if value is not None:self.data.move_to_end(key)
  return value
 def put(self,key,value):
  n=value.nbytes
  if key in self.data:self.size-=self.data.pop(key).nbytes
  if n>self.limit:return
  self.data[key]=value;self.size+=n
  while self.size>self.limit:
   _,old=self.data.popitem(last=False);self.size-=old.nbytes

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def file_sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def _link_baseline_frame(source:Path,target:Path)->None:
 try:os.link(source,target)
 except OSError as exc:
  if exc.errno!=errno.EXDEV:raise
  shutil.copy2(source,target)
def write_json(p:Path,obj):
 raw=(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n').encode()
 p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+'.partial')
 with tmp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)

def read_baseline_manifest(package:Path)->dict:
 """Return the baseline tensor manifest from legacy JSON or stored XZ bytes."""
 package=Path(package);plain=package/'baseline-manifest.json';compressed=package/'baseline-manifest.json.xz'
 if plain.is_file():raw=plain.read_bytes()
 elif compressed.is_file():
  try:raw=lzma.decompress(compressed.read_bytes())
  except lzma.LZMAError as e:raise ValueError('baseline manifest XZ is corrupt') from e
 else:raise ValueError('baseline tensor manifest missing')
 pin=json.loads((package/'manifest.json').read_text()).get('baseline_manifest_sha256')
 if pin and sha(raw)!=pin:raise ValueError('baseline manifest checksum mismatch')
 return json.loads(raw)

def read_coefficients(package:Path,candidate:dict)->np.ndarray:
 """Read validated little-endian BF16 coefficient words for a mixed candidate."""
 c=candidate.get('coefficients')
 if not isinstance(c,dict):raise ValueError('mixed coefficient metadata missing')
 path=Path(package).joinpath(*safe_rel(c.get('path')).parts)
 try:encoded=path.read_bytes()
 except OSError as e:raise ValueError('mixed coefficient file missing') from e
 if len(encoded)!=c.get('bytes') or sha(encoded)!=c.get('sha256'):raise ValueError('mixed coefficient checksum mismatch')
 shape=c.get('shape')
 if not isinstance(shape,list) or len(shape)!=2 or shape[1]!=3 or any(type(n)is not int or n<=0 for n in shape):raise ValueError('mixed coefficient shape invalid')
 expected_raw=math.prod(shape)*2
 encoding=c.get('encoding','identity')
 if encoding=='zlib' and c['bytes']>=c.get('decoded_bytes',c['bytes']):raise ValueError('zlib coefficient encoding does not save bytes')
 if encoding=='identity' and c['bytes']!=c.get('decoded_bytes',c['bytes']):raise ValueError('identity coefficient byte counts differ')
 try:
  if encoding=='zlib':
   d=zlib.decompressobj();raw=d.decompress(encoded,expected_raw+1)
   if len(raw)>expected_raw or d.unconsumed_tail:raise ValueError('mixed coefficient decompression exceeds declared shape')
   raw+=d.flush()
   if not d.eof or d.unused_data:raise ValueError('mixed coefficient compressed stream is incomplete or has trailing bytes')
  elif encoding=='identity':raw=encoded
  else:raise ValueError('unsupported coefficient encoding')
 except zlib.error as e:raise ValueError('mixed coefficient zlib data is corrupt') from e
 decoded_bytes=c.get('decoded_bytes',c.get('bytes'))
 decoded_sha=c.get('decoded_sha256',c.get('sha256'))
 if len(raw)!=decoded_bytes or sha(raw)!=decoded_sha:raise ValueError('decoded mixed coefficient checksum mismatch')
 if len(raw)!=expected_raw:raise ValueError('mixed coefficient shape/length invalid')
 return np.frombuffer(raw,dtype='<u2').reshape(shape).copy()

def _receipt_archive(source:Path)->tuple[bytes,list[dict]]:
 members=[];buf=io.BytesIO()
 paths=sorted(p for p in (source/'receipts').rglob('*') if p.is_file()) if (source/'receipts').exists() else []
 with tarfile.open(fileobj=buf,mode='w:xz',format=tarfile.PAX_FORMAT,preset=9) as tf:
  for path in paths:
   rel=path.relative_to(source).as_posix();safe_rel(rel)
   raw=path.read_bytes();members.append({'path':rel,'bytes':len(raw),'sha256':sha(raw)})
   info=tarfile.TarInfo(rel);info.size=len(raw);info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
   tf.addfile(info,io.BytesIO(raw))
 return buf.getvalue(),members

def _atomic_bytes(path:Path,payload:bytes)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
 with tmp.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def _publish_candidate(receipt_path:Path,receipt:dict,assets:list[tuple[Path,bytes]])->None:
 if receipt_path.exists():
  try:old=json.loads(receipt_path.read_text())
  except (OSError,json.JSONDecodeError) as exc:raise ValueError('existing candidate receipt is not resumable') from exc
  for key in ('target','source_sha256','reference','reference_sha256','references','published_frame_bytes','candidate_frame_bytes','frame_sha256','coefficients_sha256','derived_reference_sha256'):
   if key in old and old.get(key)!=receipt.get(key):raise ValueError('resume candidate receipt source SHA mismatch')
 for path,payload in assets:
  if path.exists() and (path.is_symlink() or not path.is_file() or path.read_bytes()!=payload):raise ValueError('resume candidate artifact SHA mismatch')
  if not path.exists():_atomic_bytes(path,payload)
 raw=(json.dumps(receipt,sort_keys=True,separators=(',',':'))+'\n').encode()
 if not receipt_path.exists():_atomic_bytes(receipt_path,raw)

def _resume_receipt(path:Path,binding:dict,profiles:tuple[tuple[str,str],...],frame_path:Path,
                    coefficient_path:Path|None=None)->dict|None:
 if not path.exists():return None
 try:receipt=json.loads(path.read_text())
 except (OSError,json.JSONDecodeError) as exc:raise ValueError('existing candidate receipt is not resumable') from exc
 if any(receipt.get(k)!=v for k,v in binding.items()):raise ValueError('resume candidate source/encoder SHA mismatch')
 if receipt.get('status')!='PASS' or receipt.get('exact') is not True:raise ValueError('resume receipt lacks exact PASS evidence')
 if (receipt.get('row_profile'),receipt.get('bin_profile')) not in profiles:raise ValueError('resume candidate profile is unsupported')
 frame_bytes=receipt.get('candidate_frame_bytes');frame_sha=receipt.get('frame_sha256')
 required=coefficient_path is not None or (type(receipt.get('saved_bytes')) is int and receipt['saved_bytes']>0)
 if required:
  if (frame_path.is_symlink() or not frame_path.is_file() or type(frame_bytes)is not int
      or frame_path.stat().st_size!=frame_bytes or file_sha(frame_path)!=frame_sha):
   raise ValueError('resume candidate frame SHA/size mismatch')
 elif frame_path.exists():raise ValueError('unexpected frame for non-beneficial resumed candidate')
 if coefficient_path is not None:
  if (coefficient_path.is_symlink() or not coefficient_path.is_file()
      or coefficient_path.stat().st_size!=receipt.get('coefficient_bytes')
      or file_sha(coefficient_path)!=receipt.get('coefficients_sha256')):
   raise ValueError('resume coefficient SHA/size mismatch')
 return receipt

def generate_candidates(baseline:Path,out:Path,baseline_tool:Path,*,codec_module=None,mix_module=None):
 """Generate exact up<-gate, down-reference, and mixed-down receipts."""
 baseline=baseline.resolve();out=out.resolve();baseline_tool=baseline_tool.resolve()
 if out==baseline or baseline in out.parents or out in baseline.parents:raise ValueError('candidate output and baseline must be separate')
 helper=baseline/'decoder'/'bitexact_context_package.py'
 if not helper.is_file() or file_sha(helper)!=file_sha(baseline_tool):raise ValueError('explicit baseline tool differs from packaged decoder helper')
 base=baseline_module(tool_path=baseline_tool)
 doc=base.verify(baseline,decode_frames=False,cache_dir=out.parent/'.scratch'/'predictive-candidate-verify')
 rows={r['name']:r for r in doc['tensors']}
 if len(rows)!=len(doc['tensors']):raise ValueError('baseline has duplicate tensor names')
 base_codec=base.decode_codec(baseline,cache_dir=out.parent/'.scratch'/'predictive-candidate-base')
 decoded={}
 def words(name):
  if name not in decoded:
   row=rows[name];frame=baseline.joinpath(*safe_rel(row['frame']).parts).read_bytes()
   arr=np.ascontiguousarray(base_codec.decode(frame))
   if arr.nbytes!=row['source_bytes'] or sha(arr.tobytes())!=row['source_sha256']:raise ValueError(f'baseline decoded source mismatch: {name}')
   decoded[name]=arr
  return decoded[name]
 codec_path=ROOT/'bitexact_predictive_codec.py';cpp_path=ROOT/'bitexact_predictive_ans.cpp';mix_path=ROOT/'bitexact_predictive_mix.py'
 if not codec_path.is_file() or not cpp_path.is_file() or not mix_path.is_file():raise ValueError('public predictive encoder sources are incomplete')
 cache=out.parent/'.scratch'/'predictive-candidate-rans';cache.mkdir(parents=True,exist_ok=True)
 codec=codec_module
 if codec is None:
  codec=types.ModuleType('candidate_predictive_codec');codec.__file__=str(codec_path);sys.modules[codec.__name__]=codec
  exec(compile(codec_path.read_bytes(),str(codec_path),'exec'),codec.__dict__)
  codec.ROOT=cache;codec.CACHE_ROOT=cache;codec.BUILD=cache/'build';shutil.copy2(cpp_path,cache/'bitexact_predictive_ans.cpp')
 mix=mix_module
 if mix is None:
  mix=types.ModuleType('candidate_predictive_mix');mix.__file__=str(mix_path);sys.modules[mix.__name__]=mix
  exec(compile(mix_path.read_bytes(),str(mix_path),'exec'),mix.__dict__)
 singles=out/'single';mixed=out/'mixed';singles.mkdir(parents=True,exist_ok=True);mixed.mkdir(parents=True,exist_ok=True)
 profiles=(('quartile','quantile'),('residual','quantile'),('tail','quantile'),('residual','lloyd'))
 default_profile=(('quartile','quantile'),)
 layers={}
 for name in rows:
  if name.startswith('mtp.'):continue
  for role in ('up_proj','gate_proj','down_proj'):
   suffix=f'.mlp.{role}.weight'
   if name.endswith(suffix):
    prefix=name[:-len(suffix)];layers.setdefault(prefix,{})[role]=name
 codec_sha=sha(codec_path.read_bytes());cpp_sha=sha(cpp_path.read_bytes());mix_sha=sha(mix_path.read_bytes())
 for prefix,group in sorted(layers.items()):
  stem_prefix=f'layer-{sha(prefix.encode())}'
  up_name=group.get('up_proj');gate_name=group.get('gate_proj');down_name=group.get('down_proj')
  if up_name and gate_name and words(up_name).shape==words(gate_name).shape:
   target_name=up_name;ref_name=gate_name;target=words(target_name);row=rows[target_name];stem=stem_prefix+'-up'
   receipt_path=singles/(stem+'.json');frame_path=singles/(stem+'.ppcx')
   binding={'target':target_name,'source_sha256':row['source_sha256'],'reference':ref_name,'reference_sha256':rows[ref_name]['source_sha256'],'transform':'identity','prototype_sha256':codec_sha,'cpp_sha256':cpp_sha,'encoder_sha256':codec_sha,'mix_source_sha256':mix_sha}
   previous=_resume_receipt(receipt_path,binding,default_profile,frame_path)
   if previous is None:
    frame=codec.encode(target,words(ref_name),row_profile='quartile',bin_profile='quantile')
    if np.ascontiguousarray(codec.decode(frame,words(ref_name))).tobytes()!=target.tobytes():raise ValueError('up<-gate PPCX candidate failed exact round trip')
    receipt={'status':'PASS','exact':True,**binding,'published_frame_bytes':row['frame_bytes'],'candidate_frame_bytes':len(frame),'saved_bytes':row['frame_bytes']-len(frame),'row_profile':'quartile','bin_profile':'quantile','frame_sha256':sha(frame)}
    _publish_candidate(receipt_path,receipt,[(frame_path,frame)] if receipt['saved_bytes']>0 else [])
  if not down_name or not up_name or not gate_name:continue
  target_name=down_name;row=rows[target_name];target=words(target_name)
  if target.ndim!=2:continue
  y=np.ascontiguousarray(target.T)
  if words(up_name).shape!=y.shape or words(gate_name).shape!=y.shape:continue
  stem=stem_prefix+'-down';receipt_path=singles/(stem+'.json');frame_path=singles/(stem+'.ppcx')
  expected={'target':target_name,'source_sha256':row['source_sha256'],'transform':'target transpose','prototype_sha256':codec_sha,'cpp_sha256':cpp_sha,'encoder_sha256':codec_sha,'mix_source_sha256':mix_sha}
  previous=_resume_receipt(receipt_path,expected,default_profile,frame_path)
  if previous is not None:
   previous_ref=previous.get('reference')
   if previous_ref not in (up_name,gate_name) or previous.get('reference_sha256')!=rows[previous_ref]['source_sha256']:
    raise ValueError('resume candidate reference SHA mismatch')
  if previous is None:
   candidates=[]
   for ref_name in (up_name,gate_name):
    ref=words(ref_name);frame=codec.encode(y,ref,row_profile='quartile',bin_profile='quantile')
    if np.ascontiguousarray(codec.decode(frame,ref)).tobytes()!=y.tobytes():raise ValueError('down PPCX candidate failed exact round trip')
    candidates.append((len(frame),ref_name,frame))
   _,ref_name,frame=min(candidates,key=lambda item:item[0])
   receipt={'status':'PASS','exact':True,**expected,'reference':ref_name,'reference_sha256':rows[ref_name]['source_sha256'],'published_frame_bytes':row['frame_bytes'],'candidate_frame_bytes':len(frame),'saved_bytes':row['frame_bytes']-len(frame),'row_profile':'quartile','bin_profile':'quantile','frame_sha256':sha(frame)}
   _publish_candidate(receipt_path,receipt,[(frame_path,frame)] if receipt['saved_bytes']>0 else [])
  mixed_receipt_path=mixed/(stem+'.json');mixed_frame_path=mixed/(stem+'.ppcx');coeff_path=mixed/(stem_prefix+'-coeff.bf16')
  mixed_binding={'target':target_name,'source_sha256':row['source_sha256'],'references':[up_name,gate_name],
                 'reference_sha256':[rows[up_name]['source_sha256'],rows[gate_name]['source_sha256']],
                 'prototype_sha256':codec_sha,'cpp_sha256':cpp_sha,'encoder_sha256':codec_sha,'mix_source_sha256':mix_sha}
  previous=_resume_receipt(mixed_receipt_path,mixed_binding,profiles,mixed_frame_path,coeff_path)
  if previous is None:
   try:coeff=mix.fit_mixed_coefficients(y,words(up_name),words(gate_name))
   except (ValueError,FloatingPointError):
    for decoded_name in (target_name,up_name,gate_name):decoded.pop(decoded_name,None)
    continue
   derived=np.ascontiguousarray(mix.predict(words(up_name),words(gate_name),coeff));profile_frames=[]
   for row_profile,bin_profile in profiles:
    candidate_frame=codec.encode(y,derived,row_profile=row_profile,bin_profile=bin_profile)
    profile_frames.append((len(candidate_frame),row_profile,bin_profile,candidate_frame))
   _,row_profile,bin_profile,mixed_frame=min(profile_frames,key=lambda item:(item[0],profiles.index((item[1],item[2]))))
   if np.ascontiguousarray(codec.decode(mixed_frame,derived)).tobytes()!=y.tobytes():raise ValueError('mixed PPCX candidate failed exact round trip')
   coeff_raw=np.ascontiguousarray(coeff,dtype='<u2').tobytes()
   receipt={'status':'PASS','exact':True,**mixed_binding,'published_frame_bytes':row['frame_bytes'],'candidate_frame_bytes':len(mixed_frame),'coefficient_bytes':len(coeff_raw),'candidate_total_bytes':len(mixed_frame)+len(coeff_raw),'saved_bytes':row['frame_bytes']-len(mixed_frame)-len(coeff_raw),'row_profile':row_profile,'bin_profile':bin_profile,'frame_sha256':sha(mixed_frame),'coefficients_sha256':sha(coeff_raw),'derived_reference_sha256':sha(derived.tobytes())}
   _publish_candidate(mixed_receipt_path,receipt,[(mixed_frame_path,mixed_frame),(coeff_path,coeff_raw)])
  for decoded_name in (target_name,up_name,gate_name):decoded.pop(decoded_name,None)
 return {'single_receipts':sum(1 for _ in singles.glob('layer*-*.json')),'mixed_receipts':sum(1 for _ in mixed.glob('layer*-*.json')),'output':str(out)}

def _verify_receipt_archive(package:Path,manifest:dict)->None:
 pin=manifest.get('baseline_receipts')
 if pin is None:return # legacy package
 path=package.joinpath(*safe_rel(pin['path']).parts)
 if not path.is_file() or path.stat().st_size!=pin['bytes'] or file_sha(path)!=pin['sha256']:raise ValueError('baseline receipt archive checksum mismatch')
 want={r['path']:r for r in pin['members']};seen=set()
 if len(want)!=len(pin['members']):raise ValueError('duplicate baseline receipt member metadata')
 with tarfile.open(path,'r:xz') as tf:
  for member in tf.getmembers():
   rel=safe_rel(member.name).as_posix()
   if not member.isfile() or not rel.startswith('receipts/') or rel in seen or rel not in want:raise ValueError('unsafe or unexpected baseline receipt member')
   raw=tf.extractfile(member).read();row=want[rel]
   if len(raw)!=row['bytes'] or sha(raw)!=row['sha256']:raise ValueError('baseline receipt raw content mismatch')
   seen.add(rel)
 if seen!=set(want):raise ValueError('baseline receipt archive is incomplete')
def safe_rel(v):
 if not isinstance(v,str):raise ValueError('unsafe relative path')
 p=PurePosixPath(v)
 if not p.parts or p.is_absolute() or '..' in p.parts: raise ValueError('unsafe relative path')
 return p

def baseline_module(tool_path:Path|None=None,package:Path|None=None):
 source=(package/'decoder'/'bitexact_context_package.py') if package is not None else tool_path
 if source is None or not source.is_file():raise ValueError('baseline package helper is missing; pass --baseline-tool during build')
 m=types.ModuleType('baseline_context_package');m.__file__=str(source);sys.modules[m.__name__]=m
 exec(compile(source.read_bytes(),str(source),'exec'),m.__dict__);return m

def load_pair_codec(package:Path,cache:Path):
 src=package/'decoder'/'bitexact_predictive_codec.py'; cpp=package/'decoder'/'bitexact_predictive_ans.cpp'
 if not src.is_file(): src=package/'decoder'/'predictive_pair_codec.py'
 if not cpp.is_file(): cpp=package/'decoder'/'predictive_ans.cpp'
 if not src.is_file() or not cpp.is_file():raise ValueError('predictive decoder source missing')
 m=types.ModuleType('packaged_predictive_pair_codec');m.__file__=str(src);sys.modules[m.__name__]=m
 exec(compile(src.read_bytes(),str(src),'exec'),m.__dict__)
 # Keep compiler output outside the package inventory.
 m.ROOT=cache; m.CACHE_ROOT=cache; m.BUILD=cache/'build'; cache.mkdir(parents=True,exist_ok=True)
 shutil.copy2(cpp,cache/('bitexact_predictive_ans.cpp' if cpp.name=='bitexact_predictive_ans.cpp' else 'predictive_ans.cpp'))
 return m

def load_mix_helper(package:Path):
 src=package/'decoder'/'bitexact_predictive_mix.py'
 if not src.is_file(): src=package/'decoder'/'predictive_mix.py'
 if not src.is_file():raise ValueError('mixed predictor helper missing')
 m=types.ModuleType('packaged_predictive_mix');m.__file__=str(src);sys.modules[m.__name__]=m
 exec(compile(src.read_bytes(),str(src),'exec'),m.__dict__);return m

def inventory(root):
 if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('package contains symlink')
 return [{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':file_sha(p)} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json']

def finalize(root,manifest):
 rows=inventory(root);manifest['files']=rows+[{'path':'manifest.json','bytes':0,'sha256':None}]
 for _ in range(30):
  raw=(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode();manifest['files'][-1]['bytes']=len(raw);manifest['package_bytes']=sum(r['bytes'] for r in manifest['files'])
  raw2=(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode()
  if len(raw2)==len(raw):break
 tmp=root/'manifest.json.partial'
 with tmp.open('wb') as f:f.write(raw2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,root/'manifest.json')

def build(baseline:Path,sweep:Path,out:Path,baseline_tool:Path,mix_sweep:Path|None=None):
 baseline=baseline.resolve();out=out.resolve()
 baseline_tool=baseline_tool.resolve();packaged_tool=baseline/'decoder'/'bitexact_context_package.py'
 if not packaged_tool.is_file() or file_sha(packaged_tool)!=file_sha(baseline_tool):raise ValueError('explicit baseline tool differs from packaged decoder helper')
 if out.is_relative_to(baseline) or baseline.is_relative_to(out):raise ValueError('baseline and output package trees must be separate')
 if out.exists() and any(out.iterdir()):raise ValueError('output must be new or empty')
 out.mkdir(parents=True,exist_ok=True)
 base=baseline_module(tool_path=baseline_tool);cache=out.parent/'.predictor-package-cache';doc=base.verify(baseline,decode_frames=False,cache_dir=cache/'baseverify')
 baseline_manifest_raw=(baseline/'manifest.json').read_bytes();baseline_manifest_xz=lzma.compress(baseline_manifest_raw,format=lzma.FORMAT_XZ,preset=9)
 receipt_archive,receipt_members=_receipt_archive(baseline)
 rows={r['name']:r for r in doc['tensors']}
 if len(rows)!=len(doc['tensors']):raise ValueError('baseline has duplicate tensor names')
 frame_names=[r['frame'] for r in doc['tensors']]
 if len(frame_names)!=len(set(frame_names)):raise ValueError('baseline reuses a frame path across tensors')
 for p in baseline.rglob('*'):
  if p.is_symlink():raise ValueError('baseline package contains symlink')
  if p.is_file():
   rel=p.relative_to(baseline)
   if rel.as_posix()=='manifest.json' or rel.parts[0]=='receipts':continue
   q=out/rel;q.parent.mkdir(parents=True,exist_ok=True)
   if rel.parts[0]=='frames':
    _link_baseline_frame(p,q)
   else:shutil.copy2(p,q)
 (out/'baseline-manifest.json.xz').write_bytes(baseline_manifest_xz)
 (out/'baseline-receipts.tar.xz').write_bytes(receipt_archive)
 decoder_names=('bitexact_predictive_codec.py','bitexact_predictive_ans.cpp','bitexact_predictive_gpu.py','bitexact_predictive_gpu_index.cpp','bitexact_predictive_gpu_kernels.py','bitexact_predictive_mix.py')
 for decoder_name in decoder_names:
  decoder_src=ROOT/decoder_name
  if not decoder_src.is_file():raise ValueError(f'public predictive decoder source missing: {decoder_name}')
  shutil.copy2(decoder_src,out/'decoder'/decoder_name)
 shutil.copy2(Path(__file__),out/'decoder'/'predictor_package.py')
 options={};single_seen=set();mix_seen=set()
 for receipt_path in sorted(sweep.glob('layer*-*.json')):
  if receipt_path.is_symlink():raise ValueError('sweep receipt cannot be a symlink')
  r=json.loads(receipt_path.read_text()); name=r.get('target')
  if not isinstance(name,str) or name in single_seen:raise ValueError('duplicate or missing single candidate target')
  if name.startswith('mtp.'):raise ValueError('MTP tensors must remain unchanged baseline BCTX frames')
  single_seen.add(name)
  if r.get('status')!='PASS' or r.get('exact') is not True:raise ValueError('candidate lacks exact PASS receipt')
  if name not in rows:raise ValueError('candidate target missing from baseline')
  refname=r.get('reference')
  if not isinstance(refname,str) or refname not in rows or refname==name:raise ValueError('invalid named reference')
  base_row=rows[name];ref_row=rows[refname]
  if r.get('source_sha256')!=base_row['source_sha256'] or r.get('reference_sha256')!=ref_row['source_sha256']:raise ValueError('candidate receipt source/reference SHA mismatch')
  if r.get('published_frame_bytes')!=base_row['frame_bytes']:raise ValueError('candidate baseline frame size mismatch')
  trans=r.get('transform')
  if trans not in ('identity','target transpose'):raise ValueError('unknown candidate transform')
  reported_size=r.get('candidate_frame_bytes');reported_saved=r.get('saved_bytes')
  if type(reported_size) is not int or reported_size<=0 or type(reported_saved) is not int or reported_saved!=base_row['frame_bytes']-reported_size:raise ValueError('candidate size/savings receipt mismatch')
  if reported_saved<=0:continue
  frame_rel=safe_rel(receipt_path.stem+'.ppcx');frame_src=sweep/frame_rel
  if type(reported_size) is not int or frame_src.is_symlink() or not frame_src.is_file() or frame_src.stat().st_size!=reported_size:raise ValueError('single candidate frame size/path mismatch')
  candidate={'schema':EXPERIMENT,'mode':'single','name':name,'source_sha256':base_row['source_sha256'],'source_bytes':base_row['source_bytes'],'source_shard':base_row['source_shard'],'shape':base_row['shape'],'frame_bytes':reported_size,'frame_sha256':r.get('frame_sha256'),'saved_bytes':reported_saved,'physical_saved_bytes':reported_saved,'references':[{'name':refname,'sha256':ref_row['source_sha256']}],'transform':trans,'receipt':receipt_path.name,'prototype_sha256':r.get('prototype_sha256'),'cpp_sha256':r.get('cpp_sha256'),'encoder_sha256':r.get('encoder_sha256'),'mix_source_sha256':r.get('mix_source_sha256'),'row_profile':r.get('row_profile'),'bin_profile':r.get('bin_profile'),'_frame_src':frame_src}
  options.setdefault(name,[]).append(candidate)
 if mix_sweep is not None:
  for receipt_path in sorted(mix_sweep.glob('layer*-down.json')):
   if receipt_path.is_symlink():raise ValueError('mix receipt cannot be a symlink')
   r=json.loads(receipt_path.read_text());name=r.get('target')
   if not isinstance(name,str) or name in mix_seen:raise ValueError('duplicate or missing mixed candidate target')
   if name.startswith('mtp.'):raise ValueError('MTP tensors must remain unchanged baseline BCTX frames')
   mix_seen.add(name)
   if r.get('status')!='PASS' or r.get('exact') is not True or name not in rows:raise ValueError('mixed candidate lacks exact source receipt')
   refs=r.get('references')
   if not isinstance(refs,list) or len(refs)!=2 or any(not isinstance(n,str) or n not in rows or n==name for n in refs) or refs[0]==refs[1]:raise ValueError('invalid mixed named references')
   base_row=rows[name];ref_rows=[rows[n] for n in refs]
   if r.get('source_sha256')!=base_row['source_sha256'] or r.get('published_frame_bytes')!=base_row['frame_bytes']:raise ValueError('mixed target source identity mismatch')
   frame_size=r.get('candidate_frame_bytes');coeff_size=r.get('coefficient_bytes');total=r.get('candidate_total_bytes');saved=r.get('saved_bytes')
   if any(type(v) is not int or v<=0 for v in (frame_size,coeff_size,total)) or total!=frame_size+coeff_size or type(saved) is not int or saved!=base_row['frame_bytes']-total:raise ValueError('mixed candidate size/savings metadata invalid')
   if len(ref_rows[0]['shape'])!=len(ref_rows[1]['shape']) or ref_rows[0]['shape']!=ref_rows[1]['shape']:raise ValueError('mixed references have different geometry')
   coeff_shape=[ref_rows[0]['shape'][0] if len(ref_rows[0]['shape'])>1 else 1,3]
   if coeff_size!=math.prod(coeff_shape)*2:raise ValueError('mixed coefficient byte count disagrees with rows x3')
   frame_path=mix_sweep/(receipt_path.stem+'.ppcx');coeff_path=mix_sweep/(receipt_path.stem.removesuffix('-down')+'-coeff.bf16')
   if frame_path.is_symlink() or coeff_path.is_symlink() or not frame_path.is_file() or frame_path.stat().st_size!=frame_size or not coeff_path.is_file() or coeff_path.stat().st_size!=coeff_size:raise ValueError('mixed frame/coefficient file size mismatch')
   coeff_raw=coeff_path.read_bytes()
   if sha(coeff_raw)!=r.get('coefficients_sha256'):raise ValueError('mixed source coefficient checksum mismatch')
   coeff_compressed=zlib.compress(coeff_raw,9);encoding='zlib' if len(coeff_compressed)<len(coeff_raw) else 'identity';coeff_payload=coeff_compressed if encoding=='zlib' else coeff_raw
   physical_saved=base_row['frame_bytes']-frame_size-len(coeff_payload)
   candidate={'schema':EXPERIMENT,'mode':'mixbf16','name':name,'source_sha256':base_row['source_sha256'],'source_bytes':base_row['source_bytes'],'source_shard':base_row['source_shard'],'shape':base_row['shape'],'frame_bytes':frame_size,'frame_sha256':r.get('frame_sha256'),'coefficients':{'bytes':len(coeff_payload),'sha256':sha(coeff_payload),'encoding':encoding,'decoded_bytes':len(coeff_raw),'decoded_sha256':sha(coeff_raw),'shape':coeff_shape,'_file_src':coeff_path,'_payload':coeff_payload},'derived_reference_sha256':r.get('derived_reference_sha256'),'saved_bytes':saved,'physical_saved_bytes':physical_saved,'references':[{'name':n,'sha256':rr['source_sha256']} for n,rr in zip(refs,ref_rows)],'transform':'target transpose','receipt':receipt_path.name,'encoder_sha256':r.get('encoder_sha256'),'mix_source_sha256':r.get('mix_source_sha256'),'row_profile':r.get('row_profile'),'bin_profile':r.get('bin_profile'),'_frame_src':frame_path}
   options.setdefault(name,[]).append(candidate)
 selections={}
 for name,candidates in options.items():
  best=max(candidates,key=lambda c:(c.get('physical_saved_bytes',c['saved_bytes']),c['mode']=='single'))
  if best.get('physical_saved_bytes',best['saved_bytes'])<=0:continue
  frame=best['_frame_src'].read_bytes()
  if len(frame)!=best['frame_bytes'] or sha(frame)!=best['frame_sha256']:raise ValueError('selected frame integrity mismatch')
  fr=f'predictive-frames/{base.key_for(name)}.ppcx';(out/fr).parent.mkdir(parents=True,exist_ok=True);(out/fr).write_bytes(frame)
  candidate={k:v for k,v in best.items() if not k.startswith('_')};candidate['frame']=fr
  if candidate['mode']=='mixbf16':
   coeff=best['coefficients']['_payload']
   if len(coeff)!=best['coefficients']['bytes'] or sha(coeff)!=best['coefficients']['sha256']:raise ValueError('selected coefficient integrity mismatch')
   cp=f'predictive-coefficients/{base.key_for(name)}.bf16';(out/cp).parent.mkdir(parents=True,exist_ok=True);(out/cp).write_bytes(coeff);candidate['coefficients']={k:v for k,v in candidate['coefficients'].items() if not k.startswith('_')};candidate['coefficients']['path']=cp
  selections[name]=candidate
  old_frame=out.joinpath(*safe_rel(rows[name]['frame']).parts)
  if old_frame.exists():old_frame.unlink()
 manifest={'schema':SCHEMA,'complete':True,'verify_on_load':True,'experimental_codec':EXPERIMENT,'baseline_manifest_sha256':sha(baseline_manifest_raw),'baseline_manifest_encoding':'xz','baseline_manifest_bytes':len(baseline_manifest_raw),'baseline_tool_sha256':file_sha(baseline_tool),'baseline_package_bytes':doc['package_bytes'],'baseline_receipts':{'path':'baseline-receipts.tar.xz','bytes':len(receipt_archive),'sha256':sha(receipt_archive),'members':receipt_members},'selected_candidates':selections,'source':doc['source'],'headers':doc['headers'],'metadata_assets':doc['metadata_assets'],'decoder_files':['decoder/bitexact_context_package.py','decoder/bitexact_context_codec.py','decoder/bitexact_context_ans.cpp',*[f'decoder/{x}' for x in decoder_names],'decoder/predictor_package.py']}
 finalize(out,manifest)
 return verify(out,decode_frames=False)

def verify(package:Path,*,decode_frames=True,check_inventory=True):
 manifest=json.loads((package/'manifest.json').read_text())
 if manifest.get('schema') not in (SCHEMA,LEGACY_SCHEMA) or manifest.get('complete') is not True:raise ValueError('unsupported/incomplete predictive package')
 if manifest.get('schema')==SCHEMA and any(k in manifest for k in ('qualification_status','build_progress','targets','measurements','source_paths')):raise ValueError('public predictive manifest contains private build metadata')
 expected={x['path']:x for x in manifest['files']}
 if len(expected)!=len(manifest['files']) or package.is_symlink() or any(p.is_symlink() for p in package.rglob('*')):raise ValueError('duplicate files or symlink')
 for rel,row in expected.items():
  p=safe_rel(rel); f=package.joinpath(*p.parts)
  if not f.is_file() or f.stat().st_size!=row['bytes'] or (check_inventory and row['sha256'] and file_sha(f)!=row['sha256']):raise ValueError('package inventory integrity mismatch')
 actual={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
 if actual!=set(expected) or manifest['package_bytes']!=sum(x['bytes'] for x in manifest['files']):raise ValueError('package inventory/accounting mismatch')
 base=baseline_module(package=package);basecodec=base.decode_codec(package,cache_dir=package.parent/'.predictor-package-cache'/f'baseverify-{os.getpid()}') if decode_frames else None;pair=load_pair_codec(package,package.parent/'.predictor-package-cache'/f'pairverify-{os.getpid()}') if decode_frames else None
 # Baseline receipt identities are retained in the copied parent manifest for every tensor.
 parent=read_baseline_manifest(package)
 if manifest.get('baseline_manifest_bytes') is not None and sum(1 for p in (package/'baseline-manifest.json.xz',package/'baseline-manifest.json') if p.is_file())!=1:raise ValueError('ambiguous compressed baseline manifest')
 if manifest.get('baseline_manifest_bytes') is not None:
  if len((lzma.decompress((package/'baseline-manifest.json.xz').read_bytes()) if manifest.get('baseline_manifest_encoding')=='xz' else (package/'baseline-manifest.json').read_bytes()))!=manifest['baseline_manifest_bytes']:raise ValueError('baseline manifest byte count mismatch')
 _verify_receipt_archive(package,manifest)
 helper=package/'decoder'/'bitexact_context_package.py'
 if not helper.is_file() or file_sha(helper)!=manifest.get('baseline_tool_sha256'):raise ValueError('packaged baseline helper checksum mismatch')
 if any(not package.joinpath(*safe_rel(x).parts).is_file() for x in manifest.get('decoder_files',[])):raise ValueError('decoder artifact missing')
 if manifest.get('schema')==SCHEMA:
  required={'decoder/bitexact_predictive_codec.py','decoder/bitexact_predictive_ans.cpp','decoder/bitexact_predictive_gpu.py','decoder/bitexact_predictive_gpu_index.cpp','decoder/bitexact_predictive_gpu_kernels.py','decoder/bitexact_predictive_mix.py'}
  if not required.issubset(set(manifest.get('decoder_files',[]))):raise ValueError('public predictive decoder inventory is incomplete')
 rows={r['name']:r for r in parent['tensors']}; selected=manifest['selected_candidates']; memo=TensorCache()
 if len(rows)!=len(parent['tensors']) or len(selected)!=len(set(selected)):raise ValueError('duplicate tensor identities')
 if manifest.get('schema')==SCHEMA and manifest.get('verify_on_load') is not True:raise ValueError('public predictive package lacks stable verification metadata')
 candidate_frames=[safe_rel(c.get('frame')).as_posix() for c in selected.values()]
 if len(candidate_frames)!=len(set(candidate_frames)):raise ValueError('candidate frame is shared by multiple tensors')
 actual_candidate_frames={p.relative_to(package).as_posix() for p in (package/'predictive-frames').rglob('*') if p.is_file()} if (package/'predictive-frames').exists() else set()
 if actual_candidate_frames!=set(candidate_frames):raise ValueError('candidate frame inventory mismatch')
 expected_coefficients=set()
 for name,c in selected.items():
  if name not in rows or c.get('schema')!=EXPERIMENT or c.get('name')!=name:raise ValueError('candidate schema/name mismatch')
  refs=c.get('references')
  nrefs=2 if c.get('mode')=='mixbf16' else 1 if c.get('mode')=='single' else 0
  if not isinstance(refs,list) or len(refs)!=nrefs:raise ValueError('candidate reference list invalid')
  for ref in refs:
   if ref.get('name') not in rows or ref.get('name')==name or ref.get('sha256')!=rows[ref['name']]['source_sha256']:raise ValueError('reference identity SHA mismatch')
  if c.get('transform') not in ('identity','target transpose') or c.get('source_sha256')!=rows[name]['source_sha256'] or c.get('source_bytes')!=rows[name]['source_bytes'] or c.get('source_shard')!=rows[name]['source_shard'] or c.get('shape')!=rows[name]['shape']:raise ValueError('candidate source identity/transform invalid')
  fp=package.joinpath(*safe_rel(c['frame']).parts)
  if not fp.is_file() or fp.stat().st_size!=c.get('frame_bytes'):raise ValueError('candidate frame length mismatch')
  if check_inventory and sha(fp.read_bytes())!=c.get('frame_sha256'):raise ValueError('candidate frame integrity mismatch')
  raw_total=fp.stat().st_size;physical_total=raw_total
  if c['mode']=='mixbf16':
   coeff=c.get('coefficients',{});cp=package.joinpath(*safe_rel(coeff.get('path')).parts);shape=coeff.get('shape')
   if not isinstance(shape,list) or len(shape)!=2 or shape[1]!=3 or any(type(x)is not int or x<=0 for x in shape) or coeff.get('decoded_bytes',coeff.get('bytes'))!=math.prod(shape)*2 or not cp.is_file() or cp.stat().st_size!=coeff['bytes']:raise ValueError('mixed coefficient shape/length invalid')
   if len(refs)!=2 or rows[refs[0]['name']]['shape']!=rows[refs[1]['name']]['shape']:raise ValueError('mixed reference geometry mismatch')
   expected_rows=rows[refs[0]['name']]['shape'][0] if len(rows[refs[0]['name']]['shape'])>1 else 1
   if shape!=[expected_rows,3]:raise ValueError('mixed coefficient rows do not match references')
   words=read_coefficients(package,c)
   expected_coefficients.add(safe_rel(coeff['path']).as_posix());raw_total+=coeff.get('decoded_bytes',coeff['bytes']);physical_total+=coeff['bytes']
   if not isinstance(c.get('derived_reference_sha256'),str) or len(c['derived_reference_sha256'])!=64:raise ValueError('derived reference SHA missing')
  elif c.get('mode')!='single':raise ValueError('candidate mode unsupported')
  if type(c.get('saved_bytes')) is not int or c['saved_bytes']!=rows[name]['frame_bytes']-raw_total:raise ValueError('candidate source savings mismatch')
  if c.get('physical_saved_bytes',c['saved_bytes'])!=rows[name]['frame_bytes']-physical_total or c.get('physical_saved_bytes',c['saved_bytes'])<=0:raise ValueError('candidate physical savings mismatch')
 actual_coefficients={p.relative_to(package).as_posix() for p in (package/'predictive-coefficients').rglob('*') if p.is_file()} if (package/'predictive-coefficients').exists() else set()
 if actual_coefficients!=expected_coefficients:raise ValueError('predictive coefficient inventory mismatch')
 mix_helper=load_mix_helper(package) if decode_frames and expected_coefficients else None
 def resolve(name,stack=()):
  cached=memo.get(name)
  if cached is not None:return cached
  if name in stack:raise ValueError('candidate reference cycle')
  if name not in rows:raise ValueError('unknown reference tensor')
  if name in selected:
   c=selected[name]
   if c.get('schema')!=EXPERIMENT or c.get('name')!=name:raise ValueError('candidate schema/name mismatch')
   refs=c['references'];refnames=[x['name'] for x in refs]
   for refrow in refs:
    if refrow['name'] not in rows or refrow['sha256']!=rows[refrow['name']]['source_sha256']:raise ValueError('reference identity SHA mismatch')
   if c.get('transform') not in ('identity','target transpose') or c.get('source_sha256')!=rows[name]['source_sha256'] or c.get('source_bytes')!=rows[name]['source_bytes'] or c.get('source_shard')!=rows[name]['source_shard'] or c.get('shape')!=rows[name]['shape'] or c.get('physical_saved_bytes',c.get('saved_bytes',0))<=0:raise ValueError('candidate source identity/transform/savings invalid')
   frame_path=package.joinpath(*safe_rel(c['frame']).parts);frame=frame_path.read_bytes()
   if len(frame)!=c.get('frame_bytes') or sha(frame)!=c.get('frame_sha256'):raise ValueError('candidate frame integrity mismatch')
   refs_decoded=[resolve(ref,stack+(name,)) for ref in refnames]
   if c['mode']=='mixbf16':
    coeff_words=read_coefficients(package,c)
    derived=np.ascontiguousarray(mix_helper.predict(refs_decoded[0],refs_decoded[1],coeff_words))
    if sha(derived.tobytes())!=c['derived_reference_sha256']:raise ValueError('derived reference checksum mismatch')
    frame_reference=derived
   else:frame_reference=refs_decoded[0]
   z=pair.decode(frame,frame_reference)
   raw=np.ascontiguousarray(z.T if c['transform']=='target transpose' else z)
   if c.get('physical_saved_bytes',c['saved_bytes'])<=0 or c['source_sha256']!=rows[name]['source_sha256']:raise ValueError('candidate not physically beneficial or source identity mismatch')
  else:
   frame_path=package.joinpath(*safe_rel(rows[name]['frame']).parts);raw=np.ascontiguousarray(basecodec.decode(frame_path.read_bytes()))
  if raw.nbytes!=rows[name]['source_bytes'] or sha(raw.tobytes())!=rows[name]['source_sha256']:raise ValueError('tensor decode/source SHA mismatch')
  memo.put(name,raw);return raw
 # Validate the reference graph even when restore defers decode verification to its shard writes.
 visiting=set();visited=set()
 def check_graph(name):
  if name in visiting:raise ValueError('candidate reference cycle')
  if name in visited:return
  visiting.add(name)
  if name in selected:
   for ref in selected[name]['references']:
    if ref['name'] not in rows:raise ValueError('unknown reference tensor')
    check_graph(ref['name'])
  visiting.remove(name);visited.add(name)
 for name in rows:check_graph(name)
 if decode_frames:
  for name in rows:resolve(name)
 return manifest

def restore(package:Path,destination:Path,shard:str|None=None):
 # A shard worker checks its own decoded tensors, header and final shard SHA;
 # full package inventory hashing is available once through `verify`/`aggregate`.
 m=verify(package,decode_frames=False,check_inventory=shard is None)
 if shard is None and destination.exists() and any(destination.iterdir()):raise ValueError('destination must be new or empty')
 destination.mkdir(parents=True,exist_ok=True)
 parent=read_baseline_manifest(package);rows={r['name']:r for r in parent['tensors']};selected=m['selected_candidates']
 base=baseline_module(package=package);basecodec=base.decode_codec(package,cache_dir=package.parent/'.predictor-package-cache'/f'restorebase-{os.getpid()}');pair=load_pair_codec(package,package.parent/'.predictor-package-cache'/f'restorepair-{os.getpid()}');mix_helper=load_mix_helper(package) if any(c['mode']=='mixbf16' for c in selected.values()) else None;memo=TensorCache()
 def resolve(name,stack=()):
  cached=memo.get(name)
  if cached is not None:return cached
  if name in stack:raise ValueError('candidate reference cycle')
  if name not in rows:raise ValueError('unknown tensor reference')
  if name in selected:
   c=selected[name];refs=c['references'];refnames=[x['name'] for x in refs]
   if any(x['sha256']!=rows[x['name']]['source_sha256'] for x in refs):raise ValueError('reference SHA mismatch')
   frame=(package/safe_rel(c['frame'])).read_bytes()
   if len(frame)!=c['frame_bytes'] or sha(frame)!=c['frame_sha256']:raise ValueError('candidate frame integrity mismatch')
   refs_decoded=[resolve(ref,stack+(name,)) for ref in refnames]
   if c['mode']=='mixbf16':
    coeff_words=read_coefficients(package,c);derived=np.ascontiguousarray(mix_helper.predict(refs_decoded[0],refs_decoded[1],coeff_words))
    if sha(derived.tobytes())!=c['derived_reference_sha256']:raise ValueError('derived reference checksum mismatch')
    frame_reference=derived
   else:frame_reference=refs_decoded[0]
   z=pair.decode(frame,frame_reference)
   arr=np.ascontiguousarray(z.T if c['transform']=='target transpose' else z)
  else:arr=np.ascontiguousarray(basecodec.decode((package/safe_rel(rows[name]['frame'])).read_bytes()))
  row=rows[name]
  if arr.nbytes!=row['source_bytes'] or sha(arr.tobytes())!=row['source_sha256']:raise ValueError('restored tensor SHA mismatch')
  memo.put(name,arr);return arr
 by_shard={}
 for row in rows.values():by_shard.setdefault(row['source_shard'],[]).append(row)
 if shard is not None and shard not in by_shard:raise ValueError('requested shard is not in package')
 selected_shards=[shard] if shard is not None else list(by_shard)
 if destination.exists() and not destination.is_dir():raise ValueError('destination must be a directory')
 destination.mkdir(parents=True,exist_ok=True)
 restored={}
 for shard_name in selected_shards:
  tensors=by_shard[shard_name]
  hentry=m['headers'][shard_name];header=(package/safe_rel(hentry['path'])).read_bytes()
  if len(header)!=hentry['bytes'] or sha(header)!=hentry['sha256']:raise ValueError('header hash mismatch')
  out=destination/shard_name;out.parent.mkdir(parents=True,exist_ok=True);partial=out.with_name(out.name+f'.partial.{os.getpid()}')
  try:
   with partial.open('wb') as f:
    f.write(header);end=max(r['offsets'][1] for r in tensors);f.truncate(len(header)+end)
    for row in tensors:
     raw=resolve(row['name']).tobytes();a,b=row['offsets']
     if b-a!=len(raw):raise ValueError('tensor offset mismatch')
     f.seek(len(header)+a);f.write(raw)
   os.replace(partial,out)
  except BaseException:
   partial.unlink(missing_ok=True);raise
  digest=file_sha(out)
  if out.stat().st_size!=m['source']['shards'][shard_name]['bytes'] or digest!=m['source']['shards'][shard_name]['sha256']:raise ValueError(f'restored original shard SHA mismatch: {shard_name}')
  restored[shard_name]={'bytes':out.stat().st_size,'sha256':digest}
 if shard is None:
  with tarfile.open(package/'model-assets.tar.xz','r:xz') as tf:
   members=tf.getmembers()
   if len(members)!=len(m['metadata_assets']) or {x.name for x in members}!=set(m['metadata_assets']):raise ValueError('metadata asset coverage mismatch')
   for member in members:
    rel=safe_rel(member.name)
    if not member.isfile() or rel.parts[0]!='sidecars':raise ValueError('unsafe archive member')
    raw=tf.extractfile(member).read();pin=m['metadata_assets'][member.name]
    if len(raw)!=pin['bytes'] or sha(raw)!=pin['sha256']:raise ValueError('sidecar hash mismatch')
    dest=destination.joinpath(*rel.parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 return {'status':'SHARD_RESTORED' if shard else 'RESTORE_EXACT','tensors':sum(len(by_shard[s]) for s in selected_shards),'shards':restored,'candidates':len(selected),'assets':len(m['metadata_assets']) if shard is None else 0}

def aggregate(package:Path,destination:Path):
 """Verify parallel shard outputs and restore metadata assets once."""
 m=verify(package,decode_frames=False,check_inventory=True)
 if destination.is_symlink() or not destination.is_dir():raise ValueError('restore directory missing or unsafe')
 expected=m['source']['shards'];actual={p.name for p in destination.glob('model-*-of-*.safetensors')}
 if actual!=set(expected):raise ValueError('restored shard population mismatch')
 hashes={}
 for name,pin in expected.items():
  path=destination/name;n=path.stat().st_size;digest=file_sha(path)
  if n!=pin['bytes'] or digest!=pin['sha256']:raise ValueError(f'final shard aggregation SHA mismatch: {name}')
  hashes[name]={'bytes':n,'sha256':digest}
 with tarfile.open(package/'model-assets.tar.xz','r:xz') as tf:
  members=tf.getmembers()
  if len(members)!=len(m['metadata_assets']) or {x.name for x in members}!=set(m['metadata_assets']):raise ValueError('metadata asset coverage mismatch')
  for member in members:
   rel=safe_rel(member.name)
   if not member.isfile() or rel.parts[0]!='sidecars':raise ValueError('unsafe archive member')
   raw=tf.extractfile(member).read();pin=m['metadata_assets'][member.name]
   if len(raw)!=pin['bytes'] or sha(raw)!=pin['sha256']:raise ValueError('sidecar hash mismatch')
   path=destination.joinpath(*rel.parts[1:]);path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists() and (path.stat().st_size!=len(raw) or file_sha(path)!=sha(raw)):raise ValueError('existing restored sidecar mismatch')
   if not path.exists():path.write_bytes(raw)
 return {'status':'RESTORE_EXACT','shards':hashes,'shard_count':len(hashes),'package_bytes':m['package_bytes'],'assets':len(m['metadata_assets'])}

def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
 c=sub.add_parser('generate-candidates');c.add_argument('--baseline',type=Path,required=True);c.add_argument('--out',type=Path,required=True);c.add_argument('--baseline-tool',type=Path,required=True)
 b=sub.add_parser('build');b.add_argument('--baseline',type=Path,required=True);b.add_argument('--sweep',type=Path,required=True);b.add_argument('--out',type=Path,required=True);b.add_argument('--baseline-tool',type=Path,required=True);b.add_argument('--mix-sweep',type=Path)
 v=sub.add_parser('verify');v.add_argument('package',type=Path)
 r=sub.add_parser('restore');r.add_argument('package',type=Path);r.add_argument('destination',type=Path);r.add_argument('--shard')
 g=sub.add_parser('aggregate');g.add_argument('package',type=Path);g.add_argument('destination',type=Path)
 a=ap.parse_args()
 if a.cmd=='generate-candidates':print(json.dumps(generate_candidates(a.baseline,a.out,a.baseline_tool)))
 elif a.cmd=='build':res=build(a.baseline,a.sweep,a.out,a.baseline_tool,a.mix_sweep);print(json.dumps({'built':True,'candidates':len(res['selected_candidates']),'package_bytes':res['package_bytes']}))
 elif a.cmd=='verify':res=verify(a.package);print(json.dumps({'verified':True,'package_bytes':res['package_bytes']}))
 elif a.cmd=='aggregate':print(json.dumps(aggregate(a.package,a.destination)))
 else:print(json.dumps(restore(a.package,a.destination,a.shard)))
if __name__=='__main__':main()
