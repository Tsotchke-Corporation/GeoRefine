"""MLX implementation of the Qwen3.5 packed vision tower.

Parameter names follow Transformers. Linear arrays preserve [out,in] layout;
patch convolution weights are flattened in PyTorch order.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _cfg(c, key, default=None):
    return c.get(key, default) if isinstance(c, dict) else getattr(c, key, default)


def checkpoint_transforms():
    """Return explicit source-name -> target-name/layout binding rules.

    Linear weights and biases bind unchanged as [out,in] / [out]. The patch
    convolution [out,in,t,h,w] is reshaped to [out,in*t*h*w] in PyTorch order;
    the patch projection helper multiplies by its transpose. Embeddings and
    norms bind unchanged.
    """
    return {"linear_weight": "identity [out,in], multiply by transpose",
            "patch_embed.proj.weight": "reshape(out, in*t*h*w)",
            "identity": "copy unchanged"}


def visual_token_count(grid_thw, spatial_merge_size):
    t, h, w = map(int, grid_thw)
    m = int(spatial_merge_size)
    if min(t, h, w, m) <= 0 or h % m or w % m:
        raise ValueError("vision grid must be positive and divisible by spatial_merge_size")
    return t * (h // m) * (w // m)


def insert_visual_embeddings(input_embeddings, visual_embeddings, visual_mask):
    """Replace masked placeholder embeddings, rejecting any count mismatch."""
    mask = mx.array(visual_mask, dtype=mx.bool_)
    if tuple(mask.shape) != tuple(input_embeddings.shape[:-1]):
        raise ValueError("visual_mask shape must match embedding token dimensions")
    n = int(mx.sum(mask).item())
    if visual_embeddings.ndim != 2 or visual_embeddings.shape[0] != n or visual_embeddings.shape[-1] != input_embeddings.shape[-1]:
        raise ValueError("visual embedding count/width does not match placeholders")
    flat = input_embeddings.reshape((-1, input_embeddings.shape[-1]))
    mflat = mask.reshape((-1,))
    # Scatter through indexed assignment is not consistently supported across
    # MLX versions; concatenate selected rows in original order instead.
    vi = 0
    rows = []
    for i, flag in enumerate(mflat.tolist()):
        if flag:
            rows.append(visual_embeddings[vi]); vi += 1
        else:
            rows.append(flat[i])
    return mx.stack(rows).reshape(input_embeddings.shape)


def vision_position_ids(grid_thw, spatial_merge_size=1, start_position=0, temp_merge_size=1, time_interval=1):
    t, h, w = map(int, grid_thw)
    m = int(spatial_merge_size)
    if min(t, h, w, m, temp_merge_size) <= 0 or h % m or w % m or t % temp_merge_size:
        raise ValueError("invalid grid or merge size")
    out = [[], [], []]
    for ti in range(t // temp_merge_size):
        for hi in range(h // m):
            for wi in range(w // m):
                out[0].append(start_position + ti * time_interval)
                out[1].append(start_position + hi)
                out[2].append(start_position + wi)
    return mx.array(out, dtype=mx.int32)


def multimodal_rope_positions(mm_token_type_ids, image_grid_thw=(), video_grid_thw=(), spatial_merge_size=1, attention_mask=None):
    """Build Qwen3.5 3-axis positions and per-row RoPE deltas.

    Inputs are integer nested lists (or MLX arrays). Modality ids: text=0,
    image=1, video=2. Videos are expanded to one temporal frame per grid.
    """
    types = mm_token_type_ids.tolist() if hasattr(mm_token_type_ids, "tolist") else mm_token_type_ids
    if types and isinstance(types[0], int): types = [types]
    mask = attention_mask.tolist() if hasattr(attention_mask, "tolist") else attention_mask
    images = [list(map(int, x)) for x in image_grid_thw]
    videos = []
    for t, h, w in video_grid_thw:
        videos.extend([[1, int(h), int(w)]] * int(t))
    cursors = {1: 0, 2: 0}; result=[]; deltas=[]
    for bi, row in enumerate(types):
        active = [i for i, typ in enumerate(row) if mask is None or mask[bi][i]]
        vals = [row[i] for i in active]
        axes=[[],[],[]]; pos=0; i=0
        while i < len(vals):
            typ=vals[i]; j=i+1
            while j < len(vals) and vals[j] == typ: j += 1
            if typ == 0:
                for axis in axes: axis.extend(range(pos, pos+j-i))
                pos += j-i
            elif typ in (1,2):
                grids=images if typ == 1 else videos
                cur=cursors[typ]
                if cur >= len(grids): raise ValueError(f"missing grid metadata for modality {typ}")
                grid=grids[cur]; cursors[typ]=cur+1
                chunk=vision_position_ids(grid, spatial_merge_size, pos)
                for a in range(3): axes[a].extend(chunk[a].tolist())
                pos += max(grid[1],grid[2]) // spatial_merge_size
            else: raise ValueError(f"unsupported modality type {typ}")
            i=j
        if any(len(a) != len(active) for a in axes):
            raise ValueError("placeholder run length disagrees with visual grid token count")
        dense=[[0]*len(row) for _ in range(3)]
        for a in range(3):
            for k, original in enumerate(active): dense[a][original]=axes[a][k]
        result.append(dense)
        deltas.append((max((max(a) for a in axes if a), default=-1)+1)-len(active))
    if cursors[1]!=len(images) or cursors[2]!=len(videos):
        raise ValueError('unused visual grid metadata')
    return mx.array(result, dtype=mx.int32).transpose((1,0,2)), mx.array(deltas, dtype=mx.int32)[:,None]


class _Linear(nn.Module):
    def __init__(self, a, b, bias=True):
        super().__init__(); self.weight=mx.zeros((b,a)); self.bias=mx.zeros((b,)) if bias else None
    def __call__(self,x): return x @ self.weight.T + self.bias if self.bias is not None else x @ self.weight.T


class _Norm(nn.Module):
    def __init__(self, n):
        super().__init__(); self.weight=mx.ones((n,)); self.bias=mx.zeros((n,)); self.eps=1e-6
    def __call__(self,x):
        dtype=x.dtype; x=x.astype(mx.float32); mean=mx.mean(x,axis=-1,keepdims=True)
        var=mx.mean((x-mean)*(x-mean),axis=-1,keepdims=True)
        return ((x-mean)*mx.rsqrt(var+self.eps)*self.weight.astype(mx.float32)+self.bias.astype(mx.float32)).astype(dtype)


class _PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__(); self.proj=_Linear(_cfg(config,'in_channels')*_cfg(config,'temporal_patch_size')*_cfg(config,'patch_size')**2,_cfg(config,'hidden_size'))
    def __call__(self,x): return self.proj(x)


class _Merger(nn.Module):
    def __init__(self,config):
        super().__init__(); dim=_cfg(config,'hidden_size'); merged=dim*_cfg(config,'spatial_merge_size')**2
        self.norm=_Norm(dim); self.linear_fc1=_Linear(merged,merged); self.linear_fc2=_Linear(merged,_cfg(config,'out_hidden_size'))


def _rotate_half(x):
    a,b=mx.split(x,2,axis=-1); return mx.concatenate((-b,a),axis=-1)


class _VisionBlock(nn.Module):
    def __init__(self,c):
        super().__init__(); d=_cfg(c,'hidden_size'); h=_cfg(c,'num_heads'); mid=_cfg(c,'intermediate_size')
        if d%h: raise ValueError('vision hidden_size must divide evenly across heads')
        self.heads=h; self.dim=d//h
        self.norm1=_Norm(d); self.norm2=_Norm(d); self.attn=_VisionAttention(d,h)
        self.mlp=_VisionMLP(d,mid,_cfg(c,'hidden_act','gelu_pytorch_tanh'))
    def __call__(self,x,cu,cos,sin):
        n,d=x.shape; qkv=self.attn.qkv(self.norm1(x)).reshape((n,3,self.heads,self.dim))
        q,k,v=[qkv[:,i] for i in range(3)]
        qdtype,kdtype=q.dtype,k.dtype
        q=(q.astype(mx.float32)*cos[:,None]+_rotate_half(q.astype(mx.float32))*sin[:,None]).astype(qdtype)
        k=(k.astype(mx.float32)*cos[:,None]+_rotate_half(k.astype(mx.float32))*sin[:,None]).astype(kdtype)
        pieces=[]
        bounds=cu.tolist()
        for lo,hi in zip(bounds[:-1],bounds[1:]):
            if hi<=lo: continue
            qs=q[lo:hi].transpose((1,0,2)); ks=k[lo:hi].transpose((1,0,2)); vs=v[lo:hi].transpose((1,0,2))
            score=(qs @ ks.transpose((0,2,1))) / math.sqrt(self.dim)
            att=mx.softmax(score,axis=-1) @ vs
            pieces.append(att.transpose((1,0,2)).reshape((hi-lo,d)))
        x=x+self.attn.proj(mx.concatenate(pieces,axis=0))
        return x+self.mlp(self.norm2(x))


class _VisionAttention(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.qkv=_Linear(d,3*d); self.proj=_Linear(d,d)


class _VisionMLP(nn.Module):
    def __init__(self,d,mid,activation):
        super().__init__(); self.linear_fc1=_Linear(d,mid); self.linear_fc2=_Linear(mid,d)
        self.activation=activation
        if self.activation not in ('gelu','gelu_pytorch_tanh'):
            raise ValueError(f"unsupported Qwen3.5 vision activation {self.activation!r}")
    def __call__(self,x):
        y=self.linear_fc1(x)
        if self.activation == 'gelu_pytorch_tanh':
            y=0.5*y*(1+mx.tanh(math.sqrt(2/math.pi)*(y+0.044715*y*y*y)))
        else:
            y=0.5*y*(1+mx.erf(y/math.sqrt(2)))
        return self.linear_fc2(y)


class Qwen35VisionModel(nn.Module):
    """Packed Qwen3.5 vision model. Config is HF config or equivalent mapping."""
    def __init__(self, config):
        super().__init__(); self.config=config
        patch=_cfg(config,'patch_size'); temporal=_cfg(config,'temporal_patch_size')
        if not isinstance(patch,int) or not isinstance(temporal,int):
            raise ValueError('MLX Qwen3.5 vision requires scalar square patch_size and temporal_patch_size')
        self.merge=int(_cfg(config,'spatial_merge_size')); self.patch=patch
        self.temporal=temporal; self.channels=int(_cfg(config,'in_channels'))
        self.dim=int(_cfg(config,'hidden_size')); self.grid_side=int(math.sqrt(_cfg(config,'num_position_embeddings')))
        if self.grid_side**2 != _cfg(config,'num_position_embeddings'): raise ValueError('position embedding count must be square')
        self.patch_embed=_PatchEmbed(config)
        self.pos_embed=nn.Embedding(_cfg(config,'num_position_embeddings'),self.dim)
        self.blocks=[_VisionBlock(config) for _ in range(_cfg(config,'depth'))]
        self.merger=_Merger(config)
        self.rotary_dim=self.dim//int(_cfg(config,'num_heads'))//2
        self._inv_freq=mx.array([1/(10000**(i/self.rotary_dim)) for i in range(0,self.rotary_dim,2)],dtype=mx.float32)

    def __call__(self, pixel_values, grid_thw):
        grids=[list(map(int,g)) for g in (grid_thw.tolist() if hasattr(grid_thw,'tolist') else grid_thw)]
        expected=sum(t*h*w for t,h,w in grids)
        if pixel_values.ndim != 2 or pixel_values.shape[0] != expected: raise ValueError('patch input count does not match grid_thw')
        x=self.patch_embed(pixel_values)
        pos=[]; cu=[0]
        for t,h,w in grids:
            if h%self.merge or w%self.merge: raise ValueError('grid dimensions must divide by spatial_merge_size')
            # Match Transformers vision_utils.linspace interpolation and its
            # block-major m×m patch ordering.
            ys=[i*(self.grid_side-1)/(h-1) if h>1 else 0 for i in range(h)]
            xs=[i*(self.grid_side-1)/(w-1) if w>1 else 0 for i in range(w)]
            for _ in range(t):
                for bh in range(h//self.merge):
                    for bw in range(w//self.merge):
                      for dy in range(self.merge):
                       for dx in range(self.merge):
                        fy=ys[bh*self.merge+dy]; fx=xs[bw*self.merge+dx]
                        y0=int(fy); x0=int(fx); y1=min(y0+1,self.grid_side-1); x1=min(x0+1,self.grid_side-1)
                        wy=fy-y0; wx=fx-x0
                        ids=mx.array([y0*self.grid_side+x0,y0*self.grid_side+x1,y1*self.grid_side+x0,y1*self.grid_side+x1])
                        weights=mx.array([(1-wy)*(1-wx),(1-wy)*wx,wy*(1-wx),wy*wx])
                        pos.append(mx.sum(self.pos_embed(ids)*weights[:,None],axis=0))
            for _ in range(t): cu.append(cu[-1]+h*w)
        x=x+mx.stack(pos).astype(x.dtype)
        posids=[]
        for t,h,w in grids:
            hp=[]; wp=[]
            for ti in range(t):
                for bh in range(h//self.merge):
                    for bw in range(w//self.merge):
                        for dh in range(self.merge):
                            for dw in range(self.merge):
                                hp.append(bh*self.merge+dh); wp.append(bw*self.merge+dw)
            posids.append(mx.array([hp,wp],dtype=mx.float32))
        p=mx.concatenate(posids,axis=1).transpose((1,0))
        f=(p[:,:,None]*self._inv_freq[None,None,:]).reshape((p.shape[0],-1))
        emb=mx.concatenate((f,f),axis=-1); cos=mx.cos(emb); sin=mx.sin(emb)
        for block in self.blocks: x=block(x,mx.array(cu),cos,sin)
        merged=[]; unit=self.merge*self.merge
        # Pixel patches and interpolated positions are already block-major as
        # required by the processor, so the reference merger only normalizes
        # and reshapes each consecutive merge group.
        offset=0
        for t,h,w in grids:
            raw=x[offset:offset+t*h*w]
            norm=self.merger.norm(raw).reshape((-1,unit*self.dim))
            y=self.merger.linear_fc1(norm); y=0.5*y*(1+mx.erf(y/math.sqrt(2)))
            merged.append(self.merger.linear_fc2(y)); offset+=t*h*w
        return mx.concatenate(merged,axis=0) if merged else mx.zeros((0,_cfg(self.config,'out_hidden_size')))
