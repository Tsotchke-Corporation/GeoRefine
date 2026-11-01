"""Dense Qwen3.5/3.8 text and MTP factories with checkpoint-native parameters.

The visual encoder is not implicitly discarded: text-only package loading must
explicitly enumerate excluded source tensors. No GPU work happens on import.
"""
from __future__ import annotations

from types import SimpleNamespace
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_next import (
    Qwen3NextAttention, Qwen3NextGatedDeltaNet, Qwen3NextMLP,
)
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import KVCache, MambaCache


def text_args(config):
    text = dict(config.get('text_config', config))
    if text.get('model_type') not in ('qwen3_5_text', 'qwen3_8_text', 'qwen3_5', 'qwen3_8'):
        raise ValueError('MLX factory supports dense Qwen3.5/3.8 text only')
    if text.get('num_experts', 0) or text.get('hidden_act', 'silu') != 'silu':
        raise ValueError('unsupported expert or activation configuration')
    required = ('hidden_size', 'num_hidden_layers', 'intermediate_size', 'vocab_size',
                'num_attention_heads', 'num_key_value_heads', 'head_dim',
                'linear_num_value_heads', 'linear_num_key_heads', 'linear_key_head_dim',
                'linear_value_head_dim', 'linear_conv_kernel_dim', 'max_position_embeddings')
    for key in required:
        if type(text.get(key)) is not int or text[key] <= 0:
            raise ValueError(f'invalid or absent architecture dimension: {key}')
    if text['linear_conv_kernel_dim'] < 2:
        raise ValueError('linear convolution requires at least two taps')
    if text['num_attention_heads'] % text['num_key_value_heads']:
        raise ValueError('attention heads must divide into KV groups')
    rope = text.get('rope_parameters', {})
    if rope.get('rope_type', 'default') != 'default':
        raise ValueError('scaled RoPE is not implemented by this factory')
    text.update(rope_theta=rope.get('rope_theta', text.get('rope_theta', 10000000.0)),
                partial_rotary_factor=rope.get('partial_rotary_factor', text.get('partial_rotary_factor', .25)),
                rope_scaling=None, rms_norm_eps=text.get('rms_norm_eps', 1e-6),
                attention_bias=text.get('attention_bias', False),
                tie_word_embeddings=text.get('tie_word_embeddings', False))
    rotary = text['head_dim'] * text['partial_rotary_factor']
    if rotary != int(rotary) or int(rotary) % 2 or not 0 < rotary <= text['head_dim']:
        raise ValueError('invalid partial rotary dimension')
    interval = text.get('full_attention_interval', 4)
    if type(interval) is not int or interval < 1:
        raise ValueError('invalid full_attention_interval')
    text['layer_types'] = text.get('layer_types', [
        'full_attention' if (i+1) % interval == 0 else 'linear_attention'
        for i in range(text['num_hidden_layers'])])
    if (len(text['layer_types']) != text['num_hidden_layers'] or
            any(x not in ('full_attention', 'linear_attention') for x in text['layer_types'])):
        raise ValueError('invalid layer_types')
    return SimpleNamespace(**text)


class OffsetRMSNorm(nn.Module):
    """Preserve the checkpoint's zero-centered norm weights without rewriting bits."""
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.zeros(dim)
        self.eps = eps

    def __call__(self, x):
        value = x.astype(mx.float32)
        value = value * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.eps)
        return (value * (1 + self.weight.astype(mx.float32))).astype(x.dtype)


def positions(starts, length):
    values=mx.array(starts,dtype=mx.int32)[:,None]+mx.arange(length)[None,:]
    return mx.stack([values,values,values],axis=0)


class Attention(Qwen3NextAttention):
    def __init__(self, args):
        super().__init__(args)
        self.q_norm = OffsetRMSNorm(args.head_dim, args.rms_norm_eps)
        self.k_norm = OffsetRMSNorm(args.head_dim, args.rms_norm_eps)
        self._rotary_dim=int(args.head_dim*args.partial_rotary_factor)
        self._rope_theta=args.rope_theta
        self._sections=tuple(getattr(args,'rope_parameters',{}).get('mrope_section',[11,11,10]))

    def _mrope(self,x,position_ids):
        half=self._rotary_dim//2
        inv=self._rope_theta**(-mx.arange(half,dtype=mx.float32)*2/self._rotary_dim)
        freq=position_ids.astype(mx.float32)[...,None]*inv
        selected=[]
        for i in range(half):
            axis=1 if i%3==1 and i<3*self._sections[1] else 2 if i%3==2 and i<3*self._sections[2] else 0
            selected.append(freq[axis,...,i])
        f=mx.stack(selected,axis=-1);f=mx.concatenate([f,f],axis=-1)[:,None]
        cos=mx.cos(f).astype(x.dtype);sin=mx.sin(f).astype(x.dtype)
        part=x[...,:self._rotary_dim]
        a,b=mx.split(part,2,axis=-1)
        rotated=part*cos+mx.concatenate([-b,a],axis=-1)*sin
        return mx.concatenate([rotated,x[...,self._rotary_dim:]],axis=-1)

    def __call__(self,x,mask=None,cache=None,position_ids=None):
        if position_ids is None:return super().__call__(x,mask=mask,cache=cache)
        from mlx_lm.models.base import scaled_dot_product_attention
        batch,length,_=x.shape
        if tuple(position_ids.shape)!=(3,batch,length):raise ValueError('invalid multimodal position shape')
        q,gate=mx.split(self.q_proj(x).reshape(batch,length,self.num_attention_heads,-1),2,axis=-1)
        q=self.q_norm(q).transpose(0,2,1,3)
        k=self.k_norm(self.k_proj(x).reshape(batch,length,self.num_key_value_heads,-1)).transpose(0,2,1,3)
        v=self.v_proj(x).reshape(batch,length,self.num_key_value_heads,-1).transpose(0,2,1,3)
        q=self._mrope(q,position_ids);k=self._mrope(k,position_ids)
        if cache is not None:k,v=cache.update_and_fetch(k,v)
        out=scaled_dot_product_attention(q,k,v,cache=cache,scale=self.scale,mask=mask)
        out=out.transpose(0,2,1,3).reshape(batch,length,-1)
        return self.o_proj(out*mx.sigmoid(gate.reshape(batch,length,-1)))


class DeltaNet(Qwen3NextGatedDeltaNet):
    """Use MLX's recurrent operation with Qwen3.5's separate projection layout."""
    def __init__(self, args):
        super().__init__(args)
        del self.in_proj_qkvz
        del self.in_proj_ba
        self.in_proj_qkv = nn.Linear(args.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(args.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)

    def __call__(self, inputs, mask=None, cache=None):
        from mlx_lm.models.gated_delta import gated_delta_update
        batch, length, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        prior = cache[0] if cache is not None else None
        if prior is None:
            prior = mx.zeros((batch,self.conv_kernel_size-1,self.conv_dim),dtype=inputs.dtype)
        history = mx.concatenate([prior,qkv],axis=1)
        if cache is not None:
            if cache.lengths is not None:
                raise ValueError('padded recurrent batches are not supported by this adapter')
            cache[0] = history[:,-(self.conv_kernel_size-1):]
        filtered = nn.silu(self.conv1d(history))
        q,k,v = mx.split(filtered,[self.key_dim,2*self.key_dim],axis=-1)
        q=q.reshape(batch,length,self.num_k_heads,self.head_k_dim)
        k=k.reshape(batch,length,self.num_k_heads,self.head_k_dim)
        v=v.reshape(batch,length,self.num_v_heads,self.head_v_dim)
        # Qwen3.5 uses epsilon on the SUM of squares, unlike Qwen3Next's
        # RMS shortcut (which adds epsilon after dividing by head dimension).
        q=q*mx.rsqrt(mx.sum(q*q,axis=-1,keepdims=True)+1e-6)*(self.head_k_dim**-.5)
        k=k*mx.rsqrt(mx.sum(k*k,axis=-1,keepdims=True)+1e-6)
        state=cache[1] if cache is not None else None
        out,state=gated_delta_update(q,k,v,self.in_proj_a(inputs),self.in_proj_b(inputs),
                                    self.A_log,self.dt_bias,state,mask,use_kernel=not self.training)
        if cache is not None:
            cache[1]=state
            cache.advance(length)
        z=self.in_proj_z(inputs).reshape(batch,length,self.num_v_heads,self.head_v_dim)
        # Preserve source norm/gate ordering and FP32 accumulation.
        value=out.astype(mx.float32)
        value=value*mx.rsqrt(mx.mean(value*value,axis=-1,keepdims=True)+self.layer_norm_epsilon)
        value=self.norm.weight*value.astype(out.dtype)
        value=(value*nn.silu(z.astype(mx.float32))).astype(out.dtype)
        return self.out_proj(value.reshape(batch,length,self.value_dim))


class Layer(nn.Module):
    def __init__(self, args, kind):
        super().__init__()
        self.is_linear = kind == 'linear_attention'
        if self.is_linear:
            self.linear_attn = DeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.input_layernorm = OffsetRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = Qwen3NextMLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x, cache=None, position_ids=None):
        mask = create_ssm_mask(x, cache) if self.is_linear else create_attention_mask(x, cache)
        op = self.linear_attn if self.is_linear else self.self_attn
        kwargs={} if self.is_linear else dict(position_ids=position_ids)
        x = x + op(self.input_layernorm(x), mask=mask, cache=cache, **kwargs)
        return x + self.mlp(self.post_attention_layernorm(x))


class Trunk(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Layer(args, kind) for kind in args.layer_types]
        self.norm = OffsetRMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, tokens, cache, inputs_embeds=None, position_ids=None):
        if len(cache) != len(self.layers):
            raise ValueError('cache count differs from model layer count')
        x = self.embed_tokens(tokens) if inputs_embeds is None else inputs_embeds
        for layer, state in zip(self.layers, cache):
            x = layer(x, state, position_ids=position_ids)
            # A deferred Metal operator must be fenced before its status can
            # escape the layer that owns its output and cache mutation.
            from glc_loader.metal.status_guard import fence
            fence(x, *_cache_arrays(state))
        return self.norm(x)


def _cache_arrays(state):
    """Return MLX arrays held by a KV/Mamba cache, without retaining state."""
    values = []
    for value in vars(state).values() if hasattr(state, '__dict__') else ():
        if hasattr(value, 'dtype') and hasattr(value, 'shape'):
            values.append(value)
        elif isinstance(value, (list, tuple)):
            values.extend(x for x in value if hasattr(x, 'dtype') and hasattr(x, 'shape'))
    return values


class Model(nn.Module):
    def __init__(self, config, *, with_vision=False):
        super().__init__()
        self.args = text_args(config)
        self._config=config
        if with_vision:
            from .mlx_vision import Qwen35VisionModel
            if 'vision_config' not in config:raise ValueError('vision_config is required')
            self.visual=Qwen35VisionModel(config['vision_config'])
        self.model = Trunk(self.args)
        if not self.args.tie_word_embeddings:
            self.lm_head = nn.Linear(self.args.hidden_size, self.args.vocab_size, bias=False)

    def project(self, hidden):
        return (self.model.embed_tokens.as_linear(hidden) if self.args.tie_word_embeddings
                else self.lm_head(hidden))

    def forward_hidden(self, tokens, cache, *, inputs_embeds=None, position_ids=None):
        return self.model(tokens, cache, inputs_embeds=inputs_embeds,position_ids=position_ids)

    def vision_working_bytes(self,visual_inputs):
        if not hasattr(self,'visual'):raise ValueError('vision was not enabled')
        grids=visual_inputs.get('image_grid_thw',[])
        grids=grids.tolist() if hasattr(grids,'tolist') else grids
        if not grids or any(len(g)!=3 or any(type(v) is not int or v<=0 for v in g) for g in grids):
            raise ValueError('invalid image grids')
        cfg=self._config['vision_config'];patches=sum(t*h*w for t,h,w in grids)
        largest=max(h*w for t,h,w in grids)
        hidden=int(cfg['hidden_size']);mid=int(cfg['intermediate_size'])
        pixels=patches*int(cfg['in_channels'])*int(cfg['temporal_patch_size'])*int(cfg['patch_size'])**2*4
        return pixels+patches*(16*hidden+4*mid)*4+3*int(cfg['num_heads'])*largest*largest*4

    def prepare_visual_inputs(self,tokens,visual_inputs):
        from .mlx_vision import insert_visual_embeddings,multimodal_rope_positions
        if not hasattr(self,'visual'):raise ValueError('vision was not enabled when loading this model')
        required={'pixel_values','image_grid_thw','mm_token_type_ids'}
        if set(visual_inputs)!=required:raise ValueError('image prefill requires pixels, grids and modality token IDs')
        kinds=mx.array(visual_inputs['mm_token_type_ids'],dtype=mx.int32)
        if kinds.shape!=tokens.shape or any(k not in (0,1) for row in kinds.tolist() for k in row):
            raise ValueError('invalid image modality token IDs')
        if not mx.array_equal(kinds==1,tokens==self._config['image_token_id']).item():
            raise ValueError('image token IDs do not match image modality mask')
        grid=visual_inputs['image_grid_thw']
        positions_,deltas=multimodal_rope_positions(kinds,grid,spatial_merge_size=self.visual.merge)
        pixels=mx.array(visual_inputs['pixel_values'],dtype=self.visual.patch_embed.proj.weight.dtype)
        visual=self.visual(pixels,grid)
        embeds=insert_visual_embeddings(self.model.embed_tokens(tokens),visual,kinds==1)
        return embeds,positions_,int(deltas[0,0].item())

    def __call__(self, tokens, cache=None):
        cache = self.make_cache() if cache is None else cache
        return self.project(self.forward_hidden(tokens, cache))

    def cache_reservation_bytes(self, tokens, *, mtp=False):
        # MLX KVCache grows in blocks of 256; reserve FP32 array payloads even
        # when loaded BF16 weights yield narrower caches. Snapshots are separate.
        a=self.args
        rounded=((tokens+255)//256)*256
        kv=2*rounded*a.num_key_value_heads*a.head_dim*4
        recurrent=(a.linear_num_value_heads*a.linear_key_head_dim*a.linear_value_head_dim*4 +
                   (a.linear_conv_kernel_dim-1)*(2*a.linear_num_key_heads*a.linear_key_head_dim+
                                                a.linear_num_value_heads*a.linear_value_head_dim)*4)
        total=sum(recurrent if layer.is_linear else kv for layer in self.model.layers)
        return total + (kv if mtp else 0) + a.hidden_size*4

    def make_cache(self):
        return [MambaCache() if layer.is_linear else KVCache() for layer in self.model.layers]


# Public name used by the standalone MLX serving adapter.
LanguageModel = Model


class ShiftedRoPE(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.rope = rope

    def __call__(self, x, offset=0):
        # MTP begins at position one, but its KV storage begins at index zero.
        return self.rope(x, offset=offset+1)


class MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        args = text_args(config)
        self.pre_fc_norm_embedding = OffsetRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.pre_fc_norm_hidden = OffsetRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.fc = nn.Linear(2*args.hidden_size, args.hidden_size, bias=False)
        self.layers = [Layer(args, 'full_attention')]
        attention = self.layers[0].self_attn
        attention.rope = ShiftedRoPE(attention.rope)
        self.norm = OffsetRMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, hidden, embeds, cache, *, position_ids=None):
        if len(cache) != 1:
            raise ValueError('MTP requires one attention cache')
        x = self.fc(mx.concatenate([self.pre_fc_norm_embedding(embeds),
                                    self.pre_fc_norm_hidden(hidden)], axis=-1))
        return self.norm(self.layers[0](x, cache[0],position_ids=position_ids))


def bindings(model, config, *, mtp=False):
    from .predictive_mlx import TensorBinding, _flatten
    result = {}
    for name, _ in _flatten(model.parameters()):
        source = 'mtp.' + name if mtp else name
        if not mtp and 'text_config' in config and name.startswith('model.'):
            source = 'model.language_model.' + name[len('model.'):]
        if not mtp and name.startswith('visual.'):
            source='model.'+name
        transpose = (0,2,1) if name.endswith('conv1d.weight') else None
        reshape=tuple(_.shape) if name=='visual.patch_embed.proj.weight' else None
        result[name] = TensorBinding((source,), transpose=transpose,reshape=reshape)
    return result
