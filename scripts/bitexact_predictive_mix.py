"""Experimental PPCX mixed-reference derivation with explicit BF16 truncation."""
from __future__ import annotations
import numpy as np

class PredictiveMixError(ValueError):
    pass

def _cpu_words(x, name):
    a=np.asarray(x)
    if a.dtype!=np.uint16 or a.ndim!=2 or not all(a.shape):
        raise PredictiveMixError(f'{name} must be a nonempty uint16 matrix')
    return np.ascontiguousarray(a,dtype='<u2')

def _finite_words(words):
    return ((words.astype(np.uint32)&0x7f80)!=0x7f80)

def fit_mixed_coefficients(target_uint16,up,gate):
    """Fit rowwise OLS coefficients and store each coefficient as BF16 words.

    Inputs and output are uint16 BF16 encodings. Rows are fitted in blocks of
    256 using centered two-predictor least squares in float64. Constant or
    singular rows use a constant-mean predictor.
    """
    target=_cpu_words(target_uint16,'target')
    u=_cpu_words(up,'up'); g=_cpu_words(gate,'gate')
    if target.shape!=u.shape or target.shape!=g.shape:
        raise PredictiveMixError('target/up/gate shape mismatch')
    if not np.all(_finite_words(target)) or not np.all(_finite_words(u)) or not np.all(_finite_words(g)):
        raise PredictiveMixError('nonfinite BF16 input')
    y=(target.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    x=(u.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    z=(g.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    coeff=np.zeros((target.shape[0],3),dtype='<u2')
    for start in range(0,target.shape[0],256):
        stop=min(start+256,target.shape[0])
        a=x[start:stop].copy(); b=z[start:stop].copy(); v=y[start:stop].copy()
        am=a.mean(axis=1); bm=b.mean(axis=1); ym=v.mean(axis=1)
        a-=am[:,None]; b-=bm[:,None]; v-=ym[:,None]
        aa=np.sum(a*a,axis=1); bb=np.sum(b*b,axis=1); ab=np.sum(a*b,axis=1)
        ay=np.sum(a*v,axis=1); by=np.sum(b*v,axis=1)
        det=aa*bb-ab*ab
        good=np.isfinite(det)&(det>0)
        alpha=np.zeros(stop-start,dtype=np.float64); beta=np.zeros_like(alpha)
        alpha[good]=(ay[good]*bb[good]-by[good]*ab[good])/det[good]
        beta[good]=(by[good]*aa[good]-ay[good]*ab[good])/det[good]
        intercept=ym-alpha*am-beta*bm
        block=np.stack((alpha,beta,intercept),axis=1).astype(np.float32)
        block[~good]=np.stack((np.zeros(np.count_nonzero(~good)),np.zeros(np.count_nonzero(~good)),ym[~good]),axis=1)
        with np.errstate(over='ignore',invalid='ignore'):
            words=(block.view(np.uint32)>>16).astype('<u2')
        finite=_finite_words(words)
        if not np.all(finite):
            bad=~np.all(finite,axis=1)
            fallback=np.stack((np.zeros(np.count_nonzero(bad)),np.zeros(np.count_nonzero(bad)),ym[bad]),axis=1).astype(np.float32)
            fw=(fallback.view(np.uint32)>>16).astype('<u2')
            words[bad]=fw
        coeff[start:stop]=words
    return coeff

def _mix_cpu(up,gate,coeff):
    u=_cpu_words(up,'up'); g=_cpu_words(gate,'gate'); c=np.asarray(coeff)
    if u.shape!=g.shape: raise PredictiveMixError('up/gate shape mismatch')
    if c.dtype!=np.uint16 or c.shape!=(u.shape[0],3): raise PredictiveMixError('coeff must be uint16 with shape (rows, 3)')
    c=np.ascontiguousarray(c,dtype='<u2')
    if not np.all(_finite_words(u)) or not np.all(_finite_words(g)) or not np.all(_finite_words(c)):
        raise PredictiveMixError('nonfinite BF16 input or coefficient')
    uf=(u.astype(np.uint32)<<16).view(np.float32); gf=(g.astype(np.uint32)<<16).view(np.float32)
    cf=(c.astype(np.uint32)<<16).view(np.float32)
    # Keep each FP32 operation separate; this matches the frozen experiment and forbids FMA.
    with np.errstate(over='ignore',invalid='ignore'):
        a=np.multiply(uf,cf[:,0,None],dtype=np.float32)
        b=np.multiply(gf,cf[:,1,None],dtype=np.float32)
        y=np.add(a,b,dtype=np.float32)
        y=np.add(y,cf[:,2,None],dtype=np.float32)
    if not np.all(np.isfinite(y)): raise PredictiveMixError('nonfinite FP32 predictor output')
    out=(y.view(np.uint32)>>16).astype('<u2')
    if not np.all(_finite_words(out)): raise PredictiveMixError('nonfinite truncated BF16 output')
    return out

def predict(up,gate,coeff):
    """Derive BF16 words on CPU (uint16) or CUDA (torch.bfloat16).

    CPU inputs and output are uint16 word matrices. CUDA inputs must both be
    CUDA bfloat16 matrices; the result is a CUDA bfloat16 tensor whose bits are
    the high 16 bits of the separately evaluated FP32 expression.
    """
    try:
        import torch
    except ImportError:
        torch=None
    if torch is not None and isinstance(up,torch.Tensor):
        if not isinstance(gate,torch.Tensor) or not isinstance(coeff,torch.Tensor): raise PredictiveMixError('CUDA inputs must all be tensors')
        if up.device.type!='cuda' or gate.device!=up.device or coeff.device!=up.device: raise PredictiveMixError('up, gate, and coeff must share a CUDA device')
        if up.dtype!=torch.bfloat16 or gate.dtype!=torch.bfloat16 or coeff.dtype!=torch.bfloat16: raise PredictiveMixError('CUDA tensors must use bfloat16')
        if up.ndim!=2 or gate.shape!=up.shape or coeff.shape!=(up.shape[0],3) or not up.shape[0] or not up.shape[1]: raise PredictiveMixError('invalid mixed-reference shapes')
        if not bool(torch.isfinite(up).all().item()) or not bool(torch.isfinite(gate).all().item()) or not bool(torch.isfinite(coeff).all().item()): raise PredictiveMixError('nonfinite BF16 input or coefficient')
        uf=up.float(); gf=gate.float(); cf=coeff.float()
        a=torch.mul(uf,cf[:,0,None]); b=torch.mul(gf,cf[:,1,None]); y=torch.add(a,b); y=torch.add(y,cf[:,2,None])
        if not bool(torch.isfinite(y).all().item()): raise PredictiveMixError('nonfinite FP32 predictor output')
        words=torch.bitwise_and(torch.bitwise_right_shift(y.contiguous().view(torch.int32),16),0xffff).to(torch.uint16)
        if bool((torch.bitwise_and(words.to(torch.int32),0x7f80)==0x7f80).any().item()): raise PredictiveMixError('nonfinite truncated BF16 output')
        return words.view(torch.bfloat16)
    return _mix_cpu(up,gate,coeff)
