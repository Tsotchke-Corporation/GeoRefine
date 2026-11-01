"""Opt-in diagnostics over the logits actually used to choose target tokens.

The 16-hex SHA256 prefix matches bidec.logits_row_digest's raw-row contract.
No dtype conversion is applied to hashed bytes. Diagnostic runs synchronize and
copy full logits to the host and must not substitute for performance runs.
"""
from __future__ import annotations

import hashlib
import numpy as np


def logits_rows(logits):
    import mlx.core as mx
    kinds={mx.float32:'float32',mx.float16:'float16',mx.bfloat16:'bfloat16'}
    if logits.ndim!=3 or min(logits.shape)<=0 or logits.dtype not in kinds:
        raise ValueError('logit digests require nonempty floating [batch,rows,vocabulary] arrays')
    mx.eval(logits)
    if not np.isfinite(np.asarray(logits.astype(mx.float32))).all():
        raise ValueError('nonfinite target logits cannot produce qualification digests')
    raw=np.asarray(mx.view(logits,mx.uint8))
    dtype=kinds[logits.dtype];vocab=int(logits.shape[-1])
    return [[dict(d=hashlib.sha256(np.ascontiguousarray(raw[b,s]).tobytes()).hexdigest()[:16],
                  dtype=dtype,vocab=vocab) for s in range(logits.shape[1])] for b in range(logits.shape[0])]


def predict(model,hidden,*,exact_digest=False):
    import mlx.core as mx
    logits=model.project(hidden)
    predicted=mx.argmax(logits,axis=-1)
    mx.eval(predicted,hidden)
    return predicted.tolist(),logits_rows(logits) if exact_digest else None
