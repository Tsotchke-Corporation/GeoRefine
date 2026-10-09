"""Native image path for the bidec engine (Qwen3.5/3.8 ``Qwen3_5ForConditionalGeneration``).

docs/serving/BIDEC_MULTIMODAL_20261004.md is the design document; this module is the code.

What "native" means here
------------------------
The text trunk of an image request runs through the SAME ``BatchDecoder.run()`` -> CUDA-graph
step every text request runs through.  Two things differ for an image request, and both are
inputs to that step, never a second forward implementation:

1. **The input embedding rows.**  ``Qwen3_5Model.forward`` builds ``inputs_embeds`` as
   ``embed_tokens(input_ids).masked_scatter(input_ids == image_token_id, image_embeds)``: every
   text row is its embedding row, and the k-th image-placeholder row (in sequence order) is row k
   of the vision tower's merged output.  The engine reproduces exactly that: its step gathers the
   embedding rows as always, then overwrites the rows flagged in the per-step metadata (``mmflag``)
   with the staged image-embedding rows (:func:`apply_embed_override`, a ``torch.where`` select --
   no arithmetic, so the bytes landing in ``h`` are the vision tower's bf16 bytes).

2. **The rotary rows.**  Qwen VL models use M-RoPE: position ids are 3-D (temporal, height,
   width), text tokens advance all three together, an image's tokens get a (t, h, w) grid.  The
   engine's attention kernel (``bidec_kernels.cu`` ``b_attn_prep``, line 216) reads the rotary
   row ``cosT[row_pos + ropeoff[slot]]`` -- one table row per row, offset per slot.  The
   text table holds HF's own rotary output at t = h = w = i.  An image request's prompt rows need
   rows the text table does not have, so the table is EXTENDED by a scratch region: at admission
   the request's prompt cos/sin rows -- computed by HF's own ``rotary_emb`` from the 3-D position
   ids, the same call ``Qwen3_5TextModel.forward`` makes -- are written to a contiguous scratch
   range ``[n_text + base, n_text + base + L)``, and while the slot is prefilling its
   ``ropeoff`` is ``n_text + base``.  Once the prompt is done the slot decodes text, and HF's
   decode positions are ``p + rope_delta`` on all three axes, i.e. the TEXT table row
   ``p + rope_delta``: the slot's ``ropeoff`` becomes ``rope_delta``.  No kernel change: the
   kernel already indexes by ``row_pos + ropeoff[slot]``.

The position ids themselves are computed by :func:`mrope_position_ids`, a replica of
``Qwen3_5Model.get_rope_index`` (transformers 5.13 ``modeling_qwen3_5.py``:1298-1389) for one
unpadded sequence, and cross-checked at request time against HF's own ``get_rope_index``
(:func:`hf_rope_index`) whenever the installed transformers carries it -- a mismatch refuses
the request rather than serving a guess.

Image embeddings are computed by :class:`VisionTower`: HF's own vision module class
(``AutoModel.from_config(config.vision_config)``, which is what ``Qwen3_5Model.__init__`` builds)
holding plain bf16 weights decoded bit-identically from the bundle (TBE entries through the
CPU codec ``glc_loader.tbe_container.decode_tbe``, each tensor checked against the manifest's
``blake2b_source``) or read from the parent's safetensors.  NOT the loader's TBE-serving vision
modules: a ``TBEServeLinear`` computes through its own fused kernel, which is not the parent's
``F.linear`` arithmetic.  One vision forward per request, over exactly that request's images --
never batched across requests (a cross-request batch changes the GEMM shapes, which is a
different arithmetic schedule from the parent's and a batch-invariance leak besides).

Placement (``--vision-placement``):
  device    the tower's bf16 weights resident on the accelerator; exact.
  ondemand  weights in pinned host RAM; uploaded to the accelerator for each image encode and
            released after (bytes recorded); exact -- the identical module, weights and kernels
            as ``device``, only the residency differs.
  host      weights in host RAM and the tower EXECUTED ON THE CPU.  CPU kernels are not the
            GPU parent's kernels, so this placement is NOT bit-exact against a GPU-run parent
            and says so (``vision_bitexact: false``) on /health and in every receipt.
  off       never loaded; image requests are refused with a 400.
"""
from __future__ import annotations

import base64
import bisect
import copy
import hashlib
import inspect
import io
import itertools
import time
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

#: ``--vision-placement`` values.  See the module docstring.
VISION_PLACEMENTS = ("device", "ondemand", "host", "off")
#: Placements whose image embeddings are bit-identical to a GPU-run HF parent BY CONSTRUCTION
#: (same module, same weights, same device kernels).  ``host`` runs the tower on the CPU.
BITEXACT_VISION_PLACEMENTS = ("device", "ondemand")
#: Token-type codes, as ``ProcessorMixin.create_mm_token_type_ids`` assigns them.
TEXT, IMAGE, VIDEO = 0, 1, 2
#: Default size of the rotary scratch region (rows).  One row is rope_dim bf16 cos + sin
#: (Qwen3.8-27B: 64 * 2 * 2 = 256 bytes), so 32768 rows is 8 MiB of device memory.  It bounds the
#: SUM of prompt lengths of image requests that may be prefilling at once.
DEFAULT_MM_ROPE_ROWS = 32768


class MMRequestError(ValueError):
    """A malformed or unservable image request (the HTTP layer answers 400)."""


# =============================================================================== content parts
def has_image_parts(messages: Sequence[Dict[str, Any]]) -> bool:
    """True if any message carries a list ``content`` (OpenAI content parts)."""
    return any(isinstance(m.get("content"), list) for m in messages or [])


def _load_image_bytes(src: str, *, max_bytes: int, fetch_urls: bool, timeout: float) -> bytes:
    if not isinstance(src, str) or not src:
        raise MMRequestError("image_url must be a non-empty string")
    if src.startswith("data:"):
        head, sep, payload = src.partition(",")
        if not sep or ";base64" not in head:
            raise MMRequestError("data: image URLs must be base64 (data:image/...;base64,...)")
        if len(payload) * 3 // 4 > max_bytes:
            raise MMRequestError(f"image larger than --mm-max-image-bytes {max_bytes}")
        try:
            data = base64.b64decode(payload, validate=True)
        except Exception as exc:  # noqa: BLE001 - any decoding failure is the client's input
            raise MMRequestError(f"invalid base64 image payload: {exc}") from None
    elif src.startswith(("http://", "https://")):
        if not fetch_urls:
            raise MMRequestError("http(s) image URLs are disabled on this server "
                                 "(--mm-fetch-urls 0); send a base64 data: URL")
        import urllib.request

        with urllib.request.urlopen(src, timeout=timeout) as r:  # noqa: S310 - opt-in operator flag
            data = r.read(max_bytes + 1)
    else:
        raise MMRequestError("unsupported image URL scheme (data: base64, or http(s) when enabled)")
    if len(data) > max_bytes:
        raise MMRequestError(f"image larger than --mm-max-image-bytes {max_bytes}")
    return data


def decode_image(data: bytes):
    """bytes -> a loaded PIL image, exactly as decoded (no resize, no mode change: the
    parent's processor does its own ``convert_rgb``, so both sides see the same object)."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise MMRequestError(f"could not decode image: {exc}") from None
    return img


def split_openai_messages(messages: Sequence[Dict[str, Any]], *, max_images: int = 4,
                          max_image_bytes: int = 20 << 20, fetch_urls: bool = False,
                          url_timeout: float = 10.0) -> Tuple[List[Dict[str, Any]], List[Any], List[str]]:
    """OpenAI chat messages -> (HF chat-template messages, PIL images in order, image sha256s).

    Content parts accepted: ``{"type": "text", "text": ...}`` and
    ``{"type": "image_url", "image_url": {"url": ...}}`` (also ``"image_url": "<url>"`` and
    ``{"type": "image", "image": "<url>"}``).  Each image part becomes ``{"type": "image"}`` in the
    HF message -- the placeholder Qwen's chat template expands to
    ``<|vision_start|><|image_pad|><|vision_end|>`` and the processor then expands
    ``<|image_pad|>`` to the image's token count.
    """
    out: List[Dict[str, Any]] = []
    images: List[Any] = []
    digests: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            raise MMRequestError("every message must be an object")
        c = m.get("content")
        rest = {k: v for k, v in m.items() if k != "content"}
        if isinstance(c, str):
            out.append({**rest, "content": c})
            continue
        if not isinstance(c, list):
            raise MMRequestError("content must be a string or a list of content parts")
        parts: List[Dict[str, Any]] = []
        for p in c:
            t = p.get("type") if isinstance(p, dict) else None
            if t == "text":
                parts.append({"type": "text", "text": str(p.get("text", ""))})
            elif t in ("image_url", "image"):
                src = p.get("image_url") if t == "image_url" else p.get("image")
                if isinstance(src, dict):
                    src = src.get("url")
                if len(images) >= max_images:
                    raise MMRequestError(f"more than --mm-max-images {max_images} images")
                data = _load_image_bytes(src, max_bytes=max_image_bytes, fetch_urls=fetch_urls,
                                         timeout=url_timeout)
                images.append(decode_image(data))
                digests.append(hashlib.sha256(data).hexdigest())
                parts.append({"type": "image"})
            else:
                raise MMRequestError(f"unsupported content part type {t!r} "
                                     "(text, image_url)")
        out.append({**rest, "content": parts})
    return out, images, digests


# ================================================================================== positions
def token_types(input_ids: Sequence[int], image_token_id: int,
                video_token_id: Optional[int] = None) -> List[int]:
    """``ProcessorMixin.create_mm_token_type_ids`` for one sequence: image 1, video 2, else 0."""
    out = []
    for t in input_ids:
        t = int(t)
        out.append(IMAGE if t == int(image_token_id) else
                   VIDEO if (video_token_id is not None and t == int(video_token_id)) else TEXT)
    return out


def mrope_position_ids(input_ids: Sequence[int], mm_token_type_ids: Sequence[int],
                       image_grid_thw: Optional[torch.Tensor],
                       spatial_merge_size: int) -> Tuple[torch.Tensor, int]:
    """3-D M-RoPE position ids [3, L] and the rope delta for ONE unpadded sequence.

    A replica of ``Qwen3_5Model.get_rope_index`` (transformers 5.13, modeling_qwen3_5.py
    :1298-1389) + ``get_vision_position_ids`` (:1246-1296), images only:

    * the sequence is grouped into maximal runs of equal token type
      (``itertools.groupby`` -- the same grouping HF does);
    * a text run of length n gets ``arange(n) + current_pos`` on all three axes, and
      ``current_pos += n``;
    * an image run takes the next ``image_grid_thw`` row (t, h, w) and gets, with
      ``gh = h // merge, gw = w // merge``: temporal ``arange(t) + current_pos``
      (time_interval 1), height ``arange(gh) + current_pos``, width ``arange(gw) +
      current_pos``, meshgrid'd in (t, h, w) order; then ``current_pos += max(h, w) // merge``;
    * ``rope_delta = max(position) + 1 - L``.

    Video runs are refused: the native path serves images only.
    """
    types_ = [int(x) for x in mm_token_type_ids]
    L = len(types_)
    if L != len(input_ids):
        raise MMRequestError(f"mm_token_type_ids has {L} entries for {len(input_ids)} tokens")
    grids = iter([] if image_grid_thw is None else [tuple(int(v) for v in g)
                                                    for g in image_grid_thw.tolist()])
    m = int(spatial_merge_size)
    parts: List[torch.Tensor] = []
    cur = 0
    pos = 0
    for key, grp in itertools.groupby(types_):
        n = len(list(grp))
        if key == TEXT:
            parts.append(torch.arange(n, dtype=torch.long).view(1, -1).expand(3, -1) + cur)
            cur += n
        elif key == IMAGE:
            try:
                t, h, w = next(grids)
            except StopIteration:
                raise MMRequestError("more image-token runs than image_grid_thw rows") from None
            gt, gh, gw = t, h // m, w // m
            if gt * gh * gw != n:
                raise MMRequestError(f"an image run of {n} tokens does not match its grid "
                                     f"(t={t}, h={h}, w={w}, merge={m}: {gt * gh * gw} tokens)")
            pt = torch.arange(gt, dtype=torch.long)
            ph = torch.arange(gh, dtype=torch.long) + cur
            pw = torch.arange(gw, dtype=torch.long) + cur
            T, Hh, W = torch.meshgrid(pt, ph, pw, indexing="ij")
            v = torch.stack([T, Hh, W], dim=0).reshape(3, -1)
            v[0] += cur
            parts.append(v)
            cur += max(h, w) // m
        else:
            raise MMRequestError("video inputs are not served by the native engine")
        pos += n
    if next(grids, None) is not None:
        raise MMRequestError("image_grid_thw has more rows than the prompt has image runs")
    pos3 = torch.cat(parts, dim=1).reshape(3, -1) if parts else torch.zeros(3, 0, dtype=torch.long)
    delta = int(pos3.max().item()) + 1 - L if L else 0
    return pos3.contiguous(), delta


def hf_rope_index(config, input_ids: Sequence[int], mm_token_type_ids: Sequence[int],
                  image_grid_thw: Optional[torch.Tensor]) -> Optional[Tuple[torch.Tensor, int]]:
    """HF's OWN ``Qwen3_5Model.get_rope_index`` on this sequence, or None when the installed
    transformers has no such method (then the cross-check is reported as not run, never as
    passed).  Called as an unbound function over a minimal ``self`` carrying ``config`` and HF's
    own ``get_vision_position_ids`` -- the engine's text skeleton is not a ``Qwen3_5Model``."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    except Exception:  # noqa: BLE001
        return None
    fn = getattr(Qwen3_5Model, "get_rope_index", None)
    if fn is None:
        return None
    shim = types.SimpleNamespace(config=config)
    gv = getattr(Qwen3_5Model, "get_vision_position_ids", None)
    if gv is not None:
        shim.get_vision_position_ids = types.MethodType(gv, shim)
    ids = torch.tensor([list(int(t) for t in input_ids)], dtype=torch.long)
    kw: Dict[str, Any] = {"image_grid_thw": image_grid_thw, "video_grid_thw": None,
                          "attention_mask": None}
    params = inspect.signature(fn).parameters
    if "mm_token_type_ids" in params:
        kw["mm_token_type_ids"] = torch.tensor([list(int(t) for t in mm_token_type_ids)],
                                               dtype=torch.long)
    pos, delta = fn(shim, ids, **kw)
    return pos[:, 0].to(torch.long).contiguous(), int(delta.reshape(-1)[0].item())


# ================================================================================ rotary rows
def rope_rows(rotary: nn.Module, pos3: torch.Tensor, device,
              dtype=torch.bfloat16) -> Tuple[torch.Tensor, torch.Tensor]:
    """HF's own rotary module on 3-D position ids [3, L] -> (cos, sin), each [L, rope_dim].

    The call is ``Qwen3_5TextModel.forward``'s (``self.rotary_emb(hidden_states, position_ids)``
    with position ids [3, 1, L] and a bf16 ``x``) and the same one ``BatchDecoder._rope_tables``
    makes for the text table: every op in it is per-position elementwise (the K=1 matmul is one
    multiply per element), so a row depends on its own position triple only.
    """
    L = int(pos3.shape[1])
    pid = pos3.view(3, 1, L).to(device)
    c, s = rotary(torch.zeros(1, device=device, dtype=dtype), pid)
    return c[0].contiguous(), s[0].contiguous()


def text_rope_table(rotary: nn.Module, n: int, device,
                    dtype=torch.bfloat16) -> Tuple[torch.Tensor, torch.Tensor]:
    """The text rotary table at t = h = w = i, i in [0, n + 64), in 1024-row calls --
    ``BatchDecoder._rope_tables``' construction exactly (bidec.py), for the CPU emulator.
    The loop itself is ``fastdec.rope_tables`` (ATTN_HEADDIM_20261004), which is this
    construction for a multimodal (mrope) rotary and the plain ``[1, n]`` form otherwise."""
    from . import fastdec as _fdm   # lazy: fastdec imports nothing from here
    return _fdm.rope_tables(rotary, n + 64, device, dtype=dtype)


# ================================================================================ vision tower
def image_embeddings(visual: nn.Module, pixel_values: torch.Tensor,
                     image_grid_thw: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    """``Qwen3_5Model.get_image_features`` + the cat/cast in ``Qwen3_5Model.forward``:
    ``visual(pixel_values.type(visual.dtype), grid_thw=...)``, its merged (pooler) output,
    split per image and concatenated back in order (an identity on the rows), cast to the
    embedding dtype.  ONE call per request: the parent runs exactly one per request too."""
    pv = pixel_values.type(visual.dtype)
    out = visual(pv, grid_thw=image_grid_thw)
    emb = getattr(out, "pooler_output", None)
    if emb is None:
        emb = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(emb, (tuple, list)):
        emb = torch.cat(list(emb), dim=0)
    return emb.to(dtype=out_dtype)


def _strip_visual(name: str) -> Optional[str]:
    """``model.visual.blocks.0.attn.qkv.weight`` -> ``blocks.0.attn.qkv.weight``."""
    n = str(name)
    if n.startswith("visual."):
        return n[len("visual."):]
    i = n.find(".visual.")
    return n[i + len(".visual."):] if i >= 0 else None


def vision_tensors_from_parent(model_dir) -> Dict[str, torch.Tensor]:
    """Every ``*.visual.*`` tensor of a dense HF checkpoint, as stored (bf16), keyed locally."""
    import json
    from pathlib import Path

    from safetensors import safe_open

    d = Path(model_dir)
    idx = d / "model.safetensors.index.json"
    files: Dict[str, List[str]] = {}
    if idx.is_file():
        for k, f in json.loads(idx.read_text())["weight_map"].items():
            if _strip_visual(k) is not None:
                files.setdefault(f, []).append(k)
    else:
        for p in sorted(d.glob("*.safetensors")):
            with safe_open(str(p), framework="pt") as h:
                ks = [k for k in h.keys() if _strip_visual(k) is not None]
            if ks:
                files[p.name] = ks
    out: Dict[str, torch.Tensor] = {}
    for f, keys in files.items():
        with safe_open(str(d / f), framework="pt", device="cpu") as h:
            for k in keys:
                out[_strip_visual(k)] = h.get_tensor(k)
    if not out:
        raise MMRequestError(f"{d}: no vision tensors (model.visual.*) in this checkpoint")
    return out


def vision_tensors_from_bundle(bundle, *, verify: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Every vision tensor of a TBE bundle DECODED to its bf16 bytes, keyed locally.

    Raw entries verbatim; TBE entries through ``glc_loader.tbe_container.decode_tbe`` (the
    pure-torch CPU codec, exact by construction and the same decoder
    ``gate_weight_parity.py`` certifies against an independent census).  With ``verify``, every
    decoded tensor is hashed and compared to the manifest's ``blake2b_source`` (the hash of the
    SOURCE bf16 bytes the transcoder recorded); a mismatch raises.
    """
    from safetensors import safe_open

    from .bundle import blake2b_tensor_hex, component_of, read_entry

    out: Dict[str, torch.Tensor] = {}
    rec = {"n_tensors": 0, "n_tbe": 0, "n_raw": 0, "n_verified": 0, "bytes": 0}
    by_shard: Dict[int, List[Dict[str, Any]]] = {}
    for t in bundle.tensors:
        if component_of(t["name"]) == "vision":
            by_shard.setdefault(int(t["shard"]), []).append(t)
    for si, entries in sorted(by_shard.items()):
        sh = bundle.shards[si]
        res = bundle.source.fetch(sh["file"], sha256=sh["sha256"], nbytes=int(sh["bytes"]))
        with safe_open(str(res.path), framework="pt", device="cpu") as h:
            for e in entries:
                payload = read_entry(h, e)
                if e["kind"] == "raw":
                    t = payload
                    rec["n_raw"] += 1
                else:
                    from glc_loader.tbe_container import decode_tbe

                    t = decode_tbe(payload).reshape([int(d) for d in e["shape"]])
                    rec["n_tbe"] += 1
                want = e.get("blake2b_source")
                if verify and want:
                    got = blake2b_tensor_hex(t)
                    if got != want:
                        raise RuntimeError(f"vision tensor {e['name']}: decoded bytes do not "
                                           f"match the manifest's blake2b_source")
                    rec["n_verified"] += 1
                local = _strip_visual(e["name"])
                if local is None:
                    raise RuntimeError(f"vision entry {e['name']!r} has no '.visual.' prefix")
                out[local] = t.contiguous()
                rec["n_tensors"] += 1
                rec["bytes"] += int(t.numel()) * int(t.element_size())
    if not out:
        raise MMRequestError("this bundle has no vision tensors")
    return out, rec


def build_vision_module(config, *, attn_implementation: str = "sdpa") -> nn.Module:
    """HF's vision tower class for ``config`` (the full ``Qwen3_5Config``), parameters on meta.

    ``AutoModel.from_config(config.vision_config)`` is exactly what ``Qwen3_5Model.__init__``
    constructs; the attention implementation is pinned to the one the parent is loaded with
    (``glc_serve.loader.ServeOptions.attn_implementation``, ``"sdpa"``) because the vision
    attention dispatches on it.
    """
    from transformers import AutoModel

    from .loader import params_on_meta

    vcfg = copy.deepcopy(config.vision_config)
    try:
        vcfg._attn_implementation = attn_implementation
    except Exception:  # noqa: BLE001 - older configs expose it as a plain attribute
        setattr(vcfg, "_attn_implementation", attn_implementation)
    with params_on_meta():
        try:
            m = AutoModel.from_config(vcfg, dtype=torch.bfloat16, attn_implementation=attn_implementation)
        except TypeError:
            m = AutoModel.from_config(vcfg, torch_dtype=torch.bfloat16,
                                      attn_implementation=attn_implementation)
    return m.eval()


class VisionTower:
    """The vision tower under one ``--vision-placement``, holding exact bf16 weights."""

    def __init__(self, config, tensors: Dict[str, torch.Tensor], *, placement: str, device,
                 attn_implementation: str = "sdpa", source: str = "", load_record=None):
        if placement not in VISION_PLACEMENTS or placement == "off":
            raise ValueError(f"VisionTower placement must be one of "
                             f"{tuple(p for p in VISION_PLACEMENTS if p != 'off')}, got {placement!r}")
        self.placement = placement
        self.device = torch.device(device)
        self.attn_implementation = attn_implementation
        self.source = source
        self.load_record = dict(load_record or {})
        t0 = time.perf_counter()
        visual = build_vision_module(config, attn_implementation=attn_implementation)
        want = {n for n, _ in visual.named_parameters()}
        have = set(tensors)
        missing, unexpected = sorted(want - have), sorted(have - want)
        if missing or unexpected:
            raise RuntimeError(f"vision weights do not match the tower: missing {missing[:6]}, "
                               f"unexpected {unexpected[:6]}")
        pin = torch.cuda.is_available() and placement == "ondemand"
        for name, p in list(visual.named_parameters()):
            mod_name, _, attr = name.rpartition(".")
            mod = visual.get_submodule(mod_name) if mod_name else visual
            t = tensors[name]
            if tuple(t.shape) != tuple(p.shape):
                raise RuntimeError(f"vision tensor {name}: shape {tuple(t.shape)} != {tuple(p.shape)}")
            if t.dtype != p.dtype:
                raise RuntimeError(f"vision tensor {name}: dtype {t.dtype} != {p.dtype}")
            t = t.detach().contiguous()
            if pin:
                t = t.pin_memory()
            mod._parameters[attr] = nn.Parameter(t, requires_grad=False)
        self.visual = visual
        self.weight_bytes = sum(int(p.numel()) * int(p.element_size()) for p in visual.parameters())
        # host copies of every param/buffer, for ondemand's upload/release swaps
        self._host_params = [(m, k, t) for m in visual.modules() for k, t in m._parameters.items()
                             if t is not None]
        self._host_bufs = [(m, k, t) for m in visual.modules() for k, t in m._buffers.items()
                           if t is not None]
        self.device_bytes = 0
        self.host_bytes = self.weight_bytes
        if placement == "device":
            self.visual.to(self.device)
            self.device_bytes, self.host_bytes = self.weight_bytes, 0
        self.encodes = 0
        self.last_encode: Dict[str, Any] = {}
        self.load_s = round(time.perf_counter() - t0, 3)

    @property
    def bitexact(self) -> bool:
        return self.placement in BITEXACT_VISION_PLACEMENTS

    @property
    def exec_device(self) -> torch.device:
        return torch.device("cpu") if self.placement == "host" else self.device

    def _upload(self) -> int:
        n = 0
        for m, k, t in self._host_params:
            m._parameters[k] = nn.Parameter(t.to(self.device, non_blocking=True), requires_grad=False)
            n += int(t.numel()) * int(t.element_size())
        for m, k, t in self._host_bufs:
            m._buffers[k] = t.to(self.device, non_blocking=True)
        return n

    def _release(self) -> None:
        for m, k, t in self._host_params:
            m._parameters[k] = nn.Parameter(t, requires_grad=False)
        for m, k, t in self._host_bufs:
            m._buffers[k] = t
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor,
               out_device) -> torch.Tensor:
        """Image embeddings [n_image_tokens, out_hidden] bf16 on ``out_device``."""
        t0 = time.perf_counter()
        dev = self.exec_device
        cuda = dev.type == "cuda" and torch.cuda.is_available()
        rec: Dict[str, Any] = {"placement": self.placement, "uploaded_bytes": 0}
        if cuda:
            torch.cuda.reset_peak_memory_stats(dev)
            rec["mem_before"] = int(torch.cuda.memory_allocated(dev))
        try:
            if self.placement == "ondemand":
                rec["uploaded_bytes"] = self._upload()
            emb = image_embeddings(self.visual, pixel_values.to(dev), image_grid_thw.to(dev))
            emb = emb.to(out_device)
            if cuda:
                torch.cuda.synchronize(dev)
                rec["peak_allocated_during_encode"] = int(torch.cuda.max_memory_allocated(dev))
        finally:
            if self.placement == "ondemand":
                self._release()
        if cuda:
            rec["mem_after_release"] = int(torch.cuda.memory_allocated(dev))
        rec["seconds"] = round(time.perf_counter() - t0, 4)
        rec["n_image_tokens"] = int(emb.shape[0])
        self.encodes += 1
        self.last_encode = rec
        return emb

    def stats(self) -> Dict[str, Any]:
        return {"vision_placement": self.placement, "vision_bitexact": self.bitexact,
                "vision_weight_bytes": self.weight_bytes,
                "vision_resident_bytes": self.device_bytes,
                "vision_host_bytes": self.host_bytes,
                "vision_attn_implementation": self.attn_implementation,
                "vision_source": self.source, "vision_load": self.load_record,
                "vision_encodes": self.encodes, "vision_last_encode": self.last_encode}


# ================================================================================ request plan
@dataclass
class MMInputs:
    """One image request's processor output (CPU): what the engine thread encodes."""
    input_ids: List[int]
    mm_token_type_ids: List[int]
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    image_sha256: List[str] = field(default_factory=list)


@dataclass
class MMPlan:
    """Everything the engine needs to run one image request's prompt natively."""
    prompt_len: int
    image_pos: np.ndarray            # sorted prompt positions of image-placeholder tokens
    embeds: torch.Tensor             # [n_image_tokens, H] bf16; row k -> image_pos[k]
    cos: torch.Tensor                # [L, rope_dim] bf16 (HF rotary on pos3)
    sin: torch.Tensor
    pos3: torch.Tensor               # [3, L] int64, CPU
    rope_delta: int
    scratch_base: int = -1
    hf_rope_check: str = "not_run"   # "equal" | "not_available" | "not_run"
    encode: Dict[str, Any] = field(default_factory=dict)


def build_plan(inputs: MMInputs, embeds: torch.Tensor, *, rotary: nn.Module, config,
               image_token_id: int, spatial_merge_size: int, device,
               check_hf: bool = True) -> MMPlan:
    ids = [int(t) for t in inputs.input_ids]
    types_ = [int(t) for t in inputs.mm_token_type_ids]
    if any(t == IMAGE for t in types_) != (inputs.image_grid_thw is not None and
                                            int(inputs.image_grid_thw.numel()) > 0):
        raise MMRequestError("image tokens and image_grid_thw disagree")
    # The placeholder mask is the TOKEN-ID mask (Qwen3_5Model.get_placeholder_mask), the
    # position grouping is the TYPE mask (get_rope_index); they must agree or HF itself
    # would mis-place the image.
    pos = np.array([i for i, t in enumerate(ids) if t == int(image_token_id)], dtype=np.int64)
    by_type = np.array([i for i, t in enumerate(types_) if t == IMAGE], dtype=np.int64)
    if not np.array_equal(pos, by_type):
        raise MMRequestError("image-token positions and mm_token_type_ids disagree")
    if int(embeds.shape[0]) != int(pos.shape[0]):
        raise MMRequestError(f"image features and image tokens do not match, tokens: "
                             f"{int(pos.shape[0])}, features: {int(embeds.shape[0])}")
    pos3, delta = mrope_position_ids(ids, types_, inputs.image_grid_thw, spatial_merge_size)
    check = "not_run"
    if check_hf:
        ref = hf_rope_index(config, ids, types_, inputs.image_grid_thw)
        if ref is None:
            check = "not_available"
        else:
            if not (torch.equal(ref[0], pos3) and ref[1] == delta):
                raise RuntimeError("native M-RoPE position ids disagree with HF get_rope_index; "
                                   "refusing to serve a guess")
            check = "equal"
    cos, sin = rope_rows(rotary, pos3, device)
    return MMPlan(prompt_len=len(ids), image_pos=pos, embeds=embeds, cos=cos, sin=sin,
                  pos3=pos3, rope_delta=int(delta), hf_rope_check=check)


class MMStraddleError(RuntimeError):
    """One ``run()`` group mixes prompt rows and generated rows of an image request."""


def step_rows(plan: MMPlan, pos0: int, n: int, n_text_rows: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """For one ``(slot, pos0, toks)`` group of an image request: (ropeoff, local image rows,
    embedding row indices).

    Prompt rows (``pos0 + n <= L``) read the slot's scratch range: ``ropeoff = n_text_rows +
    scratch_base`` so row position p reads table row ``n_text_rows + scratch_base + p`` -- that
    request's own HF rotary row for prompt position p.  Generated rows (``pos0 >= L``) read the
    text table at ``p + rope_delta``.  A group may not mix the two (``ropeoff`` is per slot per
    step); the scheduler never builds one (a prompt chunk ends at the prompt, decode / verify rows
    start after it) and the gate replay splits at the prompt boundary.
    """
    L = plan.prompt_len
    if pos0 + n <= L:
        if plan.scratch_base < 0:
            raise RuntimeError("image request prompt rows before attach_mm()")
        lo = int(np.searchsorted(plan.image_pos, pos0, side="left"))
        hi = int(np.searchsorted(plan.image_pos, pos0 + n, side="left"))
        k = np.arange(lo, hi, dtype=np.int64)
        return n_text_rows + plan.scratch_base, plan.image_pos[lo:hi] - pos0, k
    if pos0 >= L:
        return int(plan.rope_delta), np.zeros(0, np.int64), np.zeros(0, np.int64)
    raise MMStraddleError(f"rows {pos0}..{pos0 + n - 1} straddle the prompt end {L}")


class RopeScratch:
    """First-fit allocator of contiguous row ranges in the rotary scratch region."""

    def __init__(self, rows: int):
        self.rows = int(rows)
        self.used: Dict[int, int] = {}          # base -> length

    def alloc(self, n: int) -> Optional[int]:
        n = int(n)
        if n <= 0 or n > self.rows:
            return None
        cur = 0
        for b in sorted(self.used):
            if b - cur >= n:
                break
            cur = max(cur, b + self.used[b])
        if cur + n > self.rows:
            return None
        self.used[cur] = n
        return cur

    def free(self, base: int) -> None:
        self.used.pop(int(base), None)

    def can(self, n: int) -> bool:
        b = self.alloc(n)
        if b is None:
            return False
        self.free(b)
        return True

    def in_use(self) -> int:
        return sum(self.used.values())


class MMState:
    """Per-slot image plans over an EXTENDED rotary table (text rows + scratch rows).

    Shared by ``BatchDecoder`` (device tables, a device staging buffer) and the CPU references
    (``bidec_ref.RefDecoder``, :func:`emulate_step`), so the bookkeeping the gate exercises on the
    CPU is the bookkeeping the engine runs.
    """

    def __init__(self, cosT: torch.Tensor, sinT: torch.Tensor, n_text_rows: int,
                 scratch_rows: int, *, stage: Optional[torch.Tensor] = None):
        if int(cosT.shape[0]) != int(n_text_rows) + int(scratch_rows):
            raise ValueError("cosT must hold n_text_rows + scratch_rows rows")
        self.cosT, self.sinT = cosT, sinT
        self.n_text_rows = int(n_text_rows)
        self.scratch = RopeScratch(scratch_rows)
        self.stage = stage
        self.by_slot: Dict[int, MMPlan] = {}
        self.attached = 0

    def can_attach(self, prompt_len: int) -> bool:
        return self.scratch.can(prompt_len)

    def attach(self, slot: int, plan: MMPlan) -> None:
        if slot in self.by_slot:
            raise RuntimeError(f"slot {slot} already carries an image plan")
        L = plan.prompt_len
        base = self.scratch.alloc(L)
        if base is None:
            raise RuntimeError(f"rotary scratch full: {L} rows wanted, "
                               f"{self.scratch.rows - self.scratch.in_use()} free")
        try:
            a = self.n_text_rows + base
            self.cosT[a:a + L].copy_(plan.cos.to(self.cosT.device))
            self.sinT[a:a + L].copy_(plan.sin.to(self.sinT.device))
            # Tripwire: a TEXT row of the prompt (t == h == w) is, by construction, the text
            # table's row at that position.  If the two ever disagree (a math-mode flag flipped
            # between building the table and this request, a different rotary module, a
            # different device), refuse rather than serve rows from two different arithmetics.
            p3 = plan.pos3
            text = ((p3[0] == p3[1]) & (p3[1] == p3[2]) & (p3[0] < self.n_text_rows)).nonzero().view(-1)
            if text.numel():
                ti = text.to(self.cosT.device)
                tv = p3[0, text].to(self.cosT.device)
                if not (torch.equal(self.cosT[a + ti], self.cosT[tv]) and
                        torch.equal(self.sinT[a + ti], self.sinT[tv])):
                    raise RuntimeError("image request rotary rows disagree with the text table "
                                       "at text positions; refusing")
        except Exception:
            self.scratch.free(base)
            raise
        plan.scratch_base = base
        self.by_slot[int(slot)] = plan
        self.attached += 1

    def detach(self, slot: int) -> None:
        plan = self.by_slot.pop(int(slot), None)
        if plan is not None:
            self.scratch.free(plan.scratch_base)
            plan.scratch_base = -1

    def prepare_step(self, seqs: Sequence[Tuple[int, int, Sequence[int]]],
                     ranges: Sequence[Tuple[int, int]], ropeoff: np.ndarray,
                     mmflag: np.ndarray) -> List[Tuple[int, MMPlan, int]]:
        """Fill one step's ``ropeoff`` (every slot in ``seqs``: 0 for text) and ``mmflag``
        (1 on image-placeholder rows), and stage those rows' embeddings.  Returns
        ``[(row, plan, k)]`` for the image rows (k = the plan's embedding row)."""
        mmflag[:] = 0
        rows: List[Tuple[int, MMPlan, int]] = []
        for (slot, pos0, toks), (r0, n) in zip(seqs, ranges):
            plan = self.by_slot.get(int(slot))
            if plan is None:
                ropeoff[int(slot)] = 0
                continue
            off, local, k = step_rows(plan, int(pos0), int(n), self.n_text_rows)
            ropeoff[int(slot)] = off
            if local.size:
                mmflag[r0 + local] = 1
                rows.extend((int(r0 + l), plan, int(kk)) for l, kk in zip(local, k))
        if self.stage is not None and rows:
            dev = self.stage.device
            by_plan: Dict[int, Tuple[MMPlan, List[int], List[int]]] = {}
            for r, p, k in rows:
                by_plan.setdefault(id(p), (p, [], []))
                by_plan[id(p)][1].append(r)
                by_plan[id(p)][2].append(k)
            for p, rr, kk in by_plan.values():
                dst = torch.tensor(rr, dtype=torch.long, device=dev)
                src = torch.tensor(kk, dtype=torch.long, device=p.embeds.device)
                self.stage.index_copy_(0, dst, p.embeds.index_select(0, src).to(dev))
        return rows

    def stats(self) -> Dict[str, Any]:
        return {"mm_rope_rows": self.scratch.rows, "mm_rope_rows_in_use": self.scratch.in_use(),
                "mm_slots_attached": len(self.by_slot), "mm_attached_total": self.attached}


def apply_embed_override(h: torch.Tensor, mmflag: torch.Tensor, stage: torch.Tensor, Mb: int) -> None:
    """In the step (inside the CUDA graph): rows flagged in ``mmflag`` take the staged image
    embedding, every other row keeps its token embedding.  A select, not arithmetic -- the
    bytes are the vision tower's bytes, exactly what ``masked_scatter`` places."""
    sel = mmflag[:Mb].view(-1, 1) != 0
    h[:Mb].copy_(torch.where(sel, stage[:Mb], h[:Mb]))


def emulate_step(state: MMState, embed_weight: torch.Tensor,
                 seqs: Sequence[Tuple[int, int, Sequence[int]]], *, max_slots: int,
                 max_rows: int) -> Dict[str, Any]:
    """The CPU twin of what ``BatchDecoder.run`` + ``_step`` put in front of the trunk for one
    step: the same row layout (rows in ``seqs`` order, each group's tokens at pos0..), the same
    :meth:`MMState.prepare_step`, the token-embedding gather (``ext().embed`` is a row gather),
    the same :func:`apply_embed_override`, and the attention kernel's rotary indexing
    ``cosT[row_pos + ropeoff[row_slot]]`` (bidec_kernels.cu:216).  Returns the per-row trunk
    inputs ``h`` [M, H], ``cos``/``sin`` [M, rope_dim] and ``ranges``."""
    vt, rs, rp, ranges = [], [], [], []
    r = 0
    for slot, pos0, toks in seqs:
        n = len(toks)
        ranges.append((r, n))
        for t in range(n):
            vt.append(int(toks[t]))
            rs.append(int(slot))
            rp.append(int(pos0) + t)
        r += n
    M = r
    ropeoff = np.zeros(max_slots + 1, dtype=np.int32)
    mmflag = np.zeros(max_rows, dtype=np.int32)
    state.prepare_step(seqs, ranges, ropeoff, mmflag)
    h = embed_weight.index_select(0, torch.tensor(vt, dtype=torch.long)).clone()
    apply_embed_override(h, torch.from_numpy(mmflag[:M].copy()), state.stage, M)
    ri = torch.from_numpy((np.asarray(rp, dtype=np.int64) +
                           ropeoff[np.asarray(rs, dtype=np.int64)].astype(np.int64)))
    return {"h": h, "cos": state.cosT.index_select(0, ri), "sin": state.sinT.index_select(0, ri),
            "ranges": ranges, "ropeoff": ropeoff, "mmflag": mmflag[:M].copy()}


# ================================================================================ the frontend
class MMFrontend:
    """Processor + vision tower + rotary: turns an OpenAI image request into an :class:`MMPlan`.

    ``prepare`` runs in the HTTP thread (CPU only: image decode, the parent's processor);
    ``encode`` runs in the ENGINE thread at admission (the vision forward and the rotary rows
    touch the accelerator, and the engine thread owns it).
    """

    def __init__(self, *, processor, tokenizer, config, tower: VisionTower, rotary: nn.Module,
                 device, max_images: int = 4, max_image_bytes: int = 20 << 20,
                 fetch_urls: bool = False, check_hf_rope: bool = True,
                 max_image_tokens: int = 16384):
        self.processor, self.tok, self.config = processor, tokenizer, config
        self.tower, self.rotary = tower, rotary
        self.device = torch.device(device)
        self.max_images, self.max_image_bytes = int(max_images), int(max_image_bytes)
        self.fetch_urls = bool(fetch_urls)
        self.check_hf_rope = bool(check_hf_rope)
        # Bounds one request's image-token count (= its share of the prompt) and, with it, the
        # vision forward's activation memory (~linear in patches = 4x tokens with sdpa).
        self.max_image_tokens = int(max_image_tokens)
        self.image_token_id = int(config.image_token_id)
        self.video_token_id = getattr(config, "video_token_id", None)
        self.spatial_merge_size = int(config.vision_config.spatial_merge_size)
        self.requests = 0

    @property
    def bitexact(self) -> bool:
        return self.tower.bitexact

    def render(self, hf_messages, chat_template_kwargs) -> str:
        """``glc_serve.engine.Engine.render`` for a chat request: the processor's template when
        it has one (else the tokenizer's), tokenize=False, add_generation_prompt=True."""
        templ = self.processor if hasattr(self.processor, "apply_chat_template") else self.tok
        return templ.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True,
                                         **(chat_template_kwargs or {}))

    def prepare(self, messages, chat_template_kwargs=None) -> Tuple[List[int], Optional[MMInputs]]:
        hf_msgs, images, digests = split_openai_messages(
            messages, max_images=self.max_images, max_image_bytes=self.max_image_bytes,
            fetch_urls=self.fetch_urls)
        text = self.render(hf_msgs, chat_template_kwargs)
        # ``Engine.prepare``'s processor call (one request: padding is a no-op).
        enc = self.processor(text=[text], images=images or None, padding=True, return_tensors="pt")
        ids = [int(t) for t in enc["input_ids"][0].tolist()]
        if not images:
            return ids, None
        n_img_tok = int((enc["image_grid_thw"].prod(-1) // self.spatial_merge_size ** 2).sum())
        if self.max_image_tokens and n_img_tok > self.max_image_tokens:
            raise MMRequestError(f"images need {n_img_tok} tokens > --mm-max-image-tokens "
                                 f"{self.max_image_tokens}; send smaller images")
        return ids, self.inputs_from_tensors(ids, enc, digests)

    def inputs_from_tensors(self, ids: Sequence[int], extra: Dict[str, Any],
                            digests: Optional[List[str]] = None) -> MMInputs:
        """An :class:`MMInputs` from a prompt and the processor's tensors (``pixel_values``,
        ``image_grid_thw``, optional ``mm_token_type_ids``) -- also how the parity gate hands a
        reference trajectory's exact inputs to the native path."""
        ids = [int(t) for t in ids]
        tt = extra.get("mm_token_type_ids")
        if tt is not None:
            tt = [int(t) for t in torch.as_tensor(tt).reshape(-1).tolist()][:len(ids)]
        else:
            tt = token_types(ids, self.image_token_id, self.video_token_id)
        if "pixel_values_videos" in extra and extra["pixel_values_videos"] is not None:
            raise MMRequestError("video inputs are not served by the native engine")
        return MMInputs(input_ids=ids, mm_token_type_ids=tt,
                        pixel_values=torch.as_tensor(extra["pixel_values"]).detach().cpu(),
                        image_grid_thw=torch.as_tensor(extra["image_grid_thw"]).detach().cpu().long(),
                        image_sha256=list(digests or []))

    def encode(self, inputs: MMInputs) -> MMPlan:
        """Engine thread: vision forward (per request) + 3-D positions + rotary rows."""
        t0 = time.perf_counter()
        emb = self.tower.encode(inputs.pixel_values, inputs.image_grid_thw, self.device)
        plan = build_plan(inputs, emb, rotary=self.rotary, config=self.config,
                          image_token_id=self.image_token_id,
                          spatial_merge_size=self.spatial_merge_size, device=self.device,
                          check_hf=self.check_hf_rope)
        plan.encode = {**self.tower.last_encode, "plan_s": round(time.perf_counter() - t0, 4)}
        self.requests += 1
        return plan

    def stats(self) -> Dict[str, Any]:
        return {**self.tower.stats(), "mm_requests_encoded": self.requests,
                "mm_hf_rope_crosscheck": self.check_hf_rope}


def build_frontend(*, config, local_dir, placement: str, device, bundle=None, parent=None,
                   rotary: nn.Module, tokenizer, attn_implementation: str = "sdpa",
                   log=print, **kw) -> Optional[MMFrontend]:
    """The frontend for ``--vision-placement`` (None for ``off`` or a text-only checkpoint)."""
    if placement == "off":
        return None
    if getattr(config, "vision_config", None) is None:
        log("[bidec_mm] checkpoint has no vision_config: image requests disabled")
        return None
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(local_dir))
    if getattr(processor, "image_processor", None) is None:
        raise RuntimeError("multimodal checkpoint but AutoProcessor has no image_processor")
    if bundle is not None:
        tensors, rec = vision_tensors_from_bundle(bundle)
        src = "bundle"
    else:
        tensors, rec, src = vision_tensors_from_parent(parent), {}, "parent"
    tower = VisionTower(config, tensors, placement=placement, device=device,
                        attn_implementation=attn_implementation, source=src, load_record=rec)
    del tensors
    log(f"[bidec_mm] vision tower: {placement}, {tower.weight_bytes} bytes, "
        f"bitexact={tower.bitexact}, source={src}")
    return MMFrontend(processor=processor, tokenizer=tokenizer, config=config, tower=tower,
                      rotary=rotary, device=device, **kw)


__all__ = [
    "BITEXACT_VISION_PLACEMENTS", "DEFAULT_MM_ROPE_ROWS", "MMFrontend", "MMInputs", "MMPlan",
    "MMRequestError", "MMState", "MMStraddleError", "RopeScratch", "VISION_PLACEMENTS",
    "VisionTower", "apply_embed_override", "build_frontend", "build_plan", "build_vision_module",
    "emulate_step", "has_image_parts", "hf_rope_index", "image_embeddings", "mrope_position_ids",
    "rope_rows", "split_openai_messages", "step_rows", "text_rope_table", "token_types",
    "vision_tensors_from_bundle", "vision_tensors_from_parent",
]
