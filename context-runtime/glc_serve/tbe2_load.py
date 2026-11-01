"""Load a GLC codec v2 or v2.1 artifact for the batch engine (explicit ``--weight-format tbe2`` or ``tbe21``).

The artifact's manifest picks the codec: ``glc-codec-v2`` -> ``TBE2Linear`` (kernel TBE2),
``glc-codec-v2.1`` -> ``TBE21Linear`` (kernel TBE21; split checkpoints are built per fused launch
by ``TBE21Lin``, on the device, from the stored streams).

The skeleton is built with every parameter on ``meta`` (``loader.build_skeleton``), then each
artifact tensor is placed:

  * a GEMM weight (an ``nn.Linear`` weight whose v2 tensor is rows-mode, unpadded, K % 256 == 0)
    is uploaded AS STORED -- its streams, no decode -- into a ``tbe2_desc.TBE2Linear``.
    FastDecoder turns those into ``TBE2Lin`` descriptors (fusing gate|up, q|k|v, qkv|z|b|a) and
    BatchDecoder wraps them for BI-GEMM (``Ld<F_TBE2>``);
  * every other tensor (norms, conv, A_log, the embedding table, ...) is decoded on the CPU by
    ``codec_v2.decode_tensor`` (crc32 per stream + sha256 of the result checked) and placed dense;
  * a TIED ``lm_head`` (no tensor of its own: the 2B) is a ``TBE2Linear`` over the embedding's
    streams, so the head is still a coded GEMM; an untied one (the 27B) is its own tensor.

Load-time gate (``BIDEC_TBE2_G1`` = ``sample`` (default) | ``full`` | ``off``): the uploaded
streams of sampled GEMM weights are decoded ON THE DEVICE by the kernel's own decode path
(``bigemm.decode_weight`` = ``bi_decode_kernel``) and the sha256 of the bytes compared with the
manifest's source sha256.  ``full`` does every coded weight; the standalone, census-backed gate
is ``scripts/batchserve/bi_gate_tbe2.py`` G1.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

_TORCH = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
          "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
          "U8": torch.uint8, "BOOL": torch.bool}
DEVICE_DECODE_HASH_MAX_BYTES = 64 << 20


def _force_dense_tbe21_vision(v21: bool, component: str) -> bool:
    """Vision arithmetic stays in HF's dense modules even when a tensor is GEMM-ready."""
    return bool(v21 and component == "vision")


def _check_package_vision_tensor(source_tensor, source_record, skeleton_shape) -> None:
    """Validate source identity/shape while preserving its trusted recorded dtype."""
    source_dtype = _TORCH.get(source_record.get("dtype"))
    if (source_dtype is None or source_tensor.dtype != source_dtype
            or tuple(source_tensor.shape) != tuple(source_record.get("shape", ()))
            or tuple(source_tensor.shape) != tuple(skeleton_shape)):
        raise ValueError("package vision shape/dtype mismatch")


def _host_tensor(art, name: str) -> torch.Tensor:
    rec = art.stored_record(name)
    arr = art.decode_host(name)
    dt = rec["dtype"]
    if dt in ("BF16", "F16"):
        return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).view(_TORCH[dt]).reshape(rec["shape"])
    return torch.from_numpy(np.ascontiguousarray(arr)).reshape(rec["shape"])


def _runtime_decoder_record(art) -> Optional[Dict[str, str]]:
    record = getattr(art, "runtime_decoder_record", None)
    if record is None:
        record = getattr(getattr(art, "_source", None),
                         "runtime_decoder_record", None)
    return None if record is None else dict(record)


def _slice_device_rows(d, row0: int, row1: int):
    """Make a metadata-bounded row view; keep overflow word offsets absolute."""
    from glc_loader import codec_v2 as cv

    if not (0 <= row0 < row1 <= int(d.N)):
        raise ValueError(f"invalid TBE2 row range [{row0}, {row1}) for N={d.N}")
    if hasattr(d, "l1"):
        nch = int(d.K) // cv.CH
        if (int(d.l1.numel()) != int(d.N) * nch * 16
                or int(d.smb.numel()) != int(d.N) * int(d.K)
                or int(d.rowoff.numel()) != int(d.N)
                or int(d.rowcb.numel()) != int(d.N)
                or int(d.ovf.numel()) < 16):
            raise ValueError("TBE21 device stream geometry mismatch")
        return replace(d, N=row1 - row0, S=1,
                       l1=d.l1.view(int(d.N), -1)[row0:row1].reshape(-1),
                       smb=d.smb.view(int(d.N), -1)[row0:row1].reshape(-1),
                       rowoff=d.rowoff[row0:row1], rowcb=d.rowcb[row0:row1],
                       ckpt=d.ckpt[:0])
    nch = int(d.K) // cv.CH
    nchp = ((nch + 3) // 4) * 4
    ngrp = (nch + cv.GROUP - 1) // cv.GROUP
    if int(d.planes.numel()) != int(d.N) * nch * 8:
        raise ValueError("TBE2 planes geometry mismatch")
    if int(d.smb.numel()) != int(d.N) * int(d.K):
        raise ValueError("TBE2 sign/mantissa geometry mismatch")
    if int(d.grp.numel()) != int(d.N) * ngrp:
        raise ValueError("TBE2 group-index geometry mismatch")
    if int(d.len.numel()) != int(d.N) * nchp + 16:
        raise ValueError("TBE2 length geometry or required 16-byte tail mismatch")
    if int(d.ovf.numel()) < 16:
        raise ValueError("TBE2 overflow stream lacks the required 16-word tail")

    plane_rows = d.planes.view(int(d.N), nch, 8)
    smb_rows = d.smb.view(int(d.N), int(d.K))
    grp_rows = d.grp.view(int(d.N), ngrp)
    len_rows = d.len[:-16].view(int(d.N), nchp)[row0:row1].reshape(-1)
    len_tail = d.len[-16:]
    return replace(
        d, N=row1 - row0,
        planes=plane_rows[row0:row1].reshape(-1),
        smb=smb_rows[row0:row1].reshape(-1),
        grp=grp_rows[row0:row1].reshape(-1),
        # Each row subset receives the original, zero-filled masked-load tail.
        # grp values remain absolute because the complete ovf stream is retained.
        len=torch.cat((len_rows, len_tail)),
    )


def sha256_device_decode(lin, *, S: Optional[int] = None,
                         max_decode_bytes: int = DEVICE_DECODE_HASH_MAX_BYTES,
                         prefix_wg: Optional[int] = None, exec_rows: Optional[int] = None) -> str:
    """Hash every kernel-decoded BF16 byte in bounded row chunks (at most 64 MiB)."""
    if isinstance(max_decode_bytes, bool) or not isinstance(max_decode_bytes, int) or max_decode_bytes <= 0:
        raise ValueError("max_decode_bytes must be a positive integer")
    if max_decode_bytes > DEVICE_DECODE_HASH_MAX_BYTES:
        raise ValueError(f"max_decode_bytes cannot exceed {DEVICE_DECODE_HASH_MAX_BYTES}")
    d = lin.codec_v21 if hasattr(lin, "codec_v21") else lin.codec_v2
    row_bytes = int(d.K) * 2
    if int(d.N) <= 0 or int(d.K) <= 0 or row_bytes > max_decode_bytes:
        raise ValueError(f"decode budget {max_decode_bytes} cannot hold one BF16 row ({row_bytes} bytes)")
    rows_per_chunk = max(1, max_decode_bytes // row_bytes)
    digest = hashlib.sha256()
    total_bytes = 0
    for row0 in range(0, int(d.N), rows_per_chunk):
        row1 = min(int(d.N), row0 + rows_per_chunk)
        chunk = _slice_device_rows(d, row0, row1)
        # ``dense_weight()`` has always decoded with S=1; keep that traversal
        # for the default gate path even though the descriptor is row-sliced.
        if exec_rows is not None:
            if prefix_wg is not None:
                raise ValueError("EXEC and tier-prefix decoder geometries are mutually exclusive")
            decoded = _decode_device_descriptor(chunk, 1 if S is None else S, exec_rows=exec_rows)
        else:
            decoded = (_decode_device_descriptor(chunk, 1 if S is None else S)
                       if prefix_wg is None else
                       _decode_device_descriptor(chunk, 1 if S is None else S, prefix_wg=prefix_wg))
        if (tuple(decoded.shape) != (row1 - row0, int(d.K))
                or decoded.dtype != torch.bfloat16 or decoded.device != d.smb.device):
            raise RuntimeError("TBE2 native decoder returned an invalid chunk")
        byte_view = decoded.contiguous().view(torch.uint8)
        cpu = byte_view.cpu()
        raw = memoryview(cpu.numpy()).cast("B")
        expected_bytes = (row1 - row0) * row_bytes
        if len(raw) != expected_bytes or len(raw) > max_decode_bytes:
            raise RuntimeError("TBE2 native decode hash chunk exceeded its byte budget")
        digest.update(raw)
        total_bytes += len(raw)
        del chunk, decoded, byte_view, cpu, raw
    if total_bytes != int(d.N) * row_bytes:
        raise RuntimeError(f"TBE2 native decode hash covered {total_bytes} bytes, expected {int(d.N) * row_bytes}")
    return digest.hexdigest()


def _decode_device_descriptor(d, S: Optional[int], *, prefix_wg: Optional[int] = None,
                              exec_rows: Optional[int] = None) -> torch.Tensor:
    from . import bigemm as bg
    from .tbe2_desc import arrays_of, tbe21_checkpoints

    if hasattr(d, "l1"):
        split = 1 if S is None else S
        arrays = [d.l1, d.smb, d.ovf, d.rowoff, d.cb16, d.rowcb]
        arrays.append(tbe21_checkpoints(arrays, d.N, d.K, split))
        if exec_rows is not None:
            if prefix_wg is not None:
                raise ValueError("EXEC and tier-prefix decoder geometries are mutually exclusive")
            return bg.decode_weight("tbe21", d.N, d.K, arrays, S=split, exec_rows=exec_rows)
        return (bg.decode_weight("tbe21", d.N, d.K, arrays, S=split) if prefix_wg is None else
                bg.decode_weight("tbe21", d.N, d.K, arrays, S=split, prefix_wg=prefix_wg))

    if prefix_wg is not None or exec_rows is not None:
        raise ValueError("prefix geometry requires a TBE21 descriptor")
    return bg.decode_weight("tbe2", d.N, d.K, arrays_of([d], [0]), S=S)


def _decode(lin, S):
    from . import bigemm as bg
    from .tbe2_desc import TBE21Lin, arrays_of

    if hasattr(lin, "codec_v21"):
        d21 = TBE21Lin([lin], bi=False, S=S)
        return bg.decode_weight("tbe21", d21.N, d21.K, list(d21.arrays), S=S)
    d = lin.codec_v2
    return bg.decode_weight("tbe2", d.N, d.K, arrays_of([d], [0]), S=S)


def _device_g1_gate(coded, gate, weight_format):
    """Full tier geometry checks; default gate retains its original sampling policy."""
    from .loader import LoadError
    g1 = {"mode": gate, "checked": 0, "mismatched": []}
    from . import tbe21_tier_profile as tier_profile
    profile = tier_profile.from_environment(os.environ)
    from . import tbe21_v2_backend as native_backend
    exec_selected = native_backend.selected_exec_layout()
    if exec_selected:
        if weight_format != "tbe21" or gate != "full" or not coded or profile is not None:
            raise LoadError("tbe21_exec_g1_required", "EXEC requires full G1 for every coded TBE21 tensor")
        g1["exec_layout"] = dict(schema="codec.exec.full_g1.v1", oracle="selected_exec_shared_producer.v1",
            coded_tensors=len(coded), geometries=[64, 128], split_policy="S1+shape_selected",
            checked={"64": 0, "128": 0}, mismatched={"64": [], "128": []}, tensors={})
    if profile is not None:
        if weight_format != "tbe21" or gate != "full" or not coded:
            raise LoadError("tbe21_tier_g1_required", "tier-prefix requires BIDEC_TBE2_G1=full and coded TBE21 weights")
        groups = tier_profile.geometries(profile)
        g1["tier_prefix"] = dict(profile=profile, coded_tensors=len(coded),
            geometries=groups, checked={str(wg): 0 for wg in groups},
            mismatched={str(wg): [] for wg in groups})
    if gate != "off" and coded:
        keys = sorted(coded)
        if gate != "full":
            keys = keys[:: max(1, len(keys) // 8)][:8]
        for k in keys:
            mod = coded[k]
            want = (mod.codec_v21 if hasattr(mod, "codec_v21") else mod.codec_v2).sha256
            g1["checked"] += 1
            if profile is None:
                got = sha256_device_decode(mod)
                if got != want:
                    g1["mismatched"].append(k)
                if exec_selected:
                    from . import bigemm as bg
                    descriptor = mod.codec_v21
                    selected_split = bg.split_for(int(descriptor.N), int(descriptor.K),
                                                  device=descriptor.smb.device)
                    splits = sorted({1, selected_split})
                    row = dict(N=int(descriptor.N), K=int(descriptor.K), source_sha256=want,
                        shape_split=selected_split, splits=splits, checked={}, mismatched={})
                    bad = False
                    for rows in (64, 128):
                        key = str(rows); row["checked"][key] = []; row["mismatched"][key] = []
                        for split in splits:
                            actual = sha256_device_decode(mod, S=split, exec_rows=rows)
                            row["checked"][key].append(split)
                            if actual != want:
                                bad = True; row["mismatched"][key].append(split)
                        g1["exec_layout"]["checked"][key] += 1
                        if row["mismatched"][key]:g1["exec_layout"]["mismatched"][key].append(k)
                    g1["exec_layout"]["tensors"][k] = row
                    if bad and k not in g1["mismatched"]:g1["mismatched"].append(k)
            else:
                bad = False
                for wg in groups:
                    got = sha256_device_decode(mod, prefix_wg=wg)
                    g1["tier_prefix"]["checked"][str(wg)] += 1
                    if got != want:
                        bad = True
                        g1["tier_prefix"]["mismatched"][str(wg)].append(k)
                if bad:
                    g1["mismatched"].append(k)
        if g1["mismatched"]:
            raise LoadError("tbe2_g1_failed", f"device decode != source sha256 for {g1['mismatched'][:4]}")
    return g1


def load_v2_model(artifact: str, *, config_dir: Optional[str] = None, device: str = "cuda:0",
                  text_only: bool = True, spec: bool = False, log=print,
                  weight_format: str = "tbe2", embed_placement: str = "device",
                  artifact_factory: Optional[Callable[[str], Any]] = None,
                  before_tbe21_device_use: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    def progress(phase, **detail):
        from glc_serve import startup_progress
        if startup_progress.has_sink():
            startup_progress.report(phase, **detail)
    from transformers import AutoConfig, AutoTokenizer

    from glc_loader import codec_v21 as cv21
    from glc_loader.codec_v2_artifact import V2Artifact, gemm_ready, gemm_ready21, upload, upload21

    from .bundle import component_of
    from .loader import (
        LoadError,
        _disable_cuda_only_conv_kernels,
        _place_param,
        _resolver,
        _set_module,
        build_mtp_skeleton,
        build_skeleton,
        has_mtp_entries,
        split_parent,
        text_config_of,
    )
    from .tbe2_desc import TBE21Linear, TBE2Linear

    t0 = time.time()
    from . import tbe21_v2_backend as native_backend
    if native_backend.selected_exec_layout() and (
            weight_format != "tbe21" or os.environ.get("BIDEC_TBE2_G1", "sample") != "full"):
        raise LoadError("tbe21_exec_g1_required", "EXEC requires full G1 for every coded TBE21 tensor")
    if embed_placement not in ("host", "device"):
        raise ValueError(f"invalid embedding placement {embed_placement!r}")
    if artifact_factory is not None and weight_format != "tbe21":
        raise LoadError("weight_format_mismatch", "artifact_factory requires explicit tbe21")
    if before_tbe21_device_use is not None and weight_format != "tbe21":
        raise LoadError("weight_format_mismatch", "before_tbe21_device_use requires explicit tbe21")
    manifest_before = None
    progress('resolver', step='artifact_open')
    if weight_format == "tbe21":
        manifest_path = Path(artifact).expanduser() / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise LoadError("manifest_missing", "TBE21 requires a regular manifest.json")
        manifest_before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        from glc_loader.codec_v21_fullpack import V21FullPackArtifact

        art = (artifact_factory(artifact) if artifact_factory is not None
               else V21FullPackArtifact(artifact))
    elif weight_format == "tbe2":
        art = V2Artifact(artifact)
        if art.format == cv21.FORMAT:
            raise LoadError("weight_format_mismatch", "v2.1 requires explicit --weight-format tbe21")
    else:
        raise LoadError("weight_format_mismatch", f"unsupported codec path {weight_format!r}")
    try:
        if weight_format == "tbe21" and art.format != cv21.FORMAT:
            raise LoadError("weight_format_mismatch", "tbe21 requires a glc-codec-v2.1 artifact")
        v21 = art.format == cv21.FORMAT

        def coded_module(t, bias, name):
            """(module, device bytes) for a GEMM-ready coded tensor of this artifact's format."""
            if v21:
                dv = upload21(t, dev, 1)
                return TBE21Linear(dv, bias, name=name), dv.stored_bytes
            dv = upload(t, dev)
            return TBE2Linear(dv, bias, name=name), dv.bytes

        ready = gemm_ready21 if v21 else gemm_ready
        dev = torch.device(device)
        names = art.names()
        progress('resolver', step='artifact_ready', total=len(names))
        vision_names = [name for name in names if component_of(name) == "vision"]
        load_package_vision = v21 and not text_only
        metadata = None
        if v21:
            from .tbe21_metadata import load_tbe21_metadata

            metadata = load_tbe21_metadata(art, include_vision=load_package_vision)
            config = metadata.config
            # TBE21 metadata has no persistent model directory; the parent path
            # remains an ignored compatibility input on the explicit TBE21 route.
            cfg_dir = None
        else:
            cfg_dir = config_dir or (getattr(art, "config_path", art.path) if art.has_config() else None)
            if not cfg_dir:
                raise LoadError("no_config", f"{art.path} has no config.json")
            config = AutoConfig.from_pretrained(cfg_dir)
        if load_package_vision and getattr(config, "vision_config", None) is None:
            raise LoadError("vision_config_missing", "TBE21 package vision load requires a vision_config")
        if load_package_vision and not vision_names:
            raise LoadError("missing_vision_weights", "image admission requires vision tensors in the TBE21 package")
        # Keep root's Transformers 5.5 nested-text-config fix; multimodal loads
        # retain the outer config.
        skeleton_config = text_config_of(config) if text_only else config
        progress('model_construct', component='language_model')
        model = build_skeleton(skeleton_config, text_only=text_only, attn_implementation="sdpa")
        _disable_cuda_only_conv_kernels(model, dev)
        mtp = None
        if spec and has_mtp_entries(names):
            progress('model_construct', component='mtp')
            mtp = build_mtp_skeleton(config)
            _disable_cuda_only_conv_kernels(mtp, dev)
        resolve = _resolver(model)
        resolve_mtp = _resolver(mtp) if mtp is not None else None
        counts = {"tbe2_linear": 0, "decoded_dense": 0, "ignored": 0, "alias": 0}
        coded: Dict[str, Any] = {}
        stored_bytes = 0
        for index, name in enumerate(names):
            progress('transcode', done=index, total=len(names), tensor=name, step='tensor_load')
            comp = component_of(name)
            if comp == "mtp":
                if mtp is None:
                    counts["ignored"] += 1
                    continue
                root, local = mtp, resolve_mtp(name[len("mtp."):])
            else:
                if text_only and comp == "vision":
                    counts["ignored"] += 1
                    continue
                root, local = model, resolve(name)
            if local is None:
                counts["ignored"] += 1
                continue
            counts["alias"] += int(art.is_alias(name))
            parent, attr = split_parent(root, local)
            owner = local.rpartition(".")[0]
            force_dense_vision = _force_dense_tbe21_vision(v21, comp)
            if (isinstance(parent, nn.Linear) and attr == "weight"
                    and not force_dense_vision):
                t = art.read(name)                                    # crc32 checked
                if ready(t):
                    bias = parent.bias if (parent.bias is not None and not parent.bias.is_meta) else None
                    mod, nb = coded_module(t, bias, name)
                    _set_module(root, owner, mod)
                    coded[name] = mod
                    counts["tbe2_linear"] += 1
                    stored_bytes += nb
                    if hasattr(art, "release"):
                        art.release(name)
                    del t
                    continue
                del t
            if force_dense_vision:
                source_tensor = _host_tensor(art, name)
                source_record = art.stored_record(name)
                skeleton_tensor = getattr(parent, attr, None)
                try:
                    _check_package_vision_tensor(source_tensor, source_record,
                                                 getattr(skeleton_tensor, "shape", ()))
                except ValueError as exc:
                    raise LoadError("vision_tensor_mismatch", f"{name}: {exc}") from exc
                _place_param(parent, attr, source_tensor, dev)
            elif v21 and embed_placement == "host" and isinstance(parent, nn.Embedding) and attr == "weight":
                from .modules import HostEmbedding

                _set_module(root, owner, HostEmbedding(_host_tensor(art, name), dev,
                                                      padding_idx=parent.padding_idx))
            else:
                _place_param(parent, attr, _host_tensor(art, name), dev)
            counts["decoded_dense"] += 1
            if hasattr(art, "release"):
                art.release(name)
        for root in [model] + ([mtp] if mtp is not None else []):
            for mod in root.modules():
                for key, buf in list(mod._buffers.items()):
                    if buf is not None and not buf.is_meta and buf.device != dev:
                        mod._buffers[key] = buf.to(dev)
        tie = bool(getattr(text_config_of(config), "tie_word_embeddings", False)) or bool(
            getattr(config, "tie_word_embeddings", False))
        lm = model.get_output_embeddings()
        if isinstance(lm, nn.Linear) and lm.weight.is_meta and tie:
            emb_name = next(n for n in names if n.endswith("embed_tokens.weight"))
            t = art.read(emb_name)
            if ready(t):
                mod, _ = coded_module(t, None, emb_name + " (tied lm_head)")
                for path in ("lm_head",):
                    _set_module(model, path, mod)
                coded["lm_head(tied)"] = mod
                counts["tbe2_linear"] += 1
            else:
                lm.weight = model.get_input_embeddings().weight
            if hasattr(art, "release"):
                art.release(emb_name)
            del t
        residual = []
        for root in [model] + ([mtp] if mtp is not None else []):
            residual += [n for n, p in root.named_parameters() if p is not None and p.is_meta]
        if residual:
            raise LoadError("unfilled_parameters", f"{len(residual)} never supplied: {sorted(residual)[:8]}")
        model.eval()
        tok = metadata.tokenizer if metadata is not None else AutoTokenizer.from_pretrained(cfg_dir)
        if weight_format == "tbe21" and before_tbe21_device_use is not None:
            # The Context reader has now restored/transcoded all source tensors and every
            # coded weight is uploaded. Join the owned prewarm before G1 first uses BI-Lin.
            before_tbe21_device_use()
        gate = os.environ.get("BIDEC_TBE2_G1", "sample")
        progress('verify', step='device_g1', total=len(coded), gate=gate)
        g1 = _device_g1_gate(coded, gate, weight_format)
        progress('verify', step='device_g1_complete', done=g1['checked'], total=len(coded), gate=gate)
        rec = {"weight_format": weight_format, "codec": art.format, "artifact": art.path, "config": cfg_dir, "counts": counts,
               "tbe2_device_bytes": stored_bytes, "g1_load_gate": g1,
               "load_s": round(time.time() - t0, 1)}
        if manifest_before is not None:
            rec["manifest_sha256"] = manifest_before
        if metadata is not None:
            rec["metadata_source"] = "sealed_package_assets"
            rec["metadata_asset_sha256"] = metadata.asset_sha256
        if load_package_vision:
            rec["vision_source"] = "tbe21_package"
            rec["vision_tensor_count"] = len(vision_names)
        if hasattr(art, "transcode_receipts"):
            rec["context_transcode"] = {"tensors": len(art.transcode_receipts),
                                        "per_tensor": art.transcode_receipts}
        runtime_decoder = _runtime_decoder_record(art)
        if runtime_decoder is not None:
            rec["runtime_decoder"] = dict(runtime_decoder)
        startup_stats = getattr(art, "parallel_startup_stats", None)
        if startup_stats is not None:
            rec["parallel_context_startup"] = startup_stats()
        log(f"[tbe2] {art.path}: {counts} device streams {stored_bytes / 1e9:.3f} GB, load-time G1 "
            f"{g1['checked']} tensors ({gate}) in {rec['load_s']} s")
        result = {"model": model, "mtp": mtp, "tok": tok, "rec": rec}
        if metadata is not None:
            result.update(config=config, processor=metadata.processor,
                          generation_config=metadata.generation_config)
        return result

    finally:
        close = getattr(art, "close", None)
        if close is not None:
            close()
        if manifest_before is not None:
            if (manifest_path.is_symlink() or not manifest_path.is_file()
                    or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_before):
                raise LoadError("manifest_drift", "TBE21 manifest changed during model load")


__all__ = ["load_v2_model", "sha256_device_decode"]
