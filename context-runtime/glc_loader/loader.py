"""Instantiate a live model from a GLC-RELEASE artifact.

The load is written against one specific, measured failure mode:
``from_pretrained`` on a compressed checkpoint silently returns config-shaped
tensors and, on some architectures, a headless backbone -- a load that succeeds
and hands back the wrong model.  Every step below is therefore asserted rather
than assumed:

* the skeleton is built on the ``meta`` device, so no dense weight is ever
  allocated and a forgotten tensor cannot be quietly filled with garbage;
* every raw tensor's shape is checked against the safetensors header AND
  against the parameter it is being assigned to;
* after installation the model is walked in full, and **any** parameter or
  buffer still on ``meta`` aborts the load with the offending names.

If this function returns, every tensor in the model came from the artifact.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .artifact import (
    Artifact,
    GLCArtifactError,
    _ShardReader,
    _container_from_entry,
    open_artifact,
)
from .modules import (
    GLCBackendError,
    GLCEmbedding,
    GLCLinear,
    GLCTiedLMHead,
    resolve_backend,
)
from .container import decode_fwp1


@dataclass
class LoadReceipt:
    artifact: str
    backend: str
    device: str
    source_model: str
    builder_git_sha: str
    container: str
    n_coded_linears: int
    n_coded_embeddings: int
    n_raw_tensors: int
    n_tensors_not_in_served_graph: int
    weight_bytes_dense: int
    weight_bytes_resident: int
    weight_ratio: float
    load_seconds: float
    torch_peak_allocated_bytes: Optional[int]
    meta_tensors_remaining: int
    carry_plan: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# which containers to carry dense -- decided by arithmetic, not by a threshold
# ---------------------------------------------------------------------------
def plan_dense_carry(
    coded: Sequence[Tuple[str, int, int, int]],
) -> Dict[str, Any]:
    """Choose which coded tensors to decode ONCE at load rather than per call.

    A backend that keeps a container compressed and decodes it inside
    ``forward`` pays, at the instant of that call, the tensor's FULL dense size
    in transient memory.  That transient is not amortised and it is not shared:
    it is simply added to everything else resident.  So the peak of a
    "compressed" load is

        sum(resident of the coded) + sum(dense of the carried) + max(transient)

    and the largest single container decides the last term on its own.  On
    Llama-3.2-1B the tied 128256 x 2048 table is 501.0 MiB of that, which is why
    `resident` measured a peak 154.8 MiB ABOVE the dense model it was meant to
    shrink even though its steady-state weights were 514.6 MiB smaller.

    Carrying that one table dense costs ``dense - resident`` and removes it from
    the max, and whether that trades well is arithmetic on numbers the index
    already carries.  Sorting by transient descending, the candidate plans are
    exactly the prefixes of that order, so evaluating all of them is O(n log n)
    and the chosen one is the true minimum over this family rather than a
    heuristic.  No size threshold appears anywhere: a threshold inherited from a
    different model is how this repository once shipped a 1.027x artifact.

    ``coded`` is ``(name, dense_bytes, resident_bytes, transient_if_coded)``.
    ``transient_if_coded`` is 0 for a container that is never decoded whole --
    an embedding read by row lookup -- and the dense size for one that is.
    """
    items = sorted(coded, key=lambda c: (-int(c[3]), -int(c[1]), c[0]))
    n = len(items)
    if n == 0:
        return {
            "dense_names": [],
            "predicted_peak_weight_bytes": 0,
            "predicted_peak_all_coded_bytes": 0,
            "predicted_peak_all_dense_bytes": 0,
            "n_candidates_evaluated": 0,
        }
    suffix_resident = [0] * (n + 1)
    suffix_max_transient = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_resident[i] = suffix_resident[i + 1] + int(items[i][2])
        suffix_max_transient[i] = max(suffix_max_transient[i + 1], int(items[i][3]))
    prefix_dense = 0
    best_j = 0
    best_peak = suffix_resident[0] + suffix_max_transient[0]
    peak_all_coded = best_peak
    for j in range(1, n + 1):
        prefix_dense += int(items[j - 1][1])
        peak = prefix_dense + suffix_resident[j] + suffix_max_transient[j]
        if peak < best_peak:
            best_peak = peak
            best_j = j
    return {
        "dense_names": sorted(name for name, _d, _r, _t in items[:best_j]),
        "predicted_peak_weight_bytes": int(best_peak),
        "predicted_peak_all_coded_bytes": int(peak_all_coded),
        "predicted_peak_all_dense_bytes": int(sum(int(c[1]) for c in items)),
        "n_candidates_evaluated": n + 1,
    }


def _parent_and_attr(model: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    obj: nn.Module = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def _module_name_of_param(param_name: str) -> Tuple[str, str]:
    """``a.b.weight`` -> (``a.b``, ``weight``)."""
    head, _, tail = param_name.rpartition(".")
    return head, tail


def _install_derived_buffers(
    model: nn.Module, art: Artifact, dev: torch.device,
) -> int:
    entries = art.index.get("derived_buffers") or {}
    if not entries:
        return 0
    fname = art.index.get("derived_buffers_file")
    if not fname:
        raise GLCArtifactError(
            "the index declares derived buffers but names no file for them"
        )
    from safetensors.torch import load_file

    blob = load_file(str(art.model_dir / fname), device="cpu")
    bufs = dict(model.named_buffers())
    n = 0
    for name, meta in sorted(entries.items()):
        if name not in blob:
            raise GLCArtifactError(f"derived buffer {name!r} missing from {fname}")
        t = blob[name]
        if list(t.shape) != [int(x) for x in meta["shape"]]:
            raise GLCArtifactError(
                f"derived buffer {name!r} has shape {list(t.shape)}, index says "
                f"{meta['shape']}"
            )
        target = bufs.get(name)
        if target is None:
            raise GLCArtifactError(
                f"derived buffer {name!r} has no home in the instantiated model"
            )
        if tuple(target.shape) != tuple(t.shape):
            raise GLCArtifactError(
                f"derived buffer {name!r}: model expects {tuple(target.shape)}, "
                f"artifact carries {tuple(t.shape)}"
            )
        head, _, leaf = name.rpartition(".")
        parent = model.get_submodule(head) if head else model
        parent._buffers[leaf] = t.to(dev)
        n += 1
    return n


def _build_skeleton(model_dir: Path, *, trust_remote_code: bool,
                    attn_implementation: Optional[str]) -> nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(
        str(model_dir), trust_remote_code=trust_remote_code, local_files_only=True,
    )
    kwargs: Dict[str, Any] = {}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    with torch.device("meta"):
        try:
            model = AutoModelForCausalLM.from_config(
                cfg, trust_remote_code=trust_remote_code, dtype=torch.bfloat16,
                **kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_config(
                cfg, trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16, **kwargs,
            )
    return model


def load_model(
    path: os.PathLike | str,
    *,
    device: str = "cpu",
    backend: Optional[str] = None,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    require_certificate: bool = True,
    verbose: bool = False,
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """Return ``(model, tokenizer, receipt)`` for a GLC-RELEASE artifact."""
    t_start = time.perf_counter()
    art = open_artifact(path, require_certificate=require_certificate)
    if require_certificate:
        assert_certificate_usable(art)

    dev = torch.device(device)
    chosen = resolve_backend(backend, dev)
    model = _build_skeleton(
        art.model_dir,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )

    peak_stats_available = False
    if dev.type == "cuda":
        # torch.cuda.reset_peak_memory_stats raises "Invalid device argument"
        # if the CUDA context has not been created yet.  Inside the builder
        # this never fired, because transformers had already loaded a dense
        # reference model and initialised the context first; it fired the
        # moment the artifact was loaded from a cold process on a client
        # machine, which is the case that matters.  Initialise explicitly, and
        # treat the counter as telemetry that may be unavailable -- never let a
        # memory statistic decide whether a model loads.
        try:
            torch.cuda.init()
            torch.cuda.set_device(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            peak_stats_available = True
        except Exception:
            peak_stats_available = False

    tie = bool(art.manifest["structure"].get("tie_word_embeddings", False))
    embed_param = art.manifest["structure"].get("input_embedding_param")
    head_param = art.manifest["structure"].get("output_head_param")
    tied_names = set(art.manifest["structure"].get("tied_tensor_names", []))

    # --- which containers get decoded once, here, instead of in every forward
    # `materialize` decodes all of them by definition.  `triton` decodes none of
    # them ever -- its GEMV reads the compressed bytes -- so its transient is
    # the output tile, not a weight.  Only the pure-torch `resident` path has a
    # choice to make, and it makes it by arithmetic; see plan_dense_carry.
    carry_plan: Dict[str, Any] = {
        "backend": chosen,
        "policy": {
            "materialize": "all containers decoded once at load",
            "triton": "no container is ever decoded whole",
            "resident": "transient-minimising subset, chosen from the index",
        }.get(chosen, chosen),
        "dense_names": [],
    }
    if chosen == "materialize":
        carry_dense = {
            n for n, e in art.tensors.items() if e["kind"] == "fwp1"
        }
        carry_plan["dense_names"] = sorted(carry_dense)
    elif chosen == "resident":
        candidates: List[Tuple[str, int, int, int]] = []
        for name, entry in art.tensors.items():
            if entry["kind"] != "fwp1" or entry.get("in_served_graph") is False:
                continue
            dense_b = int(entry["original_bytes"])
            res_b = int(entry["resident_bytes"])
            # An embedding is read by row lookup and never decoded whole -- so
            # it contributes no transient UNLESS a tied head shares its table,
            # in which case the head decodes every row of it on every forward.
            if name == embed_param:
                transient = dense_b if (tie and head_param) else 0
            else:
                transient = dense_b
            candidates.append((name, dense_b, res_b, transient))
        plan = plan_dense_carry(candidates)
        carry_dense = set(plan["dense_names"])
        carry_plan.update(plan)
    else:
        carry_dense = set()

    reader = _ShardReader(art.model_dir)
    n_linear = 0
    n_embed = 0
    n_raw = 0
    n_skipped = 0
    installed_params: set[str] = set()
    try:
        # --- raw tensors, shape-asserted against the model AND the header ----
        model_params = dict(model.named_parameters())
        model_bufs = dict(model.named_buffers())
        raw_state: Dict[str, torch.Tensor] = {}
        for name, entry in art.tensors.items():
            if entry["kind"] != "raw":
                continue
            target = model_params.get(name, model_bufs.get(name))
            if target is None:
                if entry.get("in_served_graph") is False:
                    # The artifact's scope is the checkpoint; the served graph
                    # is a subset of it.  A vision tower that
                    # AutoModelForCausalLM does not instantiate is carried so
                    # `expand` stays complete, and skipped here.  Counted, not
                    # ignored -- the receipt reports how many.
                    n_skipped += 1
                    continue
                raise GLCArtifactError(
                    f"artifact carries {name!r} but the model built from "
                    "config.json has no such parameter or buffer, and the index "
                    "says it belongs to the served graph. The artifact and its "
                    "config disagree; refusing to load."
                )
            hdr = reader.header_shape(entry["shard"], name)
            if tuple(int(x) for x in hdr) != tuple(int(x) for x in target.shape):
                raise GLCArtifactError(
                    f"{name}: safetensors header shape {tuple(hdr)} != the "
                    f"model's parameter shape {tuple(target.shape)}. This is "
                    "the config-shaped-tensor failure; refusing to load."
                )
            raw_state[name] = reader.get(entry["shard"], name, device="cpu")
            n_raw += 1
        missing, unexpected = model.load_state_dict(
            raw_state, strict=False, assign=True,
        )
        if unexpected:
            raise GLCArtifactError(f"unexpected raw tensors: {sorted(unexpected)}")
        installed_params |= set(raw_state)

        # --- coded embedding + tied head -------------------------------------
        embed_module: Optional[GLCEmbedding] = None
        embed_weight: Optional[nn.Parameter] = None
        if embed_param is not None and art.tensors.get(embed_param, {}).get(
            "kind"
        ) == "fwp1":
            entry = art.entry(embed_param)
            c = _container_from_entry(reader, embed_param, entry, "cpu")
            mod_name, _ = _module_name_of_param(embed_param)
            parent, attr = _parent_and_attr(model, mod_name)
            old = getattr(parent, attr)
            padding_idx = getattr(old, "padding_idx", None)
            if tuple(int(x) for x in c.shape) != (
                int(old.num_embeddings), int(old.embedding_dim),
            ):
                raise GLCArtifactError(
                    f"{embed_param}: container shape {c.shape} != the model's "
                    f"embedding {(old.num_embeddings, old.embedding_dim)}"
                )
            if embed_param in carry_dense:
                fresh = nn.Embedding(
                    int(c.shape[0]), int(c.shape[1]), padding_idx=padding_idx,
                    device="meta", dtype=torch.bfloat16,
                )
                embed_weight = nn.Parameter(
                    decode_fwp1(c).to(dev), requires_grad=False,
                )
                fresh.weight = embed_weight
                setattr(parent, attr, fresh)
            else:
                embed_module = GLCEmbedding(c.to(dev), padding_idx, chosen)
                setattr(parent, attr, embed_module)
            n_embed += 1
            installed_params.add(embed_param)

        # --- coded linears ----------------------------------------------------
        for name, entry in art.tensors.items():
            if entry["kind"] != "fwp1" or name == embed_param:
                continue
            if entry.get("in_served_graph") is False:
                n_skipped += 1
                continue
            mod_name, leaf = _module_name_of_param(name)
            if leaf != "weight":
                raise GLCArtifactError(
                    f"coded tensor {name!r} is not a module weight; the loader "
                    "installs containers into modules and cannot place this"
                )
            parent, attr = _parent_and_attr(model, mod_name)
            old = getattr(parent, attr)
            if not isinstance(old, nn.Linear):
                raise GLCArtifactError(
                    f"{mod_name} is {type(old).__name__}, not nn.Linear; the "
                    "artifact declares a coded linear weight there"
                )
            c = _container_from_entry(reader, name, entry, "cpu")
            if tuple(int(x) for x in c.shape) != (
                int(old.out_features), int(old.in_features),
            ):
                raise GLCArtifactError(
                    f"{name}: container shape {c.shape} != the model's linear "
                    f"{(old.out_features, old.in_features)}"
                )
            bias_name = f"{mod_name}.bias"
            bias = raw_state.get(bias_name)
            if old.bias is not None and bias is None:
                raise GLCArtifactError(
                    f"{mod_name} has a bias but the artifact carries none"
                )
            if name in carry_dense:
                # Stay an ``nn.Linear``.  This weight has already paid for
                # itself on the wire and on disk, and either the client asked
                # for maximum compatibility or the carry plan found that
                # decoding it inside every forward would cost more transient
                # memory than keeping it compressed saves.
                old.weight = nn.Parameter(
                    decode_fwp1(c).to(dev), requires_grad=False,
                )
                if bias is not None:
                    old.bias = nn.Parameter(bias.to(dev), requires_grad=False)
            else:
                setattr(
                    parent, attr,
                    GLCLinear(
                        c.to(dev), None if bias is None else bias.to(dev), chosen,
                    ),
                )
            n_linear += 1
            installed_params.add(name)

        # --- tied output head -------------------------------------------------
        if tie and head_param is not None:
            mod_name, _ = _module_name_of_param(head_param)
            parent, attr = _parent_and_attr(model, mod_name)
            old = getattr(parent, attr)
            bias = raw_state.get(f"{mod_name}.bias")
            if embed_weight is not None:
                # A genuine tie: one Parameter object, referenced twice, so the
                # table is stored once in memory exactly as the source model
                # stored it once on disk.
                old.weight = embed_weight
                if bias is not None:
                    old.bias = nn.Parameter(bias.to(dev), requires_grad=False)
                installed_params.add(head_param)
            elif embed_module is not None:
                setattr(
                    parent, attr,
                    GLCTiedLMHead(
                        embed_module,
                        None if bias is None else bias.to(dev),
                        chosen,
                    ),
                )
                installed_params.add(head_param)
    finally:
        reader.close()

    # --- config-derived buffers ---------------------------------------------
    # Rotary inv_freq and friends are computed in ``__init__`` and registered
    # non-persistent, so they are in no checkpoint and a meta skeleton leaves
    # them on meta.  The builder computed them and put them in the artifact,
    # which is why this loader needs no private transformers API to recover
    # them and cannot be broken by a transformers upgrade on the client's box.
    n_derived = _install_derived_buffers(model, art, dev)

    # --- move the rest and prove nothing is left on meta ---------------------
    for mod in model.modules():
        for pname, p in list(mod._parameters.items()):
            if p is not None and p.device.type != "meta" and p.device != dev:
                mod._parameters[pname] = nn.Parameter(
                    p.data.to(dev), requires_grad=False,
                )
        for bname, b in list(mod._buffers.items()):
            if b is not None and b.device.type != "meta" and b.device != dev:
                mod._buffers[bname] = b.to(dev)

    stragglers = [
        n for n, t in list(model.named_parameters()) + list(model.named_buffers())
        if t.device.type == "meta"
    ]
    if stragglers:
        # If the artifact does not carry a tensor the architecture needs, the
        # model is wrong and the client must be told, not handed a plausible
        # object full of uninitialised memory.
        raise GLCArtifactError(
            "load left "
            f"{len(stragglers)} tensor(s) on the meta device, i.e. the artifact "
            "does not carry every weight this architecture requires: "
            f"{stragglers[:12]}{' ...' if len(stragglers) > 12 else ''}"
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = load_tokenizer(art, trust_remote_code=trust_remote_code)
    peak = None
    if dev.type == "cuda" and peak_stats_available:
        try:
            peak = int(torch.cuda.max_memory_allocated(dev))
        except Exception:
            peak = None
    acc = art.manifest["accounting"]
    receipt = LoadReceipt(
        artifact=str(art.root),
        backend=chosen,
        device=str(dev),
        source_model=art.manifest["provenance"]["source_model"],
        builder_git_sha=art.manifest["provenance"]["builder_git_sha"],
        container=art.manifest["container"]["name"],
        n_coded_linears=n_linear,
        n_coded_embeddings=n_embed,
        n_raw_tensors=n_raw,
        n_tensors_not_in_served_graph=n_skipped,
        weight_bytes_dense=int(acc["weight_bytes_dense"]),
        weight_bytes_resident=int(acc["weight_bytes_resident"]),
        weight_ratio=float(acc["weight_ratio"]),
        load_seconds=time.perf_counter() - t_start,
        torch_peak_allocated_bytes=peak,
        meta_tensors_remaining=0,
        carry_plan=carry_plan,
    ).to_dict()
    if verbose:
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return model, tokenizer, receipt


def load_tokenizer(art: Artifact, *, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(art.model_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )


# ---------------------------------------------------------------------------
# certificate policy -- fail closed
# ---------------------------------------------------------------------------
class CertificateError(GLCArtifactError):
    """The certificate is absent, incomplete, or does not claim a pass."""


def assert_certificate_usable(art: Artifact) -> Dict[str, Any]:
    """Refuse to load an artifact whose certificate does not stand up.

    The rules exist because each one has been violated in this project's own
    history: a gate whose verdict was missing and read as a pass; a certificate
    emitted after an internal gate error was swallowed; an all-gates-pass
    certificate on an artifact with ``param_ratio: 1.0`` that compressed
    nothing.  There is no ``.get(key, True)`` anywhere below.
    """
    cert = art.certificate
    for key in ("format", "gates", "frame", "verdict"):
        if key not in cert:
            raise CertificateError(f"certificate has no {key!r} section")
    gates = cert["gates"]
    if not isinstance(gates, dict) or not gates:
        raise CertificateError("certificate declares no gates")
    for name, g in sorted(gates.items()):
        if not isinstance(g, dict):
            raise CertificateError(f"gate {name!r} is not an object")
        if "passed" not in g:
            raise CertificateError(
                f"gate {name!r} carries no 'passed' verdict. A missing verdict "
                "is not a pass."
            )
        if not isinstance(g["passed"], bool):
            raise CertificateError(
                f"gate {name!r} has a non-boolean verdict {g['passed']!r}"
            )
        if g["passed"] is False:
            raise CertificateError(
                f"gate {name!r} FAILED at build time: {g.get('detail')}"
            )
        if g.get("measured") is not True:
            raise CertificateError(
                f"gate {name!r} claims a pass but is not marked measured"
            )
    if cert["verdict"] != "PASS":
        raise CertificateError(f"certificate verdict is {cert['verdict']!r}")
    frame = cert["frame"]
    for key in ("metric_version", "builder_git_sha", "measured_at_utc"):
        if frame.get(key) in (None, ""):
            raise CertificateError(
                f"certificate frame field {key!r} is null; a receipt that does "
                "not say when, where and under which metric version it was "
                "measured cannot be checked and is not evidence"
            )
    return cert


__all__ = [
    "CertificateError",
    "LoadReceipt",
    "assert_certificate_usable",
    "load_model",
    "load_tokenizer",
    "plan_dense_carry",
]
