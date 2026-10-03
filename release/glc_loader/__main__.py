"""Client-side command line: ``glc-loader <command>`` / ``python -m glc_loader``.

Runs from inside an artifact directory, or against one by path, with nothing
installed but this package and its three dependencies (torch, safetensors,
transformers).  Nothing here imports the repository that built the artifact.

    glc-loader info   <artifact-dir>   what it is, at which scopes, and what
                                       is NOT certified
    glc-loader verify <artifact-dir>   integrity, with honest exit codes
    glc-loader load   <artifact-dir>   smoke-load and generate a few tokens

The older ``--artifact``-flag spelling still works everywhere, so the
instructions printed inside already-released artifacts keep running.

EXIT CODES (verify).  0 verified; 1 an integrity check FAILED; 2 malformed,
unreadable, or not a GeoRefine container; 3 verified but the artifact
EXPANDS; 4 the format is recognised but this directory cannot be verified
from what it ships.  4 exists because the alternative is a misleading 2: a
v1 TBE transcode directory is a correct artifact with no self-contained
verifier, and ``experiments.georefine.lossless_audit`` on one exits 2 ("no
blobs/") purely because it is the wrong auditor for that codec.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover - direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "glc_loader"

from ._inspect import (
    EXIT_MALFORMED,
    EXIT_OK,
    EXIT_UNVERIFIABLE,
    KIND_GLC_RELEASE_V1,
    KIND_TBE_V1_TRANSCODE,
    KIND_TBE_V2,
    InspectError,
    certificate_summary,
    detect,
    not_certified,
    scope_ratios,
    source_identity,
    verify_artifact,
)
from .artifact import GLCArtifactError, expand, open_artifact


def _default_artifact() -> str:
    """The artifact this package was vendored into, if it was vendored."""
    here = Path(__file__).resolve().parent.parent
    return str(here)


def _target(args: argparse.Namespace) -> str:
    """Positional path wins; ``--artifact`` is the compatible spelling."""
    return getattr(args, "artifact_pos", None) or args.artifact


def _emit(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _fail(reason: str, detail: str, code: int = EXIT_MALFORMED) -> int:
    print(
        json.dumps(
            {"status": "FAIL", "reason": reason, "detail": detail,
             "exit_code": code},
            indent=2, sort_keys=True,
        ),
        file=sys.stderr,
    )
    return code


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
def cmd_info(args: argparse.Namespace) -> int:
    det = detect(_target(args))
    out: Dict[str, Any] = {
        "artifact": str(det.root),
        "format": det.kind,
        "source": source_identity(det),
        "ratios": scope_ratios(det),
        "ratio_note": (
            "Three scopes, never conflated. stored = bytes on disk. "
            "served_resident = bytes the device holds for weights. "
            "whole_process = measured peak process VRAM, dense vs coded, and "
            "it is the only one that includes activations, the KV cache and "
            "allocator slack. A ratio reported at one scope says nothing "
            "about the other two."
        ),
        "certificate": certificate_summary(det),
        "not_certified": not_certified(det),
    }
    if det.kind == KIND_GLC_RELEASE_V1:
        art = open_artifact(str(det.root), require_certificate=False)
        out["summary"] = art.summary()
        out["container"] = det.manifest.get("container")
        out["effective_config"] = det.manifest.get("effective_config")
    elif det.kind in (KIND_TBE_V2, KIND_TBE_V1_TRANSCODE):
        out["container"] = det.manifest.get("container")
        out["summary"] = det.manifest.get("summary")
        out["escape_band_pct"] = det.manifest.get("escape_band_pct")
        out["escape_over_band"] = [
            o.get("name") for o in (det.manifest.get("escape_over_band") or [])
        ]
        if det.compression_info:
            out["compression_info"] = {
                k: v for k, v in det.compression_info.items()
                if k != "sidecars"
            }
    if not det.understood:
        out["status"] = "REFUSED"
        out["reason"] = det.reason
        out["detail"] = det.detail
        _emit(out)
        return EXIT_UNVERIFIABLE if det.kind == KIND_TBE_V1_TRANSCODE \
            else EXIT_MALFORMED
    _emit(out)
    return EXIT_OK


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def cmd_verify(args: argparse.Namespace) -> int:
    det = detect(_target(args))
    res = verify_artifact(
        det, deep=args.deep, progress=args.progress, limit=args.limit,
    )
    res["certificate"] = certificate_summary(det)
    res["not_certified"] = not_certified(det)
    stream = sys.stdout if res["exit_code"] == EXIT_OK else sys.stderr
    print(json.dumps(res, indent=2, sort_keys=True, default=str), file=stream)
    return int(res["exit_code"])


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def _generate(model, tok, prompt: str, *, device: str, max_new_tokens: int,
              chat: bool) -> Dict[str, Any]:
    import torch

    text = prompt
    if chat and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    enc = tok(text, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    dt = time.perf_counter() - t0
    n_new = int(out.shape[-1] - enc["input_ids"].shape[-1])
    return {
        "prompt": prompt,
        "completion": tok.decode(
            out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True,
        ),
        "new_tokens": n_new,
        "seconds": dt,
        "tokens_per_second": (n_new / dt) if dt > 0 else None,
        "throughput_note": (
            "a single greedy run on whatever device this is; NOT a benchmark. "
            "Measured decode throughput for this format is 0.94x dense on a "
            "Blackwell RTX PRO 6000 and 0.57-0.62x on an A100 (single runs, "
            "+/-0.05 spread). It is a VRAM lever, not a speed lever."
        ),
    }


def cmd_load(args: argparse.Namespace) -> int:
    det = detect(_target(args))

    if det.kind == KIND_GLC_RELEASE_V1:
        from .loader import load_model

        model, tok, receipt = load_model(
            str(det.root),
            device=args.device,
            backend=args.backend,
            trust_remote_code=args.trust_remote_code,
            require_certificate=not args.no_certificate,
        )
        _emit({
            "status": "LOADED",
            "format": det.kind,
            "serving_receipt": receipt,
            "generation": _generate(
                model, tok, args.prompt, device=args.device,
                max_new_tokens=args.max_new_tokens, chat=args.chat,
            ),
            "certificate": certificate_summary(det),
        })
        return EXIT_OK

    if det.kind == KIND_TBE_V2:
        try:
            from .tbe_artifact import TBEArtifactError, load_standalone
        except ImportError as exc:
            return _fail(
                "tbe_artifact_module_unavailable",
                f"this build of glc_loader cannot load {KIND_TBE_V2} "
                f"artifacts ({exc}). Upgrade glc-loader.",
            )
        from .tbe_device_map import single_device_map

        device_map = single_device_map(args.device)
        try:
            model, tok, receipt = load_standalone(
                str(det.root), device_map=device_map,
            )
        except TBEArtifactError as exc:
            return _fail(
                getattr(exc, "reason", "load_failed"),
                getattr(exc, "detail", str(exc)),
            )
        except RuntimeError as exc:
            # Report the typed reason the serving stack raised, verbatim. Do
            # not narrate a cause: the refusals on this path include a
            # missing device map, an unsupported compute capability, a
            # non-bf16 candidate AND a parameter the container never carried,
            # and guessing between them in the CLI is how an accurate error
            # becomes a misleading one.
            return _fail(
                getattr(exc, "reason", None) or f"load_refused_{type(exc).__name__}",
                f"{getattr(exc, 'detail', None) or exc} "
                "[glc-loader did not interpret this; it is the serving "
                "stack's own typed refusal. The TBE serving path needs CUDA "
                "of a supported compute capability, or Apple Silicon with the "
                "[metal] extra; it never degrades to a slower or larger path.]",
            )
        if tok is None:
            return _fail(
                "no_tokenizer",
                "the artifact carries no tokenizer, so text cannot go in or "
                "come out. Re-issue it with its tokenizer sidecars.",
            )
        _emit({
            "status": "LOADED",
            "format": det.kind,
            "serving_receipt": receipt,
            "generation": _generate(
                model, tok, args.prompt, device=args.device,
                max_new_tokens=args.max_new_tokens, chat=args.chat,
            ),
            "certificate": certificate_summary(det),
        })
        return EXIT_OK

    if det.kind == KIND_TBE_V1_TRANSCODE:
        return _fail(det.reason, det.detail, EXIT_UNVERIFIABLE)
    return _fail(det.reason or "not_a_georefine_artifact", det.detail)


# ---------------------------------------------------------------------------
# expand + generate: GLC-RELEASE/1 only, unchanged
# ---------------------------------------------------------------------------
def cmd_expand(args: argparse.Namespace) -> int:
    res = expand(_target(args), args.out, progress=args.progress)
    _emit(res)
    return EXIT_OK


def cmd_generate(args: argparse.Namespace) -> int:
    import torch

    from .loader import load_model

    model, tok, receipt = load_model(
        _target(args),
        device=args.device,
        backend=args.backend,
        trust_remote_code=args.trust_remote_code,
        require_certificate=not args.no_certificate,
    )
    prompt = args.prompt
    if args.chat and getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    enc = tok(prompt, return_tensors="pt").to(args.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p if args.temperature > 0 else None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    dt = time.perf_counter() - t0
    n_new = int(out.shape[-1] - enc["input_ids"].shape[-1])
    text = tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
    _emit(
        {
            "receipt": receipt,
            "prompt": args.prompt,
            "completion": text,
            "new_tokens": n_new,
            "seconds": dt,
            "tokens_per_second": (n_new / dt) if dt > 0 else None,
        }
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="glc-loader",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="store_true",
                   help="print the package version and exit")
    sub = p.add_subparsers(dest="command")

    def common(sp):
        sp.add_argument(
            "artifact_pos", nargs="?", default=None, metavar="ARTIFACT-DIR",
            help="the artifact directory (default: the directory this "
                 "package was vendored into)",
        )
        sp.add_argument("--artifact", default=_default_artifact(),
                        help=argparse.SUPPRESS)
        return sp

    i = common(sub.add_parser(
        "info",
        help="format, source, licence, ratios at three scopes, and what is "
             "NOT certified",
    ))
    i.set_defaults(func=cmd_info)

    v = common(sub.add_parser(
        "verify", help="check the artifact against its own digests",
    ))
    v.add_argument(
        "--deep", action="store_true",
        help="decode every container and check it against the source digest "
             "recorded before encoding; proves bit-exactness locally",
    )
    v.add_argument("--progress", action="store_true")
    v.add_argument(
        "--limit", type=int, default=None,
        help="with --deep, check only the first N coded tensors (a smoke "
             "check, not a verification -- the exit code says so)",
    )
    v.set_defaults(func=cmd_verify)

    ld = common(sub.add_parser(
        "load", help="smoke-load the artifact and generate a few tokens",
    ))
    ld.add_argument("--device", default="cpu")
    ld.add_argument("--prompt", default="The capital of France is")
    ld.add_argument("--max-new-tokens", type=int, default=16)
    ld.add_argument("--chat", action="store_true")
    ld.add_argument("--backend", default=None,
                    choices=[None, "materialize", "resident", "triton"])
    ld.add_argument("--trust-remote-code", action="store_true")
    ld.add_argument(
        "--no-certificate", action="store_true",
        help="load even if the certificate is absent or does not claim a pass",
    )
    ld.set_defaults(func=cmd_load)

    e = common(sub.add_parser(
        "expand",
        help="write a plain HF checkpoint for any other engine "
             "(GLC-RELEASE/1 only)",
    ))
    e.add_argument("--out", required=True)
    e.add_argument("--progress", action="store_true")
    e.set_defaults(func=cmd_expand)

    g = common(sub.add_parser(
        "generate", help="load and generate text (GLC-RELEASE/1 only)",
    ))
    g.add_argument("--prompt", required=True)
    g.add_argument("--device", default="cpu")
    g.add_argument("--backend", default=None, choices=[None, "materialize",
                                                       "resident", "triton"])
    g.add_argument("--max-new-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--chat", action="store_true")
    g.add_argument("--trust-remote-code", action="store_true")
    g.add_argument(
        "--no-certificate", action="store_true",
        help="load even if the certificate is absent or does not claim a pass",
    )
    g.set_defaults(func=cmd_generate)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "version", False):
        from . import __version__

        print(__version__)
        return EXIT_OK
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_MALFORMED
    try:
        return int(args.func(args))
    except InspectError as exc:
        return _fail(exc.reason, exc.detail, exc.exit_code)
    except GLCArtifactError as exc:
        return _fail(type(exc).__name__, str(exc))
    except ModuleNotFoundError as exc:
        return _fail(
            "missing_dependency",
            f"{exc}. glc-loader needs torch, safetensors and transformers; "
            "the CUDA fast path additionally needs `pip install "
            "glc-loader[cuda]` and the Apple Silicon path `pip install "
            "glc-loader[metal]`.",
        )


if __name__ == "__main__":
    raise SystemExit(main())
