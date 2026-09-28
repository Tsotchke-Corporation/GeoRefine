"""Export a TBE bundle to ordinary Hugging Face safetensors, without the parent.

This is a portable compatibility path. It restores BF16 storage and does not
provide compressed inference or the GeoRefine native engine's speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np

from .sources import LocalFile, SourceError, download, parse_location, read_safetensors_header, sha256_source
from .tbe import TBEParams, iter_decode
from .verify import _dtype_tag


def _digest_file(path: Path) -> str:
    src = LocalFile(path)
    try:
        return sha256_source(src)
    finally:
        src.close()


def _bytes_for(entry: dict) -> int:
    return int(entry["original_bytes"])


def _header(entries: list[dict]) -> bytes:
    hdr, offset = {}, 0
    for entry in entries:
        name = entry["name"]
        size = _bytes_for(entry)
        hdr[name] = {"dtype": _dtype_tag(entry["dtype"]), "shape": entry["shape"],
                     "data_offsets": [offset, offset + size]}
        offset += size
    data = json.dumps(hdr, separators=(",", ":"), ensure_ascii=False).encode()
    return data + b" " * (-len(data) % 8)


def _write_tensor(out, entry: dict, src: LocalFile, header: dict) -> None:
    name = entry["name"]
    digest = hashlib.blake2b()

    def write(data: bytes) -> None:
        out.write(data)
        digest.update(data)

    if entry["kind"] == "raw":
        stored = header.get(name)
        if stored is None or stored.dtype != _dtype_tag(entry["dtype"]) or \
                list(stored.shape) != entry["shape"] or stored.nbytes != _bytes_for(entry):
            raise SourceError(f"raw tensor metadata differs: {name}")
        for off in range(0, stored.nbytes, 1 << 26):
            write(src.read(stored.offset + off, min(1 << 26, stored.nbytes - off)))
    elif entry["kind"] == "tbe":
        if _dtype_tag(entry["dtype"]) != "BF16":
            raise SourceError(f"TBE tensor is not BF16: {name}")
        n, k = (int(x) for x in (entry.get("coded_shape") or entry["shape"]))
        if n * k * 2 != _bytes_for(entry):
            raise SourceError(f"TBE coded shape disagrees with byte count: {name}")
        params = TBEParams(n=n, k=k, layout=entry["layout"], mode=int(entry["mode"]),
                           base=int(entry["base"]), superblock=int(entry.get("superblock", 32)))
        parts = {}
        for part in ("planes", "smb", "esc", "sbbase"):
            parts[part] = header.get(f"{name}.{part}")
            if parts[part] is None:
                raise SourceError(f"TBE part absent: {name}.{part}")
        if parts["planes"].nbytes != params.tiles * 24 or \
                parts["smb"].nbytes != params.tiles * 64 or \
                parts["esc"].nbytes != int(entry["escapes"]):
            raise SourceError(f"TBE part sizes disagree: {name}")
        sb = parts["sbbase"]
        if sb.nbytes != params.n_superblocks * 4:
            raise SourceError(f"TBE superblock index size disagrees: {name}")
        sbbase = np.frombuffer(src.read(sb.offset, sb.nbytes), dtype="<i4")

        def reader(part):
            a = parts[part]
            return lambda off, size: src.read(a.offset + off, size)

        for _, words in iter_decode(params, reader("planes"), reader("smb"),
                                    reader("esc"), sbbase, parts["esc"].nbytes):
            write(words.astype("<u2", copy=False).tobytes())
    else:
        raise SourceError(f"unknown tensor kind: {name}")
    if entry.get("blake2b_source") and digest.hexdigest() != entry["blake2b_source"]:
        raise SourceError(f"decoded tensor digest differs: {name}")


def export(bundle: str, output: Path, expected_manifest_sha256: str | None = None,
           work_dir: Path | None = None) -> dict:
    loc = parse_location(bundle)
    raw = loc.read_small("serve_manifest.json")
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        raise SourceError("manifest SHA-256 differs from the expected value")
    manifest = json.loads(raw)
    if manifest.get("format") != "georefine.tbe.serve.v1":
        raise SourceError("unsupported bundle format")
    shards = manifest["shards"]
    entries = manifest["tensors"]
    by_shard: dict[int, list[dict]] = {i: [] for i in range(len(shards))}
    if len({e["name"] for e in entries}) != len(entries):
        raise SourceError("duplicate tensor name")
    for e in entries:
        by_shard[int(e["shard"])].append(e)
    output.mkdir(parents=True, exist_ok=True)
    scratch = work_dir or (output.parent / ".scratch" / f"{output.name}-georefine-export")
    scratch.mkdir(parents=True, exist_ok=True)
    receipt_path = scratch / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"shards": {}}
    if receipt.get("manifest_sha256") not in (None, manifest_sha):
        raise SourceError("output was started from another bundle manifest")
    receipt["manifest_sha256"] = manifest_sha
    weight_map = {}
    total_size = 0
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        target = output / name
        expected = receipt["shards"].get(name)
        if expected and target.is_file() and _digest_file(target) == expected:
            pass
        else:
            source_file = scratch / f"codec-{i:05d}.safetensors"
            if not source_file.is_file() or _digest_file(source_file) != shard["sha256"]:
                if loc.kind == "local":
                    source_file = Path(loc.base) / shard["file"]
                else:
                    download(loc.url(shard["file"]), source_file, size=shard["bytes"],
                             sha256=shard["sha256"])
            if _digest_file(source_file) != shard["sha256"]:
                raise SourceError(f"codec shard hash differs: {shard['file']}")
            src = LocalFile(source_file)
            try:
                codec_header = read_safetensors_header(src)[0]
                ordered = sorted(by_shard[i], key=lambda e: e["name"])
                header = _header(ordered)
                part = scratch / f"{name}.partial"
                with part.open("wb") as out:
                    out.write(struct.pack("<Q", len(header)))
                    out.write(header)
                    for entry in ordered:
                        _write_tensor(out, entry, src, codec_header)
                os.replace(part, target)
                receipt["shards"][name] = _digest_file(target)
                receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            finally:
                src.close()
                if loc.kind != "local":
                    source_file.unlink(missing_ok=True)
        for entry in by_shard[i]:
            weight_map[entry["name"]] = name
            total_size += _bytes_for(entry)
    for sidecar in manifest["sidecars"]:
        rel = sidecar["file"]
        target = output / rel
        data = loc.read_small(rel)
        if len(data) != sidecar["bytes"] or hashlib.sha256(data).hexdigest() != sidecar["sha256"]:
            raise SourceError(f"sidecar hash differs: {rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n")
    return {"manifest_sha256": manifest_sha, "tensors": len(weight_map),
            "shards": len(shards), "bf16_bytes": total_size, "output": str(output)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="local dir, HF repo[@revision], or gs://")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, help="durable scratch and resume state")
    parser.add_argument("--expect-manifest-sha256")
    args = parser.parse_args(argv)
    result = export(args.bundle, args.output, args.expect_manifest_sha256, args.work_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
