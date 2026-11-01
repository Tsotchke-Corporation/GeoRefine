"""Opt-in, tensor-at-a-time Context source -> v2.1 descriptor adapter.

This is an in-memory startup bridge for the existing TBE21 loader. It does not
write a second weight package or register serving dispatch by default.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import importlib
import json
import lzma
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
import zlib

import numpy as np

from glc_loader import codec_v21 as cv21


FROZEN_CONTEXT_MANIFEST_SHA256 = "d1a3d29c759f2056c20049c49b6b04479e151b74b52b688ab774b2d7253e4e8c"
# The published Qwen3.8-27B-GeoRefine-Context rev f8c6b089 package manifest
# (schema bitexact-predictive-context-package-v1).  Its baseline tensor manifest
# (baseline-manifest.json.xz, decompressed) is pinned to the SAME frozen sha
# above, so both pins bind on the predictive path.
FROZEN_PREDICTIVE_MANIFEST_SHA256 = "bbac23fdeba3020f8951d7a0a005016985c40add08979f37a85df2f22ccbacee"
_REF_MEMO_BYTES = 1_400_000_000   # two 178 MB references plus slack


class ContextTranscodeError(ValueError):
    """Invalid frozen source or non-exact startup transcode."""


def _startup_detail(**detail):
    from glc_serve import startup_progress
    if startup_progress.has_sink():
        startup_progress.report('transcode', **detail)


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ContextTranscodeError(f"duplicate manifest key {key!r}")
        obj[key] = value
    return obj


def _hex_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class ContextV21TranscodeArtifact:
    """The V21 artifact reader interface backed by immutable Context frames."""

    def __init__(self, path, *, build_root,
                 expected_manifest_sha256=None):
        self.path = str(Path(path).expanduser().resolve())
        self.root = Path(self.path)
        manifest_path = self.root / "manifest.json"
        try:
            manifest_mode = manifest_path.lstat().st_mode
        except OSError as exc:
            raise ContextTranscodeError("missing Context manifest") from exc
        if not stat.S_ISREG(manifest_mode):
            raise ContextTranscodeError("Context manifest is not a regular file")
        raw = manifest_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_manifest_sha256 is None:
            # Only the two FROZEN packages are servable without an explicit pin.
            if digest not in (FROZEN_CONTEXT_MANIFEST_SHA256, FROZEN_PREDICTIVE_MANIFEST_SHA256):
                raise ContextTranscodeError("Context manifest SHA-256 is not a frozen package")
        else:
            if not _hex_sha(expected_manifest_sha256):
                raise ContextTranscodeError("expected manifest SHA-256 is required")
            if digest != expected_manifest_sha256:
                raise ContextTranscodeError("Context manifest SHA-256 mismatch")
        try:
            manifest = json.loads(raw, object_pairs_hook=_unique_object)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ContextTranscodeError("invalid Context manifest") from exc
        schema = manifest.get("schema") if isinstance(manifest, dict) else None
        if schema not in ("bitexact-context-package-v1", "bitexact-predictive-context-package-v1"):
            raise ContextTranscodeError("unsupported Context package schema")
        if not isinstance(manifest, dict) or manifest.get("complete") is not True:
            raise ContextTranscodeError("unsupported or incomplete Context package")
        self.manifest = manifest
        self.manifest_sha256 = digest
        self.codec = cv21
        self.format = cv21.FORMAT
        self.config_path = None
        self.records = {}
        self._names = []
        self._cached_name = None
        self._cached_tensor = None
        self._cached_source_name = None
        self._cached_source_words = None
        self.transcode_receipts = []
        self.runtime_decoder_record = None
        self._selected = {}
        self._pair = None
        self._mix = None
        self._ref_memo = collections.OrderedDict()
        self._ref_memo_bytes = 0
        if schema == "bitexact-predictive-context-package-v1":
            # The published package: baseline frames plus predictive candidates.
            # The embedded baseline tensor manifest must hash to BOTH the outer
            # manifest's pin and the frozen baseline sha.
            base_pin = manifest.get("baseline_manifest_sha256")
            if not _hex_sha(base_pin):
                raise ContextTranscodeError("predictive package lacks the baseline manifest pin")
            try:
                base_raw = lzma.decompress(self._regular_bytes("baseline-manifest.json.xz"))
            except lzma.LZMAError as exc:
                raise ContextTranscodeError("baseline manifest XZ is corrupt") from exc
            base_digest = hashlib.sha256(base_raw).hexdigest()
            if base_digest != base_pin or base_digest != FROZEN_CONTEXT_MANIFEST_SHA256:
                raise ContextTranscodeError("baseline manifest SHA-256 mismatch")
            try:
                base = json.loads(base_raw, object_pairs_hook=_unique_object)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ContextTranscodeError("invalid baseline manifest") from exc
            if (not isinstance(base, dict) or base.get("schema") != "bitexact-context-package-v1"
                    or base.get("complete") is not True or not isinstance(base.get("tensors"), list)
                    or not base["tensors"]):
                raise ContextTranscodeError("unsupported or incomplete baseline manifest")
            tensor_rows = base["tensors"]
        else:
            if not isinstance(manifest.get("tensors"), list) or not manifest["tensors"]:
                raise ContextTranscodeError("unsupported or incomplete Context package")
            tensor_rows = manifest["tensors"]
        decoder_sources = {}
        inventory = {row["path"]: row for row in manifest.get("files", [])
                     if isinstance(row, dict) and isinstance(row.get("path"), str)}
        for relative in manifest.get("decoder_files", []):
            item = inventory.get(relative)
            if not isinstance(item, dict) or not _hex_sha(item.get("sha256")):
                raise ContextTranscodeError(f"decoder inventory mismatch: {relative}")
            raw_source = self._regular_bytes(relative)
            if len(raw_source) != item.get("bytes") or hashlib.sha256(raw_source).hexdigest() != item["sha256"]:
                raise ContextTranscodeError(f"decoder source identity mismatch: {relative}")
            decoder_sources[PurePosixPath(relative).name] = raw_source
        if not {"bitexact_context_codec.py", "bitexact_context_ans.cpp"}.issubset(decoder_sources):
            raise ContextTranscodeError("package lacks the exact reference decoder sources")
        if schema == "bitexact-predictive-context-package-v1":
            needed = {"bitexact_predictive_codec.py", "bitexact_predictive_ans.cpp",
                      "bitexact_predictive_mix.py"}
            if not needed.issubset(decoder_sources):
                raise ContextTranscodeError("package lacks the exact predictive decoder sources")
            self._pair = self._load_pair_codec(build_root, decoder_sources)
            self._mix = self._load_mix(decoder_sources)
        self._decoder = self._load_decoder(build_root, decoder_sources)
        for row in tensor_rows:
            if not isinstance(row, dict):
                raise ContextTranscodeError("invalid Context tensor row")
            name, shape = row.get("name"), row.get("shape")
            if (not isinstance(name, str) or not name or name in self.records
                    or row.get("dtype") != "BF16" or not isinstance(shape, list)
                    or not shape or any(type(n) is not int or n <= 0 for n in shape)
                    or math.prod(shape) * 2 != row.get("source_bytes")
                    or not _hex_sha(row.get("frame_sha256"))
                    or not _hex_sha(row.get("source_sha256"))):
                raise ContextTranscodeError(f"invalid Context tensor identity: {name!r}")
            relative = row.get("frame")
            if not isinstance(relative, str) or "\\" in relative:
                raise ContextTranscodeError(f"unsafe frame path for {name}")
            parsed = PurePosixPath(relative)
            if (parsed.is_absolute() or parsed.as_posix() != relative or
                    any(part in ("", ".", "..") for part in parsed.parts)):
                raise ContextTranscodeError(f"unsafe frame path for {name}")
            self.records[name] = row
            self._names.append(name)
        if schema == "bitexact-predictive-context-package-v1":
            selected = manifest.get("selected_candidates")
            if not isinstance(selected, dict) or not selected:
                raise ContextTranscodeError("predictive package lacks selected candidates")
            for name, cand in selected.items():
                self._selected[name] = self._validated_candidate(name, cand)
            self._validate_selected_graph()
        from glc_loader import codec_v21_native
        codec_v21_native.configure_build_root(build_root)

    def _validated_candidate(self, name, cand):
        row = self.records.get(name)
        if row is None or not isinstance(cand, dict):
            raise ContextTranscodeError(f"invalid predictive candidate: {name!r}")
        if (cand.get("name") != name or cand.get("source_sha256") != row["source_sha256"]
                or not _hex_sha(cand.get("frame_sha256"))
                or type(cand.get("frame_bytes")) is not int or cand["frame_bytes"] <= 0):
            raise ContextTranscodeError(f"predictive candidate identity mismatch: {name}")
        relative = cand.get("frame")
        parsed = PurePosixPath(relative) if isinstance(relative, str) and "\\" not in relative else None
        if (parsed is None or parsed.is_absolute() or parsed.as_posix() != relative
                or any(part in ("", ".", "..") for part in parsed.parts)):
            raise ContextTranscodeError(f"unsafe candidate frame path for {name}")
        mode = cand.get("mode")
        transform = cand.get("transform")
        if transform not in ("target transpose", None, "identity"):
            raise ContextTranscodeError(f"unsupported candidate transform for {name}")
        refs = cand.get("references")
        if (not isinstance(refs, list) or not refs
                or any(not isinstance(x, dict) or x.get("name") not in self.records
                       or x.get("sha256") != self.records[x["name"]]["source_sha256"]
                       for x in refs)):
            raise ContextTranscodeError(f"predictive candidate reference mismatch: {name}")
        if any(x["name"] == name for x in refs):
            raise ContextTranscodeError(f"predictive candidate reference cycle: {name}")
        if mode == "mixbf16":
            if len(refs) != 2:
                raise ContextTranscodeError(f"mixbf16 candidate needs two references: {name}")
            c = cand.get("coefficients")
            shape = c.get("shape") if isinstance(c, dict) else None
            if (not isinstance(c, dict) or not _hex_sha(c.get("sha256"))
                    or not _hex_sha(c.get("decoded_sha256", c.get("sha256")))
                    or not _hex_sha(cand.get("derived_reference_sha256"))
                    or type(c.get("bytes")) is not int
                    or not isinstance(shape, list) or len(shape) != 2 or shape[1] != 3
                    or any(type(n) is not int or n <= 0 for n in shape)
                    or c.get("encoding", "identity") not in ("zlib", "identity")):
                raise ContextTranscodeError(f"invalid mixed coefficients for {name}")
            cp = c.get("path")
            pp = PurePosixPath(cp) if isinstance(cp, str) and "\\" not in cp else None
            if (pp is None or pp.is_absolute() or pp.as_posix() != cp
                    or any(part in ("", ".", "..") for part in pp.parts)):
                raise ContextTranscodeError(f"unsafe coefficient path for {name}")
        return cand

    def _validate_selected_graph(self):
        """Reject cycles across the complete selected-candidate dependency graph."""
        state = {}
        for start in self._selected:
            if state.get(start) == 2:
                continue
            state[start] = 1
            stack = [(start, iter(self._selected[start]["references"]))]
            while stack:
                name, references = stack[-1]
                try:
                    ref_name = next(references)["name"]
                except StopIteration:
                    state[name] = 2
                    stack.pop()
                    continue
                if ref_name not in self._selected:
                    continue
                ref_state = state.get(ref_name, 0)
                if ref_state == 1:
                    raise ContextTranscodeError(
                        f"predictive candidate reference cycle: {name} -> {ref_name}")
                if ref_state == 0:
                    state[ref_name] = 1
                    stack.append((ref_name, iter(self._selected[ref_name]["references"])))

    def _load_pair_codec(self, build_root, sources):
        root = Path(build_root).expanduser().resolve()
        if (".scratch" not in root.parts or root.is_relative_to(Path("/tmp")) or
                root.is_relative_to(Path("/private/tmp")) or root.is_relative_to(self.root)):
            raise ContextTranscodeError("decoder build root must be a durable .scratch outside the package")
        cache = root / "predictive"
        cache.mkdir(parents=True, exist_ok=True)
        cpp = cache / "bitexact_predictive_ans.cpp"
        want = sources["bitexact_predictive_ans.cpp"]
        if cpp.is_symlink() or (cpp.exists() and not stat.S_ISREG(cpp.lstat().st_mode)):
            raise ContextTranscodeError("predictive decoder build copy is not regular")
        if not cpp.exists() or hashlib.sha256(cpp.read_bytes()).digest() != hashlib.sha256(want).digest():
            cpp.write_bytes(want)
        src = sources["bitexact_predictive_codec.py"]
        module_name = "_bctx_verified_pair_" + self.manifest_sha256[:16]
        import types
        module = types.ModuleType(module_name)
        module.__file__ = str(cache / "bitexact_predictive_codec.py")
        sys.modules[module_name] = module
        exec(compile(src, module.__file__, "exec"), module.__dict__)
        # keep compiler output in the verified build root, never in the package
        module.ROOT = cache
        module.CACHE_ROOT = cache
        module.BUILD = cache / "build"
        return module

    def _load_mix(self, sources):
        import types
        module_name = "_bctx_verified_mix_" + self.manifest_sha256[:16]
        module = types.ModuleType(module_name)
        module.__file__ = "<bctx verified bitexact_predictive_mix.py>"
        sys.modules[module_name] = module
        exec(compile(sources["bitexact_predictive_mix.py"], module.__file__, "exec"), module.__dict__)
        return module

    def _read_coefficients(self, cand):
        c = cand["coefficients"]
        encoded = self._regular_bytes(c["path"])
        if len(encoded) != c["bytes"] or hashlib.sha256(encoded).hexdigest() != c["sha256"]:
            raise ContextTranscodeError("mixed coefficient checksum mismatch")
        shape = c["shape"]
        expected_raw = math.prod(shape) * 2
        encoding = c.get("encoding", "identity")
        if encoding == "zlib":
            d = zlib.decompressobj()
            raw = d.decompress(encoded, expected_raw + 1)
            if len(raw) > expected_raw or d.unconsumed_tail:
                raise ContextTranscodeError("mixed coefficient decompression exceeds declared shape")
            raw += d.flush()
            if not d.eof or d.unused_data:
                raise ContextTranscodeError("mixed coefficient stream incomplete or trailing")
        else:
            raw = encoded
        if (len(raw) != c.get("decoded_bytes", c["bytes"]) or len(raw) != expected_raw
                or hashlib.sha256(raw).hexdigest() != c.get("decoded_sha256", c["sha256"])):
            raise ContextTranscodeError("decoded mixed coefficient checksum mismatch")
        return np.frombuffer(raw, dtype="<u2").reshape(shape).copy()

    def _ref_words(self, name):
        cached = self._ref_memo.get(name)
        if cached is not None:
            self._ref_memo.move_to_end(name)
            return cached
        words = self.read_source_words(name)
        self._ref_memo[name] = words
        self._ref_memo_bytes += words.nbytes
        while self._ref_memo_bytes > _REF_MEMO_BYTES and len(self._ref_memo) > 1:
            _, old = self._ref_memo.popitem(last=False)
            self._ref_memo_bytes -= old.nbytes

        return words

    def _load_decoder(self, build_root, sources):
        root = Path(build_root).expanduser().resolve()
        if (".scratch" not in root.parts or root.is_relative_to(Path("/tmp")) or
                root.is_relative_to(Path("/private/tmp")) or root.is_relative_to(self.root)):
            raise ContextTranscodeError("decoder build root must be a durable .scratch outside the package")
        directory = root / "decoder"
        directory.mkdir(parents=True, exist_ok=True)
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise ContextTranscodeError("decoder build directory is not real")
        for name in ("bitexact_context_codec.py", "bitexact_context_ans.cpp"):
            target = directory / name
            if target.is_symlink():
                raise ContextTranscodeError(f"decoder build copy is a symlink: {name}")
            if target.exists():
                if not stat.S_ISREG(target.lstat().st_mode):
                    raise ContextTranscodeError(f"decoder build copy is not regular: {name}")
                if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(sources[name]).digest():
                    raise ContextTranscodeError(f"decoder build copy differs: {name}")
            else:
                target.write_bytes(sources[name])
        compact_choice = os.environ.get("GEOREFINE_CONTEXT_RUNTIME_COMPACT")
        if compact_choice not in (None, "0", "1"):
            raise ContextTranscodeError(
                "GEOREFINE_CONTEXT_RUNTIME_COMPACT must be 0 or 1")
        if compact_choice == "1":
            if os.environ.get("GEOREFINE_CONTEXT_COMPACT_RANS") != "1":
                raise ContextTranscodeError(
                    "runtime compact decoder requires GEOREFINE_CONTEXT_COMPACT_RANS=1")
            try:
                public_decoder = importlib.import_module(
                    "georefine_context_runtime.bitexact_context_codec")
                public_python = Path(public_decoder.__file__).resolve()
                public_cpp = public_python.with_name("bitexact_context_ans.cpp")
                if not public_python.is_file() or not public_cpp.is_file():
                    raise ContextTranscodeError("installed public decoder sources are missing")
                native = public_decoder._lib()
                compact_decode = getattr(native, "bcx_decode_compact", None)
                if compact_decode is None:
                    raise ContextTranscodeError(
                        "installed public decoder lacks bcx_decode_compact")
            except ContextTranscodeError:
                raise
            except Exception as exc:
                raise ContextTranscodeError(
                    "cannot initialize installed public compact decoder") from exc
            self.runtime_decoder_record = {
                "mode": "public_compact_rans",
                "python_sha256": hashlib.sha256(public_python.read_bytes()).hexdigest(),
                "cpp_sha256": hashlib.sha256(public_cpp.read_bytes()).hexdigest(),
            }
            return public_decoder
        module_name = "_bctx_verified_decoder_" + self.manifest_sha256[:16]
        spec = importlib.util.spec_from_file_location(module_name, directory / "bitexact_context_codec.py")
        if spec is None or spec.loader is None:
            raise ContextTranscodeError("cannot load verified reference decoder")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def names(self):
        return list(self._names)

    def stored_record(self, name):
        return self.records[name]

    def is_alias(self, name):
        self.stored_record(name)
        return False

    def has_config(self):
        return False

    def release(self, name=None):
        if name is None or self._cached_name == name:
            self._cached_name = None
            self._cached_tensor = None
        if name is None or self._cached_source_name == name:
            self._cached_source_name = None
            self._cached_source_words = None
        if name is None:
            self._ref_memo.clear()
            self._ref_memo_bytes = 0

    def _regular_bytes(self, relative):
        parsed = PurePosixPath(relative)
        if (parsed.is_absolute() or parsed.as_posix() != relative or
                any(part in ("", ".", "..") for part in parsed.parts)):
            raise ContextTranscodeError(f"unsafe package path {relative!r}")
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root) or not stat.S_ISREG(path.lstat().st_mode):
            raise ContextTranscodeError(f"package path is not a regular file: {relative}")
        return path.read_bytes()

    def _frame_bytes(self, name):
        row = self.stored_record(name)
        raw = self._regular_bytes(row["frame"])
        if len(raw) != row["frame_bytes"] or hashlib.sha256(raw).hexdigest() != row["frame_sha256"]:
            raise ContextTranscodeError(f"{name}: frozen frame identity mismatch")
        return raw

    def _baseline_words(self, name):
        row = self.stored_record(name)
        decoded_source = np.ascontiguousarray(self._decoder.decode(self._frame_bytes(name)))
        if decoded_source.dtype != np.uint16 or decoded_source.nbytes != row["source_bytes"]:
            raise ContextTranscodeError(f"{name}: Context source word identity mismatch")
        source_sha = hashlib.sha256(memoryview(decoded_source).cast("B")).hexdigest()
        if source_sha != row["source_sha256"]:
            raise ContextTranscodeError(f"{name}: Context source word SHA mismatch")
        return decoded_source.reshape(row["shape"])

    def _candidate_words(self, name):
        cand = self._selected[name]
        row = self.stored_record(name)
        frame = self._regular_bytes(cand["frame"])
        if len(frame) != cand["frame_bytes"] or hashlib.sha256(frame).hexdigest() != cand["frame_sha256"]:
            raise ContextTranscodeError(f"{name}: candidate frame identity mismatch")
        refs = [self._ref_words(x["name"]) for x in cand["references"]]
        if cand.get("mode") == "mixbf16":
            coeff = self._read_coefficients(cand)
            derived = np.ascontiguousarray(self._mix.predict(refs[0], refs[1], coeff))
            if hashlib.sha256(derived.tobytes()).hexdigest() != cand["derived_reference_sha256"]:
                raise ContextTranscodeError(f"{name}: derived reference checksum mismatch")
            reference = derived
        else:
            reference = refs[0]
        z = self._pair.decode(frame, reference)
        arr = np.ascontiguousarray(z.T if cand.get("transform") == "target transpose" else z)
        if (arr.dtype != np.uint16 or arr.nbytes != row["source_bytes"]
                or arr.shape != tuple(row["shape"])):
            raise ContextTranscodeError(f"{name}: candidate source word identity mismatch")
        if hashlib.sha256(arr.tobytes()).hexdigest() != row["source_sha256"]:
            raise ContextTranscodeError(f"{name}: candidate source word SHA mismatch")
        return arr

    def read_source_words(self, name):
        """Stage 1 only: the frozen frame decoded to source BF16 words.

        Sha256-verified against the manifest's ``source_sha256`` -- the same
        check ``read()`` performs before it ever re-encodes to v2.1.  Cached
        per name (cleared by :meth:`release`), independent of the v2.1 encode
        cache ``read()`` keeps.  This is everything a caller that wants the
        exact source bytes (and nothing else -- no v2.1 re-encode, no
        round-trip) needs.  On the published PREDICTIVE package, the 86
        reference-conditioned candidates are resolved through the package's
        own verified predictive decoder (references, mixed coefficients and
        derived-reference checksums all pinned by the manifest).
        """
        if self._cached_source_name == name and self._cached_source_words is not None:
            return self._cached_source_words
        memo = self._ref_memo.get(name)
        if memo is not None:
            row = self.stored_record(name)
            if (memo.dtype != np.uint16 or memo.shape != tuple(row["shape"])
                    or memo.nbytes != row["source_bytes"]
                    or hashlib.sha256(memoryview(np.ascontiguousarray(memo)).cast("B")).hexdigest() != row["source_sha256"]):
                raise ContextTranscodeError(f"{name}: reference memo source identity mismatch")
            self._ref_memo.move_to_end(name)
            words = memo
        elif name in self._selected:
            words = self._candidate_words(name)
        else:
            words = self._baseline_words(name)
        self._cached_source_name, self._cached_source_words = name, words
        return words

    def read(self, name, *, check_crc=True):
        if self._cached_name == name and self._cached_tensor is not None:
            return self._cached_tensor
        # Retire tensor-local buffers while retaining bounded, verified shared
        # references. Explicit release() still clears the complete memo.
        for previous in (self._cached_name, self._cached_source_name):
            if previous is not None:
                self.release(previous)
        row = self.stored_record(name)
        started = time.monotonic()
        _startup_detail(tensor=name, step='context_decode')
        words = self.read_source_words(name)
        source_seconds = time.monotonic() - started
        _startup_detail(tensor=name, step='v21_encode', context_decode_s=source_seconds)
        encoded = cv21.encode_tensor(name, words, dtype="BF16")
        if encoded.sha256 != row["source_sha256"]:
            raise ContextTranscodeError(f"{name}: v2.1 encoder source SHA mismatch")
        encoded_seconds = time.monotonic() - started - source_seconds
        _startup_detail(tensor=name, step='v21_verify', v21_encode_s=encoded_seconds)
        # Native packing is an explicit opt-in. Verify its result with the
        # bounded native inverse instead of expanding the overflow bitstream
        # into NumPy rank/mask arrays. CRC and reconstructed SHA checks remain
        # enabled, and the ordinary unselected path remains unchanged.
        if hasattr(encoded, "_native_packer_receipt"):
            from glc_loader import codec_v21_native
            decoded = codec_v21_native.decode_tensor(encoded, check_crc=True, check_sha=True)
        else:
            decoded = cv21.decode_tensor(encoded, check_crc=True, check_sha=True)
        decoded_sha = hashlib.sha256(memoryview(np.ascontiguousarray(decoded)).cast("B")).hexdigest()
        if decoded_sha != row["source_sha256"] or decoded.shape != tuple(row["shape"]):
            raise ContextTranscodeError(f"{name}: v2.1 round-trip source SHA mismatch")
        used_frame_sha = (self._selected[name]["frame_sha256"] if name in self._selected
                          else row["frame_sha256"])
        runtime_decoder_record = getattr(self, "runtime_decoder_record", None)
        self.transcode_receipts.append({
            "name": name, "frame_sha256": used_frame_sha,
            "kind": "predictive" if name in self._selected else "baseline",
            "source_sha256": row["source_sha256"],
            "v21_stream_bytes": encoded.total_bytes(),
            "context_decode_seconds": source_seconds,
            "v21_encode_seconds": encoded_seconds,
            "v21_verify_seconds": time.monotonic() - started - source_seconds - encoded_seconds,
            **({"runtime_encoder": dict(encoded._native_packer_receipt)}
               if hasattr(encoded, "_native_packer_receipt") else {}),
            **({"runtime_decoder": dict(runtime_decoder_record)}
               if runtime_decoder_record is not None else {}),
            **({"runtime_verifier": dict(encoded._native_decoder_receipt)}
               if hasattr(encoded, "_native_decoder_receipt") else {}),
        })
        self._cached_name, self._cached_tensor = name, encoded
        _startup_detail(tensor=name, step='tensor_ready', elapsed_s=time.monotonic()-started)
        return encoded

    def decode_host(self, name):
        """Return verified source words for tensors placed dense by the loader."""
        row = self.stored_record(name)
        # Dense vision, norms and embeddings do not consume V21 streams. Their
        # original Context decoder has already checked frame and source hashes;
        # encoding and decoding a second representation adds no source proof.
        decoded = np.array(self.read_source_words(name), copy=True, order="C")
        if (decoded.dtype != np.uint16 or decoded.shape != tuple(row["shape"])
                or decoded.nbytes != row["source_bytes"]
                or hashlib.sha256(memoryview(decoded).cast("B")).hexdigest() != row["source_sha256"]):
            raise ContextTranscodeError(f"{name}: host fallback source SHA mismatch")
        return decoded


__all__ = ["ContextTranscodeError", "ContextV21TranscodeArtifact",
           "FROZEN_CONTEXT_MANIFEST_SHA256", "FROZEN_PREDICTIVE_MANIFEST_SHA256"]
