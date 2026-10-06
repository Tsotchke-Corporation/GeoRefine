# Bit-exact context package builder

This standalone delivery contains the codec, C++ rANS helper, complete-model package builder, and focused tests. It packages a context-coded BF16 checkpoint for storage and can verify and restore the exact original shard and metadata bytes.

## Scope and limits

The measured artifact is a **storage package**: 36,403,357,016 bytes for 55,562,855,904 source tensor bytes (1.5263x ratio), with 1,199 tensors, 18 original shards, and 13 metadata assets. Exact restoration and package integrity were verified on the full artifact. This package does **not** provide native compressed GPU serving. GPU decode integration, serving speed, and VRAM benefit are unmeasured; after restore, normal BF16 serving consumes the restored checkpoint.

## Prerequisites

- Python 3.10+
- NumPy
- A C++17 compiler (the codec builds its helper on first use)
- For the full checkpoint: about 36.4 GB for the package and 55.6 GB for the restored tensors, plus working room

Install NumPy in the Python environment you intend to use, for example `python3 -m pip install numpy`. No model framework is required.

## Obtain and verify the package

Place the context package obtained from the approved release location at `./qwen38-context-package`, then verify it with:

```sh
python3 ./qwen38-context-package/decoder/bitexact_context_package.py verify ./qwen38-context-package --jobs 8 --cache-dir ./.scratch/context-decoder-cache
```

Verification checks the package inventory and hashes and decodes frames against their recorded source identity. Keep the decoder cache outside the package directory.

## Restore the original BF16 checkpoint

```sh
python3 ./qwen38-context-package/decoder/bitexact_context_package.py restore ./qwen38-context-package ./qwen38-restored-bf16 --jobs 8 --cache-dir ./.scratch/context-decoder-cache
```

The destination contains the original 18 safetensors shards and the archived sidecar assets. The package remains intact. Confirm that the destination has enough free disk before restoring.

## Build a package from source inputs

Inputs are a directory containing the indexed BF16 source shards, a tensor census JSON with per-tensor SHA256 and byte counts, and the metadata sidecar tar.xz. The baseline deployment pin is optional; use it when a pinned baseline comparison is part of the build contract.

```sh
python3 scripts/bitexact_context_package.py build \
  --model-dir /path/to/model \
  --out /path/to/context-package \
  --census /path/to/census.json \
  --metadata-archive /path/to/model-assets.tar.xz \
  --jobs 2
python3 scripts/bitexact_context_package.py verify /path/to/context-package --jobs 8 --cache-dir ./.scratch/context-decoder-cache
```

## Run the small CPU tests

From the repository root, with NumPy and pytest installed:

```sh
mkdir -p .scratch
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:capture -p no:cacheprovider tests/test_bitexact_context_package.py -q --basetemp=.scratch/test-package
```

The package tests use synthetic tensors and verify encoding/decoding, package integrity, exact synthetic shard and sidecar restoration, resume/reuse behavior, and fail-closed corruption handling. They do not compress the real model. `-p no:capture` avoids a known pytest capture-plugin crash in the development environment used to prepare this bundle; omit it on environments where normal pytest capture works.

## Provenance and integrity

- Package builder and package tests: commit `bb4a175e39ebd27009edd2de317649116366b734`
- Codec and C++ helper: commit `7c72d21172002c2a8123090b9a5ecb3bbf9a2a20`
- Full artifact manifest SHA256: `d1a3d29c759f2056c20049c49b6b04479e151b74b52b688ab774b2d7253e4e8c`
- Full artifact archive SHA256: `73665ab061483485a0395bb28ead9a3e9252edfa3d6932e153a07badbd6745c8`

In the standalone bundle, `SHA256SUMS` inventories every delivered file except itself. Check that bundle with `shasum -a 256 -c SHA256SUMS`.
