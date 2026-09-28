"""The loader must be installable, and its CLI must be honest.

Two things are asserted here that no other test in the tree asserts:

1. ``release/pyproject.toml`` describes a distribution that can actually be
   built and whose import surface needs NEITHER the ``[cuda]`` nor the
   ``[metal]`` extra.  The measurement is a meta-path block on ``mlx``,
   ``mlx_lm`` and ``triton`` -- this repository's dev environment HAS all
   three installed, so a plain ``import glc_loader`` would prove nothing.

2. ``glc_loader.__main__``'s ``info`` / ``verify`` / ``load`` refuse
   accurately.  The case that matters is the v1 TBE transcode directory: it
   is a CORRECT artifact with no self-contained verifier, so it must exit 4
   (unverifiable), never 2 (malformed) and never 0.

Scratch goes under ``<repo>/.scratch``, never ``/tmp``.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "release"

if str(RELEASE) not in sys.path:
    sys.path.insert(0, str(RELEASE))

torch = pytest.importorskip("torch")

from glc_loader import _inspect as gi  # noqa: E402
from glc_loader.__main__ import main as cli_main  # noqa: E402


# ---------------------------------------------------------------------------
# 1. the distribution metadata
# ---------------------------------------------------------------------------
def _pyproject() -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - the repo targets >=3.10
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads((RELEASE / "pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_exists_and_names_the_package():
    cfg = _pyproject()
    assert cfg["project"]["name"] == "glc-loader"
    find = cfg["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["."], (
        "the package root must stay at release/ -- tests import both "
        "`glc_loader` (with release/ on sys.path) and "
        "`release.glc_loader.metal.*` (with the repo root on sys.path), and "
        "moving the tree breaks one of them"
    )
    assert "glc_loader" in find["include"]
    assert "glc_loader.*" in find["include"]


def test_declared_version_matches_the_package_version():
    """The pyproject version is static, so nothing keeps the two in step but
    this assertion.  ``attr:`` was not used because resolving it can import
    the package, and importing it needs torch at build time."""
    import glc_loader

    assert _pyproject()["project"]["version"] == glc_loader.__version__


def test_extras_are_independent_and_neither_is_a_core_dependency():
    cfg = _pyproject()
    core = " ".join(cfg["project"]["dependencies"])
    extras = cfg["project"]["optional-dependencies"]
    for name in ("triton", "mlx", "mlx-lm"):
        assert name not in core, f"{name} must not be a core dependency"
    assert any(d.startswith("triton") for d in extras["cuda"])
    assert any(d.startswith("mlx") for d in extras["metal"])
    assert not any("mlx" in d for d in extras["cuda"]), (
        "[cuda] must not drag in the Metal stack"
    )
    assert not any("triton" in d for d in extras["metal"]), (
        "[metal] must not drag in the CUDA stack"
    )


def test_the_console_script_points_at_a_real_entry_point():
    cfg = _pyproject()
    target = cfg["project"]["scripts"]["glc-loader"]
    mod, _, func = target.partition(":")
    assert callable(getattr(importlib.import_module(mod), func))


def test_pinned_floors_are_declared_for_all_three_runtime_dependencies():
    core = _pyproject()["project"]["dependencies"]
    tops = {d.split(">")[0].split("=")[0].strip() for d in core}
    assert tops == {"torch", "safetensors", "transformers"}
    assert all(">=" in d for d in core), "every dependency carries a floor"


# ---------------------------------------------------------------------------
# 2. the import surface needs neither extra
# ---------------------------------------------------------------------------
_NO_EXTRAS_PROBE = r"""
import importlib, json, sys
BLOCKED = ("mlx", "mlx_lm", "triton")

class _Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(fullname, name=fullname)
        return None

sys.meta_path.insert(0, _Block())
for name in list(sys.modules):
    if name.split(".")[0] in BLOCKED:
        del sys.modules[name]

out = {}
import glc_loader
out["package"] = "ok"
for mod in %(mods)r:
    try:
        importlib.import_module(mod)
        out[mod] = "ok"
    except BaseException as exc:
        out[mod] = "%%s: %%s" %% (type(exc).__name__, exc)
try:
    importlib.import_module("glc_loader.metal")
    out["metal"] = "imported"
except ModuleNotFoundError:
    out["metal"] = "refused"
from glc_loader.container import load_kernels
out["load_kernels"] = repr(load_kernels())
print(json.dumps(out))
"""

_TOP_LEVEL_MODULES = [
    "glc_loader.__main__", "glc_loader._inspect", "glc_loader.artifact",
    "glc_loader.container", "glc_loader.loader", "glc_loader.modules",
    "glc_loader.tbe_artifact", "glc_loader.tbe_container",
    "glc_loader.tbe_device_map", "glc_loader.tbe_mma",
    "glc_loader.tbe_mma_kernels", "glc_loader.tbe_modules",
    "glc_loader.tbe_serving", "glc_loader.tbe_stream_loader",
]


def test_every_module_imports_with_mlx_and_triton_blocked(tmp_path_factory):
    """The measurement, not the assumption.

    ``glc_loader.metal.*`` DOES ``import mlx.core`` at module scope; the
    package's own ``__init__`` must not reach it, or ``pip install
    glc-loader`` would be unusable on a CUDA box.
    """
    src = _NO_EXTRAS_PROBE % {"mods": _TOP_LEVEL_MODULES}
    scratch = REPO / ".scratch" / "glc-packaging-tests"
    scratch.mkdir(parents=True, exist_ok=True)
    probe = scratch / "no_extras_probe.py"
    probe.write_text(src, encoding="utf-8")
    env = dict(os.environ)
    # release/ on the path is the same placement `pip install ./release`
    # produces -- the probe must resolve `glc_loader` as a top-level package,
    # not as `release.glc_loader`.
    env["PYTHONPATH"] = str(RELEASE) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(probe)],
        cwd=str(RELEASE), capture_output=True, text=True, timeout=600, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["package"] == "ok"
    for mod in _TOP_LEVEL_MODULES:
        assert out[mod] == "ok", f"{mod} needs an extra at import time: {out[mod]}"
    assert out["metal"] == "refused", (
        "glc_loader.metal must need the [metal] extra, or the extra is "
        "decorative"
    )
    assert out["load_kernels"] == "None", (
        "without triton the FWP1 kernel loader must return None and let the "
        "pure-torch backend take over, not raise"
    )


def test_the_package_init_does_not_import_the_metal_subpackage():
    text = (RELEASE / "glc_loader" / "__init__.py").read_text(encoding="utf-8")
    assert "from .metal" not in text and "import metal" not in text


# ---------------------------------------------------------------------------
# 3. the CLI refuses accurately
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def legacy_v1_tbe(tmp_path_factory) -> Path:
    """A v1 TBE transcode directory: a manifest and two blobs, no tag."""
    from safetensors.torch import save_file

    d = REPO / ".scratch" / "glc-packaging-tests" / "legacy_v1"
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": gi.TBE_MANIFEST_SCHEMA,
        "status": "ok",
        "container": {"layout": "mma16", "superblock": 32},
        "summary": {"n_coded": 0, "n_raw": 1, "n_total": 1},
        "escape_over_band": [],
        "escape_band_pct": 8.0,
        "source": {"model_dir": "/nowhere", "n_tensors": 1},
        "tensors": [{"name": "w", "kind": "raw", "original_bytes": 32,
                     "shape": [4, 4], "dtype": "torch.bfloat16",
                     "reason": "non_2d", "blake2b_source": "00"}],
        "accounting": {
            "stored": {"scope": "stored", "stored_bytes": 32,
                       "dense_equivalent_bytes": 32, "ratio": 1.0},
            "served_resident": {"scope": "served", "resident_bytes": 32,
                                "dense_equivalent_bytes": 32, "ratio": 1.0},
            "whole_process": {"scope": "whole", "status": "unmeasured",
                              "measured_vram_bytes": None, "ratio": None,
                              "dense_equivalent_bytes": 32},
        },
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    save_file({"w": torch.zeros(4, 4, dtype=torch.bfloat16)},
              str(d / "raw_tensors.safetensors"))
    return d


def test_a_legacy_v1_tbe_directory_is_unverifiable_not_malformed(legacy_v1_tbe):
    det = gi.detect(legacy_v1_tbe)
    assert det.kind == gi.KIND_TBE_V1_TRANSCODE
    assert det.reason == "legacy_v1_container"
    res = gi.verify_artifact(det)
    assert res["exit_code"] == gi.EXIT_UNVERIFIABLE == 4, (
        "exit 2 would call a well-formed artifact malformed; exit 0 would "
        "claim a check that did not happen"
    )
    assert res["status"] == "UNVERIFIABLE"
    assert cli_main(["verify", str(legacy_v1_tbe)]) == 4
    assert cli_main(["load", str(legacy_v1_tbe)]) == 4


def test_the_refusal_says_why_a_v1_directory_cannot_be_loaded(legacy_v1_tbe):
    det = gi.detect(legacy_v1_tbe)
    for phrase in ("compression_info.json", "config.json", "tokenizer",
                   "dense checkpoint"):
        assert phrase in det.detail


def test_a_plain_hf_checkpoint_is_declined_not_claimed(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')
    (tmp_path / "model.safetensors").write_bytes(b"")
    det = gi.detect(tmp_path)
    assert det.kind == gi.KIND_UNKNOWN
    assert det.reason == "not_a_georefine_artifact"
    assert cli_main(["verify", str(tmp_path)]) == gi.EXIT_MALFORMED


def test_an_m2_artifact_gets_its_own_auditor_named(tmp_path):
    (tmp_path / "manifest.json").write_text('{"schema": "m2"}')
    (tmp_path / "blobs").mkdir()
    det = gi.detect(tmp_path)
    assert det.kind == gi.KIND_M2
    assert "lossless_audit" in det.detail
    assert "wrong auditor" in det.detail


def test_verify_on_a_file_rather_than_a_directory_is_typed(tmp_path):
    f = tmp_path / "MANIFEST.json"
    f.write_text("{}")
    with pytest.raises(gi.InspectError) as exc:
        gi.detect(f)
    assert exc.value.reason == "not_a_directory"


# ---------------------------------------------------------------------------
# 4. the three scopes are never conflated
# ---------------------------------------------------------------------------
def test_the_three_scopes_are_always_present_and_separately_sourced(legacy_v1_tbe):
    scopes = gi.scope_ratios(gi.detect(legacy_v1_tbe))
    assert set(scopes) == {"stored", "served_resident", "whole_process"}
    assert scopes["whole_process"]["status"] == "unmeasured"
    assert scopes["whole_process"]["ratio"] is None, (
        "an unmeasured scope must not borrow another scope's number"
    )
    assert scopes["stored"]["basis"] != scopes["served_resident"]["basis"]


def test_an_unmeasured_scope_is_listed_under_what_is_not_certified(legacy_v1_tbe):
    lines = gi.not_certified(gi.detect(legacy_v1_tbe))
    assert any(line.startswith("whole_process ratio: NOT measured") for line in lines)
    speed = [line for line in lines if "decode SPEED" in line]
    assert speed and "depends on the" in speed[0] and "KERNEL_SPEED_20260925" in speed[0]
    assert not any("not a speed lever" in line for line in lines)


def test_a_missing_certificate_is_reported_as_absent_not_as_a_pass(legacy_v1_tbe):
    summary = gi.certificate_summary(gi.detect(legacy_v1_tbe))
    assert summary["present"] is False
    assert summary["verdict"] is None


def test_expansion_is_its_own_exit_code(legacy_v1_tbe, tmp_path):
    """A lossless tier that enlarges what it stores is a refusal, not a pass."""
    det = gi.detect(legacy_v1_tbe)
    det.manifest["accounting"]["stored"] = {
        "scope": "stored", "stored_bytes": 64,
        "dense_equivalent_bytes": 32, "ratio": 0.5,
    }
    assert gi._expansion_verdict(det) is not None
    assert "EXPANDS" in gi._expansion_verdict(det)


# ---------------------------------------------------------------------------
# 5. the README does not overclaim
# ---------------------------------------------------------------------------
def test_the_readme_states_the_measured_decode_numbers_and_their_spread():
    text = (RELEASE / "README.md").read_text(encoding="utf-8")
    assert "0.94" in text and "0.57" in text and "0.62" in text
    assert "single run" in text.lower()
    assert "0.05" in text, "the spread must be stated, not just the point value"
    # 2026-09-25: the engine receipts (KERNEL_SPEED_20260925.md) contradict the old
    # blanket "not a speed lever" line; speed is stated per decode path instead.
    assert "not a speed lever" not in text
    assert "37.84" in text and "27.4" in text and "KERNEL_SPEED_20260925.md" in text


def test_the_readme_scopes_bundled_tbe_and_artifact_local_fwp1():
    text = (RELEASE / "README.md").read_text(encoding="utf-8")
    assert "fwp1_kernels.py" in text and "tbe_mma_kernel.cu" in text
    assert "This wheel includes the TBE MMA source" in text
    assert "FWP1 Triton source is still artifact-local" in text
    assert (RELEASE / "glc_loader" / "tbe_mma_kernel.cu").is_file()
    assert not (RELEASE / "glc_loader" / "fwp1_kernels.py").exists()
