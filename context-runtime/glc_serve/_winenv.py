"""Platform facts the serving and benchmark code needs, in one place.

This package was written and measured on Linux and macOS.  Running it on a
Windows machine splits into two genuinely different environments, and almost
every portability question below has a different answer in each:

``wsl2``
    A real Linux kernel.  Standard Linux wheels (``torch``, ``triton``) install
    unchanged, ``nvcc`` and ``g++`` are apt packages, and the NVIDIA *Windows*
    driver is already visible through ``/dev/dxg`` -- no Linux driver and no
    CUDA toolkit download are needed for the torch wheels.  This is the
    supported path.

``windows``
    Native CPython on Win32.  ``triton`` publishes no wheel on PyPI for this
    platform, the JIT kernel builds need MSVC ``cl.exe`` rather than ``g++``
    (and ``/O2`` rather than ``-O3``), and the temporary-directory variable is
    ``TEMP``/``TMP``, not ``TMPDIR``.  The install and the CPU checks work; the
    CUDA kernel path has never been built here.

Nothing in this module measures anything.  It reports what the interpreter can
see and leaves every decision to the caller.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

__all__ = ["os_kind", "is_windows", "is_wsl", "host_cflags", "temp_env",
           "toolchain_report", "WINDOWS_README"]

WINDOWS_README = (
    "https://github.com/Tsotchke-Corporation/GeoRefine/blob/release/v1.2.0-rc2/"
    "README-WINDOWS.md")


def is_windows() -> bool:
    """Native Win32 CPython.  False inside WSL2, which is Linux."""
    return os.name == "nt"


def is_wsl() -> bool:
    """A Linux interpreter running on the WSL2 kernel.

    The kernel release string carries the marker; ``WSL_DISTRO_NAME`` is set by
    the default WSL shell init but not by every launcher, so it is only a
    fallback.
    """
    if os.name == "nt":
        return False
    try:
        if "microsoft" in Path("/proc/version").read_text().lower():
            return True
    except OSError:
        pass
    return bool(os.environ.get("WSL_DISTRO_NAME"))


def os_kind() -> str:
    """One of ``windows``, ``wsl2``, ``linux``, ``macos``, ``other``."""
    if is_windows():
        return "windows"
    if is_wsl():
        return "wsl2"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    return "other"


def host_cflags() -> list:
    """Host-compiler optimisation flags for ``torch.utils.cpp_extension``.

    ``-O3`` is a GCC/Clang spelling.  Passed to MSVC ``cl.exe`` it is not a
    no-op -- ``cl`` reads it as ``-O`` plus the undefined option ``3`` and the
    build fails -- so the Windows host compiler gets its own spelling.  nvcc
    accepts ``-O3`` on every platform and is not affected.
    """
    return ["/O2"] if is_windows() else ["-O3"]


def temp_env(env: dict, tmpdir) -> dict:
    """Point the subprocess at ``tmpdir`` for scratch, on either platform.

    ``tempfile`` reads ``TMPDIR`` on POSIX and ``TMP``/``TEMP`` on Windows, so a
    run that only sets ``TMPDIR`` silently spills Windows temporaries back onto
    the system drive -- which is the drive most likely to be the small one.
    Mutates and returns ``env``.
    """
    s = str(tmpdir)
    for var in ("TMPDIR", "TMP", "TEMP"):
        env.setdefault(var, s)
    return env


def toolchain_report() -> dict:
    """What of the JIT kernel toolchain is actually on PATH.

    The kernel modules build with ``torch.utils.cpp_extension`` at first use, so
    a missing ``nvcc`` or host compiler is not an import error -- it is a
    failure in the middle of a long run.  Reporting it in preflight turns that
    into a message before the download.

    rc3 finding G1: this probed the *bare* ``PATH``, so it reported ``ninja: None`` on a box
    where ``ninja`` was installed -- in the venv's own ``bin/``, which a console script
    invoked by absolute path never puts on ``PATH``.  The probe now runs through
    :func:`glc_loader._jitenv.build_tool_report`, which puts the interpreter's script
    directories on ``PATH`` first and then resolves exactly as the build resolves.  A
    preflight that answers a different question from the builder is worse than no preflight.
    """
    from glc_loader._jitenv import build_tool_report

    tools = build_tool_report()
    rep = {"os_kind": os_kind(), "platform": platform.platform(),
           "nvcc": tools["nvcc"], "ninja": tools["ninja"],
           "host_compiler": tools["host_compiler"],
           "host_compiler_name": tools["host_compiler_name"],
           "path_dirs_added": tools["path_dirs_added"]}
    if rep["os_kind"] == "wsl2":
        # The Windows driver's Linux-side libraries are bind-mounted here by WSL
        # itself; their absence means the distro is WSL1, or the driver predates
        # CUDA-on-WSL support.
        rep["wsl_lib_present"] = Path("/usr/lib/wsl/lib").is_dir()
        rep["nvidia_smi"] = shutil.which("nvidia-smi")
    return rep
