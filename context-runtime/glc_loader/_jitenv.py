"""Make the build tools this package declares actually findable (rc3 finding G1).

Why this module exists
----------------------
Every CUDA kernel in this package is JIT-built by ``torch.utils.cpp_extension``, which
shells out to ``ninja`` and resolves it with ``shutil.which`` -- i.e. through ``PATH``.
rc3 declared ``ninja>=1.11.0`` in the ``[cuda]`` extra, pip installed it into
``<venv>/bin/ninja``, and the next GPU run still failed every kernel build with::

    RuntimeError: Ninja is required to load C++ extensions (pip install ninja to get it)

The reason is that a console script invoked by absolute path --
``<venv>/bin/glc-bench`` -- does **not** put ``<venv>/bin`` on ``PATH``.
Activating a venv is what does that, and nobody has to activate a venv to run a console
script.  So the declared, installed dependency was invisible to the only lookup that
mattered.

The fix is to put the interpreter's own script directory on ``PATH`` before any
``cpp_extension`` call, which is where the declared dependency already is.  Nothing needs
installing and nothing needs activating.

Contract
--------
``ensure_build_tools_on_path()`` is idempotent, never raises, and never *removes* anything
from ``PATH``: it only prepends directories that are not already there.  Call it
immediately before ``cpp_extension.load`` / ``load_inline``, and once at every console-script
entry point.  ``build_tool_report()`` resolves the tools the way the build resolves them --
after the fix -- so a preflight cannot disagree with what the builder will see.
"""
from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from typing import Dict, List, Optional

__all__ = ["ensure_build_tools_on_path", "build_tool_report", "interpreter_script_dirs"]


def interpreter_script_dirs() -> List[str]:
    """Directories where *this* interpreter's console scripts and pip-installed tools live.

    ``dirname(sys.executable)`` is the venv's ``bin`` (POSIX) or the venv root (Windows);
    ``sysconfig``'s ``scripts`` path is ``Scripts`` on Windows and the same ``bin`` on POSIX.
    ``ninja.BIN_DIR`` is where the ``ninja`` PyPI wheel puts its binary when the console
    shim is not what gets used.  All three are consulted because which one holds the binary
    depends on the installer, and guessing wrong is the whole defect.
    """
    dirs: List[str] = []

    def add(d: Optional[str]) -> None:
        if d and os.path.isdir(d) and d not in dirs:
            dirs.append(d)

    exe = getattr(sys, "executable", None)
    if exe:
        add(os.path.dirname(os.path.abspath(exe)))
    try:
        add(sysconfig.get_path("scripts"))
    except (KeyError, OSError):
        pass
    try:
        import ninja                                              # noqa: PLC0415

        add(getattr(ninja, "BIN_DIR", None))
    except Exception:                                             # noqa: BLE001
        pass
    return dirs


def ensure_build_tools_on_path(env: Optional[Dict[str, str]] = None) -> List[str]:
    """Prepend this interpreter's script directories to ``PATH``.  Returns what was added.

    ``env`` defaults to ``os.environ`` (which is what ``shutil.which`` and every
    ``cpp_extension`` subprocess read).  Pass a dict to prepare a child process's
    environment instead -- a server that spawns the benchmark, for instance.
    """
    target = os.environ if env is None else env
    current = target.get("PATH", "") or ""
    have = [p for p in current.split(os.pathsep) if p]
    added = [d for d in interpreter_script_dirs() if d not in have]
    if added:
        target["PATH"] = os.pathsep.join(added + have) if have else os.pathsep.join(added)
    return added


def build_tool_report() -> Dict[str, Optional[str]]:
    """Resolve ninja / nvcc / the host compiler the way ``cpp_extension`` resolves them.

    Calls :func:`ensure_build_tools_on_path` first, so the answer is the answer the build
    will get -- not the answer a bare ``PATH`` would have given.  rc3's preflight probed the
    bare ``PATH`` and therefore reported ``ninja None`` for a ninja that was installed, and
    printed ``pip install ninja`` as the remedy for a package pip had already installed.
    """
    added = ensure_build_tools_on_path()
    win = sys.platform == "win32"
    rep: Dict[str, Optional[str]] = {
        "ninja": shutil.which("ninja"),
        "nvcc": shutil.which("nvcc"),
        "host_compiler": shutil.which("cl") if win
        else (shutil.which("c++") or shutil.which("g++")),
    }
    rep["host_compiler_name"] = "cl.exe (MSVC)" if win else "c++/g++"
    rep["path_dirs_added"] = os.pathsep.join(added)
    return rep
