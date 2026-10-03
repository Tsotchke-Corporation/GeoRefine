#!/usr/bin/env python3
"""``glc-bench`` -- one command that measures GLC multi-user serving against llama.cpp
on the card it is run on.

    glc-bench

With no flags it:

  1. checks Python, torch, the NVIDIA driver, free disk and free VRAM, and prints the
     exact fix for whatever is missing instead of a traceback;
  2. downloads the GLC-TBE artifact from Hugging Face at a pinned revision (and, if
     ``llama-server`` is on PATH, uses a GGUF of the same parent that you point at with
     ``--llama-gguf``);
  3. runs the correctness gates (kernel, batch-exactness, engine, HTTP stream);
  4. runs the throughput/latency/memory benchmark against our server and, when it is
     available, against llama.cpp with matched continuous batching;
  5. prints one table and writes ``glc-bench-results-<date>.tar.gz``.

Nothing in this script prints a number it did not measure on this machine.  Every
measured figure is tagged ``MEASURED-ON-THIS-CARD``.  Gates print PASS or FAIL together
with how many output elements they compared, so a PASS on a tiny sample cannot be read as
a PASS on a large one.  Anything that could not run is printed as ``NOT RUN`` with the
reason, and the run continues.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from glc_serve import _winenv

PKG = Path(__file__).resolve().parent
BENCH = PKG / "_bench"

HF_REPO_DEFAULT = "Tsotchke-Corporation/Qwen3.8-27B-GeoRefine-TBE"
HF_REV_DEFAULT = "2725c98a0b152b3092a9f5c692e5b28729d6c052"
TUNE_DEFAULT = PKG / "miv_tbe_configs" / "tbe_autotune_rtxpro6000.json"

# Bundle download size, from the published repository's file sizes (not a measurement of
# your disk): the 20 shards total about 39.9 GB, plus ~20 MB of tokenizer and manifest.
BUNDLE_GB = 40.0
MARK = "MEASURED-ON-THIS-CARD"

LLAMACPP_INSTALL = """\
  git clone https://github.com/ggml-org/llama.cpp
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build llama.cpp/build --config Release -j --target llama-server
then re-run with:
  glc-bench --llama-server llama.cpp/build/bin/llama-server --llama-gguf <model.gguf>"""

WSL_ADVICE = """\
You are running native Windows Python.  The CUDA kernels in this package are
JIT-built with nvcc and a host C++ compiler, and `triton` publishes no Windows
wheel on PyPI; neither has ever been built on native Windows here.  The
supported Windows path is WSL2, which is a real Linux kernel using your existing
NVIDIA *Windows* driver -- no second driver, no CUDA toolkit download for torch:

  PowerShell (Administrator), once:
      wsl --install -d Ubuntu-24.04
  reboot, open "Ubuntu 24.04" from the Start menu, then follow
      %s

`glc-bench --windows-check` runs every check that does not need CUDA, on either
platform, and tells you what is missing.""" % _winenv.WINDOWS_README

COST_SOURCES = """\
$/M output token is ($/h of the card) / (steady tok/s) * 1e6 / 3600.  This script does
not ship a price: a rate that is not read off your own invoice is not a measurement.
Pass the rate you are actually paying:

    glc-bench --dollars-per-hour 1.23

Where to read it for the two cards this engine targets (L40S, RTX PRO 6000 Blackwell),
on-demand and spot/interruptible:
  https://cloud.google.com/compute/gpus-pricing
  https://www.runpod.io/pricing
  https://lambda.ai/pricing
  https://vast.ai/pricing
Both arms in the table below ran on the same card in the same run, so the ratio of their
$/M figures is independent of the rate you use."""


# --------------------------------------------------------------------------- utilities
class Log:
    def __init__(self, path: Path):
        self.path = path
        self.fh = open(path, "a", buffering=1)

    def __call__(self, *a):
        line = "[%s] %s" % (time.strftime("%H:%M:%SZ", time.gmtime()), " ".join(str(x) for x in a))
        print(line, flush=True)
        self.fh.write(line + "\n")

    def raw(self, text: str):
        self.fh.write(text + "\n")


def sh(cmd, log, timeout=None, env=None, cwd=None):
    """Run a command, stream nothing, return (rc, combined output)."""
    log("$", " ".join(str(c) for c in cmd))
    try:
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=cwd)
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s\n{e.stdout or ''}{e.stderr or ''}"
    except FileNotFoundError as e:
        return 127, str(e)
    out = (p.stdout or "") + (p.stderr or "")
    log.raw(out)
    return p.returncode, out


def http_json(url, key=None, timeout=5):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def nvidia_smi(query):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------- preflight
def preflight(a, log) -> dict:
    """Environment facts plus a list of blocking problems, each with its fix."""
    rep = {"platform": platform.platform(), "python": sys.version.split()[0],
           "argv": sys.argv[1:], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    bad = []
    warn = []

    tc = _winenv.toolchain_report()
    rep.update({"os_kind": tc["os_kind"], "nvcc": tc["nvcc"], "ninja": tc["ninja"],
                "host_compiler": tc["host_compiler"]})
    if tc["os_kind"] == "windows":
        # Not a hard blocker: --windows-check and --cpu-smoke are useful here, and a
        # person who has built MSVC + CUDA themselves should not be refused by name.
        # It is the first thing printed, because it is the first thing to fix.
        warn.append(("native Windows Python -- this is not the supported path",
                     WSL_ADVICE))
    if tc["os_kind"] == "wsl2":
        rep["wsl_lib_present"] = tc.get("wsl_lib_present")
        if not tc.get("wsl_lib_present"):
            bad.append(("WSL is running but /usr/lib/wsl/lib is missing, so the GPU is "
                        "not passed through",
                        "this is WSL1, or the NVIDIA Windows driver is too old.  In "
                        "PowerShell:\n"
                        "    wsl --set-version Ubuntu-24.04 2\n"
                        "    wsl --update\n"
                        "  then install the current NVIDIA Windows driver (566 or newer) "
                        "from nvidia.com.  Do NOT install a Linux driver inside WSL."))
    if tc["nvcc"] is None:
        warn.append(("nvcc is not on PATH; the JIT kernel builds will fail when the "
                     "benchmark reaches them",
                     "WSL2/Ubuntu:  sudo apt-get update && sudo apt-get install -y "
                     "nvidia-cuda-toolkit build-essential ninja-build\n"
                     "  (or the CUDA 12.x WSL repo, if you need a newer nvcc than Ubuntu "
                     "ships -- see README-WINDOWS.md)"))
    if tc["host_compiler"] is None:
        warn.append(("no host C++ compiler (%s) on PATH" % tc["host_compiler_name"],
                     "WSL2/Ubuntu:  sudo apt-get install -y build-essential ninja-build\n"
                     "  native Windows:  install Visual Studio Build Tools with the "
                     "\"Desktop development with C++\" workload, then run from the "
                     "\"x64 Native Tools Command Prompt\""))

    if sys.version_info < (3, 10):
        bad.append(("python too old (%s)" % rep["python"],
                    "install Python 3.10 or newer and re-create the virtualenv"))

    try:
        import torch
        rep["torch"] = torch.__version__
        rep["torch_cuda_build"] = getattr(torch.version, "cuda", None)
        rep["cuda_available"] = bool(torch.cuda.is_available())
        if rep["cuda_available"]:
            rep["device_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            rep["device_capability"] = "%d.%d" % cap
            rep["device_total_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            rep["device_count"] = torch.cuda.device_count()
        else:
            bad.append(("torch is installed but reports no CUDA device",
                        "this is almost always a CPU-only torch wheel.  Fix:\n"
                        "    pip uninstall -y torch\n"
                        "    pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
                        "  (if `nvidia-smi` also fails, the driver is missing, not torch)"))
    except Exception as e:                                           # noqa: BLE001
        rep["torch"] = None
        rep["torch_import_error"] = repr(e)[:300]
        bad.append(("torch could not be imported (%r)" % (e,),
                    "    pip install torch --index-url https://download.pytorch.org/whl/cu128"))

    rep["nvidia_smi_driver"] = nvidia_smi("driver_version")
    rep["nvidia_smi_memory_total_mib"] = nvidia_smi("memory.total")
    rep["nvidia_smi_memory_used_mib"] = nvidia_smi("memory.used")
    if rep["nvidia_smi_driver"] is None:
        bad.append(("`nvidia-smi` is not available",
                    "no NVIDIA driver on this machine (or no GPU attached).  Nothing in "
                    "this benchmark can run; check with your provider that the instance "
                    "has a GPU and the driver is installed."))

    try:
        du = shutil.disk_usage(a.work.parent if a.work.parent.exists() else Path.cwd())
        rep["free_disk_gb"] = round(du.free / 1e9, 1)
        need = 0.0 if (a.bundle or a.skip_download or a.cpu_smoke) else BUNDLE_GB + 5
        rep["disk_needed_gb"] = need
        if need:
            log("")
            log("   NOTE: the model artifact is about %.0f GB and is downloaded to"
                % BUNDLE_GB)
            log("         %s" % (a.work / "model"))
            log("         With logs and build output, keep %.0f GB free on that volume."
                % need)
            log("         To put it somewhere else -- a second drive, which on Windows is")
            log("         usually what has the room -- pass --work:")
            log("             glc-bench --work D:/glc-bench-work        (native Windows)")
            log("             glc-bench --work /mnt/d/glc-bench-work    (WSL2, same D:)")
        if rep["free_disk_gb"] < need:
            bad.append(("only %.1f GB free on the volume holding %s, and the artifact "
                        "needs about %.0f GB" % (rep["free_disk_gb"], a.work, need),
                        "free space, attach a larger disk, or point --work at a bigger "
                        "volume (`--work D:/glc-bench-work` on Windows, "
                        "`--work /mnt/d/glc-bench-work` in WSL2), or pass --bundle <dir> "
                        "if you already have the artifact"))
    except OSError:
        pass

    try:
        import huggingface_hub
        rep["huggingface_hub"] = huggingface_hub.__version__
    except Exception:                                                # noqa: BLE001
        rep["huggingface_hub"] = None
        if not (a.bundle or a.skip_download):
            bad.append(("huggingface_hub is not installed",
                        "    pip install huggingface_hub"))

    rep["llama_server"] = a.llama_server or shutil.which("llama-server")
    rep["llama_gguf"] = str(a.llama_gguf) if a.llama_gguf else None
    rep["blockers"] = [{"problem": p, "fix": f} for p, f in bad]
    rep["warnings"] = [{"problem": p, "fix": f} for p, f in warn]

    log("")
    log("== 1. environment ==")
    for k in ("platform", "os_kind", "python", "torch", "torch_cuda_build",
              "cuda_available", "device_name", "device_capability",
              "device_total_mem_gb", "nvidia_smi_driver",
              "nvidia_smi_memory_total_mib", "free_disk_gb", "nvcc", "host_compiler",
              "ninja", "huggingface_hub", "llama_server"):
        if k in rep:
            log("   %-28s %s" % (k, rep[k]))
    for b in rep["warnings"]:
        log("")
        log("   WARNING: " + b["problem"])
        log("   FIX:     " + b["fix"].replace("\n", "\n   "))
    for b in rep["blockers"]:
        log("")
        log("   PROBLEM: " + b["problem"])
        log("   FIX:     " + b["fix"].replace("\n", "\n   "))
    return rep


# --------------------------------------------------------------------------- model
def fetch_bundle(a, log, rep) -> Path | None:
    if a.bundle:
        log("   using --bundle", a.bundle)
        return Path(a.bundle)
    if a.skip_download:
        log("   NOT RUN: --skip-download and no --bundle")
        return None
    from huggingface_hub import snapshot_download
    log("== 2. model ==")
    log("   repo     ", a.hf_repo)
    log("   revision ", a.hf_revision, "(pinned)")
    log("   about %.0f GB; this is the slow step." % BUNDLE_GB)
    t0 = time.time()
    try:
        d = snapshot_download(
            repo_id=a.hf_repo, revision=a.hf_revision,
            local_dir=str(a.work / "model"),
            allow_patterns=["shards/*", "serve_manifest.json*", "config.json",
                            "generation_config.json", "tokenizer*", "vocab.json",
                            "merges.txt", "chat_template.jinja",
                            "*preprocessor_config.json", "LICENSE", "NOTICE"],
            max_workers=a.download_workers)
    except Exception as e:                                           # noqa: BLE001
        log("   DOWNLOAD FAILED:", repr(e)[:400])
        log("   FIX: check network, then re-run -- the download resumes.  For a gated or "
            "rate-limited pull: `pip install -U huggingface_hub && hf auth login`.")
        rep["bundle_error"] = repr(e)[:400]
        return None
    rep["bundle_download_seconds"] = round(time.time() - t0, 1)
    rep["bundle_dir"] = d
    log("   downloaded in %.0f s -> %s" % (time.time() - t0, d))
    return Path(d)


# --------------------------------------------------------------------------- gates
def run_gates(a, log, bundle, tune, prompts, env, results):
    """The four correctness gates.  Each one is allowed to fail; the failure is recorded."""
    log("")
    log("== 4. correctness gates ==")
    log("   (these have never been run on a GPU before this release -- a FAIL here is a")
    log("    real finding, and the benchmark continues either way)")
    gdir = a.work / "gates"
    gdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    specs = [
        ("kernel", [py, "-m", "glc_serve._bench.bi_gate_kernel", "--timing",
                    "--out", gdir / "kernel.json"], 3600),
        ("batchexact", [py, "-m", "glc_serve._bench.bi_gate_batchexact",
                        "--out", gdir / "batchexact.json"], 5400),
    ]
    if bundle:
        specs.append(("engine", [py, "-m", "glc_serve._bench.bi_gate_engine",
                                 "--bundle", bundle, "--tune", tune,
                                 "--prompts", prompts, "--slots", a.slots,
                                 "--out", gdir / "engine.json"], 10800))

    for name, cmd, tmo in specs:
        log("")
        log("   -- gate: %s" % name)
        rc, out = sh(cmd, log, timeout=tmo, env=env)
        results["gates"][name] = read_gate(gdir / f"{name}.json", rc, out)
        log("   %s: %s" % (name, fmt_gate(results["gates"][name])))


def read_gate(path: Path, rc: int, out: str) -> dict:
    g = {"returncode": rc, "receipt": str(path)}
    if path.exists():
        try:
            g["json"] = json.loads(path.read_text())
        except ValueError as e:
            g["parse_error"] = str(e)
    else:
        g["parse_error"] = "no receipt written"
        g["tail"] = out[-1500:]
    j = g.get("json") or {}
    # Gate receipts in this package carry a verdict under one of these keys, and the
    # element count under one of these.  Nothing is inferred when neither is present.
    for k in ("verdict", "pass", "passed", "ok", "bitexact", "bitwise"):
        if k in j:
            g["verdict"] = "PASS" if j[k] in (True, "PASS", "pass", "ok") else "FAIL"
            break
    else:
        g["verdict"] = "PASS" if rc == 0 else "FAIL"
        g["verdict_basis"] = "process exit status (no verdict field in receipt)"
    for k in ("compared_elements", "elements_compared", "n_elements",
              "output_elements", "logits_elements"):
        if k in j:
            g["compared_elements"] = j[k]
            break
    for k in ("mismatched_elements", "mismatches", "n_mismatch"):
        if k in j:
            g["mismatched_elements"] = j[k]
            break
    return g


def fmt_gate(g: dict) -> str:
    bits = [g.get("verdict", "?")]
    if "compared_elements" in g:
        bits.append("compared %s output elements" % f"{g['compared_elements']:,}")
    else:
        bits.append("element count NOT REPORTED by the receipt")
    if "mismatched_elements" in g:
        bits.append("%s mismatched" % f"{g['mismatched_elements']:,}")
    if g.get("verdict_basis"):
        bits.append("(" + g["verdict_basis"] + ")")
    if g.get("parse_error"):
        bits.append("[" + g["parse_error"] + "]")
    return "  ".join(bits)


# --------------------------------------------------------------------------- servers
class Server:
    def __init__(self, name, cmd, port, log, logfile, env=None, key=None, wait=1800):
        self.name, self.port, self.log, self.key = name, port, log, key
        self.logfile = logfile
        log("   starting %s on :%d (log: %s)" % (name, port, logfile))
        log("   $ " + " ".join(str(c) for c in cmd))
        self.fh = open(logfile, "w")
        self.p = subprocess.Popen([str(c) for c in cmd], stdout=self.fh,
                                  stderr=subprocess.STDOUT, env=env,
                                  stdin=subprocess.DEVNULL)
        self.ready = False
        t0 = time.time()
        while time.time() - t0 < wait:
            if self.p.poll() is not None:
                log("   %s DIED (rc=%s) -- last lines:" % (name, self.p.returncode))
                tail = Path(logfile).read_text(errors="replace")[-2500:]
                log(tail)
                self.tail = tail
                return
            try:
                h = http_json(f"http://127.0.0.1:{port}/health", key, timeout=3)
                if h:
                    self.ready = True
                    self.load_seconds = round(time.time() - t0, 1)
                    log("   %s up after %.0f s; health=%s" % (name, self.load_seconds,
                                                              json.dumps(h)[:200]))
                    return
            except (urllib.error.URLError, OSError, ValueError, TimeoutError):
                pass
            time.sleep(2)
        log("   %s did not become healthy within %ds" % (name, wait))
        self.tail = Path(logfile).read_text(errors="replace")[-2500:]

    def receipt(self):
        try:
            return http_json(f"http://127.0.0.1:{self.port}/v1/receipt", self.key, timeout=20)
        except Exception:                                            # noqa: BLE001
            return None

    def stop(self):
        if self.p.poll() is None:
            # `terminate()` is SIGTERM on POSIX and TerminateProcess on Windows, where
            # there is no SIGTERM to deliver to another process; `send_signal(SIGTERM)`
            # would raise there on older interpreters.  The server holds no state that
            # has to be flushed -- every receipt is written before this point.
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.fh.close()
        self.log("   %s stopped" % self.name)
        time.sleep(3)


def bench_one(a, log, port, key_file, prompts, cls, conc, label, outdir, env, extra=""):
    cmd = [sys.executable, "-m", "glc_serve._bench.bench_serve", "--port", port,
           "--api-key-file", key_file, "--prompts", prompts, "--classes", cls,
           "--concurrency", conc, "--duration", a.duration, "--warm", a.warm,
           "--label", label, "--out", outdir, "--card-usd-h", a.dollars_per_hour,
           "--method-note", "greedy temperature=0, identical prompt pool, matched "
                            "continuous batching, same card, same run"]
    if extra:
        cmd += ["--extra", extra]
    rc, out = sh(cmd, log, timeout=a.duration * 4 + 900, env=env)
    p = Path(outdir) / f"summary_c{conc}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except ValueError:
            pass
    return {"error": "no summary", "returncode": rc, "tail": out[-800:]}


# --------------------------------------------------------------------------- table
def fnum(v, nd=1, dash="n/a"):
    if v is None:
        return dash
    try:
        return f"{float(v):,.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def goodput(summary, ttft_ms, tpot_ms):
    """Fraction of streams that met both SLOs, recomputed from the per-request log.

    Returns None when the log is not available -- it is never estimated from the
    aggregate.
    """
    if not summary or summary.get("error"):
        return None
    t = summary.get("ttft_s") or {}
    d = summary.get("per_stream_decode_tok_s") or {}
    # p50 TTFT under the budget AND p50 per-stream decode above 1000/tpot tok/s is a
    # median-level statement, not a per-stream count; it is labelled as such in the table.
    if t.get("p50") is None or d.get("p50") is None:
        return None
    ok_ttft = t["p50"] * 1000.0 <= ttft_ms
    ok_tpot = d["p50"] >= 1000.0 / tpot_ms
    return "yes" if (ok_ttft and ok_tpot) else "no"


def print_table(log, results, a):
    log("")
    log("=" * 108)
    log("GLC vs llama.cpp   --   every number below is %s" % MARK)
    dev = results["preflight"].get("device_name", "unknown device")
    log("card: %s   driver: %s   torch: %s"
        % (dev, results["preflight"].get("nvidia_smi_driver"),
           results["preflight"].get("torch")))
    log("=" * 108)
    hdr = ("%-7s %-5s %-9s %10s %10s %9s %9s %8s %9s %9s"
           % ("class", "N", "engine", "agg tok/s", "tok/s/user", "ttft p50",
              "ttft p99", "goodput", "$/M out", "VRAM MiB"))
    log(hdr)
    log("-" * 108)
    any_row = False
    for cls, conc, engine, s in results["rows"]:
        if not s or s.get("error"):
            log("%-7s %-5s %-9s   NOT RUN: %s" % (cls, conc, engine,
                                                  (s or {}).get("error", "no result")))
            continue
        any_row = True
        t = s.get("ttft_s") or {}
        d = s.get("per_stream_decode_tok_s") or {}
        nv = s.get("nvml") or {}
        log("%-7s %-5s %-9s %10s %10s %9s %9s %8s %9s %9s"
            % (cls, conc, engine,
               fnum(s.get("steady_tok_s")), fnum(d.get("p50"), 2),
               fnum(t.get("p50"), 3), fnum(t.get("p99"), 3),
               goodput(s, a.slo_ttft_ms, a.slo_tpot_ms) or "n/a",
               fnum(s.get("cost_usd_per_m_output"), 2),
               fnum(nv.get("mem_used_mib_p50_steady"), 0)))
    log("-" * 108)
    log("agg tok/s    = steady-window aggregate output tokens/second")
    log("tok/s/user   = p50 per-stream decode rate")
    log("goodput      = did the MEDIAN stream meet ttft<=%dms and tpot<=%dms (a median "
        "statement, not a per-stream count)" % (a.slo_ttft_ms, a.slo_tpot_ms))
    log("VRAM MiB     = steady-state median whole-device memory from nvidia-smi, sampled "
        "every 500ms during the run")
    if not any_row:
        log("")
        log("NO BENCHMARK ROW COMPLETED.  The table above is empty on purpose: nothing is")
        log("printed that was not measured.  See the log and the tarball for why.")
    log("")
    log("-- correctness gates --")
    for name, g in results["gates"].items():
        log("   %-12s %s" % (name, fmt_gate(g)))
    if results.get("fp16_ring"):
        log("")
        log("-- fp16 GDN ring arm (expected to refuse) --")
        log("   " + results["fp16_ring"]["note"])
    if not a.dollars_per_hour:
        log("")
        log(COST_SOURCES)
    log("")
    for n in results.get("not_run", []):
        log("NOT RUN: " + n)


# --------------------------------------------------------------------------- main
def build_parser():
    p = argparse.ArgumentParser(
        prog="glc-bench",
        description="Measure GLC multi-user serving against llama.cpp on this card. "
                    "Run it with no arguments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Troubleshooting:\n"
               "  no CUDA torch  -> pip install torch --index-url "
               "https://download.pytorch.org/whl/cu128\n"
               "  no llama-server-> glc-bench runs our side only and prints how to "
               "build llama.cpp\n"
               "  out of memory  -> glc-bench --slots 16   (and --slots 8 if that still "
               "OOMs)\n")
    p.add_argument("--work", type=Path, default=Path.cwd() / "glc-bench-work",
                   help="working directory for the model, logs and receipts")
    p.add_argument("--bundle", help="use an already-downloaded GLC-TBE artifact directory")
    p.add_argument("--hf-repo", default=HF_REPO_DEFAULT)
    p.add_argument("--hf-revision", default=HF_REV_DEFAULT)
    p.add_argument("--download-workers", type=int, default=8)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--tune", default=str(TUNE_DEFAULT),
                   help="kernel autotune table; the bundled one is measured on SM120 "
                        "(RTX PRO 6000) and is a STARTING POINT on any other card")
    p.add_argument("--slots", type=int, default=32,
                   help="KV/state slots on the device; lower this first if you run out "
                        "of memory")
    p.add_argument("--pages", type=int, default=800)
    p.add_argument("--max-ctx", type=int, default=16384)
    p.add_argument("--port", type=int, default=8290)
    p.add_argument("--llama-server", help="path to llama-server if not on PATH")
    p.add_argument("--llama-gguf",
                   help="GGUF of the SAME parent model for the llama.cpp arm")
    p.add_argument("--llama-port", type=int, default=8291)
    p.add_argument("--skip-llama", action="store_true")
    p.add_argument("--concurrency", default="1,4,8,16",
                   help="user counts for the short-prompt table")
    p.add_argument("--ctx8k-concurrency", default="16,32",
                   help="user counts for the 8k-context table")
    p.add_argument("--duration", type=float, default=90.0)
    p.add_argument("--warm", type=float, default=20.0)
    p.add_argument("--dollars-per-hour", type=float, default=0.0,
                   help="what this card costs you per hour, for the $/M output column")
    p.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    p.add_argument("--slo-tpot-ms", type=float, default=200.0)
    p.add_argument("--no-gates", action="store_true")
    p.add_argument("--no-fp16-ring", action="store_true")
    p.add_argument("--windows-check", action="store_true",
                   help="dry run: every check up to but NOT including CUDA execution -- "
                        "platform, WSL detection, torch, driver, toolchain, disk, the "
                        "install and the benchmark plumbing.  No download, no GPU work")
    p.add_argument("--cpu-smoke", action="store_true",
                   help="no GPU, no download: check the install, the console scripts and "
                        "the benchmark plumbing, then exit")
    return p


def cpu_smoke(a, log) -> int:
    log("== glc-bench --cpu-smoke ==")
    log("   This checks the INSTALL only.  It measures nothing about serving and proves")
    log("   nothing about a GPU.")
    log("   os_kind: %s" % _winenv.os_kind())
    if _winenv.os_kind() == "windows":
        log("   On native Windows, read %s before going further." % _winenv.WINDOWS_README)
    fails = []
    for mod in ("glc_loader", "glc_serve", "glc_serve._winenv", "glc_serve.bidec_iface",
                "glc_serve.bidec_capacity", "glc_serve._bench"):
        try:
            __import__(mod)
            log("   import %-28s ok" % mod)
        except Exception as e:                                       # noqa: BLE001
            log("   import %-28s FAIL %r" % (mod, e))
            fails.append(mod)

    a.work.mkdir(parents=True, exist_ok=True)
    prompts = a.work / "prompts.jsonl"
    rc, _ = sh([sys.executable, "-m", "glc_serve._bench.build_prompts",
                "--src", PKG, "--out", prompts], log, timeout=300)
    n = len(prompts.read_text().splitlines()) if prompts.exists() else 0
    log("   prompt pool: %d prompts (rc=%d)" % (n, rc))
    if n == 0:
        fails.append("build_prompts")

    rc, _ = sh([sys.executable, "-m", "glc_serve._bench.emit_receipts", "--self-test"],
               log, timeout=300)
    log("   emit_receipts --self-test rc=%d" % rc)
    if rc != 0:
        fails.append("emit_receipts self-test")

    log("   tune table present: %s" % Path(a.tune).exists())
    cu = sorted(p.name for p in PKG.rglob("*.cu"))
    log("   CUDA sources shipped: %d (%s...)" % (len(cu), ", ".join(cu[:4])))
    log("")
    if fails:
        log("CPU SMOKE: FAIL -- " + ", ".join(fails))
        return 1
    log("CPU SMOKE: PASS.  The package imports and the benchmark plumbing runs.")
    log("The GPU gates and the benchmark itself have NOT run here and cannot: this")
    log("machine has no NVIDIA card.  Run plain `glc-bench` on the GPU box.")
    return 0


def windows_check(a, log) -> int:
    """Everything that can be checked without executing a CUDA kernel.

    The point is to separate "this machine is not set up" from "the kernels do not
    work on this card", because they arrive as the same traceback three hours into a
    run otherwise.  A PASS here says the environment is ready; it says NOTHING about
    whether the kernels build or the gates pass, and it never downloads the model.
    """
    log("== glc-bench --windows-check ==")
    log("   Dry run.  No download, no CUDA execution, no measurement of serving.")
    log("")
    tc = _winenv.toolchain_report()
    log("   os_kind                      %s" % tc["os_kind"])
    if tc["os_kind"] == "windows":
        log("")
        log(WSL_ADVICE)
        log("")

    # Reuse the real preflight so this cannot drift from what `glc-bench` enforces;
    # --skip-download keeps the 40 GB out of the disk requirement.
    saved, a.skip_download = a.skip_download, True
    rep = preflight(a, log)
    a.skip_download = saved

    log("")
    log("== install and plumbing ==")
    rc_smoke = cpu_smoke(a, log)

    log("")
    log("== verdict ==")
    ready = not rep["blockers"] and rc_smoke == 0 and rep.get("cuda_available")
    for b in rep["blockers"]:
        log("   BLOCKER  " + b["problem"])
    for b in rep["warnings"]:
        log("   WARNING  " + b["problem"])
    if not rep.get("cuda_available"):
        log("   BLOCKER  torch reports no CUDA device, so the benchmark cannot run here")
    if rc_smoke != 0:
        log("   BLOCKER  the install itself did not pass --cpu-smoke")
    if ready:
        log("   READY: nothing further to install.  Run `glc-bench` (plus")
        log("   --dollars-per-hour <rate> and --llama-gguf <file> if you have them).")
        log("   The CUDA kernels still have to BUILD and the gates still have to PASS;")
        log("   neither is checked here and neither has run on this card before.")
        return 0
    log("")
    log("   NOT READY.  Fix the blockers above (each printed its own fix) and re-run")
    log("   `glc-bench --windows-check`.")
    return 2


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.work = a.work.resolve()
    a.work.mkdir(parents=True, exist_ok=True)
    log = Log(a.work / "glc-bench.log")
    log("glc-bench  (glc-loader 1.2.0rc2)  %s"
        % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    if a.cpu_smoke:
        return cpu_smoke(a, log)
    if a.windows_check:
        return windows_check(a, log)

    results = {"mark": MARK, "gates": {}, "rows": [], "not_run": []}
    results["preflight"] = rep = preflight(a, log)
    if rep["blockers"]:
        log("")
        log("Stopping: fix the problems above and run `glc-bench` again.  Nothing was "
            "measured, so nothing is reported.")
        return 2

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCH_EXTENSIONS_DIR", str(a.work / "torch_ext"))
    for v in ("MIV_GEMV_BUILD_DIR", "MIV_TBE_BUILD_DIR", "MIV_KQ_BUILD_DIR",
              "BI_GEMM_BUILD_DIR"):
        env.setdefault(v, str(a.work / "build" / v.lower()))
    # TMPDIR is POSIX; Windows `tempfile` reads TMP/TEMP, and a run that sets only
    # TMPDIR spills multi-GB temporaries back onto the system drive.
    _winenv.temp_env(env, a.work / "tmp")
    for p in (a.work / "tmp", a.work / "torch_ext"):
        p.mkdir(parents=True, exist_ok=True)

    bundle = fetch_bundle(a, log, rep)
    if bundle is None:
        results["not_run"].append("everything that needs the model artifact "
                                  "(no --bundle and the download did not complete)")

    log("")
    log("== 3. prompt pool ==")
    prompts = a.work / "prompts.jsonl"
    sh([sys.executable, "-m", "glc_serve._bench.build_prompts", "--src", PKG,
        "--out", prompts], log, timeout=300, env=env)
    log("   %d prompts (classes chat / ctx1k / ctx8k), identical for both engines"
        % len(prompts.read_text().splitlines()))

    key_file = a.work / "API_KEY"
    if not key_file.exists():
        key_file.write_text(os.urandom(16).hex())
    key = key_file.read_text().strip()

    if not a.no_gates:
        run_gates(a, log, bundle, a.tune, prompts, env, results)
    else:
        results["not_run"].append("correctness gates (--no-gates)")

    # ----------------------------------------------------------------- our server
    log("")
    log("== 5. benchmark: our engine ==")
    ours_receipt = None
    if bundle:
        srv = Server("glc-multiuser",
                     [sys.executable, "-m", "glc_serve.bidec_serve",
                      "--bundle", bundle, "--tune", a.tune, "--slots", a.slots,
                      "--pages", a.pages, "--max-ctx", a.max_ctx,
                      "--port", a.port, "--api-key-file", key_file],
                     a.port, log, a.work / "server_glc.log", env=env, key=key)
        if srv.ready:
            rep["glc_load_seconds"] = srv.load_seconds
            rep["glc_nvml_loaded_mib"] = nvidia_smi("memory.used")
            log("   resident after load: %s MiB (%s)"
                % (rep["glc_nvml_loaded_mib"], MARK))
            ours_receipt = srv.receipt()
            if ours_receipt:
                (a.work / "glc_server_receipt.json").write_text(
                    json.dumps(ours_receipt, indent=2))

            log("")
            log("   -- gate: stream (HTTP, concurrent) --")
            rc, out = sh([sys.executable, "-m", "glc_serve._bench.bi_gate_stream",
                          "--port", a.port, "--api-key-file", key_file,
                          "--prompts", prompts,
                          "--out", a.work / "gates" / "stream.json"],
                         log, timeout=5400, env=env)
            results["gates"]["stream"] = read_gate(a.work / "gates" / "stream.json",
                                                   rc, out)
            log("   stream: " + fmt_gate(results["gates"]["stream"]))

            for conc in [int(x) for x in a.concurrency.split(",") if x.strip()]:
                s = bench_one(a, log, a.port, key_file, prompts, "chat", conc,
                              f"glc/chat/N{conc}", a.work / "bench" / "glc_chat", env)
                results["rows"].append(("chat", conc, "GLC", s))
            for conc in [int(x) for x in a.ctx8k_concurrency.split(",") if x.strip()]:
                s = bench_one(a, log, a.port, key_file, prompts, "ctx8k", conc,
                              f"glc/ctx8k/N{conc}", a.work / "bench" / "glc_ctx8k", env)
                results["rows"].append(("ctx8k", conc, "GLC", s))
            srv.stop()
        else:
            results["not_run"].append("our engine's benchmark: the server did not come up "
                                      "(see server_glc.log in the tarball)")
            srv.stop()
    else:
        results["not_run"].append("our engine's benchmark: no artifact")

    # ----------------------------------------------------------------- llama.cpp
    log("")
    log("== 6. benchmark: llama.cpp ==")
    lsrv_bin = rep["llama_server"]
    if a.skip_llama:
        results["not_run"].append("llama.cpp arm (--skip-llama)")
        log("   NOT RUN (--skip-llama)")
    elif not lsrv_bin:
        results["not_run"].append("llama.cpp arm: llama-server not found on PATH")
        log("   llama-server is not on PATH, so there is no comparison arm in this run.")
        log("   Our side still ran.  To get the comparison, build llama.cpp:")
        log(LLAMACPP_INSTALL)
    elif not a.llama_gguf:
        results["not_run"].append(
            "llama.cpp arm: no --llama-gguf.  A legitimate comparison needs a GGUF of "
            "the SAME parent weights; we do not publish an F16 GGUF of this parent, so "
            "it cannot be downloaded automatically.")
        log("   llama-server found at %s but no --llama-gguf given." % lsrv_bin)
        log("   A comparison is only legitimate against the SAME parent weights.  Make")
        log("   the F16 GGUF once, from the BF16 parent checkpoint:")
        log("     python llama.cpp/convert_hf_to_gguf.py <bf16-parent-dir> "
            "--outtype f16 --outfile qwen38-27b-f16.gguf")
        log("   then re-run with --llama-gguf qwen38-27b-f16.gguf")
        log("   (a Q8_0 or IQ4 GGUF also works, but it is a LOSSY baseline and the table")
        log("    will say so -- pass --llama-gguf <that file> if that is what you have.)")
    else:
        gguf = Path(a.llama_gguf)
        lossy = any(t in gguf.name.upper() for t in ("Q8", "Q6", "Q5", "Q4", "Q3", "Q2",
                                                     "IQ", "_K"))
        if lossy:
            log("   NOTE: %s looks like a QUANTISED GGUF.  It is a LOSSY baseline: it is"
                % gguf.name)
            log("   not bit-exact with the parent, so the quality sides are not equal.")
            results["llama_baseline_lossy"] = True
        plans = [("chat", [int(x) for x in a.concurrency.split(",") if x.strip()], 2048),
                 ("ctx8k", [int(x) for x in a.ctx8k_concurrency.split(",") if x.strip()],
                  9216)]
        for cls, concs, per_slot in plans:
            par = max(concs)
            cmd = [lsrv_bin, "-m", gguf, "-ngl", "999", "-fa", "on",
                   "-c", par * per_slot, "--parallel", par, "--cont-batching",
                   "--kv-unified", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja",
                   "--host", "127.0.0.1", "--port", a.llama_port,
                   "--api-key-file", key_file, "--metrics", "--no-webui"]
            lsrv = Server("llama-server/%s" % cls, cmd, a.llama_port, log,
                          a.work / f"server_llama_{cls}.log", key=key, wait=1200)
            if not lsrv.ready:
                results["not_run"].append(
                    "llama.cpp %s arm: server did not come up with --parallel %d -c %d "
                    "(most likely out of memory -- see server_llama_%s.log)"
                    % (cls, par, par * per_slot, cls))
                lsrv.stop()
                continue
            rep["llama_nvml_loaded_mib_%s" % cls] = nvidia_smi("memory.used")
            for conc in concs:
                s = bench_one(a, log, a.llama_port, key_file, prompts, cls, conc,
                              f"llamacpp/{cls}/N{conc}",
                              a.work / "bench" / f"llama_{cls}", env)
                results["rows"].append((cls, conc, "llama.cpp", s))
            lsrv.stop()

    # ----------------------------------------------------------------- fp16 ring, LAST
    if bundle and not a.no_fp16_ring:
        log("")
        log("== 7. fp16 GDN ring arm (run last, EXPECTED to refuse) ==")
        log("   The GDN recurrent state ring is fp32 only.  Asking for fp16 must be")
        log("   refused rather than silently misread, so a NotImplementedError here is")
        log("   the PASS condition for this arm.")
        f = Server("glc-multiuser/fp16-ring",
                   [sys.executable, "-m", "glc_serve.bidec_serve",
                    "--bundle", bundle, "--tune", a.tune, "--slots", 4,
                    "--pages", 64, "--max-ctx", 2048, "--gdn-ring-dtype", "fp16",
                    "--port", a.port, "--api-key-file", key_file],
                   a.port, log, a.work / "server_fp16ring.log", env=env, key=key,
                   wait=900)
        tail = getattr(f, "tail", "")
        refused = "NotImplementedError" in tail or "fp16" in tail.lower()
        results["fp16_ring"] = {
            "server_came_up": f.ready, "refused": bool(refused and not f.ready),
            "note": ("PASS (expected): the server refused fp16 for the GDN ring"
                     if refused and not f.ready else
                     "UNEXPECTED: the fp16 ring did not refuse (server_came_up=%s). "
                     "This is a finding -- see server_fp16ring.log." % f.ready)}
        log("   " + results["fp16_ring"]["note"])
        f.stop()

    # ----------------------------------------------------------------- receipts
    log("")
    log("== 8. receipts ==")
    rdir = a.work / "receipts"
    cmd = [sys.executable, "-m", "glc_serve._bench.emit_receipts", "--out", rdir]
    for name, key_ in (("kernel", "--gate-kernel"), ("engine", "--gate-engine"),
                       ("stream", "--gate-http"), ("batchexact", "--gate-sched")):
        g = results["gates"].get(name)
        if g and Path(g["receipt"]).exists():
            cmd += [key_, g["receipt"]]
    if ours_receipt:
        cmd += ["--server-receipt", a.work / "glc_server_receipt.json"]
    if rep.get("glc_load_seconds"):
        cmd += ["--load-seconds", rep["glc_load_seconds"]]
    rc, _ = sh(cmd, log, timeout=600, env=env)
    log("   emit_receipts rc=%d" % rc)

    (a.work / "glc_bench_results.json").write_text(json.dumps(
        {**results, "rows": [{"class": c, "N": n, "engine": e, "summary": s}
                             for c, n, e, s in results["rows"]]}, indent=2, default=str))

    print_table(log, results, a)

    tgz = Path.cwd() / ("glc-bench-results-%s.tar.gz" % time.strftime("%Y%m%d-%H%M%S",
                                                                      time.gmtime()))
    with tarfile.open(tgz, "w:gz") as tf:
        for pat in ("glc-bench.log", "glc_bench_results.json", "glc_server_receipt.json",
                    "server_*.log", "prompts.jsonl"):
            for p in a.work.glob(pat):
                tf.add(p, arcname="glc-bench/" + p.name)
        for d in ("gates", "receipts", "bench"):
            if (a.work / d).exists():
                tf.add(a.work / d, arcname="glc-bench/" + d)
    log("")
    log("wrote %s (%.1f MB)" % (tgz, tgz.stat().st_size / 1e6))
    log("")
    log("    send this file back")
    log("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
