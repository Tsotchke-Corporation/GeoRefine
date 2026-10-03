# Running `glc-bench` on a Windows machine

You have an NVIDIA card in a Windows box. This page is the whole procedure.

**Use WSL2.** Not because Windows is unsupported in principle, but because of what
this package actually is: the CUDA kernels are compiled on your machine the first
time they run, and `triton` (needed by one optional backend) ships no Windows wheel
at all. On WSL2 those are Linux builds — the same builds this engine was written and
measured on. On native Windows they are MSVC builds that have never been made here.
WSL2 uses the NVIDIA **Windows** driver you already have: there is no second driver
to install and no CUDA download for PyTorch.

There is a native-Windows section at the bottom, with honest caveats.

Total time: about 20 minutes of typing and waiting, then 1–3 hours for the
benchmark, most of which is a 40 GB model download.

---

## Step 1 — install Ubuntu (PowerShell, once)

Open PowerShell **as Administrator** (Start → type `powershell` → right-click → Run
as administrator) and run exactly this:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot when it asks. After the reboot, open **Ubuntu 24.04** from the Start menu. It
will ask you to pick a username and password once — any value is fine, and the
password is for `sudo` inside Ubuntu only.

Everything from here on is typed in that **Ubuntu** window, not PowerShell.

Check the card is visible:

```bash
nvidia-smi
```

You should see your GPU, with a driver version. If you do not, see
[Something went wrong](#something-went-wrong) below.

---

## Step 2 — install the benchmark (Ubuntu, 3 lines)

```bash
sudo apt-get update && sudo apt-get install -y python3-venv build-essential ninja-build nvidia-cuda-toolkit
python3 -m venv ~/glc && source ~/glc/bin/activate && pip install -U pip
pip install "glc-loader[cuda] @ https://github.com/Tsotchke-Corporation/GeoRefine/releases/download/v1.2.0-rc2/glc_loader-1.2.0rc2-py3-none-any.whl"
```

The first line is the compiler and `nvcc`, which the kernels are built with.
`nvidia-cuda-toolkit` from Ubuntu is enough — **do not** install an NVIDIA *Linux*
driver inside WSL; that breaks the GPU passthrough. The second line makes an
isolated Python environment. The third installs the wheel and PyTorch.

> If you need a newer `nvcc` than Ubuntu ships (only if the build complains about
> your card's architecture being unsupported), use NVIDIA's WSL repository instead
> of `nvidia-cuda-toolkit`:
>
> ```bash
> wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
> sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
> sudo apt-get install -y cuda-toolkit-12-6
> echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
> ```

Now check the environment before committing to the download:

```bash
glc-bench --windows-check
```

This runs **every** check except actually executing a CUDA kernel: platform, WSL
detection, torch, driver, `nvcc`, host compiler, free disk, the install itself and
the benchmark plumbing. It downloads nothing. Anything missing is printed with the
one command that fixes it. Keep going when it says `READY`.

---

## Step 3 — run it

```bash
glc-bench --dollars-per-hour 1.23
```

Replace `1.23` with what the card costs you per hour — that is the only number
`glc-bench` cannot measure, and it fills in the `$/M output tokens` column. If the
card is your own and you have no hourly rate, leave the flag off; every other column
is still measured and the two engines' ratio is unaffected.

PowerShell note, if you ever run this from Windows rather than Ubuntu: a bare `$` is
a variable sigil in PowerShell, so write the flag with no `$` in it, as above
(`--dollars-per-hour 1.23`), and quote any path containing spaces.

### Disk: read this before starting

The model artifact is about **40 GB**, and `glc-bench` wants roughly **45 GB** free
with logs and build output. By default it downloads into `./glc-bench-work` in
whatever directory you are in — inside WSL that is on your `C:` drive.

If `C:` does not have 45 GB spare, point it at another drive. Your Windows drives
are mounted under `/mnt` inside Ubuntu, so `D:` is `/mnt/d`:

```bash
glc-bench --dollars-per-hour 1.23 --work /mnt/d/glc-bench-work
```

`glc-bench` prints the size and the free space before it downloads anything, and
stops with this suggestion rather than filling your system drive.

### The llama.cpp comparison arm

Without `llama-server`, `glc-bench` measures our side only and says so. For the
comparison it needs `llama-server` **inside WSL** (it starts and stops the process
itself and talks to it over loopback) and a GGUF of the **same** parent weights.

Build it — you already installed everything it needs in Step 2:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build --config Release -j --target llama-server
```

(The prebuilt CUDA zips that ggml-org publishes are `llama-server.exe` for *native*
Windows. `glc-bench` cannot drive a Windows process from inside WSL, so for this
benchmark build the Linux binary as above.)

Then the GGUF. We publish the compressed artifact, not a GGUF, and a comparison is
only honest against the same weights, so this is made once from the BF16 parent
checkpoint:

```bash
pip install -r ~/llama.cpp/requirements.txt
python ~/llama.cpp/convert_hf_to_gguf.py <bf16-parent-dir> --outtype f16 --outfile /mnt/d/qwen38-27b-f16.gguf
```

A `Q8_0` or `IQ4` GGUF also runs, but it is a **lossy** baseline — the quality sides
are then not equal, and the table says so. Full run with both arms:

```bash
glc-bench --dollars-per-hour 1.23 \
  --work /mnt/d/glc-bench-work \
  --llama-server ~/llama.cpp/build/bin/llama-server \
  --llama-gguf /mnt/d/qwen38-27b-f16.gguf
```

---

## Step 4 — send the tarball

`glc-bench` ends by writing `glc-bench-results-<date>.tar.gz` in the directory you
ran it from. **Send that file back.** It has the log, the gate receipts, the
per-request benchmark records and the environment block. Nothing in it is a number
we filled in from elsewhere; everything measured is tagged `MEASURED-ON-THIS-CARD`.

A `FAIL` on a gate is a real result and we want the tarball either way. The four GPU
gates in this release have never run on a GPU before — your card is where they first
execute.

To get the file into Windows so you can attach it, copy it to your Windows desktop:

```bash
cp glc-bench-results-*.tar.gz /mnt/c/Users/<your-windows-username>/Desktop/
```

---

## Something went wrong

The four things that actually happen, with the one-line fix for each:

| What you see | Fix |
|---|---|
| `nvidia-smi` not found, or no GPU listed, inside Ubuntu | The GPU is not passed through. In PowerShell as Administrator: `wsl --update`, then `wsl --set-version Ubuntu-24.04 2`, then install the current NVIDIA **Windows** driver from nvidia.com. Never install a Linux driver inside WSL. |
| `torch is installed but reports no CUDA device` | CPU-only PyTorch wheel. `pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| Out of memory, or the server dies while loading | Fewer concurrent slots: `glc-bench --slots 16`, and `--slots 8` if that still fails. |
| `no --llama-gguf` / `llama-server is not on PATH` | The run continues with our engine only. For the comparison arm, follow [the llama.cpp section](#the-llamacpp-comparison-arm). |

For anything else: run `glc-bench --windows-check`, and send us that output. It is
short and it names what is missing.

---

## Native Windows, without WSL2

This works for installing and inspecting, and is **unverified** for the benchmark.

```powershell
py -m venv %USERPROFILE%\glc
%USERPROFILE%\glc\Scripts\activate
pip install -U pip
pip install "glc-loader[cuda] @ https://github.com/Tsotchke-Corporation/GeoRefine/releases/download/v1.2.0-rc2/glc_loader-1.2.0rc2-py3-none-any.whl"
pip install torch --index-url https://download.pytorch.org/whl/cu128
glc-bench --windows-check
```

The double quotes around the requirement are required in both PowerShell and `cmd`:
unquoted, the `[cuda]` is a wildcard pattern to PowerShell. As of 1.2.0rc2 the
`[cuda]` extra no longer pulls `triton` on Windows (it has no Windows wheel on
PyPI), so the install itself succeeds — `triton` is only used by an optional
`glc_loader` backend that this benchmark never calls.

What is known, honestly:

| | WSL2 (Ubuntu 24.04) | Native Windows |
|---|---|---|
| wheel install, console scripts, `--cpu-smoke` | works | works; checked in CI on `windows-latest` |
| `--windows-check` | works | works |
| PyTorch CUDA wheel | standard Linux wheel | standard Windows wheel |
| `triton` (optional FWP1 backend only) | PyPI wheel | no PyPI wheel; community `triton-windows` exists, untested here |
| JIT CUDA kernel build (the benchmark needs this) | g++ + `nvcc`, as measured | MSVC `cl.exe` + `nvcc`; **never built here** |
| the gates and the benchmark | the supported path | unverified |

If you do try it, you need the full CUDA Toolkit for Windows and Visual Studio Build
Tools with the "Desktop development with C++" workload, and you must run from the
"x64 Native Tools Command Prompt for VS" so `cl.exe` is on `PATH`. 1.2.0rc2 passes
MSVC's `/O2` rather than GCC's `-O3` to the host compiler, which removes one known
failure, but the build as a whole is untested. Long paths are the other known
hazard: keep `--work` short (`--work D:\glc`) and enable Win32 long paths
(`New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`).

We would rather have a WSL2 result than a native-Windows debugging session. If
native Windows is the only option, say so and we will work the kernel build with
you.
