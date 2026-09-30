"""Standalone nvcc JIT for the ported CUDA kernels (de-tvm_ffi of FreeToken's
kernel JIT loader).

FreeToken compiles its JIT kernels through tvm_ffi's ``load_inline``, which is
not available here; torch's ``cpp_extension`` cannot find the WSL toolkit's
``cuda_runtime.h`` (CUDA 13.2 lives off torch's include path). So the same
shape -- write a generated TU, invoke nvcc, dlopen the result -- implemented
directly:

  * nvcc compiles one generated ``.cu`` per (template-args) combo; the template
    is instantiated in the TU and exported as an ``extern "C"`` C-ABI entry
    point (tvm_ffi's ``TVM_FFI_DLL_EXPORT_TYPED_FUNC`` replacement).
  * The ``.so`` is cached under ``~/.cache/plastic_infer/kernels``, keyed by the
    source hash + nvcc version + target arch, and dlopen'd via ctypes.
  * Kernels launch on an explicit stream (torch's current stream), so they
    participate in CUDA graph capture the same way tvm_ffi launches did.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import shutil
import subprocess
from functools import lru_cache

# Where the copied kernel sources live (kernel/csrc/jit/*.cuh).
KERNEL_SRC_DIR = pathlib.Path(__file__).parent / "csrc" / "jit"

_CACHE_DIR_ENV = "PLASTIC_INFER_KERNEL_CACHE_DIR"
_DEFAULT_CUDA_CFLAGS = ["-std=c++17", "-O3", "--expt-relaxed-constexpr"]


def kernel_cache_dir() -> pathlib.Path:
    if env := os.environ.get(_CACHE_DIR_ENV):
        return pathlib.Path(env).expanduser()
    return pathlib.Path.home() / ".cache" / "plastic_infer" / "kernels"


def _nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError(
            "nvcc not found on PATH: the offload-MoE index-copy kernel is built "
            "with standalone nvcc JIT (torch cpp_extension cannot see the WSL "
            "toolkit's include paths)"
        )
    return nvcc


def _nvcc_version_stamp(nvcc: str) -> str:
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "release" in line:
            return line.strip()
    return out.strip()[:64]


def _cuda_arch_flags() -> list[str]:
    """Target the local GPU's SASS arch (the JIT runs where the kernels run)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("kernel JIT requires a CUDA device (SASS arch target)")
    major, minor = torch.cuda.get_device_capability()
    cc = f"{major}{minor}"
    return [f"-gencode=arch=compute_{cc},code=sm_{cc}"]


def _included_sources_material() -> bytes:
    """Bytes of every ``*.cuh`` under ``KERNEL_SRC_DIR`` (folded into the cache key).

    Generated TUs ``#include`` the kernels by path, so the TU text alone does not
    change when a ``.cuh`` is edited -- without folding the included sources into
    the key, an edit would silently reuse a stale cached ``.so``.
    """
    parts = []
    for path in sorted(KERNEL_SRC_DIR.glob("*.cuh")):
        parts.append(b"\0--- %s ---\0" % str(path).encode())
        parts.append(path.read_bytes())
    return b"".join(parts)


@lru_cache(maxsize=64)
def compile_cuda_source(source: str, *, name: str) -> ctypes.CDLL:
    """JIT-compile ``source`` (a complete, self-contained ``.cu`` TU) and dlopen it.

    Cached on the source text, the included ``.cuh`` bytes, the nvcc version, and
    the target arch: the same template combo rebuilds only once per (machine,
    toolchain), and a source/toolchain change gets a fresh key instead of a stale
    ``.so``.
    """
    import torch

    nvcc = _nvcc()
    key = hashlib.sha256(
        f"{source}|{_included_sources_material()}|{_nvcc_version_stamp(nvcc)}|"
        f"{torch.version.cuda}|{torch.cuda.get_device_capability()}".encode()
    ).hexdigest()[:24]
    cache_dir = kernel_cache_dir() / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    so_path = cache_dir / f"{name}.so"
    if not so_path.exists():
        cu_path = cache_dir / f"{name}.cu"
        cu_path.write_text(source)
        cmd = [
            nvcc,
            *_DEFAULT_CUDA_CFLAGS,
            "-shared",
            "-Xcompiler", "-fPIC",
            *_cuda_arch_flags(),
            "-o", str(so_path),
            str(cu_path),
            "-lcudart",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            so_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"nvcc JIT failed for {name}:\n{proc.stderr[-4000:]}"
            )
    return ctypes.CDLL(str(so_path))
