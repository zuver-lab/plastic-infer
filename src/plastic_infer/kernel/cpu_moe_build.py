"""Standalone build of the ``_cpu_moe`` C++ CPU-MoE executor extension (M2).

FreeToken builds ``freetoken.kernel._cpu_moe`` through setuptools + torch's
``CppExtension`` (g++ + pybind11 headers + ``-lcudart``, ``-O3 -std=c++17
-pthread``). This repo has no setup.py build step, so the same compilation is
reproduced here as a first-use JIT, matching the M1b kernel JIT's
build-on-demand philosophy: torch's ``cpp_extension`` supplies the torch /
bundled-pybind11 headers and the correct C++ ABI flags, we add the CUDA
include/lib dirs exactly as FreeToken's setup.py does, and the resulting
``.so`` is placed next to this package as ``kernel/_cpu_moe.so``. The
``kernel/_cpu_moe.py`` shim makes the first ``from ..kernel import _cpu_moe``
build it transparently; once the ``.so`` exists, Python's extension loader
imports it natively and the shim is never touched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
from pathlib import Path

_KERNEL_DIR = Path(__file__).parent
_SRC = _KERNEL_DIR / "csrc" / "cpu_moe" / "cpu_moe_ext.cpp"
_SO = _KERNEL_DIR / "_cpu_moe.so"
# Source hash stamp: an edited .cpp rebuilds, an untouched checkout does not.
_STAMP = _KERNEL_DIR / "_cpu_moe.build_stamp"


def _source_hash() -> str:
    return hashlib.sha256(_SRC.read_bytes()).hexdigest()[:24]


def _cuda_dirs() -> tuple[list[str], list[str]]:
    """(include_dirs, library_dirs) for the CUDA runtime, as FreeToken's setup does."""
    from torch.utils.cpp_extension import CUDA_HOME

    home = Path(os.environ.get("CUDA_HOME") or CUDA_HOME)
    if not home.exists():
        raise RuntimeError(
            "CUDA_HOME is required to build _cpu_moe (it links cuda_runtime_api.h "
            "for the cudaLaunchHostFunc submit/sync graph nodes)"
        )
    library_dirs = [str(home / "lib64")]
    if (home / "lib").exists():
        library_dirs.append(str(home / "lib"))
    return [str(home / "include")], library_dirs


def _load_from_so(so_path: Path):
    """importlib-load a compiled extension module straight from its file path."""
    spec = importlib.util.spec_from_file_location(
        "plastic_infer.kernel._cpu_moe", so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(*, force: bool = False):
    """Compile ``_cpu_moe`` into ``kernel/_cpu_moe.so`` (no-op if fresh).

    Returns the loaded module (imported from the built ``.so``).
    """
    import torch
    from torch.utils.cpp_extension import load

    stamp = _STAMP.read_text() if _STAMP.exists() else ""
    if not force and _SO.exists() and stamp == _source_hash():
        return _load_from_so(_SO)

    cuda_include_dirs, cuda_library_dirs = _cuda_dirs()
    build_dir = Path.home() / ".cache" / "plastic_infer" / "kernels" / \
        f"cpu_moe_{_source_hash()}"
    build_dir.mkdir(parents=True, exist_ok=True)
    # The same geometry FreeToken's setup.py passes to CppExtension: explicit
    # cuda include/lib dirs + cudart, -O3 -std=c++17 -pthread. torch resolves
    # the torch/pybind11 headers and the CXX11 ABI flag; load() has no
    # libraries/library_dirs kwargs, so cudart comes in via the link flags.
    load(
        name="_cpu_moe",
        sources=[str(_SRC)],
        build_directory=str(build_dir),
        extra_include_paths=cuda_include_dirs,
        extra_cflags=["-O3", "-std=c++17", "-pthread"],
        extra_ldflags=["-pthread", "-lcudart",
                       *[f"-L{d}" for d in cuda_library_dirs]],
        with_cuda=True,
        verbose=True,
    )
    built = build_dir / "_cpu_moe.so"
    if not built.exists():
        raise RuntimeError(f"_cpu_moe build produced no {built}")
    shutil.copy2(built, _SO)
    _STAMP.write_text(_source_hash())
    # Drop the module torch's load() cached so the native-import path below is
    # the one everyone sees (identical module, single source of truth).
    if "plastic_infer.kernel._cpu_moe" in __import__("sys").modules:
        del __import__("sys").modules["plastic_infer.kernel._cpu_moe"]
    return _load_from_so(_SO)


def ensure_built():
    """Build if missing or stale; return the loaded ``_cpu_moe`` module."""
    if not _SO.exists() or _STAMP.read_text() != _source_hash():
        return build()
    return _load_from_so(_SO)


if __name__ == "__main__":
    m = build(force="--force" in sys.argv)
    print(f"built _cpu_moe -> {_SO} (isa={m is not None})")
