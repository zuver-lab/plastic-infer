"""cudaMemcpyBatchAsync host binding for the prefill hit-D2D misses (de-tvm_ffi
port of FreeToken's ``kernel/batch_memcpy.py``).

FreeToken wraps cudaMemcpyBatchAsync in a tvm_ffi module; here the same host
wrapper is compiled with the standalone nvcc JIT and the call becomes ctypes.
There is no device code -- the .cuh is a pure host function. ``load_batch_memcpy``
builds once, runs one real 16-byte H2D probe (so a driver without batch-memcpy
support surfaces as an exception the caller's fallback can catch), and returns a
callable matching FreeToken's ``fn(dst_ptrs, src_ptrs, sizes, stream)`` signature.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache

import torch

from .jit import KERNEL_SRC_DIR, compile_cuda_source


def _cuh_include() -> str:
    return f'#include "{KERNEL_SRC_DIR / "batch_memcpy.cuh"}"'


@lru_cache(maxsize=1)
def _batch_memcpy_module() -> ctypes.CDLL:
    source = f"""{_cuh_include()}

extern "C" void plasticinfer_batch_memcpy(
    const int64_t* dst_ptrs, const int64_t* src_ptrs, const int64_t* sizes,
    int64_t n, cudaStream_t stream) {{
    BatchMemcpy::run(
        reinterpret_cast<const void* const*>(dst_ptrs),
        reinterpret_cast<const void* const*>(src_ptrs),
        reinterpret_cast<const std::size_t*>(sizes),
        static_cast<std::size_t>(n),
        stream);
}}
"""
    lib = compile_cuda_source(source, name="batch_memcpy")
    lib.plasticinfer_batch_memcpy.argtypes = [
        ctypes.c_void_p,  # dst_ptrs
        ctypes.c_void_p,  # src_ptrs
        ctypes.c_void_p,  # sizes
        ctypes.c_int64,  # n
        ctypes.c_void_p,  # cudaStream_t
    ]
    return lib


def _probe(fn) -> None:
    """Two real 16-byte H2D copies through the binding. A version-gated build (panic
    branch) or a driver without batch-memcpy support loads cleanly and only fails
    at call time; probing here turns every such mode into a load_batch_memcpy
    exception the caller's fallback path can catch. Two copies because some drivers
    honor the first batch and silently drop subsequent ones -- one copy would pass
    a probe that the runtime calls then fail (WSL2/CUDA 13.0 observed this)."""
    src = torch.arange(16, dtype=torch.uint8).pin_memory()
    for _ in range(2):
        dst = torch.zeros(16, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.Stream()
        fn(
            torch.tensor([dst.data_ptr()]),
            torch.tensor([src.data_ptr()]),
            torch.tensor([16]),
            stream.cuda_stream,
        )
        stream.synchronize()
        if not torch.equal(dst.cpu(), src):
            raise RuntimeError("cudaMemcpyBatchAsync probe copied wrong bytes")


def load_batch_memcpy():
    """Build (once), probe, and return the batch-memcpy entry point, or raise.

    The 8-argument cudaMemcpyBatchAsync signature this binding uses is CUDA 13.0's
    (12.8/12.9 had an extra failIdx parameter); gate on the torch runtime version
    before paying for the JIT build, then verify with a real copy.
    """
    cuda = torch.version.cuda
    if cuda is None or tuple(int(x) for x in cuda.split(".")[:2]) < (13, 0):
        raise RuntimeError(
            f"cudaMemcpyBatchAsync binding requires CUDA >= 13.0 (torch built with {cuda})"
        )
    launch = _batch_memcpy_module().plasticinfer_batch_memcpy

    def fn(dst_ptrs: torch.Tensor, src_ptrs: torch.Tensor, sizes: torch.Tensor, stream: int) -> None:
        """Enqueue one cudaMemcpyBatchAsync of ``len(sizes)`` independent copies.

        ``dst_ptrs``/``src_ptrs``/``sizes`` are same-length CPU int64 tensors of raw
        addresses and byte counts; ``stream`` is a raw cudaStream_t handle
        (``torch.cuda.Stream.cuda_stream``), which must not be the legacy NULL stream.
        """
        launch(
            dst_ptrs.data_ptr(),
            src_ptrs.data_ptr(),
            sizes.data_ptr(),
            dst_ptrs.numel(),
            stream,
        )

    _probe(fn)
    return fn


def batch_memcpy_jit(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    sizes: torch.Tensor,
    stream: int,
) -> None:
    """Enqueue one cudaMemcpyBatchAsync of ``len(sizes)`` independent copies."""
    load_batch_memcpy()(dst_ptrs, src_ptrs, sizes, stream)
