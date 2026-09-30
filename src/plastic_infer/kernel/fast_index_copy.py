"""Index-copy kernels for the MoE slot cache (de-tvm_ffi port of
FreeToken's ``kernel/fast_index_copy.py``).

The CUDA kernels live in ``kernel/csrc/jit/fast_index_copy.cuh`` (copied
verbatim from FreeToken, host glue rewritten to a C ABI); this module is the
Python half -- the same functions, the same template-arg resolution, but
tvm_ffi's ``load_jit``/``Module`` replaced by the standalone nvcc JIT in
:mod:`kernel.jit` and the module calls become ctypes calls. The entry points
accept raw data pointers and launch on torch's current CUDA stream.
"""

from __future__ import annotations

import ctypes
import math
import os
from functools import lru_cache

import torch

from .jit import KERNEL_SRC_DIR, compile_cuda_source

DEFAULT_NUM_BLOCKS = 4
SKIP_FAST_INDEX_COPY_ENV = "FREETOKEN_SKIP_FAST_INDEX_COPY"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _skip_fast_index_copy_enabled() -> bool:
    return os.getenv(SKIP_FAST_INDEX_COPY_ENV, "").strip().lower() in _TRUE_VALUES


def _cuh_include() -> str:
    return f'#include "{KERNEL_SRC_DIR / "fast_index_copy.cuh"}"'


def _default_worker_threads(feature_size: int) -> int:
    if feature_size <= 1024:
        return 8
    if feature_size <= 2048:
        return 16
    return 32


def _shrink_worker_feature_size(feature_size: int, worker_feature_size: int) -> int:
    if feature_size < worker_feature_size:
        worker_feature_size = feature_size
    while feature_size % worker_feature_size != 0 and worker_feature_size > 128:
        worker_feature_size //= 2
    return worker_feature_size


def default_worker_args(feature_size: int) -> tuple[int, int, int]:
    """(worker_threads, worker_feature_size, num_block) of the default call path.

    Matches FreeToken's ``default_worker_args``; the JIT instantiates exactly
    these template args, so a default ``fast_index_copy_jit`` call reuses the
    cached build.
    """
    return (
        _default_worker_threads(feature_size),
        _shrink_worker_feature_size(feature_size, 2048),
        DEFAULT_NUM_BLOCKS,
    )


# The original kernel requires kWorkersFeatures % 128 == 0 (load_vec's
# kBytesPerLoop), which the shrink above preserves (2048..128); the feature
# size itself must therefore be a multiple of 128. Cheap Python-side guard so a
# bad shape fails before nvcc does.
def _require_feature_alignment(feature_size: int) -> None:
    if feature_size % 128 != 0:
        raise ValueError(
            f"fast_index_copy: feature byte size {feature_size} must be a "
            "multiple of 128 (the kernel's per-row transfer granularity)"
        )


_LAUNCH_ARGTYPES = [
    ctypes.c_void_p,  # dst
    ctypes.c_void_p,  # dst_indices
    ctypes.c_void_p,  # src
    ctypes.c_void_p,  # src_indices
    ctypes.c_int64,  # length
    ctypes.c_void_p,  # num_indices (may be None)
    ctypes.c_int,  # use_int32
    ctypes.c_void_p,  # sync_flag (may be None)
    ctypes.c_int,  # mode: 0 default, 1 high, 2 normal
    ctypes.c_void_p,  # cudaStream_t
]
_MULTI_LAUNCH_ARGTYPES = [
    ctypes.c_void_p,  # dst_ptrs
    ctypes.c_void_p,  # src_ptrs
    ctypes.c_void_p,  # feat_bytes
    ctypes.c_void_p,  # dst_indices
    ctypes.c_void_p,  # src_indices
    ctypes.c_int64,  # length
    ctypes.c_void_p,  # num_indices (may be None)
    ctypes.c_int,  # num_banks
    ctypes.c_int,  # use_int32
    ctypes.c_void_p,  # cudaStream_t
]


@lru_cache(maxsize=None)
def _fast_index_copy_module(
    feature_size: int, worker_threads: int, worker_feature_size: int, num_block: int
) -> ctypes.CDLL:
    source = f"""{_cuh_include()}

extern "C" void plasticinfer_fast_index_copy_launch(
    void* dst, const void* dst_indices, void* src, const void* src_indices,
    int64_t length, const int64_t* num_indices, int use_int32,
    int32_t* sync_flag, int mode, cudaStream_t stream) {{
    using K = FastIndexCopyKernel<{feature_size}, {worker_threads}, {worker_feature_size}, 1024, {num_block}, 1>;
    K::run_impl(dst, dst_indices, src, src_indices, static_cast<std::size_t>(length),
                num_indices, use_int32 != 0, sync_flag, mode, stream);
}}
"""
    lib = compile_cuda_source(source, name=f"fast_index_copy_f{feature_size}")
    lib.plasticinfer_fast_index_copy_launch.argtypes = _LAUNCH_ARGTYPES
    return lib


def fast_index_copy_jit(
    dst: torch.Tensor,
    dst_indices: torch.Tensor,
    src: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    worker_threads: int | None = None,
    worker_feature_size: int = 2048,
    num_block: int | None = None,
    priority: str | None = None,
    sync_flag: torch.Tensor | None = None,
) -> None:
    """Copy the rows ``src_indices`` -> ``dst_indices`` of one bank (one launch).

    ``dst``/``src`` are ``[rows, features]`` banks (any dtype; row bytes derived
    from the shape); the index tensors are int32/int64 CUDA tensors and
    ``num_indices``, when given, is a CUDA int64 ``[1]`` valid-length override
    (so the same launch is reusable across varying miss counts). ``priority``
    ``"high"``/``"normal"`` gates the copy on ``sync_flag`` (CUDA int32 ``[1]``)
    for the CPU-executor priority plumbing; ``None`` is the plain path.
    """
    num_dst_feature = math.prod(dst.shape[1:])
    num_src_feature = math.prod(src.shape[1:])
    assert num_src_feature == num_dst_feature

    # Debug/perf ablation: keep miss bookkeeping intact, but make the copy free to
    # approximate a zero-copy-miss runtime. Outputs are only meaningful if callers
    # already have valid cache contents for the requested indices.
    if _skip_fast_index_copy_enabled():
        return

    dst = dst.as_strided(size=(dst.size(0), num_dst_feature), stride=(num_dst_feature, 1))
    src = src.as_strided(size=(src.size(0), num_src_feature), stride=(num_src_feature, 1))

    feature_size = dst.size(-1) * dst.element_size()
    num_block = num_block or DEFAULT_NUM_BLOCKS
    worker_threads = worker_threads or _default_worker_threads(feature_size)
    worker_feature_size = _shrink_worker_feature_size(feature_size, worker_feature_size)
    assert worker_threads in (8, 16, 32)
    assert feature_size % worker_feature_size == 0
    _require_feature_alignment(feature_size)

    assert dst_indices.shape == src_indices.shape, "index shape mismatch"
    assert dst_indices.device.type == "cuda" and src_indices.device.type == "cuda"

    lib = _fast_index_copy_module(feature_size, worker_threads, worker_feature_size, num_block)
    launch = lib.plasticinfer_fast_index_copy_launch
    mode = {"high": 1, "normal": 2}.get(priority, 0)
    if priority is not None:
        assert sync_flag is not None and sync_flag.numel() == 1
    launch(
        dst.data_ptr(),
        dst_indices.data_ptr(),
        src.data_ptr(),
        src_indices.data_ptr(),
        dst_indices.numel(),
        num_indices.data_ptr() if num_indices is not None else None,
        1 if dst_indices.dtype == torch.int32 else 0,
        sync_flag.data_ptr() if sync_flag is not None else None,
        mode,
        torch.cuda.current_stream().cuda_stream,
    )


@lru_cache(maxsize=None)
def _fast_index_copy_multi_module(num_threads: int, blocks_per_bank: int) -> ctypes.CDLL:
    source = f"""{_cuh_include()}

extern "C" void plasticinfer_fast_index_copy_multi_launch(
    const int64_t* dst_ptrs, const int64_t* src_ptrs, const int64_t* feat_bytes,
    const void* dst_indices, const void* src_indices, int64_t length,
    const int64_t* num_indices, int num_banks, int use_int32, cudaStream_t stream) {{
    using K = MultiIndexCopyKernel<{num_threads}, {blocks_per_bank}>;
    K::run(dst_ptrs, src_ptrs, feat_bytes, dst_indices, src_indices,
           static_cast<std::size_t>(length), num_indices, num_banks,
           use_int32 != 0, stream);
}}
"""
    lib = compile_cuda_source(source, name=f"fast_index_copy_multi_t{num_threads}_b{blocks_per_bank}")
    lib.plasticinfer_fast_index_copy_multi_launch.argtypes = _MULTI_LAUNCH_ARGTYPES
    return lib


def fast_index_copy_multi_jit(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    feat_bytes: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    num_threads: int = 1024,
    blocks_per_bank: int = 8,
) -> None:
    """Fused multi-bank index copy: copy the same rows for every bank in ONE launch.

    ``dst_ptrs``/``src_ptrs``/``feat_bytes`` are int64 ``[num_banks]`` device
    tensors built once by the caller (per-bank slot-cache base addr, host-source
    base addr, per-row byte size). Every bank's per-row byte size must be a
    multiple of 16, and the base addresses 16-byte aligned (true for contiguous
    torch allocations of these banks). This is the path ``copy_missing`` uses
    once the cache's fused-copy plan is built (see ``_build_copy_plan``).
    """
    if _skip_fast_index_copy_enabled():
        return
    lib = _fast_index_copy_multi_module(num_threads, blocks_per_bank)
    launch = lib.plasticinfer_fast_index_copy_multi_launch
    launch(
        dst_ptrs.data_ptr(),
        src_ptrs.data_ptr(),
        feat_bytes.data_ptr(),
        dst_indices.data_ptr(),
        src_indices.data_ptr(),
        dst_indices.numel(),
        num_indices.data_ptr() if num_indices is not None else None,
        dst_ptrs.numel(),
        1 if dst_indices.dtype == torch.int32 else 0,
        torch.cuda.current_stream().cuda_stream,
    )


@lru_cache(maxsize=1)
def _update_flag_module() -> ctypes.CDLL:
    source = f"""{_cuh_include()}

extern "C" void plasticinfer_fast_index_copy_update_flag(
    int32_t* flag_ptr, int32_t delta, cudaStream_t stream) {{
    update_copy_flag_kernel<<<1, 1, 0, stream>>>(flag_ptr, delta);
}}
"""
    lib = compile_cuda_source(source, name="fast_index_copy_flag")
    lib.plasticinfer_fast_index_copy_update_flag.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p,
    ]
    return lib


def update_copy_flag_jit(sync_flag: torch.Tensor, delta: int) -> None:
    """Atomically bump ``sync_flag`` by ``delta`` (priority plumbing; see .cuh)."""
    assert sync_flag.is_cuda
    assert sync_flag.numel() == 1
    assert sync_flag.dtype == torch.int32
    _update_flag_module().plasticinfer_fast_index_copy_update_flag(
        sync_flag.data_ptr(), delta, torch.cuda.current_stream().cuda_stream
    )
