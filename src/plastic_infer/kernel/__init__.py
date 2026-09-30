"""CUDA kernels + pinned-memory helpers for the offload MoE stack.

Port of FreeToken's ``freetoken.kernel`` package: pure-triton grouped-GEMM /
routing kernels, the flashlib LRU slot-cache kernel, pinned helpers, and (M1b)
the standalone nvcc-JIT fast_index_copy extension.

Only the pure-torch/triton pieces are importable at package import time; the
compiled pieces (``_cpu_moe``, ``fast_index_copy``) load lazily where used.
"""

from .moe_impl import (
    fused_moe_decode_kernel_triton,
    fused_moe_kernel_triton,
    moe_align_block_size_triton,
    moe_sum_reduce_triton,
)
from .pinned import copy_to_pinned_tensor, create_pinned_tensor_like

__all__ = [
    "fused_moe_decode_kernel_triton",
    "fused_moe_kernel_triton",
    "moe_align_block_size_triton",
    "moe_sum_reduce_triton",
    "copy_to_pinned_tensor",
    "create_pinned_tensor_like",
]
