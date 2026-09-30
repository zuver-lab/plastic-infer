"""Pinned host-tensor helpers (port of FreeToken's kernel/pinned.py).

FreeToken delegates to a ``_pinned_tensor`` torch extension for exact-size
``cudaMallocHost`` storage. That extension can't be built here (WSL toolkit at
``/home/kitty/.cuda/cuda-13.2/`` is not on torch's include path), so this module
reimplements the same surface with ctypes against libcudart where it matters and
torch's pinned pool elsewhere:

  * ``alloc_pinned_tensor`` / ``create_pinned_tensor_like`` / ``copy_to_pinned_tensor``:
    pinned allocations via ``torch.empty(pin_memory=True)``. On this machine the only
    pinned allocations are small (CPU-executor IO buffers, flag arrays, the optional
    born-pinned HostBank path); the expert banks themselves are OS-locked (LOCKED,
    no CUDA pin quota) -- the huge-exact-size case FreeToken's custom allocator
    protects against never runs here.
  * ``host_register``: real ``cudaHostRegister`` (portable+mapped) via ctypes -- the
    pin-after-fill path used to register filled mmap'd banks.
  * ``device_ptr`` / ``host_ptr_identity``: FreeToken's probe (uva == 1 &&
    CanUseHostPointerForRegisteredMem == 1). Under Linux/UVA that is True and
    ``device_ptr`` returns ``data_ptr()`` directly; WSL2 reports the registered-mem
    attribute as 0 (False), so host banks go through ``cudaHostGetDevicePointer``
    (which maps them at their host VA -- the same address, but the explicit call is
    required to validate the mapping).
"""

from __future__ import annotations

import ctypes
from functools import lru_cache

import torch

# cudaHostRegister / cudaHostAlloc flags
_CUDA_HOST_REGISTER_PORTABLE = 0x01
_CUDA_HOST_REGISTER_MAPPED = 0x02
_CUDA_HOST_REGISTER_DEVICE_MAP = 0x04
_CUDA_HOST_ALLOC_PORTABLE = 0x01
_CUDA_HOST_ALLOC_MAPPED = 0x02


@lru_cache(maxsize=1)
def _cudart():
    try:
        return ctypes.CDLL("libcudart.so")
    except OSError as exc:
        raise RuntimeError(
            "pinned.py: libcudart.so not loadable (CUDA runtime required for "
            "host_register/alloc_pinned_tensor)" if False else
            f"pinned.py: libcudart.so not loadable: {exc}"
        ) from exc


def create_pinned_tensor_like(input: torch.Tensor) -> torch.Tensor:
    """Create a CPU pinned tensor with the same size, stride, and dtype as input."""
    return torch.empty_strided(
        input.shape, input.stride(), dtype=input.dtype, device="cpu", pin_memory=True
    )


def copy_to_pinned_tensor(input: torch.Tensor) -> torch.Tensor:
    """Copy a CPU tensor into pinned storage."""
    output = create_pinned_tensor_like(input)
    with torch.no_grad():
        output.copy_(input)
    return output


def alloc_pinned_tensor(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a pinned host tensor (torch pinned pool, page-aligned for cuda)."""
    return torch.empty(tuple(int(s) for s in shape), dtype=dtype, pin_memory=True)


def host_register(addr: int, nbytes: int) -> None:
    """cudaHostRegister ``nbytes`` at ``addr`` as portable+mapped (pin-after-fill)."""
    lib = _cudart()
    lib.cudaHostRegister.restype = ctypes.c_int
    lib.cudaHostRegister.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
    ]
    err = lib.cudaHostRegister(
        ctypes.c_void_p(addr), ctypes.c_size_t(nbytes),
        ctypes.c_uint(_CUDA_HOST_REGISTER_PORTABLE | _CUDA_HOST_REGISTER_MAPPED),
    )
    if err != 0:
        fn = lib.cudaGetErrorString
        fn.restype = ctypes.c_char_p
        fn.argtypes = [ctypes.c_int]
        raise RuntimeError(
            f"cudaHostRegister({nbytes} bytes) failed: "
            f"cudaError {err} ({fn(ctypes.c_int(err)).decode()})"
        )


@lru_cache(maxsize=1)
def _host_ptr_identity() -> bool:
    # FreeToken's probe: uva == 1 && CanUseHostPointerForRegisteredMem == 1. On
    # Linux/UVA registered host memory is device-visible at its host VA (True);
    # WSL2 reports the registered-mem attribute as 0 (False), so host pointers need
    # the explicit cudaHostGetDevicePointer mapping. The cudaDeviceAttr enum numbering
    # moved between CUDA 12 (UnifiedAddressing=35, CanUseHostPointerForRegisteredMem=40)
    # and CUDA 13 (41, 91); resolve the ids from torch's paired CUDA major.
    major = int((torch.version.cuda or "0").split(".")[0])
    uva_attr, reg_attr = (41, 91) if major >= 13 else (35, 40)
    lib = _cudart()
    lib.cudaDeviceGetAttribute.restype = ctypes.c_int
    lib.cudaDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
    ]
    uva = ctypes.c_int(0)
    reg = ctypes.c_int(0)
    lib.cudaDeviceGetAttribute(ctypes.byref(uva), uva_attr, 0)
    lib.cudaDeviceGetAttribute(ctypes.byref(reg), reg_attr, 0)
    return bool(uva.value == 1 and reg.value == 1)


def device_ptr(t: torch.Tensor) -> int:
    """Base address of ``t`` as the GPU must dereference it.

    Under identity (``_host_ptr_identity``) that is ``data_ptr()``; otherwise host
    tensors are mapped with ``cudaHostGetDevicePointer`` (the address the CUDA kernels
    must dereference -- WSL2 returns the host VA, but the explicit call validates it).
    """
    if t.is_cuda or _host_ptr_identity():
        return t.data_ptr()
    lib = _cudart()
    lib.cudaHostGetDevicePointer.restype = ctypes.c_int
    lib.cudaHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint,
    ]
    mapped = ctypes.c_void_p()
    err = lib.cudaHostGetDevicePointer(ctypes.byref(mapped), t.data_ptr(), 0)
    if err != 0:
        raise RuntimeError(
            "pinned.device_ptr: cudaHostGetDevicePointer failed "
            f"(cudaError {err}) for a host tensor; host banks must be pinned+mapped "
            "(register filled banks with pinned.host_register)"
        )
    return mapped.value
