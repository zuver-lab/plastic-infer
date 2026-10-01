"""Build-on-first-use shim for the compiled ``_cpu_moe`` extension (M2).

Once ``_cpu_moe.so`` sits next to this file, Python's extension loader wins and
this module is never imported. Before that (a fresh checkout), importing
``from ..kernel import _cpu_moe`` lands here, which builds the ``.so`` via
:mod:`kernel.cpu_moe_build` and swaps itself out for the real compiled module
so every attribute (``CpuMoeExecutor``, ``memops_probe``, ...) resolves.
"""

from __future__ import annotations

import sys

from .cpu_moe_build import ensure_built

_impl = ensure_built()
sys.modules[__name__] = _impl
