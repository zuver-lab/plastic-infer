"""Live hybrid bandwidth probe (M2+): offload->hybrid pick + the fetch split.

Port of FreeToken's ``ft bench bw`` core (``benchbw.py``) as a live, cached
one-shot instead of a separate ``ft bench bw`` subcommand + JSON profile: this
repo has no such CLI, so the engine measures the two bandwidths on the first
hybrid build. The hybrid backend computes which fraction of a decode step's
expert misses go over PCIe to the GPU (vs. computed on the CPU) so both sides
finish together: fetch fraction = pcie_bw / cpu_bw (FreeToken's standalone
fallback in ``load_hybrid_fetch_fraction``; its overlap refinement
``pcie_ov/(pcie_ov+cpu_ov)`` is omitted here). ``recommend`` upgrades offload ->
hybrid only when the CPU MoE bandwidth clears the PCIe gather bandwidth by the
bench threshold (default 2x), exactly FreeToken's rule.
"""

from __future__ import annotations

import functools
import statistics
import time
from types import SimpleNamespace

import torch

from ..kernel.pinned import alloc_pinned_tensor
from .offload_cache import OffloadMoeCache


def _expert_bytes(hidden: int, inter: int) -> int:
    """One bf16 expert's total bytes (gate_up + down)."""
    return (2 * inter * hidden + inter * hidden) * 2


def _pin_budget_bytes() -> int | None:
    """Host pinning ceiling (WSL ~1 GiB); None on uncapped hosts. Lazy import so
    the moe package never pulls in the runtime engine at import time."""
    from ..runtime.engine import _pin_budget_bytes as _b
    return _b()


def _synth_experts(num_experts: int, per_expert: int,
                   cap: int | None = None) -> int:
    """Cap synthetic experts so the pinned rig stays under ``cap`` bytes (WSL's
    cudaHostAlloc ceiling is ~1 GiB; the loaders hit the same wall). Default cap
    is 1 GiB on uncapped hosts, shrunk inside a measured WSL pin budget so the
    probe itself does not trip the wall it exists to quantify."""
    if cap is None:
        budget = _pin_budget_bytes()
        cap = min(1 << 30, int(budget * 0.7)) if budget is not None else 1 << 30
    return min(num_experts, max(1, cap // per_expert))


def measure_pcie_gather_bw(device: torch.device, num_experts: int, hidden: int,
                           inter: int, iters: int = 20) -> dict:
    """Real PCIe gather bandwidth (GB/s): pinned host banks -> GPU slot cache.

    Drives the production ``OffloadMoeCache.copy_missing`` (fused multi-bank
    ``fast_index_copy`` when 16-byte aligned) refilling a full synthetic layer;
    timed with CUDA events (FreeToken ``measure_pcie_gather_bw``).
    """
    E = _synth_experts(num_experts, _expert_bytes(hidden, inter))
    cache = OffloadMoeCache(num_layers=1, num_experts=E, cache_size=E,
                            device=device, quant_format="bf16")
    total_bytes = 0
    for name, elems in (("gate_up", 2 * inter * hidden), ("down", hidden * inter)):
        src = alloc_pinned_tensor(E, elems, dtype=torch.bfloat16)
        dst = torch.empty(E, elems, dtype=torch.bfloat16, device=device)
        cache.bank_sources[name] = [src]
        cache.bank_caches[name] = dst
        cache.banks.append(([src], dst))
        total_bytes += E * elems * 2
    cache._build_copy_plan()  # enable the fused path (production default) when aligned
    cache.evict_slots[:E].copy_(torch.arange(E, dtype=torch.int32, device=device))
    cache.src_indices[:E].copy_(torch.randperm(E, device=device).to(torch.int32))
    cache.num_indices.fill_(E)  # int64, matches the kernel contract
    cache._pending_src_layer = 0  # normally set by ensure_experts; bypassed here

    for _ in range(3):
        cache.copy_missing()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        cache.copy_missing()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / iters
    return {"bw_gbs": total_bytes / (ms / 1e3) / 1e9, "fused": bool(cache._copy_fused_ok)}


def measure_cpu_moe_bw(device: torch.device, num_experts: int, hidden: int,
                       inter: int, top_k: int, iters: int = 32,
                       num_threads: int = 0) -> float:
    """Real CPU MoE GEMV bandwidth (GB/s) at bs=1, experts read from pinned host
    banks (FreeToken ``measure_cpu_moe_bw`` / ``_time_cpu_moe``). Consecutive steps
    route to disjoint experts so the working set stays past the LLC -- the DRAM-bound
    regime hybrid decode actually runs in."""
    from .cpu_executor import CpuMoeExecutor

    E = _synth_experts(num_experts, _expert_bytes(hidden, inter))
    gate_up = alloc_pinned_tensor(E, 2 * inter, hidden, dtype=torch.bfloat16)
    down = alloc_pinned_tensor(E, hidden, inter, dtype=torch.bfloat16)
    cache = SimpleNamespace(
        quant_format="bf16",
        bank_sources={"gate_up": [gate_up], "down": [down]},
        num_layers=1, num_experts=E,
        decode_target="cpu", cpu_executor=None,
    )
    ex = CpuMoeExecutor(cache, top_k=top_k, activation="silu",
                        apply_router_weight_on_input=False,
                        num_threads=num_threads, max_tokens=1, device=device)
    bs = 1
    io = ex._io_for(bs)
    io["x"].copy_(torch.randn(bs, hidden, dtype=torch.bfloat16) * 0.1)
    io["w"].copy_(torch.rand(bs, top_k, dtype=torch.float32))
    task = ex._task_for(0, bs)
    base = torch.arange(top_k, dtype=torch.int32)

    def run(i: int) -> None:
        io["ids"].copy_((((i * top_k) + base) % E).view(bs, top_k))
        ex._ext.run_task(task)

    for i in range(8):  # warmup
        run(i)
    samples = []
    for i in range(iters):
        t0 = time.perf_counter()
        run(8 + i)
        samples.append(time.perf_counter() - t0)
    ms = statistics.median(samples) * 1e3
    return (top_k * _expert_bytes(hidden, inter)) / (ms / 1e3) / 1e9


def recommend(cpu_bw_gbs: float, pcie_bw_gbs: float, threshold: float = 2.0) -> str:
    """``hybrid`` iff CPU bandwidth exceeds ``threshold`` x PCIe bandwidth (FreeToken)."""
    return "hybrid" if cpu_bw_gbs > threshold * pcie_bw_gbs else "offload"


@functools.lru_cache(maxsize=1)
def probe(device: torch.device, num_experts: int, hidden: int, inter: int,
          top_k: int) -> dict | None:
    """Measure both bandwidths once: ``{recommended, fraction, cpu_bw, pcie_bw}``.

    ``fraction`` = pcie_bw / cpu_bw (fetch this share of each step's misses over
    PCIe, compute the rest on the CPU, clamped to [0, 1]); ``None`` if either
    measurement is unusable (caller keeps offload / a fixed fetch cap of 1).
    """
    try:
        if not torch.cuda.is_available():
            return None
        # torch.cuda.set_device rejects an unindexed torch.device("cuda");
        # normalize to the current device's index first (they are equivalent).
        device = torch.device(device.type,
                              device.index if device.index is not None
                              else torch.cuda.current_device())
        torch.cuda.set_device(device)
        pcie = measure_pcie_gather_bw(device, num_experts, hidden, inter)
        cpu = measure_cpu_moe_bw(device, num_experts, hidden, inter, top_k)
    except (ImportError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        import logging

        logging.getLogger(__name__).warning(
            "hybrid bandwidth probe unavailable (%s); staying on offload / cap 1", exc)
        return None
    if not pcie["bw_gbs"] or not cpu:
        return None
    return {
        "recommended": recommend(cpu, pcie["bw_gbs"]),
        "fraction": min(1.0, pcie["bw_gbs"] / cpu),
        "cpu_bw": cpu,
        "pcie_bw": pcie["bw_gbs"],
        "fused": pcie["fused"],
    }
