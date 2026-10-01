"""Shared CUDA + bf16 OffloadMoeCache construction for the MoE equivalence tests.

The FreeToken port is CUDA-only (fused expert kernels + cudaHostGetDevicePointer
bank movement), so these tests build the same banks ``load_expert_banks`` would
settle (pinned host rows in the gate_up/down schema) directly from the per-expert
w1/w2/w3 ``ExpertBank`` the legacy tests used.
"""

from __future__ import annotations

import pytest
import torch

from plastic_infer.moe.offload_cache import OffloadMoeCache
from plastic_infer.store.experts import ExpertBank


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("FreeToken offload-MoE tests require CUDA")
    return torch.device("cuda")


def banks_from_expert_bank(bank: ExpertBank, *, dtype: torch.dtype = torch.bfloat16):
    """Per-expert w1/w2/w3 ExpertBank -> FreeToken bank schema.

    gate_up[L] = cat([stack(w1s), stack(w3s)], dim=1) -> [E, 2*I, H]
    down[L]    = stack(w2s)                           -> [E, H, I]
    Pinned host tensors, matching load_expert_banks' settled PINNED banks.
    """
    layers = sorted({l for l, _ in bank.keys()})
    experts = sorted({e for _, e in bank.keys()})
    gate_up, down = [], []
    for l in layers:
        w1s, w2s, w3s = [], [], []
        for e in experts:
            w = bank[(l, e)]
            w1s.append(w.w1)
            w2s.append(w.w2)
            w3s.append(w.w3)
        w1 = torch.stack(w1s).to("cpu", dtype)   # [E, I, H] host
        w3 = torch.stack(w3s).to("cpu", dtype)   # [E, I, H] host
        gate_up.append(torch.cat([w1, w3], dim=1).pin_memory())  # [E, 2I, H]
        down.append(torch.stack(w2s).to("cpu", dtype).pin_memory())  # [E, H, I]
    return {"gate_up": gate_up, "down": down}


def make_moe_cache(bank: ExpertBank, n_slots: int, *,
                   device: torch.device | None = None,
                   prefill_overlap: bool = False) -> OffloadMoeCache:
    """OffloadMoeCache over an ExpertBank with ``n_slots`` expert slots.

    The unified slot cache can never hold fewer than one slot per expert
    (validate_rebuild), so a tight budget clamps to num_experts — the old
    "n_slots=2/4" tests become "num_experts", which still forces constant
    cross-layer eviction during decode (D5).
    """
    dev = device or require_cuda()
    layers = sorted({l for l, _ in bank.keys()})
    experts = sorted({e for _, e in bank.keys()})
    cache = OffloadMoeCache(
        num_layers=len(layers),
        num_experts=len(experts),
        cache_size=max(int(n_slots), len(experts)),
        device=dev,
        quant_format="bf16",
        prefill_overlap=prefill_overlap,
    )
    cache.collect_stats = True
    cache.set_bank_sources(banks_from_expert_bank(bank, dtype=torch.bfloat16))
    return cache
