"""CPU MoE executor (``CpuMoeExecutor``, the M2 ``_cpu_moe`` extension) tests.

Part 1 -- numerical alignment: the CPU SwiGLU MoE GEMV vs the production GPU
decode kernel on identical bf16 banks and routing across batch sizes
(fp32-accumulate, so the only spread is reduction order -> tight tol).

Part 2 -- end-to-end through the runner: an ``OffloadMoeCache`` whose
``decode_target`` is "cpu" with every layer CPU-locked, plus a live
``CpuMoeExecutor``, served to ``prefill_forward_moe`` (GPU) / ``decode_step_moe``
(CPU executor); every decode logit must match HF. Runs under both the default
stream-memop flag handshake and ``FREETOKEN_CPU_MOE_FLAG_SYNC=0`` (the
``cudaLaunchHostFunc`` sync), so the whole D2H -> CPU GEMV -> H2D path is
exercised either way.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from transformers import MixtralConfig, MixtralForCausalLM

from plastic_infer.exec.runner import DenseKVCache, DenseWeights, ModelConfig
from plastic_infer.exec.runner_moe import decode_step_moe, prefill_forward_moe
from plastic_infer.kernel.pinned import alloc_pinned_tensor
from plastic_infer.moe.offload_cache import OffloadMoeCache
from plastic_infer.store.experts import ExpertBank

from _moe_cache_helpers import banks_from_expert_bank, require_cuda
from test_equiv_smoke_moe import (
    _extract_moe_weights,
    _make_moe_config,
    _tiny_mixtral_config,
)


def _make_cpu_cache(bank: ExpertBank, device: torch.device, *,
                    top_k: int) -> tuple[OffloadMoeCache, object]:
    """An OffloadMoeCache with every layer CPU-locked + a live executor attached.

    Imports ``CpuMoeExecutor`` at call time so a reloaded module (flag-sync env
    toggle) is picked up.
    """
    from plastic_infer.moe.cpu_executor import CpuMoeExecutor

    layers = sorted({l for l, _ in bank.keys()})
    experts = sorted({e for _, e in bank.keys()})
    cache = OffloadMoeCache(
        num_layers=len(layers), num_experts=len(experts),
        cache_size=len(experts), device=device, quant_format="bf16",
        decode_target="cpu",
    )
    cache.collect_stats = True
    cache.cpu_layer_ids = frozenset(layers)
    cache.set_bank_sources(banks_from_expert_bank(bank, dtype=torch.bfloat16))
    executor = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0, max_tokens=1, device=device,
    )
    cache.set_cpu_executor(executor)
    return cache, executor


# ---------------------------------------------------------------------------
# Part 1 -- raw GEMV alignment (CPU executor vs GPU fused decode kernel)
# ---------------------------------------------------------------------------

def _make_random_bf16_cache(L: int, E: int, H: int, I: int):
    gate_up = alloc_pinned_tensor(L * E, 2 * I, H, dtype=torch.bfloat16)
    down = alloc_pinned_tensor(L * E, H, I, dtype=torch.bfloat16)
    gate_up.copy_(torch.randn(L * E, 2 * I, H) * 0.1)
    down.copy_(torch.randn(L * E, H, I) * 0.1)
    return SimpleNamespace(
        quant_format="bf16",
        bank_sources={"gate_up": list(gate_up.split(E)),
                      "down": list(down.split(E))},
        num_layers=L, num_experts=E,
        decode_target="cpu", cpu_executor=None,
    )


@pytest.mark.parametrize("bs", [1, 4])
def test_cpu_decode_matches_gpu_decode_kernel(bs: int) -> None:
    from plastic_infer.moe.cpu_executor import CpuMoeExecutor
    from plastic_infer.moe.fused import fused_experts_decode_impl

    torch.manual_seed(bs)
    L, E, H, I, top_k = 4, 16, 1024, 512, 4
    layer = 2
    dev = require_cuda()
    cache = _make_random_bf16_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0, max_tokens=bs, device=dev,
    )
    try:
        hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=dev)[:top_k]
                           for _ in range(bs)]).to(torch.int32)
        w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

        cpu_out = ex.decode(layer, hidden, w, ids).float()
        torch.cuda.synchronize()

        gu = cache.bank_sources["gate_up"][layer].to(dev)
        dn = cache.bank_sources["down"][layer].to(dev)
        gpu_out = fused_experts_decode_impl(
            hidden, gu, dn, w, ids.clone(), "silu", False,
        ).float()

        rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
        assert rel < 2e-2, f"bs={bs} rel err {rel.item()}"
    finally:
        del ex
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Part 2 -- end-to-end through the runner, both sync handshakes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hf_model() -> MixtralForCausalLM:
    dev = require_cuda()
    torch.manual_seed(42)
    model = MixtralForCausalLM(_tiny_mixtral_config())
    model.to(device=dev, dtype=torch.bfloat16)
    model.eval()
    return model


@pytest.fixture(scope="module")
def config(hf_model: MixtralForCausalLM) -> ModelConfig:
    return _make_moe_config(hf_model.config)


@pytest.fixture(scope="module")
def weights(hf_model: MixtralForCausalLM) -> DenseWeights:
    dev = require_cuda()
    dense, _bank = _extract_moe_weights(hf_model)
    return DenseWeights({k: v.to(device=dev, dtype=torch.bfloat16)
                         for k, v in dense.items()})


@pytest.fixture(scope="module")
def bank(hf_model: MixtralForCausalLM) -> ExpertBank:
    _dense, bank = _extract_moe_weights(hf_model)
    return bank


def _assert_allclose(a: torch.Tensor, b: torch.Tensor, where: str) -> None:
    assert a.shape == b.shape
    assert torch.allclose(a.float(), b.float(), atol=0.05, rtol=0.05), (
        f"{where}: max diff = {(a - b).float().abs().max().item():.6f}"
    )


@pytest.mark.parametrize("flag_sync", ["1", "0"])
def test_cpu_locked_decode_matches_hf(
    flag_sync: str,
    weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
    hf_model: MixtralForCausalLM, monkeypatch,
) -> None:
    """All-layers-CPU decode == HF, via both the default stream-memop flag
    handshake and FREETOKEN_CPU_MOE_FLAG_SYNC=0 (cudaLaunchHostFunc sync)."""
    monkeypatch.setenv("FREETOKEN_CPU_MOE_FLAG_SYNC", flag_sync)
    # _FLAG_SYNC is read at module import; reload so the env toggle applies.
    import plastic_infer.moe.cpu_executor as cpe
    importlib.reload(cpe)

    torch.manual_seed(3)
    prompt_len, n_decode = 6, 4
    input_ids = torch.randint(0, config.vocab_size, (prompt_len,),
                              device=weights.device)
    cache, executor = _make_cpu_cache(bank, weights.device,
                                      top_k=config.n_experts_per_tok)
    try:
        if flag_sync == "0":
            assert executor._flag_sync is False

        kv = DenseKVCache(config, max_seq_len=128, dtype=torch.bfloat16,
                          device=weights.device)
        our_prefill = prefill_forward_moe(weights, config, cache, kv,
                                          input_ids)
        our_tokens: list[int] = [int(our_prefill.argmax().item())]
        our_logits: list[torch.Tensor] = [our_prefill]
        for _ in range(n_decode):
            logits = decode_step_moe(weights, config, cache, kv,
                                     our_tokens[-1])
            our_logits.append(logits)
            our_tokens.append(int(logits.argmax().item()))

        with torch.no_grad():
            full_ids = torch.tensor(
                input_ids.tolist() + our_tokens[:n_decode],
            ).unsqueeze(0).to(weights.device)
            hf_logits_all = hf_model(full_ids).logits[0]

        for step in range(n_decode + 1):
            pos = prompt_len - 1 + step
            _assert_allclose(our_logits[step], hf_logits_all[pos],
                             f"cpu-locked decode step {step} (pos {pos})")
    finally:
        del cache
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Part 3 -- hybrid pageable fetch (WSL: unpinned banks, no device alias)
# ---------------------------------------------------------------------------

def _make_pageable_hybrid_cache(L: int, E: int, H: int, I: int, cache_size: int,
                                device: torch.device):
    """A ``decode_target="hybrid"`` cache whose banks are pageable (no device
    alias): every decode fetch must go through ``copy_missing``'s pageable
    gather. Mirrors the engine's WSL outcome -- pin budget << one layer's bank,
    so no bank is registered and no layer is a pure CPU layer."""
    gate_up = torch.empty(L * E, 2 * I, H, dtype=torch.bfloat16)
    down = torch.empty(L * E, H, I, dtype=torch.bfloat16)
    gate_up.normal_().mul_(0.1)
    down.normal_().mul_(0.1)
    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=cache_size,
        device=device, quant_format="bf16", decode_target="hybrid",
    )
    from plastic_infer.moe.host_banks import HostResidency

    cache.cpu_layer_ids = frozenset()  # no CPU-only layers: every layer is hybrid
    cache.set_bank_sources(
        {"gate_up": list(gate_up.split(E)), "down": list(down.split(E))},
        layer_residency=[HostResidency.LOCKED.value] * L,
    )
    assert len(cache._unpinned_layers) == L
    return cache, gate_up, down


def test_hybrid_pageable_fetch_matches_gpu_kernel() -> None:
    """Hybrid decode over unpinned (pageable) banks: the staged misses are
    gathered on the host into their LRU slots (``copy_missing``'s pageable
    branch) and the overflow runs on the CPU executor; the merged output must
    equal the GPU fused kernel on identical banks."""
    from plastic_infer.exec.runner_moe import _decode_hybrid
    from plastic_infer.moe.cpu_executor import CpuMoeExecutor
    from plastic_infer.moe.fused import fused_experts_decode_impl
    from plastic_infer.moe.host_banks import HostResidency

    torch.manual_seed(11)
    L, E, H, I, top_k = 2, 8, 256, 128, 4
    dev = require_cuda()
    cache, gate_up, down = _make_pageable_hybrid_cache(
        L, E, H, I, cache_size=E, device=dev)
    cache.hybrid_max_fetch = 2  # cap below the miss count so the CPU executor runs
    ex = CpuMoeExecutor(cache, top_k=top_k, activation="silu",
                        apply_router_weight_on_input=False,
                        num_threads=0, max_tokens=1, device=dev)
    cache.set_cpu_executor(ex)
    layer = 1
    try:
        for step in range(3):
            cache.reset()  # fresh empty mapping: every active expert is a miss
            hidden = torch.randn(1, H, device=dev, dtype=torch.bfloat16) * 0.1
            ids = torch.stack([torch.randperm(E, device=dev)[:top_k]
                               for _ in range(1)]).to(torch.int32)
            w = torch.rand(1, top_k, device=dev)
            out = _decode_hybrid(cache, layer, hidden, w, ids.clone()).float()
            torch.cuda.synchronize()
            gu = gate_up.split(E)[layer].to(dev)
            dn = down.split(E)[layer].to(dev)
            ref = fused_experts_decode_impl(
                hidden, gu, dn, w, ids.clone(), "silu", False).float()
            rel = (out - ref).abs().max() / (ref.abs().max() + 1e-6)
            assert rel < 2e-2, f"step {step}: rel err {rel.item()}"
    finally:
        del cache, ex
        torch.cuda.empty_cache()
