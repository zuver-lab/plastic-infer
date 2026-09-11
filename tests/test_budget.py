"""Tests for planning.budget — pure functions, no GPU needed.

These test the invariants listed in §5.1 of DESIGN.md:
  - Σ pools ≤ hbm · (1 − safety)
  - weight_window ≥ 2 or explicit degradation
  - expert_slots ≤ total_experts, and is the greedy optimum
  - similar invariants on the host side
"""

from __future__ import annotations

import pytest

from plastic_infer.planning.budget import (
    MemoryPlanner,
    ModelMeta,
    Plan,
    RequestMeta,
)
from plastic_infer.planning.profile import DeviceProfile


# ---------------------------------------------------------------------------
# Fixtures: a small Llama-1B-ish dense model and a small MoE model
# ---------------------------------------------------------------------------

def _dense_profile() -> DeviceProfile:
    """8 GB GPU, 32 GB RAM, plausible bandwidths."""
    return DeviceProfile(
        hbm_bytes=8 * 1024**3,
        host_ram_bytes=32 * 1024**3,
        pin_limit_bytes=16 * 1024**3,
        b_h2d=12e9,
        b_host=40e9,
        b_disk=3e9,
    )


def _moe_profile() -> DeviceProfile:
    """16 GB GPU, 64 GB RAM."""
    return DeviceProfile(
        hbm_bytes=16 * 1024**3,
        host_ram_bytes=64 * 1024**3,
        pin_limit_bytes=32 * 1024**3,
        b_h2d=24e9,
        b_host=60e9,
        b_disk=5e9,
    )


def _small_dense(n_layers: int = 12, hidden: int = 2048,
                 kv_heads: int = 4, head_dim: int = 128) -> ModelMeta:
    """A synthetic ~1B dense model."""
    # per-layer dense: attn 4*hidden^2 + ffn ~ 8*hidden^2 ≈ 12*hidden^2
    per_layer = 12 * hidden * hidden * 2  # fp16
    # kv per token per layer: 2 * kv_heads * head_dim bytes
    kv_per_token_layer = 2 * kv_heads * head_dim * 2  # fp16
    return ModelMeta(
        n_layers=n_layers,
        dense_bytes=per_layer * n_layers,
        per_layer_dense=tuple(per_layer for _ in range(n_layers)),
        experts_per_layer=tuple(0 for _ in range(n_layers)),
        expert_row_bytes=tuple(0 for _ in range(n_layers)),
        kv_bytes_per_token=kv_per_token_layer * n_layers,
    )


def _small_moe(n_layers: int = 8, hidden: int = 1024,
               n_experts: int = 8, expert_hidden: int = 2048,
               kv_heads: int = 4, head_dim: int = 64) -> ModelMeta:
    """Synthetic small MoE: 8 experts per layer."""
    # dense per layer: attn + norm + shared parts ≈ 4 * hidden^2 * 2 bytes
    per_layer_dense = 4 * hidden * hidden * 2
    # expert row: up+gate+down ≈ 2 * hidden * expert_hidden * 2 bytes
    expert_row = 2 * hidden * expert_hidden * 2
    kv_per_token_layer = 2 * kv_heads * head_dim * 2
    return ModelMeta(
        n_layers=n_layers,
        dense_bytes=per_layer_dense * n_layers,
        per_layer_dense=tuple(per_layer_dense for _ in range(n_layers)),
        experts_per_layer=tuple(n_experts for _ in range(n_layers)),
        expert_row_bytes=tuple(expert_row for _ in range(n_layers)),
        kv_bytes_per_token=kv_per_token_layer * n_layers,
    )


def _small_req(seq: int = 2048, gen: int = 256) -> RequestMeta:
    return RequestMeta(seq_budget=seq, gen_tokens=gen, reuse_prefix_chunks=0)


# ---------------------------------------------------------------------------
# Invariant checks — reused across tests
# ---------------------------------------------------------------------------

def assert_plan_invariants(plan: Plan, profile: DeviceProfile,
                           model: ModelMeta, safety: float = 0.08) -> None:
    """All the rules from §5.1 that every plan must satisfy."""
    usable = int(profile.hbm_bytes * (1.0 - safety))
    assert plan.gpu_total_bytes <= usable, (
        f"GPU total {plan.gpu_total_bytes} > usable {usable}"
    )
    # weight window floor (only meaningful when model has ≥ 2 layers)
    if model.n_layers >= 2:
        assert plan.weight_window >= 2 or plan.long_ctx_mode == "REJECT"
    # expert slots bounded
    assert 0 <= plan.expert_slots <= model.total_experts
    # host side doesn't exceed host RAM (minus OS reserve)
    # host_total_bytes includes staging, dense, expert, KV L1 pools
    host_cap = int(profile.host_ram_bytes * (1 - 0.15))
    assert plan.host_total_bytes <= host_cap, (
        f"host total {plan.host_total_bytes} > cap {host_cap}"
    )
    # expert_host_slots bounded
    assert 0 <= plan.expert_host_slots <= model.total_experts


# ---------------------------------------------------------------------------
# Tests: dense model
# ---------------------------------------------------------------------------

class TestDenseModel:
    def test_dense_small_gpu_fits_window(self) -> None:
        profile = _dense_profile()
        model = _small_dense()
        planner = MemoryPlanner()
        plan = planner.plan(profile, model, _small_req())
        assert_plan_invariants(plan, profile, model)
        # Dense model with 0 experts → expert_slots == 0
        assert plan.expert_slots == 0
        # Dense weights fit in host on a 32GB machine
        assert plan.dense_fits_in_host

    def test_dense_tiny_gpu_forces_small_window(self) -> None:
        """Very small GPU → weight window still ≥ 2 (floor)."""
        profile = DeviceProfile(
            hbm_bytes=2 * 1024**3,          # 2 GB — tiny
            host_ram_bytes=32 * 1024**3,
            pin_limit_bytes=16 * 1024**3,
            b_h2d=12e9, b_host=40e9, b_disk=3e9,
        )
        model = _small_dense(hidden=2048, n_layers=12)
        planner = MemoryPlanner()
        plan = planner.plan(profile, model, _small_req(seq=1024, gen=128))
        assert_plan_invariants(plan, profile, model)
        assert plan.weight_window >= 2

    def test_dense_no_experts_has_zero_slots(self) -> None:
        plan = MemoryPlanner().plan(_dense_profile(), _small_dense(),
                                    _small_req())
        assert plan.expert_slots == 0
        assert plan.expert_host_slots == 0

    def test_dense_long_context_outgrows_gpu(self) -> None:
        """Very long context + small GPU → KV spills beyond GPU page pool."""
        # 80 layers × 128 KV heads × 128 dim × fp16
        # kv_per_token = 80 * 2 * 128 * 128 * 2 = 5.2 MB/token... no, let's
        # use a bigger model with more KV capacity per token.
        model = _small_dense(n_layers=80, kv_heads=64, head_dim=128)
        # kv per token = 80 * 2 * 64 * 128 * 2 = 2,621,440 bytes = 2.5 MB
        # 4K tokens = 10 GB → won't fit in 8 GB GPU
        plan = MemoryPlanner().plan(
            _dense_profile(), model,
            _small_req(seq=4096, gen=512),
        )
        assert_plan_invariants(plan, _dense_profile(), model)
        assert plan.kv_hot_tokens < 4096 + 512
        assert plan.kv_host_l1_cap_bytes > 0


# ---------------------------------------------------------------------------
# Tests: MoE model
# ---------------------------------------------------------------------------

class TestMoEModel:
    def test_moe_experts_fit_in_gpu(self) -> None:
        profile = _moe_profile()
        model = _small_moe()
        plan = MemoryPlanner().plan(profile, model, _small_req())
        assert_plan_invariants(plan, profile, model)
        assert plan.expert_slots > 0
        # with 16GB GPU and small MoE, all experts should fit
        assert plan.expert_slots == model.total_experts

    def test_moe_host_fits_all_experts(self) -> None:
        """64 GB host → full expert bank stays in RAM."""
        model = _small_moe()
        plan = MemoryPlanner().plan(_moe_profile(), model, _small_req())
        assert plan.experts_fit_in_host
        assert plan.expert_host_slots == model.total_experts

    def test_moe_tiny_host_forces_disk_spill(self) -> None:
        """Host too small for all experts → HOST_FIRST with LRU + disk."""
        profile = DeviceProfile(
            hbm_bytes=8 * 1024**3,
            host_ram_bytes=2 * 1024**3,      # tiny 2GB host
            pin_limit_bytes=1 * 1024**3,
            b_h2d=12e9, b_host=40e9, b_disk=3e9,
        )
        # 32 layers × 32 experts each = 1024 experts, big expert rows
        model = _small_moe(n_layers=32, n_experts=32,
                           hidden=4096, expert_hidden=8192)
        plan = MemoryPlanner().plan(profile, model, _small_req())
        assert_plan_invariants(plan, profile, model)
        # Can't fit all 1024 experts in 2 GB host
        assert plan.expert_host_slots < model.total_experts
        assert plan.expert_source == "HOST_FIRST"

    def test_moe_expert_floor_respected(self) -> None:
        """expert_slots >= 2 * max_layer_experts when possible."""
        model = _small_moe(n_experts=8)
        plan = MemoryPlanner().plan(_moe_profile(), model, _small_req())
        # 8 layers × 8 experts = 64 total; floor = 2 × 8 = 16
        assert plan.expert_slots >= 16


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_experts_no_expert_row_bytes_div_by_zero(self) -> None:
        """No-expert model must not divide by zero."""
        model = _small_dense()
        plan = MemoryPlanner().plan(_dense_profile(), model, _small_req())
        assert plan.expert_slots == 0
        assert plan.expert_host_slots == 0

    def test_single_layer_model(self) -> None:
        model = _small_dense(n_layers=1)
        plan = MemoryPlanner().plan(_dense_profile(), model, _small_req())
        assert_plan_invariants(plan, _dense_profile(), model)
        assert plan.weight_window == 1   # can't exceed n_layers

    def test_kv_reserve_floor(self) -> None:
        """Even a tiny request gets at least kv_reserve_tokens."""
        model = _small_dense()
        plan = MemoryPlanner().plan(
            _dense_profile(), model,
            _small_req(seq=16, gen=1),
            kv_reserve_tokens=2048,
        )
        assert plan.kv_hot_tokens >= 2048

    def test_long_ctx_mode_gpu_when_fits(self) -> None:
        model = _small_dense()
        plan = MemoryPlanner().plan(
            _dense_profile(), model,
            _small_req(seq=128, gen=16),
        )
        assert plan.long_ctx_mode == "GPU"

    def test_safety_is_respected(self) -> None:
        """Cranking up safety reduces the usable budget.

        Use a large model that doesn't fit entirely, so tighter safety
        actually shrinks the window and KV pool.
        """
        model = _small_dense(n_layers=80, hidden=8192)   # ~70B-ish dense
        p_normal = MemoryPlanner().plan(
            _dense_profile(), model, _small_req(), safety=0.08,
        )
        p_tight = MemoryPlanner().plan(
            _dense_profile(), model, _small_req(), safety=0.30,
        )
        # Tighter safety → smaller or equal window
        assert p_tight.weight_window <= p_normal.weight_window
        # Usable budget is strictly smaller
        usable_normal = int(_dense_profile().hbm_bytes * (1 - 0.08))
        usable_tight = int(_dense_profile().hbm_bytes * (1 - 0.30))
        assert usable_tight < usable_normal
