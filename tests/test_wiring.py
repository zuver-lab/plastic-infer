"""Tests for planning.wiring — plan -> runtime pool budgets (M3).

Gates:
  1. Budgets derive exactly from the plan's shares (internal consistency
     with budget.py's own arithmetic).
  2. Request-level replanning actually moves shares: short vs long request
     -> different KV pool / dense window budgets.
  3. The three GPU pools never exceed the plan's allocated GPU total.
  4. A model too big for host RAM is planned for a DISK source — the
     decision the M3 streamed runner realizes.
"""

from __future__ import annotations

import pytest

from plastic_infer.planning.budget import MemoryPlanner, RequestMeta
from plastic_infer.planning.wiring import (
    dense_window_bytes,
    expert_slot_budget_bytes,
    kv_page_pool_pages,
    kv_pool_bytes,
    pool_budgets,
)

from test_budget import _moe_profile, _small_dense, _small_moe  # noqa: F401


@pytest.fixture
def planner() -> MemoryPlanner:
    return MemoryPlanner()


def test_budgets_derive_from_plan(planner: MemoryPlanner) -> None:
    profile = _moe_profile()
    model = _small_moe()
    request = RequestMeta(seq_budget=2048, gen_tokens=256)
    plan = planner.plan(profile, model, request)

    b = pool_budgets(plan, model)

    avg_dense = model.dense_bytes // model.n_layers
    assert b.dense_window_bytes == plan.weight_window * avg_dense
    assert dense_window_bytes(plan, model) == b.dense_window_bytes
    avg_expert = model.total_expert_bytes // model.total_experts
    assert b.expert_slots_bytes == plan.expert_slots * avg_expert
    assert expert_slot_budget_bytes(plan, model) == b.expert_slots_bytes
    assert b.kv_pool_bytes == plan.kv_hot_tokens * model.kv_bytes_per_token
    assert kv_pool_bytes(plan, model) == b.kv_pool_bytes
    assert b.kv_pages == max(1, plan.kv_hot_tokens // 16)
    assert kv_page_pool_pages(plan, model) == b.kv_pages
    # page budget matches byte budget
    assert b.kv_pages * 16 * model.kv_bytes_per_token >= b.kv_pool_bytes - \
        (16 * model.kv_bytes_per_token)  # within one page


def test_replan_changes_shares(planner: MemoryPlanner) -> None:
    """Request-level replanning: long request eats KV pool, shrinks window."""
    profile = _moe_profile()
    model = _small_moe()

    short = planner.plan(profile, model,
                         RequestMeta(seq_budget=2048, gen_tokens=256))
    long_req = planner.plan(profile, model,
                            RequestMeta(seq_budget=4096, gen_tokens=1024))

    b_short = pool_budgets(short, model)
    b_long = pool_budgets(long_req, model)

    assert b_long.kv_pool_bytes >= b_short.kv_pool_bytes
    assert b_long.dense_window_bytes <= b_short.dense_window_bytes
    # and the wiring reflects the different plan
    assert b_long.kv_pages != b_short.kv_pages or \
        b_long.kv_pool_bytes != b_short.kv_pool_bytes


def test_budgets_within_gpu_total(planner: MemoryPlanner) -> None:
    profile = _moe_profile()
    model = _small_moe()
    plan = planner.plan(profile, model,
                        RequestMeta(seq_budget=4096, gen_tokens=512))
    b = pool_budgets(plan, model)
    pools = b.dense_window_bytes + b.expert_slots_bytes + b.kv_pool_bytes
    # pools are strict subsets of gpu_total (which also reserves activations)
    assert pools <= plan.gpu_total_bytes


def test_disk_source_when_model_exceeds_host(planner: MemoryPlanner) -> None:
    """Dense model bigger than host RAM -> the planner chooses DISK."""
    profile = _moe_profile()          # 64 GiB host
    model = _small_dense(n_layers=48, hidden=8192)   # ~77 GiB dense model
    assert model.dense_bytes > profile.host_ram_bytes

    plan = planner.plan(profile, model,
                        RequestMeta(seq_budget=1024, gen_tokens=128))
    assert plan.seq_weight_source == "DISK"
    # a disk-streamed model still keeps a window of layers resident
    assert plan.weight_window >= 2


def test_no_experts_no_expert_budget(planner: MemoryPlanner) -> None:
    profile = _moe_profile()
    model = _small_dense()
    plan = planner.plan(profile, model,
                        RequestMeta(seq_budget=2048, gen_tokens=256))
    assert expert_slot_budget_bytes(plan, model) == 0
    assert dense_window_bytes(plan, model) > 0
