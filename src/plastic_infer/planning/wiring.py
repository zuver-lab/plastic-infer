"""Plan -> runtime pool budgets (M3 "elastic replanning" wiring).

budget.py decides *shares* (how many dense layers / expert slots / KV
tokens each pool gets); this module converts those shares into the
concrete byte/page budgets the runtime pools are constructed with.
Pure arithmetic, no side effects — like budget.py, every branch is
unit-testable.

"Elastic replanning" (§3.4 of DESIGN.md) is exactly this: at a request
boundary the engine calls plan() again (different request -> different
shares), derives new pool budgets here, and rebuilds the pools. No
tensor migrates mid-request; the pools are sized per request.
"""

from __future__ import annotations

from dataclasses import dataclass

from .budget import MemoryPlanner, ModelMeta, Plan


@dataclass(frozen=True)
class PoolBudgets:
    """Concrete budgets for the three GPU pools of one request."""

    dense_window_bytes: int   # DenseWindow byte budget (rotating window)
    expert_slots_bytes: int   # ExpertSlotPool byte budget
    kv_pool_bytes: int        # KV page pool byte budget
    kv_pages: int             # KV page pool size in pages (PAGE_TOKENS each)


def _avg_per_layer(model: ModelMeta, total: int, n: int) -> int:
    return total // n if n > 0 else 0


def dense_window_bytes(plan: Plan, model: ModelMeta) -> int:
    """Byte budget for the dense rotating window (W layers resident)."""
    avg = _avg_per_layer(model, model.dense_bytes, model.n_layers)
    return plan.weight_window * avg


def expert_slot_budget_bytes(plan: Plan, model: ModelMeta) -> int:
    """Byte budget for the expert slot pool (slot count -> bytes)."""
    avg = _avg_per_layer(model, model.total_expert_bytes,
                         model.total_experts)
    return plan.expert_slots * avg


def kv_pool_bytes(plan: Plan, model: ModelMeta) -> int:
    """Byte budget for the KV page pool (hot tokens * bytes/token)."""
    return plan.kv_hot_tokens * model.kv_bytes_per_token


def kv_page_pool_pages(plan: Plan, model: ModelMeta) -> int:
    """KV page pool size in pages. At least one page (a request needs
    to allocate pages before writing KV)."""
    return max(1, plan.kv_hot_tokens // MemoryPlanner.PAGE_TOKENS)


def pool_budgets(plan: Plan, model: ModelMeta) -> PoolBudgets:
    return PoolBudgets(
        dense_window_bytes=dense_window_bytes(plan, model),
        expert_slots_bytes=expert_slot_budget_bytes(plan, model),
        kv_pool_bytes=kv_pool_bytes(plan, model),
        kv_pages=kv_page_pool_pages(plan, model),
    )
