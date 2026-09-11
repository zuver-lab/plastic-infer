"""Memory planner — pure arithmetic: device + model + request -> Plan.

No side effects. No tensors. Everything is ints and floats, so every
branch is unit-testable.

The planner answers one question: "For this device, this model, this
request, how should GPU VRAM and host RAM be divided among the three
pools, and where should each class of data live?"

GPU pools (§3.3 in DESIGN.md):
  1. activations & scratch   (fixed reservation)
  2. KV page pool
  3. expert slot pool
  4. dense weight window     (gets whatever's left)

Host budget follows the same shape but with different defaults —
dense and expert each decide whether they fit in host RAM as a full
bank or must fall back to a smaller buffer + disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .profile import DeviceProfile


WeightSource = Literal["HOST", "DISK", "HOST_FIRST"]
LongCtxMode = Literal["GPU", "HOST_STREAM", "REJECT"]


# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelMeta:
    """Describes the static shape/size of a model.

    All byte counts are per-dtype (the dtype you plan to run at).
    """

    n_layers: int
    dense_bytes: int                  # total dense weight bytes (B_d)
    per_layer_dense: tuple[int, ...]  # W_d(l) for each layer
    experts_per_layer: tuple[int, ...]   # E_l per layer
    expert_row_bytes: tuple[int, ...]    # b_e(l) per layer
    kv_bytes_per_token: int           # sum of kv_tok(l) across all layers

    def __post_init__(self) -> None:
        assert len(self.per_layer_dense) == self.n_layers
        assert len(self.experts_per_layer) == self.n_layers
        assert len(self.expert_row_bytes) == self.n_layers
        assert self.kv_bytes_per_token > 0

    @property
    def total_expert_bytes(self) -> int:
        return sum(e * b for e, b in zip(self.experts_per_layer,
                                         self.expert_row_bytes))

    @property
    def total_weight_bytes(self) -> int:
        return self.dense_bytes + self.total_expert_bytes

    @property
    def max_expert_row(self) -> int:
        return max(self.expert_row_bytes) if self.expert_row_bytes else 0

    @property
    def total_experts(self) -> int:
        return sum(self.experts_per_layer)


@dataclass(frozen=True)
class RequestMeta:
    """What we know about an incoming request before running it."""

    seq_budget: int          # model's context window cap (tokens)
    gen_tokens: int          # max_new_tokens estimate
    reuse_prefix_chunks: int = 0  # how many prefix chunks we expect to hit


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """A complete residency/budget plan for one request."""

    # -- GPU pools --
    weight_window: int           # W: dense layers resident on GPU
    expert_slots: int            # total expert slots across all layers
    kv_hot_tokens: int           # how many tokens fit in KV page pool

    # -- sourcing --
    seq_weight_source: WeightSource   # where dense weights come from
    expert_source: WeightSource       # where expert weights come from
    expert_host_slots: int            # how many expert slots fit in host LRU

    # -- KV --
    kv_host_l1_cap_bytes: int    # host-side KV L1 cache budget
    long_ctx_mode: LongCtxMode   # what to do when KV outgrows GPU + host

    # -- sanity totals (for assertions) --
    gpu_total_bytes: int         # sum of all GPU pool bytes
    host_total_bytes: int        # sum of all host pool bytes

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def dense_fits_in_host(self) -> bool:
        return self.seq_weight_source in ("HOST", "HOST_FIRST")

    @property
    def experts_fit_in_host(self) -> bool:
        return self.expert_source in ("HOST", "HOST_FIRST")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class MemoryPlanner:
    """Greedy pool allocator.

    Priority order (§3.3):
      1. activations & scratch reservation
      2. KV page pool
      3. expert slot pool
      4. dense window (gets the rest)

    Host budget follows a similar logic:
      - Dense gets a full bank if it fits (highest priority because dense
        has no hot-subset escape — every layer is used every step).
      - Expert bank fits what it can; overflow goes to disk (experts
        have natural hot-cold skew so LRU + disk spill degrades gracefully).
      - Remaining goes to KV L1 cache.
    """

    # Tokens per KV page — must match kv.paged.
    PAGE_TOKENS = 16

    def plan(
        self,
        profile: DeviceProfile,
        model: ModelMeta,
        request: RequestMeta,
        *,
        safety: float = 0.08,
        kv_reserve_tokens: int = 2048,
        min_weight_window: int = 2,
        activation_bytes: int | None = None,
        host_os_reserve_ratio: float = 0.15,
        host_staging_bytes: int | None = None,
    ) -> Plan:
        """Compute a plan. Never raises; never silently over-commits."""

        usable_hbm = int(profile.hbm_bytes * (1.0 - safety))

        # 0. activation/scratch reservation
        if activation_bytes is None:
            # rough: 2 * max_layer_dense (for the intermediate tensors)
            # plus a hidden-state buffer
            max_dense = max(model.per_layer_dense)
            hidden = int(model.kv_bytes_per_token / max(1, model.n_layers)
                         * 2)  # K + V heads -> rough hidden estimate
            activation_bytes = max_dense + hidden * request.seq_budget
        activation_bytes = min(activation_bytes, usable_hbm // 4)
        remaining = usable_hbm - activation_bytes
        assert remaining > 0, "activation reservation exceeds usable HBM"

        # Dense minimum: we always need at least min_weight_window layers
        # in GPU memory to make forward progress. This is a hard floor;
        # everything else gets carved out of what's left.
        avg_dense = model.dense_bytes / max(1, model.n_layers)
        dense_min_bytes = min(
            min_weight_window * int(avg_dense),
            model.dense_bytes,
        )
        assert dense_min_bytes < remaining, (
            f"dense floor {dense_min_bytes} > remaining {remaining}"
        )

        # 1. KV page pool — gets up to half of what's left after dense floor
        kv_need_tokens = request.seq_budget + request.gen_tokens
        kv_need_bytes = kv_need_tokens * model.kv_bytes_per_token
        kv_pool_cap = (remaining - dense_min_bytes) // 2
        kv_hot_bytes = min(kv_need_bytes, kv_pool_cap)
        kv_hot_tokens = max(
            (kv_hot_bytes // model.kv_bytes_per_token),
            kv_reserve_tokens,
        )
        kv_hot_bytes_actual = kv_hot_tokens * model.kv_bytes_per_token
        remaining -= kv_hot_bytes_actual

        # 2. Expert slot pool — as many as we can afford,
        #    but always leave room for the dense floor.
        avg_expert = (
            model.total_expert_bytes / max(1, model.total_experts)
            if model.total_experts > 0 else 0
        )
        max_layer_experts = (
            max(model.experts_per_layer) if model.experts_per_layer else 0
        )
        expert_floor = min(2 * max_layer_experts, model.total_experts)

        expert_budget = remaining - dense_min_bytes  # don't eat into dense
        if avg_expert > 0 and expert_budget > 0:
            max_affordable = int(expert_budget / avg_expert)
            expert_slots = min(model.total_experts, max_affordable)
            # If we can hit the floor, great; otherwise take what fits.
            expert_slots = max(0, expert_slots)
            expert_bytes = int(expert_slots * avg_expert)
        else:
            expert_slots = 0
            expert_bytes = 0
        remaining -= expert_bytes

        # 3. Dense weight window — whatever is left
        max_window_fit = int(remaining / avg_dense) if avg_dense > 0 else 0
        weight_window = min(max_window_fit, model.n_layers)
        # Floor: at least min_weight_window, but never more than n_layers.
        weight_window = max(weight_window, min_weight_window)
        weight_window = min(weight_window, model.n_layers)
        weight_bytes = weight_window * int(avg_dense)
        # Safety: ensure we don't overshoot
        if weight_bytes > remaining:
            weight_window = max(min_weight_window,
                                int(remaining / avg_dense))
            weight_window = min(weight_window, model.n_layers)
            weight_bytes = weight_window * int(avg_dense)
        remaining -= weight_bytes

        # gpu_total = what we actually allocated (sum of pools)
        gpu_total = (
            activation_bytes
            + kv_hot_bytes_actual
            + expert_bytes
            + weight_bytes
        )

        # -------------------------
        # Host budget
        # -------------------------
        host_usable = int(profile.host_ram_bytes * (1 - host_os_reserve_ratio))
        if host_staging_bytes is None:
            host_staging_bytes = min(
                int(profile.b_disk * 2),  # 2 seconds of disk read
                host_usable // 20,
            )
        host_usable -= host_staging_bytes

        # Dense first — it has no hot-subset escape.
        dense_fits = model.dense_bytes <= host_usable
        if dense_fits:
            seq_weight_source: WeightSource = "HOST"
            host_dense = model.dense_bytes
        else:
            seq_weight_source = "DISK"
            # leave a reasonably-sized host ring buffer
            host_dense = min(model.dense_bytes,
                             max(4 * max(model.per_layer_dense),
                                 host_usable // 4))

        host_remaining = host_usable - host_dense

        # Experts — fit what we can, rest goes to disk.
        if avg_expert > 0 and host_remaining >= model.total_expert_bytes:
            expert_source: WeightSource = "HOST"
            expert_host_slots = model.total_experts
            host_expert = model.total_expert_bytes
        elif avg_expert > 0 and host_remaining > 0:
            expert_source = "HOST_FIRST"  # host LRU + disk spill
            max_fit = int(host_remaining / avg_expert)
            # Try the floor; if it doesn't fit, take what we can (even 0).
            if max_fit >= max_layer_experts:
                expert_host_slots = max_fit
            else:
                expert_host_slots = max(0, max_fit)
            expert_host_slots = min(expert_host_slots, model.total_experts)
            host_expert = int(expert_host_slots * avg_expert)
        else:
            expert_source = "DISK"
            expert_host_slots = 0
            host_expert = 0
        # Safety: never overshoot
        if host_expert > host_remaining:
            expert_host_slots = max(0, int(host_remaining / max(1, avg_expert)))
            expert_host_slots = min(expert_host_slots, model.total_experts)
            host_expert = int(expert_host_slots * avg_expert)
        host_remaining -= host_expert

        # KV L1 — whatever remains
        kv_host_l1_cap_bytes = max(0, host_remaining)

        # Long-context mode
        if kv_hot_tokens >= request.seq_budget + request.gen_tokens:
            long_ctx_mode: LongCtxMode = "GPU"
        elif kv_host_l1_cap_bytes > 0:
            long_ctx_mode = "HOST_STREAM"
        else:
            long_ctx_mode = "REJECT"

        # Host total = explicit sum (not derived from remaining)
        host_total = host_staging_bytes + host_dense + host_expert + kv_host_l1_cap_bytes

        return Plan(
            weight_window=weight_window,
            expert_slots=expert_slots,
            kv_hot_tokens=kv_hot_tokens,
            seq_weight_source=seq_weight_source,
            expert_source=expert_source,
            expert_host_slots=expert_host_slots,
            kv_host_l1_cap_bytes=kv_host_l1_cap_bytes,
            long_ctx_mode=long_ctx_mode,
            gpu_total_bytes=gpu_total,
            host_total_bytes=host_total,
        )
