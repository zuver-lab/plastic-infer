"""Expert residency: host bank + GPU LRU slot pool (M1).

Implements the sparse-expert data class (§3.1 of DESIGN.md):
  - ExpertBank: *all* experts resident in host RAM. This is the M1
    assumption (weights fully resident in memory; the disk tier that
    feeds the bank arrives in M3).
  - ExpertSlotPool: a GPU budget-bounded LRU cache of the routed
    subset — FreeToken's host-bank + GPU-slot separation (D3).
    Eviction reuses the shared pin/refcount kernel (LruPool) and the
    single hard invariant: eviction only reclaims pinned == False (D7).

Correctness is independent of how many experts fit in GPU slots (D5):
with even one slot the runner produces identical logits, it just
loads and evicts more often.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .resident import LruPool
from .tiers import Tier


@dataclass
class ExpertWeights:
    """The weight tensors of one Mixtral-style expert (SwiGLU).

    Roles (HF Mixtral naming): w1 = gate (silu applied), w3 = up,
    w2 = down. Shapes: w1, w3 -> [inter_dim, hidden_dim];
    w2 -> [hidden_dim, inter_dim].
    """

    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for t in (self.w1, self.w2, self.w3))


class ExpertBank:
    """Host-resident bank of every expert, keyed (layer, expert_id)."""

    def __init__(self) -> None:
        self._experts: dict[tuple[int, int], ExpertWeights] = {}
        self._bytes: dict[tuple[int, int], int] = {}

    def add(self, layer: int, expert_id: int, weights: ExpertWeights) -> None:
        key = (layer, expert_id)
        assert key not in self._experts, f"duplicate expert {key}"
        self._experts[key] = weights
        self._bytes[key] = weights.bytes()

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._experts

    def __getitem__(self, key: tuple[int, int]) -> ExpertWeights:
        return self._experts[key]

    def bytes(self, key: tuple[int, int]) -> int:
        return self._bytes[key]

    def __len__(self) -> int:
        return len(self._experts)

    def keys(self):
        return self._experts.keys()

    @property
    def total_bytes(self) -> int:
        return sum(self._bytes.values())


class ExpertSlotPool:
    """GPU LRU cache of the routed expert subset, budgeted in bytes.

    `ensure(layer, expert_ids)` loads missing experts from the bank
    onto the pool device (evicting least-recently-used unpinned
    experts first, D7), pins the requested set for the duration of
    compute, and returns GPU-resident weights. Caller must call
    `release(layer, expert_ids)` after compute.

    Hit/miss counters are exposed for the M1 gate: decode expert
    hit-rate must be observable.
    """

    def __init__(self, bank: ExpertBank, budget_bytes: int,
                 device: torch.device | None = None) -> None:
        # A slot pool must hold at least one expert, else loading any
        # expert would silently overflow the budget (no OOM by design).
        if len(bank) > 0:
            assert budget_bytes >= max(
                bank.bytes(k) for k in bank.keys()
            ), "slot budget smaller than a single expert"
        self.bank = bank
        self.device = device or torch.device("cpu")
        self.pool = LruPool(budget_bytes)
        self._tensors: dict[tuple[int, int], ExpertWeights] = {}
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    def ensure(self, layer: int, expert_ids: list[int] | torch.Tensor
               ) -> dict[int, ExpertWeights]:
        """Load (or confirm) the routed experts of one layer and pin them.

        Returns {expert_id: ExpertWeights} with pool-device tensors.
        Caller must call `release(layer, expert_ids)` after compute.
        """
        eids = [int(e) for e in expert_ids]
        keys = [(layer, e) for e in eids]

        # Visit each requested expert in order, pinning it immediately so
        # a later eviction (to make room for the next miss) can never take
        # an expert we are about to return (D7). A "hit" evicted while
        # making room for another miss is simply re-detected as a miss.
        for k in keys:
            if k not in self.pool:
                for page in self.pool.evict_to_fit(self.bank.bytes(k)):
                    del self._tensors[page.key]
                self.pool.add(k, self.bank.bytes(k), tier=Tier.GPU)
                self._tensors[k] = self._load(k)
                self.misses += 1
            else:
                self.hits += 1
            self.pool.pin(k)
        return {e: self._tensors[(layer, e)] for e in eids}

    def release(self, layer: int, expert_ids: list[int] | torch.Tensor
                ) -> None:
        """Unpin the experts after compute. Double-release raises."""
        for e in expert_ids:
            self.pool.unpin((layer, int(e)))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def _load(self, key: tuple[int, int]) -> ExpertWeights:
        """Copy one expert from the host bank to the pool device."""
        src = self.bank[key]
        return ExpertWeights(*(t.to(self.device)
                               for t in (src.w1, src.w2, src.w3)))
