"""Tests for store.experts — host bank + GPU LRU slot pool (M1).

Covers residency budget, LRU eviction order, the D7 pin invariant,
and hit/miss accounting. All tensors on CPU (no GPU needed).
"""

from __future__ import annotations

import pytest
import torch

from plastic_infer.store.experts import ExpertBank, ExpertSlotPool, ExpertWeights


def _expert(seed: int, H: int = 8, I: int = 16) -> ExpertWeights:
    g = torch.Generator().manual_seed(seed)
    return ExpertWeights(
        w1=torch.randn(I, H, generator=g),
        w2=torch.randn(H, I, generator=g),
        w3=torch.randn(I, H, generator=g),
    )


EXPERT_BYTES = _expert(0).bytes()   # 3 * 128 floats * 4 bytes = 1536


def _make_bank(n_layers: int = 2, n_experts: int = 4) -> ExpertBank:
    bank = ExpertBank()
    for l in range(n_layers):
        for e in range(n_experts):
            bank.add(l, e, _expert(l * 100 + e))
    return bank


@pytest.fixture
def bank() -> ExpertBank:
    return _make_bank()


def _pool(bank: ExpertBank, n_slots: int) -> ExpertSlotPool:
    return ExpertSlotPool(bank, budget_bytes=n_slots * EXPERT_BYTES)


class TestExpertBank:
    def test_add_get_roundtrip(self, bank: ExpertBank) -> None:
        assert (0, 2) in bank
        w = bank[(0, 2)]
        assert w.w1.shape == (16, 8)
        assert len(bank) == 8

    def test_bytes_accounting(self, bank: ExpertBank) -> None:
        assert bank.bytes((0, 0)) == EXPERT_BYTES
        assert bank.total_bytes == len(bank) * EXPERT_BYTES

    def test_duplicate_add_raises(self, bank: ExpertBank) -> None:
        with pytest.raises(AssertionError):
            bank.add(0, 0, _expert(1))

    def test_unknown_key_missing(self, bank: ExpertBank) -> None:
        assert (9, 9) not in bank


class TestSlotPoolBudget:
    def test_budget_holds_subset(self, bank: ExpertBank) -> None:
        pool = _pool(bank, n_slots=2)
        pool.ensure(0, [0, 1])
        pool.release(0, [0, 1])
        assert pool.pool.used_bytes == 2 * EXPERT_BYTES
        assert pool.pool.free_bytes == 0

    def test_evicts_lru_when_over_budget(self, bank: ExpertBank) -> None:
        pool = _pool(bank, n_slots=2)
        pool.ensure(0, [0, 1])
        pool.release(0, [0, 1])

        pool.ensure(0, [2, 3])
        pool.release(0, [2, 3])

        assert (0, 0) not in pool.pool   # evicted
        assert (0, 1) not in pool.pool
        assert (0, 2) in pool.pool
        assert (0, 3) in pool.pool
        assert pool.pool.used_bytes == 2 * EXPERT_BYTES

    def test_lru_order_respected(self, bank: ExpertBank) -> None:
        pool = _pool(bank, n_slots=3)
        pool.ensure(0, [0, 1])
        pool.release(0, [0, 1])
        # Re-touch expert 0 so expert 1 becomes the LRU entry
        pool.ensure(0, [0])
        pool.release(0, [0])

        pool.ensure(0, [2, 3])   # needs 1 slot beyond free → evicts LRU (1)
        pool.release(0, [2, 3])

        assert (0, 1) not in pool.pool   # LRU evicted
        assert (0, 0) in pool.pool       # recently used survived
        assert (0, 2) in pool.pool
        assert (0, 3) in pool.pool

    def test_d7_pinned_survives_eviction(self, bank: ExpertBank) -> None:
        pool = _pool(bank, n_slots=3)
        pool.ensure(0, [0, 1])
        pool.release(0, [0])     # unpin only expert 0; expert 1 stays pinned

        pool.ensure(0, [2, 3])   # needs 2 slots, only 1 free
        pool.release(0, [2, 3])

        assert (0, 1) in pool.pool    # pinned expert not evicted (D7)
        assert (0, 0) not in pool.pool
        assert (0, 2) in pool.pool
        assert (0, 3) in pool.pool

    def test_all_pinned_blocks_eviction(self, bank: ExpertBank) -> None:
        pool = _pool(bank, n_slots=2)
        pool.ensure(0, [0, 1])   # both pinned

        evicted = pool.pool.evict_to_fit(EXPERT_BYTES)
        assert evicted == []     # D7: nothing unpinned to reclaim
        assert (0, 0) in pool.pool
        assert (0, 1) in pool.pool
        pool.release(0, [0, 1])

    def test_budget_below_one_expert_rejected(self, bank: ExpertBank) -> None:
        with pytest.raises(AssertionError):
            ExpertSlotPool(bank, budget_bytes=EXPERT_BYTES - 1)


class TestEnsure:
    def test_miss_loads_then_hit_reuses(self, bank: ExpertBank) -> None:
        pool = _pool(bank, n_slots=4)

        r1 = pool.ensure(0, [0, 1])
        pool.release(0, [0, 1])
        assert pool.misses == 2
        assert pool.hits == 0

        r2 = pool.ensure(0, [0, 1])
        pool.release(0, [0, 1])
        assert pool.hits == 2
        assert pool.misses == 2
        # Same resident tensors returned on hit
        assert r2[0] is r1[0] and r2[1] is r1[1]

    def test_hit_rate(self, bank: ExpertBank) -> None:
        pool = _pool(bank, n_slots=4)
        pool.ensure(0, [0, 1])
        pool.release(0, [0, 1])
        pool.ensure(0, [0, 1])
        pool.release(0, [0, 1])
        pool.ensure(0, [2, 3])     # 2 more misses
        pool.release(0, [2, 3])

        assert pool.misses == 4
        assert pool.hits == 2
        assert pool.hit_rate == pytest.approx(2 / 6)

    def test_data_matches_bank(self, bank: ExpertBank) -> None:
        pool = _pool(bank, n_slots=4)
        got = pool.ensure(0, [1, 3])
        pool.release(0, [1, 3])

        for e, w in got.items():
            ref = bank[(0, e)]
            assert torch.allclose(w.w1, ref.w1)
            assert torch.allclose(w.w2, ref.w2)
            assert torch.allclose(w.w3, ref.w3)

    def test_double_release_raises(self, bank: ExpertBank) -> None:
        pool = _pool(bank, n_slots=4)
        pool.ensure(0, [0])
        pool.release(0, [0])
        with pytest.raises(AssertionError):
            pool.release(0, [0])

    def test_correct_with_single_slot(self, bank: ExpertBank) -> None:
        """Even 1 slot serves every expert correctly (D5 at store level)."""
        pool = _pool(bank, n_slots=1)
        for l in range(2):
            for e in range(4):
                got = pool.ensure(l, [e])
                pool.release(l, [e])
                ref = bank[(l, e)]
                assert torch.allclose(got[e].w1, ref.w1)
        assert pool.pool.used_bytes <= EXPERT_BYTES
