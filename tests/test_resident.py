"""Tests for store.resident — LRU pool with pin/refcount.

Design invariants (§5.2 / D7):
  - Eviction never reclaims a pinned page.
  - Pool usage never exceeds budget.
  - LRU order matches touch order.
"""

from __future__ import annotations

import pytest

from plastic_infer.store.resident import LruPool
from plastic_infer.store.tiers import Tier


@pytest.fixture
def small_pool() -> LruPool[str]:
    """1000-byte pool, 5 pages of 100 bytes each."""
    pool: LruPool[str] = LruPool(1000)
    for i in range(5):
        pool.add(f"p{i}", 100, Tier.GPU)
    return pool


class TestBasicOps:
    def test_add_and_get(self, small_pool: LruPool[str]) -> None:
        assert "p0" in small_pool
        assert small_pool["p0"].bytes_ == 100
        assert small_pool["p0"].tier == Tier.GPU

    def test_used_bytes(self, small_pool: LruPool[str]) -> None:
        assert small_pool.used_bytes == 500
        assert small_pool.free_bytes == 500

    def test_can_fit(self, small_pool: LruPool[str]) -> None:
        assert small_pool.can_fit(500)
        assert not small_pool.can_fit(501)

    def test_remove(self, small_pool: LruPool[str]) -> None:
        p = small_pool.remove("p0")
        assert p.key == "p0"
        assert small_pool.used_bytes == 400
        assert "p0" not in small_pool

    def test_remove_pinned_raises(self, small_pool: LruPool[str]) -> None:
        small_pool.pin("p0")
        with pytest.raises(AssertionError):
            small_pool.remove("p0")


class TestPinRefcount:
    def test_pin_increments(self, small_pool: LruPool[str]) -> None:
        small_pool.pin("p0")
        assert small_pool["p0"].pinned == 1
        small_pool.pin("p0")
        assert small_pool["p0"].pinned == 2

    def test_unpin_decrements(self, small_pool: LruPool[str]) -> None:
        small_pool.pin("p0")
        small_pool.pin("p0")
        small_pool.unpin("p0")
        assert small_pool["p0"].pinned == 1
        small_pool.unpin("p0")
        assert small_pool["p0"].pinned == 0

    def test_double_unpin_raises(self, small_pool: LruPool[str]) -> None:
        with pytest.raises(AssertionError):
            small_pool.unpin("p0")


class TestLruOrder:
    def test_evict_lru_oldest_first(self, small_pool: LruPool[str]) -> None:
        # Pages added in order p0..p4; p0 is oldest
        evicted = small_pool.evict_lru(200)
        assert [p.key for p in evicted] == ["p0", "p1"]
        assert small_pool.used_bytes == 300

    def test_touch_makes_newer(self, small_pool: LruPool[str]) -> None:
        # Pin p0 (touches it → moves to newest)
        small_pool.pin("p0")
        small_pool.unpin("p0")
        # Now p1 is oldest
        evicted = small_pool.evict_lru(100)
        assert [p.key for p in evicted] == ["p1"]

    def test_evict_skips_pinned(self, small_pool: LruPool[str]) -> None:
        """D7 invariant: pinned pages are never evicted."""
        small_pool.pin("p0")
        small_pool.pin("p1")
        # Need 300 bytes; p0,p1 pinned → evict p2,p3,p4
        evicted = small_pool.evict_lru(300)
        keys = {p.key for p in evicted}
        assert "p0" not in keys
        assert "p1" not in keys
        assert "p2" in keys

    def test_evict_all_pinned_returns_partial(self,
                                              small_pool: LruPool[str]) -> None:
        """If everything is pinned, evict returns empty."""
        for i in range(5):
            small_pool.pin(f"p{i}")
        evicted = small_pool.evict_lru(500)
        assert evicted == []
        assert small_pool.all_pinned()

    def test_evict_more_than_available(self,
                                        small_pool: LruPool[str]) -> None:
        evicted = small_pool.evict_lru(99999)
        assert len(evicted) == 5  # all 5
        assert small_pool.used_bytes == 0


class TestEvictToFit:
    def test_already_fits(self, small_pool: LruPool[str]) -> None:
        evicted = small_pool.evict_to_fit(100)
        assert evicted == []
        assert small_pool.used_bytes == 500

    def test_needs_eviction(self, small_pool: LruPool[str]) -> None:
        evicted = small_pool.evict_to_fit(800)  # need 300 more
        assert len(evicted) == 3   # p0, p1, p2 = 300 bytes freed
        assert small_pool.free_bytes >= 800

    def test_cant_fit_all_pinned(self, small_pool: LruPool[str]) -> None:
        for i in range(5):
            small_pool.pin(f"p{i}")
        evicted = small_pool.evict_to_fit(900)
        assert evicted == []
        assert small_pool.free_bytes == 500  # unchanged


class TestTierTransitions:
    def test_demote_reduces_usage(self, small_pool: LruPool[str]) -> None:
        """Moving a page off-GPU reduces GPU budget usage."""
        assert small_pool.used_bytes == 500
        small_pool.set_tier("p0", Tier.HOST)
        assert small_pool.used_bytes == 400
        assert small_pool["p0"].tier == Tier.HOST

    def test_promote_adds_usage(self, small_pool: LruPool[str]) -> None:
        small_pool.set_tier("p0", Tier.HOST)
        assert small_pool.used_bytes == 400
        small_pool.set_tier("p0", Tier.GPU)
        assert small_pool.used_bytes == 500

    def test_evict_skips_non_gpu(self, small_pool: LruPool[str]) -> None:
        small_pool.set_tier("p0", Tier.HOST)
        small_pool.set_tier("p1", Tier.DISK)
        # Only p2,p3,p4 are GPU-resident
        evicted = small_pool.evict_lru(500)  # need 500, only 300 available
        assert len(evicted) == 3
        keys = {p.key for p in evicted}
        assert keys == {"p2", "p3", "p4"}

    def test_demoted_page_still_in_pool(self,
                                        small_pool: LruPool[str]) -> None:
        small_pool.set_tier("p0", Tier.HOST)
        assert "p0" in small_pool
        assert len(small_pool) == 5
