"""Residency management: pin/refcount, LRU eviction, per-pool budgets.

This is the shared kernel used by all three pools (dense window,
expert slots, KV chunks). It enforces the single hard safety
invariant from §5.2 of DESIGN.md:

  Eviction may only reclaim objects where pinned == False.

The Page here is a *logical* page — a tracked object with a key, a
size, a current tier, and a pin count. Actual tensor storage lives
in the mover / backend modules; this module only tracks ownership
and residency state.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Generic, Hashable, TypeVar

from .tiers import Tier


K = TypeVar("K", bound=Hashable)


@dataclass
class Page(Generic[K]):
    """A tracked, tier-resident object.

    pinned == True means "there is an in-flight copy or an active
    compute reference" — the page must not be evicted or moved.
    """

    key: K
    bytes_: int
    tier: Tier
    pinned: int = 0           # reference count, not a boolean
    # For LRU ordering: lower = older (used less recently)
    _lru_seq: int = field(default=0, repr=False, compare=False)


class LruPool(Generic[K]):
    """A budget-bounded pool of Page objects with LRU eviction.

    Threading note: v1 is single-writer (the main compute thread
    pins/unpins; the copy thread only transitions tier after a
    transfer completes). For true concurrency, wrap in a lock —
    not needed for C=1.
    """

    def __init__(self, budget_bytes: int) -> None:
        self._budget = budget_bytes
        self._pages: dict[K, Page[K]] = {}
        # LRU order: oldest first
        self._lru: OrderedDict[K, None] = OrderedDict()
        self._seq = 0
        self._used_bytes = 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def free_bytes(self) -> int:
        return self._budget - self._used_bytes

    def __contains__(self, key: K) -> bool:
        return key in self._pages

    def get(self, key: K) -> Page[K] | None:
        return self._pages.get(key)

    def __getitem__(self, key: K) -> Page[K]:
        return self._pages[key]

    def __len__(self) -> int:
        return len(self._pages)

    # ------------------------------------------------------------------
    # Insertion / removal
    # ------------------------------------------------------------------

    def add(self, key: K, bytes_: int, tier: Tier = Tier.GPU) -> Page[K]:
        """Add a new page. Caller must ensure it fits (or evict first)."""
        assert key not in self._pages, f"duplicate key {key}"
        assert bytes_ >= 0
        page = Page(key=key, bytes_=bytes_, tier=tier)
        self._pages[key] = page
        self._used_bytes += bytes_
        self._touch(page)
        return page

    def remove(self, key: K) -> Page[K]:
        """Remove a page from the pool. Must not be pinned."""
        page = self._pages.pop(key)
        assert page.pinned == 0, f"cannot remove pinned page {key}"
        self._used_bytes -= page.bytes_
        del self._lru[key]
        return page

    # ------------------------------------------------------------------
    # Pin / unpin (reference counting)
    # ------------------------------------------------------------------

    def pin(self, key: K) -> Page[K]:
        """Increment pin count. Page cannot be evicted while pinned > 0."""
        page = self._pages[key]
        page.pinned += 1
        self._touch(page)
        return page

    def unpin(self, key: K) -> None:
        """Decrement pin count. Raises if already 0 (double-unpin bug)."""
        page = self._pages[key]
        assert page.pinned > 0, f"unpin of already-unpinned page {key}"
        page.pinned -= 1
        self._touch(page)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def can_fit(self, bytes_: int) -> bool:
        """Whether `bytes_` would fit without evicting anything."""
        return self._used_bytes + bytes_ <= self._budget

    def evict_lru(self, need_bytes: int) -> list[Page[K]]:
        """Evict least-recently-used *unpinned* pages until we have
        `need_bytes` free (or run out of evictable pages).

        Returns the list of evicted pages (already removed from pool).
        Only evicts GPU-resident pages; HOST/DISK pages don't count
        against the GPU budget.

        Safety: never evicts a pinned page (D7 invariant).
        """
        evicted: list[Page[K]] = []
        freed = 0

        # Walk from oldest to newest
        for key in list(self._lru.keys()):
            if freed >= need_bytes:
                break
            page = self._pages[key]
            if page.pinned > 0:
                continue
            if page.tier != Tier.GPU:
                # Already off-GPU — doesn't count against GPU budget.
                # Still occupies the slot in the index though.
                continue
            evicted.append(self.remove(key))
            freed += page.bytes_

        return evicted

    def evict_to_fit(self, bytes_: int) -> list[Page[K]]:
        """Evict until `bytes_` fits. Returns evicted pages (may be [])."""
        if self.can_fit(bytes_):
            return []
        need = bytes_ - self.free_bytes
        return self.evict_lru(need)

    def all_pinned(self) -> bool:
        """True if every GPU page is pinned (nothing can be evicted)."""
        return all(
            p.pinned > 0 or p.tier != Tier.GPU
            for p in self._pages.values()
        )

    # ------------------------------------------------------------------
    # Tier transitions
    # ------------------------------------------------------------------

    def set_tier(self, key: K, new_tier: Tier) -> None:
        """Update a page's tier. Caller is responsible for the actual
        data movement; this just updates bookkeeping."""
        page = self._pages[key]
        old = page.tier
        page.tier = new_tier
        if old == Tier.GPU and new_tier != Tier.GPU:
            # No longer counts against GPU budget
            self._used_bytes -= page.bytes_
        elif old != Tier.GPU and new_tier == Tier.GPU:
            # Now counts against GPU budget
            self._used_bytes += page.bytes_
        self._touch(page)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _touch(self, page: Page[K]) -> None:
        """Mark page as most-recently-used."""
        self._seq += 1
        page._lru_seq = self._seq
        # Move to end (most recent)
        if page.key in self._lru:
            del self._lru[page.key]
        self._lru[page.key] = None
