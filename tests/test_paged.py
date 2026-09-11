"""Tests for kv.paged — page allocator + block table (pure logic)."""

from __future__ import annotations

import pytest

from plastic_infer.kv.paged import (
    DEFAULT_CHUNK_PAGES,
    DEFAULT_PAGE_SIZE,
    BlockTable,
    PageAllocator,
)


# ---------------------------------------------------------------------------
# PageAllocator tests
# ---------------------------------------------------------------------------

class TestPageAllocator:
    def test_initial_state(self) -> None:
        a = PageAllocator(100)
        assert a.total_pages == 100
        assert a.free_pages == 100
        assert a.used_pages == 0

    def test_alloc_and_free(self) -> None:
        a = PageAllocator(10)
        p = a.alloc()
        assert 0 <= p < 10
        assert a.free_pages == 9
        assert a.used_pages == 1
        assert a.is_used(p)
        a.free(p)
        assert a.free_pages == 10
        assert not a.is_used(p)

    def test_alloc_unique(self) -> None:
        a = PageAllocator(50)
        ids = [a.alloc() for _ in range(50)]
        assert len(set(ids)) == 50
        assert a.free_pages == 0

    def test_alloc_full_raises(self) -> None:
        a = PageAllocator(3)
        for _ in range(3):
            a.alloc()
        with pytest.raises(IndexError):
            a.alloc()

    def test_alloc_many(self) -> None:
        a = PageAllocator(20)
        ids = a.alloc_many(5)
        assert len(ids) == 5
        assert len(set(ids)) == 5
        assert a.free_pages == 15

    def test_alloc_many_overflow(self) -> None:
        a = PageAllocator(5)
        with pytest.raises(IndexError):
            a.alloc_many(10)

    def test_free_reuse(self) -> None:
        """Freed pages get reallocated (LIFO reuse pattern)."""
        a = PageAllocator(10)
        p1 = a.alloc()
        p2 = a.alloc()
        a.free(p2)
        a.free(p1)
        # LIFO: p1 was freed last → reallocated first
        assert a.alloc() == p1
        assert a.alloc() == p2

    def test_free_many(self) -> None:
        a = PageAllocator(20)
        ids = a.alloc_many(10)
        assert a.free_pages == 10
        a.free_many(ids[:5])
        assert a.free_pages == 15
        # Free the rest one by one
        for pid in ids[5:]:
            a.free(pid)
        assert a.free_pages == 20

    def test_free_unused_raises(self) -> None:
        a = PageAllocator(10)
        with pytest.raises(AssertionError):
            a.free(42)

    def test_no_leak_through_random_allocation(self) -> None:
        """Allocate and free in random pattern — total stays consistent."""
        import random
        random.seed(12345)
        a = PageAllocator(100)
        used: list[int] = []
        for _ in range(1000):
            if used and random.random() < 0.5:
                idx = random.randrange(len(used))
                pid = used.pop(idx)
                a.free(pid)
            else:
                if a.free_pages > 0:
                    used.append(a.alloc())
        # Free everything
        for pid in used:
            a.free(pid)
        assert a.free_pages == 100
        assert a.used_pages == 0


# ---------------------------------------------------------------------------
# BlockTable tests
# ---------------------------------------------------------------------------

class TestBlockTable:
    def test_empty(self) -> None:
        bt = BlockTable(n_layers=4, page_size=16)
        for l in range(4):
            assert bt.num_logical_pages(l) == 0
            assert bt.num_tokens(l) == 0

    def test_append_and_get(self) -> None:
        bt = BlockTable(n_layers=2, page_size=16)
        bt.append_page(0, 10)
        bt.append_page(0, 20)
        bt.append_page(1, 30)
        assert bt.num_logical_pages(0) == 2
        assert bt.num_logical_pages(1) == 1
        assert bt.num_tokens(0) == 32
        assert bt.get_page(0, 0) == 10
        assert bt.get_page(0, 1) == 20
        assert bt.get_page(1, 0) == 30

    def test_get_page_for_token(self) -> None:
        bt = BlockTable(n_layers=1, page_size=16)
        bt.append_page(0, 100)
        bt.append_page(0, 200)
        # Tokens 0-15 → page 0, tokens 16-31 → page 1
        assert bt.get_page_for_token(0, 0) == 100
        assert bt.get_page_for_token(0, 15) == 100
        assert bt.get_page_for_token(0, 16) == 200
        assert bt.get_page_for_token(0, 31) == 200

    def test_all_page_ids(self) -> None:
        bt = BlockTable(n_layers=3, page_size=16)
        bt.append_page(0, 1)
        bt.append_page(0, 2)
        bt.append_page(1, 3)
        bt.append_page(2, 2)  # page 2 shared across layers? not in v1, but
                              # the data structure doesn't prohibit it
        assert bt.all_page_ids() == {1, 2, 3}

    def test_chunk_page_ids(self) -> None:
        bt = BlockTable(n_layers=2, page_size=16)
        # Chunk 0 = pages 0..15 per layer
        for l in range(2):
            for i in range(32):  # 32 pages per layer = 2 chunks
                bt.append_page(l, l * 100 + i)

        chunk0 = bt.chunk_page_ids(0, pages_per_chunk=16)
        chunk1 = bt.chunk_page_ids(1, pages_per_chunk=16)
        # Chunk 0 has pages 0..15 in both layers = 32 pages total
        assert len(chunk0) == 32
        assert 0 in chunk0
        assert 100 in chunk0
        assert 15 in chunk0
        assert 115 in chunk0
        # Chunk 1 has pages 16..31
        assert len(chunk1) == 32
        assert 16 in chunk1
        assert 116 in chunk1

    def test_is_chunk_complete(self) -> None:
        bt = BlockTable(n_layers=2, page_size=16)
        # Add 20 pages per layer (1 full chunk + 4 partial)
        for l in range(2):
            for i in range(20):
                bt.append_page(l, l * 100 + i)
        assert bt.is_chunk_complete(0, pages_per_chunk=16)
        assert not bt.is_chunk_complete(1, pages_per_chunk=16)
        assert not bt.is_chunk_complete(2, pages_per_chunk=16)

    def test_chunk_partial(self) -> None:
        """Requesting a chunk that's only partially written returns
        only the existing pages (fewer than pages_per_chunk)."""
        bt = BlockTable(n_layers=2, page_size=16)
        for l in range(2):
            for i in range(10):  # less than one full chunk
                bt.append_page(l, l * 10 + i)
        chunk_pages = bt.chunk_page_ids(0, pages_per_chunk=16)
        assert len(chunk_pages) == 20  # 10 pages × 2 layers


# ---------------------------------------------------------------------------
# Integration: allocator + block table working together
# ---------------------------------------------------------------------------

class TestAllocatorBlockTableIntegration:
    def test_append_allocates(self) -> None:
        """Pages are allocated on demand and tracked in the block table."""
        alloc = PageAllocator(100)
        bt = BlockTable(n_layers=4, page_size=16)

        # Simulate writing 100 tokens → 7 pages per layer (ceil(100/16))
        for token_idx in range(100):
            for l in range(4):
                page_idx = token_idx // 16
                if bt.num_logical_pages(l) <= page_idx:
                    pid = alloc.alloc()
                    bt.append_page(l, pid)

        # ceil(100/16) = 7 pages per layer
        assert bt.num_logical_pages(0) == 7
        # 4 layers × 7 pages = 28 pages used
        assert alloc.used_pages == 28
        assert alloc.free_pages == 72

    def test_evict_chunk_frees_pages(self) -> None:
        """Evicting a chunk frees its pages back to the allocator."""
        alloc = PageAllocator(100)
        bt = BlockTable(n_layers=2, page_size=16)

        # Fill 3 full chunks (16 pages each, 2 layers)
        for l in range(2):
            for i in range(48):  # 3 chunks worth
                bt.append_page(l, alloc.alloc())

        assert alloc.used_pages == 96  # 48 * 2

        # Evict chunk 0
        chunk0_pages = bt.chunk_page_ids(0, pages_per_chunk=16)
        assert len(chunk0_pages) == 32
        alloc.free_many(chunk0_pages)
        assert alloc.free_pages == 36  # 32 freed + 4 original

    def test_cross_request_reuse(self) -> None:
        """After request 1 frees all pages, request 2 reuses them."""
        alloc = PageAllocator(50)

        # Request 1: use 30 pages
        bt1 = BlockTable(n_layers=2, page_size=16)
        for l in range(2):
            for _ in range(15):
                bt1.append_page(l, alloc.alloc())
        assert alloc.used_pages == 30

        # End request 1: free all
        alloc.free_many(list(bt1.all_page_ids()))
        assert alloc.free_pages == 50
        assert alloc.used_pages == 0

        # Request 2: use 40 pages
        bt2 = BlockTable(n_layers=4, page_size=16)
        for l in range(4):
            for _ in range(10):
                bt2.append_page(l, alloc.alloc())
        assert alloc.used_pages == 40
