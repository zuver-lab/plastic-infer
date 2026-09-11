"""Paged KV cache: free-list page allocator + block table.

This is the allocation/addressing substrate for KV on GPU (§5.5).
A page (block) holds PAGE_SIZE tokens for one layer's K and V heads.

v1 design:
  - Global free-list of physical page IDs.
  - One block table per active request (C=1 → one request at a time).
  - Block table maps (layer, logical_token) → physical_page_id.
  - Pages are allocated on demand; freed back to free-list on eviction
    or when the request ends.
  - No radix tree, no cross-request page-level sharing (out of scope).

This module is pure data-structure logic — no tensors, no GPU calls.
The actual page buffers (torch tensors) live in kv_store; this module
just tracks which pages exist and which logical positions they map to.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Defaults (matching DESIGN.md §5.5)
DEFAULT_PAGE_SIZE: int = 16   # tokens per page
DEFAULT_CHUNK_PAGES: int = 16  # pages per chunk = 256 tokens


@dataclass
class BlockTable:
    """Per-request mapping from logical positions to physical pages.

    Layout: for each layer l, `pages[l]` is a list where index i
    is the physical page ID holding logical tokens [i*PAGE_SIZE, (i+1)*PAGE_SIZE).

    C=1 and single-request: this is just a linear list that grows on append.
    No holes, no fragmentation within a request's logical space.
    """

    n_layers: int
    page_size: int
    pages: list[list[int]] = field(default_factory=list)  # [layer][logical_page_idx] -> phys_page_id

    def __post_init__(self) -> None:
        if not self.pages:
            self.pages = [[] for _ in range(self.n_layers)]

    def num_logical_pages(self, layer: int) -> int:
        return len(self.pages[layer])

    def num_tokens(self, layer: int) -> int:
        return self.num_logical_pages(layer) * self.page_size

    def get_page(self, layer: int, logical_page_idx: int) -> int:
        return self.pages[layer][logical_page_idx]

    def get_page_for_token(self, layer: int, token_idx: int) -> int:
        """Return the physical page holding token_idx in layer l."""
        return self.pages[layer][token_idx // self.page_size]

    def append_page(self, layer: int, phys_page_id: int) -> None:
        """Append a new physical page to the logical end of layer l."""
        self.pages[layer].append(phys_page_id)

    def all_page_ids(self) -> set[int]:
        """All physical pages referenced by this block table."""
        ids: set[int] = set()
        for layer_pages in self.pages:
            ids.update(layer_pages)
        return ids

    def chunk_page_ids(self, chunk_idx: int, pages_per_chunk: int) -> list[int]:
        """Return all physical page IDs for chunk chunk_idx across all layers.

        A chunk covers logical page indices [chunk_idx*pages_per_chunk,
        (chunk_idx+1)*pages_per_chunk) in every layer.
        """
        start = chunk_idx * pages_per_chunk
        end = start + pages_per_chunk
        ids: list[int] = []
        for layer_pages in self.pages:
            for pidx in range(start, min(end, len(layer_pages))):
                ids.append(layer_pages[pidx])
        return ids

    def is_chunk_complete(self, chunk_idx: int, pages_per_chunk: int) -> bool:
        """True if chunk chunk_idx has all its pages in all layers."""
        start = chunk_idx * pages_per_chunk
        end = start + pages_per_chunk
        for layer_pages in self.pages:
            if len(layer_pages) < end:
                return False
        return True


class PageAllocator:
    """Global free-list of physical page IDs.

    Pages are numbered 0..total_pages-1. We reuse freed pages
    immediately — no compaction needed because pages are uniform size
    and addressed by ID, not by offset.
    """

    def __init__(self, total_pages: int) -> None:
        assert total_pages > 0
        self._total = total_pages
        # Free list: stack of available page IDs (LIFO — best for cache
        # locality when we immediately reuse recently-freed pages).
        self._free: list[int] = list(range(total_pages - 1, -1, -1))
        self._used: set[int] = set()

    @property
    def total_pages(self) -> int:
        return self._total

    @property
    def free_pages(self) -> int:
        return len(self._free)

    @property
    def used_pages(self) -> int:
        return self._total - len(self._free)

    def alloc(self) -> int:
        """Allocate one page. Returns its physical page ID.

        Raises IndexError if no pages are free. Caller should evict
        chunks to free up pages before calling alloc again.
        """
        if not self._free:
            raise IndexError("out of pages")
        pid = self._free.pop()
        self._used.add(pid)
        return pid

    def alloc_many(self, count: int) -> list[int]:
        """Allocate `count` pages. Returns list of page IDs."""
        if count > self.free_pages:
            raise IndexError(
                f"need {count} pages, only {self.free_pages} free"
            )
        ids = [self._free.pop() for _ in range(count)]
        self._used.update(ids)
        return ids

    def free(self, page_id: int) -> None:
        """Return a page to the free list."""
        assert page_id in self._used, f"page {page_id} not in use"
        self._used.discard(page_id)
        self._free.append(page_id)

    def free_many(self, page_ids: list[int]) -> None:
        """Return multiple pages to the free list."""
        for pid in page_ids:
            assert pid in self._used, f"page {pid} not in use"
            self._used.discard(pid)
        self._free.extend(page_ids)

    def is_used(self, page_id: int) -> bool:
        return page_id in self._used
