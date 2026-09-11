"""Tests for kv.kv_store — paged KV + chunk sinking + prefix loading.

Tests use fake store (all CPU tensors) — no GPU needed.
"""

from __future__ import annotations

import pytest
import torch

from plastic_infer.kv.kv_store import KVConfig, KVStore


@pytest.fixture
def cfg() -> KVConfig:
    """Small KV config for fast tests."""
    return KVConfig(
        n_layers=2,
        n_kv_heads=2,
        head_dim=4,
        page_size=4,          # 4 tokens per page
        pages_per_chunk=2,    # 8 tokens per chunk
        max_pages=64,         # 64 pages total
        max_host_chunks=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


@pytest.fixture
def store(cfg: KVConfig) -> KVStore:
    return KVStore(cfg)


def _rand_kv(cfg: KVConfig, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random K/V for one token: [n_layers, n_kv_heads, head_dim]."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, generator=g)
    v = torch.randn(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, generator=g)
    return k, v


class TestBasicAppend:
    def test_append_increments_length(self, store: KVStore) -> None:
        cache = store.new_request()
        k, v = _rand_kv(store.config, seed=0)
        cache.append_token(k, v)
        assert cache.length == 1
        assert cache.block_table.num_logical_pages(0) == 1

    def test_append_page_full_allocates_new_page(
        self, store: KVStore,
    ) -> None:
        cache = store.new_request()
        page_size = store.config.page_size  # 4
        for i in range(page_size + 1):   # 5 tokens → 2 pages
            k, v = _rand_kv(store.config, seed=i)
            cache.append_token(k, v)
        assert cache.length == page_size + 1
        assert cache.block_table.num_logical_pages(0) == 2
        assert store.gpu.allocator.used_pages == 4  # 2 layers × 2 pages

    def test_data_written_correctly(self, store: KVStore) -> None:
        cache = store.new_request()
        k, v = _rand_kv(store.config, seed=42)
        cache.append_token(k, v)

        # Check page 0, token 0, layer 0
        phys_0 = cache.block_table.get_page(0, 0)
        stored_k = store.gpu.k_pages[phys_0, :, :, 0]
        assert torch.allclose(stored_k, k[0])

        stored_v = store.gpu.v_pages[phys_0, :, :, 0]
        assert torch.allclose(stored_v, v[0])

    def test_free_request_reclaims_pages(self, store: KVStore) -> None:
        cache = store.new_request()
        for i in range(20):
            k, v = _rand_kv(store.config, seed=i)
            cache.append_token(k, v)
        used = store.gpu.allocator.used_pages
        assert used > 0
        store.free_request(cache)
        assert store.gpu.allocator.used_pages == 0
        assert store.gpu.allocator.free_pages == store.config.max_pages


class TestChunkSinking:
    def test_sink_one_chunk(self, store: KVStore) -> None:
        cache = store.new_request()
        chunk_size = store.config.tokens_per_chunk  # 8
        tokens = list(range(chunk_size))

        for i in range(chunk_size):
            k, v = _rand_kv(store.config, seed=i)
            cache.append_token(k, v)
        cache.set_token_ids(tokens)

        assert cache.num_complete_chunks() == 1
        assert len(cache.chunk_keys) == 1

        sunk = store.sink_completed_chunks(cache)
        assert sunk == 1
        assert store.host_cache_size == 1

    def test_sink_incremental(self, store: KVStore) -> None:
        cache = store.new_request()
        chunk_size = store.config.tokens_per_chunk
        all_tokens: list[int] = []

        # Write 1.5 chunks
        for i in range(int(chunk_size * 1.5)):
            k, v = _rand_kv(store.config, seed=i)
            cache.append_token(k, v)
            all_tokens.append(i)
        cache.set_token_ids(all_tokens)

        sunk = store.sink_completed_chunks(cache)
        assert sunk == 1    # only full chunk sinks
        assert store.host_cache_size == 1

        # Finish second chunk
        for i in range(int(chunk_size * 1.5), chunk_size * 2):
            k, v = _rand_kv(store.config, seed=i)
            cache.append_token(k, v)
            all_tokens.append(i)
        cache.set_token_ids(all_tokens)

        sunk = store.sink_completed_chunks(cache)
        assert sunk == 1    # one more chunk
        assert store.host_cache_size == 2

    def test_sunk_data_roundtrip(self, store: KVStore) -> None:
        """Data sunk to host should match what was in GPU pages."""
        cache = store.new_request()
        chunk_size = store.config.tokens_per_chunk
        tokens = list(range(chunk_size))

        for i in range(chunk_size):
            k, v = _rand_kv(store.config, seed=i * 7)
            cache.append_token(k, v)
        cache.set_token_ids(tokens)

        # Read back from GPU pages
        gpu_k = torch.zeros(store.config.n_layers, store.config.n_kv_heads,
                           store.config.head_dim, chunk_size)
        gpu_v = torch.zeros_like(gpu_k)
        for layer in range(store.config.n_layers):
            for t in range(chunk_size):
                page = t // store.config.page_size
                off = t % store.config.page_size
                phys = cache.block_table.get_page(layer, page)
                gpu_k[layer, :, :, t] = store.gpu.k_pages[phys, :, :, off]
                gpu_v[layer, :, :, t] = store.gpu.v_pages[phys, :, :, off]

        # Sink and compare
        store.sink_completed_chunks(cache)
        key = cache.chunk_keys[0].full_key
        host_k, host_v = store._host_cache[key]
        assert torch.allclose(gpu_k, host_k, atol=1e-6)
        assert torch.allclose(gpu_v, host_v, atol=1e-6)

    def test_host_eviction(self, store: KVStore) -> None:
        """Host cache respects max_host_chunks and evicts LRU."""
        cache = store.new_request()
        chunk_size = store.config.tokens_per_chunk
        n_chunks = store.config.max_host_chunks + 3  # 11 chunks
        total_tokens = n_chunks * chunk_size

        for i in range(total_tokens):
            k, v = _rand_kv(store.config, seed=i * 11)
            cache.append_token(k, v)
        all_tokens = list(range(total_tokens))
        cache.set_token_ids(all_tokens)

        sunk = store.sink_completed_chunks(cache)
        assert sunk == n_chunks
        assert store.host_cache_size == store.config.max_host_chunks


class TestPrefixLoad:
    def test_load_existing_prefix(self, store: KVStore) -> None:
        """Write + sink a prefix, then load it into a new request."""
        chunk_size = store.config.tokens_per_chunk

        # Request 1: write 2 chunks and sink
        cache1 = store.new_request()
        tokens1 = list(range(chunk_size * 2))
        for i in range(len(tokens1)):
            k, v = _rand_kv(store.config, seed=i)
            cache1.append_token(k, v)
        cache1.set_token_ids(tokens1)
        store.sink_completed_chunks(cache1)
        assert store.host_cache_size == 2

        # Save data from cache1 for comparison
        ref_k = [torch.zeros(store.config.n_layers, store.config.n_kv_heads,
                             store.config.head_dim, chunk_size)
                 for _ in range(2)]
        ref_v = [torch.zeros_like(ref_k[0]) for _ in range(2)]
        for ci in range(2):
            for layer in range(store.config.n_layers):
                for t in range(chunk_size):
                    global_t = ci * chunk_size + t
                    page = global_t // store.config.page_size
                    off = global_t % store.config.page_size
                    phys = cache1.block_table.get_page(layer, page)
                    ref_k[ci][layer, :, :, t] = store.gpu.k_pages[phys, :, :, off]
                    ref_v[ci][layer, :, :, t] = store.gpu.v_pages[phys, :, :, off]

        store.free_request(cache1)

        # Request 2: load prefix
        cache2 = store.new_request()
        matched, keys = store.load_prefix(cache2, tokens1)
        assert matched == chunk_size * 2
        assert len(keys) == 2
        assert cache2.length == chunk_size * 2

        # Compare loaded data with reference
        for ci in range(2):
            for layer in range(store.config.n_layers):
                for t in range(chunk_size):
                    global_t = ci * chunk_size + t
                    page = global_t // store.config.page_size
                    off = global_t % store.config.page_size
                    phys = cache2.block_table.get_page(layer, page)
                    assert torch.allclose(
                        store.gpu.k_pages[phys, :, :, off],
                        ref_k[ci][layer, :, :, t],
                    )

    def test_prefix_partial_hit(self, store: KVStore) -> None:
        """Load prefix with only first chunk matching."""
        chunk_size = store.config.tokens_per_chunk

        # Sink chunk 0
        cache1 = store.new_request()
        tokens_a = list(range(chunk_size))
        for i in range(chunk_size):
            k, v = _rand_kv(store.config, seed=i)
            cache1.append_token(k, v)
        cache1.set_token_ids(tokens_a)
        store.sink_completed_chunks(cache1)
        store.free_request(cache1)

        # Load a longer prefix that only matches chunk 0
        cache2 = store.new_request()
        long_tokens = list(range(chunk_size * 3))
        matched, keys = store.load_prefix(cache2, long_tokens)
        assert matched == chunk_size
        assert len(keys) == 1

    def test_prefix_no_hit(self, store: KVStore) -> None:
        cache = store.new_request()
        tokens = [999, 998, 997]  # random tokens, no match
        matched, keys = store.load_prefix(cache, tokens)
        assert matched == 0
        assert keys == []
        assert cache.length == 0

    def test_prefix_empty_cache(self, store: KVStore) -> None:
        cache = store.new_request()
        matched, keys = store.load_prefix(cache, [1, 2, 3])
        assert matched == 0


class TestEviction:
    def test_evict_frees_pages(self, store: KVStore) -> None:
        """Evicting chunks frees GPU pages."""
        chunk_size = store.config.tokens_per_chunk
        cache = store.new_request()
        # Write 4 full chunks
        for i in range(chunk_size * 4):
            k, v = _rand_kv(store.config, seed=i)
            cache.append_token(k, v)
        cache.set_token_ids(list(range(chunk_size * 4)))
        store.sink_completed_chunks(cache)

        used_before = store.gpu.allocator.used_pages
        # Need 1 page → evict at least one chunk (2 layers × 2 pages = 4 pages)
        freed = store.evict_chunks(need_pages=1, protected_request=cache)
        assert freed >= 1
        assert store.gpu.allocator.used_pages == used_before - freed
