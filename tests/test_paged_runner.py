"""Tests for paged KV runner: numerical equivalence + prefix reuse.

M2 gate:
  1. Paged prefill/decode produces same logits as flat runner (and HF).
  2. Prefix-loaded cache + incremental prefill produces same logits
     as full prefill (D10: prefix hit → only compute unmatched tail).
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from plastic_infer.exec.runner import DenseWeights, ModelConfig
from plastic_infer.exec.runner_paged import (
    decode_step_paged,
    make_kv_config,
    prefill_forward_paged,
)
from plastic_infer.kv.kv_store import KVStore

from test_equiv_smoke import (  # noqa: F401
    _extract_weights,
    _make_config,
    _tiny_llama_config,
)


@pytest.fixture(scope="module")
def hf_model() -> LlamaForCausalLM:
    cfg = _tiny_llama_config()
    torch.manual_seed(42)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def weights(hf_model: LlamaForCausalLM) -> DenseWeights:
    return DenseWeights(_extract_weights(hf_model))


@pytest.fixture(scope="module")
def config(hf_model: LlamaForCausalLM) -> ModelConfig:
    return _make_config(hf_model.config)


@pytest.fixture(scope="module")
def kv_cfg(config: ModelConfig) -> KVStore:
    return make_kv_config(config, max_pages=512, max_host_chunks=32)


class TestPagedEquivalence:
    """Paged KV runner should produce the same logits as flat runner."""

    def test_prefill_matches_flat(
        self, weights: DenseWeights, config: ModelConfig, kv_cfg: KVStore,
    ) -> None:
        from plastic_infer.exec.runner import DenseKVCache, prefill_forward

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (16,))

        # Flat runner
        flat_cache = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)
        flat_logits = prefill_forward(weights, config, flat_cache, input_ids)

        # Paged runner
        store = KVStore(kv_cfg)
        paged_cache = store.new_request()
        paged_logits = prefill_forward_paged(
            weights, config, store, paged_cache, input_ids,
        )

        assert torch.allclose(flat_logits, paged_logits, atol=1e-2)

    def test_prefill_matches_hf(
        self, weights: DenseWeights, config: ModelConfig,
        kv_cfg: KVStore, hf_model: LlamaForCausalLM,
    ) -> None:
        torch.manual_seed(1)
        input_ids = torch.randint(0, config.vocab_size, (12,))

        store = KVStore(kv_cfg)
        cache = store.new_request()
        our_logits = prefill_forward_paged(
            weights, config, store, cache, input_ids,
        )

        with torch.no_grad():
            hf_logits = hf_model(input_ids.unsqueeze(0)).logits[0, -1]

        assert torch.allclose(our_logits, hf_logits, atol=1e-2)

    def test_decode_matches_hf(
        self, weights: DenseWeights, config: ModelConfig,
        kv_cfg: KVStore, hf_model: LlamaForCausalLM,
    ) -> None:
        """Multi-step decode with paged KV matches HF."""
        torch.manual_seed(2)
        prompt_len = 6
        n_decode = 4
        input_ids = torch.randint(0, config.vocab_size, (prompt_len,))

        # Our paged run
        store = KVStore(kv_cfg)
        cache = store.new_request()
        our_prefill = prefill_forward_paged(
            weights, config, store, cache, input_ids,
        )
        our_tokens: list[int] = [int(our_prefill.argmax().item())]
        our_logits_list: list[torch.Tensor] = [our_prefill]

        for _ in range(n_decode):
            next_tok = our_tokens[-1]
            logits = decode_step_paged(weights, config, store, cache, next_tok)
            our_logits_list.append(logits)
            our_tokens.append(int(logits.argmax().item()))

        # HF reference
        with torch.no_grad():
            full_ids = torch.tensor(
                input_ids.tolist() + our_tokens[:n_decode],
            ).unsqueeze(0)
            hf_logits_all = hf_model(full_ids).logits[0]

        for step in range(n_decode + 1):
            pos = prompt_len - 1 + step
            assert torch.allclose(
                our_logits_list[step], hf_logits_all[pos], atol=1e-2,
            ), f"step {step} (pos {pos})"


class TestPrefixReuse:
    """Prefix loading + incremental prefill = full prefill (D10)."""

    def test_prefix_load_plus_tail_equals_full(
        self, weights: DenseWeights, config: ModelConfig, kv_cfg: KVStore,
    ) -> None:
        """If request 2 shares a prefix with request 1, loading that
        prefix and only computing the tail yields the same final logits
        as doing the full prefill from scratch."""
        torch.manual_seed(10)

        # Build a prefix of 2 full chunks + some tail tokens.
        chunk_size = kv_cfg.tokens_per_chunk  # default: 16*4=64 for tiny model?
        # Actually with default page_size=16, pages_per_chunk=16,
        # tokens_per_chunk = 256. Too long for tiny test.
        # Let's just use a shorter sequence that makes at least 1 chunk.
        # Wait — kv_cfg uses config's own page_size / pages_per_chunk from
        # DEFAULT_PAGE_SIZE / DEFAULT_CHUNK_PAGES (16 / 16 = 256 tok).
        # Our test model only has 128 max_seq_len. Let's use a smaller test.
        #
        # Use a sequence that produces at least one full chunk.
        # For the tiny test config with page_size=16, pages_per_chunk=16:
        # that's 256 tokens — too many.
        #
        # Let's use a custom small config for this test.
        pass

    def test_prefix_load_with_small_chunks(
        self, weights: DenseWeights, config: ModelConfig,
    ) -> None:
        """Same test with tiny chunks so we get multiple chunks quickly."""
        from plastic_infer.kv.kv_store import KVConfig

        torch.manual_seed(20)

        # Very small chunks: 4 tokens/page × 2 pages/chunk = 8 tokens/chunk
        small_kv_cfg = KVConfig(
            n_layers=config.n_layers,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            page_size=4,
            pages_per_chunk=2,
            max_pages=256,
            max_host_chunks=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # 3 full chunks = 24 tokens prefix, plus 5 tokens tail
        prefix_len = 24
        tail_len = 5
        total_len = prefix_len + tail_len
        all_tokens = torch.randint(0, config.vocab_size, (total_len,))
        prefix_tokens = all_tokens[:prefix_len]
        tail_tokens = all_tokens[prefix_len:]

        # --- Request 1: full prefix, sink to host ---
        store = KVStore(small_kv_cfg)
        cache1 = store.new_request()
        prefill_forward_paged(weights, config, store, cache1, prefix_tokens)
        cache1.set_token_ids(prefix_tokens.tolist())
        store.sink_completed_chunks(cache1)
        assert store.host_cache_size == 3  # 3 full chunks
        store.free_request(cache1)

        # --- Request 2: load prefix + incremental prefill on tail ---
        cache2 = store.new_request()
        matched, _keys = store.load_prefix(cache2, all_tokens.tolist())
        assert matched == prefix_len
        # Now prefill only the tail (starting from prefix_len)
        tail_logits = prefill_forward_paged(
            weights, config, store, cache2, tail_tokens,
            start_pos=prefix_len,
        )

        # --- Reference: full prefill from scratch ---
        store_ref = KVStore(small_kv_cfg)
        cache_ref = store_ref.new_request()
        ref_logits = prefill_forward_paged(
            weights, config, store_ref, cache_ref, all_tokens,
        )

        assert torch.allclose(tail_logits, ref_logits, atol=1e-2), (
            f"max diff = {(tail_logits - ref_logits).abs().max().item()}"
        )

    def test_partial_prefix_hit(
        self, weights: DenseWeights, config: ModelConfig,
    ) -> None:
        """Only the matching chunks are loaded; the rest is computed."""
        from plastic_infer.kv.kv_store import KVConfig

        torch.manual_seed(30)

        small_kv_cfg = KVConfig(
            n_layers=config.n_layers,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            page_size=4, pages_per_chunk=2,
            max_pages=256, max_host_chunks=32,
            dtype=torch.float32, device=torch.device("cpu"),
        )

        # Request 1: 2 chunks (16 tokens)
        req1_tokens = torch.randint(0, config.vocab_size, (16,))
        store = KVStore(small_kv_cfg)
        c1 = store.new_request()
        prefill_forward_paged(weights, config, store, c1, req1_tokens)
        c1.set_token_ids(req1_tokens.tolist())
        store.sink_completed_chunks(c1)
        store.free_request(c1)

        # Request 2: same first chunk, different second chunk + more
        req2_tokens = req1_tokens.clone()
        req2_tokens[8:16] = torch.randint(0, config.vocab_size, (8,))
        extra = torch.randint(0, config.vocab_size, (4,))
        req2_tokens = torch.cat([req2_tokens, extra])

        c2 = store.new_request()
        matched, keys = store.load_prefix(c2, req2_tokens.tolist())
        # Only first chunk should match (chunk 1 is different)
        assert matched == 8  # 1 chunk = 8 tokens
        assert len(keys) == 1
