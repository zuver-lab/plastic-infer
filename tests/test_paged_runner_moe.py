"""Tests for the composed MoE + paged-KV runner (M1 + M2).

Gates:
  1. Paged MoE prefill/decode == flat MoE runner and HF Mixtral.
  2. Prefix-loaded cache + incremental prefill == full prefill (D10),
     with experts served by the slot pool.
  3. Tight slot budget still identical (D5); hit-rate observable.
"""

from __future__ import annotations

import pytest
import torch
from transformers import MixtralForCausalLM

from plastic_infer.exec.runner import DenseKVCache, DenseWeights, ModelConfig
from plastic_infer.exec.runner_moe import decode_step_moe, prefill_forward_moe
from plastic_infer.exec.runner_moe_paged import (
    decode_step_moe_paged,
    prefill_forward_moe_paged,
)
from plastic_infer.exec.runner_paged import make_kv_config
from plastic_infer.kv.kv_store import KVConfig, KVStore
from plastic_infer.store.experts import ExpertBank, ExpertSlotPool

from test_equiv_smoke_moe import (  # noqa: F401
    _extract_moe_weights,
    _make_moe_config,
    _tiny_mixtral_config,
)


@pytest.fixture(scope="module")
def hf_model() -> MixtralForCausalLM:
    torch.manual_seed(42)
    model = MixtralForCausalLM(_tiny_mixtral_config())
    model.eval()
    return model


@pytest.fixture(scope="module")
def config(hf_model: MixtralForCausalLM) -> ModelConfig:
    return _make_moe_config(hf_model.config)


@pytest.fixture(scope="module")
def weights(hf_model: MixtralForCausalLM) -> DenseWeights:
    dense, _bank = _extract_moe_weights(hf_model)
    return DenseWeights(dense)


@pytest.fixture(scope="module")
def bank(hf_model: MixtralForCausalLM) -> ExpertBank:
    _dense, bank = _extract_moe_weights(hf_model)
    return bank


def _pool(bank: ExpertBank, n_slots: int) -> ExpertSlotPool:
    one_expert = bank.bytes(next(iter(bank.keys())))
    return ExpertSlotPool(bank, budget_bytes=n_slots * one_expert)


def _assert_allclose(a: torch.Tensor, b: torch.Tensor, where: str) -> None:
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-2), (
        f"{where}: max diff = {(a - b).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Equivalence: paged MoE == flat MoE == HF
# ---------------------------------------------------------------------------

class TestPagedMoEEquivalence:
    def test_prefill_matches_flat(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
    ) -> None:
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (16,))

        flat_pool = _pool(bank, n_slots=16)
        flat_kv = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)
        flat_logits = prefill_forward_moe(
            weights, config, flat_pool, flat_kv, input_ids,
        )

        store = KVStore(make_kv_config(config, max_pages=512))
        paged_pool = _pool(bank, n_slots=16)
        cache = store.new_request()
        paged_logits = prefill_forward_moe_paged(
            weights, config, paged_pool, store, cache, input_ids,
        )

        _assert_allclose(paged_logits, flat_logits, "paged vs flat prefill")

    def test_prefill_matches_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: MixtralForCausalLM,
    ) -> None:
        torch.manual_seed(1)
        input_ids = torch.randint(0, config.vocab_size, (12,))

        store = KVStore(make_kv_config(config, max_pages=512))
        pool = _pool(bank, n_slots=16)
        cache = store.new_request()
        our_logits = prefill_forward_moe_paged(
            weights, config, pool, store, cache, input_ids,
        )

        with torch.no_grad():
            hf_logits = hf_model(input_ids.unsqueeze(0)).logits[0, -1]

        _assert_allclose(our_logits, hf_logits, "paged prefill vs HF")

    def test_decode_matches_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: MixtralForCausalLM,
    ) -> None:
        torch.manual_seed(2)
        prompt_len, n_decode = 6, 4
        input_ids = torch.randint(0, config.vocab_size, (prompt_len,))

        store = KVStore(make_kv_config(config, max_pages=512))
        pool = _pool(bank, n_slots=16)
        cache = store.new_request()
        our_prefill = prefill_forward_moe_paged(
            weights, config, pool, store, cache, input_ids,
        )
        our_tokens: list[int] = [int(our_prefill.argmax().item())]
        our_logits: list[torch.Tensor] = [our_prefill]

        for _ in range(n_decode):
            logits = decode_step_moe_paged(
                weights, config, pool, store, cache, our_tokens[-1],
            )
            our_logits.append(logits)
            our_tokens.append(int(logits.argmax().item()))

        with torch.no_grad():
            full_ids = torch.tensor(
                input_ids.tolist() + our_tokens[:n_decode],
            ).unsqueeze(0)
            hf_logits_all = hf_model(full_ids).logits[0]

        for step in range(n_decode + 1):
            pos = prompt_len - 1 + step
            _assert_allclose(our_logits[step], hf_logits_all[pos],
                             f"paged decode step {step}")

    def test_decode_tight_slots_matches_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: MixtralForCausalLM,
    ) -> None:
        """D5: 2 expert slots + paged KV — constant eviction, same logits."""
        torch.manual_seed(3)
        prompt_len, n_decode = 3, 3
        input_ids = torch.randint(0, config.vocab_size, (prompt_len,))

        store = KVStore(make_kv_config(config, max_pages=512))
        pool = _pool(bank, n_slots=2)
        cache = store.new_request()
        our_prefill = prefill_forward_moe_paged(
            weights, config, pool, store, cache, input_ids,
        )
        our_tokens: list[int] = [int(our_prefill.argmax().item())]
        our_logits: list[torch.Tensor] = [our_prefill]

        for _ in range(n_decode):
            logits = decode_step_moe_paged(
                weights, config, pool, store, cache, our_tokens[-1],
            )
            our_logits.append(logits)
            our_tokens.append(int(logits.argmax().item()))

        with torch.no_grad():
            full_ids = torch.tensor(
                input_ids.tolist() + our_tokens[:n_decode],
            ).unsqueeze(0)
            hf_logits_all = hf_model(full_ids).logits[0]

        for step in range(n_decode + 1):
            pos = prompt_len - 1 + step
            _assert_allclose(our_logits[step], hf_logits_all[pos],
                             f"tight paged decode step {step}")

        assert 0.0 <= pool.hit_rate <= 1.0
        assert pool.pool.used_bytes <= pool.pool.budget_bytes


# ---------------------------------------------------------------------------
# Prefix reuse (D10) with experts served by the slot pool
# ---------------------------------------------------------------------------

class TestPagedMoEPrefixReuse:
    def test_prefix_load_plus_tail_equals_full(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
    ) -> None:
        """Request 2 shares a 24-token prefix with request 1: loading it
        and only computing the 5-token tail yields the same final logits
        as a full prefill from scratch."""
        torch.manual_seed(10)
        small_cfg = KVConfig(
            n_layers=config.n_layers,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            page_size=4, pages_per_chunk=2,   # 8 tokens per chunk
            max_pages=512, max_host_chunks=32,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        prefix_len, tail_len = 24, 5
        all_tokens = torch.randint(0, config.vocab_size, (prefix_len + tail_len,))
        prefix, tail = all_tokens[:prefix_len], all_tokens[prefix_len:]

        # One store holds the host L1 cache across requests.
        store = KVStore(small_cfg)
        c1 = store.new_request()
        prefill_forward_moe_paged(
            weights, config, _pool(bank, n_slots=8), store, c1, prefix,
        )
        c1.set_token_ids(prefix.tolist())
        assert store.sink_completed_chunks(c1) == 3
        store.free_request(c1)

        # Request 2: load prefix, prefill only the tail
        c2 = store.new_request()
        matched, keys = store.load_prefix(c2, all_tokens.tolist())
        assert matched == prefix_len
        assert len(keys) == 3
        tail_logits = prefill_forward_moe_paged(
            weights, config, _pool(bank, n_slots=8), store, c2, tail,
            start_pos=prefix_len,
        )

        # Reference: full prefill from scratch
        store_ref = KVStore(small_cfg)
        c_ref = store_ref.new_request()
        ref_logits = prefill_forward_moe_paged(
            weights, config, _pool(bank, n_slots=8), store_ref, c_ref,
            all_tokens,
        )

        _assert_allclose(tail_logits, ref_logits, "prefix reuse tail")

    def test_prefix_load_tight_slots_equals_full(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
    ) -> None:
        """D10 + D5: prefix reuse under a 2-expert slot budget."""
        torch.manual_seed(11)
        small_cfg = KVConfig(
            n_layers=config.n_layers, n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            page_size=4, pages_per_chunk=2,
            max_pages=512, max_host_chunks=32,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        prefix_len, tail_len = 16, 4
        all_tokens = torch.randint(0, config.vocab_size, (prefix_len + tail_len,))
        prefix, tail = all_tokens[:prefix_len], all_tokens[prefix_len:]

        store = KVStore(small_cfg)
        c1 = store.new_request()
        prefill_forward_moe_paged(
            weights, config, _pool(bank, n_slots=2), store, c1, prefix,
        )
        c1.set_token_ids(prefix.tolist())
        store.sink_completed_chunks(c1)
        store.free_request(c1)

        c2 = store.new_request()
        matched, _ = store.load_prefix(c2, all_tokens.tolist())
        assert matched == prefix_len
        tail_logits = prefill_forward_moe_paged(
            weights, config, _pool(bank, n_slots=2), store, c2, tail,
            start_pos=prefix_len,
        )

        store_ref = KVStore(small_cfg)
        c_ref = store_ref.new_request()
        ref_logits = prefill_forward_moe_paged(
            weights, config, _pool(bank, n_slots=2), store_ref, c_ref,
            all_tokens,
        )

        _assert_allclose(tail_logits, ref_logits, "tight prefix reuse tail")

    def test_partial_prefix_hit(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
    ) -> None:
        """Only matching chunks are loaded; the rest is computed (D10)."""
        torch.manual_seed(12)
        small_cfg = KVConfig(
            n_layers=config.n_layers, n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            page_size=4, pages_per_chunk=2,
            max_pages=512, max_host_chunks=32,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        chunk_tokens = small_cfg.tokens_per_chunk  # 8

        req1 = torch.randint(0, config.vocab_size, (chunk_tokens * 2,))
        store = KVStore(small_cfg)
        c1 = store.new_request()
        prefill_forward_moe_paged(
            weights, config, _pool(bank, n_slots=8), store, c1, req1,
        )
        c1.set_token_ids(req1.tolist())
        store.sink_completed_chunks(c1)
        store.free_request(c1)

        # Same first chunk, different second chunk + extra tokens
        req2 = req1.clone()
        req2[chunk_tokens:] = torch.randint(0, config.vocab_size,
                                            (chunk_tokens,))
        req2 = torch.cat([req2, torch.randint(0, config.vocab_size, (4,))])

        c2 = store.new_request()
        matched, keys = store.load_prefix(c2, req2.tolist())
        assert matched == chunk_tokens          # only first chunk hits
        assert len(keys) == 1
