"""Qwen3Moe equivalence anchor: our composed runner == HuggingFace.

This is the real-model milestone's anchor: a *Qwen3Moe*-shaped tiny model
(norm_topk_prob, per-head QK-norm, fused 3D experts, router named
mlp.gate) exercised through the same mapping/splitting helpers the
converter uses, then run through the composed runner (paged KV + slot
pool). If logits match HF token-by-token, the following are all correct
together:

  - HF -> canonical name mapping (mlp.gate -> mlp.router)
  - fused gate_up_proj/down_proj -> per-expert w1/w2/w3 split
  - per-head QK RMSNorm (q_norm/k_norm) placement before RoPE
  - norm_topk_prob routing == our softmax-over-top-k
  - qwen3_moe head_dim from config (16 here, 128 for the real model)

Runs in fp32 on CPU so numerical noise is negligible (atol=1e-2).
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from plastic_infer.exec.runner import DenseWeights, ModelConfig
from plastic_infer.exec.runner_moe_paged import (
    decode_step_moe_paged,
    prefill_forward_moe_paged,
)
from plastic_infer.exec.runner_paged import make_kv_config
from plastic_infer.kv.kv_store import KVStore
from plastic_infer.store.experts import ExpertBank, ExpertSlotPool
from plastic_infer.weights.convert import split_state_dict


# ---------------------------------------------------------------------------
# Tiny Qwen3Moe model + shared build helpers (reused by the other tests)
# ---------------------------------------------------------------------------

def tiny_qwen3moe_config() -> Qwen3MoeConfig:
    """A tiny Qwen3Moe mirroring the real model's *shape*: explicit
    head_dim, per-head QK-norm, norm_topk_prob, fused experts."""
    return Qwen3MoeConfig(
        vocab_size=320,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )


def tiny_qwen3moe_config_dict() -> dict:
    """The config as it would appear in a converted model's config.json
    (real-model field names: num_experts, head_dim, rope_theta)."""
    return {
        "model_type": "qwen3_moe",
        "vocab_size": 320,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "norm_topk_prob": True,
    }


def make_model_config(hf_cfg: Qwen3MoeConfig) -> ModelConfig:
    """Our runner config from an HF Qwen3Moe config.

    head_dim is explicit in Qwen3 (not hidden//heads); rope_base comes
    from rope_parameters (or top-level rope_theta for the real model).
    """
    return ModelConfig(
        n_layers=hf_cfg.num_hidden_layers,
        n_heads=hf_cfg.num_attention_heads,
        n_kv_heads=hf_cfg.num_key_value_heads,
        head_dim=hf_cfg.head_dim,
        hidden_dim=hf_cfg.hidden_size,
        intermediate_dim=hf_cfg.moe_intermediate_size,
        vocab_size=hf_cfg.vocab_size,
        max_seq_len=hf_cfg.max_position_embeddings,
        rope_base=float(hf_cfg.rope_parameters["rope_theta"]),
        dtype=torch.float32,
        qk_norm=True,
        n_experts=hf_cfg.num_local_experts,
        n_experts_per_tok=hf_cfg.num_experts_per_tok,
    )


def build_weights_and_bank(
    hf_model: Qwen3MoeForCausalLM,
) -> tuple[DenseWeights, ExpertBank]:
    """HF state_dict -> (DenseWeights, ExpertBank) via the converter's
    own mapping/splitting helpers (this is what convert.py does on disk)."""
    dense, experts = split_state_dict(
        hf_model.state_dict(),
        num_experts=hf_model.config.num_local_experts,
        moe_intermediate_size=hf_model.config.moe_intermediate_size,
    )
    bank = ExpertBank()
    for (layer, eid), w in experts.items():
        bank.add(layer, eid, w)
    return DenseWeights(dense), bank


@pytest.fixture(scope="module")
def hf_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(42)
    model = Qwen3MoeForCausalLM(tiny_qwen3moe_config())
    model.eval()
    return model


@pytest.fixture(scope="module")
def config(hf_model: Qwen3MoeForCausalLM) -> ModelConfig:
    return make_model_config(hf_model.config)


@pytest.fixture(scope="module")
def weights(hf_model: Qwen3MoeForCausalLM) -> DenseWeights:
    w, _bank = build_weights_and_bank(hf_model)
    return w


@pytest.fixture(scope="module")
def bank(hf_model: Qwen3MoeForCausalLM) -> ExpertBank:
    _w, bank = build_weights_and_bank(hf_model)
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
# Prefill equivalence
# ---------------------------------------------------------------------------

class TestQwen3MoePrefillEquivalence:
    def test_prefill_last_logits_match_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: Qwen3MoeForCausalLM,
    ) -> None:
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (16,))
        store = KVStore(make_kv_config(config, max_pages=512))
        cache = store.new_request()
        our_logits = prefill_forward_moe_paged(
            weights, config, _pool(bank, n_slots=16), store, cache, input_ids,
        )

        with torch.no_grad():
            hf_logits = hf_model(input_ids.unsqueeze(0)).logits[0, -1]

        _assert_allclose(our_logits, hf_logits, "qwen3moe prefill last token")

    def test_prefill_all_positions_match_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: Qwen3MoeForCausalLM,
    ) -> None:
        """All positions match (prefixes of increasing length)."""
        torch.manual_seed(1)
        input_ids = torch.randint(0, config.vocab_size, (8,))

        with torch.no_grad():
            hf_logits_all = hf_model(input_ids.unsqueeze(0)).logits[0]

        for i in range(1, 9):
            store = KVStore(make_kv_config(config, max_pages=512))
            cache = store.new_request()
            our_logits = prefill_forward_moe_paged(
                weights, config, _pool(bank, n_slots=16), store, cache,
                input_ids[:i],
            )
            _assert_allclose(our_logits, hf_logits_all[i - 1],
                             f"position {i - 1}")

    def test_prefill_tight_slots_still_match_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: Qwen3MoeForCausalLM,
    ) -> None:
        """D5: 2 slots force constant eviction yet logits are identical."""
        torch.manual_seed(2)
        input_ids = torch.randint(0, config.vocab_size, (8,))
        store = KVStore(make_kv_config(config, max_pages=512))
        pool = _pool(bank, n_slots=2)
        cache = store.new_request()
        our_logits = prefill_forward_moe_paged(
            weights, config, pool, store, cache, input_ids,
        )

        with torch.no_grad():
            hf_logits = hf_model(input_ids.unsqueeze(0)).logits[0, -1]

        _assert_allclose(our_logits, hf_logits, "qwen3moe tight prefill")
        assert pool.misses > 0                       # eviction actually ran
        assert pool.pool.used_bytes <= pool.pool.budget_bytes


# ---------------------------------------------------------------------------
# Decode equivalence
# ---------------------------------------------------------------------------

class TestQwen3MoeDecodeEquivalence:
    def test_decode_multi_step_matches_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: Qwen3MoeForCausalLM,
    ) -> None:
        torch.manual_seed(3)
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
                             f"decode step {step} (pos {pos})")

    def test_decode_tight_slots_matches_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: Qwen3MoeForCausalLM,
    ) -> None:
        """D5 + paged KV under a 2-slot budget."""
        torch.manual_seed(4)
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
                             f"tight decode step {step}")

        assert 0.0 <= pool.hit_rate <= 1.0
        assert pool.pool.used_bytes <= pool.pool.budget_bytes
