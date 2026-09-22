"""Smoke integration test: MoE model prefill + decode vs HF reference.

M1 gate from DESIGN.md §7:
  - test_ops / test_equiv_smoke(MoE) green
  - decode expert hit-rate observable

The sparse expert path (routed subset through a budget-bounded LRU
slot pool) must produce the *same* logits as HF — with any slot
budget, even one that forces constant eviction (D5: numerics
independent of placement).

Uses transformers 5.14's Mixtral (no shared expert; experts stored as
fused gate_up_proj / down_proj). The router there softmaxes over all
experts, top-k-picks probabilities, then renormalizes — which equals
softmax over the top-k logits, exactly what topk_route does.
"""

from __future__ import annotations

import pytest
import torch
from transformers import MixtralConfig, MixtralForCausalLM

from plastic_infer.exec.runner import DenseKVCache, DenseWeights, ModelConfig
from plastic_infer.exec.runner_moe import decode_step_moe, prefill_forward_moe
from plastic_infer.store.experts import ExpertBank, ExpertSlotPool, ExpertWeights


# ---------------------------------------------------------------------------
# Build a tiny Mixtral model + extract weights
# ---------------------------------------------------------------------------

def _tiny_mixtral_config() -> MixtralConfig:
    return MixtralConfig(
        vocab_size=320,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        num_local_experts=8,
        num_experts_per_tok=2,
    )


def _rope_theta(cfg: MixtralConfig) -> float:
    return cfg.rope_parameters["rope_theta"]


def _extract_moe_weights(
    hf_model: MixtralForCausalLM,
) -> tuple[dict[str, torch.Tensor], ExpertBank]:
    """Extract HF Mixtral weights into dense dict + expert bank."""
    sd = hf_model.state_dict()
    cfg = hf_model.config
    inter = cfg.intermediate_size
    dense: dict[str, torch.Tensor] = {}
    bank = ExpertBank()

    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}"
        dense[f"layers.{i}.input_layernorm.weight"] = \
            sd[f"{prefix}.input_layernorm.weight"].clone()
        dense[f"layers.{i}.self_attn.q_proj.weight"] = \
            sd[f"{prefix}.self_attn.q_proj.weight"].clone()
        dense[f"layers.{i}.self_attn.k_proj.weight"] = \
            sd[f"{prefix}.self_attn.k_proj.weight"].clone()
        dense[f"layers.{i}.self_attn.v_proj.weight"] = \
            sd[f"{prefix}.self_attn.v_proj.weight"].clone()
        dense[f"layers.{i}.self_attn.o_proj.weight"] = \
            sd[f"{prefix}.self_attn.o_proj.weight"].clone()
        dense[f"layers.{i}.post_attention_layernorm.weight"] = \
            sd[f"{prefix}.post_attention_layernorm.weight"].clone()

        # Router (fused gate_up_proj = [E, 2I, H] -> w1 (gate) | w3 (up))
        dense[f"layers.{i}.mlp.router.weight"] = \
            sd[f"{prefix}.mlp.gate.weight"].clone()
        gate_up = sd[f"{prefix}.mlp.experts.gate_up_proj"]
        down = sd[f"{prefix}.mlp.experts.down_proj"]
        for e in range(cfg.num_local_experts):
            bank.add(i, e, ExpertWeights(
                w1=gate_up[e, :inter, :].clone(),   # [I, H] gate (silu)
                w2=down[e].clone(),                 # [H, I] down
                w3=gate_up[e, inter:, :].clone(),   # [I, H] up
            ))

    dense["embed_tokens.weight"] = sd["model.embed_tokens.weight"].clone()
    dense["norm.weight"] = sd["model.norm.weight"].clone()
    dense["lm_head.weight"] = sd["lm_head.weight"].clone()
    return dense, bank


def _make_moe_config(hf_cfg: MixtralConfig) -> ModelConfig:
    return ModelConfig(
        n_layers=hf_cfg.num_hidden_layers,
        n_heads=hf_cfg.num_attention_heads,
        n_kv_heads=hf_cfg.num_key_value_heads,
        head_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        hidden_dim=hf_cfg.hidden_size,
        intermediate_dim=hf_cfg.intermediate_size,
        vocab_size=hf_cfg.vocab_size,
        max_seq_len=hf_cfg.max_position_embeddings,
        rope_base=_rope_theta(hf_cfg),
        dtype=torch.float32,
        n_experts=hf_cfg.num_local_experts,
        n_experts_per_tok=hf_cfg.num_experts_per_tok,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
# Prefill equivalence
# ---------------------------------------------------------------------------

class TestMoEPrefillEquivalence:
    def test_prefill_last_logits_match_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: MixtralForCausalLM,
    ) -> None:
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (16,))
        pool = _pool(bank, n_slots=16)   # everything fits

        kv = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)
        our_logits = prefill_forward_moe(weights, config, pool, kv, input_ids)

        with torch.no_grad():
            hf_logits = hf_model(input_ids.unsqueeze(0)).logits[0, -1]

        _assert_allclose(our_logits, hf_logits, "prefill last token")

    def test_prefill_all_positions_match_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: MixtralForCausalLM,
    ) -> None:
        torch.manual_seed(1)
        input_ids = torch.randint(0, config.vocab_size, (8,))
        pool = _pool(bank, n_slots=8)    # half the experts — forces churn

        with torch.no_grad():
            hf_logits_all = hf_model(input_ids.unsqueeze(0)).logits[0]

        for i in range(1, 9):
            kv = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)
            our_logits = prefill_forward_moe(
                weights, config, pool, kv, input_ids[:i],
            )
            _assert_allclose(our_logits, hf_logits_all[i - 1],
                             f"position {i - 1}")

    def test_prefill_tight_slots_still_match_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: MixtralForCausalLM,
    ) -> None:
        """D5: with only 4 slots (2 layers * up-to-4 routed experts),
        the pool must evict constantly yet produce identical logits."""
        torch.manual_seed(2)
        input_ids = torch.randint(0, config.vocab_size, (2,))
        pool = _pool(bank, n_slots=4)

        kv = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)
        our_logits = prefill_forward_moe(weights, config, pool, kv, input_ids)

        with torch.no_grad():
            hf_logits = hf_model(input_ids.unsqueeze(0)).logits[0, -1]

        _assert_allclose(our_logits, hf_logits, "tight-slot prefill")
        assert pool.misses > 0                       # eviction actually ran
        assert pool.pool.used_bytes <= pool.pool.budget_bytes  # never over


# ---------------------------------------------------------------------------
# Decode equivalence
# ---------------------------------------------------------------------------

class TestMoEDecodeEquivalence:
    def test_decode_multi_step_matches_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: MixtralForCausalLM,
    ) -> None:
        torch.manual_seed(3)
        prompt_len, n_decode = 6, 4
        input_ids = torch.randint(0, config.vocab_size, (prompt_len,))
        pool = _pool(bank, n_slots=16)

        kv = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)
        our_prefill = prefill_forward_moe(weights, config, pool, kv, input_ids)
        our_tokens: list[int] = [int(our_prefill.argmax().item())]
        our_logits: list[torch.Tensor] = [our_prefill]

        for _ in range(n_decode):
            logits = decode_step_moe(weights, config, pool, kv, our_tokens[-1])
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

    def test_decode_tight_slots_still_match_hf(
        self, weights: DenseWeights, config: ModelConfig, bank: ExpertBank,
        hf_model: MixtralForCausalLM,
    ) -> None:
        """D5 under pressure: 2 slots, so every layer's routed experts
        are evicted by the next layer each step — output unchanged."""
        torch.manual_seed(4)
        prompt_len, n_decode = 3, 3
        input_ids = torch.randint(0, config.vocab_size, (prompt_len,))
        pool = _pool(bank, n_slots=2)

        kv = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)
        our_prefill = prefill_forward_moe(weights, config, pool, kv, input_ids)
        our_tokens: list[int] = [int(our_prefill.argmax().item())]
        our_logits: list[torch.Tensor] = [our_prefill]

        for _ in range(n_decode):
            logits = decode_step_moe(weights, config, pool, kv, our_tokens[-1])
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

        # hit-rate observable (M1 gate): the metric is exposed and sane
        assert 0.0 <= pool.hit_rate <= 1.0
        assert pool.pool.used_bytes <= pool.pool.budget_bytes
