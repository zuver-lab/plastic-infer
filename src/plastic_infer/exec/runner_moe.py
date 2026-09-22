"""MoE model runner (M1): dense attention + routed expert FFN.

Same forward logic as runner.py for the dense part; the FFN is a
Mixtral-style MoE block whose experts are served by an
ExpertSlotPool. Experts are loaded on demand by routing and
LRU-evicted from the GPU slot pool — the output is numerically
identical to keeping every expert resident (D5).

KV stays in the flat contiguous cache (DenseKVCache) for this
milestone's equivalence anchor; the paged path composes later.

Weight layout (canonical, dense + MoE):
  layers.{i}.input_layernorm.weight / self_attn.{q,k,v,o}_proj.weight
  layers.{i}.post_attention_layernorm.weight
  layers.{i}.mlp.router.weight
  layers.{i}.mlp.experts.{eid}.{w1,w2,w3}.weight
  embed_tokens.weight / norm.weight / lm_head.weight
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..store.experts import ExpertSlotPool
from .attention import build_rope_cache, rms_norm, rope_positions
from .moe import moe_forward_sparse, topk_route
from .runner import DenseKVCache, DenseWeights, ModelConfig, linear


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
          config: ModelConfig, *, causal: bool) -> torch.Tensor:
    """SDPA over [1, seq, heads, dim] tensors -> [1, seq, hidden]."""
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    if config.n_heads != config.n_kv_heads:
        n_groups = config.n_heads // config.n_kv_heads
        k_t = k_t.repeat_interleave(n_groups, dim=1)
        v_t = v_t.repeat_interleave(n_groups, dim=1)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2).contiguous().view(
        1, q.shape[1], config.n_heads * config.head_dim)


def _moe_ffn(x: torch.Tensor, weights: DenseWeights, pool: ExpertSlotPool,
             layer_idx: int, config: ModelConfig) -> torch.Tensor:
    """Routed-expert FFN. x is [1, T, H]; returns [1, T, H]."""
    T = x.shape[1]
    xf = x.reshape(-1, config.hidden_dim)   # [T, H] (Mixtral flattens b*s)

    router_logits = linear(xf, weights[f"layers.{layer_idx}.mlp.router.weight"])
    eids, routing_w = topk_route(router_logits, config.n_experts_per_tok)

    unique = eids.reshape(-1).unique().tolist()
    exp = pool.ensure(layer_idx, unique)
    try:
        moe_out = moe_forward_sparse(xf, exp, eids, routing_w)
    finally:
        pool.release(layer_idx, unique)

    return moe_out.reshape(1, T, config.hidden_dim)


def prefill_forward_moe(
    weights: DenseWeights,
    config: ModelConfig,
    pool: ExpertSlotPool,
    kv_cache: DenseKVCache,
    input_ids: torch.Tensor,   # [seq_len]
) -> torch.Tensor:
    """Full MoE prefill. Returns logits for the last token [vocab_size]."""
    assert input_ids.dim() == 1
    seq_len = input_ids.shape[0]
    device = weights.device
    dtype = weights.dtype

    x = weights["embed_tokens.weight"][input_ids].to(dtype).unsqueeze(0)
    cos, sin = build_rope_cache(config.max_seq_len, config.head_dim,
                                base=config.rope_base, device=device,
                                dtype=dtype)
    positions = torch.arange(seq_len, device=device)

    for layer_idx in range(config.n_layers):
        residual = x
        x = rms_norm(x, weights[f"layers.{layer_idx}.input_layernorm.weight"])

        q = linear(x, weights[f"layers.{layer_idx}.self_attn.q_proj.weight"])
        k = linear(x, weights[f"layers.{layer_idx}.self_attn.k_proj.weight"])
        v = linear(x, weights[f"layers.{layer_idx}.self_attn.v_proj.weight"])
        q = q.view(1, seq_len, config.n_heads, config.head_dim)
        k = k.view(1, seq_len, config.n_kv_heads, config.head_dim)
        v = v.view(1, seq_len, config.n_kv_heads, config.head_dim)
        q = rope_positions(cos, sin, positions, q)
        k = rope_positions(cos, sin, positions, k)

        kv_cache.k[layer_idx, :seq_len] = k[0]
        kv_cache.v[layer_idx, :seq_len] = v[0]

        attn_out = _sdpa(q, k, v, config, causal=True)
        attn_out = linear(attn_out,
                          weights[f"layers.{layer_idx}.self_attn.o_proj.weight"])
        x = residual + attn_out

        residual = x
        x = rms_norm(x,
                     weights[f"layers.{layer_idx}.post_attention_layernorm.weight"])
        x = _moe_ffn(x, weights, pool, layer_idx, config)
        x = residual + x

    kv_cache._len = seq_len

    x = rms_norm(x, weights["norm.weight"])
    logits = linear(x[:, -1, :], weights["lm_head.weight"])
    return logits.squeeze(0)


def decode_step_moe(
    weights: DenseWeights,
    config: ModelConfig,
    pool: ExpertSlotPool,
    kv_cache: DenseKVCache,
    input_id: int,
) -> torch.Tensor:
    """Single MoE decode step. Returns logits [vocab_size]."""
    device = weights.device
    dtype = weights.dtype
    pos = kv_cache.length

    x = weights["embed_tokens.weight"][input_id].to(dtype).view(1, 1, -1)
    cos, sin = build_rope_cache(config.max_seq_len, config.head_dim,
                                base=config.rope_base, device=device,
                                dtype=dtype)
    pos_tensor = torch.tensor([pos], device=device)

    for layer_idx in range(config.n_layers):
        residual = x
        x = rms_norm(x, weights[f"layers.{layer_idx}.input_layernorm.weight"])

        q = linear(x, weights[f"layers.{layer_idx}.self_attn.q_proj.weight"])
        k = linear(x, weights[f"layers.{layer_idx}.self_attn.k_proj.weight"])
        v = linear(x, weights[f"layers.{layer_idx}.self_attn.v_proj.weight"])
        q = q.view(1, 1, config.n_heads, config.head_dim)
        k = k.view(1, 1, config.n_kv_heads, config.head_dim)
        v = v.view(1, 1, config.n_kv_heads, config.head_dim)
        q = rope_positions(cos, sin, pos_tensor, q)
        k = rope_positions(cos, sin, pos_tensor, k)

        kv_cache.append(layer_idx, k, v)
        k_full, v_full = kv_cache.get_slice(layer_idx, end=pos + 1)

        attn_out = _sdpa(q, k_full, v_full, config, causal=False)
        attn_out = linear(attn_out,
                          weights[f"layers.{layer_idx}.self_attn.o_proj.weight"])
        x = residual + attn_out

        residual = x
        x = rms_norm(x,
                     weights[f"layers.{layer_idx}.post_attention_layernorm.weight"])
        x = _moe_ffn(x, weights, pool, layer_idx, config)
        x = residual + x

    kv_cache.end_token()

    x = rms_norm(x, weights["norm.weight"])
    logits = linear(x[:, -1, :], weights["lm_head.weight"])
    return logits.squeeze(0)
