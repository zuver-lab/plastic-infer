"""MoE model runner with paged KV (M1 + M2 composition).

Composes the two prior milestones:
  - M1: sparse routed-expert FFN served by an ExpertSlotPool
  - M2: paged KV through KVStore (page pool, chunk sinking, prefix reuse)

Attention/KV handling is identical to runner_paged.py (reuses its
paged-attention and page-write helpers); the FFN is the MoE block from
runner_moe.py. Numerically equal to runner_moe.py — placement of
experts and KV never changes logits (D5), prefix reuse only skips
computation of the matched tail (D10).
"""

from __future__ import annotations

import torch

from ..kv.kv_store import KVRequestCache, KVStore
from ..store.experts import ExpertSlotPool
from .attention import build_rope_cache, qk_norm, rms_norm, rope_positions
from .moe import moe_forward_sparse, topk_route
from .runner import DenseWeights, ModelConfig, linear
from .runner_moe import _moe_ffn
from .runner_paged import (
    _allocate_pages_for_tokens,
    _attn_paged,
    _write_kv_layer,
    make_kv_config,
)


def prefill_forward_moe_paged(
    weights: DenseWeights,
    config: ModelConfig,
    pool: ExpertSlotPool,
    store: KVStore,
    cache: KVRequestCache,
    input_ids: torch.Tensor,   # [seq_len]
    start_pos: int = 0,
) -> torch.Tensor:
    """Full MoE prefill with paged KV.

    If start_pos > 0, the first `start_pos` tokens are already in the
    cache (prefix loading); only the tail is computed (D10).

    Returns logits for the last token [vocab_size].
    """
    assert input_ids.dim() == 1
    seq_len = input_ids.shape[0]
    device = weights.device
    dtype = weights.dtype

    _allocate_pages_for_tokens(cache, store, seq_len)

    x = weights["embed_tokens.weight"][input_ids].to(dtype).unsqueeze(0)
    cos, sin = build_rope_cache(config.max_seq_len, config.head_dim,
                                base=config.rope_base, device=device,
                                dtype=dtype)
    positions = torch.arange(start_pos, start_pos + seq_len, device=device)

    for layer_idx in range(config.n_layers):
        residual = x
        x = rms_norm(x, weights[f"layers.{layer_idx}.input_layernorm.weight"])

        q = linear(x, weights[f"layers.{layer_idx}.self_attn.q_proj.weight"])
        k = linear(x, weights[f"layers.{layer_idx}.self_attn.k_proj.weight"])
        v = linear(x, weights[f"layers.{layer_idx}.self_attn.v_proj.weight"])
        q = q.view(1, seq_len, config.n_heads, config.head_dim)
        k = k.view(1, seq_len, config.n_kv_heads, config.head_dim)
        v = v.view(1, seq_len, config.n_kv_heads, config.head_dim)
        if config.qk_norm:
            q, k = qk_norm(
                q, k,
                weights[f"layers.{layer_idx}.self_attn.q_norm.weight"],
                weights[f"layers.{layer_idx}.self_attn.k_norm.weight"],
                config.head_dim)
        q = rope_positions(cos, sin, positions, q)
        k = rope_positions(cos, sin, positions, k)

        _write_kv_layer(cache, store, layer_idx, start_pos, k, v)

        attn_out = _attn_paged(q, cache, layer_idx, store,
                               kv_num_tokens=start_pos + seq_len,
                               causal=True, q_start_pos=start_pos)
        attn_out = attn_out.contiguous().view(
            1, seq_len, config.n_heads * config.head_dim)
        attn_out = linear(attn_out,
                          weights[f"layers.{layer_idx}.self_attn.o_proj.weight"])
        x = residual + attn_out

        residual = x
        x = rms_norm(x,
                     weights[f"layers.{layer_idx}.post_attention_layernorm.weight"])
        x = _moe_ffn(x, weights, pool, layer_idx, config)
        x = residual + x

    x = rms_norm(x, weights["norm.weight"])
    logits = linear(x[:, -1, :], weights["lm_head.weight"])
    return logits.squeeze(0)


def decode_step_moe_paged(
    weights: DenseWeights,
    config: ModelConfig,
    pool: ExpertSlotPool,
    store: KVStore,
    cache: KVRequestCache,
    input_id: int,
) -> torch.Tensor:
    """Single MoE decode step with paged KV. Returns logits [vocab_size]."""
    device = weights.device
    dtype = weights.dtype
    pos = cache.length

    dummy_k = torch.zeros(config.n_layers, config.n_kv_heads, config.head_dim,
                          dtype=dtype, device=device)
    dummy_v = torch.zeros_like(dummy_k)
    cache.append_token(dummy_k, dummy_v)

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
        if config.qk_norm:
            q, k = qk_norm(
                q, k,
                weights[f"layers.{layer_idx}.self_attn.q_norm.weight"],
                weights[f"layers.{layer_idx}.self_attn.k_norm.weight"],
                config.head_dim)
        q = rope_positions(cos, sin, pos_tensor, q)
        k = rope_positions(cos, sin, pos_tensor, k)

        _write_kv_layer(cache, store, layer_idx, pos, k, v)

        attn_out = _attn_paged(q, cache, layer_idx, store,
                               kv_num_tokens=pos + 1, causal=False)
        attn_out = attn_out.contiguous().view(
            1, 1, config.n_heads * config.head_dim)
        attn_out = linear(attn_out,
                          weights[f"layers.{layer_idx}.self_attn.o_proj.weight"])
        x = residual + attn_out

        residual = x
        x = rms_norm(x,
                     weights[f"layers.{layer_idx}.post_attention_layernorm.weight"])
        x = _moe_ffn(x, weights, pool, layer_idx, config)
        x = residual + x

    x = rms_norm(x, weights["norm.weight"])
    logits = linear(x[:, -1, :], weights["lm_head.weight"])
    return logits.squeeze(0)
