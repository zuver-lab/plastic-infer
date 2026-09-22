"""Layer-by-layer streaming dense runner (M3).

The dense counterpart of runner.py that draws each layer's weights
from a DenseWindow instead of a flat in-memory dict. This is what
makes seq_weight_source = DISK / HOST_FIRST actually work (§3.4 of
DESIGN.md): with a window sized W=1, each layer is read from the
source, computed, and evicted before the next layer loads — the
AirLLM-style rotating residency that bounds resident dense weights.

The per-layer body is byte-identical to runner.py's; only the weight
lookup changes. So logits are identical whether a model runs fully in
memory or streams layer-by-layer from disk (D5 across the disk tier).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..store.weights import DenseWindow
from .attention import build_rope_cache, rms_norm, rope_positions
from .runner import DenseKVCache, ModelConfig, linear


def prefill_forward_streamed(
    window: DenseWindow,
    shared: dict[str, torch.Tensor],   # embed / norm / lm_head
    config: ModelConfig,
    kv_cache: DenseKVCache,
    input_ids: torch.Tensor,           # [seq_len]
) -> torch.Tensor:
    """Full prefill streaming layers through the window.

    Returns logits for the last token [vocab_size]. Same architecture
    as runner.prefill_forward; each layer is pinned only for the
    duration of its own forward pass.
    """
    assert input_ids.dim() == 1
    seq_len = input_ids.shape[0]
    device = window.device
    dtype = config.dtype

    x = shared["embed_tokens.weight"][input_ids].to(dtype).unsqueeze(0)

    cos, sin = build_rope_cache(config.max_seq_len, config.head_dim,
                                base=config.rope_base, device=device,
                                dtype=dtype)
    positions = torch.arange(seq_len, device=device)

    for layer_idx in range(config.n_layers):
        w = window.ensure(layer_idx)
        try:
            residual = x
            x = rms_norm(
                x, w[f"layers.{layer_idx}.input_layernorm.weight"])

            q = linear(x, w[f"layers.{layer_idx}.self_attn.q_proj.weight"])
            k = linear(x, w[f"layers.{layer_idx}.self_attn.k_proj.weight"])
            v = linear(x, w[f"layers.{layer_idx}.self_attn.v_proj.weight"])
            q = q.view(1, seq_len, config.n_heads, config.head_dim)
            k = k.view(1, seq_len, config.n_kv_heads, config.head_dim)
            v = v.view(1, seq_len, config.n_kv_heads, config.head_dim)
            q = rope_positions(cos, sin, positions, q)
            k = rope_positions(cos, sin, positions, k)

            kv_cache.k[layer_idx, :seq_len] = k[0]
            kv_cache.v[layer_idx, :seq_len] = v[0]

            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            if config.n_heads != config.n_kv_heads:
                n_groups = config.n_heads // config.n_kv_heads
                k_t = k_t.repeat_interleave(n_groups, dim=1)
                v_t = v_t.repeat_interleave(n_groups, dim=1)

            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True,
            )
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.view(
                1, seq_len, config.n_heads * config.head_dim)
            attn_out = linear(
                attn_out, w[f"layers.{layer_idx}.self_attn.o_proj.weight"])
            x = residual + attn_out

            residual = x
            x = rms_norm(
                x, w[f"layers.{layer_idx}.post_attention_layernorm.weight"])
            gate = linear(x, w[f"layers.{layer_idx}.mlp.gate_proj.weight"])
            up = linear(x, w[f"layers.{layer_idx}.mlp.up_proj.weight"])
            ffn_out = F.silu(gate) * up
            ffn_out = linear(
                ffn_out, w[f"layers.{layer_idx}.mlp.down_proj.weight"])
            x = residual + ffn_out
        finally:
            window.release(layer_idx)

    kv_cache._len = seq_len

    x = rms_norm(x, shared["norm.weight"])
    logits = linear(x[:, -1, :], shared["lm_head.weight"])
    return logits.squeeze(0)


def decode_step_streamed(
    window: DenseWindow,
    shared: dict[str, torch.Tensor],
    config: ModelConfig,
    kv_cache: DenseKVCache,
    input_id: int,
) -> torch.Tensor:
    """Single decode step streaming layers through the window."""
    device = window.device
    dtype = config.dtype
    pos = kv_cache.length

    x = shared["embed_tokens.weight"][input_id].to(dtype).view(1, 1, -1)

    cos, sin = build_rope_cache(config.max_seq_len, config.head_dim,
                                base=config.rope_base, device=device,
                                dtype=dtype)
    pos_tensor = torch.tensor([pos], device=device)

    for layer_idx in range(config.n_layers):
        w = window.ensure(layer_idx)
        try:
            residual = x
            x = rms_norm(
                x, w[f"layers.{layer_idx}.input_layernorm.weight"])

            q = linear(x, w[f"layers.{layer_idx}.self_attn.q_proj.weight"])
            k = linear(x, w[f"layers.{layer_idx}.self_attn.k_proj.weight"])
            v = linear(x, w[f"layers.{layer_idx}.self_attn.v_proj.weight"])
            q = q.view(1, 1, config.n_heads, config.head_dim)
            k = k.view(1, 1, config.n_kv_heads, config.head_dim)
            v = v.view(1, 1, config.n_kv_heads, config.head_dim)
            q = rope_positions(cos, sin, pos_tensor, q)
            k = rope_positions(cos, sin, pos_tensor, k)

            kv_cache.append(layer_idx, k, v)
            k_full, v_full = kv_cache.get_slice(layer_idx, end=pos + 1)

            q_t = q.transpose(1, 2)
            k_t = k_full.transpose(1, 2)
            v_t = v_full.transpose(1, 2)
            if config.n_heads != config.n_kv_heads:
                n_groups = config.n_heads // config.n_kv_heads
                k_t = k_t.repeat_interleave(n_groups, dim=1)
                v_t = v_t.repeat_interleave(n_groups, dim=1)

            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=False,
            )
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.view(
                1, 1, config.n_heads * config.head_dim)
            attn_out = linear(
                attn_out, w[f"layers.{layer_idx}.self_attn.o_proj.weight"])
            x = residual + attn_out

            residual = x
            x = rms_norm(
                x, w[f"layers.{layer_idx}.post_attention_layernorm.weight"])
            gate = linear(x, w[f"layers.{layer_idx}.mlp.gate_proj.weight"])
            up = linear(x, w[f"layers.{layer_idx}.mlp.up_proj.weight"])
            ffn_out = F.silu(gate) * up
            ffn_out = linear(
                ffn_out, w[f"layers.{layer_idx}.mlp.down_proj.weight"])
            x = residual + ffn_out
        finally:
            window.release(layer_idx)

    kv_cache.end_token()

    x = rms_norm(x, shared["norm.weight"])
    logits = linear(x[:, -1, :], shared["lm_head.weight"])
    return logits.squeeze(0)
