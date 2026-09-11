"""Dense model runner with paged KV cache (M2).

Same forward logic as runner.py, but KV is paged and managed
through KVStore. This enables on-demand page allocation, chunk
eviction, and prefix loading from host L1 cache.

Numerical output should match runner.py exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import build_rope_cache, paged_attention_v1, rms_norm, rope_positions
from ..kv.kv_store import KVConfig, KVRequestCache, KVStore
from .runner import DenseWeights, ModelConfig, linear


def make_kv_config(model_cfg: ModelConfig, max_pages: int = 256,
                   max_host_chunks: int = 64,
                   device: torch.device | None = None,
                   dtype: torch.dtype | None = None) -> KVConfig:
    """Build a KVConfig from a ModelConfig."""
    return KVConfig(
        n_layers=model_cfg.n_layers,
        n_kv_heads=model_cfg.n_kv_heads,
        head_dim=model_cfg.head_dim,
        max_pages=max_pages,
        max_host_chunks=max_host_chunks,
        dtype=dtype or model_cfg.dtype,
        device=device or torch.device("cpu"),
    )


def _attn_paged(
    q: torch.Tensor,
    cache: KVRequestCache,
    layer_idx: int,
    store: KVStore,
    kv_num_tokens: int | None = None,
    causal: bool = True,
    q_start_pos: int = 0,
) -> torch.Tensor:
    """Paged attention.

    When q_start_pos > 0 (incremental prefill / decode), Q's first
    row corresponds to logical position q_start_pos in the KV cache.
    Causal mask is built accordingly: Q[i] attends to KV[0 : q_start_pos + i + 1].
    """
    cfg = store.config
    block_table = cache.get_block_table_for_layer(layer_idx)

    if not causal or q_start_pos == 0:
        return paged_attention_v1(
            q, store.gpu.k_pages, store.gpu.v_pages,
            block_table, cfg.page_size,
            kv_num_tokens=kv_num_tokens, causal=causal,
        )

    # Build custom causal mask for offset Q positions.
    # SDPA with attn_mask expects [1, 1, q_len, kv_len] or broadcastable.
    import torch.nn.functional as F
    from .attention import gather_paged_kv

    q_len = q.shape[1]
    kv_len = kv_num_tokens or (len(block_table) * cfg.page_size)
    device = q.device

    # Mask: True/attend if kv_pos <= q_pos (causal)
    q_pos = torch.arange(q_start_pos, q_start_pos + q_len, device=device)  # [Q]
    kv_pos = torch.arange(kv_len, device=device)                       # [KV]
    mask = kv_pos[None, :] <= q_pos[:, None]   # [Q, KV], bool
    # Convert to float mask (0 = attend, -inf = mask)
    attn_mask = torch.zeros(q_len, kv_len, dtype=q.dtype, device=device)
    attn_mask = attn_mask.masked_fill(~mask, float("-inf"))
    attn_mask = attn_mask[None, None, :, :]   # [1, 1, Q, KV]

    # Gather KV then do SDPA with custom mask
    k, v = gather_paged_kv(
        store.gpu.k_pages, store.gpu.v_pages,
        block_table, cfg.page_size, kv_num_tokens,
    )
    q_t = q.transpose(1, 2)       # [1, H, Q, D]
    k_t = k.transpose(1, 2)       # [1, H_kv, KV, D]
    v_t = v.transpose(1, 2)
    n_heads = q_t.shape[1]
    n_kv_heads = k_t.shape[1]
    if n_heads != n_kv_heads:
        n_groups = n_heads // n_kv_heads
        k_t = k_t.repeat_interleave(n_groups, dim=1)
        v_t = v_t.repeat_interleave(n_groups, dim=1)

    out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask)
    return out.transpose(1, 2)   # [1, Q, H, D]


def _allocate_pages_for_tokens(
    cache: KVRequestCache,
    store: KVStore,
    n_tokens: int,
) -> None:
    """Pre-allocate pages for `n_tokens` new tokens (append zeros)."""
    cfg = store.config
    dummy_k = torch.zeros(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
                         dtype=cfg.dtype, device=cfg.device)
    dummy_v = torch.zeros_like(dummy_k)
    for _ in range(n_tokens):
        cache.append_token(dummy_k, dummy_v)


def _write_kv_layer(
    cache: KVRequestCache,
    store: KVStore,
    layer_idx: int,
    start_pos: int,
    k: torch.Tensor,   # [1, seq_len, n_kv_heads, head_dim]
    v: torch.Tensor,
) -> None:
    """Write K/V for one layer into the pre-allocated pages."""
    cfg = store.config
    seq_len = k.shape[1]
    for t in range(seq_len):
        global_t = start_pos + t
        page_idx = global_t // cfg.page_size
        page_off = global_t % cfg.page_size
        phys_page = cache.block_table.get_page(layer_idx, page_idx)
        store.gpu.k_pages[phys_page, :, :, page_off] = k[0, t]
        store.gpu.v_pages[phys_page, :, :, page_off] = v[0, t]


def prefill_forward_paged(
    weights: DenseWeights,
    config: ModelConfig,
    store: KVStore,
    cache: KVRequestCache,
    input_ids: torch.Tensor,   # [seq_len]
    start_pos: int = 0,
) -> torch.Tensor:
    """Full prefill pass with paged KV.

    If start_pos > 0, the first `start_pos` tokens are already in
    the cache (from prefix loading). We only compute attention for
    new tokens, but KV is written for the full new range.

    Returns logits for the last token [vocab_size].
    """
    assert input_ids.dim() == 1
    seq_len = input_ids.shape[0]
    device = weights.device
    dtype = weights.dtype
    cfg = config

    # Pre-allocate pages for the new tokens
    _allocate_pages_for_tokens(cache, store, seq_len)

    # Embedding
    x = weights["embed_tokens.weight"][input_ids].to(dtype).unsqueeze(0)

    # RoPE
    cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim,
                                 base=cfg.rope_base, device=device, dtype=dtype)
    positions = torch.arange(start_pos, start_pos + seq_len, device=device)

    for layer_idx in range(cfg.n_layers):
        residual = x
        x = rms_norm(x, weights[f"layers.{layer_idx}.input_layernorm.weight"])

        # Projections
        q = linear(x, weights[f"layers.{layer_idx}.self_attn.q_proj.weight"])
        k = linear(x, weights[f"layers.{layer_idx}.self_attn.k_proj.weight"])
        v = linear(x, weights[f"layers.{layer_idx}.self_attn.v_proj.weight"])

        q = q.view(1, seq_len, cfg.n_heads, cfg.head_dim)
        k = k.view(1, seq_len, cfg.n_kv_heads, cfg.head_dim)
        v = v.view(1, seq_len, cfg.n_kv_heads, cfg.head_dim)

        q = rope_positions(cos, sin, positions, q)
        k = rope_positions(cos, sin, positions, k)

        # Write K/V into pre-allocated pages
        _write_kv_layer(cache, store, layer_idx, start_pos, k, v)

        # Paged attention
        attn_out = _attn_paged(q, cache, layer_idx, store,
                               kv_num_tokens=start_pos + seq_len,
                               causal=True, q_start_pos=start_pos)
        attn_out = attn_out.contiguous().view(1, seq_len, cfg.n_heads * cfg.head_dim)
        attn_out = linear(attn_out,
                          weights[f"layers.{layer_idx}.self_attn.o_proj.weight"])
        x = residual + attn_out

        # FFN (SwiGLU)
        residual = x
        x = rms_norm(x, weights[f"layers.{layer_idx}.post_attention_layernorm.weight"])
        gate = linear(x, weights[f"layers.{layer_idx}.mlp.gate_proj.weight"])
        up = linear(x, weights[f"layers.{layer_idx}.mlp.up_proj.weight"])
        ffn_out = F.silu(gate) * up
        ffn_out = linear(ffn_out,
                         weights[f"layers.{layer_idx}.mlp.down_proj.weight"])
        x = residual + ffn_out

    # Final norm + lm_head
    x = rms_norm(x, weights["norm.weight"])
    logits = linear(x[:, -1, :], weights["lm_head.weight"])
    return logits.squeeze(0)


def decode_step_paged(
    weights: DenseWeights,
    config: ModelConfig,
    store: KVStore,
    cache: KVRequestCache,
    input_id: int,
) -> torch.Tensor:
    """Single decode step with paged KV. Returns logits [vocab_size]."""
    device = weights.device
    dtype = weights.dtype
    cfg = config
    pos = cache.length

    # Pre-allocate this token's page
    dummy_k = torch.zeros(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
                         dtype=dtype, device=device)
    dummy_v = torch.zeros_like(dummy_k)
    cache.append_token(dummy_k, dummy_v)

    # Embedding
    x = weights["embed_tokens.weight"][input_id].to(dtype).view(1, 1, -1)

    cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim,
                                 base=cfg.rope_base, device=device, dtype=dtype)
    pos_tensor = torch.tensor([pos], device=device)

    for layer_idx in range(cfg.n_layers):
        residual = x
        x = rms_norm(x, weights[f"layers.{layer_idx}.input_layernorm.weight"])

        q = linear(x, weights[f"layers.{layer_idx}.self_attn.q_proj.weight"])
        k = linear(x, weights[f"layers.{layer_idx}.self_attn.k_proj.weight"])
        v = linear(x, weights[f"layers.{layer_idx}.self_attn.v_proj.weight"])

        q = q.view(1, 1, cfg.n_heads, cfg.head_dim)
        k = k.view(1, 1, cfg.n_kv_heads, cfg.head_dim)
        v = v.view(1, 1, cfg.n_kv_heads, cfg.head_dim)

        q = rope_positions(cos, sin, pos_tensor, q)
        k = rope_positions(cos, sin, pos_tensor, k)

        # Write into the pre-allocated page
        _write_kv_layer(cache, store, layer_idx, pos, k, v)

        # Paged attention (no causal mask needed for Q len=1)
        attn_out = _attn_paged(q, cache, layer_idx, store,
                               kv_num_tokens=pos + 1, causal=False)
        attn_out = attn_out.contiguous().view(1, 1, cfg.n_heads * cfg.head_dim)
        attn_out = linear(attn_out,
                          weights[f"layers.{layer_idx}.self_attn.o_proj.weight"])
        x = residual + attn_out

        # FFN
        residual = x
        x = rms_norm(x, weights[f"layers.{layer_idx}.post_attention_layernorm.weight"])
        gate = linear(x, weights[f"layers.{layer_idx}.mlp.gate_proj.weight"])
        up = linear(x, weights[f"layers.{layer_idx}.mlp.up_proj.weight"])
        ffn_out = F.silu(gate) * up
        ffn_out = linear(ffn_out,
                         weights[f"layers.{layer_idx}.mlp.down_proj.weight"])
        x = residual + ffn_out

    # Final norm + lm_head
    x = rms_norm(x, weights["norm.weight"])
    logits = linear(x[:, -1, :], weights["lm_head.weight"])
    return logits.squeeze(0)
