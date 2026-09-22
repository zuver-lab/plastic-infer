"""Core attention ops: paged KV attention + RoPE.

Design (§5.4 of DESIGN.md):
  - v1 backend: gather resident pages into a contiguous K/V buffer,
    then call torch.nn.functional.scaled_dot_product_attention.
  - Interface is page-aware so we can swap in a real paged kernel
    later without changing callers.
  - CPU fallback works without any GPU (for testing).

This module is pure tensor ops — no knowledge of layers, models, or
weight loading.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def rope_positions(
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to x.

    Args:
        cos, sin: precomputed [seq_len, head_dim]
        positions: [seq_len] int tensor of positions (can be non-contiguous)
        x: [batch, seq_len, n_heads, head_dim]

    Returns:
        [batch, seq_len, n_heads, head_dim] with RoPE applied.
    """
    # Gather cos/sin for each position
    cos_p = cos[positions]   # [seq_len, head_dim]
    sin_p = sin[positions]   # [seq_len, head_dim]
    # Broadcast over batch and n_heads
    cos_p = cos_p[None, :, None, :]   # [1, S, 1, D]
    sin_p = sin_p[None, :, None, :]

    # RoPE: rotate the 2D pairs
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos_p + rotated * sin_p


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.outer(t, inv_freq)
    # For classic RoPE, cos and sin each have head_dim elements (repeated pairs)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS normalization."""
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


# ---------------------------------------------------------------------------
# Paged attention — v1: gather + SDPA
# ---------------------------------------------------------------------------


def gather_paged_kv(
    k_pages: torch.Tensor,     # [num_pages, n_kv_heads, head_dim, page_size]
    v_pages: torch.Tensor,     # [num_pages, n_kv_heads, head_dim, page_size]
    block_table: list[int],    # logical_page_idx -> phys_page_id, length = n_pages_needed
    page_size: int,
    num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather pages from paged KV storage into a contiguous tensor.

    This is the v1 fallback — correct but not maximally efficient.
    A real paged-attention kernel would skip this gather and read
    directly from the page table.

    Args:
        k_pages, v_pages: physical page storage
        block_table: which physical pages to read, in logical order
        page_size: tokens per page
        num_tokens: total tokens to read (may be less than len(block_table)*page_size
                    for the last partial page). If None, all pages are read fully.

    Returns:
        K, V each of shape [1, num_tokens, n_kv_heads, head_dim]
    """
    if not block_table:
        # Empty KV — return empty tensor with correct dims
        n_heads = k_pages.shape[1]
        head_dim = k_pages.shape[2]
        empty = torch.empty(0, n_heads, head_dim, device=k_pages.device, dtype=k_pages.dtype)
        return empty.unsqueeze(0), empty.unsqueeze(0)

    # Gather all pages
    phys_ids = torch.tensor(block_table, dtype=torch.long, device=k_pages.device)
    k_gathered = k_pages.index_select(0, phys_ids)   # [n_pages, H, D, P]
    v_gathered = v_pages.index_select(0, phys_ids)

    # Reshape to [1, n_pages * page_size, H, D]
    n_pages, n_heads, head_dim, _ = k_gathered.shape
    k = k_gathered.permute(0, 3, 1, 2).reshape(n_pages * page_size, n_heads, head_dim)
    v = v_gathered.permute(0, 3, 1, 2).reshape(n_pages * page_size, n_heads, head_dim)

    # Trim to actual token count
    if num_tokens is not None:
        k = k[:num_tokens]
        v = v[:num_tokens]

    return k.unsqueeze(0), v.unsqueeze(0)   # add batch dim


def paged_attention_v1(
    q: torch.Tensor,                  # [1, seq_len, n_heads, head_dim]
    k_pages: torch.Tensor,            # [num_pages, n_kv_heads, head_dim, page_size]
    v_pages: torch.Tensor,            # [num_pages, n_kv_heads, head_dim, page_size]
    block_table: list[int],           # logical_page -> phys_page_id
    page_size: int,
    kv_num_tokens: int | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Paged attention (v1 gather-then-SDPA implementation).

    Args:
        q: query tensor [batch=1, q_len, n_heads, head_dim]
        k_pages, v_pages: physical page storage for keys/values
        block_table: page IDs making up the KV cache for this request
        page_size: tokens per page
        kv_num_tokens: total KV tokens (may be < len(block_table)*page_size)
        causal: whether to apply causal mask

    Returns:
        attention output [1, q_len, n_heads, head_dim]
    """
    k, v = gather_paged_kv(k_pages, v_pages, block_table, page_size, kv_num_tokens)

    # SDPA expects [batch, heads, seq, head_dim]
    q_t = q.transpose(1, 2)        # [1, H, S_q, D]
    k_t = k.transpose(1, 2)        # [1, H_kv, S_kv, D]
    v_t = v.transpose(1, 2)

    # GQA: expand KV heads if needed
    n_heads = q_t.shape[1]
    n_kv_heads = k_t.shape[1]
    if n_heads != n_kv_heads:
        assert n_heads % n_kv_heads == 0
        n_groups = n_heads // n_kv_heads
        k_t = k_t.repeat_interleave(n_groups, dim=1)
        v_t = v_t.repeat_interleave(n_groups, dim=1)

    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)     # back to [1, S_q, H, D]


def host_stream_attention_v1(
    q: torch.Tensor,                    # [1, q_len, n_heads, head_dim]
    host_chunks: list[tuple[torch.Tensor, torch.Tensor]],
    # [(k_c, v_c)] each [n_kv_heads, head_dim, C], oldest chunk first;
    # together they cover positions [0, host_len)
    resident_k: torch.Tensor,           # [1, R, n_kv_heads, head_dim] (GPU tail)
    resident_v: torch.Tensor,           # [1, R, n_kv_heads, head_dim]
    q_start_pos: int = 0,
    causal: bool = True,
) -> torch.Tensor:
    """Attention over KV split between host chunks and a GPU-resident tail.

    This is the long-sequence HOST_STREAM primitive (M3, optional
    switch): when a request's KV outgrows the GPU page pool, completed
    chunks are sunk to host and only a recent tail stays resident. This
    op materializes host chunks + resident tail into one contiguous
    buffer, then SDPA — a correctness-first v1. A flash-incremental
    kernel that reads host chunks without materializing is a later
    swap behind the same interface.

    `q` rows are at global positions [q_start_pos, q_start_pos + q_len);
    the KV covers positions [0, host_len + R). Causal masking matches
    _attn_paged: Q[i] attends to KV[0 : q_start_pos + i + 1].
    """
    # [n_kv_heads, head_dim, C] -> [C, n_kv_heads, head_dim], oldest first
    k_parts = [k_c.permute(2, 0, 1) for k_c, _ in host_chunks]
    v_parts = [v_c.permute(2, 0, 1) for _, v_c in host_chunks]
    # host_chunks is empty when everything is resident
    k_full = torch.cat(k_parts + [resident_k[0]], dim=0).unsqueeze(0)
    v_full = torch.cat(v_parts + [resident_v[0]], dim=0).unsqueeze(0)

    q_len = q.shape[1]
    kv_len = k_full.shape[1]

    q_t = q.transpose(1, 2)            # [1, H, Q, D]
    k_t = k_full.transpose(1, 2)       # [1, H_kv, KV, D]
    v_t = v_full.transpose(1, 2)
    n_heads = q_t.shape[1]
    n_kv_heads = k_t.shape[1]
    if n_heads != n_kv_heads:
        assert n_heads % n_kv_heads == 0
        n_groups = n_heads // n_kv_heads
        k_t = k_t.repeat_interleave(n_groups, dim=1)
        v_t = v_t.repeat_interleave(n_groups, dim=1)

    if causal and (q_start_pos > 0 or q_len != kv_len):
        device = q.device
        q_pos = torch.arange(q_start_pos, q_start_pos + q_len, device=device)
        kv_pos = torch.arange(kv_len, device=device)
        mask = kv_pos[None, :] <= q_pos[:, None]        # [Q, KV]
        attn_mask = torch.zeros(q_len, kv_len, dtype=q.dtype, device=device)
        attn_mask = attn_mask.masked_fill(~mask, float("-inf"))
        attn_mask = attn_mask[None, None, :, :]
        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=attn_mask)
    else:
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)         # [1, Q, H, D]
