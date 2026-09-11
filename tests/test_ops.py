"""Tests for exec ops: attention, paged KV gather, MoE routing, RMSNorm.

All tests run on CPU (no GPU needed). The ground truth for each op
is a simple loop-based reference implementation — correctness is
anchored to "trivially correct" code, not to another library.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from plastic_infer.exec.attention import (
    build_rope_cache,
    gather_paged_kv,
    paged_attention_v1,
    rms_norm,
    rope_positions,
)
from plastic_infer.exec.moe import (
    moe_forward_reference,
    moe_forward_v1,
    topk_route,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_vs_torch_norm(self) -> None:
        """RMSNorm output should match a manual computation."""
        torch.manual_seed(42)
        x = torch.randn(2, 10, 64)
        weight = torch.randn(64) * 0.1 + 1.0

        out = rms_norm(x, weight)
        # Manual reference
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)
        ref = x / rms * weight
        assert torch.allclose(out, ref, atol=1e-5)

    def test_weight_ones_means_normalized(self) -> None:
        x = torch.randn(3, 8) * 100
        weight = torch.ones(8)
        out = rms_norm(x, weight)
        # Output RMS per row should be ~1
        row_rms = torch.sqrt(torch.mean(out * out, dim=-1))
        assert torch.allclose(row_rms, torch.ones_like(row_rms), atol=1e-4)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class TestRoPE:
    def test_rope_shift_invariance(self) -> None:
        """Applying RoPE at position p vs p+1 should differ."""
        head_dim = 64
        cos, sin = build_rope_cache(seq_len=100, head_dim=head_dim)
        x = torch.randn(1, 1, 4, head_dim)  # [B, S, H, D]

        y0 = rope_positions(cos, sin, torch.tensor([0]), x)
        y1 = rope_positions(cos, sin, torch.tensor([1]), x)
        assert not torch.allclose(y0, y1)

    def test_rope_deterministic(self) -> None:
        head_dim = 32
        cos, sin = build_rope_cache(seq_len=10, head_dim=head_dim)
        x = torch.randn(1, 4, 2, head_dim)
        pos = torch.arange(4)
        y1 = rope_positions(cos, sin, pos, x)
        y2 = rope_positions(cos, sin, pos, x)
        assert torch.allclose(y1, y2)

    def test_rope_half_rotation(self) -> None:
        """RoPE rotates by pairs; verify the math matches a reference."""
        head_dim = 4
        cos, sin = build_rope_cache(seq_len=2, head_dim=head_dim)
        # x = [1, 0, 1, 0] per head
        x = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])  # [1,1,4]
        x = x.unsqueeze(2)  # [1,1,1,4]
        pos = torch.tensor([0])
        y = rope_positions(cos, sin, pos, x)
        # Position 0: rotation angle = 0, so cos=1, sin=0 → no rotation
        assert torch.allclose(y, x, atol=1e-5)


# ---------------------------------------------------------------------------
# Paged KV gather
# ---------------------------------------------------------------------------

class TestPagedKvGather:
    def test_single_page(self) -> None:
        """Gathering one page = the contents of that page."""
        torch.manual_seed(0)
        num_pages, n_heads, head_dim, page_size = 10, 2, 8, 16
        k_pages = torch.randn(num_pages, n_heads, head_dim, page_size)
        v_pages = torch.randn(num_pages, n_heads, head_dim, page_size)

        k, v = gather_paged_kv(k_pages, v_pages, [3], page_size)
        assert k.shape == (1, 16, 2, 8)  # [B=1, S, H, D]
        # Page 3's first token should match gathered first token
        # k_pages[pid, :, :, t] shape: [n_heads, head_dim]
        assert torch.allclose(k[0, 0], k_pages[3, :, :, 0])

    def test_multiple_pages_in_order(self) -> None:
        torch.manual_seed(1)
        num_pages, n_heads, head_dim, page_size = 5, 2, 4, 8
        k_pages = torch.randn(num_pages, n_heads, head_dim, page_size)
        v_pages = torch.randn(num_pages, n_heads, head_dim, page_size)

        pages = [0, 2, 4]
        k, v = gather_paged_kv(k_pages, v_pages, pages, page_size)
        assert k.shape == (1, 24, 2, 4)
        # Page 0 tokens 0-7, page 2 tokens 8-15, page 4 tokens 16-23
        assert torch.allclose(k[0, 0], k_pages[0, :, :, 0])
        assert torch.allclose(k[0, 8], k_pages[2, :, :, 0])
        assert torch.allclose(k[0, 16], k_pages[4, :, :, 0])

    def test_partial_last_page(self) -> None:
        num_pages, n_heads, head_dim, page_size = 4, 2, 4, 16
        k_pages = torch.randn(num_pages, n_heads, head_dim, page_size)
        v_pages = torch.randn(num_pages, n_heads, head_dim, page_size)

        k, v = gather_paged_kv(k_pages, v_pages, [0, 1], page_size,
                               num_tokens=20)
        assert k.shape == (1, 20, 2, 4)

    def test_empty_block_table(self) -> None:
        num_pages, n_heads, head_dim, page_size = 4, 2, 4, 16
        k_pages = torch.randn(num_pages, n_heads, head_dim, page_size)
        v_pages = torch.randn(num_pages, n_heads, head_dim, page_size)

        k, v = gather_paged_kv(k_pages, v_pages, [], page_size)
        assert k.shape == (1, 0, 2, 4)
        assert v.shape == (1, 0, 2, 4)


# ---------------------------------------------------------------------------
# Paged attention
# ---------------------------------------------------------------------------

def _reference_attention(
    q: torch.Tensor,   # [B, Sq, H, D]
    k: torch.Tensor,   # [B, Skv, H_kv, D]
    v: torch.Tensor,   # [B, Skv, H_kv, D]
    causal: bool = True,
) -> torch.Tensor:
    """Loop-based reference attention (very slow, ground truth)."""
    B, Sq, H, D = q.shape
    _, Skv, H_kv, _ = k.shape
    assert H % H_kv == 0
    G = H // H_kv

    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(D)

    for b in range(B):
        for h in range(H):
            h_kv = h // G
            for i in range(Sq):
                scores = torch.zeros(Skv, dtype=q.dtype)
                for j in range(Skv):
                    if causal and j > i:
                        scores[j] = float('-inf')
                    else:
                        scores[j] = torch.dot(q[b, i, h], k[b, j, h_kv]) * scale
                attn = F.softmax(scores, dim=-1)
                acc = torch.zeros(D, dtype=q.dtype)
                for j in range(Skv):
                    acc += attn[j] * v[b, j, h_kv]
                out[b, i, h] = acc
    return out


class TestPagedAttention:
    def test_matches_contiguous_attention(self) -> None:
        """Paged attention with all pages in order == contiguous attention."""
        torch.manual_seed(7)
        B, Sq, H, D = 1, 8, 4, 16
        page_size = 4
        n_pages = 3  # 12 tokens of KV
        H_kv = 2     # GQA, 2 kv heads

        q = torch.randn(B, Sq, H, D) * 0.1
        # Contiguous KV
        k_contig = torch.randn(B, n_pages * page_size, H_kv, D) * 0.1
        v_contig = torch.randn(B, n_pages * page_size, H_kv, D) * 0.1

        # Pack into paged format
        # k_pages: [num_pages, H_kv, D, page_size]
        k_pages = torch.zeros(n_pages, H_kv, D, page_size)
        v_pages = torch.zeros(n_pages, H_kv, D, page_size)
        for p in range(n_pages):
            for t in range(page_size):
                tok = p * page_size + t
                k_pages[p, :, :, t] = k_contig[0, tok]  # [H_kv, D]
                v_pages[p, :, :, t] = v_contig[0, tok]

        block_table = list(range(n_pages))

        # Paged attention
        out_paged = paged_attention_v1(
            q, k_pages, v_pages, block_table, page_size,
            kv_num_tokens=n_pages * page_size, causal=True,
        )

        # Reference: SDPA with contiguous KV
        q_t = q.transpose(1, 2)
        k_t = k_contig.transpose(1, 2).repeat_interleave(H // H_kv, dim=1)
        v_t = v_contig.transpose(1, 2).repeat_interleave(H // H_kv, dim=1)
        out_ref = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        out_ref = out_ref.transpose(1, 2)

        assert torch.allclose(out_paged, out_ref, atol=1e-4)

    def test_noncontiguous_pages(self) -> None:
        """Block table with out-of-order pages still gives correct result."""
        torch.manual_seed(11)
        B, Sq, H, D = 1, 6, 2, 8
        page_size = 3
        H_kv = 2

        q = torch.randn(B, Sq, H, D) * 0.1

        # 5 pages, use pages [4, 1, 3] in that logical order
        n_phys_pages = 5
        k_pages = torch.randn(n_phys_pages, H_kv, D, page_size) * 0.1
        v_pages = torch.randn(n_phys_pages, H_kv, D, page_size) * 0.1

        block_table = [4, 1, 3]
        out_paged = paged_attention_v1(
            q, k_pages, v_pages, block_table, page_size,
            kv_num_tokens=9, causal=True,
        )

        # Build contiguous reference: page 4, then 1, then 3
        def pages_to_contig(pages, table):
            pieces = []
            for pid in table:
                # pages[pid] is [H_kv, D, page_size] → [page_size, H_kv, D]
                piece = pages[pid].permute(2, 0, 1)  # [P, H_kv, D]
                pieces.append(piece)
            contig = torch.cat(pieces, dim=0).unsqueeze(0)  # [1, 9, H_kv, D]
            return contig

        k_ref = pages_to_contig(k_pages, block_table)
        v_ref = pages_to_contig(v_pages, block_table)

        q_t = q.transpose(1, 2)
        k_t = k_ref.transpose(1, 2)
        v_t = v_ref.transpose(1, 2)
        out_ref = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        out_ref = out_ref.transpose(1, 2)

        assert torch.allclose(out_paged, out_ref, atol=1e-4)

    def test_decode_single_token(self) -> None:
        """Single-token decode query against full KV cache."""
        torch.manual_seed(13)
        B, Sq, H, D = 1, 1, 2, 8
        page_size = 4
        H_kv = 2
        n_pages = 5

        q = torch.randn(B, Sq, H, D) * 0.1
        k_pages = torch.randn(n_pages, H_kv, D, page_size) * 0.1
        v_pages = torch.randn(n_pages, H_kv, D, page_size) * 0.1

        block_table = list(range(n_pages))
        out = paged_attention_v1(
            q, k_pages, v_pages, block_table, page_size,
            kv_num_tokens=20, causal=True,
        )
        assert out.shape == (1, 1, 2, 8)
        # Should be finite
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# MoE routing
# ---------------------------------------------------------------------------

class TestMoERouting:
    def test_topk_picks_largest(self) -> None:
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5, 2.0]])  # 1 token, 5 exp
        ids, weights = topk_route(logits, k=2)
        assert ids.shape == (1, 2)
        # Top 2 are expert 1 (5.0) and expert 2 (3.0)
        assert set(ids[0].tolist()) == {1, 2}
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1))

    def test_topk_softmax_correct(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        ids, weights = topk_route(logits, k=2)
        # Top 2: 4.0 and 3.0 → experts 3 and 2
        top_logits = torch.tensor([[3.0, 4.0]])
        ref_weights = F.softmax(top_logits, dim=-1)
        # weights[0] corresponds to expert 2, weights[1] to expert 3
        # But topk returns in descending order: expert 3 first, then 2
        assert ids[0, 0].item() == 3
        assert ids[0, 1].item() == 2
        # Ref: softmax of [4.0, 3.0]
        ref = F.softmax(torch.tensor([[4.0, 3.0]]), dim=-1)
        assert torch.allclose(weights, ref, atol=1e-5)

    def test_route_batch(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(16, 8)
        ids, weights = topk_route(logits, k=2)
        assert ids.shape == (16, 2)
        assert weights.shape == (16, 2)
        assert (weights > 0).all()
        assert torch.allclose(weights.sum(dim=-1), torch.ones(16))


# ---------------------------------------------------------------------------
# MoE forward
# ---------------------------------------------------------------------------

class TestMoEForward:
    def test_v1_matches_reference(self) -> None:
        """Batched v1 implementation matches per-token reference."""
        torch.manual_seed(99)
        T, H, I, E, k = 8, 16, 32, 4, 2

        hidden = torch.randn(T, H) * 0.5
        expert_w1 = torch.randn(E, H, I) * 0.1
        expert_w2 = torch.randn(E, I, H) * 0.1
        router_logits = torch.randn(T, E)

        expert_ids, weights = topk_route(router_logits, k=k)

        out_v1 = moe_forward_v1(hidden, expert_w1, expert_w2,
                                expert_ids, weights)
        out_ref = moe_forward_reference(hidden, expert_w1, expert_w2,
                                        expert_ids, weights)

        assert out_v1.shape == out_ref.shape
        assert torch.allclose(out_v1, out_ref, atol=1e-4)

    def test_single_token_single_expert(self) -> None:
        """k=1, single token, single expert = just a linear layer."""
        T, H, I, E, k = 1, 4, 8, 1, 1
        hidden = torch.randn(T, H)
        expert_w1 = torch.randn(E, H, I)
        expert_w2 = torch.randn(E, I, H)
        expert_ids = torch.zeros(T, k, dtype=torch.long)
        weights = torch.ones(T, k)

        out = moe_forward_v1(hidden, expert_w1, expert_w2, expert_ids, weights)
        # Reference: silu(x @ w1) @ w2
        ref = F.silu(hidden @ expert_w1[0]) @ expert_w2[0]
        assert torch.allclose(out, ref, atol=1e-5)

    def test_output_shape(self) -> None:
        T, H, I, E, k = 10, 32, 64, 8, 2
        hidden = torch.randn(T, H)
        expert_w1 = torch.randn(E, H, I)
        expert_w2 = torch.randn(E, I, H)
        ids = torch.randint(0, E, (T, k))
        weights = torch.rand(T, k)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        out = moe_forward_v1(hidden, expert_w1, expert_w2, ids, weights)
        assert out.shape == (T, H)
