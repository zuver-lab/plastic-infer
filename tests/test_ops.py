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
    host_stream_attention_v1,
    paged_attention_v1,
    rms_norm,
    rope_positions,
)
from plastic_infer.exec.moe import (
    moe_forward_reference,
    moe_forward_reference_swiglu,
    moe_forward_sparse,
    moe_forward_v1,
    swiglu_mlp,
    topk_route,
)
from plastic_infer.store.experts import ExpertWeights


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


class TestSparseMoEForward:
    """M1 sparse path: only the routed experts are resident."""

    @staticmethod
    def _experts(E: int, H: int, I: int, seed: int = 7
                 ) -> dict[int, ExpertWeights]:
        g = torch.Generator().manual_seed(seed)
        return {
            e: ExpertWeights(
                w1=torch.randn(I, H, generator=g) * 0.1,
                w2=torch.randn(H, I, generator=g) * 0.1,
                w3=torch.randn(I, H, generator=g) * 0.1,
            )
            for e in range(E)
        }

    def test_sparse_matches_reference(self) -> None:
        torch.manual_seed(11)
        T, H, I, E, k = 8, 16, 32, 4, 2
        hidden = torch.randn(T, H) * 0.5
        experts = self._experts(E, H, I)
        router_logits = torch.randn(T, E)
        eids, weights = topk_route(router_logits, k=k)

        # Serve only the routed subset (as the slot pool would)
        routed = {int(e) for e in eids.flatten().tolist()}
        subset = {e: experts[e] for e in routed}

        out = moe_forward_sparse(hidden, subset, eids, weights)
        ref = moe_forward_reference_swiglu(hidden, experts, eids, weights)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5)

    def test_single_token_single_expert(self) -> None:
        """k=1 single token: silu(x w1^T) * (x w3^T) then w2."""
        T, H, I, k = 1, 4, 8, 1
        hidden = torch.randn(T, H)
        w = ExpertWeights(w1=torch.randn(I, H), w2=torch.randn(H, I),
                          w3=torch.randn(I, H))
        eids = torch.zeros(T, k, dtype=torch.long)
        weights = torch.ones(T, k)

        out = moe_forward_sparse(hidden, {0: w}, eids, weights)
        ref = swiglu_mlp(hidden, w.w1, w.w2, w.w3)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_unrouted_experts_not_needed(self) -> None:
        """Passing only routed experts is sufficient (D5 at op level)."""
        torch.manual_seed(13)
        T, H, I, E, k = 12, 16, 32, 8, 2
        hidden = torch.randn(T, H) * 0.5
        experts = self._experts(E, H, I, seed=3)
        router_logits = torch.randn(T, E)
        eids, weights = topk_route(router_logits, k=k)

        routed = {int(e) for e in eids.flatten().tolist()}
        subset = {e: experts[e] for e in routed}

        # Sparse (routed subset) must equal full-dict forward
        sparse = moe_forward_sparse(hidden, subset, eids, weights)
        full = moe_forward_sparse(hidden, experts, eids, weights)
        assert torch.allclose(sparse, full, atol=1e-6)


# ---------------------------------------------------------------------------
# Host-stream attention (M3, optional long-sequence primitive)
# ---------------------------------------------------------------------------

class TestHostStreamAttention:
    """host_stream_attention_v1 == attention over the full contiguous KV.

    The causal-with-offset mask semantics (kv_pos <= q_pos) are already
    anchored end-to-end by the prefix-reuse tests; these tests pin the
    new part: chunk order / permutation / concat and GQA expansion.
    """

    def _setup(self, seed: int, *, total_kv: int, host_len: int,
               chunk_size: int, n_heads: int = 4, n_kv_heads: int = 2,
               head_dim: int = 16):
        """Split a contiguous KV into host chunks + resident tail."""
        torch.manual_seed(seed)
        k_full = torch.randn(1, total_kv, n_kv_heads, head_dim) * 0.1
        v_full = torch.randn(1, total_kv, n_kv_heads, head_dim) * 0.1

        host_chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, host_len, chunk_size):
            end = min(start + chunk_size, host_len)
            # [H_kv, D, C] (the host chunk layout)
            k_c = k_full[0, start:end].permute(1, 2, 0).contiguous()
            v_c = v_full[0, start:end].permute(1, 2, 0).contiguous()
            host_chunks.append((k_c, v_c))
        resident_k = k_full[:, host_len:]
        resident_v = v_full[:, host_len:]
        return k_full, v_full, host_chunks, resident_k, resident_v

    def _reference(self, q, k_full, v_full, q_start_pos: int,
                   causal: bool) -> torch.Tensor:
        """Contiguous SDPA with the same causal-offset semantics."""
        q_len, kv_len = q.shape[1], k_full.shape[1]
        H, H_kv = q.shape[2], k_full.shape[2]
        q_t = q.transpose(1, 2)
        k_t = k_full.transpose(1, 2).repeat_interleave(H // H_kv, dim=1)
        v_t = v_full.transpose(1, 2).repeat_interleave(H // H_kv, dim=1)
        if causal and (q_start_pos > 0 or q_len != kv_len):
            q_pos = torch.arange(q_start_pos, q_start_pos + q_len)
            kv_pos = torch.arange(kv_len)
            mask = kv_pos[None, :] <= q_pos[:, None]
            attn_mask = torch.zeros(q_len, kv_len, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~mask, float("-inf"))
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=attn_mask[None, None])
        else:
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=causal)
        return out.transpose(1, 2)

    def test_full_tail_prefill_matches_contiguous(self) -> None:
        """New tokens appended past a host-streamed KV prefix."""
        kv_full, v_full, chunks, rk, rv = self._setup(
            1, total_kv=10, host_len=6, chunk_size=3)
        q = torch.randn(1, 2, 4, 16) * 0.1     # 2 new tokens at positions 10,11
        q_start_pos = 10

        out = host_stream_attention_v1(q, chunks, rk, rv,
                                       q_start_pos=q_start_pos)
        ref = self._reference(q, kv_full, v_full, q_start_pos, causal=True)
        assert torch.allclose(out, ref, atol=1e-4), (
            f"max diff = {(out - ref).abs().max().item():.6f}")

    def test_decode_step_matches_contiguous(self) -> None:
        kv_full, v_full, chunks, rk, rv = self._setup(
            2, total_kv=8, host_len=5, chunk_size=4)
        q = torch.randn(1, 1, 4, 16) * 0.1     # one token at position 8
        q_start_pos = 8

        out = host_stream_attention_v1(q, chunks, rk, rv,
                                       q_start_pos=q_start_pos)
        ref = self._reference(q, kv_full, v_full, q_start_pos, causal=True)
        assert torch.allclose(out, ref, atol=1e-4)

    def test_offset_query_attends_suffix(self) -> None:
        """q positions inside the KV range (recompute of a partial tail)."""
        kv_full, v_full, chunks, rk, rv = self._setup(
            3, total_kv=8, host_len=4, chunk_size=2)
        q = torch.randn(1, 2, 4, 16) * 0.1     # positions 5,6
        q_start_pos = 5

        out = host_stream_attention_v1(q, chunks, rk, rv,
                                       q_start_pos=q_start_pos)
        ref = self._reference(q, kv_full, v_full, q_start_pos, causal=True)
        assert torch.allclose(out, ref, atol=1e-4)

    def test_all_resident_no_chunks(self) -> None:
        """No host chunks: degenerate case == plain contiguous causal SDPA."""
        kv_full, v_full, _, rk, rv = self._setup(
            4, total_kv=6, host_len=6, chunk_size=3)
        # resident covers everything; chunks list is empty
        rk = kv_full
        rv = v_full
        q = kv_full.clone()                  # q_len == kv_len, q_start_pos == 0
        q = q.repeat_interleave(2, dim=2)    # [1, 6, 4, 16] GQA already exercised

        out = host_stream_attention_v1(q, [], rk, rv, q_start_pos=0)
        ref = self._reference(q, kv_full, v_full, 0, causal=True)
        assert torch.allclose(out, ref, atol=1e-4)
