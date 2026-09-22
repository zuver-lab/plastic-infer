"""MoE ops: top-k routing + grouped expert GEMM.

Design (§5.4 of DESIGN.md):
  - v1: "verifiability first" implementation.
    Router maps (token, expert) → per-expert mini-batches;
    index_select → batched matmul for each expert.
  - Not optimized (triton grouped GEMM is later). Interface is
    stable so we can swap in faster kernels without touching callers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..store.experts import ExpertWeights


def topk_route(
    router_logits: torch.Tensor,   # [num_tokens, num_experts]
    k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k routing.

    Returns:
        (expert_ids, weights)
        expert_ids: [num_tokens, k]   int64
        weights:     [num_tokens, k]   softmax of top-k logits

    Routing weights are computed in fp32 regardless of logit dtype
    (Qwen3's norm_topk_prob renormalization: softmax of the selected
    top-k is mathematically identical to softmax-all -> topk ->
    renormalize, so no special case is needed — fp32 keeps the expert
    selection stable for bf16 checkpoints).
    """
    top_logits, top_ids = torch.topk(router_logits, k, dim=-1)
    weights = F.softmax(top_logits.float(), dim=-1).to(router_logits.dtype)
    return top_ids, weights


def moe_forward_v1(
    hidden: torch.Tensor,           # [num_tokens, hidden_dim]
    expert_w1: torch.Tensor,        # [num_experts, hidden_dim, inter_dim]
    expert_w2: torch.Tensor,        # [num_experts, inter_dim, hidden_dim]
    expert_ids: torch.Tensor,       # [num_tokens, k]
    weights: torch.Tensor,          # [num_tokens, k]
) -> torch.Tensor:
    """MoE forward pass (v1: simple per-token expert dispatch).

    Correctness-first: for each token, gather its k expert weights,
    compute each expert's output, sum with routing weights.
    O(num_tokens * k * hidden * inter) — slow but easy to verify.

    Args:
        hidden: input tokens [T, H]
        expert_w1: up/gate weights [E, H, I] (v1 has one up-proj)
        expert_w2: down-proj weights [E, I, H]
        expert_ids: top-k expert per token [T, k]
        weights: routing weights [T, k]

    Returns:
        output [T, H]
    """
    T, H = hidden.shape
    k = expert_ids.shape[1]
    output = torch.zeros_like(hidden)

    for ki in range(k):
        e_ids = expert_ids[:, ki]       # [T]
        e_w1 = expert_w1[e_ids]         # [T, H, I]
        e_w2 = expert_w2[e_ids]         # [T, I, H]

        # Per-token matmul: hidden[T,H] x w1[T,H,I] -> [T, I]
        up = torch.bmm(hidden.unsqueeze(1), e_w1).squeeze(1)  # [T, I]
        up = F.silu(up)
        # up [T,I] x w2[T,I,H] -> [T, H]
        down = torch.bmm(up.unsqueeze(1), e_w2).squeeze(1)    # [T, H]

        output += down * weights[:, ki:ki+1]

    return output


# Reference (super slow, for testing only)
def moe_forward_reference(
    hidden: torch.Tensor,
    expert_w1: torch.Tensor,
    expert_w2: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Extremely naive reference implementation — per-token loop.

    Used only as the ground truth for unit tests.
    """
    T, H = hidden.shape
    k = expert_ids.shape[1]
    out = torch.zeros_like(hidden)

    for t in range(T):
        x = hidden[t]  # [H]
        acc = torch.zeros(H, dtype=hidden.dtype, device=hidden.device)
        for ki in range(k):
            eid = int(expert_ids[t, ki].item())
            w1 = expert_w1[eid]   # [H, I]
            w2 = expert_w2[eid]   # [I, H]
            up = x @ w1           # [I]
            up = F.silu(up)
            down = up @ w2        # [H]
            acc += down * weights[t, ki]
        out[t] = acc

    return out


# ---------------------------------------------------------------------------
# Sparse routed experts (M1): only the routed subset is resident
# ---------------------------------------------------------------------------


def swiglu_mlp(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
               w3: torch.Tensor) -> torch.Tensor:
    """One Mixtral-style expert MLP: silu(x @ w1^T) * (x @ w3^T) then w2."""
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)


def moe_forward_sparse(
    hidden: torch.Tensor,               # [num_tokens, hidden_dim]
    experts: dict[int, ExpertWeights],  # routed experts (eid -> weights)
    expert_ids: torch.Tensor,           # [num_tokens, k]
    weights: torch.Tensor,              # [num_tokens, k] routing weights
) -> torch.Tensor:
    """MoE forward with only the routed experts resident (M1).

    For each unique expert, gather the tokens routed to it and run a
    batched SwiGLU GEMM; weighted results are scattered back with
    index_add_. Matches the §5.4 plan: route maps (token, expert) to
    per-expert mini-batches, index_select then batched matmul.

    Correctness-first (v1). The interface stays put for a grouped-GEMM
    kernel later.
    """
    T, H = hidden.shape
    k = expert_ids.shape[1]
    out = torch.zeros_like(hidden)
    ids = expert_ids  # [T, k]

    for eid, w in experts.items():
        pairs = (ids == eid).nonzero()   # [n, 2] (token, slot)
        if pairs.shape[0] == 0:
            continue
        rows = pairs[:, 0]
        wts = weights[rows, pairs[:, 1]].unsqueeze(1)   # [n, 1]

        x = hidden[rows]                                # [n, H]
        inter = F.silu(F.linear(x, w.w1)) * F.linear(x, w.w3)  # [n, I]
        d = F.linear(inter, w.w2)                       # [n, H]
        out.index_add_(0, rows, d * wts)

    return out


def moe_forward_reference_swiglu(
    hidden: torch.Tensor,
    experts: dict[int, ExpertWeights],
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Per-token loop ground truth for the sparse SwiGLU forward.

    Used only as the reference for unit tests.
    """
    T, H = hidden.shape
    k = expert_ids.shape[1]
    out = torch.zeros_like(hidden)

    for t in range(T):
        for s in range(k):
            eid = int(expert_ids[t, s])
            w = experts[eid]
            inter = F.silu(F.linear(hidden[t], w.w1)) * F.linear(hidden[t], w.w3)
            out[t] += F.linear(inter, w.w2) * weights[t, s]

    return out
