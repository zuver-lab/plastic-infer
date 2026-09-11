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


def topk_route(
    router_logits: torch.Tensor,   # [num_tokens, num_experts]
    k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k routing.

    Returns:
        (expert_ids, weights)
        expert_ids: [num_tokens, k]   int64
        weights:     [num_tokens, k]   softmax of top-k logits
    """
    top_logits, top_ids = torch.topk(router_logits, k, dim=-1)
    weights = F.softmax(top_logits, dim=-1)
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
