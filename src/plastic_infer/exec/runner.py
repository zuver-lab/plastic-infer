"""Layer-by-layer forward pass runner.

Strings together weights, KV cache, and ops into a full prefill/decode
loop. This is the heart of the engine — everything else exists to make
this loop correct and fast.

v1 (M0 milestone):
  - Dense models only (no MoE).
  - All weights already in host memory (HOST source).
  - KV cache in a flat contiguous buffer (simpler to verify against HF).
    Paged KV comes later; the interface is stable so we can swap in
    the paged implementation without changing this file's structure.
  - No streaming, no async, no overlap — just a plain sequential loop.
    Correctness first; performance comes after the equivalence anchor.

The runner takes:
  - weights: dict of layer_name -> torch.Tensor (dense weights)
  - kv_cache: KVCache object (contiguous or paged)
  - model config (n_layers, n_heads, etc.)

And produces logits one token at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .attention import build_rope_cache, rms_norm, rope_positions


@dataclass
class ModelConfig:
    """Minimal model description for the runner."""
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    intermediate_dim: int
    vocab_size: int
    max_seq_len: int
    rope_base: float = 10000.0
    dtype: torch.dtype = torch.float32
    qk_norm: bool = False            # Qwen3-style per-head QK RMSNorm
    # MoE fields (0 = pure dense model).
    n_experts: int = 0               # experts per layer (Mixtral-style)
    n_experts_per_tok: int = 0       # top-k routing width


class DenseKVCache:
    """Simple contiguous KV cache (v1, for correctness verification).

    Stores K and V as [n_layers, max_seq_len, n_kv_heads, head_dim].
    Allocated up-front; append only.
    """

    def __init__(self, config: ModelConfig, max_seq_len: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        dtype = dtype or config.dtype
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")
        self.k = torch.zeros(config.n_layers, max_seq_len, config.n_kv_heads,
                             config.head_dim, dtype=dtype, device=self.device)
        self.v = torch.zeros(config.n_layers, max_seq_len, config.n_kv_heads,
                             config.head_dim, dtype=dtype, device=self.device)
        self._len = 0

    @property
    def length(self) -> int:
        return self._len

    def append(self, layer_idx: int, k_token: torch.Tensor,
               v_token: torch.Tensor) -> None:
        """Append one token's K/V for a single layer.

        k_token, v_token: [1, n_kv_heads, head_dim] (unsqueezed seq dim)
        """
        assert self._len < self.max_seq_len
        pos = self._len
        self.k[layer_idx, pos] = k_token.squeeze(0)
        self.v[layer_idx, pos] = v_token.squeeze(0)

    def end_token(self) -> None:
        """Call after all layers have appended for this token step."""
        self._len += 1

    def get_slice(self, layer_idx: int, end: int | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get K/V for layer_idx up to position end (exclusive)."""
        if end is None:
            end = self._len
        # Add batch dim for SDPA: [1, seq, n_kv_heads, head_dim]
        k = self.k[layer_idx, :end].unsqueeze(0)
        v = self.v[layer_idx, :end].unsqueeze(0)
        return k, v


# ---------------------------------------------------------------------------
# Weight access
# ---------------------------------------------------------------------------


class DenseWeights:
    """Dense model weights stored as a plain dict (all in memory for v1).

    Keys follow the pattern:
      "layers.{i}.input_layernorm.weight"
      "layers.{i}.self_attn.q_proj.weight"
      "layers.{i}.self_attn.k_proj.weight"
      "layers.{i}.self_attn.v_proj.weight"
      "layers.{i}.self_attn.o_proj.weight"
      "layers.{i}.post_attention_layernorm.weight"
      "layers.{i}.mlp.gate_proj.weight"
      "layers.{i}.mlp.up_proj.weight"
      "layers.{i}.mlp.down_proj.weight"
      "embed_tokens.weight"
      "norm.weight"
      "lm_head.weight"
    """

    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        self.w = weights
        self.dtype = next(iter(weights.values())).dtype
        self.device = next(iter(weights.values())).device

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.w[key]


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Linear layer: x @ weight.T (standard nn.Linear weight layout)."""
    return F.linear(x, weight)


def prefill_forward(
    weights: DenseWeights,
    config: ModelConfig,
    kv_cache: DenseKVCache,
    input_ids: torch.Tensor,   # [seq_len]
) -> torch.Tensor:
    """Full prefill pass. Returns logits for the last token [vocab_size].

    Uses standard dense Llama-style architecture:
      embed → for each layer: RMSNorm → attn → add → RMSNorm → FFN → add
    final RMSNorm → lm_head
    """
    assert input_ids.dim() == 1
    seq_len = input_ids.shape[0]
    device = weights.device
    dtype = weights.dtype

    # Embedding
    x = weights["embed_tokens.weight"][input_ids].to(dtype)  # [S, H]
    x = x.unsqueeze(0)  # [1, S, H]

    # RoPE cache
    cos, sin = build_rope_cache(config.max_seq_len, config.head_dim,
                                 base=config.rope_base, device=device,
                                 dtype=dtype)
    positions = torch.arange(seq_len, device=device)

    for layer_idx in range(config.n_layers):
        # Pre-norm
        residual = x
        x = rms_norm(x, weights[f"layers.{layer_idx}.input_layernorm.weight"])

        # Attention projections
        q = linear(x, weights[f"layers.{layer_idx}.self_attn.q_proj.weight"])
        k = linear(x, weights[f"layers.{layer_idx}.self_attn.k_proj.weight"])
        v = linear(x, weights[f"layers.{layer_idx}.self_attn.v_proj.weight"])

        # Reshape to [1, S, H, D]
        q = q.view(1, seq_len, config.n_heads, config.head_dim)
        k = k.view(1, seq_len, config.n_kv_heads, config.head_dim)
        v = v.view(1, seq_len, config.n_kv_heads, config.head_dim)

        # RoPE
        q = rope_positions(cos, sin, positions, q)
        k = rope_positions(cos, sin, positions, k)

        # Save KV to cache
        kv_cache.k[layer_idx, :seq_len] = k[0]
        kv_cache.v[layer_idx, :seq_len] = v[0]

        # SDPA (causal)
        q_t = q.transpose(1, 2)    # [1, H, S, D]
        k_t = k.transpose(1, 2)    # [1, H_kv, S, D]
        v_t = v.transpose(1, 2)
        if config.n_heads != config.n_kv_heads:
            n_groups = config.n_heads // config.n_kv_heads
            k_t = k_t.repeat_interleave(n_groups, dim=1)
            v_t = v_t.repeat_interleave(n_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous()  # [1, S, H, D]
        attn_out = attn_out.view(1, seq_len, config.n_heads * config.head_dim)

        # Output projection
        attn_out = linear(attn_out,
                          weights[f"layers.{layer_idx}.self_attn.o_proj.weight"])
        x = residual + attn_out

        # FFN (SwiGLU: gate_proj + up_proj → silu(gate) * up → down_proj)
        residual = x
        x = rms_norm(x, weights[f"layers.{layer_idx}.post_attention_layernorm.weight"])
        gate = linear(x, weights[f"layers.{layer_idx}.mlp.gate_proj.weight"])
        up = linear(x, weights[f"layers.{layer_idx}.mlp.up_proj.weight"])
        ffn_out = F.silu(gate) * up
        ffn_out = linear(ffn_out,
                         weights[f"layers.{layer_idx}.mlp.down_proj.weight"])
        x = residual + ffn_out

    # Update cache length
    kv_cache._len = seq_len

    # Final norm + lm_head (last token only)
    x = rms_norm(x, weights["norm.weight"])
    logits = linear(x[:, -1, :], weights["lm_head.weight"])  # [1, V]
    return logits.squeeze(0)  # [V]


def decode_step(
    weights: DenseWeights,
    config: ModelConfig,
    kv_cache: DenseKVCache,
    input_id: int,
) -> torch.Tensor:
    """Single decode step. Returns logits [vocab_size]."""
    device = weights.device
    dtype = weights.dtype
    pos = kv_cache.length

    # Embedding
    x = weights["embed_tokens.weight"][input_id].to(dtype)
    x = x.view(1, 1, -1)  # [1, 1, H]

    # RoPE
    cos, sin = build_rope_cache(config.max_seq_len, config.head_dim,
                                 base=config.rope_base, device=device,
                                 dtype=dtype)
    pos_tensor = torch.tensor([pos], device=device)

    for layer_idx in range(config.n_layers):
        # Pre-norm
        residual = x
        x = rms_norm(x, weights[f"layers.{layer_idx}.input_layernorm.weight"])

        # Attention projections
        q = linear(x, weights[f"layers.{layer_idx}.self_attn.q_proj.weight"])
        k = linear(x, weights[f"layers.{layer_idx}.self_attn.k_proj.weight"])
        v = linear(x, weights[f"layers.{layer_idx}.self_attn.v_proj.weight"])

        q = q.view(1, 1, config.n_heads, config.head_dim)
        k = k.view(1, 1, config.n_kv_heads, config.head_dim)
        v = v.view(1, 1, config.n_kv_heads, config.head_dim)

        # RoPE on this token
        q = rope_positions(cos, sin, pos_tensor, q)
        k = rope_positions(cos, sin, pos_tensor, k)

        # Append to cache (writes position `pos`, length still = pos)
        kv_cache.append(layer_idx, k, v)

        # Full KV so far (including the token we just wrote)
        k_full, v_full = kv_cache.get_slice(layer_idx, end=pos + 1)
        # [1, pos+1, H_kv, D]

        # SDPA
        q_t = q.transpose(1, 2)   # [1, H, 1, D]
        k_t = k_full.transpose(1, 2)  # [1, H_kv, pos+1, D]
        v_t = v_full.transpose(1, 2)
        if config.n_heads != config.n_kv_heads:
            n_groups = config.n_heads // config.n_kv_heads
            k_t = k_t.repeat_interleave(n_groups, dim=1)
            v_t = v_t.repeat_interleave(n_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=False,  # not needed: Q len=1, all K valid
        )
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(1, 1, config.n_heads * config.head_dim)

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

    kv_cache.end_token()

    # Final norm + lm_head
    x = rms_norm(x, weights["norm.weight"])
    logits = linear(x[:, -1, :], weights["lm_head.weight"])
    return logits.squeeze(0)
