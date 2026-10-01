"""MoE model runner (M1): dense attention + routed expert FFN.

Same forward logic as runner.py for the dense part; the FFN is a
Mixtral-style MoE block whose experts are served by an
OffloadMoeCache (the FreeToken port). Movement and kernel dispatch
follow FreeToken's ``OffloadMoELayer`` routed paths verbatim: prefill
streams whole layers (double-buffered under ``prefill_overlap``),
decode loads on demand into the GPU slot cache (or computes on the
CPU executor for CPU-locked layers); the output is numerically
identical to keeping every expert resident.

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

import os

import torch
import torch.nn.functional as F

from ..moe.fused import fused_experts_decode_impl, fused_experts_impl, fused_topk
from .attention import build_rope_cache, qk_norm, rms_norm, rope_positions
from .runner import DenseKVCache, DenseWeights, ModelConfig, linear


# Qwen3 MoE routing hyperparams (the only routed-MoE family PlasticInfer serves;
# FreeToken keeps these on the MoE layer object).
_MOE_RENORMALIZE = True
_MOE_ACTIVATION = "silu"
_MOE_APPLY_ROUTER_WEIGHT_ON_INPUT = False

# Hybrid decode knob (FreeToken FREETOKEN_HYBRID_OVERLAP): 0 serializes the CPU
# overflow before the PCIe fetch + GPU GEMM so an A/B isolates the overlap win.
_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1").strip().lower() \
    not in {"0", "false", "no", "off"}


def _expert_gemm_bf16(hidden_states, topk_weights, topk_ids, *, views, is_prefill):
    """bf16 branch of FreeToken's ``_expert_gemm`` (the only quant format here)."""
    gate_up, down = views
    impl = fused_experts_impl if is_prefill else fused_experts_decode_impl
    return impl(hidden_states, gate_up, down, topk_weights, topk_ids,
                _MOE_ACTIVATION, _MOE_APPLY_ROUTER_WEIGHT_ON_INPUT)


def _wait_prefill_overlap(cache, layer_id: int) -> tuple[torch.Tensor, ...]:
    """Double-buffer choreography for this layer's overlap prefill: kick off the
    next layer's full-layer H2D copy, then return this layer's bank views (buffer
    position == expert id, so routing ids pass through unmapped)."""
    if layer_id == 0:
        cache.begin_prefill()
    cache.prefetch_prefill_layer(layer_id)
    cache.prefetch_prefill_layer(layer_id + 1)
    return cache.wait_prefill_layer(layer_id)


def _prefill_routed(hidden_states, topk_weights, topk_ids, cache, layer_id: int,
                    num_experts: int) -> torch.Tensor:
    """Prefill movement (FreeToken ``_prefill_routed``): stream whole layers --
    double-buffered behind the previous layer's GEMMs when ``prefill_overlap`` is
    on, else a synchronous ``materialize_layer``. In both, position == expert id."""
    if cache.prefill_overlap:
        views = _wait_prefill_overlap(cache, layer_id)
        out = _expert_gemm_bf16(hidden_states, topk_weights, topk_ids,
                                views=views, is_prefill=True)
        cache.release_prefill_layer(layer_id)
        return out
    cache.materialize_layer(layer_id)
    cache.copy_missing()
    return _expert_gemm_bf16(hidden_states, topk_weights, topk_ids,
                             views=cache.bank_views(num_experts), is_prefill=True)


def _decode_hybrid(cache, layer_id: int, hidden_states, topk_weights, topk_ids):
    """Hybrid decode (FreeToken ``_decode_hybrid``): GPU computes cache hits +
    freshly-fetched experts, the CPU executor computes the overflow misses,
    overlapped, then the partials merge. Requires the M2 CPU executor."""
    executor = cache.cpu_executor
    assert executor is not None, "CPU MoE executor was not initialized"
    raw = topk_ids.clone()  # raw expert ids for the CPU partial
    cache.ensure_experts_hybrid(layer_id, topk_ids)  # -> slot (hit/fetched) or -1
    if cache.collect_stats:
        cache.record_decode_stats_hybrid(layer_id)
    on_gpu = topk_ids >= 0
    cpu_ids = torch.where(on_gpu, raw.new_full((), -1), raw).contiguous()
    pending = executor.decode_submit(layer_id, hidden_states, topk_weights, cpu_ids)
    cpu_routed_early = (
        executor.decode_sync(pending) if not _HYBRID_OVERLAP else None
    )
    cache.copy_missing()
    gpu_slots = topk_ids.clamp_min(0)  # -1 -> slot 0 (zero-weighted below)
    gpu_w = torch.where(on_gpu, topk_weights, topk_weights.new_zeros(())).contiguous()
    gpu_routed = _expert_gemm_bf16(hidden_states, gpu_w, gpu_slots,
                                   views=cache.bank_views(), is_prefill=False)
    cpu_routed = cpu_routed_early if not _HYBRID_OVERLAP else executor.decode_sync(pending)
    return gpu_routed + cpu_routed


def _decode_routed(hidden_states, topk_weights, topk_ids, cache,
                   layer_id: int) -> torch.Tensor:
    """On-demand decode (FreeToken ``_decode_routed``): ``ensure_experts`` rewrites
    ``topk_ids`` into cache slot ids in place, then the GEMM reads the full slot
    cache. CPU-locked layers compute on the executor instead (slot cache untouched,
    ids stay raw)."""
    if cache.is_cpu_layer(layer_id):
        executor = cache.cpu_executor
        assert executor is not None, "CPU MoE executor was not initialized"
        return executor.decode(layer_id, hidden_states, topk_weights, topk_ids)
    if cache.decode_target == "hybrid":
        return _decode_hybrid(cache, layer_id, hidden_states, topk_weights, topk_ids)
    cache.ensure_experts(layer_id, topk_ids)
    cache.copy_missing()
    return _expert_gemm_bf16(hidden_states, topk_weights, topk_ids,
                             views=cache.bank_views(), is_prefill=False)


def _moe_ffn(x: torch.Tensor, weights: DenseWeights, moe_cache,
             layer_idx: int, config: ModelConfig, *, is_decode: bool) -> torch.Tensor:
    """Routed-expert FFN. x is [1, T, H]; returns [1, T, H].

    Prefill writes into the input in place (``fused_experts_impl``), decode
    allocates (``fused_experts_decode_impl``); the caller captured ``residual``
    before the call either way, so the overwrite is safe.
    """
    T = x.shape[1]
    xf = x.reshape(-1, config.hidden_dim)   # [T, H]

    router_logits = linear(xf, weights[f"layers.{layer_idx}.mlp.router.weight"])
    topk_weights, topk_ids = fused_topk(
        xf, router_logits, config.n_experts_per_tok, _MOE_RENORMALIZE)

    if is_decode:
        out = _decode_routed(xf, topk_weights, topk_ids, moe_cache, layer_idx)
    else:
        out = _prefill_routed(xf, topk_weights, topk_ids, moe_cache, layer_idx,
                              config.n_experts)
    return out.reshape(1, T, config.hidden_dim)


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


def prefill_forward_moe(
    weights: DenseWeights,
    config: ModelConfig,
    moe_cache,
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
        if config.qk_norm:
            q, k = qk_norm(
                q, k,
                weights[f"layers.{layer_idx}.self_attn.q_norm.weight"],
                weights[f"layers.{layer_idx}.self_attn.k_norm.weight"],
                config.head_dim)
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
        x = _moe_ffn(x, weights, moe_cache, layer_idx, config, is_decode=False)
        x = residual + x

    kv_cache._len = seq_len

    x = rms_norm(x, weights["norm.weight"])
    logits = linear(x[:, -1, :], weights["lm_head.weight"])
    return logits.squeeze(0)


def decode_step_moe(
    weights: DenseWeights,
    config: ModelConfig,
    moe_cache,
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
        if config.qk_norm:
            q, k = qk_norm(
                q, k,
                weights[f"layers.{layer_idx}.self_attn.q_norm.weight"],
                weights[f"layers.{layer_idx}.self_attn.k_norm.weight"],
                config.head_dim)
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
        x = _moe_ffn(x, weights, moe_cache, layer_idx, config, is_decode=True)
        x = residual + x

    kv_cache.end_token()

    x = rms_norm(x, weights["norm.weight"])
    logits = linear(x[:, -1, :], weights["lm_head.weight"])
    return logits.squeeze(0)
