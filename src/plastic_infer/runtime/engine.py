"""Engine: plan -> pools -> runner for a converted model directory.

Loads a directory produced by weights.convert (layout.index.json +
per-layer safetensors + config.json), plans each request against a
DeviceProfile, builds the three-tier pools, and runs prefill + decode
through the composition runner (runner_moe_paged).

For Qwen3-30B-A3B the plan comes out as: dense weights fully in host
RAM and copied to GPU (HOST source, W = n_layers), experts HOST_FIRST
(host LRU + disk spill), KV paged on GPU. Long-context KV beyond the
GPU pool (HOST_STREAM/REJECT) is future work and is rejected clearly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..exec.runner import DenseWeights, ModelConfig
from ..exec.runner_moe_paged import (
    decode_step_moe_paged,
    make_kv_config,
    prefill_forward_moe_paged,
)
from ..kv.kv_store import KVStore
from ..planning.budget import MemoryPlanner, ModelMeta, Plan, RequestMeta
from ..planning.profile import DeviceProfile
from ..planning.wiring import pool_budgets
from ..store.experts import ExpertSlotPool, HostExpertLru
from ..weights.disk import DiskExpertSource, DiskLayerSource
from ..weights.layout import LayoutIndex


def build_model_meta(layout: LayoutIndex, cfg: dict) -> ModelMeta:
    """ModelMeta from the layout's measured sizes + config shapes."""
    n_layers = layout.n_layers
    dtype_bytes = layout.dtype_bytes
    head_dim = cfg.get("head_dim") or (
        cfg["hidden_size"] // cfg["num_attention_heads"])
    return ModelMeta(
        n_layers=n_layers,
        dense_bytes=layout.total_dense_bytes,
        per_layer_dense=tuple(
            layout.dense_per_layer_bytes(l) for l in range(n_layers)),
        experts_per_layer=tuple(
            layout.num_experts(l) for l in range(n_layers)),
        expert_row_bytes=tuple(
            layout.expert_total_bytes(l) for l in range(n_layers)),
        kv_bytes_per_token=(
            n_layers * 2 * cfg["num_key_value_heads"] * head_dim * dtype_bytes),
    )


def build_runner_config(cfg: dict, *, dtype: torch.dtype,
                        qk_norm: bool) -> ModelConfig:
    """Runner ModelConfig from an HF config dict (head_dim is explicit
    in Qwen3, not derived; rope_theta may live top-level or in
    rope_parameters)."""
    head_dim = cfg.get("head_dim") or (
        cfg["hidden_size"] // cfg["num_attention_heads"])
    rope_base = cfg.get("rope_theta")
    if rope_base is None:
        rope_base = cfg.get("rope_parameters", {}).get("rope_theta", 10000.0)
    return ModelConfig(
        n_layers=cfg["num_hidden_layers"],
        n_heads=cfg["num_attention_heads"],
        n_kv_heads=cfg["num_key_value_heads"],
        head_dim=head_dim,
        hidden_dim=cfg["hidden_size"],
        intermediate_dim=cfg.get("moe_intermediate_size",
                                 cfg["intermediate_size"]),
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_position_embeddings"],
        rope_base=float(rope_base),
        dtype=dtype,
        qk_norm=qk_norm,
        n_experts=cfg.get("num_experts", 0),
        n_experts_per_tok=cfg.get("num_experts_per_tok", 0),
    )


@dataclass
class RunResult:
    tokens: list[int]
    plan: Plan
    metrics: dict


class Engine:
    def __init__(self, model_dir: str | Path, profile: DeviceProfile,
                 device: torch.device | None = None) -> None:
        self.dir = Path(model_dir)
        self.profile = profile
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.layout = LayoutIndex.load(self.dir / "layout.index.json")
        with open(self.dir / "config.json") as f:
            self.cfg = json.load(f)
        self.dtype = torch.bfloat16 if self.cfg.get("torch_dtype") == "bfloat16" \
            else torch.float32
        self.model = build_model_meta(self.layout, self.cfg)
        qk_norm = any("self_attn.q_norm.weight" in n
                      for n in self.layout.layer_tensor_names(0))
        self.config = build_runner_config(self.cfg, dtype=self.dtype,
                                          qk_norm=qk_norm)
        self.dense_src = DiskLayerSource(self.dir, self.layout)
        self.expert_src = DiskExpertSource(self.dir, self.layout)

    def _load_dense(self) -> DenseWeights:
        """Load all dense + shared weights onto the run device."""
        flat = dict(self.dense_src.shared())
        for l in range(self.layout.n_layers):
            flat.update(self.dense_src.layer(l))
        return DenseWeights({k: v.to(self.device) for k, v in flat.items()})

    def run(self, input_ids: list[int], *, max_new_tokens: int = 16,
            request: RequestMeta | None = None) -> RunResult:
        """Prefill + greedy decode. Returns tokens (input + generated)."""
        seq_len = len(input_ids)
        req = request or RequestMeta(
            seq_budget=self.cfg["max_position_embeddings"],
            gen_tokens=max_new_tokens)
        plan = MemoryPlanner().plan(self.profile, self.model, req)
        b = pool_budgets(plan, self.model)

        if plan.seq_weight_source != "HOST":
            raise NotImplementedError(
                f"MoE engine requires dense weights in host RAM (plan says "
                f"{plan.seq_weight_source}); a model whose dense weights "
                f"exceed host RAM needs the streamed dense×MoE runner "
                f"(future work)")
        if seq_len + max_new_tokens > plan.kv_hot_tokens:
            raise ValueError(
                f"request ({seq_len} prompt + {max_new_tokens} gen) exceeds "
                f"the GPU KV pool ({plan.kv_hot_tokens} tokens); "
                f"HOST_STREAM long context is future work")

        weights = self._load_dense()

        per_expert = self.layout.expert_total_bytes(0)
        host = HostExpertLru(self.expert_src,
                             budget_bytes=plan.expert_host_slots * per_expert,
                             per_expert_bytes=per_expert)
        pool = ExpertSlotPool(host, b.expert_slots_bytes, device=self.device)

        kv_cfg = make_kv_config(
            self.config,
            max_pages=b.kv_pages * self.config.n_layers,  # physical pages
            max_host_chunks=64,
            device=self.device,
            dtype=self.config.dtype)
        store = KVStore(kv_cfg)
        cache = store.new_request()

        ids = torch.tensor(input_ids, device=self.device)
        t0 = time.monotonic()
        logits = prefill_forward_moe_paged(weights, self.config, pool, store,
                                           cache, ids)
        prefill_s = time.monotonic() - t0

        tokens = list(input_ids)
        # First generated token comes from the prefill logits (they
        # predict the token after the prompt). decode_step then takes
        # that token as input and predicts the next, writing its KV at
        # the new position — same convention the equivalence tests use.
        tokens.append(int(logits.argmax().item()))
        t_dec = time.monotonic()
        for _ in range(max_new_tokens - 1):
            logits = decode_step_moe_paged(weights, self.config, pool, store,
                                           cache, tokens[-1])
            tokens.append(int(logits.argmax().item()))
        decode_s = time.monotonic() - t_dec

        store.free_request(cache)
        metrics = {
            "prefill_seconds": prefill_s,
            "decode_seconds": decode_s,
            "decode_tokens_per_s": (max_new_tokens / decode_s
                                    if decode_s > 0 else 0.0),
            "expert_gpu_hit_rate": pool.hit_rate,
            "expert_gpu_misses": pool.misses,
            "expert_host_hit_rate": host.hit_rate,
            "expert_host_misses": host.misses,
        }
        return RunResult(tokens=tokens, plan=plan, metrics=metrics)
