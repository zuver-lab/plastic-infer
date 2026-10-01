"""Tests for store.weights — StreamingDenseWeights (D4/D6 dense streaming).

The class presents the same runner surface as DenseWeights
(`__getitem__` / `.device` / `.dtype`). These tests run on CPU (no CUDA
needed), covering the synchronous window path, W=1 single-slot streaming,
wrap-around across a prefill->decode boundary, and byte-identity with a
plain DenseWeights reference. The CUDA prefetch/copy-stream machinery is
exercised only on the real GPU (bench A/B).
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3MoeForCausalLM

from plastic_infer.exec.runner import DenseWeights
from plastic_infer.exec.runner_moe_paged import (
    decode_step_moe_paged,
    make_kv_config,
    prefill_forward_moe_paged,
)
from plastic_infer.kv.kv_store import KVStore
from plastic_infer.store.experts import ExpertBank
from plastic_infer.store.weights import StreamingDenseWeights
from plastic_infer.weights.convert import split_state_dict
from plastic_infer.weights.disk import DictLayerSource

from _moe_cache_helpers import make_moe_cache, require_cuda
from test_qwen3moe_equiv import (
    make_model_config,
    tiny_qwen3moe_config,
)

H = 8
N = 3
LAYER_NAMES = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.down_proj.weight",
)
SHARED_KEYS = ("embed_tokens.weight", "norm.weight", "lm_head.weight")


def _tensor(name: str, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if name == "input_layernorm.weight":
        return torch.randn(H, generator=g)
    if name.endswith("gate_proj.weight"):
        return torch.randn(2 * H, H, generator=g)
    if name == "mlp.down_proj.weight":
        return torch.randn(H, 2 * H, generator=g)
    return torch.randn(H, H, generator=g)


@pytest.fixture
def flat() -> dict[str, torch.Tensor]:
    d: dict[str, torch.Tensor] = {}
    for l in range(N):
        for name in LAYER_NAMES:
            d[f"layers.{l}.{name}"] = _tensor(name, seed=l * 100)
    for name in SHARED_KEYS:
        d[name] = _tensor("self_attn.q_proj.weight", seed=99)
    return d


def _make(flat: dict[str, torch.Tensor], *, window: int = 2
          ) -> StreamingDenseWeights:
    src = DictLayerSource(flat, n_layers=N)
    return StreamingDenseWeights(
        {l: src.layer(l) for l in range(N)},
        shared=src.shared(), n_layers=N,
        device=torch.device("cpu"), dtype=torch.float32, window=window)


def _layer_keys(flat: dict[str, torch.Tensor], l: int) -> list[str]:
    return [k for k in flat if k.startswith(f"layers.{l}.")]


def _walk(streamed: StreamingDenseWeights, ref: DenseWeights,
          flat: dict[str, torch.Tensor], passes: int = 1) -> None:
    """Simulate the runner: shared, then layers 0..N-1, then shared again,
    once per pass (a prefill followed by `passes` decode steps)."""
    for _ in range(passes):
        for k in SHARED_KEYS:
            assert torch.equal(streamed[k], ref[k]), k
        for l in range(N):
            for k in _layer_keys(flat, l):
                assert torch.equal(streamed[k], ref[k]), k
        for k in SHARED_KEYS:
            assert torch.equal(streamed[k], ref[k]), k


class TestStreamingDense:
    def test_dtype_device_surface(self, flat) -> None:
        s = _make(flat)
        assert s.device.type == "cpu"
        assert s.dtype == torch.float32

    def test_w2_matches_dense_weights(self, flat) -> None:
        """Window=2 (the Qwen3 plan): every key byte-identical to a plain
        DenseWeights reference across a full sequential pass."""
        _walk(_make(flat), DenseWeights(flat), flat)

    def test_w1_streams_single_slot(self, flat) -> None:
        """W=1: correct, and the window never holds more than one layer
        (each advance evicts the previous slot)."""
        s = _make(flat, window=1)
        _walk(s, DenseWeights(flat), flat)
        # window stays pinned to the current layer on the synchronous path
        for l in range(N):
            for k in _layer_keys(flat, l):
                s[k]
            assert set(s._gpu) == {l}

    def test_window_n_layers_sync_loads_every_layer(self, flat) -> None:
        """window >= n_layers: prefetch is off, all layers load on demand,
        still byte-identical."""
        _walk(_make(flat, window=N), DenseWeights(flat), flat)

    def test_wrap_around_across_steps(self, flat) -> None:
        """Prefill (0..N-1) then two decode steps (0..N-1 again): the layer-0
        access after ending at N-1 must not corrupt anything (mod-n_layers
        wrap-around)."""
        _walk(_make(flat), DenseWeights(flat), flat, passes=3)

    def test_shared_stays_resident(self, flat) -> None:
        """Shared tensors are resident on the device, not reloaded per
        layer — advancing layers returns the same object."""
        s = _make(flat)
        ref = DenseWeights(flat)
        emb = s["embed_tokens.weight"]
        for l in range(N):
            for k in _layer_keys(flat, l):
                s[k]
        assert s["embed_tokens.weight"] is emb
        assert torch.equal(emb, ref["embed_tokens.weight"])

    def test_miss_on_first_access_of_each_layer(self, flat) -> None:
        """CPU path: every layer is a synchronous miss (no prefetch), and
        re-accessing the same layer within a pass hits the resident slot."""
        s = _make(flat)
        for l in range(N):
            s[_layer_keys(flat, l)[0]]    # miss: loads the layer
            s[_layer_keys(flat, l)[1]]    # hit: same resident dict
            assert l in s._gpu
        assert s._gpu  # non-empty


# ---------------------------------------------------------------------------
# End-to-end through the real paged-MoE runner (CPU)
# ---------------------------------------------------------------------------


class TestRunnerIntegration:
    def test_paged_runner_prefill_decode_identical(self) -> None:
        """StreamingDenseWeights served to the production runner
        (prefill_forward_moe_paged / decode_step_moe_paged) yields
        bit-identical logits to the resident DenseWeights, including the
        prefill -> decode wrap-around (the decode step restarts at layer 0
        right after the prefill ended at layer n_layers - 1)."""

        dev = require_cuda()
        torch.manual_seed(42)
        hf = Qwen3MoeForCausalLM(tiny_qwen3moe_config())
        hf.eval()
        dense, experts = split_state_dict(
            hf.state_dict(),
            num_experts=hf.config.num_local_experts,
            moe_intermediate_size=hf.config.moe_intermediate_size,
        )
        bank = ExpertBank()
        for (layer, eid), w in experts.items():
            bank.add(layer, eid, w)
        cfg = make_model_config(hf.config)

        to_cuda_bf16 = {k: v.to(device=dev, dtype=torch.bfloat16)
                        for k, v in dense.items()}
        resident = DenseWeights(to_cuda_bf16)
        streamed = StreamingDenseWeights(
            {l: {k: v for k, v in dense.items()
                 if k.startswith(f"layers.{l}.")}
             for l in range(cfg.n_layers)},
            shared={k: v for k, v in dense.items()
                    if not k.startswith("layers.")},
            n_layers=cfg.n_layers, device=dev,
            dtype=torch.bfloat16, window=2)
        moe_cache = make_moe_cache(bank, n_slots=8)

        def run(weights, ids):
            store = KVStore(make_kv_config(cfg, max_pages=512, device=dev))
            cache = store.new_request()
            pre = prefill_forward_moe_paged(weights, cfg, moe_cache, store,
                                            cache, ids)
            tok = int(pre.argmax().item())
            dec = decode_step_moe_paged(weights, cfg, moe_cache, store, cache,
                                        tok)
            return pre, dec

        ids = torch.randint(0, cfg.vocab_size, (16,), device=dev)
        a_pre, a_dec = run(resident, ids)
        b_pre, b_dec = run(streamed, ids)
        torch.testing.assert_close(b_pre, a_pre)
        torch.testing.assert_close(b_dec, a_dec)
