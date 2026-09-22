"""Tests for store.weights — the DenseWindow rotating layer window (M3).

Covers byte budget, rotating residency (W=1 streaming), the D7 pin
invariant, and miss/hit accounting. Like the expert pool, all tensors
stay on CPU — no GPU needed.
"""

from __future__ import annotations

import pytest
import torch

from plastic_infer.store.weights import DenseWindow
from plastic_infer.weights.disk import DictLayerSource

H = 8
LAYER_NAMES = (
    "input_layernorm.weight",        # [8]       32 B
    "self_attn.q_proj.weight",       # [8,8]    256 B
    "self_attn.o_proj.weight",       # [8,8]    256 B
    "mlp.gate_proj.weight",          # [16,8]   512 B
    "mlp.down_proj.weight",          # [8,16]   512 B
)
LAYER_BYTES = 32 + 256 + 256 + 512 + 512  # 1568


def _tensor(name: str, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if name == "input_layernorm.weight":
        shape = (H,)
    elif name.endswith("gate_proj.weight"):
        shape = (2 * H, H)
    elif name == "mlp.down_proj.weight":
        shape = (H, 2 * H)
    else:  # q_proj / o_proj
        shape = (H, H)
    return torch.randn(*shape, generator=g)


def _source(n_layers: int = 3) -> DictLayerSource:
    flat: dict[str, torch.Tensor] = {}
    for l in range(n_layers):
        for name in LAYER_NAMES:
            flat[f"layers.{l}.{name}"] = _tensor(name, seed=l * 100)
    flat["norm.weight"] = torch.randn(H)
    return DictLayerSource(flat, n_layers=n_layers)


@pytest.fixture
def source() -> DictLayerSource:
    return _source()


def _window(source, n_layers: int) -> DenseWindow:
    return DenseWindow(source, budget_bytes=n_layers * LAYER_BYTES)


class TestWindowBudget:
    def test_ensure_release_roundtrip(self, source: DictLayerSource) -> None:
        window = _window(source, n_layers=1)
        w = window.ensure(0)
        assert set(w.keys()) == {f"layers.0.{n}" for n in LAYER_NAMES}
        assert all(t.dtype == torch.float32 for t in w.values())
        assert window.pool.used_bytes == LAYER_BYTES
        window.release(0)
        assert window.pool.used_bytes == LAYER_BYTES  # released, still resident

    def test_miss_then_hit(self, source: DictLayerSource) -> None:
        window = _window(source, n_layers=1)
        w1 = window.ensure(0)
        window.release(0)
        w2 = window.ensure(0)
        window.release(0)
        assert window.hits == 1
        assert window.misses == 1
        assert w2["layers.0.input_layernorm.weight"] is \
            w1["layers.0.input_layernorm.weight"]   # same resident tensor

    def test_data_matches_source(self, source: DictLayerSource) -> None:
        window = _window(source, n_layers=1)
        got = window.ensure(2)
        window.release(2)
        ref = source.layer(2)
        for name, t in ref.items():
            assert torch.allclose(got[name], t)

    def test_budget_too_small_raises_on_ensure(
        self, source: DictLayerSource,
    ) -> None:
        window = DenseWindow(source, budget_bytes=LAYER_BYTES - 1)
        with pytest.raises(AssertionError):
            window.ensure(0)

    def test_double_release_raises(self, source: DictLayerSource) -> None:
        window = _window(source, n_layers=1)
        window.ensure(0)
        window.release(0)
        with pytest.raises(AssertionError):
            window.release(0)


class TestRotatingResidency:
    def test_window_1_streams_every_layer(self, source: DictLayerSource) -> None:
        """W=1: each layer evicts the previous one; all loads are misses."""
        window = _window(source, n_layers=1)
        for l in range(3):
            window.ensure(l)
            window.release(l)
            assert window.pool.used_bytes <= LAYER_BYTES
        assert window.misses == 3
        assert window.hits == 0
        assert window.miss_rate == 1.0

    def test_window_1_evicts_previous_layer(self, source: DictLayerSource) -> None:
        window = _window(source, n_layers=1)
        window.ensure(0)
        window.release(0)
        window.ensure(1)
        window.release(1)
        assert 0 not in window.pool    # evicted to make room
        assert 1 in window.pool
        assert window.pool.used_bytes == LAYER_BYTES

    def test_window_n_layers_keeps_all_resident(self, source) -> None:
        window = _window(source, n_layers=3)
        for l in range(3):
            window.ensure(l)
            window.release(l)
        assert window.misses == 3
        assert all(l in window.pool for l in range(3))
        # Re-touch: all hits now
        window.ensure(0)
        window.release(0)
        assert window.hits == 1
        assert window.miss_rate < 1.0

    def test_d7_pinned_survives_eviction(self, source: DictLayerSource) -> None:
        """Budget 2 layers: layer 1 pinned during load of layer 2 (D7)."""
        window = _window(source, n_layers=2)
        window.ensure(0)
        window.ensure(1)          # both resident and pinned
        window.release(0)         # only layer 0 unpinned

        window.ensure(2)          # needs a slot -> evicts unpinned LRU (0)
        window.release(1)
        window.release(2)

        assert 0 not in window.pool    # evicted (was the unpinned LRU)
        assert 1 in window.pool        # survived (D7)
        assert 2 in window.pool
        assert window.pool.used_bytes == 2 * LAYER_BYTES

    def test_all_pinned_blocks_overflow(self, source: DictLayerSource) -> None:
        """Budget 1 layer, layer 0 pinned: loading layer 1 must raise
        (D7 forbids evicting it, and the window refuses to overflow)."""
        window = _window(source, n_layers=1)
        window.ensure(0)
        with pytest.raises(AssertionError):
            window.ensure(1)
        window.release(0)
