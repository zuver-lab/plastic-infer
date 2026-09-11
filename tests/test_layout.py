"""Tests for weights.layout — pure logic, no I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plastic_infer.weights.layout import (
    ExpertLocation,
    LayoutIndex,
    TensorLocation,
    build_layout_from_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _dense_tensors() -> dict[str, int]:
    """Per-layer dense tensor sizes (bytes, fp16)."""
    hidden = 1024
    return {
        "input_layernorm.weight": hidden * 2,
        "self_attn.q_proj.weight": hidden * hidden * 2,
        "self_attn.k_proj.weight": hidden * hidden * 2,
        "self_attn.v_proj.weight": hidden * hidden * 2,
        "self_attn.o_proj.weight": hidden * hidden * 2,
        "post_attention_layernorm.weight": hidden * 2,
        "mlp.gate_proj.weight": hidden * hidden * 4,  # ffn up
        "mlp.up_proj.weight": hidden * hidden * 4,
        "mlp.down_proj.weight": hidden * 4 * hidden,
    }


def _expert_tensors() -> dict[str, int]:
    """Per-expert tensor sizes (MoE ffn)."""
    hidden, inter = 1024, 2048
    return {
        "w1.weight": hidden * inter * 2,   # gate
        "w2.weight": inter * hidden * 2,   # down
        "w3.weight": hidden * inter * 2,   # up
    }


@pytest.fixture
def dense_layout() -> LayoutIndex:
    return build_layout_from_manifest(
        n_layers=4,
        per_layer_dense_tensors=_dense_tensors(),
        shared_tensors={
            "model.embed_tokens.weight": 32000 * 1024 * 2,
            "model.norm.weight": 1024 * 2,
            "lm_head.weight": 32000 * 1024 * 2,
        },
    )


@pytest.fixture
def moe_layout() -> LayoutIndex:
    return build_layout_from_manifest(
        n_layers=4,
        per_layer_dense_tensors={
            k: v for k, v in _dense_tensors().items()
            if "mlp" not in k   # MoE replaces dense FFN
        },
        experts_per_layer=8,
        per_expert_tensors=_expert_tensors(),
        shared_tensors={
            "model.embed_tokens.weight": 32000 * 1024 * 2,
            "model.norm.weight": 1024 * 2,
            "lm_head.weight": 32000 * 1024 * 2,
        },
    )


# ---------------------------------------------------------------------------
# Dense tests
# ---------------------------------------------------------------------------

class TestDenseLayout:
    def test_n_layers(self, dense_layout: LayoutIndex) -> None:
        assert dense_layout.n_layers == 4

    def test_is_not_moe(self, dense_layout: LayoutIndex) -> None:
        assert not dense_layout.is_moe()

    def test_layer_file(self, dense_layout: LayoutIndex) -> None:
        assert dense_layout.layer_file(0) == "model.layers.0.safetensors"
        assert dense_layout.layer_file(3) == "model.layers.3.safetensors"

    def test_tensor_offsets_sequential(self,
                                       dense_layout: LayoutIndex) -> None:
        """Tensors within a layer are packed sequentially."""
        names = dense_layout.layer_tensor_names(0)
        prev_end = 0
        for name in names:
            loc = dense_layout.tensor(name)
            assert loc.offset == prev_end
            assert loc.file == "model.layers.0.safetensors"
            prev_end = loc.offset + loc.nbytes

    def test_layer_total_bytes(self, dense_layout: LayoutIndex) -> None:
        expected = sum(_dense_tensors().values())
        assert dense_layout.layer_total_bytes(0) == expected
        assert dense_layout.layer_total_bytes(3) == expected

    def test_dense_per_layer_equals_layer_total(
        self, dense_layout: LayoutIndex,
    ) -> None:
        for i in range(4):
            assert (dense_layout.dense_per_layer_bytes(i)
                    == dense_layout.layer_total_bytes(i))

    def test_total_dense_bytes(self, dense_layout: LayoutIndex) -> None:
        layer_bytes = dense_layout.layer_total_bytes(0)
        assert dense_layout.total_dense_bytes == 4 * layer_bytes

    def test_total_expert_zero(self, dense_layout: LayoutIndex) -> None:
        assert dense_layout.total_expert_bytes == 0

    def test_shared_tensors(self, dense_layout: LayoutIndex) -> None:
        loc = dense_layout.tensor("model.embed_tokens.weight")
        assert loc.file == "model.shared.safetensors"
        assert loc.offset == 0
        assert loc.nbytes == 32000 * 1024 * 2


# ---------------------------------------------------------------------------
# MoE tests
# ---------------------------------------------------------------------------

class TestMoELayout:
    def test_is_moe(self, moe_layout: LayoutIndex) -> None:
        assert moe_layout.is_moe()

    def test_num_experts(self, moe_layout: LayoutIndex) -> None:
        assert moe_layout.num_experts(0) == 8
        assert moe_layout.num_experts(3) == 8

    def test_expert_location(self, moe_layout: LayoutIndex) -> None:
        exp = moe_layout.expert(0, 0)
        assert isinstance(exp, ExpertLocation)
        assert exp.layer == 0
        assert exp.expert_id == 0
        assert len(exp.tensors) == 3  # w1, w2, w3

    def test_expert_bytes_consistent(self, moe_layout: LayoutIndex) -> None:
        exp0 = moe_layout.expert(0, 0)
        exp3 = moe_layout.expert(0, 3)
        assert exp0.total_bytes == exp3.total_bytes
        assert exp0.total_bytes == sum(_expert_tensors().values())

    def test_experts_are_sequential(self, moe_layout: LayoutIndex) -> None:
        """Expert 1 starts right after expert 0 ends."""
        exp0_end = max(
            t.offset + t.nbytes for t in moe_layout.expert(0, 0).tensors
        )
        exp1_start = min(
            t.offset for t in moe_layout.expert(0, 1).tensors
        )
        assert exp0_end == exp1_start

    def test_expert_file_is_layer_file(self, moe_layout: LayoutIndex) -> None:
        exp = moe_layout.expert(1, 5)
        for t in exp.tensors:
            assert t.file == moe_layout.layer_file(1)

    def test_total_expert_bytes(self, moe_layout: LayoutIndex) -> None:
        per_expert = moe_layout.expert_total_bytes(0)
        expected = 4 * 8 * per_expert
        assert moe_layout.total_expert_bytes == expected

    def test_dense_per_layer_excludes_experts(
        self, moe_layout: LayoutIndex,
    ) -> None:
        dense_bytes = moe_layout.dense_per_layer_bytes(0)
        layer_bytes = moe_layout.layer_total_bytes(0)
        # Layer total = dense + (8 experts × per_expert)
        expected_layer = dense_bytes + 8 * moe_layout.expert_total_bytes(0)
        assert layer_bytes == expected_layer


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_roundtrip_dense(self, dense_layout: LayoutIndex,
                             tmp_path: Path) -> None:
        path = tmp_path / "index.json"
        dense_layout.save(path)
        loaded = LayoutIndex.load(path)
        assert loaded.n_layers == dense_layout.n_layers
        assert loaded.is_moe() == dense_layout.is_moe()
        # Spot-check a tensor
        loc_orig = dense_layout.tensor("model.layers.2.self_attn.q_proj.weight")
        loc_loaded = loaded.tensor("model.layers.2.self_attn.q_proj.weight")
        assert loc_orig == loc_loaded
        assert loaded.layer_total_bytes(0) == dense_layout.layer_total_bytes(0)

    def test_roundtrip_moe(self, moe_layout: LayoutIndex,
                           tmp_path: Path) -> None:
        path = tmp_path / "index.json"
        moe_layout.save(path)
        loaded = LayoutIndex.load(path)
        assert loaded.is_moe()
        assert loaded.num_experts(0) == 8
        exp_orig = moe_layout.expert(2, 5)
        exp_loaded = loaded.expert(2, 5)
        assert exp_orig.tensors == exp_loaded.tensors
