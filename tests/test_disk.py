"""Tests for weights.disk — disk-backed dense weight sources (M3).

Gates:
  1. DiskLayerSource reads a layer's safetensors file back byte-for-byte.
  2. Sources are lazy: construction touches nothing; reads happen per
     layer() / shared() call.
  3. DictLayerSource (the HOST anchor) slices the flat dict correctly.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from plastic_infer.weights.disk import DiskLayerSource, DictLayerSource
from plastic_infer.weights.layout import build_layout_from_manifest


# ---------------------------------------------------------------------------
# Synthetic model: two layers + shared file, all tensors fp32
# ---------------------------------------------------------------------------

LAYER_TENSORS = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _shape_for(name: str) -> tuple[int, ...]:
    shapes = {
        "input_layernorm.weight": (16,),
        "self_attn.q_proj.weight": (16, 16),
        "self_attn.k_proj.weight": (8, 16),
        "self_attn.v_proj.weight": (8, 16),
        "self_attn.o_proj.weight": (16, 16),
        "post_attention_layernorm.weight": (16,),
        "mlp.gate_proj.weight": (32, 16),
        "mlp.up_proj.weight": (32, 16),
        "mlp.down_proj.weight": (16, 32),
    }
    return shapes[name]


def _layer_tensors(layer: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for name in LAYER_TENSORS:
        out[f"layers.{layer}.{name}"] = torch.randn(
            *_shape_for(name), generator=g)
    return out


def _shared_tensors(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "embed_tokens.weight": torch.randn(64, 16, generator=g),
        "norm.weight": torch.randn(16, generator=g),
        "lm_head.weight": torch.randn(64, 16, generator=g),
    }


def _write_model(tmp_path: pytest.TempPathFactory, n_layers: int = 2) -> None:
    """Write one safetensors file per layer plus a shared file."""
    for l in range(n_layers):
        save_file(_layer_tensors(l, seed=100 + l),
                  str(tmp_path / f"model.layers.{l}.safetensors"))
    save_file(_shared_tensors(seed=7), str(tmp_path / "model.shared.safetensors"))


def _layout(n_layers: int = 2) -> object:
    per_layer: dict[str, int] = {}
    for name, t in _layer_tensors(0, seed=100).items():
        rel = name.split(".", 2)[2]
        per_layer[rel] = t.numel() * t.element_size()
    shared = {k: v.numel() * v.element_size()
              for k, v in _shared_tensors(seed=7).items()}
    return build_layout_from_manifest(
        n_layers=n_layers,
        per_layer_dense_tensors=per_layer,
        shared_tensors=shared,
        layer_prefix="layers.{layer}",
    )


class TestDiskLayerSource:
    def test_layer_roundtrip_byte_identical(
        self, tmp_path,
    ) -> None:
        _write_model(tmp_path)
        layout = _layout()
        src = DiskLayerSource(tmp_path, layout)

        got = src.layer(0)
        ref = _layer_tensors(0, seed=100)
        assert set(got.keys()) == set(ref.keys())
        for name, t in ref.items():
            assert torch.equal(got[name], t), f"layer0 {name} differs"

        got1 = src.layer(1)
        ref1 = _layer_tensors(1, seed=101)
        assert torch.equal(got1["layers.1.mlp.gate_proj.weight"],
                           ref1["layers.1.mlp.gate_proj.weight"])

    def test_shared_roundtrip(self, tmp_path) -> None:
        _write_model(tmp_path)
        src = DiskLayerSource(tmp_path, _layout())
        got = src.shared()
        ref = _shared_tensors(seed=7)
        assert set(got.keys()) == set(ref.keys())
        for name, t in ref.items():
            assert torch.equal(got[name], t)

    def test_keys_are_canonical_full_names(self, tmp_path) -> None:
        _write_model(tmp_path)
        src = DiskLayerSource(tmp_path, _layout())
        got = src.layer(0)
        assert "layers.0.input_layernorm.weight" in got
        assert "layers.0.mlp.down_proj.weight" in got

    def test_construction_is_lazy(self, tmp_path) -> None:
        """No I/O at construction: a missing file is only a problem on read."""
        missing_dir = tmp_path / "does-not-exist"
        src = DiskLayerSource(missing_dir, _layout())
        with pytest.raises(FileNotFoundError):
            src.layer(0)

    def test_missing_layer_file_raises(self, tmp_path) -> None:
        # Only write the shared file, no layer files
        save_file(_shared_tensors(seed=7),
                  str(tmp_path / "model.shared.safetensors"))
        src = DiskLayerSource(tmp_path, _layout())
        with pytest.raises(FileNotFoundError):
            src.layer(0)


class TestDictLayerSource:
    def test_slices_layer_and_shared(self) -> None:
        flat = {}
        flat.update(_shared_tensors(seed=7))
        for l in range(2):
            flat.update(_layer_tensors(l, seed=100 + l))
        src = DictLayerSource(flat, n_layers=2)

        l0 = src.layer(0)
        assert set(l0.keys()) == set(_layer_tensors(0, seed=100).keys())
        assert all(not k.startswith("layers.") for k in src.shared())
        assert torch.equal(src.layer(1)["layers.1.mlp.up_proj.weight"],
                           flat["layers.1.mlp.up_proj.weight"])
