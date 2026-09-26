"""Converter tests: HF state_dict -> canonical disk layout -> read-back.

Verifies the real-model conversion path (weights/convert.py):
  1. Name mapping: mlp.gate.weight -> mlp.router.weight; fused 3D expert
     tensors (gate_up_proj / down_proj) are *not* emitted as dense.
  2. Per-expert split: each expert's w1 (gate, first I rows), w3 (up,
     rest), w2 (down) read back byte-identical to the HF originals.
  3. QK-norm tensors (q_norm / k_norm) survive the round trip.
  4. convert() (shard + index.json entry point) == convert_from_dict
     (in-memory entry point) on the same weights, and the layout index
     survives serialize -> load.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeForCausalLM

from plastic_infer.weights.convert import (
    convert,
    convert_from_dict,
    map_hf_tensor,
    split_expert,
    split_state_dict,
)
from plastic_infer.weights.disk import DiskExpertSource, DiskLayerSource
from plastic_infer.weights.layout import LayoutIndex

from test_qwen3moe_equiv import (
    build_weights_and_bank,
    tiny_qwen3moe_config,
    tiny_qwen3moe_config_dict,
)


@pytest.fixture(scope="module")
def hf_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(7)
    model = Qwen3MoeForCausalLM(tiny_qwen3moe_config())
    model.eval()
    return model


@pytest.fixture(scope="module")
def hf_state_dict(hf_model: Qwen3MoeForCausalLM) -> dict[str, torch.Tensor]:
    return hf_model.state_dict()


@pytest.fixture(scope="module")
def config_dict() -> dict:
    return tiny_qwen3moe_config_dict()


# ---------------------------------------------------------------------------
# Name mapping / split helpers
# ---------------------------------------------------------------------------

class TestNameMapping:
    def test_gate_renamed_to_router(self) -> None:
        assert (map_hf_tensor("model.layers.3.mlp.gate.weight")
                == "layers.3.mlp.router.weight")

    def test_dense_tensors_strip_model_prefix(self) -> None:
        assert (map_hf_tensor("model.layers.0.self_attn.q_norm.weight")
                == "layers.0.self_attn.q_norm.weight")
        assert (map_hf_tensor("model.embed_tokens.weight")
                == "embed_tokens.weight")
        assert map_hf_tensor("model.norm.weight") == "norm.weight"
        assert map_hf_tensor("lm_head.weight") == "lm_head.weight"

    def test_fused_experts_return_none(self) -> None:
        assert (map_hf_tensor("model.layers.0.mlp.experts.gate_up_proj")
                is None)
        assert (map_hf_tensor("model.layers.0.mlp.experts.down_proj")
                is None)

    def test_per_expert_tensors_renamed_to_w1_w3_w2(self) -> None:
        """The released-checkpoint layout: experts.{eid}.{gate,up,down}."""
        assert (map_hf_tensor("model.layers.0.mlp.experts.7.gate_proj.weight")
                == "layers.0.mlp.experts.7.w1.weight")
        assert (map_hf_tensor("model.layers.0.mlp.experts.7.up_proj.weight")
                == "layers.0.mlp.experts.7.w3.weight")
        assert (map_hf_tensor("model.layers.0.mlp.experts.7.down_proj.weight")
                == "layers.0.mlp.experts.7.w2.weight")
        # A per-expert name must not be mistaken for the fused marker.
        assert (map_hf_tensor("model.layers.11.mlp.experts.100.gate_proj.weight")
                == "layers.11.mlp.experts.100.w1.weight")

    def test_split_state_dict_has_no_fused_or_gate(self,
                                                   hf_state_dict) -> None:
        """split_state_dict emits canonical dense names only."""
        dense, experts = split_state_dict(
            hf_state_dict, num_experts=4, moe_intermediate_size=16)
        assert "layers.0.mlp.router.weight" in dense
        assert not any("gate_up_proj" in k or "down_proj" in k
                       for k in dense)
        assert not any(".gate." in k for k in dense)


# ---------------------------------------------------------------------------
# convert_from_dict -> disk -> read-back
# ---------------------------------------------------------------------------

class TestConvertRoundTrip:
    def test_layer_read_back_matches_hf(self, hf_state_dict, config_dict,
                                        tmp_path) -> None:
        out = tmp_path / "model"
        convert_from_dict(hf_state_dict, config_dict, out,
                          dtype=torch.float32)

        layout = LayoutIndex.load(out / "layout.index.json")
        src = DiskLayerSource(out, layout)

        # Shared tensors round-trip.
        shared = src.shared()
        assert set(shared) == {"embed_tokens.weight", "norm.weight",
                               "lm_head.weight"}
        assert torch.equal(shared["embed_tokens.weight"],
                           hf_state_dict["model.embed_tokens.weight"])
        assert torch.equal(shared["norm.weight"],
                           hf_state_dict["model.norm.weight"])
        assert torch.equal(shared["lm_head.weight"],
                           hf_state_dict["lm_head.weight"])

        # Layer 0 dense tensors: router renamed, q_norm present,
        # fused experts absent.
        layer = src.layer(0)
        assert "layers.0.mlp.router.weight" in layer
        assert "layers.0.self_attn.q_norm.weight" in layer
        assert "layers.0.self_attn.k_norm.weight" in layer
        assert torch.equal(layer["layers.0.mlp.router.weight"],
                           hf_state_dict["model.layers.0.mlp.gate.weight"])
        assert torch.equal(
            layer["layers.0.self_attn.q_norm.weight"],
            hf_state_dict["model.layers.0.self_attn.q_norm.weight"])
        assert not any("experts." in k or ".gate." in k for k in layer)

    def test_expert_read_back_byte_identical(self, hf_state_dict,
                                             config_dict, tmp_path) -> None:
        out = tmp_path / "model"
        convert_from_dict(hf_state_dict, config_dict, out,
                          dtype=torch.float32)

        layout = LayoutIndex.load(out / "layout.index.json")
        esrc = DiskExpertSource(out, layout)

        # Compare against split_expert of the fused originals.
        for layer in range(2):
            gu = hf_state_dict[f"model.layers.{layer}.mlp.experts.gate_up_proj"]
            dn = hf_state_dict[f"model.layers.{layer}.mlp.experts.down_proj"]
            for eid in range(4):
                want = split_expert(gu, dn, eid, 16)
                got = esrc.expert(layer, eid)
                for tag in ("w1", "w2", "w3"):
                    assert torch.equal(getattr(got, tag),
                                       getattr(want, tag)), (
                        f"layer {layer} expert {eid} {tag}")

    def test_layout_expert_names_and_sizes(self, hf_state_dict,
                                           config_dict, tmp_path) -> None:
        out = tmp_path / "model"
        convert_from_dict(hf_state_dict, config_dict, out,
                          dtype=torch.float32)

        layout = LayoutIndex.load(out / "layout.index.json")
        assert layout.n_layers == 2
        assert layout.num_experts(0) == 4
        names = layout.expert_tensor_names(0, 0)
        assert names == ["layers.0.mlp.experts.0.w1.weight",
                         "layers.0.mlp.experts.0.w2.weight",
                         "layers.0.mlp.experts.0.w3.weight"]
        # 16*64*4 + 64*16*4 + 16*64*4 (fp32)
        assert layout.expert_total_bytes(0) == 12 * 1024


def _to_per_expert_layout(flat: dict[str, torch.Tensor], num_experts: int,
                          inter: int) -> dict[str, torch.Tensor]:
    """Rewrite a fused 3D expert state dict into the released-checkpoint
    per-expert layout (experts.{eid}.{gate,up,down}_proj.weight).

    This is the layout the real Qwen3-30B-A3B on the Hub actually ships,
    as opposed to the fused tensors transformers builds in memory.
    """
    out: dict[str, torch.Tensor] = {}
    for name, t in flat.items():
        if name.endswith(".mlp.experts.gate_up_proj"):
            layer = name.split(".")[2]
            for eid in range(num_experts):
                base = f"model.layers.{layer}.mlp.experts.{eid}"
                out[f"{base}.gate_proj.weight"] = t[eid][:inter]
                out[f"{base}.up_proj.weight"] = t[eid][inter:]
        elif name.endswith(".mlp.experts.down_proj"):
            layer = name.split(".")[2]
            for eid in range(num_experts):
                out[f"model.layers.{layer}.mlp.experts.{eid}.down_proj.weight"] \
                    = t[eid]
        else:
            out[name] = t
    return out


# ---------------------------------------------------------------------------
# Released-checkpoint (per-expert) layout == fused in-memory layout
# ---------------------------------------------------------------------------

class TestPerExpertLayout:
    def test_converts_identically_to_fused(self, hf_state_dict, config_dict,
                                           tmp_path) -> None:
        """Both expert layouts must convert to byte-identical canonical
        files — the real model ships per-expert, the tests use fused."""
        per_expert = _to_per_expert_layout(hf_state_dict, 4, 16)

        a = tmp_path / "fused"
        convert_from_dict(hf_state_dict, config_dict, a, dtype=torch.float32)
        b = tmp_path / "per_expert"
        convert_from_dict(per_expert, config_dict, b, dtype=torch.float32)

        la = LayoutIndex.load(a / "layout.index.json")
        lb = LayoutIndex.load(b / "layout.index.json")
        assert la.to_dict() == lb.to_dict()

        for l in range(2):
            da = DiskLayerSource(a, la).layer(l)
            db = DiskLayerSource(b, lb).layer(l)
            assert set(da) == set(db)
            for k in da:
                assert torch.equal(da[k], db[k]), k

    def test_dense_tensors_exclude_per_expert_names(self, hf_state_dict,
                                                    tmp_path) -> None:
        """Dense reads must never pull the 384 expert tensors of a layer."""
        per_expert = _to_per_expert_layout(hf_state_dict, 4, 16)
        out = tmp_path / "model"
        convert_from_dict(per_expert, tiny_qwen3moe_config_dict(), out,
                          dtype=torch.float32)
        layer = DiskLayerSource(
            out, LayoutIndex.load(out / "layout.index.json")).layer(0)
        assert "layers.0.mlp.router.weight" in layer
        assert not any("experts." in k for k in layer)


# ---------------------------------------------------------------------------
# Shard-based convert() entry point == convert_from_dict
# ---------------------------------------------------------------------------

class TestConvertEntryPoint:
    def _write_hf_shards(self, hf_dir, state_dict, n_shards=2):
        """Write HF-style shards + model.safetensors.index.json."""
        names = sorted(state_dict)
        hf_dir.mkdir(parents=True, exist_ok=True)
        weight_map = {}
        chunk_size = (len(names) + n_shards - 1) // n_shards
        for i in range(0, len(names), chunk_size):
            shard_name = f"model-0000{i // chunk_size:02d}-of-0000{n_shards:02d}.safetensors"
            shard = {n: state_dict[n] for n in names[i:i + chunk_size]}
            save_file(shard, hf_dir / shard_name)
            for n in shard:
                weight_map[n] = shard_name
        index = {"metadata": {"total_size": 0},
                 "weight_map": weight_map}
        (hf_dir / "model.safetensors.index.json").write_text(
            json.dumps(index))
        (hf_dir / "config.json").write_text(
            json.dumps(tiny_qwen3moe_config_dict()))

    def test_convert_matches_convert_from_dict(self, hf_state_dict,
                                               config_dict, tmp_path) -> None:
        hf_dir = tmp_path / "hf"
        self._write_hf_shards(hf_dir, hf_state_dict, n_shards=2)

        a = tmp_path / "via_dict"
        convert_from_dict(hf_state_dict, config_dict, a,
                          dtype=torch.float32)

        b = tmp_path / "via_convert"
        convert(hf_dir, b, dtype=torch.float32)

        # Same set of tensor files. config.json is metadata copied only
        # by convert() (convert_from_dict writes no metadata by design).
        tensor_files = {p.name for p in b.iterdir()}
        assert {p.name for p in a.iterdir()} == tensor_files - {"config.json"}
        assert "model.shared.safetensors" in tensor_files
        assert "layout.index.json" in tensor_files

        la = LayoutIndex.load(a / "layout.index.json")
        lb = LayoutIndex.load(b / "layout.index.json")
        assert la.to_dict() == lb.to_dict()

        # Byte-identical tensors (fp32, deterministic).
        for l in range(2):
            sa, sb = DiskLayerSource(a, la), DiskLayerSource(b, lb)
            da, db = sa.layer(l), sb.layer(l)
            assert set(da) == set(db)
            for k in da:
                assert torch.equal(da[k], db[k]), k

    def test_config_copied(self, hf_state_dict, tmp_path) -> None:
        hf_dir = tmp_path / "hf"
        self._write_hf_shards(hf_dir, hf_state_dict)
        out = tmp_path / "out"
        convert(hf_dir, out, dtype=torch.float32)
        assert (out / "config.json").exists()
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["num_experts"] == 4
        assert cfg["head_dim"] == 16


# ---------------------------------------------------------------------------
# End-to-end: converted dir drives the runner identically to in-memory
# ---------------------------------------------------------------------------

class TestConvertedDirRuns:
    def test_disk_sources_feed_runner(self, hf_model, hf_state_dict,
                                      config_dict, tmp_path) -> None:
        """The disk layout is a faithful copy: dense + expert sources
        reconstruct the same weights build_weights_and_bank uses."""
        out = tmp_path / "model"
        convert_from_dict(hf_state_dict, config_dict, out,
                          dtype=torch.float32)
        layout = LayoutIndex.load(out / "layout.index.json")

        dlayer = DiskLayerSource(out, layout)
        desrc = DiskExpertSource(out, layout)
        dense = dict(dlayer.shared())
        for l in range(layout.n_layers):
            dense.update(dlayer.layer(l))
        dense = {k: v.to(torch.float32) for k, v in dense.items()}

        dense_ref, bank = build_weights_and_bank(hf_model)
        assert set(dense) == set(dense_ref.w)
        for k in dense:
            assert torch.equal(dense[k], dense_ref.w[k]), k
        # Spot-check every expert through the disk source.
        for layer in range(2):
            for eid in range(4):
                want = bank[(layer, eid)]
                got = desrc.expert(layer, eid)
                for tag in ("w1", "w2", "w3"):
                    assert torch.equal(getattr(got, tag),
                                       getattr(want, tag)), (layer, eid, tag)
