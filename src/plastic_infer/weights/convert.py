"""HF -> custom disk layout converter (end-to-end milestone).

Reads a HuggingFace Qwen3MoE directory (config.json + safetensors
shards + model.safetensors.index.json) and writes the canonical
per-layer format consumed by DiskLayerSource / DiskExpertSource:

  model.layers.{L}.safetensors    dense tensors + per-expert w1/w2/w3
  model.shared.safetensors        embed_tokens / norm / lm_head
  layout.index.json               the LayoutIndex (offsets, names)
  config.json (+ tokenizer files) copied for the engine

Name mapping (HF -> canonical, see map_hf_tensor):
  - router:  mlp.gate.weight  ->  mlp.router.weight  (runner uses router)
  - experts: fused 3D gate_up_proj / down_proj are split per expert into
    layers.{L}.mlp.experts.{eid}.{w1,w2,w3}.weight, where w1 = gate
    (first moe_intermediate_size rows), w3 = up (rest), w2 = down.

Memory-controlled: every HF tensor is read through safe_open (mmap), so
only the bytes actually sliced are paged in; at most one layer's weights
(~1.2GB for Qwen3-30B-A3B) is materialized at a time.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..store.experts import ExpertWeights
from .layout import build_layout_from_manifest

_DENSE_FUSED_EXPERTS = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
_SHARED_HF_NAMES = ("model.embed_tokens.weight", "model.norm.weight",
                    "lm_head.weight")


def map_hf_tensor(hf_name: str) -> str | None:
    """Map an HF Qwen3Moe tensor name to our canonical name.

    Returns None for the fused expert tensors (gate_up_proj /
    down_proj), which are split per-expert by the converter loop.
    """
    if hf_name == "lm_head.weight" or hf_name.startswith("model.embed_tokens.weight") \
            or hf_name.startswith("model.norm.weight"):
        return hf_name.removeprefix("model.")
    if hf_name.startswith("model.layers.") and not any(
            f in hf_name for f in _DENSE_FUSED_EXPERTS):
        out = hf_name.removeprefix("model.")        # layers.{L}....
        if ".mlp.gate.weight" in out:               # router rename
            return out.replace(".mlp.gate.weight", ".mlp.router.weight")
        return out
    return None


def split_expert(fused_gate_up: torch.Tensor, fused_down: torch.Tensor,
                 eid: int, inter: int) -> ExpertWeights:
    """Split fused 3D expert tensors into one expert's w1/w2/w3.

    gate_up_proj[eid] is [2*inter, hidden]; the first `inter` rows are
    the gate (w1), the rest the up (w3). down_proj[eid] is [hidden,
    inter] = w2. Order matches HF: gate, up = linear(x, gate_up_proj[eid])
    .chunk(2, dim=-1).
    """
    gu = fused_gate_up[eid]              # [2I, H]
    return ExpertWeights(
        w1=gu[:inter],                   # gate
        w2=fused_down[eid],              # down
        w3=gu[inter:],                   # up
    )


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


# ---------------------------------------------------------------------------
# Layered conversion core (shared by shard and dict sources)
# ---------------------------------------------------------------------------


class _DictSource:
    """Reads HF-named tensors from an in-memory state dict (tests)."""

    def __init__(self, flat: dict[str, torch.Tensor]) -> None:
        self.flat = flat

    def layer_dense_names(self, layer: int) -> list[str]:
        prefix = f"model.layers.{layer}."
        return sorted(
            n for n in self.flat
            if n.startswith(prefix) and not any(f in n for f in _DENSE_FUSED_EXPERTS)
        )

    def shared_names(self) -> list[str]:
        return sorted(n for n in self.flat if n in _SHARED_HF_NAMES)

    def get(self, name: str) -> torch.Tensor:
        return self.flat[name]

    def expert_row(self, fused_name: str, eid: int) -> torch.Tensor:
        return self.flat[fused_name][eid]


class _ShardSource:
    """Reads HF-named tensors from safetensors shards (mmap'd).

    Handles are cached per shard for the duration of a conversion, so
    the fused-expert loop (128 get_slice calls per layer) does not
    reopen files. mmap pages in only the sliced bytes.
    """

    def __init__(self, hf_dir: Path, weight_map: dict[str, str]) -> None:
        self.dir = hf_dir
        self.weight_map = weight_map
        self._handles: dict[str, safe_open] = {}

    def _handle(self, name: str):
        shard = self.weight_map[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.dir / shard), framework="pt")
        return self._handles[shard]

    def layer_dense_names(self, layer: int) -> list[str]:
        prefix = f"model.layers.{layer}."
        return sorted(
            n for n in self.weight_map
            if n.startswith(prefix) and not any(f in n for f in _DENSE_FUSED_EXPERTS)
        )

    def shared_names(self) -> list[str]:
        return sorted(n for n in self.weight_map if n in _SHARED_HF_NAMES)

    def get(self, name: str) -> torch.Tensor:
        return self._handle(name).get_tensor(name)

    def expert_row(self, fused_name: str, eid: int) -> torch.Tensor:
        return self._handle(fused_name).get_slice(fused_name)[eid]

    def close(self) -> None:
        # safe_open 0.8 has no close(); drop the cached handles and let
        # GC unmap the mmaps (they are released once out of scope).
        self._handles.clear()


def _convert_layers(source, config: dict, out_dir: Path, dtype: torch.dtype) -> None:
    """Convert all layers + shared tensors via a named-tensor source."""
    n_layers = config["num_hidden_layers"]
    n_experts = config["num_experts"]
    inter = config["moe_intermediate_size"]
    out_dir.mkdir(parents=True, exist_ok=True)

    per_layer_dense: dict[str, int] = {}
    per_expert: dict[str, int] = {}
    for layer in range(n_layers):
        layer_dict: dict[str, torch.Tensor] = {}
        for hf_name in source.layer_dense_names(layer):
            canon = map_hf_tensor(hf_name)
            assert canon is not None, hf_name
            t = source.get(hf_name).to(dtype)
            layer_dict[canon] = t
            per_layer_dense[canon.split(".", 2)[2]] = \
                t.numel() * t.element_size()

        gu_name = f"model.layers.{layer}.mlp.experts.gate_up_proj"
        dn_name = f"model.layers.{layer}.mlp.experts.down_proj"
        for eid in range(n_experts):
            w1 = source.expert_row(gu_name, eid)[:inter].contiguous().to(dtype)
            w3 = source.expert_row(gu_name, eid)[inter:].contiguous().to(dtype)
            w2 = source.expert_row(dn_name, eid).to(dtype)
            for tag, t in (("w1", w1), ("w2", w2), ("w3", w3)):
                layer_dict[f"layers.{layer}.mlp.experts.{eid}.{tag}.weight"] = t
            if eid == 0:
                per_expert = {
                    f"{tag}.weight": t.numel() * t.element_size()
                    for tag, t in (("w1", w1), ("w2", w2), ("w3", w3))
                }
        save_file(layer_dict, out_dir / f"model.layers.{layer}.safetensors")

    shared: dict[str, torch.Tensor] = {}
    shared_sizes: dict[str, int] = {}
    for hf_name in source.shared_names():
        canon = map_hf_tensor(hf_name)
        t = source.get(hf_name).to(dtype)
        shared[canon] = t
        shared_sizes[canon] = t.numel() * t.element_size()
    save_file(shared, out_dir / "model.shared.safetensors")

    layout = build_layout_from_manifest(
        n_layers=n_layers,
        per_layer_dense_tensors=per_layer_dense,
        experts_per_layer=n_experts,
        per_expert_tensors=per_expert,
        shared_tensors=shared_sizes,
        dtype_bytes=_element_size(dtype),
        layer_prefix="layers.{layer}",
        expert_prefix="layers.{layer}.mlp.experts.{eid}",
    )
    layout.save(out_dir / "layout.index.json")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def convert(hf_dir: str | Path, out_dir: str | Path,
            dtype: torch.dtype = torch.bfloat16) -> None:
    """Convert a HuggingFace Qwen3MoE directory to our disk layout.

    Reads model.safetensors.index.json (or a single model.safetensors)
    to locate tensors, then streams layer by layer. Copies config.json
    and tokenizer files for the engine.
    """
    hf_dir = Path(hf_dir)
    out_dir = Path(out_dir)
    with open(hf_dir / "config.json") as f:
        config = json.load(f)

    index_path = hf_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        single = hf_dir / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(
                f"{hf_dir}: no model.safetensors.index.json or model.safetensors")
        with safe_open(str(single), framework="pt") as f:
            weight_map = {n: "model.safetensors" for n in f.keys()}

    source = _ShardSource(hf_dir, weight_map)
    try:
        _convert_layers(source, config, out_dir, dtype)
    finally:
        source.close()

    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json",
               "generation_config.json"):
        p = hf_dir / fn
        if p.exists():
            shutil.copy2(p, out_dir / fn)


def convert_from_dict(flat: dict[str, torch.Tensor], config: dict,
                      out_dir: str | Path,
                      dtype: torch.dtype = torch.float32) -> None:
    """Convert an in-memory HF state dict (tests; no metadata copying)."""
    _convert_layers(_DictSource(flat), config, Path(out_dir), dtype)


def split_state_dict(flat: dict[str, torch.Tensor], num_experts: int,
                     moe_intermediate_size: int
                     ) -> tuple[dict[str, torch.Tensor],
                                dict[tuple[int, int], ExpertWeights]]:
    """Map an HF Qwen3Moe state dict to our canonical in-memory form.

    Returns (dense: dict[str, Tensor], experts: dict[(layer, eid),
    ExpertWeights]) using the same naming map and fused-expert split as
    convert(). This is the runner-equivalence anchor (no disk I/O).
    """
    dense: dict[str, torch.Tensor] = {}
    experts: dict[tuple[int, int], ExpertWeights] = {}
    for name, t in flat.items():
        canon = map_hf_tensor(name)
        if canon is not None:
            dense[canon] = t
    layer_ids = {int(n.split(".")[2]) for n in flat
                 if n.startswith("model.layers.")}
    for layer in range(max(layer_ids, default=-1) + 1):
        gu = flat[f"model.layers.{layer}.mlp.experts.gate_up_proj"]
        dn = flat[f"model.layers.{layer}.mlp.experts.down_proj"]
        for eid in range(num_experts):
            experts[(layer, eid)] = split_expert(gu, dn, eid,
                                                 moe_intermediate_size)
    return dense, experts
