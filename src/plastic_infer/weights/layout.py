"""Weight layout: how model weights are organized on disk.

Design follows AirLLM's conclusion (§5.3 of DESIGN.md):
  - One safetensors file = one streaming unit (per layer)
  - Dense model: one file per layer (model.layers.0.safetensors ...)
  - MoE: per-expert sub-ranges within each layer's file, seekable
  - Shared / non-layer tensors: one extra file (model.shared.safetensors)

This module is pure logic — no I/O, no tensors. It just maps
"layer N expert K" to (file, offset, nbytes) so the loader knows
what to read.

Layout is described by a LayoutIndex built from a manifest dict
(think: model.safetensors.index.json from HF).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorLocation:
    """Where a single tensor lives."""
    file: str              # safetensors file name (relative to model dir)
    offset: int            # byte offset within the file's data section
    nbytes: int            # tensor size in bytes


@dataclass(frozen=True)
class ExpertLocation:
    """Where one expert's weight row lives (multiple tensors)."""
    layer: int
    expert_id: int
    tensors: tuple[TensorLocation, ...]  # w1, w2, w3 (up/gate/down) etc.

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)


class LayoutIndex:
    """Immutable index of where every weight tensor lives on disk.

    Two representations coexist:
      - flat: tensor_name -> TensorLocation
      - structured: per-layer / per-expert views
    """

    def __init__(
        self,
        layer_files: dict[int, str],          # layer_idx -> file_name
        shared_file: str | None,              # non-layer tensors
        tensor_map: dict[str, TensorLocation],# name -> location
        layer_tensors: dict[int, list[str]],  # layer_idx -> [tensor names]
        expert_tensors: dict[int, dict[int, list[str]]] | None,
        # layer_idx -> { expert_id -> [tensor names] }
        dtype_bytes: int = 2,                 # fp16/bf16 = 2
        n_layers: int = 0,
    ) -> None:
        self._layer_files = dict(layer_files)
        self._shared_file = shared_file
        self._tensor_map = dict(tensor_map)
        self._layer_tensors = {k: list(v) for k, v in layer_tensors.items()}
        if expert_tensors is None:
            self._expert_tensors: dict[int, dict[int, list[str]]] = {}
        else:
            self._expert_tensors = {
                layer: {eid: list(tensors) for eid, tensors in exp.items()}
                for layer, exp in expert_tensors.items()
            }
        self.dtype_bytes = dtype_bytes
        self.n_layers = n_layers

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def tensor(self, name: str) -> TensorLocation:
        return self._tensor_map[name]

    def layer_file(self, layer_idx: int) -> str:
        return self._layer_files[layer_idx]

    def shared_file(self) -> str | None:
        """File name holding the non-layer tensors (embed/norm/lm_head)."""
        return self._shared_file

    def shared_tensor_names(self) -> list[str]:
        """Names of the tensors stored in the shared file."""
        if self._shared_file is None:
            return []
        return [n for n, loc in self._tensor_map.items()
                if loc.file == self._shared_file]

    def layer_tensor_names(self, layer_idx: int) -> list[str]:
        return list(self._layer_tensors[layer_idx])

    def layer_total_bytes(self, layer_idx: int) -> int:
        return sum(self._tensor_map[n].nbytes
                   for n in self._layer_tensors[layer_idx])

    def is_moe(self) -> bool:
        return len(self._expert_tensors) > 0

    def num_experts(self, layer_idx: int) -> int:
        return len(self._expert_tensors.get(layer_idx, {}))

    def expert(self, layer_idx: int, expert_id: int) -> ExpertLocation:
        """Location of all tensors for one expert in one layer."""
        names = self._expert_tensors[layer_idx][expert_id]
        tensors = tuple(self._tensor_map[n] for n in names)
        return ExpertLocation(
            layer=layer_idx, expert_id=expert_id, tensors=tensors,
        )

    def expert_total_bytes(self, layer_idx: int) -> int:
        if layer_idx not in self._expert_tensors:
            return 0
        first_eid = next(iter(self._expert_tensors[layer_idx]))
        return self.expert(layer_idx, first_eid).total_bytes

    def dense_per_layer_bytes(self, layer_idx: int) -> int:
        """Bytes per layer *excluding* expert tensors (dense part only)."""
        dense_names = [
            n for n in self._layer_tensors[layer_idx]
            if not _is_expert_tensor(n)
        ]
        return sum(self._tensor_map[n].nbytes for n in dense_names)

    @property
    def total_dense_bytes(self) -> int:
        return sum(self.layer_total_bytes(l) for l in range(self.n_layers)) \
               - self.total_expert_bytes

    @property
    def total_expert_bytes(self) -> int:
        total = 0
        for layer_idx in self._expert_tensors:
            e0 = next(iter(self._expert_tensors[layer_idx]))
            per_expert = self.expert(layer_idx, e0).total_bytes
            total += per_expert * self.num_experts(layer_idx)
        return total

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for index.json on disk)."""
        return {
            "n_layers": self.n_layers,
            "dtype_bytes": self.dtype_bytes,
            "layer_files": {str(k): v for k, v in self._layer_files.items()},
            "shared_file": self._shared_file,
            "tensor_map": {
                k: {"file": v.file, "offset": v.offset, "nbytes": v.nbytes}
                for k, v in self._tensor_map.items()
            },
            "layer_tensors": {
                str(k): v for k, v in self._layer_tensors.items()
            },
            "expert_tensors": {
                str(layer): {
                    str(eid): names
                    for eid, names in exp.items()
                }
                for layer, exp in self._expert_tensors.items()
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "LayoutIndex":
        tensor_map = {
            k: TensorLocation(file=v["file"], offset=v["offset"],
                              nbytes=v["nbytes"])
            for k, v in d["tensor_map"].items()
        }
        layer_tensors = {int(k): v for k, v in d["layer_tensors"].items()}
        expert_raw = d.get("expert_tensors") or {}
        expert_tensors = {
            int(layer): {int(eid): names for eid, names in exp.items()}
            for layer, exp in expert_raw.items()
        }
        return cls(
            layer_files={int(k): v for k, v in d["layer_files"].items()},
            shared_file=d.get("shared_file"),
            tensor_map=tensor_map,
            layer_tensors=layer_tensors,
            expert_tensors=expert_tensors if expert_tensors else None,
            dtype_bytes=d.get("dtype_bytes", 2),
            n_layers=d["n_layers"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "LayoutIndex":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Build a layout from a synthetic manifest (used by tests + conversion tools)
# ---------------------------------------------------------------------------


def _is_expert_tensor(name: str) -> bool:
    return ".experts." in name


def build_layout_from_manifest(
    n_layers: int,
    per_layer_dense_tensors: dict[str, int],   # name -> nbytes
    experts_per_layer: int = 0,
    per_expert_tensors: dict[str, int] | None = None,  # name -> nbytes
    shared_tensors: dict[str, int] | None = None,
    dtype_bytes: int = 2,
    layer_prefix: str = "model.layers.{layer}",
    expert_prefix: str = "model.layers.{layer}.experts.{eid}",
) -> LayoutIndex:
    """Build a LayoutIndex from a description of tensor sizes.

    All tensors for one layer go into one safetensors file, packed
    densely in a fixed order: dense tensors first, then experts
    sequentially. Offset 0 is the start of the *data* section
    (safetensors header is separate — caller handles that).
    """
    per_expert_tensors = per_expert_tensors or {}

    layer_files: dict[int, str] = {}
    tensor_map: dict[str, TensorLocation] = {}
    layer_tensors: dict[int, list[str]] = {}
    expert_tensors: dict[int, dict[int, list[str]]] = {}

    for layer_idx in range(n_layers):
        file_name = f"model.layers.{layer_idx}.safetensors"
        layer_files[layer_idx] = file_name
        offset = 0
        names: list[str] = []
        layer_exp: dict[int, list[str]] = {}

        # Dense tensors first (fixed order)
        for tname, tbytes in per_layer_dense_tensors.items():
            full_name = f"{layer_prefix.format(layer=layer_idx)}.{tname}"
            tensor_map[full_name] = TensorLocation(
                file=file_name, offset=offset, nbytes=tbytes,
            )
            names.append(full_name)
            offset += tbytes

        # Expert tensors (sequential per expert)
        for eid in range(experts_per_layer):
            exp_names: list[str] = []
            for tname, tbytes in per_expert_tensors.items():
                full_name = (
                    f"{expert_prefix.format(layer=layer_idx, eid=eid)}"
                    f".{tname}"
                )
                tensor_map[full_name] = TensorLocation(
                    file=file_name, offset=offset, nbytes=tbytes,
                )
                names.append(full_name)
                exp_names.append(full_name)
                offset += tbytes
            layer_exp[eid] = exp_names

        layer_tensors[layer_idx] = names
        if experts_per_layer > 0:
            expert_tensors[layer_idx] = layer_exp

    # Shared tensors
    shared_file = None
    if shared_tensors:
        shared_file = "model.shared.safetensors"
        offset = 0
        for name, nbytes in shared_tensors.items():
            tensor_map[name] = TensorLocation(
                file=shared_file, offset=offset, nbytes=nbytes,
            )
            offset += nbytes

    return LayoutIndex(
        layer_files=layer_files,
        shared_file=shared_file,
        tensor_map=tensor_map,
        layer_tensors=layer_tensors,
        expert_tensors=expert_tensors if expert_tensors else None,
        dtype_bytes=dtype_bytes,
        n_layers=n_layers,
    )
