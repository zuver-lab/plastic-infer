"""Disk-backed weight sources (M3).

M3 completes the tiering story for *dense* weights: the planner can now
decide that a model's sequential layers stream from disk (seq_weight_source
= DISK/HOST_FIRST) instead of living fully in host RAM. This module is the
disk tier of the loading path (§5.3 of DESIGN.md):

  - One safetensors file = one streaming unit (per layer).
  - DiskLayerSource opens only the requested layer's file and reads just
    that layer's tensors — the "sequential read of one layer" that keeps
    disk bandwidth proportional to what is actually resident.
  - DictLayerSource serves the same interface from an in-memory dict, which
    is the HOST source and the test anchor (proves the streaming *runner*,
    not the I/O, is what changes).

Both sources are lazy: nothing is loaded until `layer(l)` is called.
"""

from __future__ import annotations

from pathlib import Path

from safetensors import safe_open
import torch

from .layout import LayoutIndex


class DiskLayerSource:
    """Serves layer weights from per-layer safetensors files on disk.

    The file's tensor names are the layout's canonical names, so a layer
    file written by a converter (or a test) is read back verbatim.
    """

    def __init__(self, model_dir: str | Path, layout: LayoutIndex) -> None:
        self.dir = Path(model_dir)
        self.layout = layout

    def layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Read one layer's tensors (a sequential read of one file)."""
        file_name = self.layout.layer_file(layer_idx)
        names = self.layout.layer_tensor_names(layer_idx)
        with safe_open(str(self.dir / file_name), framework="pt") as f:
            return {n: f.get_tensor(n) for n in names}

    def shared(self) -> dict[str, torch.Tensor]:
        """Read the non-layer tensors (embed/norm/lm_head) once."""
        file_name = self.layout.shared_file()
        if file_name is None:
            return {}
        names = self.layout.shared_tensor_names()
        with safe_open(str(self.dir / file_name), framework="pt") as f:
            return {n: f.get_tensor(n) for n in names}


class DictLayerSource:
    """Serves layers from an in-memory flat dict (HOST source / test anchor)."""

    def __init__(self, weights: dict[str, torch.Tensor], n_layers: int) -> None:
        self.weights = weights
        self.n_layers = n_layers

    def layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        prefix = f"layers.{layer_idx}."
        return {k: v for k, v in self.weights.items() if k.startswith(prefix)}

    def shared(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.weights.items()
                if not k.startswith("layers.")}
