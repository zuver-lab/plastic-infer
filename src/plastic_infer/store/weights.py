"""Dense weight residency: a budget-bounded rotating window (M3).

Implements the sequential-dense data class's resident window (§3.1 /
§5.2 of DESIGN.md): for a disk or host source, only a *window* of
layers is resident in GPU at once, and the runner decides which layer
is current (rotating residency, no LRU competition — unlike experts,
there is no working set to exploit, so the window is a FIFO ring the
runner walks in order).

`ensure(l)` loads layer `l` from the source if absent (evicting to
fit) and pins it for compute; `release(l)` unpins. Same D7 invariant
as the expert slot pool: eviction never reclaims a pinned layer.

Correctness is independent of window size (D5): with W=1 the runner
streams every layer through one slot and produces identical logits;
with W=n_layers everything is resident and nothing reloads.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .resident import LruPool
from .tiers import Tier


class LayerSource(Protocol):
    """Something that can serve one layer's tensors on demand.

    Both disk (DiskLayerSource) and host (DictLayerSource) sources
    implement this; the window is agnostic to the tier.
    """

    def layer(self, layer_idx: int) -> dict[str, torch.Tensor]: ...


class DenseWindow:
    """Budget-bounded resident window of dense layers over a source.

    `ensure(layer_idx)` returns the layer's weight dict on the pool
    device, loading it from the source on a miss. Caller must call
    `release(layer_idx)` after compute. Window size is set by the
    byte budget; the runner controls which layer is resident.
    """

    def __init__(self, source: LayerSource, budget_bytes: int,
                 device: torch.device | None = None) -> None:
        self.source = source
        self.device = device or torch.device("cpu")
        self.pool = LruPool(budget_bytes)
        self._tensors: dict[int, dict[str, torch.Tensor]] = {}
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    def ensure(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Load (or confirm) one layer and pin it. Returns its tensors.

        On a miss the layer is read from the source and copied to the
        pool device; a too-small budget is caught here rather than
        silently overflowing the pool (no OOM by design).
        """
        if layer_idx not in self.pool:
            src = self.source.layer(layer_idx)
            nbytes = sum(t.numel() * t.element_size() for t in src.values())
            for page in self.pool.evict_to_fit(nbytes):
                del self._tensors[page.key]
            assert self.pool.can_fit(nbytes), (
                f"dense window budget too small for layer {layer_idx}: "
                f"{nbytes}B > {self.pool.budget_bytes}B budget")
            self.pool.add(layer_idx, nbytes, tier=Tier.GPU)
            self._tensors[layer_idx] = {
                k: t.to(self.device) for k, t in src.items()
            }
            self.misses += 1
        else:
            self.hits += 1
        self.pool.pin(layer_idx)
        return self._tensors[layer_idx]

    def release(self, layer_idx: int) -> None:
        """Unpin a layer after compute. Double-release raises."""
        self.pool.unpin(layer_idx)

    @property
    def miss_rate(self) -> float:
        total = self.hits + self.misses
        return self.misses / total if total else 0.0
