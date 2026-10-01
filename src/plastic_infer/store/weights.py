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

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol

import torch

from .resident import LruPool
from .tiers import Tier


def _strip_layer_prefix(key: str) -> str:
    """'layers.7.self_attn.q_proj.weight' -> 'self_attn.q_proj.weight'."""
    return key.split(".", 2)[2]


def _load_gilcopy():
    """The GIL-releasing memcpy used by the prefetch worker.

    `copy_`/`pin_memory` run under the GIL, and the runner's Python loop
    holds it nearly continuously, so a worker thread doing a plain torch
    copy can starve for the whole layer's compute window. This tiny C
    module (memcpy wrapped in Py_BEGIN_ALLOW_THREADS) stages a layer into
    its pinned slot without holding the GIL, so the transfer overlaps the
    previous layer's GPU work. Returns None when unavailable (fall back to
    a normal `copy_`).
    """
    try:
        import gilcopy  # type: ignore
        return gilcopy
    except ImportError:
        return None


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


class StreamingDenseWeights:
    """DenseWeights-compatible layer window with background prefetch.

    `shared` tensors (embed_tokens / norm / lm_head) stay resident on the
    device; per-layer tensors stream host -> device through a window of
    two layers (the current one plus the next, prefetched). A single
    background thread copies the next layer on a dedicated CUDA stream
    while the current layer is being computed, hiding the H2D copy behind
    the layer's math (DESIGN.md D4/D6). Prefetch wraps around
    ((L+1) % n_layers) so a decode step that restarts at layer 0 finds its
    weights already staged when the previous step ended at layer 47.

    The runner sees exactly DenseWeights' surface — `__getitem__`,
    `.device`, `.dtype` — so exec/ never changes. Correctness is
    independent of window size (D5): W=1 streams one slot with prefetch
    off, W=n_layers degrades to a synchronous load per layer. On CPU every
    lookup is a plain dict access and prefetch is a no-op.

    Slot lifecycle (main thread only touches `_gpu`; the worker returns
    the copied dict through the Future, so no locking is needed):
      advance(L):  consume the in-flight prefetch if it targets L and make
                   the compute stream wait for the copy; else load L
                   synchronously on a miss; trim to {L}; submit prefetch
                   of (L+1) % n_layers.
    The copy waits (via a start event) only for work submitted *before*
    the prefetch was requested — the previous layers' compute — so it runs
    concurrent with L's compute, not after it. Freed slots are reclaimed by
    the caching allocator with stream semantics, so no kernel ever reads a
    reused block.
    """

    def __init__(self, host_layers: dict[int, dict[str, torch.Tensor]], *,
                 shared: dict[str, torch.Tensor], n_layers: int,
                 device: torch.device, dtype: torch.dtype,
                 window: int = 2, prefetch: bool = True) -> None:
        self.device = device
        self.dtype = dtype
        self.n_layers = int(n_layers)
        self._window = max(1, min(int(window), self.n_layers))
        self._prefetch = bool(prefetch) and self._window >= 2
        self._shared = {k: t.to(device=device, dtype=dtype)
                        for k, t in shared.items()}
        self._gpu: dict[int, dict[str, torch.Tensor]] = {}
        self._current = -1
        self._future: tuple[int, Future] | None = None
        self._executor = None
        self._copy_stream = None
        self._dbg_wait_future = 0.0   # TEMP diagnostics
        self._dbg_n = 0
        self._dbg_copy_s = 0.0
        self._dbg_copy_n = 0
        self._dbg_stall = 0.0  # TEMP: slot-reuse DMA wait
        self._dbg_cpys = 0.0   # TEMP: staging copy wall
        self._dbg_cpy_times = []  # TEMP: per-copy wall, ms
        self._dbg_tos = 0.0    # TEMP: DMA issue wall
        self._dbg_inflight = 0 # TEMP: DMA not done when consumed
        # Host tensors stay unpinned: pinning the whole dense footprint at
        # once exceeds the host's pinned-memory allowance (WSL2 caps it
        # near 1 GiB). Each layer is staged into a *rotating* pinned slot
        # right before its DMA, so at most a layer's worth of pinned RAM is
        # live at any time.
        self._host = dict(host_layers)
        if device.type == "cuda":
            # Contiguous flat copies of each layer (built once) so the
            # worker can stage the whole layer with a single memcpy and
            # issue a single contiguous DMA — 9 per-tensor transfers measured
            # ~2x the wall of one flat transfer, and one GIL-released memcpy
            # beats nine GIL-bound ones.
            self._host_flat: dict[int, torch.Tensor] = {}
            self._meta: dict[int, list[tuple[str, int, tuple]] ] = {}
            for l, d in self._host.items():
                keys = sorted(d)
                flat = torch.cat([d[k].flatten() for k in keys]).to(self.dtype)
                off = 0
                meta = []
                for k in keys:
                    t = d[k]
                    meta.append((k, off, t.shape))
                    off += t.numel()
                self._host_flat[l] = flat
                self._meta[l] = meta
            nbytes0 = self._host_flat[0].numel() * \
                self._host_flat[0].element_size()
            assert all(f.numel() * f.element_size() == nbytes0
                       for f in self._host_flat.values()), \
                "streaming dense window assumes uniform layer sizes"
            self._nbytes = nbytes0
            self._gilcopy = _load_gilcopy()
            # Two rotating slots: pinned staging + device buffer, so DMA into
            # slot i never overwrites weights the compute stream is still
            # reading (they live in the other device buffer two layers back).
            self._pin_slots = [
                torch.empty_like(self._host_flat[0], pin_memory=True)
                for _ in range(2)
            ]
            self._gpu_bufs = [
                torch.empty_like(self._host_flat[0], device=device)
                for _ in range(2)
            ]
            self._pin_done: list[torch.cuda.Event | None] = [None, None]
            self._pin_slot = 0
            if self._prefetch:
                self._executor = ThreadPoolExecutor(max_workers=1)
                self._copy_stream = torch.cuda.Stream()

    # ------------------------------------------------------------------
    # Window advancement
    # ------------------------------------------------------------------

    def _copy_to_device(self, layer: int) -> tuple[
            dict[str, torch.Tensor], torch.cuda.Event | None]:
        """Stage one layer into the next free pinned slot and issue the DMA.

        Returns `(gpu, done_ev)`. The whole layer is staged with a single
        contiguous memcpy from the pre-flattened host copy — GIL-released
        when `gilcopy` is available so the worker's copy overlaps the
        compute stream's Python dispatch — then one contiguous async H2D
        into the persistent device buffer. `done_ev` fires when the DMA
        completes; the prefetch worker hands it back so the compute stream
        can wait for the copy, while the miss path (same stream as the
        following kernels) needs no wait.
        """
        if self.device.type != "cuda":
            return {k: t.to(self.dtype) for k, t in self._host[layer].items()}, None
        slot = self._pin_slots[self._pin_slot]
        buf = self._gpu_bufs[self._pin_slot]
        done_ev = self._pin_done[self._pin_slot]
        if done_ev is not None and not done_ev.query():
            import time as _t
            _a = _t.perf_counter()
            done_ev.synchronize()  # rare: this slot's DMA still in flight
            self._dbg_stall += _t.perf_counter() - _a
        src = self._host_flat[layer]
        import time as _t
        _a = _t.perf_counter()
        if self._gilcopy is not None:
            self._gilcopy.memcpy_async(slot.data_ptr(), src.data_ptr(),
                                       self._nbytes)
        else:
            slot.copy_(src)
        self._dbg_cpys += (_dt := _t.perf_counter() - _a)
        self._dbg_cpy_times.append(_dt * 1e3)
        _a = _t.perf_counter()
        buf.copy_(slot, non_blocking=True)
        self._dbg_tos += _t.perf_counter() - _a
        gpu = {}
        for k, off, shape in self._meta[layer]:
            n = 1
            for s in shape:
                n *= s
            gpu[k] = buf.narrow(0, off, n).view(shape)
        done_ev = torch.cuda.Event()
        done_ev.record(torch.cuda.current_stream())
        self._pin_done[self._pin_slot] = done_ev
        self._pin_slot ^= 1
        return gpu, done_ev

    def __getitem__(self, key: str) -> torch.Tensor:
        if key.startswith("layers."):
            layer = int(key.split(".")[1])
            if layer != self._current:
                self._advance(layer)
            return self._gpu[layer][key]
        return self._shared[key]

    # ------------------------------------------------------------------
    # Window advancement
    # ------------------------------------------------------------------

    def _advance(self, layer: int) -> None:
        if self._future is not None:
            target, fut = self._future
            self._future = None
            if target == layer:
                import time as _t
                _a = _t.perf_counter()
                gpu, done_ev = fut.result()
                self._dbg_wait_future += _t.perf_counter() - _a
                self._gpu[layer] = gpu
                if done_ev is not None:
                    if not done_ev.query():
                        self._dbg_inflight += 1
                    torch.cuda.current_stream().wait_event(done_ev)
                self._dbg_n += 1
                self._current = layer
                self._trim(layer)
                self._prefetch_ahead(layer)
                return
            fut.cancel()          # stale: leftover prefetch from a prior pass
        if layer not in self._gpu:
            # miss: load on the compute stream (W=1, prefetch disabled, or
            # the first layer of a run).
            gpu, _ = self._copy_to_device(layer)
            self._gpu[layer] = gpu
        self._current = layer
        self._trim(layer)
        self._prefetch_ahead(layer)

    def _trim(self, layer: int) -> None:
        for stale in [k for k in self._gpu if k != layer]:
            del self._gpu[stale]

    def _prefetch_ahead(self, layer: int) -> None:
        if self._executor is None:
            return
        nxt = (layer + 1) % self.n_layers
        if nxt == layer:
            return
        start_ev = torch.cuda.Event()
        start_ev.record(torch.cuda.current_stream())
        self._future = (nxt,
                        self._executor.submit(self._copy_layer, nxt, start_ev))

    def _copy_layer(self, layer: int, start_ev):
        import time as _t
        _a = _t.perf_counter()
        with torch.cuda.stream(self._copy_stream):
            # Wait only for work submitted before this prefetch was
            # requested (previous layers), not the current layer's — the
            # copy then overlaps the current layer's compute.
            self._copy_stream.wait_event(start_ev)
            gpu, done_ev = self._copy_to_device(layer)
        self._dbg_copy_s += _t.perf_counter() - _a
        self._dbg_copy_n += 1
        return gpu, done_ev
