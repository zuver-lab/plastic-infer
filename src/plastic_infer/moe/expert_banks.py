"""Per-layer expert bank loading for the offload MoE cache.

Port of FreeToken's bank-loading core. Two halves:

* The streaming assembly machinery (``_PlainBank`` / ``_alloc_expert_bank`` /
  ``_copy_expert_layer_into_bank`` / ``_packed_expert_source_info`` /
  ``stream_moe_expert_sources``) -- verbatim from ``freetoken.models.loader``:
  each packed whole-layer ``[num_experts, ...]`` tensor is written into its own
  per-layer ``HostBank`` (pin/lock-after-fill, honoring the ambient
  :func:`~plastic_infer.moe.host_banks.requested_residency` plan).
* ``load_expert_banks`` -- the PlasticInfer entry point. Where FreeToken reads a
  raw HF checkpoint through a per-arch adapter, PlasticInfer's converted layout
  already stores per-expert ``w1/w2/w3`` tensors per layer, so the reader packs
  them into FreeToken's bf16 bank schema directly:

      gate_up[layer] = [E, 2I, H]   rows [0,I) = w1 (gate), [I,2I) = w3 (up)
      down[layer]    = [E, H, I]    = w2 (down)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
from safetensors import safe_open

from ..utils import init_logger
from .host_banks import HostBank

logger = init_logger(__name__)


@dataclass(frozen=True)
class ExpertBanks:
    """Loaded expert banks, normalized for ``OffloadMoeCache`` wiring."""

    quant_format: str  # _BANK_SCHEMAS key
    # Pinned host banks, keyed by the format's schema: one [num_experts, ...]
    # tensor per layer (independent allocations -> per-layer host attributes).
    sources: dict[str, list[torch.Tensor]]
    # per-expert global scales ([L*E]); None for formats without them (bf16)
    gate_up_alpha: torch.Tensor | None = field(default=None)
    down_alpha: torch.Tensor | None = field(default=None)
    # per-layer HostResidency values actually applied by the loader; None -> all pinned
    layer_residency: list[str] | None = field(default=None)
    # True iff a ``layer_sink`` was engaged (each layer streamed straight to its sink
    # instead of staying materialized here) -- not used by the engine path.
    streamed: bool = False


class _PlainBank:
    """CPU-only fallback bank: a plain unpinned tensor with a no-op pin (no CUDA)."""

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def pin(self) -> None:
        pass


def _alloc_expert_bank(shape: tuple[int, ...], *, dtype: torch.dtype):
    """Allocate an UNPINNED bank (lazy host mmap), to be filled then pinned at the end --
    pin-after-fill. Registering already-resident pages skips cudaHostAlloc's slow commit
    (~2.8 GiB/s) / zero-fill. Returns a bank object exposing ``.tensor`` and ``.pin()``."""
    if torch.cuda.is_available():
        return HostBank(tuple(shape), dtype)
    return _PlainBank(torch.empty(shape, dtype=dtype))


def _copy_expert_layer_into_bank(
    banks: dict[str, list],
    row_shape: dict[str, tuple[int, ...]],
    seen_layers: dict[str, set[int]],
    *,
    bank_name: str,
    tensor: torch.Tensor,
    layer: int,
    num_experts: int,
    dtype: torch.dtype,
) -> None:
    if tensor.size(0) != num_experts:
        raise ValueError(
            f"Unexpected {bank_name} expert count {tensor.size(0)}; "
            f"expected {num_experts}"
        )
    expected_shape = row_shape.setdefault(bank_name, tuple(tensor.shape[1:]))
    if tuple(tensor.shape[1:]) != expected_shape:
        raise ValueError(
            f"Inconsistent {bank_name} expert shape {tuple(tensor.shape[1:])}; "
            f"expected {expected_shape}"
        )

    bank = banks[bank_name][layer]
    if bank is None:
        banks[bank_name][layer] = bank = _alloc_expert_bank(
            (num_experts, *tensor.shape[1:]), dtype=dtype
        )
    bank.tensor.copy_(tensor)  # whole-layer arrival; pinned later, after fully resident
    seen_layers[bank_name].add(layer)


def _packed_expert_source_info(key: str) -> tuple[int, str] | None:
    parts = key.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None
    if parts[-2] != "experts" or parts[-1] not in {"gate_up_proj", "down_proj"}:
        return None
    try:
        return int(parts[2]), parts[-1]
    except ValueError:
        return None


def stream_moe_expert_sources(
    tensors,
    *,
    num_layers: int,
    num_experts: int,
    dtype: torch.dtype,
    layer_sink=None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Stream packed per-layer BF16 expert tensors into final offload banks.

    The reader normalizes experts to ``model.layers.{L}.mlp.experts.gate_up_proj``
    and ``.down_proj`` with shape ``[num_experts, ...]``. Each arrives whole-layer,
    so it's written directly into its own ``[num_experts, ...]`` per-layer bank
    (independent allocation).

    ``layer_sink=None`` (serving): pin each layer's banks as its writes complete,
    via an internally-owned :class:`PinPipeline`. ``layer_sink`` given (converter):
    the tracker fires into it instead -- nothing is pinned, and the sink may release
    banks it has written out, so the returned tensors are only valid until then.
    """
    from .host_banks import LayerCompletionTracker, PinPipeline

    banks: dict[str, list] = {  # name -> per-layer [bank obj (HostBank/_PlainBank) or None]
        "gate_up": [None] * num_layers,
        "down": [None] * num_layers,
    }
    row_shape: dict[str, tuple[int, ...]] = {}
    seen_layers: dict[str, set[int]] = {"gate_up": set(), "down": set()}

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, banks, sink)  # gate_up + down per layer
        for name, tensor in tensors:
            expert_info = _packed_expert_source_info(name)
            if expert_info is None:
                raise ValueError(f"Unexpected expert weight key: {name}")
            layer, packed_name = expert_info
            bank_name = "gate_up" if packed_name == "gate_up_proj" else "down"
            _copy_expert_layer_into_bank(
                banks,
                row_shape,
                seen_layers,
                bank_name=bank_name,
                tensor=tensor,
                layer=layer,
                num_experts=num_experts,
                dtype=dtype,
            )
            tracker.note(layer)

        expected_layers = set(range(num_layers))
        missing = {
            name: sorted(expected_layers - seen)
            for name, seen in seen_layers.items()
            if seen != expected_layers
        }
        if missing:
            raise ValueError(f"Missing MoE expert source layers: {missing}")

    if layer_sink is not None:
        _load(layer_sink)
    else:
        with PinPipeline() as pins:
            _load(pins)
    return (
        [bank.tensor for bank in banks["gate_up"]],
        [bank.tensor for bank in banks["down"]],
    )


# ---------------------------------------------------------------------------
# PlasticInfer disk layout -> packed whole-layer tensors (bf16)
# ---------------------------------------------------------------------------


def _iter_packed_layers(model_dir, layout, dtype: torch.dtype):
    """Yield ``(name, whole_layer_tensor)`` pairs for every layer's experts.

    Reads one converted ``model.layers.{L}.safetensors`` per layer (mmap'd), packs
    the 128 per-expert w1/w2/w3 tensors into FreeToken's bf16 bank schema, and
    yields ``gate_up_proj`` then ``down_proj`` for that layer. Peak host memory is
    ~2x one layer's experts (the read tensors + the packed banks), sequential.
    """
    n_layers = layout.n_layers
    for layer in range(n_layers):
        file_name = layout.layer_file(layer)
        names_by_eid = [
            layout.expert_tensor_names(layer, eid)
            for eid in range(layout.num_experts(layer))
        ]
        w1s, w2s, w3s = [], [], []
        with safe_open(str(os.path.join(model_dir, file_name)), framework="pt") as f:
            for (w1n, w2n, w3n) in names_by_eid:
                w1s.append(f.get_tensor(w1n).to(dtype))
                w2s.append(f.get_tensor(w2n).to(dtype))
                w3s.append(f.get_tensor(w3n).to(dtype))
        # [E, 2I, H]: cat of the stacked gates and ups along the I axis
        gate_up = torch.cat([torch.stack(w1s, dim=0), torch.stack(w3s, dim=0)], dim=1)
        down = torch.stack(w2s, dim=0)  # [E, H, I]
        yield f"model.layers.{layer}.mlp.experts.gate_up_proj", gate_up
        yield f"model.layers.{layer}.mlp.experts.down_proj", down


def _dummy_expert_sources(num_layers: int, num_experts: int, inter: int,
                          hidden: int, dtype: torch.dtype):
    """Random banks matching the bf16 schema (no disk read; tests / dry runs)."""
    gen = torch.Generator().manual_seed(0)
    gate_up = [torch.randn(num_experts, 2 * inter, hidden, generator=gen, dtype=dtype)
               for _ in range(num_layers)]
    down = [torch.randn(num_experts, hidden, inter, generator=gen, dtype=dtype)
            for _ in range(num_layers)]
    return gate_up, down


def _echo_residency(banks: ExpertBanks, requested: list[str] | None) -> ExpertBanks:
    """Stamp an honored residency request onto the ExpertBanks (the settle point --
    PinPipeline via requested_residency -- recorded the achieved labels)."""
    from .host_banks import requested_residency

    if requested is None or banks.layer_residency is not None:
        return banks
    return banks  # requested_residency's plan was ambient; labels already honored


def load_expert_banks(
    model_dir: str,
    layout,
    *,
    dtype: torch.dtype,
    dummy: bool = False,
    layer_residency: list[str] | None = None,
    layer_sink=None,
) -> ExpertBanks:
    """Load (or fabricate, with ``dummy=True``) the bf16 expert banks.

    ``layer_residency``: per-layer ``HostResidency`` labels applied at settle time
    (pin vs OS-lock). The load runs under :func:`requested_residency`, so every
    layer's ``HostBank`` settles to its label. Echoes the requested labels on the
    returned ``ExpertBanks`` (a failed lock downgrades a layer to PAGEABLE, which
    ``OffloadMoeCache`` treats the same as LOCKED).
    """
    num_layers = layout.n_layers
    num_experts = layout.num_experts(0)
    # inter / hidden from the first expert's shapes (w1 is [inter, hidden])
    first = layout.expert_tensor_names(0, 0)
    with safe_open(os.path.join(model_dir, layout.layer_file(0)), framework="pt") as f:
        w1 = f.get_tensor(first[0])
    inter, hidden = int(w1.shape[0]), int(w1.shape[1])

    from .host_banks import requested_residency

    with requested_residency(layer_residency):
        if dummy:
            gate_up_source, down_source = _dummy_expert_sources(
                num_layers, num_experts, inter, hidden, dtype
            )
            # dummy is materialize-only (fabricated in one shot); no sink/settle needed
            banks = ExpertBanks(
                "bf16", {"gate_up": gate_up_source, "down": down_source},
                layer_residency=layer_residency,
            )
            return banks
        sink = None if layer_sink is None else layer_sink
        tensors = _iter_packed_layers(model_dir, layout, dtype)
        gate_up_source, down_source = stream_moe_expert_sources(
            tensors,
            num_layers=num_layers,
            num_experts=num_experts,
            dtype=dtype,
            layer_sink=sink,
        )
    return ExpertBanks(
        "bf16", {"gate_up": gate_up_source, "down": down_source},
        layer_residency=layer_residency,
        streamed=sink is not None,
    )
