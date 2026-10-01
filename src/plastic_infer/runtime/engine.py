"""Engine: plan -> pools -> runner for a converted model directory.

Loads a directory produced by weights.convert (layout.index.json +
per-layer safetensors + config.json), plans each request against a
DeviceProfile, builds the three-tier pools, and runs prefill + decode
through the composition runner (runner_moe_paged).

For Qwen3-30B-A3B the plan comes out as: dense weights fully in host
RAM and copied to GPU (HOST source, W = n_layers), experts served by
the FreeToken offload-MoE cache (host banks -> GPU slot cache, decode
on demand; CPU executor for layers the WSL pin budget OS-locks), KV
paged on GPU. Long-context KV beyond the GPU pool (HOST_STREAM/REJECT)
is future work and is rejected clearly.
"""

from __future__ import annotations

import ctypes
import functools
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..exec.runner import DenseWeights, ModelConfig
from ..exec.runner_moe_paged import (
    decode_step_moe_paged,
    make_kv_config,
    prefill_forward_moe_paged,
)
from ..kv.kv_store import KVStore
from ..moe.expert_banks import load_expert_banks
from ..moe.offload_cache import OffloadMoeCache
from ..planning.budget import MemoryPlanner, ModelMeta, Plan, RequestMeta
from ..planning.profile import DeviceProfile
from ..planning.wiring import pool_budgets
from ..store.weights import StreamingDenseWeights
from ..weights.disk import DiskLayerSource
from ..weights.layout import LayoutIndex

logger = logging.getLogger(__name__)


def build_model_meta(layout: LayoutIndex, cfg: dict) -> ModelMeta:
    """ModelMeta from the layout's measured sizes + config shapes."""
    n_layers = layout.n_layers
    dtype_bytes = layout.dtype_bytes
    head_dim = cfg.get("head_dim") or (
        cfg["hidden_size"] // cfg["num_attention_heads"])
    return ModelMeta(
        n_layers=n_layers,
        dense_bytes=layout.total_dense_bytes,
        per_layer_dense=tuple(
            layout.dense_per_layer_bytes(l) for l in range(n_layers)),
        experts_per_layer=tuple(
            layout.num_experts(l) for l in range(n_layers)),
        expert_row_bytes=tuple(
            layout.expert_total_bytes(l) for l in range(n_layers)),
        kv_bytes_per_token=(
            n_layers * 2 * cfg["num_key_value_heads"] * head_dim * dtype_bytes),
    )


def build_runner_config(cfg: dict, *, dtype: torch.dtype,
                        qk_norm: bool) -> ModelConfig:
    """Runner ModelConfig from an HF config dict (head_dim is explicit
    in Qwen3, not derived; rope_theta may live top-level or in
    rope_parameters)."""
    head_dim = cfg.get("head_dim") or (
        cfg["hidden_size"] // cfg["num_attention_heads"])
    rope_base = cfg.get("rope_theta")
    if rope_base is None:
        rope_base = cfg.get("rope_parameters", {}).get("rope_theta", 10000.0)
    return ModelConfig(
        n_layers=cfg["num_hidden_layers"],
        n_heads=cfg["num_attention_heads"],
        n_kv_heads=cfg["num_key_value_heads"],
        head_dim=head_dim,
        hidden_dim=cfg["hidden_size"],
        intermediate_dim=cfg.get("moe_intermediate_size",
                                 cfg["intermediate_size"]),
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_position_embeddings"],
        rope_base=float(rope_base),
        dtype=dtype,
        qk_norm=qk_norm,
        n_experts=cfg.get("num_experts", 0),
        n_experts_per_tok=cfg.get("num_experts_per_tok", 0),
    )


# Expert activations the CPU MoE executor supports (csrc ActKind).
_CPU_MOE_ACTS = (
    "silu", "swish", "gelu", "gelu_tanh", "gelu_pytorch_tanh", "swigluoai",
)


def _cpu_moe_executor_viable() -> bool:
    """Whether an automatic CPU-decode decision may target the CPU MoE executor.

    A default boot must degrade to GPU offload instead of crashing in the executor
    after the whole load (explicit cpu/hybrid picks still fail loudly).
    """
    try:
        from ..kernel import _cpu_moe  # noqa: F401
    except ImportError:
        return False
    if "silu" not in _CPU_MOE_ACTS:  # Qwen3 hidden_act
        return False
    try:
        from ..moe.cpu_executor import _WFMT_IDS  # noqa: F401
    except ImportError:
        return False
    return "bf16" in _WFMT_IDS  # bf16 expert banks


@functools.lru_cache(maxsize=1)
def _probe_wsl_pin_budget() -> int | None:
    """Measured cumulative CUDA pin ceiling; None if unavailable."""
    try:
        if not torch.cuda.is_available():
            return None
        torch.zeros(1, device="cuda")  # warm the CUDA context
        rt = ctypes.CDLL(f"libcudart.so.{(torch.version.cuda or '0').split('.')[0]}")
        for fn in ("cudaHostAlloc", "cudaFreeHost", "cudaGetLastError"):
            getattr(rt, fn).restype = ctypes.c_int
        rt.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.c_size_t, ctypes.c_uint]
        rt.cudaFreeHost.argtypes = [ctypes.c_void_p]
    except Exception:
        return None
    chunk = 256 << 20  # small chunks get the most usable budget
    max_probe = 8 << 30  # bound probe time/RAM on uncapped hosts
    held: list[ctypes.c_void_p] = []
    total = 0
    try:
        while total + chunk <= max_probe:
            ptr = ctypes.c_void_p()
            if rt.cudaHostAlloc(ctypes.byref(ptr), chunk, 0) != 0 or not ptr.value:
                break  # hit the pin wall
            ctypes.memset(ptr, 0, 1 << 20)  # fault pages, matching pin-after-fill banks
            held.append(ptr)
            total += chunk
    finally:
        for ptr in held:  # free the probe buffers so the real banks get the budget
            rt.cudaFreeHost(ptr)
    rt.cudaGetLastError()  # clear sticky error from the refused alloc or torch OOMs
    return int(total * 0.8) if total else None


def _pin_budget_bytes() -> int | None:
    """Bytes safe to cudaHostRegister, or None when the platform does not cap
    pinning (plain Linux). WSL/WDDM caps near ~1 GiB; FREETOKEN_PIN_BUDGET_GB
    overrides anywhere."""
    if env := os.environ.get("FREETOKEN_PIN_BUDGET_GB"):
        return int(float(env) * 2**30)
    if not hasattr(os, "uname") or "microsoft" not in os.uname().release.lower():
        return None
    return _probe_wsl_pin_budget()


def _resolve_cpu_layers() -> frozenset[int]:
    """MoE layer ids whose decode runs entirely on the CPU executor (offload
    path). Under hybrid every layer is hybrid-eligible and pageable fetch serves
    the unpinned ones, so this set stays empty there. No CLI knob requests an
    explicit set today -- the auto resolution in the engine is the mechanism."""
    return frozenset()


def _auto_cpu_layers(num_moe_layers: int, bank_bytes: int) -> frozenset[int]:
    """Pick CPU (locked) MoE layers automatically when the banks exceed the pin
    budget. Locks just enough head+tail layers: per-layer decode miss rates are
    U-shaped, so the ends are the cheapest to move off the slot cache."""
    budget = _pin_budget_bytes()
    if budget is None or not bank_bytes or bank_bytes <= budget:
        return frozenset()
    if not _cpu_moe_executor_viable():
        logger.info(
            "moe-cpu-layers auto: banks %.2f GiB exceed the pin budget %.2f GiB, "
            "but the CPU MoE executor cannot serve this model; keeping every layer "
            "pinned on the GPU offload path",
            bank_bytes / 2**30, budget / 2**30)
        return frozenset()
    n = min(num_moe_layers, math.ceil(num_moe_layers * (1 - budget / bank_bytes)))
    head = (n + 1) // 2
    ids = frozenset(range(head)) | frozenset(
        range(num_moe_layers - (n - head), num_moe_layers))
    logger.info(
        "moe-cpu-layers auto: banks %.2f GiB > pin budget %.2f GiB; locking %d "
        "head+tail MoE layers for CPU decode (%s)",
        bank_bytes / 2**30, budget / 2**30, n, sorted(ids))
    return ids


@dataclass
class RunResult:
    tokens: list[int]
    plan: Plan
    metrics: dict


class Engine:
    def __init__(self, model_dir: str | Path, profile: DeviceProfile,
                 device: torch.device | None = None) -> None:
        self.dir = Path(model_dir)
        self.profile = profile
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.layout = LayoutIndex.load(self.dir / "layout.index.json")
        with open(self.dir / "config.json") as f:
            self.cfg = json.load(f)
        self.dtype = torch.bfloat16 if self.cfg.get("torch_dtype") == "bfloat16" \
            else torch.float32
        self.model = build_model_meta(self.layout, self.cfg)
        qk_norm = any("self_attn.q_norm.weight" in n
                      for n in self.layout.layer_tensor_names(0))
        self.config = build_runner_config(self.cfg, dtype=self.dtype,
                                          qk_norm=qk_norm)
        self.dense_src = DiskLayerSource(self.dir, self.layout)
        # FreeToken offload-MoE cache (banks + slot cache + executor), built once
        # on the first request (loading all expert banks is the expensive part).
        self._moe_cache: OffloadMoeCache | None = None
        # Persistent CPU MoE executor (decode-time expert compute) for the CPU-
        # locked / hybrid layers, built alongside the cache; None for plain GPU
        # offload. Per-step raise_if_unhealthy() mirrors FreeToken's forward_batch.
        self.cpu_moe_executor = None

    def _load_dense(self) -> DenseWeights:
        """Load all dense + shared weights onto the run device."""
        flat = dict(self.dense_src.shared())
        for l in range(self.layout.n_layers):
            flat.update(self.dense_src.layer(l))
        return DenseWeights({k: v.to(self.device) for k, v in flat.items()})

    def _load_host_dense(self) -> dict[str, dict]:
        """Read shared + every dense layer into host RAM (no GPU copy).

        The dense source reads whole files (safetensors get_tensor), so
        the full 3.5GB lands in host RAM once; StreamingDenseWeights then
        pins and streams it layer by layer, keeping the window resident.
        """
        return {
            "shared": dict(self.dense_src.shared()),
            "layers": {l: self.dense_src.layer(l)
                       for l in range(self.layout.n_layers)},
        }

    def _build_moe_cache(self, expert_slots_bytes: int) -> OffloadMoeCache:
        """Build (once) the FreeToken offload-MoE cache for this model.

        Wires banks -> slot cache exactly as FreeToken's engine does: resolve the
        backend (hybrid by default) and the CPU-locked layer set (offload
        auto-lock, when the pinned banks exceed the WSL pin budget), load all
        expert banks into host with the matching residency, then construct the
        cache with ``cpu_layer_ids`` set BEFORE ``set_bank_sources`` (the residency
        validation and the copy plan's skip of non-pinned layers key on that set).
        ``decode_target`` picks the per-decode mechanism: "hybrid" is GPU slot-cache
        + CPU-overflow co-compute (unpinned layers fetch via ``copy_missing``'s
        pageable gather), "cpu" routes the locked layers to the CPU executor, "gpu"
        is plain GPU offload.
        """
        num_moe_layers = self.config.n_layers
        # FreeToken sizes this as layers * experts * per-expert
        # (bank_bytes_estimate); expert_total_bytes is one expert's rows.
        bank_bytes = sum(self.layout.expert_total_bytes(l) *
                         self.layout.num_experts(l)
                         for l in range(num_moe_layers))
        # backend (FreeToken --moe-backend, default hybrid): the CPU executor
        # computes each step's overflow misses while the GPU computes the cache
        # hits + the bandwidth-matched fetched share. FREETOKEN_MOE_BACKEND=offload
        # restores plain GPU offload; =auto applies FreeToken's hardware gate
        # (offload -> hybrid only when the CPU MoE bw clears the PCIe gather bw
        # by the bench threshold).
        backend = os.environ.get("FREETOKEN_MOE_BACKEND", "hybrid").strip().lower()
        if backend == "hybrid":
            hybrid = True
        elif backend == "auto":
            from ..moe.bandwidth import probe
            p = probe(self.device, self.config.n_experts, self.config.hidden_dim,
                      self.config.intermediate_dim, self.config.n_experts_per_tok)
            hybrid = bool(p and p["recommended"] == "hybrid")
        else:  # "offload" (or anything unknown)
            hybrid = False
        # a default hybrid boot degrades to GPU offload when the CPU executor cannot
        # serve this model (an explicit FREETOKEN_MOE_BACKEND=hybrid fails loudly in
        # the executor build instead).
        if hybrid and not _cpu_moe_executor_viable():
            logger.info(
                "moe backend hybrid defaulted but the CPU MoE executor cannot serve "
                "this model; using plain GPU offload")
            hybrid = False
        cpu_layer_ids = _resolve_cpu_layers()
        if hybrid:
            decode_target = "hybrid"
        else:
            # offload semantics (FreeToken): CPU-locked layers when the banks
            # exceed the pin budget; is_cpu_layer wins over "gpu" at decode.
            if not cpu_layer_ids and _pin_budget_bytes() is not None:
                cpu_layer_ids = _auto_cpu_layers(num_moe_layers, bank_bytes)
            decode_target = "cpu" if cpu_layer_ids else "gpu"
        # residency: which banks get a device address. Offload pins every bank and
        # OS-locks the CPU layers (resident for the executor, unregistered). Hybrid
        # pins the layers whose full banks fit the pin budget (device-side fused
        # fetch); the rest stay LOCKED/PAGEABLE but hybrid-eligible -- their decode
        # fetch goes through copy_missing's pageable gather, whose pinned staging is
        # bounded by the per-step fetch count, not the bank size. A capped pin budget
        # therefore no longer forces every layer onto the CPU executor (FreeToken's
        # WSL all-locked outcome); hybrid keeps the GPU slot cache in play per layer.
        budget = _pin_budget_bytes()
        split_residency = False
        requested_residency = None
        if hybrid:
            if budget is not None and bank_bytes > budget:
                pinnable = int(budget // (bank_bytes / num_moe_layers))
                from ..moe.host_banks import HostResidency
                requested_residency = [
                    HostResidency.PINNED.value if i < pinnable
                    else HostResidency.LOCKED.value
                    for i in range(num_moe_layers)
                ]
                split_residency = True
        else:
            split_residency = bool(cpu_layer_ids) and budget is not None
            if split_residency:
                from ..moe.host_banks import HostResidency
                requested_residency = [
                    HostResidency.LOCKED.value if i in cpu_layer_ids
                    else HostResidency.PINNED.value
                    for i in range(num_moe_layers)
                ]
        banks = load_expert_banks(
            self.dir, self.layout, dtype=self.dtype,
            layer_residency=requested_residency)
        per_expert = self.layout.expert_total_bytes(0)
        cache_size = max(self.config.n_experts,
                         expert_slots_bytes // per_expert)
        # overlap DMAs from registered banks, and locked layers cannot feed it;
        # FreeToken's solver also self-disables it below 2*num_experts slots
        # (cache_budget.plan_cache_budget), so mirror that guard here.
        prefill_overlap = not split_residency and \
            cache_size >= 2 * self.config.n_experts
        cache = OffloadMoeCache(
            num_layers=num_moe_layers,
            num_experts=self.config.n_experts,
            cache_size=cache_size,
            device=self.device,
            prefill_overlap=prefill_overlap,
            prefill_hit_d2d=False,  # FreeToken default; the batch-memcpy probe gates it
            quant_format=banks.quant_format,
            decode_target=decode_target,
        )
        # before set_bank_sources: the residency validation and the copy plan's
        # skip of non-pinned layers key on the CPU-layer set.
        cache.cpu_layer_ids = cpu_layer_ids
        cache.set_bank_sources(banks.sources,
                               layer_residency=banks.layer_residency)
        cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
        cache.collect_stats = True
        if decode_target == "hybrid":
            self._resolve_hybrid_fetch(cache)
        if decode_target in ("cpu", "hybrid"):
            self._init_cpu_moe_executor(cache)
        logger.info(
            "MoE offload cache: decode_target=%s, %d CPU-locked layer(s), "
            "cache_size=%d (%.2f GiB), residency=%s",
            decode_target, len(cpu_layer_ids), cache.cache_size,
            cache.cache_size * per_expert / 2**30,
            "split" if split_residency else "pinned")
        return cache

    def _resolve_hybrid_fetch(self, cache) -> None:
        """FreeToken ``_resolve_hybrid_fetch``: cap each hybrid step's fetch by the
        bandwidth-matched fraction (pcie_bw / cpu_bw) so the PCIe pull and the CPU
        overflow GEMV finish together; the fixed cap of 1 applies only when no
        usable measurement exists. Sets ``hybrid_max_fetch`` to the slot count,
        which makes the fraction the operative cap (exactly FreeToken)."""
        from ..moe.bandwidth import probe
        p = probe(self.device, self.config.n_experts, self.config.hidden_dim,
                  self.config.intermediate_dim, self.config.n_experts_per_tok)
        if p is None or p["fraction"] is None:
            logger.warning(
                "hybrid fetch: no usable bandwidth measurement; capping each "
                "step's PCIe fetch at 1 miss (cache.hybrid_max_fetch)")
            return  # cache.hybrid_max_fetch stays its default of 1
        cache.hybrid_max_fetch = cache.num_experts  # inert: fraction is the cap
        cache.hybrid_fetch_fraction = p["fraction"]
        logger.info(
            "hybrid fetch: PCIe pulls %.0f%% of each step's misses (pcie %.2f "
            "GB/s vs cpu %.2f GB/s, fused=%s); the rest run on the CPU",
            p["fraction"] * 100, p["pcie_bw"], p["cpu_bw"], p["fused"])

    def _init_cpu_moe_executor(self, cache) -> None:
        """Build the persistent CPU MoE executor (decode-time expert compute).

        Faithful port of FreeToken's engine ``_init_cpu_moe_executor``: construct
        the executor over this cache's host banks, attach it (``set_cpu_executor``
        gates on decode_target in {"cpu", "hybrid"}) and keep a reference for the
        per-step ``raise_if_unhealthy`` watchdog check. FreeToken reads
        top_k/activation/apply_router_weight_on_input off the first MoE layer
        module; PlasticInfer's runner carries them as module constants, so they
        come from there. Must run before any decode (the worker pool has to be
        live and the pinned IO buffers stable).
        """
        from ..exec.runner_moe import (
            _MOE_ACTIVATION,
            _MOE_APPLY_ROUTER_WEIGHT_ON_INPUT,
        )
        from ..moe.cpu_executor import CpuMoeExecutor

        executor = CpuMoeExecutor(
            cache,
            top_k=self.config.n_experts_per_tok,
            activation=_MOE_ACTIVATION,
            apply_router_weight_on_input=_MOE_APPLY_ROUTER_WEIGHT_ON_INPUT,
            num_threads=0,          # auto: one thread per physical core
            max_tokens=1,           # this engine decodes one request at a time
            device=self.device,
            swiglu_alpha=1.702,     # silu activation; swiglu defaults unused
            swiglu_limit=None,
        )
        cache.set_cpu_executor(executor)
        self.cpu_moe_executor = executor

    def stream(self, input_ids: list[int], *, max_new_tokens: int = 16,
               request: RequestMeta | None = None):
        """Prefill + greedy decode, yielding each generated token with the
        cumulative wall-clock since the request started (pool setup +
        prefill + decode so far). The generator's return value is the
        full RunResult; `run()` is a thin wrapper that drains it.

        Timing semantics for benchmarks:
          TTFT  = elapsed_s of the first yielded pair
          TPOT  = gaps between consecutive yielded pairs
          e2e   = elapsed_s of the last yielded pair (plus argmax)
        """
        seq_len = len(input_ids)
        req = request or RequestMeta(
            seq_budget=self.cfg["max_position_embeddings"],
            gen_tokens=max_new_tokens)
        plan = MemoryPlanner().plan(self.profile, self.model, req)
        b = pool_budgets(plan, self.model)

        # The planner sizes a *rotating* dense window on the GPU (W layers
        # resident) but assumes a HOST source is copied in full. When the
        # model is too big to keep every layer, we run that window for
        # real (streaming + prefetch); otherwise every dense weight is
        # resident. Either way, charge any bytes resident beyond the
        # planner's W-based window budget to the expert slot pool so dense
        # + experts + KV still fit in HBM — otherwise the expert LRU would
        # be free to grow into the space dense occupies.
        #
        # Note dense_bytes counts per-layer tensors only: shared
        # (embed/norm/lm_head) lives in model.shared.safetensors and is
        # resident in full in both branches.
        streamed = plan.weight_window < self.layout.n_layers
        shared_bytes = sum(self.layout.tensor(n).nbytes
                           for n in self.layout.shared_tensor_names())
        if streamed:
            per_layer = self.layout.dense_per_layer_bytes(0)
            resident_dense = shared_bytes + plan.weight_window * per_layer
        else:
            resident_dense = shared_bytes + self.model.dense_bytes
        per_expert = self.layout.expert_total_bytes(0)
        expert_slots_bytes = max(
            per_expert, b.expert_slots_bytes
            - max(0, resident_dense - b.dense_window_bytes))

        if plan.seq_weight_source != "HOST":
            raise NotImplementedError(
                f"MoE engine requires dense weights in host RAM (plan says "
                f"{plan.seq_weight_source}); a model whose dense weights "
                f"exceed host RAM needs the streamed dense×MoE runner "
                f"(future work)")
        if seq_len + max_new_tokens > plan.kv_hot_tokens:
            raise ValueError(
                f"request ({seq_len} prompt + {max_new_tokens} gen) exceeds "
                f"the GPU KV pool ({plan.kv_hot_tokens} tokens); "
                f"HOST_STREAM long context is future work")

        t_start = time.monotonic()
        if streamed:
            host = self._load_host_dense()
            weights: DenseWeights | StreamingDenseWeights = \
                StreamingDenseWeights(
                    host["layers"], shared=host["shared"],
                    n_layers=self.layout.n_layers, device=self.device,
                    dtype=self.dtype, window=plan.weight_window)
        else:
            weights = self._load_dense()

        if self._moe_cache is None:
            self._moe_cache = self._build_moe_cache(expert_slots_bytes)
        moe_cache = self._moe_cache
        moe_cache.reset_stats()  # per-request metrics window

        kv_cfg = make_kv_config(
            self.config,
            max_pages=b.kv_pages * self.config.n_layers,  # physical pages
            max_host_chunks=64,
            device=self.device,
            dtype=self.config.dtype)
        store = KVStore(kv_cfg)
        cache = store.new_request()

        ids = torch.tensor(input_ids, device=self.device)
        t_setup = time.monotonic()
        logits = prefill_forward_moe_paged(weights, self.config, moe_cache,
                                           store, cache, ids)
        prefill_s = time.monotonic() - t_setup
        if self.cpu_moe_executor is not None:
            # one pinned read per forward (FreeToken forward_batch): a dead
            # flag-handshake coordinator surfaces loudly, not as stale outputs
            self.cpu_moe_executor.raise_if_unhealthy()

        tokens = list(input_ids)
        # First generated token comes from the prefill logits (they
        # predict the token after the prompt). decode_step then takes
        # that token as input and predicts the next, writing its KV at
        # the new position — same convention the equivalence tests use.
        tokens.append(int(logits.argmax().item()))
        t_first = time.monotonic()
        yield tokens[-1], t_first - t_start

        t_dec = time.monotonic()
        for _ in range(max_new_tokens - 1):
            logits = decode_step_moe_paged(weights, self.config, moe_cache,
                                           store, cache, tokens[-1])
            if self.cpu_moe_executor is not None:
                self.cpu_moe_executor.raise_if_unhealthy()
            tokens.append(int(logits.argmax().item()))
            yield tokens[-1], time.monotonic() - t_start
        decode_s = time.monotonic() - t_dec

        store.free_request(cache)
        e2e_s = time.monotonic() - t_start
        metrics = {
            "setup_seconds": t_setup - t_start,
            "prefill_seconds": prefill_s,
            "prefill_tokens_per_s": (seq_len / prefill_s
                                     if prefill_s > 0 else 0.0),
            "ttft_seconds": t_first - t_start,
            "decode_seconds": decode_s,
            "tpot_seconds": (decode_s / (max_new_tokens - 1)
                             if max_new_tokens > 1 else 0.0),
            "decode_tokens_per_s": (max_new_tokens / decode_s
                                    if decode_s > 0 else 0.0),
            "e2e_seconds": e2e_s,
            "e2e_tokens_per_s": (max_new_tokens / e2e_s
                                 if e2e_s > 0 else 0.0),
        }
        # Expert movement from the offload cache's realized LRU (per-step averages
        # over this request's decode window; the cache's host side is the bank
        # source, not an LRU, so there is no host hit/miss pair anymore).
        mstats = moe_cache.decode_miss_stats()
        calls = mstats["layer_calls"]
        metrics["expert_gpu_hit_rate"] = (
            1.0 - mstats["miss_rate"] if calls else 0.0)
        metrics["expert_gpu_misses"] = round(
            mstats["missing_per_layer"] * calls)
        metrics["expert_host_hit_rate"] = 0.0
        metrics["expert_host_misses"] = 0

        return RunResult(tokens=tokens, plan=plan, metrics=metrics)

    def run(self, input_ids: list[int], *, max_new_tokens: int = 16,
            request: RequestMeta | None = None) -> RunResult:
        """Prefill + greedy decode. Returns tokens (input + generated)."""
        it = self.stream(input_ids, max_new_tokens=max_new_tokens,
                         request=request)
        try:
            while True:
                next(it)
        except StopIteration as e:
            return e.value
