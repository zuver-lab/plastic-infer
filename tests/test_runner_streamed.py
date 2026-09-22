"""Tests for the streamed dense runner (M3: disk weight layer).

Gates:
  1. Streamed prefill/decode == HF Llama, with every layer actually read
     from per-layer safetensors files on disk (DISK source, W=1).
  2. D5 across the disk tier: W=1 (stream everything) == W=n_layers
     (all resident) == in-memory runner == HF.
  3. The window hits the disk exactly n_layers times for a W=1 prefill.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaForCausalLM

from plastic_infer.exec.runner import (
    DenseKVCache,
    DenseWeights,
    ModelConfig,
    prefill_forward,
)
from plastic_infer.exec.runner_streamed import (
    decode_step_streamed,
    prefill_forward_streamed,
)
from plastic_infer.store.weights import DenseWindow
from plastic_infer.weights.disk import DiskLayerSource
from plastic_infer.weights.layout import build_layout_from_manifest

from test_equiv_smoke import (  # noqa: F401
    _extract_weights,
    _make_config,
    _tiny_llama_config,
)


@pytest.fixture(scope="module")
def hf_model() -> LlamaForCausalLM:
    torch.manual_seed(42)
    model = LlamaForCausalLM(_tiny_llama_config())
    model.eval()
    return model


@pytest.fixture(scope="module")
def flat_weights(hf_model: LlamaForCausalLM) -> dict[str, torch.Tensor]:
    return _extract_weights(hf_model)


@pytest.fixture(scope="module")
def config(hf_model: LlamaForCausalLM) -> ModelConfig:
    return _make_config(hf_model.config)


def _write_model_files(tmp_path, flat: dict[str, torch.Tensor]
                       ) -> tuple[DiskLayerSource, int]:
    """Write per-layer + shared safetensors files; return source + one-layer bytes."""
    layer_tensors: dict[int, dict[str, torch.Tensor]] = {}
    shared: dict[str, torch.Tensor] = {}
    for name, t in flat.items():
        if name.startswith("layers."):
            l = int(name.split(".")[1])
            layer_tensors.setdefault(l, {})[name] = t
        else:
            shared[name] = t

    for l, tensors in layer_tensors.items():
        save_file(tensors, tmp_path / f"model.layers.{l}.safetensors")
    save_file(shared, tmp_path / "model.shared.safetensors")

    per_layer = {name.split(".", 2)[2]: t.numel() * t.element_size()
                 for name, t in layer_tensors[0].items()}
    shared_nbytes = {name: t.numel() * t.element_size()
                     for name, t in shared.items()}
    layout = build_layout_from_manifest(
        n_layers=len(layer_tensors),
        per_layer_dense_tensors=per_layer,
        shared_tensors=shared_nbytes,
        layer_prefix="layers.{layer}",
    )
    src = DiskLayerSource(tmp_path, layout)
    layer_bytes = sum(t.numel() * t.element_size()
                      for t in src.layer(0).values())
    return src, layer_bytes


def _shared_on(src: DiskLayerSource) -> dict[str, torch.Tensor]:
    return dict(src.shared())


def _streamed(src: DiskLayerSource, layer_bytes: int,
              window_layers: int) -> DenseWindow:
    return DenseWindow(src, budget_bytes=window_layers * layer_bytes)


def _assert_allclose(a: torch.Tensor, b: torch.Tensor, where: str) -> None:
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-2), (
        f"{where}: max diff = {(a - b).abs().max().item():.6f}"
    )


class TestStreamedPrefill:
    def test_prefill_from_disk_matches_hf(
        self, tmp_path, hf_model: LlamaForCausalLM,
        flat_weights: dict[str, torch.Tensor], config: ModelConfig,
    ) -> None:
        """W=1: every layer read from disk, logits == HF."""
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (16,))

        src, layer_bytes = _write_model_files(tmp_path, flat_weights)
        window = _streamed(src, layer_bytes, window_layers=1)
        shared = _shared_on(src)
        kv = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)
        our_logits = prefill_forward_streamed(window, shared, config, kv,
                                              input_ids)

        with torch.no_grad():
            hf_logits = hf_model(input_ids.unsqueeze(0)).logits[0, -1]

        _assert_allclose(our_logits, hf_logits, "streamed prefill vs HF")

        # Every layer was read from disk exactly once (W=1 streaming)
        assert window.misses == config.n_layers
        assert window.pool.used_bytes <= layer_bytes

    def test_streamed_equals_inmemory_equals_window_full(
        self, tmp_path, flat_weights, config,
    ) -> None:
        """D5 across the disk tier: placement never changes logits."""
        torch.manual_seed(1)
        input_ids = torch.randint(0, config.vocab_size, (12,))

        src, layer_bytes = _write_model_files(tmp_path, flat_weights)
        shared = _shared_on(src)

        # In-memory anchor
        kv_mem = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)
        mem_logits = prefill_forward(DenseWeights(dict(flat_weights)), config,
                                     kv_mem, input_ids)

        # Streamed, W=1 (every layer reloaded)
        w1 = _streamed(src, layer_bytes, window_layers=1)
        kv1 = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)
        s1 = prefill_forward_streamed(w1, shared, config, kv1, input_ids)

        # Streamed, W=n_layers (everything resident, no reloads)
        wfull = _streamed(src, layer_bytes, window_layers=config.n_layers)
        kvf = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)
        sfull = prefill_forward_streamed(wfull, shared, config, kvf, input_ids)

        _assert_allclose(s1, mem_logits, "streamed W=1 vs in-memory")
        _assert_allclose(sfull, mem_logits, "streamed W=n vs in-memory")
        _assert_allclose(s1, sfull, "streamed W=1 vs W=n")
        assert wfull.hits == 0 and wfull.misses == config.n_layers


class TestStreamedDecode:
    def test_decode_matches_hf(
        self, tmp_path, hf_model: LlamaForCausalLM,
        flat_weights, config,
    ) -> None:
        """Prefill + decode, each step streaming from disk == HF."""
        torch.manual_seed(3)
        prompt_len, n_decode = 6, 3
        input_ids = torch.randint(0, config.vocab_size, (prompt_len,))

        src, layer_bytes = _write_model_files(tmp_path, flat_weights)
        window = _streamed(src, layer_bytes, window_layers=1)
        shared = _shared_on(src)
        kv = DenseKVCache(config, max_seq_len=128, dtype=torch.float32)

        prefill_logits = prefill_forward_streamed(
            window, shared, config, kv, input_ids)
        our_logits: list[torch.Tensor] = [prefill_logits]
        our_tokens: list[int] = [int(prefill_logits.argmax().item())]
        for _ in range(n_decode):
            logits = decode_step_streamed(
                window, shared, config, kv, our_tokens[-1])
            our_logits.append(logits)
            our_tokens.append(int(logits.argmax().item()))

        with torch.no_grad():
            full_ids = torch.tensor(
                input_ids.tolist() + our_tokens[:n_decode]).unsqueeze(0)
            hf_logits_all = hf_model(full_ids).logits[0]

        for step in range(n_decode + 1):
            pos = prompt_len - 1 + step
            _assert_allclose(our_logits[step], hf_logits_all[pos],
                             f"streamed decode step {step}")
