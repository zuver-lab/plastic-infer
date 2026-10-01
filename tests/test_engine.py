"""Engine tests: plan -> pools -> runner over a converted model dir.

End-to-end through the runtime layer (runtime/engine.py): a converted
tiny Qwen3Moe directory (dense HOST-resident, experts host LRU over
disk, paged KV) is planned and run on CUDA (the FreeToken-ported engine
is CUDA+bf16 only, like FreeToken itself). The engine exposes tokens +
metrics, not intermediate logits, so equivalence is asserted on the
greedy decode stream: if prefill logits and every decode step matched
HF, the emitted tokens equal HF greedy generation token-for-token.

Also checks the plan that comes out is the expected shape for this
model (dense HOST, experts HOST_FIRST) and that metrics are populated.
"""

from __future__ import annotations

import json

import pytest
import torch
from transformers import Qwen3MoeForCausalLM

from plastic_infer.planning.budget import MemoryPlanner, RequestMeta
from plastic_infer.planning.profile import DeviceProfile
from plastic_infer.runtime.engine import Engine
from plastic_infer.weights.convert import convert_from_dict

from test_qwen3moe_equiv import (
    tiny_qwen3moe_config,
    tiny_qwen3moe_config_dict,
)
from _moe_cache_helpers import require_cuda


def _profile() -> DeviceProfile:
    """A generous host-RAM / modest-HBM profile: the tiny model's dense
    weights fit host, experts fit host, KV fits the GPU pool."""
    return DeviceProfile(
        hbm_bytes=8 << 30,
        host_ram_bytes=16 << 30,
        pin_limit_bytes=4 << 30,
        b_h2d=12e9,
        b_host=30e9,
        b_disk=7e9,
    )


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> tuple:
    """Convert a tiny Qwen3Moe into a model dir (with config.json)."""
    torch.manual_seed(21)
    model = Qwen3MoeForCausalLM(tiny_qwen3moe_config())
    model.eval()
    out = tmp_path_factory.mktemp("model") / "converted"
    convert_from_dict(model.state_dict(), tiny_qwen3moe_config_dict(),
                      out, dtype=torch.bfloat16)
    cfg = dict(tiny_qwen3moe_config_dict())
    cfg["torch_dtype"] = "bfloat16"
    (out / "config.json").write_text(json.dumps(cfg))
    return out, model


@pytest.fixture(scope="module")
def hf_model(model_dir) -> Qwen3MoeForCausalLM:
    dev = require_cuda()
    model = model_dir[1]
    model.to(device=dev, dtype=torch.bfloat16)
    model.eval()
    return model


@pytest.fixture(scope="module")
def engine(model_dir) -> Engine:
    return Engine(model_dir[0], _profile(), device=require_cuda())


class TestEnginePlan:
    def test_plan_shape(self, engine: Engine) -> None:
        """Dense fits host (HOST); experts fit host (HOST_FIRST for a
        model this small collapses to HOST — both are resident)."""
        seq = [10, 20, 30]
        result = engine.run(seq, max_new_tokens=4)
        plan = result.plan
        assert plan.seq_weight_source == "HOST"
        assert plan.weight_window == engine.layout.n_layers
        assert plan.experts_fit_in_host

    def test_metrics_populated(self, engine: Engine) -> None:
        result = engine.run([5, 6, 7], max_new_tokens=4)
        m = result.metrics
        assert m["prefill_seconds"] > 0
        assert m["decode_tokens_per_s"] > 0
        assert 0.0 <= m["expert_gpu_hit_rate"] <= 1.0
        assert 0.0 <= m["expert_host_hit_rate"] <= 1.0
        assert result.plan.gpu_total_bytes > 0


class TestEngineEquivalence:
    def test_greedy_decode_matches_hf(self, engine: Engine,
                                      hf_model: Qwen3MoeForCausalLM) -> None:
        torch.manual_seed(31)
        prompt = torch.randint(0, engine.config.vocab_size, (8,)).tolist()
        n_gen = 6

        result = engine.run(prompt, max_new_tokens=n_gen)
        assert result.tokens[:len(prompt)] == prompt
        assert len(result.tokens) == len(prompt) + n_gen

        with torch.no_grad():
            hf_ids = hf_model.generate(
                torch.tensor([prompt], device=hf_model.device),
                max_new_tokens=n_gen, do_sample=False, use_cache=True,
            )
        hf_gen = hf_ids[0, len(prompt):].tolist()

        assert result.tokens[len(prompt):] == hf_gen, (
            f"our {result.tokens[len(prompt):]} != HF {hf_gen}")

    def test_deterministic_across_runs(self, engine: Engine) -> None:
        prompt = [3, 9, 14]
        a = engine.run(prompt, max_new_tokens=5).tokens
        b = engine.run(prompt, max_new_tokens=5).tokens
        assert a == b

    def test_cpu_locked_decode_matches_hf(self, model_dir, hf_model,
                                          monkeypatch) -> None:
        """Shrinking the WSL pin budget below the tiny banks auto-locks every
        MoE layer to CPU decode (_auto_cpu_layers -> _init_cpu_moe_executor):
        the _cpu_moe executor is built and greedy decode still matches HF
        token-for-token."""
        monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "0.000001")
        eng = Engine(model_dir[0], _profile(), device=require_cuda())
        torch.manual_seed(32)
        prompt = torch.randint(0, eng.config.vocab_size, (8,)).tolist()
        n_gen = 4
        result = eng.run(prompt, max_new_tokens=n_gen)
        assert eng.cpu_moe_executor is not None
        assert eng._moe_cache is not None
        assert eng._moe_cache.decode_target == "cpu"

        with torch.no_grad():
            hf_ids = hf_model.generate(
                torch.tensor([prompt], device=hf_model.device),
                max_new_tokens=n_gen, do_sample=False, use_cache=True)
        assert result.tokens[len(prompt):] == hf_ids[0, len(prompt):].tolist(), (
            f"cpu-locked decode {result.tokens[len(prompt):]} != HF "
            f"{hf_ids[0, len(prompt):].tolist()}")

    def test_stream_timing_and_run_equivalence(self, engine: Engine) -> None:
        """stream() yields one (token, elapsed) per generated token with
        non-decreasing elapsed, and run() (which drains it) gives the
        same tokens. TTFT is the first pair's elapsed."""
        prompt = [4, 8, 15]
        n_gen = 4
        it = engine.stream(prompt, max_new_tokens=n_gen)
        elapsed: list[float] = []
        gen: list[int] = []
        try:
            while True:
                tok, t = next(it)
                gen.append(tok)
                elapsed.append(t)
        except StopIteration as e:
            result = e.value

        assert len(gen) == n_gen
        assert elapsed == sorted(elapsed)      # monotonic cumulative time
        assert elapsed[0] > 0                  # TTFT
        assert 0 < result.metrics["ttft_seconds"] <= elapsed[-1]
        assert result.metrics["e2e_seconds"] >= elapsed[-1]
        assert result.tokens == prompt + gen

        run_result = engine.run(prompt, max_new_tokens=n_gen)
        assert run_result.tokens == result.tokens
        assert run_result.metrics["ttft_seconds"] > 0

    def test_rejects_oversized_request(self, engine: Engine) -> None:
        """Requests beyond the GPU KV pool are rejected clearly
        (HOST_STREAM long context is future work)."""
        req = RequestMeta(seq_budget=2 ** 31, gen_tokens=1)
        plan = MemoryPlanner().plan(engine.profile, engine.model, req)
        too_long = plan.kv_hot_tokens + 10
        with pytest.raises(ValueError, match="exceeds the GPU KV pool"):
            engine.run([0] * too_long, max_new_tokens=1)
