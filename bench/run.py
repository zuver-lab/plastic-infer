"""Performance benchmark for a converted model directory.

Usage:
  python -m bench.run models/Qwen3-30B-A3B-pi --prompt "The capital of France is" \
      --max-new-tokens 32 --repeats 3

Each run is a cold start (the engine builds fresh dense load + KV pool +
expert pools per request), so TTFT covers the full setup -> first-token
path. Per request it reports:

  TTFT             time from request start to the first generated token
  TPOT             per-output-token decode latency (median / p95)
  prefill tok/s    prompt tokens processed per second
  decode tok/s     generated tokens per second during the decode loop
  e2e tok/s        end-to-end generation throughput
  peak GPU (GiB)   high-water CUDA allocation for the request

plus the engine's three-tier metrics (expert GPU hit rate / host LRU hit
rate / disk spills). Repeats aggregate into min / median / max columns.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

from plastic_infer.runtime.cli import _default_profile, _tokenize
from plastic_infer.runtime.engine import Engine


def _p95(values: list[float]) -> float:
    s = sorted(values)
    return s[max(0, min(len(s) - 1, int(0.95 * len(s))))]


def _bench_one(engine: Engine, ids: list[int], max_new_tokens: int,
               seed: int) -> dict:
    """Run one request through engine.stream(), measuring per-token
    timing. Returns the per-run metric dict (plus raw per-step gaps)."""
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    it = engine.stream(ids, max_new_tokens=max_new_tokens)
    t0 = time.monotonic()
    first_elapsed: float | None = None
    prev_elapsed: float | None = None
    tpot_gaps: list[float] = []
    try:
        while True:
            _tok, elapsed = next(it)
            if first_elapsed is None:
                first_elapsed = elapsed
            else:
                tpot_gaps.append(elapsed - prev_elapsed)
            prev_elapsed = elapsed
    except StopIteration as e:
        result = e.value
    e2e_s = time.monotonic() - t0

    m = result.metrics
    n_decode = max(0, max_new_tokens - 1)
    return {
        "ttft_s": first_elapsed,
        "prefill_s": m["prefill_seconds"],
        "prefill_tok_per_s": m["prefill_tokens_per_s"],
        "tpot_gaps_s": tpot_gaps,
        "tpot_median_s": (statistics.median(tpot_gaps) if tpot_gaps
                          else float("nan")),
        "tpot_p95_s": (_p95(tpot_gaps) if tpot_gaps else float("nan")),
        "decode_tok_per_s": (n_decode / m["decode_seconds"]
                             if m["decode_seconds"] > 0 else 0.0),
        "e2e_tok_per_s": (max_new_tokens / e2e_s if e2e_s > 0 else 0.0),
        "peak_gpu_gib": (torch.cuda.max_memory_allocated(engine.device) / 2**30
                         if engine.device.type == "cuda" else 0.0),
        "expert_gpu_hit": m["expert_gpu_hit_rate"],
        "expert_gpu_misses": m["expert_gpu_misses"],
        "expert_host_hit": m["expert_host_hit_rate"],
        "disk_reads": m["expert_host_misses"],
    }


def _col(runs: list[dict], key: str) -> list[float]:
    return [r[key] for r in runs]


def _fmt(v: float) -> str:
    return "—" if v != v else f"{v:.4g}"


def _report(engine: Engine, prompt: str, n_prompt: int, n_gen: int,
            runs: list[dict]) -> None:
    p = engine.profile
    n = len(runs)
    print(f"\n== {prompt!r}: {n_prompt} prompt -> {n_gen} new tokens"
          f" ({n} run{'s' if n != 1 else ''}) ==")
    print(f"  profile: hbm {p.hbm_bytes/2**30:.1f} GiB | host "
          f"{p.host_ram_bytes/2**30:.0f} GiB")

    if n == 1:
        r = runs[0]
        rows = [
            ("TTFT (s)", r["ttft_s"]),
            ("TPOT median (s)", r["tpot_median_s"]),
            ("TPOT p95 (s)", r["tpot_p95_s"]),
            ("prefill tok/s", r["prefill_tok_per_s"]),
            ("decode tok/s", r["decode_tok_per_s"]),
            ("e2e tok/s", r["e2e_tok_per_s"]),
            ("peak GPU (GiB)", r["peak_gpu_gib"]),
            ("expert GPU hit", r["expert_gpu_hit"]),
            ("host LRU hit", r["expert_host_hit"]),
            ("disk reads", float(r["disk_reads"])),
        ]
        w = max(len(k) for k, _ in rows) + 2
        for k, v in rows:
            print(f"  {k:<{w}}{_fmt(v)}")
    else:
        rows = [
            ("TTFT (s)", "ttft_s"),
            ("TPOT median (s)", "tpot_median_s"),
            ("TPOT p95 (s)", "tpot_p95_s"),
            ("prefill tok/s", "prefill_tok_per_s"),
            ("decode tok/s", "decode_tok_per_s"),
            ("e2e tok/s", "e2e_tok_per_s"),
            ("peak GPU (GiB)", "peak_gpu_gib"),
            ("expert GPU hit", "expert_gpu_hit"),
            ("host LRU hit", "expert_host_hit"),
            ("disk reads", "disk_reads"),
        ]
        w = max(len(k) for k, _ in rows) + 2
        print(f"  {'metric':<{w}}{'min':>10}{'median':>10}{'max':>10}")
        for k, key in rows:
            vs = _col(runs, key)
            print(f"  {k:<{w}}{_fmt(min(vs)):>10}{_fmt(statistics.median(vs)):>10}"
                  f"{_fmt(max(vs)):>10}")
    # totals over all generated tokens
    tot = sum(len(r["tpot_gaps_s"]) + 1 for r in runs)
    print(f"  total tokens generated: {tot}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python -m bench.run models/Qwen3-30B-A3B-pi \\\n"
               "      --prompt 'The capital of France is' --max-new-tokens 32\n"
               "  python -m bench.run models/Qwen3-30B-A3B-pi --repeats 5 \\\n"
               "      --prompt 'The capital of' --prompt 'Hello world'")
    parser.add_argument("model_dir")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--input-ids", default=None,
                       help="space-separated token ids (skips tokenizer; "
                            "replaces --prompt)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"))
    model_dir = Path(args.model_dir)
    engine = Engine(model_dir, _default_profile(device), device=device)

    print(f"bench: {model_dir} on {device}")
    # `--prompt` default is [] so argparse does not append to it; fall
    # back here so a bare invocation still benchmarks something.
    prompts = ["<ids>"] if args.input_ids else (args.prompt
                                                or ["The capital of France is"])
    for prompt in prompts:
        ids = _tokenize(model_dir, prompt, args.input_ids)
        if not ids:
            raise SystemExit("empty input")
        runs = []
        for rep in range(args.repeats):
            torch.manual_seed(args.seed + rep)
            runs.append(_bench_one(engine, ids, args.max_new_tokens,
                                   args.seed + rep))
        _report(engine, prompt, len(ids), args.max_new_tokens, runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
