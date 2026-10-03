# bench

Performance benchmark for a converted model directory.

## Usage

```bash
python -m bench.run models/Qwen3-30B-A3B-pi \
    --prompt "The capital of France is" \
    --max-new-tokens 32 \
    --repeats 3
```

Options:

| flag | default | meaning |
|------|---------|---------|
| `--prompt` | `"The capital of France is"` | repeatable; one benchmark per prompt |
| `--max-new-tokens` | 32 | generated tokens per run |
| `--repeats` | 1 | runs per prompt (min/median/max columns) |
| `--device` | cuda:0 if present | run device |
| `--seed` | 0 | base seed (offset per repeat) |

## Metrics

Each run is a **cold start** — the engine builds fresh dense load + KV
pool + expert pools per request, so TTFT covers the whole setup→first
token path:

- **TTFT** (s) — request start to first generated token
- **TPOT** (s) — per-output-token decode latency, median + p95
- **prefill tok/s** — prompt tokens / prefill time
- **decode tok/s** — generated tokens / decode loop time
- **e2e tok/s** — end-to-end generation throughput
- **peak GPU (GiB)** — high-water CUDA allocation (cuda only)
- **expert GPU hit / host LRU hit / disk reads** — the three-tier
  placement metrics from the engine

`--repeats N > 1` reports min / median / max of each metric.

## How it works

`bench/run.py` drives `Engine.stream()`, which yields `(token,
elapsed_s)` per generated token (elapsed = cumulative wall-clock since
request start). TTFT is the first pair's elapsed; TPOT is the gaps
between consecutive pairs. `stream()` is also what `Engine.run()`
delegates to, so the benchmark measures the exact production path.
