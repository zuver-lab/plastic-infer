"""CLI for plastic-infer: convert and run.

  plastic-infer convert <hf_dir> <out_dir> [--dtype bf16|fp32]
  plastic-infer run <out_dir> --prompt "..." [--max-new-tokens N]
                          [--device cuda:0] [--seed 0] [--input-ids "1 2 3"]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from ..planning.profile import DeviceProfile
from ..weights.convert import convert
from .engine import Engine


def _default_profile(device: torch.device) -> DeviceProfile:
    if device.type == "cuda":
        hbm = torch.cuda.get_device_properties(device).total_memory
    else:
        hbm = 8 << 30
    page_size = os.sysconf("SC_PAGE_SIZE")
    phys_pages = os.sysconf("SC_PHYS_PAGES")
    ram = phys_pages * page_size
    return DeviceProfile(
        hbm_bytes=int(hbm),
        host_ram_bytes=int(ram),
        pin_limit_bytes=int(hbm // 2),
        b_h2d=12e9,   # rough PCIe effective
        b_host=30e9,  # rough RAM read bandwidth
        b_disk=7e9,   # rough NVMe sequential read
    )


def _tokenize(model_dir: Path, prompt: str,
              input_ids: str | None) -> list[int]:
    if input_ids:
        return [int(x) for x in input_ids.split()]
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit(
            f"{model_dir}: no tokenizer.json (pass --input-ids instead)")
    from tokenizers import Tokenizer  # lazy: not a hard runtime dependency
    return Tokenizer.from_file(str(tok_path)).encode(prompt).ids


def _run_cli(args) -> None:
    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"))
    model_dir = Path(args.model_dir)
    ids = _tokenize(model_dir, args.prompt, args.input_ids)
    if not ids:
        raise SystemExit("empty input")

    torch.manual_seed(args.seed)
    engine = Engine(model_dir, _default_profile(device), device=device)
    result = engine.run(ids, max_new_tokens=args.max_new_tokens)

    m = result.metrics
    plan = result.plan
    print(f"plan: dense {plan.seq_weight_source} W={plan.weight_window} | "
          f"experts {plan.expert_source} "
          f"(gpu slots={plan.expert_slots}, host={plan.expert_host_slots}) | "
          f"kv {plan.kv_hot_tokens} tokens / {plan.long_ctx_mode}")
    print(f"prefill: {m['prefill_seconds']:.3f}s | "
          f"decode: {m['decode_tokens_per_s']:.2f} tok/s "
          f"({m['decode_seconds']:.3f}s for {args.max_new_tokens})")
    print(f"expert GPU hit {m['expert_gpu_hit_rate']:.3f} "
          f"({m['expert_gpu_misses']} misses) | "
          f"host LRU hit {m['expert_host_hit_rate']:.3f} "
          f"({m['expert_host_misses']} misses)")

    if args.prompt:
        tok_path = model_dir / "tokenizer.json"
        if tok_path.exists():
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(tok_path))
            gen = result.tokens[len(ids):]
            print(f"--- generated ---")
            print(tok.decode(gen))
    else:
        print(f"tokens: {' '.join(map(str, result.tokens))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plastic-infer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="HF dir -> custom disk layout")
    p_conv.add_argument("hf_dir")
    p_conv.add_argument("out_dir")
    p_conv.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p_conv.add_argument("--delete-original", action="store_true",
                        help="rm the HF source dir after a successful convert")
    p_conv.set_defaults(cmd="convert")

    p_run = sub.add_parser("run", help="run a converted model")
    p_run.add_argument("model_dir")
    p_run.add_argument("--prompt", default="")
    p_run.add_argument("--input-ids", default=None,
                       help="space-separated token ids (skips tokenizer)")
    p_run.add_argument("--max-new-tokens", type=int, default=32)
    p_run.add_argument("--device", default=None)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.set_defaults(cmd="run")

    args = parser.parse_args(argv)
    if args.cmd == "convert":
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        convert(args.hf_dir, args.out_dir, dtype=dtype,
                delete_original=args.delete_original)
        print(f"converted {args.hf_dir} -> {args.out_dir}")
    else:
        _run_cli(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
