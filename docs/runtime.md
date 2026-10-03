# runtime —— 引擎与 CLI（把 plan→池→runner 串起来）

> 对应包：`src/plastic_infer/runtime/` · 设计依据：DESIGN.md 全篇的执行入口

## 1. 职责

`runtime` 是系统的**装配层**：一个请求到达后，它（1）加载模型 layout/config，（2）调规划器出 `Plan`，（3）把预算实例化成真正的池（dense 权重、专家三层池、KV 页池），（4）选 runner 跑 prefill + decode，（5）产出 token 与一组性能指标。对外只暴露两个入口：`Engine`（Python API）与 `plastic-infer` CLI。

## 2. 文件地图

| 文件 | 内容 |
|------|------|
| `engine.py` | `Engine`：装配与请求生命周期；`stream()`/`run()`；`build_model_meta` / `build_runner_config` |
| `cli.py` | `convert` / `run` 子命令；默认设备画像；tokenizer 封装 |
| `__init__.py` | 包声明 |

## 3. 装配辅助函数（`engine.py`）

### `build_model_meta(layout, cfg) -> ModelMeta`

从 layout（实测字节）与 config.json 拼出规划器需要的模型元数据：`dense_bytes`/`per_expert_bytes` 直接来自 layout 实测；`kv_bytes_per_token` 由 `n_layers × 2 × n_kv_heads × head_dim × dtype_bytes` 计算。

### `build_runner_config(cfg, *, dtype, ...) -> ModelConfig`

把 HF config 转成 `exec.ModelConfig`：**`head_dim` 取 config 显式值**（Qwen3 的 128 ≠ `hidden//n_heads` 的 64，见 [exec.md](exec.md)）、`rope_base=1e6`、`n_experts=128`、`n_experts_per_tok=8`、`qk_norm` 由「layout 中是否存在 `self_attn.q_norm.weight`」判定。

## 4. `Engine` 与请求生命周期

### `RunResult`

```python
@dataclass(frozen=True)
class RunResult:
    tokens: list[int]           # prompt + 生成 token
    metrics: dict               # 见 §5
    plan: Plan
```

### `stream()` 与 `run()`

```python
def stream(self, input_ids, *, max_new_tokens=16, request=None):
    # 生成器：yield (token, elapsed_s)，elapsed 自请求开始累计；
    # 迭代耗尽时 return RunResult（经 StopIteration.value 传递）
def run(self, input_ids, *, max_new_tokens=16, request=None) -> RunResult:
    # 薄封装：drain 完 stream() 并返回其 return value
```

**时序语义**：每次 yield 的 `elapsed` 是单调递增的累计墙钟（setup + prefill + 已发生的 decode）。由此基准直接定义：

- **TTFT** = 第一次 yield 的 `elapsed`
- **TPOT** = 相邻两次 yield 的 `elapsed` 之差
- **e2e** = 最后一次 yield 的 `elapsed`

`run()` 与 `stream()` 走完全同一条路径（bench 即用 `stream()`，见 [bench/README.md](../bench/README.md)）。

### 请求生命周期（`stream` 内部）

```
1. RequestMeta(seq_budget, gen_tokens) → MemoryPlanner().plan(profile, model, req)
2. pool_budgets(plan, model) → PoolBudgets
3. dense：seq_weight_source == "HOST"（Qwen3 满足）
     → DiskLayerSource 全量装入 DenseWeights（device=GPU）
   "DISK" 分支：对 MoE 引擎暂不支持 → 明确报错
4. 专家：HostExpertLru(disk_expert_src, plan.expert_host_slots × per_expert_bytes)
     → ExpertSlotPool(host_lru, b.expert_slots_bytes, device=GPU)
5. KV：make_kv_config(config, max_pages=b.kv_pages * n_layers, device=GPU, dtype=bf16)
     → KVStore.new_request()   （×n_layers 后为全部物理页；kv_pages 是每层页数）
6. 预检：len(input_ids) + max_new_tokens ≤ plan.kv_hot_tokens，否则 ValueError
7. prefill_forward_moe_paged → 采样 decode_step_moe_paged 循环
8. 记录 metrics，return RunResult
```

### 4.3 GPU 超订修正（M5 的关键修复）

规划器按「dense 旋转窗口」切预算，但 `Engine` 实际让 dense **全量常驻**（HOST 源读一次拷贝进 GPU）。若直接采用 `PoolBudgets.expert_slots_bytes`，专家 LRU 会自由增长进 dense 常驻占用的显存 → 24 GB 卡必然 OOM。修复：把超差从专家槽预算中扣掉：

```python
per_expert = self.layout.expert_total_bytes(0)
expert_slots_bytes = max(
    per_expert,
    b.expert_slots_bytes - max(0, self.model.dense_bytes - b.dense_window_bytes))
```

（详见 [planning.md](planning.md) §7 的说明。）M5 真机实测峰值 24.25 / 24.58 GiB，紧贴 24 GB 上限 —— 该修正不可或缺。

## 5. 指标字典（`stream` 产出）

```
setup_seconds / prefill_seconds / prefill_tokens_per_s
ttft_seconds / decode_seconds / tpot_seconds / decode_tokens_per_s
e2e_seconds / e2e_tokens_per_s
expert_gpu_hit_rate / expert_gpu_misses      # ExpertSlotPool 命中
expert_host_hit_rate / expert_host_misses    # HostExpertLru 命中（miss = 磁盘读）
```

`expert_host_misses` 即**磁盘读取次数**。

## 6. CLI（`cli.py`）

```bash
plastic-infer convert <hf_dir> <out_dir> [--dtype bf16]
plastic-infer run    <out_dir> --prompt "..." [--max-new-tokens N]
                    [--device cuda:0] [--seed 0] [--input-ids]
```

- `convert`：调用 [weights.md](weights.md) 的 `convert()`，把 HF 目录转成自定义布局。
- `run`：构造 `_default_profile(device)` → `Engine` → `_tokenize`（用 `tokenizers.Tokenizer.from_file(out_dir/tokenizer.json)`，懒加载、缺文件给明确错误）→ `engine.run` → 打印 tokens 与指标。
- `--input-ids`：跳过 tokenizer，直接给空格分隔的 id 列表（用于无 tokenizer 的小模型 / bench）。
- 默认 prompt 行为：`--prompt` 可重复，`default=[]`，裸调用回退 `["The capital of France is"]`。

## 7. 真机实测（Qwen3-30B-A3B，24 GB RTX 3090）

| 指标 | 冷启动（首次） | 热启动（页缓存已热） |
|------|------|------|
| TTFT | **295.6 s** | **31.45 s** |
| TPOT median / p95 | 1.62 s / 12.97 s | **0.449 s / 1.826 s** |
| decode tok/s | 0.27 | **1.31** |
| e2e tok/s | 0.078 | 0.57 |
| peak GPU | 23.1 GiB | 23.13 GiB |
| expert GPU hit | 0.820 | 0.820 |
| host LRU hit | 0.032 | 0.032 |
| disk reads | 2293 | 2293 |

冷启动的 264 s 差主要来自 61 GB 权重文件的冷 mmap first-touch；热启动后 TTFT 仍高达 31 s，根因是 setup 的 dense 全量加载 + KV 池预分配 + 384 次/层的串行专家 dispatch（compute-load 流水线未实现，见 [exec.md](exec.md) §7）。

## 8. 设计约定与现状

- **每请求冷启动**：`stream()` 每次请求都重建 dense 加载、KV 池、专家池 —— 没有跨请求保活。bench 的 TTFT 因此覆盖完整 setup→首 token 路径。
- **请求超预算保护**：`len + gen > kv_hot_tokens` 直接 `ValueError`（HOST_STREAM 长上下文是未来工作，有明确信息而非静默截断）。
- **DISK-dense × MoE**：`runner_streamed` 仅 dense；MoE 引擎只支持 HOST dense，`DISK` 分支明确报错。

## 9. 相关文档

- [planning.md](planning.md) —— Plan 从哪来、池预算语义
- [exec.md](exec.md) —— prefill/decode runner 的实际计算
- [store.md](store.md) —— dense/专家池的实例化
- [weights.md](weights.md) —— 模型目录与 layout 格式
