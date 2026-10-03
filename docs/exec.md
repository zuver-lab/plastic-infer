# exec —— 执行层（张量算子与前向/解码 runner）

> 对应包：`src/plastic_infer/exec/` · 设计依据：DESIGN.md §5.4（MoE 算子）、§5.5（KV 分页）

## 1. 职责

`exec` 是系统的**纯计算层**：它只接收「张量 + 权重容器」，产出 logits / token，**不触碰磁盘、不分配存储、不了解规划结果**。I/O（读权重、换入换出）全部由 `store` 层在 runner 的取数点完成；本层通过传入的 `DenseWeights` / `ExpertSlotPool` / KV 存储来间接感知数据所在层级。

一个层的前向 = 注意力（含 RoPE、QK-norm）+ 逐专家 FFN，其中 MoE 部分是三层存储的**唯一消费点**（compute-load 流水线的挂钩处，见 §7）。

## 2. 文件地图

| 文件 | 内容 |
|------|------|
| `runner.py` | 稠密模型：`ModelConfig`/`DenseWeights`/`DenseKVCache`/`linear` + 连续 KV 的 prefill/decode |
| `runner_paged.py` | 稠密模型 × 分页 KV：`make_kv_config` + paged prefill/decode |
| `runner_moe.py` | MoE 模型 × 连续 KV：`topk_route` 路由 + `_moe_ffn` 专家循环 |
| `runner_moe_paged.py` | MoE × 分页 KV：**生产路径**（Qwen3-30B-A3B 实际使用） |
| `runner_streamed.py` | dense 窗口流式执行（仅 dense；MoE 未接入，见 §7） |
| `attention.py` | 位置编码与注意力算子：RoPE、RMSNorm、QK-norm、分页/流式 attention |
| `moe.py` | MoE 算子：路由、稀疏散射、SwiGLU，及其参考实现 |

## 3. 核心数据类型（`runner.py`）

### `ModelConfig`

一个层中 MoE 前向需要的全部超参数（`runner.py:36`）。dense-only 模型该对象中 `n_experts=0`、`n_experts_per_tok=0`。

```python
@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    intermediate_dim: int
    vocab_size: int
    max_seq_len: int
    rope_base: float = 10000.0
    dtype: torch.dtype = torch.float32
    qk_norm: bool = False          # Qwen3 逐头 QK RMSNorm（见 §6.2）
    n_experts: int = 0             # 每层专家数；0 = 纯稠密
    n_experts_per_tok: int = 0     # top-k 路由宽度
```

### `DenseKVCache`

连续（非分页）KV 缓存，`append(layer, k_token, v_token)` 逐 token 追加，`get_slice(layer, end)` 取前缀。仅 dense runner 与 MoE×连续KV runner 使用；分页路径走 `kv` 包。

### `DenseWeights`

`dict[str, Tensor]` 的薄封装，`__getitem__(key)` 取权重。键为 `layers.{L}.self_attn.q_proj.weight`、`layers.{L}.mlp.router.weight` 这类规范名（见 [weights.md](weights.md)）。它**不区分数据所在层**——无论张量常驻 GPU 还是从 host/磁盘现取，都以同一接口暴露。

## 4. 注意力与位置编码（`attention.py`）

| 函数 | 签名要点 | 说明 |
|------|----------|------|
| `rope_positions` | `(cos, sin, positions, x)` | 把 RoPE 旋转应用到 x（q 或 k）上，返回同形张量 |
| `build_rope_cache` | `(seq_len, head_dim, base=1e4, device, dtype) -> (cos, sin)` | 预计算全量 cos/sin 缓存 |
| `rms_norm` | `(x, weight, eps=1e-5)` | 标准 RMSNorm |
| `qk_norm` | `(q, k, q_weight, k_weight, head_dim) -> (q, k)` | 对 reshape 后的 q/k **逐头**做 RMSNorm（Qwen3 独有，见 §6.2） |
| `gather_paged_kv` | `(k_pages, v_pages, block_table, page_size, num_tokens=None) -> (k, v)` | 按块表把物理页 gather 成连续 k/v 前缀（v1 回退实现，非真 paged kernel） |
| `paged_attention_v1` | `(q, k_pages, v_pages, block_table, page_size, kv_num_tokens=None, causal=True)` | 分页注意力（v1：先 gather 后 SDPA） |
| `host_stream_attention_v1` | `(q, host_chunks, resident_k, resident_v, q_start_pos=0, ...)` | 面向 host KV 流式前向的注意力（**已实现未接线**，见 §7） |

> **RoPE 布局约定**：本仓的 `build_rope_cache` 采用 transformers ≥5.14 的 **half-repeat** 布局（每 `head_dim` 前半为 cos、后半为 sin，而非逐对交替）。这是与 HF 对拍逐字节一致的关键前提，改动会静默破坏端到端等价（小模型注意力缩放可能掩盖错误）。详见记忆 [[rope-layout-convention]]。

## 5. MoE 算子（`moe.py`）

| 函数 | 说明 |
|------|------|
| `topk_route(router_logits, k=2)` | 输入 router 线性层的 logits（`[T, n_experts]`）→ top-k → **fp32** softmax → 回原 dtype（≡ Qwen3 `norm_topk_prob=True`）。返回 `(expert_ids, weights)` |
| `moe_forward_sparse(hidden, experts, expert_ids, weights)` | `experts: dict[int, ExpertWeights]`；对每个专家 `(ids==eid)` 挑行 → `swiglu_mlp` → `index_add_` 加权加回。**生产路径** |
| `swiglu_mlp(x, w1, w2, w3)` | `silu(x@w1ᵀ) * (x@w3ᵀ)` 后 `@w2`，对应切分后的 w1=gate / w3=up / w2=down |
| `moe_forward_v1` / `moe_forward_reference` / `moe_forward_reference_swiglu` | v1（逐 token dispatch）与超慢参考实现，**仅供测试对拍** |

`topk_route` 的 fp32 归一化是一个刻意的 bf16 稳定性修正：`norm_topk_prob` 语义上等于「softmax 后取 top-k 再重归一」，而在 bf16 下直接做会导致专家选择不稳；fp32 计算、结果再转回原 dtype 后，fp32 测试路径逐字节不变。

## 6. runner 矩阵

五个 runner 由两个正交维度组合：**模型是否 MoE** × **KV 是否分页**，外加一个 dense-only 的流式变体。

| runner | MoE | KV | 生产用途 |
|--------|-----|----|----------|
| `prefill_forward` / `decode_step`（`runner.py`） | ✗ | 连续 | dense 小模型、单测锚 |
| `prefill_forward_paged` / `decode_step_paged`（`runner_paged.py`） | ✗ | 分页 | dense × 长上下文 |
| `prefill_forward_moe` / `decode_step_moe`（`runner_moe.py`） | ✓ | 连续 | MoE 小模型、单测锚 |
| **`prefill_forward_moe_paged` / `decode_step_moe_paged`（`runner_moe_paged.py`）** | ✓ | 分页 | **生产路径（Qwen3-30B-A3B）** |
| `prefill_forward_streamed` / `decode_step_streamed`（`runner_streamed.py`） | ✗ | 连续 | dense 窗口流式（M3） |

每个 runner 都持有同一套 48 层主循环骨架：RMSNorm → Q 投影（+QK-norm +RoPE）→ 注意力 → KV 写回 → 后 LN → MoE/FFN。差异只在注意力取数（`DenseKVCache.get_slice` vs `gather_paged_kv`）和 FFN（稠密 `linear×3` vs `_moe_ffn`）。

### 6.1 `_moe_ffn` —— 逐专家串行循环（`runner_moe.py:46`）

```python
for eid in unique:
    exp = pool.ensure(layer_idx, [eid])   # ← 取数点：{eid: 专家权重}，命中 GPU / miss host / miss 磁盘
    try:
        moe_out += moe_forward_sparse(xf, exp, eids, routing_w)
    finally:
        pool.release(layer_idx, [eid])    # 用毕放回，LRU 可见
```

对每个被路由到的专家：先 `ensure`（经 `ExpertSlotPool` 懒加载，见 [store.md](store.md)），算完即 `release`。**每次循环只处理一个专家**，load 与 compute 严格串行——这是 compute-load 流水线尚未落地的具体体现（见 §7）。

### 6.2 QK-norm 钩子

`runner_moe.py` / `runner_moe_paged.py` 在 `q/k.view(...)`（拆头）之后、RoPE 之前插入：

```python
if config.qk_norm:
    q, k = qk_norm(
        q, k,
        weights[f"layers.{L}.self_attn.q_norm.weight"],
        weights[f"layers.{L}.self_attn.k_norm.weight"],
        config.head_dim)
```

`qk_norm=False` 时该分支不执行，与 M1–M3 的稠密路径逐字节一致（既有测试不动）。

## 7. 设计约定与现状

- **compute-load 流水线（DESIGN.md D4/D6/§3.5/§5.4）**：设计的意图是在解码循环里用 CUDA Stream 做「下一批专家加载」与「当前批计算」重叠，`store` 的 `sink_completed_chunks`、`host_stream_attention_v1` 等已就位。**但 M0–M5 均未实现该流水线**：全仓无 `torch.cuda.Stream`、无 double buffer、无 prefetch、无 `mover.py`；`_moe_ffn` 对每个专家串行 load→compute→release。真模型上这是 TTFT/TPOT 的主要延迟来源之一（冷启动 295.6 s / 热启动 TPOT median 0.449 s，实测见 [runtime.md](runtime.md)）。
- **qk_norm**：由引擎在构造 `ModelConfig` 时按 layout 中是否存在 `self_attn.q_norm.weight` 决定。
- **RoPE half-repeat**：见 §4 约定，勿改动。
- **稀疏散射 vs 参考实现**：`moe_forward_sparse` 是生产实现，`moe_forward_*reference*` 仅供对拍，二者结果必须一致。
- **runner 单一来源**：任何新增算子改动，需同时在对应 runner 及其参考实现对拍保持逐字节一致。

## 8. 相关文档

- [kv.md](kv.md) —— runner 的 KV 取数与分页分配
- [store.md](store.md) —— `_moe_ffn` 取数点背后的三层专家池
- [runtime.md](runtime.md) —— 哪个 runner 被选用、参数如何从 config/layout 构造
- [weights.md](weights.md) —— 权重命名（`w1/w2/w3`、`router`）的来源
