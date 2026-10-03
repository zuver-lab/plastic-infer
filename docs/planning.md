# planning —— 规划层（设备画像 + 内存规划 + 池预算）

> 对应包：`src/plastic_infer/planning/` · 设计依据：DESIGN.md §3.3（GPU 池优先级）、§3.4

## 1. 职责

在**每个请求开始前**，把「设备硬件 + 模型体积 + 请求规模」三份输入综合成一份 `Plan`：决定四块 GPU 显存各自分多少 —— 激活 / KV 页池 / 专家槽池 / dense 窗口 —— 以及 dense 与专家分别放哪一层（GPU / host / 磁盘）。`planning` 是纯决策层，**不碰显存**；产出的 `Plan` 交给 [runtime](runtime.md) 去实例化各池。

## 2. 文件地图

| 文件 | 内容 |
|------|------|
| `tiers.py` | `Tier` 枚举与冷热比较 |
| `profile.py` | `DeviceProfile`：本机硬件画像（显存/内存/带宽） |
| `budget.py` | `ModelMeta` / `RequestMeta` / `Plan` / `MemoryPlanner.plan`（核心） |
| `wiring.py` | 把 `Plan` 拆成每个池的字节预算：`PoolBudgets` |

## 3. 分层模型（`tiers.py`）

```python
class Tier(Enum):
    GPU   = "gpu"      # 显存
    HOST  = "host"     # CPU 内存
    DISK  = "disk"     # 磁盘
    DROPPED = "dropped"  # 规划外丢弃

TIER_ORDER = (Tier.GPU, Tier.HOST, Tier.DISK)
tier_colder(a, b) -> bool   # 比较两层谁更冷
```

三层按「容量递增、带宽递减」排序：GPU 最快最贵、磁盘最慢最廉。一切 placement 决策都表达为「把某个张量集放到哪个 Tier」，由 [store](store.md) 层执行。

## 4. 设备画像（`profile.py`）

```python
@dataclass(frozen=True)
class DeviceProfile:
    hbm_bytes: int      # 显存总量
    host_ram_bytes: int # 内存总量
    pin_limit_bytes: int  # 可用作 host 权重驻留的上限（一般 hbm//2）
    b_h2d: float        # 显存↔内存带宽（字节/秒）
    b_host: float       # 内存带宽
    b_disk: float       # 磁盘带宽
```

CLI 的默认画像（`runtime/cli._default_profile`）：`pin_limit = hbm // 2`，`b_h2d = 12e9`、`b_host = 30e9`、`b_disk = 7e9`（字节/秒量级）。带宽数字目前只作信息展示，规划器实际按**容量**决策。

## 5. 输入元数据（`budget.py`）

```python
@dataclass(frozen=True)
class ModelMeta:
    n_layers: int
    dense_bytes: int          # 全量 dense（含 embed/lm_head）字节
    per_expert_bytes: int     # 单个专家字节
    n_experts: int            # 每层专家数
    dense_per_layer_bytes: int
    kv_bytes_per_token: int   # 每 token 全层 KV 字节
    # 派生：total_expert_bytes / total_weight_bytes / max_expert_row / total_experts

@dataclass(frozen=True)
class RequestMeta:
    seq_budget: int           # 请求承诺的序列预算
    gen_tokens: int           # 生成步数
```

`kv_bytes_per_token = n_layers * 2 * n_kv_heads * head_dim * dtype_bytes`，由引擎 `build_model_meta` 从 layout 实测字节算出（见 [runtime.md](runtime.md)）。

## 6. `Plan` 与四池优先级

`MemoryPlanner.plan(profile, model, req) -> Plan`（`budget.py:150`），`Plan` 的核心字段：

| 字段 | 含义 |
|------|------|
| `seq_weight_source` | dense 所在层：`"HOST"`（常驻）或 `"DISK"`（窗口流式） |
| `weight_window` | dense 窗口层数 W |
| `kv_hot_tokens` | GPU KV 页池可容纳的 token 数 |
| `expert_slots_bytes` | GPU 专家槽池预算 |
| `expert_host_slots` / `experts_fit_in_host` | host 层是否放得下全部专家 |
| `gpu_total_bytes` | 规划占用总和（应 ≈ usable） |

**切分顺序（高优先级先保证，DESIGN.md §3.3）：**

```
usable = hbm × (1 - safety=0.08) × (1 - host_os_reserve_ratio=0.15)
1. 激活保留   → max(dense_per_layer_bytes, hidden×seq_budget)，上限 usable//4
2. KV 页池    → (remaining − dense_min_bytes) // 2，封顶使 kv_hot_tokens ≥ kv_reserve_tokens(2048)
3. 专家槽池   → remaining − dense_min_bytes
4. dense 窗口 → 剩余部分，W ≥ 2 才启用窗口；否则全量常驻
```

关键约束：

- **safety=0.08**：HBM 只敢用 92%，给驱动/页表/运行时留余量。
- **host_os_reserve_ratio=0.15**：host RAM 只计划 85%。
- **dense_min_bytes**：无论优先级怎么挤，dense 至少留 W≥2 的窗口或全量常驻所需的字节。
- **激活优先于 KV**：`hidden×seq_budget` 是单层激活下限，防止 KV 池吞光计算所需的中间张量空间。
- 各池上限均 **≤ usable**，即规划输出永远不会要求超过「可用显存」的总和。

## 7. 池预算（`wiring.py`）

```python
@dataclass(frozen=True)
class PoolBudgets:
    dense_window_bytes: int
    expert_slots_bytes: int
    kv_pool_bytes: int
    kv_pages: int          # = kv_pool_bytes // (bytes_per_page)
```

`pool_budgets(plan, model) -> PoolBudgets` 把 `Plan` 的语义字段换算成每个池构造时直接可用的字节数。其中 `kv_pages = kv_hot_tokens // PAGE_TOKENS(16)`，是**每层**需要的页数；引擎构造 KVConfig 时传 `max_pages = kv_pages × n_layers` 作为全部物理页（一页 = 某一层的 16 token K+V，见 [kv.md](kv.md)）。

> **引擎侧修正（重要）**：规划器假设 dense 用「旋转窗口」按需进显存；而 `Engine` 目前把 dense **全量常驻**（HOST 源一次读入 GPU，M5 实测 2.88 GiB）。为不超预算，engine 把该超差从专家槽预算中扣掉（见 [runtime.md](runtime.md) §4.3）。规划器的 W 值此时仅反映「dense 应该留多少」，实际池以 engine 修正后为准。

## 8. 真实模型样例（Qwen3-30B-A3B，24 GB）

| 池 | 规划值 | 引擎实际 | 说明 |
|----|--------|----------|------|
| 激活 | ~0.19 GiB | — | 上限 usable//4 |
| KV 页池 | 3.75 GiB（40992 tokens） | 3.75 GiB | 预分配整块 |
| 专家槽池 | 18.0 GiB / 2050 槽 | **15.26 GiB** | 扣除 dense 常驻超差 ≈2.8 GiB |
| dense 窗口 | W=2（0.12 GiB） | **2.88 GiB**（全量常驻） | 引擎行为 ≠ 规划假设 |
| **合计** | **22.09 GiB** | **≈24.25/24.58 GiB 峰值** | usable = 24×0.92 |

`seq_weight_source="HOST"`（dense 3.5 GB ≪ host 52 GB），专家 `HOST_FIRST`（57.6 GB > host 可用 → host LRU + 磁盘溢出）。

## 9. 相关文档

- [runtime.md](runtime.md) —— `Engine` 如何调 `plan` → `pool_budgets` → 建池
- [store.md](store.md) —— 各池预算如何被 `ExpertSlotPool`/`DenseWindow`/`KVPagedStorage` 消费
- [kv.md](kv.md) —— `kv_pages` 与预分配的对应
