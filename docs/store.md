# store —— 常驻管理（LRU/pin 内核、dense 窗口、专家三层池）

> 对应包：`src/plastic_infer/store/` · 设计依据：DESIGN.md D5/D7、§3.3

## 1. 职责

`store` 是系统的**数据驻留层**：把规划层给出的字节预算，实现为三个可用的池 —— dense 旋转窗口、GPU 专家槽 LRU、host 专家 LRU（磁盘兜底）。它回答「某个权重现在在不在显存、不在的话去上一层取」这个问题，并以 **pin/refcount** 保证被正在计算的权重不会被逐出。

## 2. 文件地图

| 文件 | 内容 |
|------|------|
| `tiers.py` | 转发 [planning.md](planning.md) 的 `Tier` 枚举 |
| `resident.py` | 通用 LRU 内核：`Page`、`LruPool`（pin/refcount、只逐出未 pin 项） |
| `weights.py` | `LayerSource` 协议 + `DenseWindow`（dense 旋转窗口） |
| `experts.py` | 专家三层：`ExpertWeights`、`ExpertBank`、`HostExpertLru`、`ExpertSlotPool` |

## 3. LRU 内核（`resident.py`）

### `Page[K]`

```python
@dataclass
class Page(Generic[K]):
    key: K
    value: Any
    bytes_: int
    pinned: int        # refcount：>0 时禁止逐出
    tier: Tier
    last_access: int   # LRU 时间戳
```

### `LruPool[K]`（D7 不变式所在）

| 方法 | 语义 |
|------|------|
| `add(key, bytes_, tier)` | 入池，先 `evict_to_fit` 腾位 |
| `__getitem__(key)` | 取页并 `_touch` 更新 LRU 次序 |
| `pin(key)` / `unpin(key)` | 引用计数 ±1；pin 住表示「正在被计算」 |
| `evict_lru(need_bytes)` / `evict_to_fit(bytes_)` | 从 LRU 尾端逐出，**跳过 pinned 页** |
| `set_tier(key, new_tier)` | 页所在层迁移（GPU⇄host）的记账 |
| `all_pinned()` | 全部页都被 pin（诊断/断言用） |

**D7 不变式**：只逐出 `pinned == 0` 的页。正在参与计算的权重一旦被 `pin`，LRU 无论如何都不会把它挤走 —— 计算与驱逐天然隔离，单线程内无需锁。

## 4. dense 旋转窗口（`weights.py`）

```python
class LayerSource(Protocol):
    def layer(self, layer_idx) -> dict[str, torch.Tensor]: ...
    # 磁盘/host 实现见 weights.md；DictLayerSource 作 host 锚

class DenseWindow:
    def __init__(self, source: LayerSource, budget_bytes, n_layers):
        # 预算 = dense_window_bytes；W = budget // 每层字节
    def ensure(self, layer_idx) -> dict[str, torch.Tensor]   # 缺层则从 source 载入、按预算逐出
    def release(self, layer_idx)                             # 显式释放一层
    @property
    def miss_rate(self) -> float
```

设计上是「按层旋转的窗口」，供 `runner_streamed`（dense-only 流式，M3）使用。**注意**：MoE 生产路径（Qwen3）不走窗口 —— `Engine` 把 dense 全量常驻（见 [runtime.md](runtime.md) §4.3），本窗口目前仅服务 dense 流式分支与单测。

## 5. 专家三层池（`experts.py`）

一个专家的数据描述 `ExpertWeights(w1=gate, w2=down, w3=up)`，`bytes` 属性 = 三层权重总字节。三层由「三个都鸭子类型兼容 `bank` 的源」串联：

```
DiskExpertSource ──► HostExpertLru ──► ExpertSlotPool ──► _moe_ffn 计算
   (磁盘)               (host LRU)        (GPU 槽 LRU)
```

### `ExpertBank`

内存 dict 锚：`add(layer, eid, weights)` / `__getitem__` / `bytes` / `keys` / `total_bytes`。小模型（全专家进 host）时直接用它，等价于「HOST 常驻，无磁盘层」。

### `HostExpertLru`（HOST_FIRST 的第三层）

```python
def __init__(self, source, budget_bytes, per_expert_bytes):
    # budget = plan.expert_host_slots × per_expert_bytes

def __getitem__(self, key: tuple[int, int]) -> ExpertWeights:
    # 命中 → hit；未命中 → source.expert(...) 读盘 → evict_to_fit 腾位 → 入池 → miss
```

完全鸭子类型兼容 `ExpertBank`（`__getitem__/bytes/__contains__/keys/__len__`），因此 `ExpertSlotPool` 无需感知自己背后是内存 dict 还是磁盘 LRU。预算小于单个专家时 `ensure` 会断言（不静默溢出）。`hit_rate` 属性暴露命中率。

### `ExpertSlotPool`（GPU 槽，懒加载）

```python
def __init__(self, bank, budget_bytes, *, device, ...)
    # bank = ExpertBank 或 HostExpertLru（鸭子类型）

def ensure(self, layer, expert_ids) -> ExpertWeights:
    # 已在 GPU 槽 → 命中；否则 _load：bank[...] → 拷贝 GPU → evict_to_fit → 入槽
    # 返回前 pin
def release(self, layer, expert_ids):   # unpin 放回，LRU 可见
@property
def hit_rate(self) -> float
```

**三层自动成立**：`_load` 只依赖 `bank[key]`，而 `bank` 可以是「内存」或「host LRU（磁盘兜底）」。Qwen3 上即 `ExpertSlotPool(host_lru, ...)`，一次 `ensure` miss 时：host 命中 → 拷贝 GPU；host miss → 磁盘读 → 进 host LRU → 拷贝 GPU。

## 6. 数据流：一次专家取数

```
_moe_ffn 需要专家 eid（exec/runner_moe.py）
  └─ ExpertSlotPool.ensure(L, [eid])
       ├─ GPU 槽命中 → pin → 直接计算
       └─ miss → HostExpertLru[L,eid]
            ├─ host 命中 → hit
            └─ miss → DiskExpertSource.expert(L,eid) 读盘 → 入 host LRU（逐出最冷）
       → 拷贝 GPU → 入槽（evict_to_fit 逐出未 pin 的最冷槽）→ pin
  计算 → ExpertSlotPool.release(L, [eid])  → unpin
```

真机（Qwen3）数字：expert GPU hit 0.820、host LRU hit 0.032、磁盘读 2293 次 —— 绝大多数专家稳定驻留 GPU，其余多数命中 host，少数落盘。

## 7. 设计约定与现状

- **D7 逐出只碰未 pin 页**：见 §3，是 `_moe_ffn`「ensure→计算→release」循环能安全工作的前提。
- **鸭子类型的三层**：`ExpertSlotPool` 不感知 host 层实现，替换 host 源（`ExpertBank` ↔ `HostExpertLru`）零改动。
- **host LRU 的 pin 语义**：host 层单线程、计算期间专家已被 GPU 槽 pin，因此 host 层不 pin 也安全。
- **`DenseWindow` 与 MoE 生产路径的差异**：MoE 引擎用全量常驻而非窗口，窗口目前只在 dense 流式分支生效（见 §4）。

## 8. 相关文档

- [planning.md](planning.md) —— 各池预算的语义
- [runtime.md](runtime.md) —— 池如何被 engine 实例化与串联
- [exec.md](exec.md) —— `_moe_ffn` 如何消费 `ExpertSlotPool`
- [weights.md](weights.md) —— `LayerSource` 的磁盘实现
