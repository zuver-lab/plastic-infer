# PlasticInfer 模块文档

PlasticInfer 是一台把 **GPU 显存 / CPU 内存 / 磁盘** 当作三种容量、三种带宽来用的弹性推理内核（整体设计见 [DESIGN.md](DESIGN.md)）。本目录按 `src/plastic_infer/` 下的包结构，为每个模块提供一份独立的代码文档。

## 模块总览

```
                    ┌────────────────────────────────────────────┐
                    │              runtime（引擎 + CLI）          │
                    │   plan → pools → runner，请求生命周期      │
                    └──────┬───────────────┬────────────────────┘
                           │               │
            ┌──────────────▼──────┐  ┌────▼──────────────────┐
            │   planning（规划层） │  │   weights（权重格式） │
            │  device+model+req   │  │  layout/disk/convert  │
            │      → Plan         │  │  HF ⇄ 自定义磁盘格式  │
            └──────────────┬──────┘  └────┬──────────────────┘
                           │               │
                    ┌──────▼──────┐  ┌────▼──────────────┐
                    │  store（常驻层）│ │   kv（KV 存储）   │
                    │ 三池：dense 窗口│ │ paged 分配 +      │
                    │ 专家槽 LRU，     │ │ chunk 前缀哈希    │
                    │ host LRU/磁盘   │ │ + host 影子副本  │
                    └──────┬──────┘  └────┬──────────────┘
                           │               │
                    ┌──────▼──────────────▼──────┐
                    │        exec（执行层）       │
                    │  注意力 / RoPE / MoE 路由 / │
                    │ 5 个 runner 前向 + 解码     │
                    └────────────────────────────┘
```

## 文档清单

| 文档 | 对应包 | 职责 |
|------|--------|------|
| [exec.md](exec.md) | `exec/` | 张量算子（RoPE/QK-norm/注意力/MoE 路由）与全部前向/解码 runner |
| [kv.md](kv.md) | `kv/` | 分页 KV：页分配器、块表、chunk 前缀哈希、host 影子下沉 |
| [planning.md](planning.md) | `planning/` | 设备画像、内存规划器（四池切分）、plan→池预算 |
| [runtime.md](runtime.md) | `runtime/` | 引擎（plan→池→runner）与 CLI |
| [store.md](store.md) | `store/` | 常驻管理：LRU/pin 内核、dense 窗口、专家三层池 |
| [weights.md](weights.md) | `weights/` | 磁盘权重布局、磁盘/host 权重源、HF→自定义格式转换器 |

## 阅读建议

- 想理解一次请求如何被处理：先读 [runtime.md](runtime.md)，再按调用链向下看 [planning.md](planning.md) → [store.md](store.md) → [kv.md](kv.md) → [exec.md](exec.md)
- 想理解数据如何组织在磁盘/内存/显存：读 [weights.md](weights.md) 和 [store.md](store.md)
- 想对照设计意图与当前实现（哪些是蓝图、哪些已落地）：读 [DESIGN.md](DESIGN.md) 的 D 系列结论，各模块文档的"设计约定"小节与之对应
