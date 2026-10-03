# weights —— 权重格式与转换（layout / 磁盘源 / HF 转换器）

> 对应包：`src/plastic_infer/weights/` · 设计依据：DESIGN.md §3.1、M4 转换器里程碑

## 1. 职责

`weights` 定义权重在**磁盘上怎么摆**（layout）、怎么**读**（磁盘/内存源），以及怎么从 HuggingFace 检查点**转换**成这套格式。它是 store/exec 的底层供给：`DiskLayerSource.layer()` 给出的 dense 权重命名，必须与 `exec.ModelConfig` 的取数键严格一致；每层/每专家的字节数，必须被规划器实测引用。

## 2. 磁盘格式约定

转换后的模型目录：

```
out_dir/
  config.json                    # 拷贝自 HF（引擎读 head_dim/rope_base/…）
  tokenizer.json / tokenizer_config.json
  layout.index.json              # LayoutIndex 序列化（见 §3）
  model.shared.safetensors       # embed_tokens / norm / lm_head
  model.layers.{L}.safetensors   # L=0..47；每层一个文件
```

层文件内张量（**规范命名**）：

```
layers.{L}.input_layernorm.weight
layers.{L}.self_attn.{q,k,v,o}_proj.weight
layers.{L}.self_attn.q_norm.weight / k_norm.weight   # Qwen3 独有
layers.{L}.post_attention_layernorm.weight
layers.{L}.mlp.router.weight                          # 改名自 HF 的 mlp.gate.weight
layers.{L}.mlp.experts.{eid}.w1.weight  # = gate 投影
layers.{L}.mlp.experts.{eid}.w2.weight  # = down 投影
layers.{L}.mlp.experts.{eid}.w3.weight  # = up 投影
```

**专家不融合**：HF 的 fused `gate_up_proj [128, 2I, H]` + `down_proj [128, H, I]` 被按行切成 128 个独立 `(w1=gate前半, w3=up后半, w2=down)` 二维张量。这一选择让 `DiskExpertSource.expert()` 只触碰单个专家的字节（safetensors 按名+切片读，见 §4），代价是每层多 128 个文件内张量。

## 3. Layout（`layout.py`）

```python
@dataclass
class TensorLocation:
    file: str      # 所在文件
    dtype: str
    shape: list[int]

@dataclass
class ExpertLocation:
    file: str
    w1_name: str; w2_name: str; w3_name: str
    @property total_bytes

class LayoutIndex:
    # __init__(layer_prefix, expert_prefix, per_expert_tensors, ...)
    def tensor(name) -> TensorLocation
    def layer_file(layer) / shared_file() / shared_tensor_names()
    def layer_tensor_names(layer) / expert_tensor_names(layer, eid)
    def is_moe() / num_experts(layer)
    def expert_total_bytes(layer) / dense_per_layer_bytes(layer)
    @property total_dense_bytes / total_expert_bytes
    def to_dict() / save(path) / from_dict(d) / load(path)
```

`LayoutIndex` 由 `build_layout_from_manifest(layer_prefix, expert_prefix, per_expert_tensors, ...)` 从转换输出构建，可序列化成 `layout.index.json` 并重新加载。它向规划器/引擎暴露三个关键量：**每层 dense 字节**、**每层专家总字节**、**专家张量名** —— 引擎据此实测算 `ModelMeta`（见 [runtime.md](runtime.md)）。

## 4. 磁盘 / 内存源（`disk.py`）

| 类 | 读取方式 | 说明 |
|----|----------|------|
| `DiskLayerSource(model_dir, layout)` | `layer(layer)` → `safe_open(layer_file)[...]` | **排除专家张量**（`if ".experts." not in name`）；`shared()` 读 embed/lm_head |
| `DiskExpertSource(model_dir, layout)` | `expert(layer, eid)` → 按 `expert_tensor_names` 三张量读 | 每专家独立读 |
| `DictExpertSource(bank)` | 从 `ExpertBank` 取 | host 全常驻锚（测试/小模型） |
| `DictLayerSource(weights, n_layers)` | 从内存 dict 取 | host dense 锚 |

实现要点：safetensors 0.8 的 `safe_open` 默认 mmap，`get_tensor(name)` 按名读取只触碰所需字节 —— 单专家读取不会把整层页调入内存，冷启动的 first-touch 按需进行（这是热/冷 TTFT 差异 264 s 的来源之一，见 [runtime.md](runtime.md) §7）。

## 5. 转换器（`convert.py`）

### 纯函数（供测试复用）

```python
def map_hf_tensor(hf_name: str) -> str | None
    # HF 名 → 规范名；非本仓覆盖的张量返回 None（如 rotary_emb）
    # 含改名：mlp.gate.weight → mlp.router.weight；
    # 含专家识别：{...}.mlp.experts.{eid}.{gate,up,down}_proj.weight → w1/w2/w3

def split_expert(fused_gate_up, fused_down, eid) -> ExpertWeights
    # 从 fused [128, 2I, H] 切第 eid 行：w1=gate 前半、w3=up 后半、w2=down
```

### 入口

```python
def convert(hf_dir, out_dir, dtype=torch.bfloat16):
    # 读 model.safetensors.index.json 定张量→分片；safe_open（mmap）缓存句柄；
    # 逐层：dense 按名读 + 专家逐行切 → save_file 写层文件；
    # 最后写 model.shared.safetensors、layout.index.json、拷贝 config/tokenizer

def convert_from_dict(flat_state_dict, config_dict, out_dir, dtype=torch.float32):
    # 测试用入口：同一管线，输入内存 dict（等价 HF state_dict），让单测跑真实转换代码

def split_state_dict(flat, num_experts, ...):
    # 把 fused 专家展平为每专家张量（供 runner 测试用）
```

`_DictSource` / `_ShardSource` 抽象了「输入来自 dict 还是多分片磁盘」，`_convert_layers` 对二者跑同一套逐层逻辑。单层峰值 ~2.5 GB（bf16 48 层全尺寸），可接受。

### HF → 规范的命名映射（Qwen3Moe）

| HF state_dict | 本仓规范名 | 处理 |
|---------------|-----------|------|
| `model.embed_tokens.weight` | `embed_tokens.weight` | 直写 shared |
| `model.layers.{L}.mlp.gate.weight` | `layers.{L}.mlp.router.weight` | **改名** |
| `model.layers.{L}.mlp.experts.gate_up_proj.weight` | 每个 `experts.{eid}.w1.weight` + `w3.weight` | 按行切（gate 前半 / up 后半） |
| `model.layers.{L}.mlp.experts.down_proj.weight` | 每个 `experts.{eid}.w2.weight` | 按行切 |
| `model.layers.{L}.self_attn.q_norm.weight` | `self_attn.q_norm.weight` | 直写（Qwen3 独有） |
| `model.layers.{L}.input_layernorm.weight` 等 | 同名 | 直写 |
| `lm_head.weight` / `norm.weight` | 同名 | 直写 shared |
| `rotary_emb.*`、无 tie embed 时无对应 | — | `map_hf_tensor` 返回 None，跳过 |

## 6. 真实规模参考（Qwen3-30B-A3B）

- 转换产物 **57 GB**，49 个 safetensors 文件（1 shared + 48 层），与 HF 源**字节级一致**（M5 验收）。
- 每层：dense ≈ 60 MB + 128 专家 ≈ 1.2 GB；专家总量 ≈ 57.6 GB。
- layout 加载即可运行，无需重扫权重。

## 7. 相关文档

- [store.md](store.md) —— `LayerSource`/专家源如何被池消费
- [exec.md](exec.md) —— 规范权重名如何被 runner 取用
- [planning.md](planning.md) —— layout 实测字节如何进入 `ModelMeta`
- [runtime.md](runtime.md) —— `convert` 子命令与引擎加载
