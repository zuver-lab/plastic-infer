# kv —— 分页 KV 存储与 chunk 前缀缓存

> 对应包：`src/plastic_infer/kv/` · 设计依据：DESIGN.md §5.5、D9/D10

## 1. 职责

`kv` 负责**注意力 KV 的物理管理**：以固定大小的页为单位在 GPU 上预分配一块显存，为每个请求用块表（BlockTable）映射逻辑 token → 物理页；同时把「已完成的 chunk」下沉到 host LRU，并用**前缀哈希**支持 prefix 复用（D9/D10）。它把 KV 从「每个请求一个变长数组」提升为「可分配、可回收、可跨请求复用的页池」。

## 2. 文件地图

| 文件 | 内容 |
|------|------|
| `paged.py` | 物理页分配：`PageAllocator`（页的分配/回收）、`BlockTable`（层×逻辑页→物理页映射） |
| `chunk.py` | chunk 前缀哈希：`ChunkKey`、滚动哈希、`longest_prefix_hits` |
| `kv_store.py` | 总装：`KVPagedStorage`（显存页张量）、`KVRequestCache`（单请求上下文）、`KVStore`（池 + host 下沉 + prefix 加载） |

## 3. 基本参数

```
PAGE_SIZE    = 16   # 每页 token 数（paged.py）
CHUNK_PAGES  = 16   # 每个 chunk 的页数 = 256 tokens / chunk
```

## 4. 物理页（`paged.py`）

### `BlockTable`

```python
@dataclass
class BlockTable:
    n_layers: int
    pages_per_chunk: int        # = CHUNK_PAGES
    layer_pages: list[list[int]]  # layer -> [逻辑页 → 物理页 id]
```

核心查询：

| 方法 | 语义 |
|------|------|
| `get_page_for_token(layer, token_idx)` | 逻辑 token → 物理页 id |
| `append_page(layer, phys_page_id)` | 给某层追加一个物理页 |
| `chunk_page_ids(chunk_idx, pages_per_chunk)` | 第 n 个 chunk 覆盖的物理页 id 列表 |
| `is_chunk_complete(chunk_idx, pages_per_chunk)` | 该 chunk 是否已写满（可下沉） |

> 注意 `get_page_for_token` 与 `get_page(layer, logical_page_idx)` 的区别：前者按**全局 token 下标**取页，后者按**层内逻辑页号**取页。`BlockTable` 的各层页列表可长短不一（decode 阶段各层同步增长，但接口允许分层）。

### `PageAllocator`

`total_pages` 个物理页的自由表：`alloc()/alloc_many(count)/free()/free_many()`，`free_pages/used_pages` 统计。物理页 id 是**全局**的：所有层共用同一个页池，一页固定承载「某一层的 16 个 token 的 K+V」。页池总页数 = 每层页数 × 层数（engine 以 `kv_pages × n_layers` 传入，见 [runtime.md](runtime.md) §4.2）。

## 5. chunk 前缀哈希（`chunk.py`）

把 token 流切成 chunk，为每个 chunk 算一个滚动哈希（`compute_prefix_hashes_and_final` 返回每个 chunk 的 `prefix_hash` 与最终 `final_hash`）。`ChunkKey`（`prefix_hash` + `offset`）作为 host 缓存的键，`longest_prefix_hits(hashes, cached_keys)` 求最大可复用前缀长度。

这套机制服务于 D9/D10 的「prefix 缓存」设计：prompt 与历史请求共享前缀时，只加载不重叠的尾部 chunk。**当前 engine 路径尚未调用**（见 §7）。

## 6. 总装（`kv_store.py`）

### `KVConfig`

```python
@dataclass(frozen=True)
class KVConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    page_size: int = PAGE_SIZE
    # 派生：bytes_per_page = n_kv_heads * head_dim * page_size * 2 * dtype_bytes
    #        （单层单页的字节数，K+V；不含 n_layers）
    #      tokens_per_chunk = page_size * CHUNK_PAGES
```

### `KVPagedStorage`

```python
def __init__(self, config, total_pages, device):
    # 预分配整块：torch.zeros((total_pages, n_kv_heads, head_dim, page_size))
    # 物理布局：页主序，每页内 (kv_heads × head_dim × page_size) 连续
```

`get_page_k(page_id)` / `get_page_v(page_id)` 返回某物理页的 k/v 视图。**一次性预分配**保证了显存峰值可控（Qwen3 上 KV 池 3.75 GiB = 40992 tokens，见 [planning.md](planning.md)），页的分配/回收只是逻辑记账，不触碰显存。

### `KVRequestCache`

单个请求的 KV 上下文：持有一个 `BlockTable` 与每层「当前页内偏移」；`append_token(k_token, v_token)` 写入当前页、满页时 `alloc` 新页；`set_token_ids` 记录 token 序列并重算 chunk 键。

### `KVStore`

```python
class KVStore:
    def __init__(self, config):        # 建 KVPagedStorage + 页分配器 + host LRU
    def new_request(request_id=0) -> KVRequestCache
    def free_request(cache)            # 归还全部物理页
    def sink_completed_chunks(cache) -> int   # 已写满的 chunk 下沉 host LRU，返回释放的页数
    def load_prefix(cache, tokens)     # 命中前缀缓存时预填充（D9）
    def evict_chunks(need_pages)       # host LRU 回吐 / 页池扩容时的逐出
```

host 影子副本：完整 chunk 在 GPU 页写满后，其 k/v 会拷贝进 `_host_cache`（LRU，`host_cache_size` 可查），显存不足时逐出对应 GPU 页 —— 这是「长上下文 KV 放 host」的落点。

## 7. 设计约定与现状

- **预分配一整块页池**：显存峰值在规划期就锁定，运行期零分配。代价是容量上限固定为 `kv_pool_bytes / bytes_per_page`。
- **`sink_completed_chunks` 已实现但未接线**：engine 的 `prefill/decode` 路径当前直接经 `KVRequestCache.append_token` 写 GPU 页，chunk 下沉与 `load_prefix` 均未在运行期触发 —— 属 D9/D10 的存量实现、运行期待接入。
- **超预算保护在 engine 层**：请求长度超 `kv_hot_tokens` 时 engine 明确报错（见 [runtime.md](runtime.md)），`KVStore` 本身不拒绝。

## 8. 相关文档

- [exec.md](exec.md) —— `gather_paged_kv` / `paged_attention_v1` 如何消费页张量
- [planning.md](planning.md) —— KV 池字节预算与 `kv_pages` 的由来
- [runtime.md](runtime.md) —— engine 如何构造 KVConfig 并按请求新建/回收缓存
