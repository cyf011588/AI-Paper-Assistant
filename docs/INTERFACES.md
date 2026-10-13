# 层间数据契约（INTERFACES）

> 目的：让「不同人、不同机器、各自方向」并行优化时**互不打架**。
> 规则只有一条：**层内随便改，跨层只认本文件定义的字段与语义。**
> 只要不动契约，你改 `smart_retriever.py` 的算法，别人改 `pdf_parser.py` 的解析，两边无感。

---

## 0. 五层划分与归属

| 层 | 职责 | 主要文件 | 产物 |
|---|---|---|---|
| **L1 语料与解析** | PDF → 结构化 → 分块 | `pdf_parser.py`、`chunker.py`、`multimodal_parser.py` | `data/processed/*_structured.json`、`*_chunks.json` |
| **L2 向量索引** | 文本 → 向量库 | `vector_store.py`、`vision_extractor.py` | `data/chroma_db/`、`data/cache/` |
| **L3 检索** | 问题 → 候选片段 | `smart_retriever.py`、`reranker.py`、`query_understanding.py` | 见契约 ② |
| **L4 生成** | 片段 → 带引用回答 | `generator.py`、`writing_agent.py` | 见契约 ③ |
| **L5 交互与持久化** | 界面、会话 | `app.py`、`session_store.py` | `data/session.db` |
| 横切 | 配置 / 脚本 / 评测 | `config.py`、`scripts/`、`evaluation/` | — |

数据流：`L1 → L2 → L3 → L4 → L5`，逐层通过下面三张契约传递。

---

## ① Chunk 契约（L1 → L2 / L3）

**定义位置**：`src/chunker.py` 的 `@dataclass Chunk`
**持久化形式**：`data/processed/{paper}_chunks.json`，是一个 `Chunk` 列表的 JSON 数组
**消费方式**：`Chunk(**c)` 直接反序列化（`evaluate_v2.py` / `scripts/build_index.py` 均如此）

| 字段 | 类型 | 必有 | 语义 |
|---|---|---|---|
| `id` | str | ✅ | 全局唯一。格式 `{paper_name}_{section_path}_{chunk_index:04d}` |
| `text` | str | ✅ | **子块**文本（用于 embedding 与检索，小而精） |
| `paper_name` | str | ✅ | 来源论文名（与 `data/paper_metadata.json` 的 `paper_name` 对齐） |
| `section_path` | str | ✅ | 章节路径，层级用 `" / "` 连接，如 `第五章 / 5.1 试验概况` |
| `page_start` | int | ✅ | 起始页码（1-based） |
| `page_end` | int | ✅ | 结束页码 |
| `parent_id` | str | ✅（可空串） | 父块 ID，格式 `{paper_name}_parent_{parent_index:04d}` |
| `parent_text` | str | ✅（可空串） | **父块**全文（= 整个小节）；命中后作为给 LLM 的上下文 |
| `block_types` | List[str] | ✅（可空列表） | 组成该块的原始块类型，如 `HEADING`/`TEXT`/`FIGURE_CAPTION` |

**修改约定**

- ✅ **可以**改切分策略、`chunk_size`、父子聚合方式——只要字段不变。
- ⛔ **不要**删除或改名任何字段；新增字段**必须带默认值**（否则旧 JSON 反序列化 `Chunk(**c)` 会报错）。
- ⛔ **不要**改变 `id` 的粒度语义：评测集的 `gold_chunk_ids` 直接指向 `id`，改了 = 历史评测全部失效。
- ⚠️ 改了分块 → **必须重跑评测**（见 `docs/COLLABORATION.md`），因为 Recall 分母变了。

---

## ② 检索结果契约（L3 → L4）

**定义位置**：`src/smart_retriever.py` 的 `SmartRetriever.search()`

```python
search(question: str, top_k: int = TOP_K, allow_decompose: bool = True) -> dict
```

返回结构：

```jsonc
{
  "results": [ RetrievedChunk, ... ],   // 已按相关性排序
  "query_info": QueryInfo | null,        // 查询理解结果
  "debug_info": DebugInfo | null         // 诊断信息（评测/归因用）
}
```

### RetrievedChunk（`results` 的元素）

| 字段 | 类型 | 语义 |
|---|---|---|
| `id` | str | 对应契约 ① 的 `id` |
| `text` | str | ⚠️ **L3 出口处已被替换为父块全文 `parent_text`**（给 LLM 看完整上下文） |
| `original_text` | str | 仅当发生父块替换时存在，保留原始**子块**文本 |
| `paper_name` / `section_path` / `page_start` / `page_end` | 同契约 ① | — |
| `score` | float | 相关性得分；扩展进来的邻居块固定为 `0.5` |
| `parent_id` / `parent_text` | 同契约 ① | — |
| `group_paper` | str | 仅 `comparison_decompose` 模式下存在，标记该块属于哪个子问题 |

> **关键语义**：`text` 在 L3 出口是「父块」，不是子块。谁在 L3 之后想拿原始子块，请读 `original_text`。
> 改检索算法时**必须保持这一语义**，否则 L4 的上下文构造会静默退化。

### QueryInfo（`query_info`，来自 `query_understanding.parse()`）

| 字段 | 类型 | 语义 |
|---|---|---|
| `target_papers` | List[str] | 问题中明确指向的论文名（可能为空 = 全库检索） |
| `content_query` | str | 去掉论文/作者引用后的「真正内容问题」 |
| `is_paper_specific` | bool | 是否限定到具体论文 |

### DebugInfo（`debug_info`）

| 字段 | 类型 | 语义 |
|---|---|---|
| `retrieval_mode` | str | ∈ `hybrid_rerank` / `section_shortcut` / `comparison_decompose` / `section_list` |
| `pre_expansion_ids` | List[str] | 扩展前 Top-N 的 chunk id（**评测的 Strict Recall@10 就算在它上面**） |
| `post_expansion_ids` | List[str] | 章节聚合 + 邻居扩展后的全部 id |
| `pre_rerank_ids` / `post_rerank_ids` | List[str] | RRF 融合后 / reranker 精排后的 id（部分分支提供） |

**修改约定**

- ✅ 可以换 embedding、换 reranker、改 RRF 权重、加查询改写——只要返回结构不变。
- ⛔ 不要改 `results` 的字段名；不要改 `debug_info.retrieval_mode` 的取值集合（评测脚本按它分类）。
- ⛔ 不要改变 `extraction_mode` 之外 `pre_expansion_ids` 的含义——它就是 Strict Recall@10 的分子来源。

---

## ③ 引用契约（L4 → L5）

**定义位置**：`src/generator.py` 的 `Generator.generate()`

```python
generate(question: str, contexts: List[dict]) -> dict
```

```jsonc
{
  "answer": "……正文……",
  "citations": [ Citation, ... ]
}
```

### Citation

| 字段 | 类型 | 语义 |
|---|---|---|
| `ref_id` | str | 展示序号，格式 `"[1]"`、`"[2]"` |
| `paper` | str | 论文名 |
| `section` | str | 章节路径（L5 展示时截断到 60 字符） |
| `page` | int | 页码 |

> `answer` 正文里的 `[n]` 编号已被正则清洗（`re.sub(r'\[\d+\]', '', ...)`），引用统一由结构化的 `citations` 承载——**不要**在正文里再拼引用号。

**修改约定**

- ✅ 可以改 Prompt、改 temperature、加多模态看图、加写作 Agent。
- ⛔ 不要让 `citations` 的字段缺项：`app.py` 和 `session_store.py` 都直接按这 4 个 key 取值。

---

## ④ 向量库元数据契约（L2，附）

**定义位置**：`src/vector_store.py` 的 `add_chunks()`

- `documents[i] = f"[{section_path}] {text}"`（**带章节前缀**，检索时依赖它）
- `ids[i] = chunk.id`
- `metadatas[i] = { paper_name, section_path, page_start, page_end, parent_id, parent_text[:2000] }`

> `parent_text` 入库时**截断到 2000 字符**。若你依赖更长的父块，请连同这一截断一起改，否则会出现「离线有、检索没」的诡异现象。

---

## ⑤ 契约变更流程

任何**破坏性**变更（删/改名、改语义、改 id 粒度）：

1. 在本文件 `## 变更记录` 追加一条；
2. 在 `EXPERIMENTS.md` 登记为一次实验（含动机与影响面）；
3. 同步更新所有下游消费者；
4. **重跑 30 题评测**并在 PR 里贴出对比。

## 变更记录

| 日期 | 契约 | 变更 | 提出人 | 影响 |
|---|---|---|---|---|
| 2026-10-13 | ①②③ | 首次冻结（据现有实现归纳，未改动任何代码语义） | — | — |
