# 评测集目录

> 本目录**不随仓库发布评测集**。
>
> 原因：评测集里的问题、标准答案、`gold chunk id` 都直接来自具体论文，属于论文内容的
> 衍生品。公开仓库只发布「可运行的框架」，论文与论文衍生内容由各人/各方向在本地自备。
>
> 这里只保留一个**纯格式模板** `example_dataset.json`（不含任何真实论文内容）。

## 怎么用

1. 参考 `example_dataset.json` 的字段，写自己的评测集（文件名任意，如 `my_dataset.json`）。
2. 放进本目录 `evaluation/dataset/`。
3. 跑评测：

   ```bash
   python evaluate_v2.py
   ```

   `src/config.py` 的 `EVAL_DATASET_PATH` 会自动选中本目录下的第一个 `*.json`
   （自动跳过 `example_dataset.json`）。要指定别的路径，设环境变量或在脚本里改 `config`。

## 为什么必须是「同一套题」

多人多方向并行优化时，**只有当所有人跑同一套评测集、同一个脚本**，分数才可比。
这是全项目唯一的「统一裁判」。规则见 [`docs/COLLABORATION.md`](../../docs/COLLABORATION.md)。

换语料 = 换评测集。换了之后，旧分数不可直接比较，必须在 `EXPERIMENTS.md` 里注明语料版本。

## 字段格式

顶层是一个数组，每个元素是一道题：

| 字段 | 类型 | 说明 |
|---|---|---|
| `question_id` | string | 题号，如 `"q001"`。评测结果按它对齐，**改名等于换题**。 |
| `question` | string | 问题原文（喂给检索器与生成器） |
| `question_type` | string | 题型，用于分维度统计。参考取值：`按作者问` / `按章节问` / `按内容问` / `事实细节` / `中英混合` / `跨论文对比` / `总结归纳` |
| `difficulty` | string | `简单` / `中等` / `困难`（仅用于分层统计，不影响打分） |
| `retrieval_gt.relevant_chunk_ids` | string[] | **标准答案所对应的 chunk id 列表**。检索指标 `Strict Recall@K` 就是看这些 id 有多少被召回到 Top-K。 |
| `retrieval_gt.relevant_chapters` | string[] | 相关章节号（如 `["摘要", "1.3"]`），仅作说明与人工核对用 |
| `retrieval_gt.notes` | string | 出题理由 / 判分口径备注 |
| `generation_gt.standard_answer` | string | 标准答案，生成质量打分（LLM-as-judge）的参照 |
| `generation_gt.source_chunks` | string[] | 标准答案的出处 chunk id（用于人工溯源） |
| `generation_gt.answer_type` | string | `事实型` / `对比型` / `总结型` 等 |

### `relevant_chunk_ids` 怎么填

chunk id 的生成规则见 [`docs/INTERFACES.md`](../../docs/INTERFACES.md)（契约一）。
格式是：

```
<论文文件名去扩展名>_<章节路径，各级用 " / " 连接>_<序号4位>
```

例如本地跑完 `python scripts/build_index.py` 之后，可以在
`data/processed/<论文名>_chunks.json` 里按 `text` 找到对应片段，抄下它的 `id`。

⚠️ chunk id 与**解析器 / 分块器的行为**强绑定：改了 `pdf_parser.py` 或 `chunker.py`
之后，id 会变，旧评测集的 `gold chunk id` 会失配。这类改动必须在 `EXPERIMENTS.md`
里登记，并重新生成 gold id。

## 评测指标（`evaluate_v2.py` 输出）

- **Strict Recall@10**：Top-10 里命中 `relevant_chunk_ids` 的比例（严格按 id 匹配）
- **Recall after expansion**：把召回的父块展开成子块后再匹配（宽松口径）
- **生成评分**：LLM 按「准确性 / 引用 / 完整性」给 0–10 分
