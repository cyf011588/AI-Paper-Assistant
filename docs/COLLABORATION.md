# 协作机制（COLLABORATION）

> 目标：多人、多机、各自方向并行优化，互不覆盖、且**结果可比较**。
> 三条铁律：**① 统一裁判 ② 只动自己那层 ③ 数据不进 git。**

---

## 一、谁只碰哪些文件（分工边界）

先看 `docs/INTERFACES.md` 的五层划分。下面把「优化方向」映射到「允许改的文件」：

| 优化方向 | 只允许改 | 评测依据 |
|---|---|---|
| 解析 / 双栏 / 公式 / 表格 | `pdf_parser.py`、`chunker.py`、`multimodal_parser.py` | 30 题 Retrieval |
| 换 embedding / 多向量 / 索引结构 | `vector_store.py`、`config.py` 里模型名相关项 | 30 题 Retrieval |
| 召回 / 精排 / 查询改写 | `smart_retriever.py`、`reranker.py`、`query_understanding.py` | 30 题 Retrieval |
| Prompt / 写作 Agent / 多模态看图 | `generator.py`、`writing_agent.py`、`vision_extractor.py` | 生成评分 + 引用正确率 |
| 前端 / 产品化 / 会话 | `app.py`、`session_store.py` | 人工体验 |
| 换学科语料 | `data/` + **新建**评测集 | 新领域评测 |

> 跨层改动（例如同时改 `chunker.py` 和 `smart_retriever.py`）是允许的，但**必须在 PR 描述里说明是「契约变更」**，并走 `INTERFACES.md` ⑤ 的流程。

---

## 二、分支策略

```bash
# 主干永远保持「评测通过」的状态
main / master

# 每个方向一条分支，命名 feat/<方向>
git checkout -b feat/parser-two-column
git checkout -b feat/retriever-rrf-weight
git checkout -b feat/embedding-bge-m3
```

规则：
1. **一个方向一条分支**，不要把两个无关优化塞进同一分支（否则评测归因不清）。
2. 主干只合并**评测通过**的改动；未通过的改动留在分支上——**负结果同样有价值**，但如实标注。
3. 合并前先 `git rebase origin/main`（或 merge），确保评测是在**最新主干**上跑的。
4. **负结果不要删分支**：现有 `Conditional-TOC`（-4.1pp）就是可复用的负结果证据。

---

## 三、统一裁判（最重要）

**`evaluation/dataset/` 下的 30 题评测集是全项目唯一的对比口径。**
任何优化，必须跑同一套题，否则「结果不认」。

### 怎么跑

```bash
# 在仓库根目录执行
python evaluate_v2.py                # 检索层 + 生成层的分层评测
```

评测集路径由 `config.EVAL_DATASET_PATH` 统一定位（仓库内优先，老机器绝对路径回退），
因此**任何机器跑的都是同一份题**，不再依赖某台机器的本地目录。

### 怎么记账

1. 结果落盘到 `evaluation/<方向>_<YYYYMMDD>.json`（与现有 `eval_v3_result.json` 同结构）。
2. 在 `EXPERIMENTS.md` 追加一行：问题 → 假设 → 改动 → 结果 → 结论。
3. 在 PR 描述里贴出**对照表**（基线 vs 你的改动），格式：

   | 指标 | 基线(baseline) | 本次改动 | Δ |
   |---|---|---|---|
   | Strict Recall@10 | 78.9% | ?? | ?? |
   | 扩展后 Recall | 89.2% | ?? | ?? |
   | 生成均分(/10) | 7.53 | ?? | ?? |

### 口径纪律（血泪教训）

- **Strict Recall@10 ≠ 准确率。** 它是「扩展前 Top-10 命中 gold chunk 的比例」，是**检索**指标，不要对外说成准确率。
- `eval_v2` 的 `recall_at_10`（73.9%）与 `eval_v3` 的 `strict_recall_at_10`（78.9%）**口径不同，不可直接比**。
- 生成分是 LLM 打分，有噪声；**单题差异不构成结论**，看十题以上的均值。

### 当前基线（据 `evaluation/eval_v3_result.json`）

| 指标 | 值 |
|---|---|
| Strict Recall@10（30 题均值） | **78.9%** |
| 扩展后 Recall | **89.2%** |
| 生成均分 | **7.53 / 10** |
| 检索模式分布 | hybrid_rerank 18 / section_shortcut 7 / comparison_decompose 5 |

按题型看短板（**优化的优先级参考**）：

| 题型 | n | Strict Recall@10 |
|---|---|---|
| 跨论文对比 | 3 | **58.3%** ← 最弱 |
| 按章节问 | 6 | **65.3%** ← |
| 按内容问 | 7 | 76.3% |
| 中英混合 | 3 | 83.3% |
| 事实细节 | 4 | 87.5% |
| 按作者问 | 6 | 94.5% |
| 总结归纳 | 1 | 100% |

---

## 四、依赖分治

- **通用依赖**只加在 `requirements.txt`（运行核心）里，改动需要说明理由。
- **某方向专属依赖**加在 `requirements-experiments.txt`，或干脆只写进你自己分支的说明里，**不要污染主干**。
- 想固定整个环境：`pip freeze > requirements-lock.txt`（锁版本，供复现）。
- 任何人都**不要**把 `site-packages`、`chroma_db`、`cache` 提交进 git。

---

## 五、数据不进 git

`.gitignore` 已排除：`data/papers/`、`data/processed/`、`data/chroma_db/`、`data/cache/`、`*.pdf`。

原因：语料是**受版权保护的学位论文**，且索引体积大、可重建。

**新同事怎么拿到数据和索引？**

```bash
# 1) 把 PDF 放进 data/papers/
# 2) 一键重建解析 + 分块 + 向量库
python scripts/build_index.py
```

**确需共享语料时**（例如换学科）：走网盘 / 对象存储，给出**文件清单 + MD5**，不要直接塞 git。

> 评测集是**例外**：`evaluation/dataset/*.json` **必须进 git**，它是统一裁判，人人必须有同一份。

---

## 六、提交 / 合并前检查清单

- [ ] 只改了自己方向的文件（或已声明为契约变更）
- [ ] `python -m py_compile` 通过
- [ ] 跑通 `python evaluate_v2.py`，结果落盘 + 登记 `EXPERIMENTS.md`
- [ ] PR 描述含「基线 vs 改动」对照表
- [ ] 没有提交 `.env`、`data/`、`*.pdf`
- [ ] 契约若有变更，已更新 `docs/INTERFACES.md`

---

## 七、环境统一（避免「在我机器上能跑」）

统一约定（`scripts/bootstrap.py` 会自检）：

| 项 | 约定 |
|---|---|
| Python | **3.13**（本机为 `F:/anaconda/python.exe`，3.13.9） |
| 依赖 | `pip install -r requirements.txt` |
| 密钥 | 每人自己 `cp .env.example .env` 填 Key；`.env` 不入库 |
| 配置优先级 | **`.env` > 系统环境变量 > 代码默认值** |
| 国内镜像 | `HF_ENDPOINT=https://hf-mirror.com`（已默认） |

> 注意：**`.env` 优先于系统环境变量**是刻意设计的。否则机器上其他项目的同名 `DEEPSEEK_API_KEY` 会悄悄覆盖你的配置。
