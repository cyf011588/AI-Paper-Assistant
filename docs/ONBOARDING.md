# 协作者指南（ONBOARDING）

> 一句话：这是一个「论文知识库 + 文献问答 + 论文写作」的 RAG 应用。
> 本文告诉你：**项目里有什么、哪些归你改、哪些绝对不能动、你要配置什么、从零怎么跑起来。**
>
> 先读 `README.md` 拿总览，再读本文拿细节，**动手改代码前必读 `docs/INTERFACES.md`**。

> ### ⚠️ 本组当前分工（先看这个）
>
> **检索层已冻结，只开放写作层。**
>
> - **不用改**：怎么切分（`chunker.py`）、用哪个 embedding（`vector_store.py`）、怎么召回与精排
>   （`smart_retriever.py` / `reranker.py` / `query_understanding.py`）、PDF 怎么解析。
> - **你要改的**：`src/writing_agent.py` 里的 6 个节点与 Prompt（论文写作 Tab 的全部逻辑）、
>   `src/vision_extractor.py`（看图描述）、导师风格库（`parse_word_revisions.py`）。
> - **因此不需要跑评测**：`evaluate_v2.py` 那套 30 题评测是给「改检索」的人比分数用的；
>   你改写作 Prompt，**改完直接看生成结果**即可（是不是引用了你图里的数据、有没有编造、像不像导师的改法）。
>
> 详见本文 **第 5 节**。

---

## 0. 三十秒了解这个项目

- **它做什么**：把论文 PDF 放进知识库 → 用自然语言提问 → 得到**带「论文 | 章节 | 页码」引用**的回答；另有一个辅助写论文的模块。
- **它怎么做的**：`PDF → 结构感知解析 → 父子分块 → 向量库 + BM25 双路召回 → RRF 融合 → cross-encoder 精排 → LLM 生成带引用回答`。
- **现在什么水平**：30 题评测，**Strict Recall@10 = 78.9%**，扩展后 89.2%，生成 **7.53/10**（数据见 `evaluation/eval_v3_result.json`）。
- **现在要做什么**：从「单机单人」变成「**多人、多机、各自方向**可并行优化」——这就是整个协作层存在的理由。

---

## 1. 目录地图：里面有什么

### 1.1 核心代码 —— `src/`（13 个 `.py`）

| 文件 | 层 | 职责 | 改动风险 |
|---|---|---|---|
| `config.py` | L0 横切 | 全局配置，**全部从 `.env` 读取**，不含任何密钥 | 可新增，**不可删/改现有导出名** |
| `pdf_parser.py` | L1 解析 | 结构感知 PDF 解析（自适应字号阈值，区分页眉/正文/标题/图注） | 可改（解析方向） |
| `chunker.py` | L1 解析 | 父子分块：父块 = 完整小节，子块 = 语义片段 | 可改，但**必须重跑评测** |
| `multimodal_parser.py` | L1 解析 | Excel/CSV → Markdown 表格 | 可改 |
| `vector_store.py` | L2 索引 | ChromaDB 持久化向量库（bge-base-zh），增量更新 | 可改（embedding 方向） |
| `vision_extractor.py` | L2 索引 | Qwen-VL 提取实验图片特征（曲线/云图/照片） | 可改（多模态方向） |
| `smart_retriever.py` | L3 检索 | 混合检索 + 查询理解 + 问题头脑风暴 + 章节聚合 | 可改（召回方向，主战场） |
| `reranker.py` | L3 检索 | BGE cross-encoder 精排 | 可改 |
| `query_understanding.py` | L3 检索 | 识别问题中的作者/论文名并映射到具体论文 | 可改 |
| `generator.py` | L4 生成 | DeepSeek 生成回答 + 整理引用 | 可改（Prompt 方向） |
| `writing_agent.py` | L4 生成 | LangGraph 五节点写作工作流（检索→机制推理→看图写作→去重→审稿） | 可改（写作方向） |
| `app.py` | L5 交互 | Gradio 网页界面，三个 Tab（问答 / 论文写作 / 知识库管理） | 可改（前端方向） |
| `session_store.py` | L5 交互 | SQLite 对话持久化与恢复 | 可改 |

> 层的定义与三张数据契约见 **`docs/INTERFACES.md`**。

### 1.2 脚本 —— `scripts/`

| 文件 | 作用 | 你什么时候用 |
|---|---|---|
| `bootstrap.py` | **只读环境自检**：Python / 依赖 / 密钥 / 数据 / 索引，并打印下一步命令 | **第一次进来先跑它** |
| `build_index.py` | 从 `data/papers/*.pdf` 一键重建「解析 → 分块 → 向量库」 | 换机器后必跑；改了 `pdf_parser.py` / `chunker.py` 后必跑 |

### 1.3 根目录脚本 —— 18 个 `.py`

| 类别 | 文件 | 你需要关心吗 |
|---|---|---|
| **写作层工具** | `parse_word_revisions.py` | ✅ **要**：从 Word 修订记录提取导师修改偏好 → `style_db` |
| 写作层自检 | `test_vl.py` | ✅ **要**：测 Qwen-VL 看图能力是否可用 |
| 检索层评测 | `evaluate_v2.py`、`eval_30_focused.py`、`eval_30_cond_toc.py` | ❌ 不用：只在做检索优化时才跑 |
| 检索层消融档案 | `ablation_*.py`（13 个） | ❌ 不用：已完成的检索对照实验，只读记录 |

> 一句话：**根目录里除 `parse_word_revisions.py` 与 `test_vl.py` 外，其余都可以不看。**
> 这些检索实验的结论（含负结果）汇总在 `EXPERIMENTS.md`，需要时看那个就行。

### 1.4 数据 —— `data/`（**全部不进 git**）

| 项 | 内容 | 是否进 git |
|---|---|---|
| `papers/` | 12 篇论文 PDF（受版权保护，**不要公开**） | 不进 |
| `processed/` | 13 组 `*_chunks.json` + `*_structured.json`（解析产物） | 不进（可重建） |
| `chroma_db/` | ChromaDB 持久化（当前 1601 条向量） | 不进（可重建） |
| `cache/` | 约 1500 个 embedding 向量缓存 | 不进（可重建） |
| `paper_metadata.json` | 论文题录（论文名/作者/年份/期刊）—— 用 Web UI 上传论文时自动生成 | **不进** |
| `paper_metadata.example.json` | 题录格式模板 | 进 |
| `mechanics_mapping.json` | 力学现象 → 通俗解释映射（供图片描述用，不含论文内容） | 进 |
| `session.db` | SQLite 会话库 | 不进 |

### 1.5 评测 —— `evaluation/`

> 只改写作层的话，**这一整节可以跳过**。它服务于「改检索 → 比分数」的场景。

| 项 | 说明 | 是否进 git |
|---|---|---|
| `dataset/README.md` | **评测集字段说明 + `gold chunk id` 怎么定** | 进 |
| `dataset/example_dataset.json` | 纯格式模板（不含任何真实论文内容） | 进 |
| `dataset/*.json`（你自己的评测集） | 评测集。**本仓库不发布**，人人本地自备 | **不进** |
| `eval_*.json`（评测结果，如 `eval_v3_result.json`） | 含论文衍生答案，本地跑出来即可 | **不进** |

> 本仓库**不发布评测集**：它的问题、标准答案、`gold chunk id` 都来自具体论文。
> 历史上的基线数字（strict 78.9% / 扩展后 89.2% / 生成 7.53）记录在
> [`EXPERIMENTS.md`](../EXPERIMENTS.md) 里，那只是数字，不含论文内容。

### 1.6 文档

| 文件 | 什么时候看 |
|---|---|
| `README.md` | 第一次进来，拿总览 |
| `docs/ONBOARDING.md` | 本文：完整讲解 |
| `docs/INTERFACES.md` | **改代码前**：五层划分 + 三张数据契约 |
| `docs/COLLABORATION.md` | 分工、分支策略、评测规范、合并检查清单 |
| `docs/DISTRIBUTION.md` | 怎么把项目给到别人（为什么不给安装包） |
| `EXPERIMENTS.md` | 实验登记表（含历史实验与负结果） |
| `CHANGELOG.md` | 版本史（v0.1 → v0.9） |

### 1.7 配置类文件

| 文件 | 是否进 git | 说明 |
|---|---|---|
| `requirements.txt` | 进 | 完整运行依赖（已锁定版本） |
| `requirements-experiments.txt` | 进 | 只做检索/评测的最小依赖 |
| `.env.example` | 进 | 环境变量**模板**（不含真实密钥） |
| `.env` | **不进** | **你自己的真实密钥**，每人一份 |
| `src/config.py` | 进 | 配置读取逻辑，**不含密钥** |
| `src/config.py.example` | 进 | 参数总览（脱敏） |
| `.gitignore` | 进 | 排除 `.env` / `data/` / `*.pdf` / 模型文件 |

---

## 2. 你要怎么做（从零到跑起来，6 步）

```bash
# 1. 拿代码（只有代码 + 文档；论文、索引、评测集都在本地自备）
git clone https://github.com/<账号>/AI-Paper-Assistant.git
cd AI-Paper-Assistant

# 2. 自检：它会逐项告诉你缺什么、下一步敲什么
python scripts/bootstrap.py

# 3. 装依赖（只做检索/评测的话，可用更小的 requirements-experiments.txt）
pip install -r requirements.txt

# 4. 配密钥：复制模板为 .env，填入自己的 Key
cp .env.example .env          # Windows: copy .env.example .env

# 5. 放语料并重建索引（把论文 PDF 拷进 data/papers/ 后）
python scripts/build_index.py

# 6. 启动
python src/app.py             # 浏览器打开 http://127.0.0.1:7860
```

跑评测（**只有改检索层时才需要，改写作 Prompt 不用**）：

```bash
python evaluate_v2.py
```

---

## 3. 你需要配置什么（只有 2 个是必填）

写在 `.env` 里，格式 `KEY=value`。

### 3.1 必填（不填就没有大模型和视觉能力）

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek 密钥，用于问答生成 / 检索规划 / 评测打分 |
| `SILICONFLOW_API_KEY` | 硅基流动密钥，用于问题头脑风暴（Qwen2.5-7B）与看图（Qwen-VL） |

### 3.2 可选（都有默认值，想换模型 / 调参数才写）

| 变量 | 默认值 |
|---|---|
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | `deepseek-chat` |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` |
| `TEXT_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct`（问题头脑风暴） |
| `VL_MODEL_NAME` | `Qwen/Qwen3-VL-30B-A3B-Instruct`（看图） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `80` |
| `TOP_K` / `RECALL_K` | `5` / `100` |
| `EMBEDDING_MODEL` / `RERANKER_MODEL` | `BAAI/bge-base-zh-v1.5` / `BAAI/bge-reranker-base` |
| `HEADER_HEIGHT_RATIO` / `FOOTER_HEIGHT_RATIO` | `0.08` / `0.92` |
| `SMALL_FONT_SIZE` / `MAX_HEADING_LEN` | `7` / `100` |
| `HF_ENDPOINT` | `https://hf-mirror.com`（模型下载镜像） |

**优先级：`.env` > 系统环境变量 > 代码默认值。**
所以写进 `.env` 的项一定生效，不会被你机器上其他项目的同名变量悄悄覆盖。

> 标定用一句话：**只要 2 个 Key 必填，其余全部有默认值**——不配置也能跑，只是没有大模型能力。

---

## 4. 绝对不能动的东西（红线）

| 编号 | 不能动 | 为什么 |
|---|---|---|
| 1 | `evaluation/dataset/` 的评测集 | 它是**统一裁判**。改了，你的结果就无法和任何人对齐（*仅改检索时适用*） |
| 2 | 三张数据契约的**字段名与语义**（见 `docs/INTERFACES.md`） | 改了下游代码会**静默崩**（不报错但结果错） |
| 3 | `config.py` 中现有的**导出名** | 删/改名会让全项目 import 失败。新增字段必须给默认值 |
| 4 | `Chunk.id` 的**粒度** | 评测集的 `gold_chunk_ids` 直接指向它，改了历史评测全废（*仅改检索时适用*） |
| 5 | **`.env`** | 绝不能提交。它已在 `.gitignore` 里，别去删这行 |
| 6 | `data/` 里的语料 | 受版权保护，不进 git，也不要对外分享原 PDF |

**动这些之前**：走 `docs/INTERFACES.md` 的「契约变更流程」——登记 → 声明 → 重跑评测 → 在 PR 里给对照表。

---

## 5. 归你改的（当前：只开放写作层）

**约定：检索层已冻结，不开放。** 切分方式、embedding 模型、召回与精排都不用动 ——
所以**你也不需要跑评测**，改完直接看生成结果。

### 5.1 你的主场：论文写作（✍️ 论文写作 Tab）

主文件 `src/writing_agent.py`，一条 LangGraph 工作流，6 个节点：

| 节点 | 干什么 | 你想改的大概是 |
|---|---|---|
| `_retriever_node` | 提取「作者 + 章节」→ 硬过滤 ChromaDB 取模板段落；顺带取风格案例与图片特征 | 取几段模板、取几条风格案例 |
| `_intent_router_node` | 判断写作场景：整体趋势 / 局部状态 / 参数对比 / 理论推演 | 加新的写作场景类型 |
| `_mechanism_node` | 按场景选 Prompt，推理力学因果链（读 `data/mechanics_mapping.json`） | **各场景的推理 Prompt** |
| `_writer_node` | **Qwen3-VL 看图写作**（最关键的一个节点） | **写作主 Prompt**：语气、结构、字数、禁用复制模板数据 |
| `_dedup_node` | 8-gram 对比模板，有重复就调 DeepSeek 同义改写 | 重复判定阈值（当前 8-gram） |
| `_reviewer_node` | DeepSeek 审稿：数据合理 / 表达严谨 / 有无抄袭，不合格退回重写 | 审查标准与通过条件 |

改法：**直接改节点函数里的 Prompt 文本即可**，不需要碰检索层。改完在 Web UI 的
「论文写作」Tab 上传一张实验图 + 写一句需求，看输出效果。

配套可改的还有：

| 文件 | 作用 |
|---|---|
| `src/vision_extractor.py` | 图片特征描述 Prompt（曲线 / 云图 / 照片三类） |
| `parse_word_revisions.py` | 从 Word 修订记录提取导师修改偏好 → `style_db`（风格库） |
| `src/generator.py` | 问答生成的 Prompt（若你想顺带调问答口径） |

**改完怎么判断变好**：人工看三条 —— ① 数据是否都来自你上传的图（有没有编造）；
② 有没有照抄模板里的具体数值 / 试件号；③ 读起来像不像导师改出来的。

### 5.2 明确不用你碰的（已冻结）

| 模块 | 文件 |
|---|---|
| PDF 解析 / 分块 | `pdf_parser.py`、`chunker.py`、`multimodal_parser.py` |
| Embedding / 向量库 / 索引 | `vector_store.py`、`config.py` 里模型名相关项 |
| 召回 / 精排 / 查询理解 | `smart_retriever.py`、`reranker.py`、`query_understanding.py` |
| 前端 / 会话 | `app.py`、`session_store.py` |

> <details><summary>若将来要重开检索优化（现在不需要）</summary>
>
> 届时才需要跑评测，且**必须用同一套评测集**：`python evaluate_v2.py`，
> 再用「基线 vs 改动」对照表说明效果。规范见 `docs/COLLABORATION.md`。
> </details>

**通用底线**：只改自己那层，别人就不会被你打断。跨层改动允许，但要**声明为契约变更**
（见 `docs/INTERFACES.md`）。

---

## 6. 协作流程（写作层版）

1. **一条分支一个方向**：`git checkout -b feat/writing-<你的方向>`。不要把两个无关改动塞进同一分支。
2. **改完自己看效果**：在 Web UI 的「论文写作」Tab，用**同一张实验图 + 同一条需求**跑改前 / 改后两版对比。
   当前不做量化评测（检索层已冻结，写作质量本来也只能人工判断）。
3. **登记一行**：在 `EXPERIMENTS.md` 追加「问题 → 假设 → 改动 → 结果 → 结论」。**负结果同样登记**，不要删。
4. **PR 里说清楚**：改了哪个节点 / 哪段 Prompt、为什么这样改、贴上改前改后的生成示例。

细节见 `docs/COLLABORATION.md`。

---

## 7. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 跑起来知识库是空的 | `data/` 没进 git | 放 PDF 进 `data/papers/`，跑 `python scripts/build_index.py` |
| 模型下载卡住 | 没设镜像 | 已在 `config.py` 默认 `HF_ENDPOINT=https://hf-mirror.com`；要换就写进 `.env` |
| 配了 `.env` 却不生效 | 曾以为是系统变量优先 | 本项目已定为 `.env` 优先；确认你在项目根目录下运行 |
| 生成结果抄了模板里的数字 | `_writer_node` 的约束不够狠 | 加强「硬性要求」那段的措辞，见本文 5.1 |
| 改完觉得变好但说不清 | 没有固定对照 | 用**同一张图 + 同一条需求**跑改前 / 改后两版，按 5.1 的三条标准逐条对 |
| 不知道从哪改起 | — | 先读本文 5.1 的节点表，从 `_writer_node` 的写作主 Prompt 下手 |

---

## 8. 现在这个项目的真实状态（不粉饰）

- **已完成**：RAG 问答闭环、写作工作流、知识库管理、可复现化改造、协作层（契约/规范/分发）。
- **检索层已知短板（已冻结，你不用管）**：跨论文对比题（58.3%）与按章节问（65.3%）召回偏弱；
  `q026` 单题检索失败（strict=0.0）。
- **已知负结果**：Conditional-TOC 方案反而变差（对比题 −25pp），保留在 `EXPERIMENTS.md` 作为反例证据。
- **尚未做**：目录解耦（把 `src/` 分成 `core/` 与 `experiments/`）——因会打断 import 路径，暂缓。
