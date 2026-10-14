# 版本记录

## v0.9.2 (2026-10-14) — 仓库瘦身：移除一次性脚本，明确「只开放写作层」

> 背景：本组分工调整为**检索层冻结、只改论文写作模块**。既然切分 / embedding /
> 召回 / 精排都不动，评测体系对写作任务就用不上，那些「针对具体题目、具体论文」的
> 一次性排查脚本更是纯噪音。本次做减法。

### 移除（41 个，已备份到仓库外 `_backup_AIPA_delete_20261014/`）

- `debug/` 整个目录（30 个）：针对具体论文 / 具体题号的一次性诊断脚本
  （如 `debug_wxt.py`、`fix_section_path.py`、`reindex_wang.py` 只对王鑫涛那篇有效，
  `debug_q013/q014/q021/q026.py`、`show_q0*.py` 只对特定题目有效）。
- 根目录 `_*.py`（5 个）：下划线前缀＝用完即弃。
- 根目录 `debug_*.py`（4 个）、`test_5qs.py`、`test_shortcut_fix.py`：
  一次性排查与内嵌真题的自测脚本。
- `docs/项目规划-AI_PM方向.md`：属于个人求职材料（简历模板 / PM 转型路线），
  已移出到 `F:/AI_Project/作品集/`，不再随项目分发。

### 保留（检索层实验资产完整保留）

- `evaluate_v2.py`、`eval_30_focused.py`、`eval_30_cond_toc.py`、`evaluation/`：
  评测能力与评测集格式模板（`EXPERIMENTS.md` 里 E1–E5 数字的复现依据）。
- `ablation_*.py`（13 个）：`EXPERIMENTS.md` 的 E2–E5 直接引用它们，且 E4/E5 结论
  「未落盘 JSON」，脚本是唯一证据，故保留为只读档案。
- `EXPERIMENTS.md`、`requirements-experiments.txt`、`test_vl.py`、`parse_word_revisions.py`。

### 文档同步

- `README.md` / `docs/ONBOARDING.md` / `docs/COLLABORATION.md` 统一为
  **「检索层已冻结 → 只改 `writing_agent.py` 等写作层 → 不需要跑评测」** 的口径：
  - ONBOARDING 第 5 节重写：给出写作工作流 6 个节点的职责表 + 「改完怎么判断变好」。
  - ONBOARDING 第 1.3 节：标注根目录里只有 `parse_word_revisions.py` / `test_vl.py`
    与写作任务相关。
  - COLLABORATION 第一节：分工表收窄到写作层，明确列出「已冻结」的 9 个文件；
    第三节评测规范标注为「仅重开检索优化时适用」。

### 未改动

- `src/` 下所有业务逻辑、Prompt、默认参数**一行未改**。
- 仓库文件数：97 → 56。

---

## v0.9.1 (2026-10-13) — 修复「新 clone 直接崩溃」

- `src/query_understanding.py`：`__init__` 无条件读取 `paper_metadata.json`，
  新 clone 没有该文件时首次检索即崩。改为**缺失时降级为空 metadata**，
  只失去「按作者 / 论文名硬过滤」能力，其余检索正常。
- `src/writing_agent.py`：改用 `config.DATA_DIR` 绝对路径读
  `data/mechanics_mapping.json`，不再依赖当前工作目录。

## v0.9 (2026-10-13) — 补齐新同事上手会踩的三个坑

- **空目录不进 git**：`data/papers/`、`data/processed/`、`data/chroma_db/`、
  `data/cache/` 在新 clone 里不存在 → `.gitignore` 改用 `data/*/*` + `!data/*/.gitkeep`，
  并新增 4 个 `.gitkeep` 占位。
- **`.env` 静默失效**：未安装 `python-dotenv` 时 `.env` 会被忽略且无任何提示 →
  `config.py` 检测到 `.env` 存在但导入失败时打印明确告警。
- `src/app.py` 端口支持 `APP_PORT` 环境变量覆盖（默认 `7860` 不变）。

---

## v0.8 (2026-10-13) — 公开仓库只发布「可运行的框架」：论文与论文衍生数据全部移出

> 目标：把这个仓库公开出去时，**只给别人能跑起来的东西**，不发布任何论文。
> 课题组要用，自己把 PDF 放进 `data/papers/` 重建即可（`scripts/build_index.py`）。

### 变更

- **论文语料全面移出版本管理**（原已在 `.gitignore`，本次补齐遗漏项）：
  `data/papers/`、`data/processed/`、`data/chroma_db/`、`data/cache/`、`*.pdf`。
- **`data/paper_metadata.json` 移出**（论文题录：标题/作者）→ 新增格式模板
  `data/paper_metadata.example.json`。题录会在用 Web UI 上传论文时自动生成。
- **评测集与评测结果移出**：
  - `evaluation/dataset/RAG分层评测数据集_*.json` → 移出；新增
    `evaluation/dataset/README.md`（字段说明 + `gold chunk id` 怎么定）与
    `evaluation/dataset/example_dataset.json`（纯格式模板，不含真实论文内容）。
  - `evaluation/eval_v2_result.json`、`eval_v3_result.json`、`eval_cond_toc_3rounds.json`
    → 移出（含论文衍生答案）。基线**数字**仍记录在 `EXPERIMENTS.md`（只是数字）。
- **重写 git 历史**：上述文件在旧提交里已存在，仅 `git rm --cached` 不足以隐藏，
  因此用 `git filter-branch` 从**所有历史提交**中移除 `data/paper_metadata.json` 与
  `evaluation/`，再 `reflog expire` + `gc --prune=now` 清掉游离对象。
  提交哈希因此全部改变（`initial`→`8629efc`、`v0.3`→`3e7a3bf`、`v0.7`→`3d5618e`）。
- **`config.py`**：
  - `_resolve_eval_dataset()` 改为自动选取 `evaluation/dataset/` 下第一个 `*.json`
    （跳过 `example_dataset.json`），不再写死文件名。
  - 新增 `EVAL_DATASET_AVAILABLE` 与 `require_eval_dataset()`——缺少评测集时抛出
    **带操作指引**的异常。
- **评测入口容错**：`evaluate_v2.py`、`eval_30_focused.py`、`eval_30_cond_toc.py`
  在缺少评测集/baseline 时打印下一步该怎么做并退出，不再裸抛 `FileNotFoundError`。
- **`scripts/bootstrap.py`**：评测集缺失不再标成 `[MISS]`（新 clone 本来就该没有），
  改为 `[INFO]` + 放置说明。
- **文档同步**：`README.md`、`docs/ONBOARDING.md`、`docs/COLLABORATION.md`、
  `docs/DISTRIBUTION.md` 全部改为「论文/评测集本地自备」的口径。

### 不变

- 业务逻辑、检索/生成算法、默认参数**一行未改**。
- `data/mechanics_mapping.json`（力学现象→通俗解释，通用领域知识）仍在库中。

## v0.7 (2026-10-13) — 协作层：统一裁判 + 数据契约 + 分发方案

> 目标：让「不同电脑、不同人、各自方向」并行优化时**互不打架、结果可比较**。
> 本次只做「让分享出去能跑 + 让协作有规则」的事，不改任何检索/生成算法。

### 新增

- **评测集入仓（统一裁判）**：把原先散落在仓库外的 30 题评测集复制进
  `evaluation/dataset/`，并在 `config.py` 新增 `EVAL_DATASET_PATH`（仓库内优先、
  老机器绝对路径回退）。任何机器跑的都是同一套题。
- **层间数据契约**：新增 `docs/INTERFACES.md`，冻结三张契约
  （① Chunk ② 检索结果 ③ 引用），并给出五层划分与契约变更流程。
- **协作规范**：新增 `docs/COLLABORATION.md`（分工边界、分支策略、评测纪律、依赖分治、
  合并前检查清单）。
- **实验登记**：新增 `EXPERIMENTS.md`，回填历史实验（口径修正、Conditional-TOC 负结果、
  q026 归因、RRF vs Reranker 等）。
- **分发方案**：新增 `docs/DISTRIBUTION.md`，说明「为什么不给安装包 / Git 该怎么传 / 什么进库什么不进库」。
- **环境自检**：新增 `scripts/bootstrap.py`，只读检查 Python / 依赖 / 密钥 / 数据 / 索引，
  并打印下一步该敲的命令。

### 修复（换机器必崩的硬编码路径）

- `src/app.py`（2 处）、`src/query_understanding.py`（1 处）写死的
  `F:/AI_Project/.../paper_metadata.json` → 改用 `config.METADATA_PATH`。
- 14 个评测 / 消融 / 调试脚本里写死的评测集绝对路径 → 改用 `config.EVAL_DATASET_PATH`
  或仓库内相对路径。
- 5 个 `ablation_q026_*.py` 里写死的 `sys.path.insert("F:/AI_Project/.../src")`
  → 改为基于 `__file__` 推导。

> 修复后全仓库 `.py` 中不再有任何 `F:\` 绝对路径（仅 `config.py` 保留一处
> 供老机器兼容的**回退**分支）。

### 未改动

- 所有业务逻辑、接口、默认参数不变；回归验证：全部脚本编译通过、10 个业务模块导入成功、
  config 导出名与原版逐项一致。

***

## v0.6 (2026-10-13) — 可复现化改造（不改变任何功能）

> 目标：让项目能在不同电脑、不同人手上直接跑起来，支撑「各自方向」并行优化。
> 本次改动**只动外围配置，不改任何业务逻辑**，默认行为与 v0.5 完全一致。

### 变更

- **依赖清单补全**：`requirements.txt` 由 2 个包补全为完整运行依赖并锁定版本；
  新增 `requirements-experiments.txt`（只做检索/评测的最小依赖，不含 gradio/langgraph/opencv）。
- **配置外置**：`src/config.py` 改为从 `.env` / 环境变量读取密钥与可调参数；
  新增 `.env.example` 模板；`.env` 由使用者各自维护（已在 `.gitignore` 中）。
  所有参数默认值与原版逐项一致，未配置 `.env` 时行为不变。
- **一键重建索引**：新增 `scripts/build_index.py`，从 `data/papers/` 的 PDF
  重建 `processed/*.json` 与 `chroma_db`，等价于 Web UI 逐篇上传，新机器无需传输索引。
- **文档**：README 增加「快速开始 / 配置 / 重建索引 / 多机多人协作要点」，移除硬编码路径。

### 未改动

- `src/` 下所有业务模块（解析、分块、检索、生成、写作、UI）逻辑与接口不变。
- 根目录的消融 / 评测 / 调试脚本不变。

***

## v0.5 (2026-10-05) — 图片特征入库 + 知识库管理 Tab

### 新增



* **🗂️ 知识库管理 Tab**：论文列表 + 图片列表 + 单张删除 / 清空

* **figure\_db 集合**：批量上传实验图片 (png/jpg/tif)，Qwen3-VL-30B 自动生成结构化描述

* 图片元数据：试件编号、荷载步、类型 (curve/contour/photo)

* TIF 中文路径修复：numpy 二进制读取 + cv2.imdecode 解码

### 修复



* SILICONFLOW\_MODEL 导入错误 → TEXT\_MODEL\_NAME

* ChromaDB figure\_db 默认下载 all-MiniLM → 指定 bge-base-zh

* 旧 figure\_db embedding 冲突 → 删除重建

* Writer 节点 TIF 直接发 Qwen-VL 报 400 → TIF→PNG 转换

* load\_existing\_papers 返回值数量不匹配 → 返回两份给两个 dataframe

### 关键配置



* VL 模型：Qwen/Qwen3-VL-30B-A3B-Instruct（硅基流动）

* 文本模型：Qwen/Qwen2.5-7B-Instruct（硅基流动）

* 问答模型：deepseek-chat（DeepSeek）

* Embedding：BAAI/bge-base-zh-v1.5

* Reranker：BAAI/bge-reranker-base



***

## v0.4 (2026-10 初) — LangGraph 写作工作流

### 新增



* **五节点写作工作流**（writing\_agent.py）：

1. Retriever：DeepSeek 提取 author+chapter，硬过滤 ChromaDB

2. MechanismReasoner：DeepSeek 分析 A→B→C 纵向对比塑性发展因果链

3. Writer：Qwen3-VL 看图写作（参考模板逻辑，不复制数据）

4. De-duplication：8-gram n-gram 对比模板，重复则同义改写

5. Reviewer：DeepSeek 审查数据合理性和抄袭

* **导师风格库**：parse\_word\_revisions.py 提取 Word 修订记录→style\_db

* 写作 Prompt 强化：严禁从模板复制具体数值 / 试件编号 / 宽高比

* A/B/C 点明确为荷载步而非结构位置



***

## v0.3 (2026-09 末) — 混合检索 + 评估体系

### 新增



* **混合检索**：BM25 + 向量检索 + RRF (k=60) 融合

* **BGE Reranker**精排

* **问题头脑风暴**：Qwen2.5-7B 扩展检索词

* **查询理解**：作者名→paper\_name 映射，硬过滤

* **跨论文对比**：查询分解→分别检索→合并对比

* **评估体系**（evaluate\_v2.py）：Recall@10 + DeepSeek 打分 (0-10)

* 30 题专业评测数据集

### 指标



* Recall@10 = 74%

* 生成均分 = 7.1/10



***

## v0.2 (2026-09 中) — ChromaDB 持久化

### 新增



* 内存 cosine → ChromaDB PersistentClient

* 增量更新：上传新 PDF 只解析新文档，不重建索引

* 父子分块：父块 = 完整小节，子块 = 语义片段

* 章节路径 metadata：第六章 > 6.1 > 6.1.1



***

## v0.1 (2026-09 初) — 基线



* PyMuPDF 解析 PDF

* 内存向量 + cosine 相似度

* Gradio 简单问答界面

* 13 篇论文，1601 个 chunk