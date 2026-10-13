# 版本记录

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