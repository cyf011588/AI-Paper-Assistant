# AI Paper Assistant

面向科研人员的个人论文知识库 + 论文写作助手。

> **给课题组同事的一句话**：检索层（解析 / 切分 / embedding / 召回 / 精排）**已冻结、不用改**，
> 你只需要改 **论文写作模块**（`src/writing_agent.py`）——因此**不需要跑评测**，改完直接看效果。
> 详见 [`docs/ONBOARDING.md`](docs/ONBOARDING.md) 第 5 节。

## 项目能力

### 1. 文献问答（💬 问答 Tab）

- 上传 PDF（中文 / 英文期刊论文、硕士学位论文），自动解析入库
- 混合检索：BM25 关键词 + 向量检索 (bge-base-zh-v1.5) + RRF 融合 + BGE Reranker 精排
- 作者 / 论文名自动识别：问 "张三硕士论文结论" 自动锁定对应论文，不跨论文召回
- 跨论文对比：问 "两篇论文共同导师是谁" 自动拆解子问题分别检索
- 问题头脑风暴：LLM 扩展检索词，定位答案可能在的章节类型
- 引用来源格式：年份，作者。标题。期刊 | 章节 | 页码

### 2. 论文写作（✍️ 论文写作 Tab）

- LangGraph 五节点工作流：检索员→机制推理员→写作员→去重→审稿人
- **写作员用 Qwen3-VL-30B**直接看用户上传的实验图片 / 曲线 / 云图
- 参考课题组论文的逻辑结构（不复制原文数据）
- 导师风格库：从 Word 修订记录提取修改偏好，模仿导师改法
- 8-gram 去重：自动对比模板文本，防止抄袭

### 3. 知识库管理（🗂️ 知识库管理 Tab）

- 查看所有已入库论文及 chunk 数
- 批量上传实验图片（png/jpg/tif），自动调 Qwen-VL 生成结构化描述
- 图片元数据：试件编号、荷载步、类型 (curve/contour/photo)
- 单张删除 / 清空全部

## 技术栈

| 模块        | 工具                               | 说明                       |
| --------- | -------------------------------- | ------------------------ |
| PDF 解析    | PyMuPDF                          | 提取文本、图片、表格               |
| 分块        | StructuralChunker                | 父子分块：父块 = 完整小节，子块 = 语义片段 |
| Embedding | BAAI/bge-base-zh-v1.5            | 中文向量                     |
| 向量库       | ChromaDB (PersistentClient)      | 持久化，增量更新                 |
| 关键词检索     | rank_bm25                        | BM25                     |
| 融合        | RRF (Reciprocal Rank Fusion)     | k=60                     |
| 精排        | BAAI/bge-reranker-base           | cross-encoder 重排         |
| 问答 LLM    | DeepSeek (deepseek-chat)         | 问答 / 检索规划 / 审稿           |
| 写作 LLM    | Qwen3-VL-30B-A3B-Instruct (硅基流动) | 看图写作 / 图片描述              |
| 文本 LLM    | Qwen2.5-7B-Instruct (硅基流动)       | 问题头脑风暴                   |
| 前端        | Gradio                           | Web UI                   |
| 工作流       | LangGraph                        | 多智能体写作                   |

## 项目结构

```
AI-Paper-Assistant/
├── src/
│   ├── app.py              # Gradio Web UI（三个Tab）
│   ├── config.py           # API密钥、模型名配置
│   ├── pdf_parser.py       # PDF解析器（PyMuPDF）
│   ├── chunker.py          # StructuralChunker父子分块
│   ├── vector_store.py     # ChromaDB持久化向量库
│   ├── smart_retriever.py  # 混合检索+查询理解+问题头脑风暴
│   ├── generator.py         # DeepSeek LLM生成回答
│   ├── query_understanding.py # 作者/论文名识别
│   ├── reranker.py         # BGE reranker精排
│   ├── writing_agent.py    # LangGraph五节点写作工作流
│   ├── vision_extractor.py  # Qwen-VL图片特征提取（曲线/云图/照片）
│   └── multimodal_parser.py # Excel/CSV→Markdown表格
├── scripts/
│   ├── build_index.py      # 一键重建索引（新机器只需放 PDF 即可复现）
│   └── bootstrap.py        # 环境自检（新机器第一步：缺什么、下一步敲什么）
├── docs/
│   ├── ONBOARDING.md       # 协作者指南：改哪里、怎么跑（新同事从这里开始）
│   ├── SETUP-新同事全流程.md # 零基础装机分步（要装什么、怎么配 API）
│   ├── INTERFACES.md       # 层间数据契约（改代码前必读）
│   ├── COLLABORATION.md    # 分支 / 分工协作规范
│   └── DISTRIBUTION.md     # 怎么把项目给到别人（Git vs 安装包）
├── data/                   # 语料与索引：全部不入库，放论文后本地重建
│   ├── papers/             #   PDF 原文（各人自备）
│   ├── processed/          #   解析产物（可重建）
│   └── chroma_db/          #   向量库（可重建）
├── parse_word_revisions.py # 提取Word修订记录→导师风格库
├── test_vl.py              # 测试Qwen-VL
├── .env.example            # 环境变量模板（复制为 .env 填自己的 Key）
└── requirements.txt        # 完整运行依赖（已锁定版本）
```

> 本仓库只发布**能跑起来的框架 + 写作模块**。原作者的检索层实验资产
> （`ablation_*.py`、`evaluate_v2.py`、`EXPERIMENTS.md`、`evaluation/` 等）
> 含论文作者名与题号，**不随本仓库分发**。

## ChromaDB 集合

| 集合名          | 内容                  |
| ------------ | ------------------- |
| paper_chunks | 论文父子块（1601 条）       |
| style_db     | 导师修改对（Before→After） |
| my_results   | 用户上传的 Excel/CSV 数据  |
| figure_db    | 实验图片特征描述            |

## 快速开始（在一台新机器上）

### 0. 先自检（推荐）

```bash
python scripts/bootstrap.py     # 只读检查：Python / 依赖 / 密钥 / 数据 / 索引，并打印下一步
```

### 1. 准备环境

```bash
# 建议 Python 3.13
pip install -r requirements.txt
```

### 2. 配置密钥（不再改源码）

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

打开 `.env` 填入自己的 Key（`DEEPSEEK_API_KEY` / `SILICONFLOW_API_KEY`）。  
`.env` 已在 `.gitignore` 中，不会被提交。`config.py` 会自动读取它；  
**不配置 .env 时用的是代码里的默认值，行为与旧版一致。**

优先级为 **`.env` > 系统环境变量 > 代码默认值**：写进 `.env` 的项一定生效，  
没写的项才回退到系统环境变量，因此不会被机器上其他项目的同名变量意外遮蔽。

### 3. 准备语料并重建索引

```bash
# 把论文 PDF 放进 data/papers/，然后：
python scripts/build_index.py
```

`build_index.py` 会完成「解析 → 分块 → 写入 ChromaDB」全流程，  
等价于在 Web UI 里逐篇上传，因此**新机器无需传输向量库 / 缓存**。

常用参数：

| 命令                                               | 作用                          |
| ------------------------------------------------ | --------------------------- |
| `python scripts/build_index.py`                  | 增量：复用已有解析结果，逐篇入库            |
| `python scripts/build_index.py --rebuild`        | 清空 `paper_chunks` 集合后全量重建   |
| `python scripts/build_index.py --force-parse`    | 忽略已有解析结果，重新解析 PDF           |
| `python scripts/build_index.py --from-processed` | 只处理已有 `structured.json` 的论文 |

### 4. 启动

```bash
python src/app.py
# 浏览器打开 http://127.0.0.1:7860
```

## 多机器 / 多人协作要点

- **密钥不入库**：每个人在自己机器上维护各自的 `.env`。
- **论文与论文衍生数据一律不入库**：`data/papers`、`data/processed`、`data/chroma_db`、
  `data/cache`、`data/paper_metadata.json` 都在 `.gitignore` 中。  
  论文由各人自备，索引由 `scripts/build_index.py` 从 PDF 重建 ——
  仓库里只有「能跑起来的框架」。
- **本组分工：检索层已冻结，只改写作层** —— 切分 / embedding / 召回 / 精排都不用动，
  因此**不需要跑评测**，改完 Prompt 直接看效果。见
  [`docs/ONBOARDING.md`](docs/ONBOARDING.md) 第 5 节。
- **只改 .env 也能切模型**：`DEEPSEEK_MODEL`、`VL_MODEL_NAME`、`TEXT_MODEL_NAME` 等
  都可写进 `.env` 覆盖，无需改动源码。

## 文档地图（协作必读）

| 文档 | 什么时候看 |
|---|---|
| [`docs/SETUP-新同事全流程.md`](docs/SETUP-新同事全流程.md) | **第一次装环境时看**：电脑要装什么、要不要 VSCode、没有 Python 怎么办、要不要虚拟环境、怎么配 API（零基础分步） |
| [`docs/ONBOARDING.md`](docs/ONBOARDING.md) | **新协作者从这里开始**：项目里有什么、归谁改、不能动什么、要配置什么 |
| [`docs/INTERFACES.md`](docs/INTERFACES.md) | **改代码前**：五层划分 + 三张数据契约（chunk / 检索结果 / 引用） |
| [`docs/COLLABORATION.md`](docs/COLLABORATION.md) | 分工、分支策略、合并前检查清单 |
| [`docs/DISTRIBUTION.md`](docs/DISTRIBUTION.md) | 怎么把项目给到别人：为什么**不该给安装包**、Git 该怎么传 |
