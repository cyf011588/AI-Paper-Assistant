# AI Paper Assistant

面向科研人员的个人论文知识库 + 论文写作助手。

## 项目能力

### 1. 文献问答（💬 问答 Tab）



* 上传 PDF（中文 / 英文期刊论文、硕士学位论文），自动解析入库

* 混合检索：BM25 关键词 + 向量检索 (bge-base-zh-v1.5) + RRF 融合 + BGE Reranker 精排

* 作者 / 论文名自动识别：问 "雷杰鑫大论文结论" 自动锁定对应论文，不跨论文召回

* 跨论文对比：问 "两篇论文共同导师是谁" 自动拆解子问题分别检索

* 问题头脑风暴：LLM 扩展检索词，定位答案可能在的章节类型

* 引用来源格式：年份，作者。标题。期刊 | 章节 | 页码

### 2. 论文写作（✍️ 论文写作 Tab）



* LangGraph 五节点工作流：检索员→机制推理员→写作员→去重→审稿人

* **写作员用 Qwen3-VL-30B**直接看用户上传的实验图片 / 曲线 / 云图

* 参考课题组论文的逻辑结构（不复制原文数据）

* 导师风格库：从 Word 修订记录提取修改偏好，模仿导师改法

* 8-gram 去重：自动对比模板文本，防止抄袭

### 3. 知识库管理（🗂️ 知识库管理 Tab）



* 查看所有已入库论文及 chunk 数

* 批量上传实验图片（png/jpg/tif），自动调 Qwen-VL 生成结构化描述

* 图片元数据：试件编号、荷载步、类型 (curve/contour/photo)

* 单张删除 / 清空全部

## 技术栈



| 模块        | 工具                               | 说明                       |
| --------- | -------------------------------- | ------------------------ |
| PDF 解析    | PyMuPDF                          | 提取文本、图片、表格               |
| 分块        | StructuralChunker                | 父子分块：父块 = 完整小节，子块 = 语义片段 |
| Embedding | BAAI/bge-base-zh-v1.5            | 中文向量                     |
| 向量库       | ChromaDB (PersistentClient)      | 持久化，增量更新                 |
| 关键词检索     | rank\_bm25                       | BM25                     |
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
├── data/
│   ├── papers/             # PDF原始文件
│   ├── paper_metadata.json # 论文元数据（作者、年份、期刊）
│   └── chroma_db/          # ChromaDB持久化数据
├── evaluate_v2.py          # 分层评估脚本
├── parse_word_revisions.py # 提取Word修订记录→导师风格库
├── test_vl.py              # 测试Qwen-VL
└── requirements.txt
```

## ChromaDB 集合



| 集合名           | 内容                  |
| ------------- | ------------------- |
| paper\_chunks | 论文父子块（1601 条）       |
| style\_db     | 导师修改对（Before→After） |
| my\_results   | 用户上传的 Excel/CSV 数据  |
| figure\_db    | 实验图片特征描述            |

## 运行



```
set HF_ENDPOINT=https://hf-mirror.com
F:\anaconda\python.exe src/app.py
# 浏览器打开 http://127.0.0.1:7860
```

## 评估



```
F:\anaconda\python.exe evaluate_v2.py
# 当前：总分7.1/10，Recall@10=74%
```