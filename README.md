# AI Paper Assistant

面向科研人员的个人论文知识库与 AI 科研助手——基于 RAG 的学术论文问答系统。

## 项目简介

针对学术论文双栏排版、章节层级复杂、长文本语义割裂等问题，构建结构感知的 PDF 解析与 RAG Pipeline，实现论文结构化处理、语义检索、证据溯源与科研问答。

## 技术架构

```
PDF上传 → PDF解析 → 结构切分 → Embedding → 向量索引
                                              ↓
用户提问 → 查询理解 → 中文翻译 → 语义检索 → 章节聚合 → LLM生成 → 带引用的回答
```

## 核心功能

- **结构感知 PDF 解析**：自适应识别双栏排版、章节层级、页眉页脚
- **结构感知切分**：按章节边界 + 句子边界切分，保留论文/章节/页码元数据
- **混合检索**：语义检索 + 章节级聚合 + 相邻上下文扩展
- **查询理解**：自动识别作者名/论文名，过滤检索范围
- **跨语言支持**：中文问题自动翻译成英文检索英文论文
- **证据溯源**：回答附带来源论文、章节、页码

## 文件说明

```
AI-Paper-Assistant/
├── src/                    # 源代码
│   ├── app.py              # Gradio Web 界面（主入口）
│   ├── config.py           # 全局配置（路径、API key、参数）
│   ├── pdf_parser.py       # PDF 解析器（自适应结构感知）
│   ├── chunker.py          # 文本切分器（结构感知 + 朴素切分）
│   ├── semantic_retriever.py # BGE 语义检索器
│   ├── smart_retriever.py  # 智能检索器（查询理解 + 过滤 + 章节聚合）
│   ├── query_understanding.py # 查询理解（识别作者/论文名）
│   ├── generator.py        # DeepSeek LLM 生成器
│   ├── retriever.py        # TF-IDF 检索器（基线）
│   └── batch_process.py    # 批量处理脚本
├── data/
│   ├── papers/             # 原始 PDF 论文
│   ├── processed/          # 解析后的结构化 JSON + chunks
│   └── paper_metadata.json  # 论文元数据（标题/作者/年份/别名）
├── evaluation/
│   ├── test_set.json       # 10 题测试集
│   ├── evaluate.py         # 三组对照评估脚本
│   └── evaluation_result.json # 实验结果
├── docs/
│   └── 项目规划-AI_PM方向.md  # 完整项目规划文档
├── requirements.txt        # Python 依赖
└── README.md               # 项目说明
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 DeepSeek API key（在 src/config.py 中）
DEEPSEEK_API_KEY = "your-api-key"

# 启动应用
cd src
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

## 实验结果

| 指标 | 基线（TF-IDF） | 结构切分+语义检索 | 智能检索（查询理解+过滤） |
|---|---|---|---|
| 论文级 Top-1 | 10% | 10% | 60% |
| 论文级 Top-5 | 30% | 20% | 80% |
| 页码级 Top-5 | 10% | 10% | 70% |

## 技术栈

- **PDF 解析**：PyMuPDF
- **Embedding**：BAAI/bge-small-zh-v1.5
- **LLM**：DeepSeek Chat
- **Web UI**：Gradio
- **向量化**：sentence-transformers + numpy

## 已知局限

- 图片中的公式和图表无法提取
- 中文论文标题识别仍需优化
- 未实现跨论文证据汇总
- 单用户本地使用，无多用户支持
