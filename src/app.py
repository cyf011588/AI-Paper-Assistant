"""
app.py — Gradio 网页界面

功能：
1. 上传 PDF 论文，自动解析+切分+入库
2. 聊天问答，带引用来源
3. 显示当前知识库中有哪些论文
"""

import json
import os
from pathlib import Path

# 设置 HuggingFace 国内镜像（必须在 import sentence_transformers 之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import gradio as gr

from config import PAPERS_DIR, PROCESSED_DIR
from pdf_parser import StructuredPDFParser
from chunker import StructuralChunker, Chunk
from smart_retriever import SmartRetriever
from generator import AnswerGenerator


# =========================
# 全局状态
# =========================

retriever = SmartRetriever()
generator = AnswerGenerator()
chunker = StructuralChunker()
paper_list = []  # 已入库的论文列表


def load_existing_papers():
    """启动时加载已有的 chunks"""
    global retriever, paper_list

    chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))
    if not chunk_files:
        return "还没有论文，请先上传"

    all_chunks = []
    papers_seen = set()
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            chunk_dicts = json.load(f)
        for c in chunk_dicts:
            all_chunks.append(Chunk(**c))
            papers_seen.add(c["paper_name"])

    retriever.build_index(all_chunks)
    paper_list = sorted(papers_seen)
    return f"已加载 {len(paper_list)} 篇论文，共 {len(all_chunks)} 个文本块"


def upload_pdf(file_path):
    """上传新 PDF：解析 → 切分 → 重新建索引"""
    global retriever, paper_list

    if file_path is None:
        return "请先选择文件", ""

    pdf_path = Path(file_path)
    filename = pdf_path.name

    # 1. 解析
    parser = StructuredPDFParser(pdf_path)
    sections = parser.parse()

    # 保存结构化 JSON
    structured_path = PROCESSED_DIR / f"{pdf_path.stem}_structured.json"
    parser.save(sections, structured_path)

    # 2. 切分
    chunks = chunker.process_paper(structured_path)

    # 保存 chunks JSON
    chunks_path = PROCESSED_DIR / f"{pdf_path.stem}_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump([{
            "id": c.id, "text": c.text, "paper_name": c.paper_name,
            "section_path": c.section_path, "page_start": c.page_start,
            "page_end": c.page_end,
        } for c in chunks], f, ensure_ascii=False, indent=2)

    # 2.5 自动更新paper_metadata.json
    metadata_path = Path("F:/AI_Project/AI-Paper-Assistant/data/paper_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    else:
        all_metadata = []

    # 检查是否已存在
    existing = [m for m in all_metadata if m["paper_name"] == pdf_path.stem]
    if not existing:
        # 新增
        meta = {
            "paper_name": pdf_path.stem,
            "title": parser.metadata["title"] or pdf_path.stem,
            "authors": parser.metadata["authors"],
            "author_aliases": [],
            "language": "en",
            "year": parser.metadata["year"],
            "journal": "",
        }
        all_metadata.append(meta)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    # 3. 重建索引（把所有 chunks 重新加载）
    all_chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))
    all_chunks = []
    papers_seen = set()
    for cf in all_chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            chunk_dicts = json.load(f)
        for c in chunk_dicts:
            all_chunks.append(Chunk(**c))
            papers_seen.add(c["paper_name"])

    retriever.build_index(all_chunks)
    paper_list = sorted(papers_seen)

    paper_info = "\n".join([f"  • {p[:50]}..." for p in paper_list])
    upload_msg = f"✅ 上传成功！\n论文：{filename}\n章节数：{len(sections)}\n文本块：{len(chunks)}"
    kb_msg = f"当前知识库：\n{paper_info}"
    return upload_msg, kb_msg


def ask_question(question, history):
    """提问：检索 → 生成回答"""
    if not question.strip():
        return history, ""

    if not retriever.is_trained or len(retriever.all_chunks) == 0:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "知识库为空，请先上传论文"},
        ]
        return history, ""

    # 智能检索（自动识别论文引用并过滤），章节级聚合后返回更多chunks
    output = retriever.search(question, top_k=15)
    contexts = output["results"]
    query_info = output["query_info"]

    if not contexts:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "没有找到相关内容，请换个问法"},
        ]
        return history, ""

    # 生成回答
    result = generator.generate(question, contexts)

    # 整理引用（按章节去重，显示所有命中章节）
    seen_sections = set()
    cite_lines = []
    for cite in result["citations"]:
        paper = cite["paper"]
        section = cite['section'][:60] if cite['section'] else ""
        key = (paper, section)
        if key in seen_sections:
            continue
        seen_sections.add(key)
        citation_str = retriever.format_citation(paper)
        page = cite['page']
        if section:
            cite_lines.append(f"{citation_str}\n   章节：{section} | 第 {page} 页")
        else:
            cite_lines.append(f"{citation_str}\n   第 {page} 页")
    citations_text = "\n\n".join(cite_lines)

    answer_with_citations = f"{result['answer']}\n\n---\n📎 引用来源：\n{citations_text}"
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer_with_citations},
    ]

    return history, ""


# =========================
# 构建 Gradio 界面
# =========================

with gr.Blocks(title="AI Paper Assistant") as demo:
    gr.Markdown("# 📚 AI Paper Assistant\n面向科研人员的个人论文知识库")

    with gr.Row():
        # 左侧：上传 + 知识库状态
        with gr.Column(scale=1):
            gr.Markdown("### 📤 上传论文")
            file_input = gr.File(label="选择 PDF 文件", file_types=[".pdf"])
            upload_btn = gr.Button("上传并解析", variant="primary")
            upload_status = gr.Textbox(label="状态", lines=8, interactive=False)

            gr.Markdown("### 📋 知识库")
            kb_status = gr.Textbox(label="已加载论文", lines=5, interactive=False)

        # 右侧：聊天问答
        with gr.Column(scale=2):
            gr.Markdown("### 💬 提问")
            chatbot = gr.Chatbot(height=500, label="对话记录")
            question_input = gr.Textbox(
                label="你的问题",
                placeholder="例如：这篇论文的主要结论是什么？",
                lines=2,
            )
            ask_btn = gr.Button("提问", variant="primary")

    # 事件绑定
    upload_btn.click(upload_pdf, inputs=[file_input], outputs=[upload_status, kb_status])
    ask_btn.click(ask_question, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
    question_input.submit(ask_question, inputs=[question_input, chatbot], outputs=[chatbot, question_input])

    # 启动时加载已有数据
    demo.load(load_existing_papers, outputs=[kb_status])


if __name__ == "__main__":
    print("启动 AI Paper Assistant...")
    print("浏览器打开 http://localhost:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)
