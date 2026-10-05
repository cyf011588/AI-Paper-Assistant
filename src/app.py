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
current_papers = []  # 当前讨论的论文列表（对话记忆，支持多篇）


def load_existing_papers():
    """启动时加载已有的 chunks"""
    global retriever, paper_list

    chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))
    if not chunk_files:
        return "还没有论文，请先上传", []

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

    # 生成Dataframe数据
    table_data = []
    for p in paper_list:
        count = sum(1 for c in all_chunks if c.paper_name == p)
        table_data.append([p, count])

    return table_data, table_data


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

    # 2.5 自动更新paper_metadata.json（用LLM提取metadata）
    metadata_path = Path("F:/AI_Project/AI-Paper-Assistant/data/paper_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    else:
        all_metadata = []

    # 检查是否已存在
    existing = [m for m in all_metadata if m["paper_name"] == pdf_path.stem]
    if not existing:
        # 用LLM从论文前几页提取metadata
        from openai import OpenAI
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

        # 取前3页文本给LLM
        preview_text = ""
        for s in sections[:3]:
            preview_text += f"[{s.get('heading','')}] {s.get('text','')[:500]}\n"

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        llm_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": """你是学术论文元数据提取助手。从论文前几页内容提取以下信息，输出JSON格式：
{"title": "论文标题", "authors": ["作者1","作者2"], "year": 年份或null, "language": "zh或en", "thesis_type": "journal或thesis", "institution": "学校名或期刊名"}
thesis_type: 页数>40页或含"硕士学位论文/博士学位论文"→thesis；否则→journal。
只输出JSON，不要其他内容。"""},
                {"role": "user", "content": preview_text[:3000]}
            ],
            temperature=0.1,
        )
        import json as _json
        extracted = _json.loads(llm_response.choices[0].message.content)

        # 作者别名：中文名+拼音+论文名
        author_names = extracted.get("authors", [])
        aliases = list(author_names)
        # 如果有中文名，加上"XXX大论文""XXX小论文"等变体
        for name in author_names:
            if any('\u4e00' <= c <= '\u9fff' for c in name):
                aliases.append(f"{name}大论文")
                aliases.append(f"{name}毕业论文")
                aliases.append(f"{name}小论文")

        meta = {
            "paper_name": pdf_path.stem,
            "title": extracted.get("title") or pdf_path.stem,
            "authors": author_names,
            "author_aliases": aliases,
            "language": extracted.get("language", "zh"),
            "year": extracted.get("year"),
            "journal": extracted.get("institution", ""),
            "thesis_type": extracted.get("thesis_type", "journal"),
        }
        all_metadata.append(meta)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    # 3. 把新chunks加入ChromaDB（增量更新）
    #    如果之前上传过同名论文，先删掉旧的
    retriever.vector_db.delete_by_paper(pdf_path.stem)
    retriever.vector_db.add_chunks(chunks)

    # 4. 重新加载paper_chunks_map（用于章节聚合）
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

    # 生成Dataframe数据
    table_data = []
    for p in paper_list:
        count = sum(1 for c in all_chunks if c.paper_name == p)
        table_data.append([p, count])

    upload_msg = f"✅ 上传成功！\n论文：{filename}\n章节数：{len(sections)}\n文本块：{len(chunks)}"
    return upload_msg, table_data


def delete_paper(selected_row):
    """删除选中的论文（从Dataframe选中行获取）"""
    global retriever, paper_list
    if not selected_row or len(selected_row) == 0:
        return gr.update(value=[])

    # selected_row是Dataframe选中的行，取第一列（论文名）
    paper_name = selected_row[0][0] if isinstance(selected_row[0], list) else selected_row[0]
    if not paper_name:
        return gr.update(value=[])

    # 1. 从ChromaDB删除
    retriever.vector_db.delete_by_paper(paper_name)

    # 2. 删除chunks JSON文件
    chunks_file = PROCESSED_DIR / f"{paper_name}_chunks.json"
    if chunks_file.exists():
        chunks_file.unlink()
    structured_file = PROCESSED_DIR / f"{paper_name}_structured.json"
    if structured_file.exists():
        structured_file.unlink()

    # 3. 从metadata删除
    metadata_path = Path("F:/AI_Project/AI-Paper-Assistant/data/paper_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
        all_metadata = [m for m in all_metadata if m["paper_name"] != paper_name]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    # 4. 重新加载索引
    chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))
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

    # 更新Dataframe
    table_data = []
    for p in paper_list:
        count = sum(1 for c in all_chunks if c.paper_name == p)
        table_data.append([p, count])

    print(f"✅ 已删除：{paper_name}")
    return gr.update(value=table_data)


def resolve_context(question, history):
    """对话消解：把指代词替换成具体论文名，更新current_papers"""
    global current_papers

    # 指代词列表：单篇和多篇
    single_pronouns = ["这篇", "这个", "它", "该文", "这篇论文", "该论文"]
    multi_pronouns = ["两篇", "这两篇", "那两篇", "它们", "两者", "两篇论文"]

    has_single = any(p in question for p in single_pronouns)
    has_multi = any(p in question for p in multi_pronouns)

    # 如果问题里有指代词，且有记忆，替换成具体论文名
    if (has_single or has_multi) and current_papers:
        if has_multi and len(current_papers) >= 2:
            # 多篇指代词：把"两篇"替换成所有记忆中的论文名
            paper_refs = "、".join([f"《{p}》" for p in current_papers])
            for p in multi_pronouns:
                question = question.replace(p, paper_refs)
        elif has_single and len(current_papers) == 1:
            # 单篇指代词：用最近一篇替换
            question = question.replace("这篇", f"《{current_papers[-1]}》")
            question = question.replace("这个", f"《{current_papers[-1]}》")
            question = question.replace("它", f"《{current_papers[-1]}》")
            question = question.replace("该文", f"《{current_papers[-1]}》")
        return question

    # 如果没有指代词，用LLM判断这轮问题提到了哪篇论文
    from openai import OpenAI
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # 列出所有论文名供LLM参考
    paper_names_str = "\n".join([f"- {p}" for p in paper_list])
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": f"""你是对话消解助手。用户问的问题涉及哪些论文？
可选论文列表：
{paper_names_str}

如果问题明确提到了某篇或某几篇论文，输出这些论文的完整名称，用逗号分隔。
如果问题没有指定论文，输出"无"。
只输出论文名称或"无"，不要解释。"""},
            {"role": "user", "content": question},
        ],
        temperature=0.1,
        max_tokens=200,
    )
    result = response.choices[0].message.content.strip().strip("《》")
    if result != "无":
        # 可能是多篇，用逗号分隔
        mentioned = [r.strip().strip("《》") for r in result.split(",")]
        new_papers = [p for p in mentioned if p in paper_list]
        if new_papers:
            current_papers = new_papers  # 更新当前讨论的论文列表

    return question


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

    # 对话消解：把"这篇论文"替换成具体论文名
    resolved_question = resolve_context(question, history)
    print(f"[DEBUG] 原始问题: {question}")
    print(f"[DEBUG] 消解后问题: {resolved_question}")
    print(f"[DEBUG] current_papers: {current_papers}")

    # 智能检索（自动识别论文引用并过滤），章节级聚合后返回更多chunks
    output = retriever.search(resolved_question, top_k=15)
    contexts = output["results"]
    query_info = output["query_info"]
    print(f"[DEBUG] 查询理解: target_papers={query_info['target_papers']}")
    print(f"[DEBUG] 检索到 {len(contexts)} 个chunk:")
    for i, r in enumerate(contexts[:5]):
        print(f"  [{i+1}] {r['paper_name'][:30]} | {r['section_path'][:40]}")

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

    with gr.Tabs():
        with gr.Tab("📖 问答"):
            with gr.Row():
                # 左侧：上传 + 知识库状态
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 上传论文")
                    file_input = gr.File(label="选择 PDF 文件", file_types=[".pdf"])
                    upload_btn = gr.Button("上传并解析", variant="primary")
                    upload_status = gr.Textbox(label="状态", lines=8, interactive=False)

                    gr.Markdown("### 📋 知识库")
                    kb_table = gr.Dataframe(
                        headers=["论文名称", "文本块数"],
                        datatype=["str", "number"],
                        col_count=2,
                        interactive=False,
                    )
                    delete_btn = gr.Button("🗑️ 删除选中论文", variant="stop")

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

        with gr.Tab("✍️ 论文写作"):
            gr.Markdown("### AI论文写作助手（多智能体：检索→看图写作→去重→审稿）")

            write_request = gr.Textbox(
                label="写作需求",
                placeholder='例如：按照王鑫涛大论文4.2节的样式，分析我上传的滞回曲线',
                lines=3,
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📊 上传实验图片")
                    batch_images = gr.File(
                        label="批量上传图片（png/jpg/tif，支持多选）",
                        file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                        file_count="multiple",
                    )
                    specimen_input = gr.Textbox(label="试件编号（如 SP10）", placeholder="SP10")
                    load_step_input = gr.Textbox(label="荷载步（如 A点/B点/C点）", placeholder="A点")
                    fig_type_input = gr.Dropdown(
                        choices=["curve", "contour", "photo"],
                        value="curve",
                        label="图片类型",
                    )
                    extract_btn = gr.Button("提取图片特征并入库", variant="secondary")
                    extract_status = gr.Textbox(label="提取状态", lines=5, interactive=False)

            write_my_file = gr.File(
                label="补充Excel数据（可选）",
                file_types=[".xlsx", ".xls", ".csv"],
            )
            write_btn = gr.Button("开始写作", variant="primary")
            write_output = gr.Markdown(label="写作结果")
            write_history = gr.Textbox(label="Agent内部思考过程", lines=10, interactive=False)

        with gr.Tab("🗂️ 知识库管理"):
            gr.Markdown("### 📄 论文列表")
            kb_table_mgmt = gr.Dataframe(
                headers=["论文名称", "文本块数"],
                datatype=["str", "number"],
                col_count=2,
                interactive=False,
            )
            gr.Markdown("### 🖼️ 已入库图片")
            fig_list_text = gr.Textbox(label="图片列表", lines=8, interactive=False)
            fig_refresh_btn = gr.Button("刷新列表", variant="secondary")
            fig_delete_id = gr.Textbox(label="输入要删除的图号", placeholder="sp-11-m")
            with gr.Row():
                fig_delete_btn = gr.Button("删除指定图片", variant="stop")
                fig_clear_btn = gr.Button("清空所有图片", variant="stop")
            fig_status = gr.Textbox(label="状态", lines=2, interactive=False)

    # 事件绑定
    upload_btn.click(upload_pdf, inputs=[file_input], outputs=[upload_status, kb_table])
    ask_btn.click(ask_question, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
    question_input.submit(ask_question, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
    delete_btn.click(delete_paper, inputs=[kb_table], outputs=[kb_table])

    # 写作助手
    def do_writing(request, batch_imgs, excel_file):
        from writing_agent import WritingAgent
        from multimodal_parser import parse_file, save_to_my_results

        # 处理Excel
        if excel_file:
            try:
                text, ftype = parse_file(excel_file)
                save_to_my_results(text, excel_file.split("\\")[-1], ftype)
            except:
                pass

        # 取第一张图传给Qwen-VL直接看
        img_path = batch_imgs[0] if batch_imgs else ""

        agent = WritingAgent(retriever=retriever)
        result = agent.run(request, img_path)
        history_text = "\n".join(result.get("history", []))
        return result.get("final_text") or result.get("draft", ""), history_text

    write_btn.click(do_writing, inputs=[write_request, batch_images, write_my_file], outputs=[write_output, write_history])

    # 批量提取图片特征
    def do_extract(files, specimen, load_step, fig_type):
        if not files:
            return "请先上传图片"
        from vision_extractor import extract_figure_features
        import chromadb
        from chromadb.utils import embedding_functions
        results = []
        client = chromadb.PersistentClient(path="data/chroma_db")
        bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-base-zh-v1.5",
        )
        col = client.get_or_create_collection("figure_db", embedding_function=bge_ef)
        for f in files:
            try:
                fname = os.path.basename(f)
                features = extract_figure_features(f, fig_type)
                fid = os.path.splitext(fname)[0]
                text = json.dumps(features, ensure_ascii=False)
                col.add(
                    ids=[fid],
                    documents=[text],
                    metadatas=[{"figure_id": fid, "specimen": specimen, "load_step": load_step, "type": fig_type}],
                )
                results.append(f"✓ {fid} (已入库)")
            except Exception as e:
                results.append(f"✗ {fname}: {e}")
        return "\n".join(results)

    extract_btn.click(do_extract, inputs=[batch_images, specimen_input, load_step_input, fig_type_input], outputs=[extract_status])

    # 知识库管理tab
    def list_figures():
        import chromadb
        from chromadb.utils import embedding_functions
        client = chromadb.PersistentClient(path="data/chroma_db")
        try:
            bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-zh-v1.5")
            col = client.get_collection("figure_db", embedding_function=bge_ef)
            data = col.get(include=["metadatas"])
            lines = [f"{m.get('figure_id','')} | {m.get('specimen','')} | {m.get('load_step','')} | {m.get('type','')}"
                     for m in data["metadatas"]]
            return "\n".join(lines) if lines else "（暂无图片）"
        except:
            return "（暂无图片）"

    def delete_figure(fid):
        import chromadb
        client = chromadb.PersistentClient(path="data/chroma_db")
        try:
            col = client.get_collection("figure_db")
            col.delete(where={"figure_id": fid})
            return f"已删除: {fid}"
        except Exception as e:
            return f"删除失败: {e}"

    def clear_figures():
        import chromadb
        client = chromadb.PersistentClient(path="data/chroma_db")
        try:
            client.delete_collection("figure_db")
            return "已清空所有图片记录"
        except:
            return "没有图片记录"

    fig_refresh_btn.click(list_figures, outputs=[fig_list_text])
    fig_delete_btn.click(delete_figure, inputs=[fig_delete_id], outputs=[fig_status]).then(list_figures, outputs=[fig_list_text])
    fig_clear_btn.click(clear_figures, outputs=[fig_status]).then(list_figures, outputs=[fig_list_text])

    # 启动时加载已有数据
    demo.load(load_existing_papers, outputs=[kb_table, kb_table_mgmt])


if __name__ == "__main__":
    print("启动 AI Paper Assistant...")
    print("浏览器打开 http://localhost:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)
