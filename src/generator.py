"""
generator.py — LLM 生成模块

把检索到的相关段落 + 用户问题一起发给 DeepSeek，
让它基于材料回答，并且标注引用来源。
"""

import re
import jieba
from openai import OpenAI
from typing import List, Dict

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


class AnswerGenerator:
    """调用大模型生成回答"""

    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        self.model = DEEPSEEK_MODEL

    def _compress_context(self, text: str, question: str, max_sentences: int = 3) -> str:
        """上下文压缩：按句子与问题的相关度打分，只保留最相关的2-3句。"""
        # 按句子切分（中文句号/分号/换行）
        sentences = re.split(r'(?<=[。；！？\n])', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) <= max_sentences:
            return text

        # 提取问题的关键词
        question_words = set(jieba.cut(question))
        question_words = {w for w in question_words if len(w) >= 2}

        # 给每个句子打分：与问题关键词的重合度
        scored = []
        for sent in sentences:
            sent_words = set(jieba.cut(sent))
            overlap = len(question_words & sent_words)
            scored.append((overlap, sent))

        # 取分数最高的max_sentences句，保持原文顺序
        scored.sort(key=lambda x: -x[0])
        top_sents = [s for _, s in scored[:max_sentences]]
        # 按原文顺序排列
        result = [s for s in sentences if s in top_sents]
        return "".join(result) if result else text

    def generate(self, question: str, contexts: List[Dict]) -> Dict:
        """
        基于检索到的上下文生成回答。
        """
        # 1. 上下文压缩 + 拼成材料
        # 列举型问题不压缩（需要完整列出多个部件/步骤）
        list_keywords = ["分别", "有哪些", "列出", "哪些", "哪几个", "包括", "单元类型"]
        is_list_question = any(kw in question for kw in list_keywords)
        if not is_list_question:
            for ctx in contexts:
                ctx["text"] = self._compress_context(ctx["text"], question)
        else:
            print(f"  [不压缩] 列举型问题，保留完整上下文")

        # 1.5 视觉分析暂时关闭
        figure_chunks = []  # 暂时关闭视觉分析
        if figure_chunks:
            print(f"  [视觉] 检测到{len(figure_chunks)}个图表chunk，提取原图...")
            vision_descriptions = self._describe_figures(figure_chunks, question)
            if vision_descriptions:
                contexts.append({
                    "paper_name": "（视觉分析）",
                    "section_path": "图表直接识别",
                    "page_start": 0,
                    "text": vision_descriptions,
                })

        context_text = self._build_context(contexts)

        # 2. 检测是否是对比问题
        is_comparison = any(kw in question for kw in ["对比", "区别", "不同", "差异", "共同点", "分别", "比较", "相比", "异同"])
        comparison_instruction = ""
        if is_comparison:
            comparison_instruction = """
注意：这是一个跨论文对比问题。请按以下结构回答：
1. 先分别列出每篇论文的相关内容（标明论文名）
2. 再总结共同点和不同点
3. 如果某篇论文的材料缺失，明确说明该论文相关内容无法从材料中获取
"""

        # 3. 构造 Prompt
        system_prompt = self._build_system_prompt()
        user_prompt = f"""用户问题：{question}

以下是从论文中检索到的相关材料（按相关性排序）：
{context_text}

请严格基于以上材料回答。如果材料中没有明确提及答案，直接回答"根据现有材料无法回答"，绝不允许跨章节推测或使用外部知识。
{comparison_instruction}
"""

        # 4. 调用 DeepSeek
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )

        answer_text = response.choices[0].message.content

        # 5. 清洗正文里的引用编号（[1] [2] [50] 等都去掉）
        answer_text = re.sub(r'\[\d+\]', '', answer_text)
        answer_text = re.sub(r' +', ' ', answer_text)

        # 6. 整理引用信息
        citations = []
        for i, ctx in enumerate(contexts):
            citations.append({
                "ref_id": f"[{i+1}]",
                "paper": ctx["paper_name"],
                "section": ctx["section_path"],
                "page": ctx["page_start"],
            })

        return {
            "answer": answer_text,
            "citations": citations,
        }

    def _describe_figures(self, figure_chunks: List[Dict], question: str) -> str:
        """提取图表对应的PDF原图，发给Qwen-VL直接看图回答。"""
        import base64, fitz
        from pathlib import Path
        from openai import OpenAI
        from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, PAPERS_DIR

        if not SILICONFLOW_API_KEY:
            return ""

        # 建立 paper_name → PDF路径 的映射
        pdf_map = {}
        for pdf_path in Path(PAPERS_DIR).glob("*.pdf"):
            pdf_map[pdf_path.stem] = str(pdf_path)

        client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
        descriptions = []

        for fc in figure_chunks[:3]:  # 最多看3张图
            paper_name = fc["paper_name"]
            page = fc["page_start"]
            caption = fc["section_path"][:50]

            # 找PDF
            pdf_path = pdf_map.get(paper_name)
            if not pdf_path:
                print(f"    找不到PDF: {paper_name}")
                continue

            try:
                doc = fitz.open(pdf_path)
                page_obj = doc[page - 1]  # 0-indexed
                # 渲染整页为图片（比提取单个图片更可靠）
                pix = page_obj.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                doc.close()

                b64 = base64.b64encode(img_bytes).decode()
                resp = client.chat.completions.create(
                    model="Qwen/Qwen2.5-VL-72B-Instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": f"这是论文第{page}页（图注：{caption}）。用户问：{question}\n请直接看图回答，描述图中展示了什么（曲线形状、坐标轴、关键数据点、趋势）。100字以内。"}
                        ]
                    }],
                    temperature=0.1,
                    max_tokens=200,
                )
                desc = resp.choices[0].message.content.strip()
                descriptions.append(f"【视觉分析 p.{page}】{desc}")
                print(f"    p.{page} 视觉描述完成")
            except Exception as e:
                print(f"    p.{page} 视觉分析失败: {e}")

        return "\n".join(descriptions)

    def _build_context(self, contexts: List[Dict]) -> str:
        """把检索结果拼成带编号的材料。如果有group_paper字段，按论文分组标注。"""
        # 检查是否是对比问题（有group_paper字段）
        has_groups = any(ctx.get("group_paper") for ctx in contexts)
        if not has_groups:
            parts = []
            for i, ctx in enumerate(contexts):
                ref = f"[{i+1}] {ctx['paper_name']} | {ctx['section_path']} | p.{ctx['page_start']}"
                parts.append(f"{ref}\n内容：{ctx['text']}\n")
            return "\n".join(parts)

        # 对比问题：按论文分组
        groups = {}
        for ctx in contexts:
            g = ctx.get("group_paper", ctx["paper_name"])
            if g not in groups:
                groups[g] = []
            groups[g].append(ctx)

        parts = []
        idx = 1
        for paper_name, chunks in groups.items():
            parts.append(f"===== 【{paper_name} 的上下文】 =====")
            for ctx in chunks:
                ref = f"[{idx}] {ctx['section_path']} | p.{ctx['page_start']}"
                parts.append(f"{ref}\n内容：{ctx['text']}\n")
                idx += 1
        return "\n".join(parts)

    def _build_system_prompt(self) -> str:
        return """你是一个科研论文问答助手。你的任务是根据用户提供的参考材料，准确回答关于学术论文的问题。

铁律：
- 必须严格基于提供的【章节上下文】回答
- 如果上下文没有明确提及，直接回答"根据现有材料无法回答"
- 绝不允许跨章节推测，绝不允许使用外部知识
- 回答要简洁、准确、有条理
- 用中文回答，专业术语保留英文"""


# =========================
# 命令行测试
# =========================

if __name__ == "__main__":
    import sys
    import json
    from config import PROCESSED_DIR
    from chunker import Chunk
    from retriever import TFIDFRetriever

    # 1. 加载 chunks 并建检索索引
    chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))
    if not chunk_files:
        print("没有找到 chunks 文件")
        sys.exit()

    with open(chunk_files[0], "r", encoding="utf-8") as f:
        chunk_dicts = json.load(f)
    chunks = [Chunk(**c) for c in chunk_dicts]

    retriever = TFIDFRetriever()
    retriever.build_index(chunks)

    # 2. 初始化生成器
    generator = AnswerGenerator()

    # 3. 测试
    test_question = "What is the main failure mode of CPSW-PF-SF?"
    print(f"问题: {test_question}\n")

    # 检索
    contexts = retriever.search(test_question, top_k=5)
    print(f"检索到 {len(contexts)} 个相关段落\n")

    # 生成回答
    result = generator.generate(test_question, contexts)

    print("=" * 70)
    print("回答:")
    print("=" * 70)
    print(result["answer"])
    print()
    print("=" * 70)
    print("引用来源:")
    print("=" * 70)
    for cite in result["citations"]:
        print(f"  {cite['ref_id']} {cite['paper'][:40]}... | {cite['section'][:40]}... | p.{cite['page']}")
