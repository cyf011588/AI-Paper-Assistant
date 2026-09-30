"""
generator.py — LLM 生成模块

把检索到的相关段落 + 用户问题一起发给 DeepSeek，
让它基于材料回答，并且标注引用来源。
"""

import re
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

    def generate(self, question: str, contexts: List[Dict]) -> Dict:
        """
        基于检索到的上下文生成回答。
        """
        # 1. 把上下文拼成带编号的材料
        context_text = self._build_context(contexts)

        # 2. 构造 Prompt
        system_prompt = self._build_system_prompt()
        user_prompt = f"""用户问题：{question}

以下是从论文中检索到的相关材料（按相关性排序）：
{context_text}

请根据以上材料回答用户问题。严格遵守：
1. 只使用材料中的信息，不要编造材料里没有的内容
2. 回答要结构清晰：如果问题问"有几条/分别是什么"，先回答数量，再逐条列出
3. 如果材料不足以完整回答，明确说明哪些部分无法回答，不要猜测
4. 专业术语保留原文，其余用中文回答
"""

        # 3. 调用 DeepSeek
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )

        answer_text = response.choices[0].message.content

        # 4. 清洗正文里的引用编号（[1] [2] [50] 等都去掉）
        answer_text = re.sub(r'\[\d+\]', '', answer_text)
        # 清理多余空格
        answer_text = re.sub(r' +', ' ', answer_text)

        # 5. 整理引用信息
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

    def _build_context(self, contexts: List[Dict]) -> str:
        """把检索结果拼成带编号的材料"""
        parts = []
        for i, ctx in enumerate(contexts):
            ref = f"[{i+1}] {ctx['paper_name']} | {ctx['section_path']} | p.{ctx['page_start']}"
            parts.append(f"{ref}\n内容：{ctx['text']}\n")
        return "\n".join(parts)

    def _build_system_prompt(self) -> str:
        return """你是一个科研论文问答助手。你的任务是根据用户提供的参考材料，准确回答关于学术论文的问题。

规则：
- 只使用参考材料中的信息，不要依赖你自己的知识
- 回答要简洁、准确、有条理
- 如果材料不足以回答问题，明确说明"根据现有材料无法回答"
- 用中文回答，专业术语可以保留英文"""


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
