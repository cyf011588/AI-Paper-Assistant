"""
chunker.py — 结构感知 Chunker

核心思路：
1. 先按章节边界切（不把不同章节的内容混在一个 chunk 里）
2. 章节太长就在内部按句子切（不在句子中间断开）
3. 每个 chunk 带上完整元数据（论文名 / 章节路径 / 页码范围）

对比基线：固定大小切分（naive chunking）
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List

from config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DIR


@dataclass
class Chunk:
    """一个切分后的文本块"""
    id: str                    # 唯一标识，如 "paper1_sec2.1_003"
    text: str                  # 实际文本内容
    paper_name: str            # 来自哪篇论文
    section_path: str          # 属于哪个章节，如 "3 Experimental results / 3.1 ..."
    page_start: int            # 起始页码
    page_end: int              # 结束页码
    block_types: List[str] = field(default_factory=list)  # 这个 chunk 里有哪些类型的块


class StructuralChunker:
    """结构感知切分器：在章节边界和句子边界切，不破坏语义"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_paper(self, paper_json_path: Path) -> List[Chunk]:
        """
        读取结构化 JSON，输出切分后的 chunk 列表
        """
        with open(paper_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        paper_name = data["paper_name"]
        chunks = []
        chunk_index = 0

        for section in data["sections"]:
            section_path = section["section_path"]
            blocks = section["blocks"]

            if not blocks:
                continue

            # 把同一章节的所有正文块拼成一段连续文本
            section_text = ""
            page_start = blocks[0]["page"]
            page_end = blocks[-1]["page"]

            for block in blocks:
                if block["block_type"] in ("FIGURE_CAPTION", "TABLE_CAPTION"):
                    # 图表标题单独成段，用括号标注
                    section_text += f"\n\n[{block['block_type']}: {block['text']}]\n\n"
                else:
                    section_text += block["text"] + " "

            # 把这一段长文本切成多个 chunk
            text_chunks = self._split_text(section_text)

            for i, chunk_text in enumerate(text_chunks):
                chunk_id = f"{paper_name}_{section_path}_{chunk_index:04d}"
                chunk_index += 1

                chunks.append(Chunk(
                    id=chunk_id,
                    text=chunk_text.strip(),
                    paper_name=paper_name,
                    section_path=section_path,
                    page_start=page_start,
                    page_end=page_end,
                ))

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        把一段长文本切成合适大小的 chunks。
        策略：先按句号/问号/感叹号切成句子，然后贪心拼接直到接近 chunk_size。
        """
        # 先按句子切（中英文都兼容）
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 如果当前 chunk 加这句话还没超长，就加进去
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                # 当前 chunk 满了，存起来
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 开始新 chunk，带上一点 overlap（上一句的尾巴）
                if self.overlap > 0 and chunks:
                    # 从上一个 chunk 末尾取 overlap 长度的文字
                    prev = chunks[-1]
                    overlap_text = prev[-self.overlap:]
                    current_chunk = overlap_text + " " + sentence + " "
                else:
                    current_chunk = sentence + " "

        # 别忘了最后一个 chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """把文本切成句子（按 . ! ? 和中文句号问号感叹号）"""
        # 先把换行符替换成空格，避免切断
        text = text.replace("\n", " ")
        # 在中英文句末标点后切分
        text = re.sub(r'([.!?。！？；;])\s+', r'\1\n', text)
        sentences = text.split("\n")
        return [s.strip() for s in sentences if s.strip()]


class NaiveChunker:
    """
    基线切分器：固定大小切分，不管句子边界。
    用来跟 StructuralChunker 做对比实验。
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_paper(self, paper_json_path: Path) -> List[Chunk]:
        with open(paper_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        paper_name = data["paper_name"]
        chunks = []
        chunk_index = 0

        # 把整篇论文所有正文拼在一起，不区分章节
        full_text = ""
        page_map = []  # 记录每个字符大致在第几页

        for section in data["sections"]:
            for block in section["blocks"]:
                if block["block_type"] in ("HEADER", "FOOTER"):
                    continue
                text = block["text"]
                start_pos = len(full_text)
                full_text += text + " "
                # 记录这段文字在第几页
                page_map.append((start_pos, len(full_text), block["page"]))

        # 固定大小切分
        for start in range(0, len(full_text), self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, len(full_text))
            chunk_text = full_text[start:end].strip()
            if not chunk_text:
                continue

            # 找这个 chunk 大概在第几页
            page = self._find_page(start, page_map)

            chunk_id = f"naive_{paper_name}_{chunk_index:04d}"
            chunk_index += 1

            chunks.append(Chunk(
                id=chunk_id,
                text=chunk_text,
                paper_name=paper_name,
                section_path="(naive, no section)",
                page_start=page,
                page_end=page,
            ))

        return chunks

    def _find_page(self, char_pos: int, page_map: list) -> int:
        """根据字符位置找大概在第几页"""
        for start, end, page in page_map:
            if start <= char_pos < end:
                return page
        return page_map[-1][2] if page_map else 1


# =========================
# 命令行测试
# =========================

if __name__ == "__main__":
    import sys

    # 找结构化 JSON 文件
    json_files = list(PROCESSED_DIR.glob("*_structured.json"))
    if not json_files:
        print("没有找到解析后的 JSON，请先运行 pdf_parser.py")
        sys.exit()

    json_path = json_files[0]
    print(f"处理文件: {json_path.name}\n")

    # 对比两种切分方式
    structural = StructuralChunker()
    naive = NaiveChunker()

    structural_chunks = structural.process_paper(json_path)
    naive_chunks = naive.process_paper(json_path)

    print(f"结构感知切分: {len(structural_chunks)} 个 chunks")
    print(f"固定大小切分: {len(naive_chunks)} 个 chunks")
    print()

    print("=" * 70)
    print("结构感知切分示例（前 3 个）:")
    print("=" * 70)
    for c in structural_chunks[:3]:
        print(f"\n[ID] {c.id}")
        print(f"[章节] {c.section_path}")
        print(f"[页码] p.{c.page_start}")
        print(f"[内容] {c.text[:150]}...")

    print()
    print("=" * 70)
    print("固定大小切分示例（前 3 个）:")
    print("=" * 70)
    for c in naive_chunks[:3]:
        print(f"\n[ID] {c.id}")
        print(f"[章节] {c.section_path}")
        print(f"[页码] p.{c.page_start}")
        print(f"[内容] {c.text[:150]}...")

    # 保存结构感知切分的结果
    output_path = PROCESSED_DIR / f"{json_path.stem.replace('_structured', '_chunks.json')}"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in structural_chunks], f, ensure_ascii=False, indent=2)
    print(f"\n结构感知 chunks 已保存到: {output_path}")
