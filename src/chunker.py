"""
chunker.py — 结构感知 Chunker（父子分块版）

核心思路：
1. 父块 = 一个完整小节（如"6.1 本文结论"整节）
2. 子块 = 父块内按字数切分的语义片段
3. 检索时embedding用子块（小而精），命中后返回整个父块给LLM

这样解决的问题：
- 子块小，embedding更精准
- 父块完整，LLM能看到完整上下文
- 不会因为chunk切碎而丢失结论的完整性
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DIR


@dataclass
class Chunk:
    """一个切分后的文本块（子块）"""
    id: str                    # 唯一标识
    text: str                  # 子块文本（用于embedding和检索）
    paper_name: str            # 来自哪篇论文
    section_path: str          # 属于哪个章节
    page_start: int            # 起始页码
    page_end: int              # 结束页码
    parent_id: str = ""        # 父块ID
    parent_text: str = ""      # 父块完整文本（命中后返回给LLM）
    block_types: List[str] = field(default_factory=list)


class StructuralChunker:
    """结构感知切分器：父子分块"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_paper(self, paper_json_path: Path) -> List[Chunk]:
        """
        读取结构化 JSON，输出父子分块后的 chunk 列表
        """
        with open(paper_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        paper_name = data["paper_name"]
        chunks = []
        chunk_index = 0
        parent_index = 0

        for section in data["sections"]:
            section_path = section["section_path"]
            blocks = section["blocks"]

            if not blocks:
                continue

            # 跳过纯图表目录、参考文献、致谢等（不入库）
            skip_sections = ["目录", "参考文献", "参 考 文 献", "致谢", "致  谢",
                           "作者简介", "独创性声明", "发表论文", "Research Area",
                           "附录", "Appendix"]
            if any(skip in section_path for skip in skip_sections):
                continue

            # 1. 把同一章节的所有正文块拼成父块文本
            parent_text = ""
            page_start = blocks[0]["page"]
            page_end = blocks[-1]["page"]

            for block in blocks:
                btype = block["block_type"]
                text = block["text"]

                if btype == "FIGURE_CAPTION":
                    # 图注保留，标注为图
                    parent_text += f"\n\n[图: {text}]\n\n"
                elif btype == "TABLE_CAPTION":
                    # 表注保留，标注为表
                    parent_text += f"\n\n[表: {text}]\n\n"
                elif btype in ("HEADER", "FOOTER", "SKIP"):
                    # 页眉页脚跳过
                    continue
                else:
                    parent_text += text + " "

            parent_text = parent_text.strip()
            if len(parent_text) < 50:  # 太短的章节跳过
                continue

            # 父块ID
            parent_id = f"{paper_name}_parent_{parent_index:04d}"
            parent_index += 1

            # 2. 父块不太长时，整个就是一个子块
            if len(parent_text) <= self.chunk_size * 2:
                chunk_id = f"{paper_name}_{section_path}_{chunk_index:04d}"
                chunk_index += 1
                chunks.append(Chunk(
                    id=chunk_id,
                    text=parent_text,
                    paper_name=paper_name,
                    section_path=section_path,
                    page_start=page_start,
                    page_end=page_end,
                    parent_id=parent_id,
                    parent_text=parent_text,
                ))
                continue

            # 3. 父块太长，切成多个子块
            sub_chunks = self._split_text(parent_text)

            for i, sub_text in enumerate(sub_chunks):
                chunk_id = f"{paper_name}_{section_path}_{chunk_index:04d}"
                chunk_index += 1
                chunks.append(Chunk(
                    id=chunk_id,
                    text=sub_text,
                    paper_name=paper_name,
                    section_path=section_path,
                    page_start=page_start,
                    page_end=page_end,
                    parent_id=parent_id,
                    parent_text=parent_text,  # 子块都引用同一个父块
                ))

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """把一段长文本切成合适大小的子块"""
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """把文本切成句子"""
        text = text.replace("\n", " ")
        text = re.sub(r'([.!?。！？；;])\s+', r'\1\n', text)
        sentences = text.split("\n")
        return [s.strip() for s in sentences if s.strip()]


# =========================
# 命令行测试
# =========================

if __name__ == "__main__":
    import sys

    json_files = list(PROCESSED_DIR.glob("*_structured.json"))
    if not json_files:
        print("没有找到解析后的 JSON")
        sys.exit()

    # 测试大论文
    for jf in json_files:
        if "毕业论文" in jf.name or "大论文" in jf.name:
            json_path = jf
            break
    else:
        json_path = json_files[0]

    print(f"处理文件: {json_path.name}\n")

    chunker = StructuralChunker()
    chunks = chunker.process_paper(json_path)

    print(f"生成 {len(chunks)} 个子块\n")

    # 统计有多少个父块
    parent_ids = set(c.parent_id for c in chunks)
    print(f"对应 {len(parent_ids)} 个父块\n")

    # 展示前5个
    for c in chunks[:5]:
        print(f"[ID] {c.id}")
        print(f"[章节] {c.section_path}")
        print(f"[子块长度] {len(c.text)} 字")
        print(f"[父块长度] {len(c.parent_text)} 字")
        print(f"[子块内容] {c.text[:100]}...")
        print()

    # 保存
    output_path = PROCESSED_DIR / f"{json_path.stem.replace('_structured', '_chunks')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
    print(f"已保存到: {output_path}")
