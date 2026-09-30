"""
batch_process.py — 批量导入论文

把 data/papers/ 目录下所有 PDF 都跑一遍：
PDF → 结构解析 → 切分 → 保存 chunks
最后统一重建语义索引
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from pathlib import Path

from config import PAPERS_DIR, PROCESSED_DIR
from pdf_parser import StructuredPDFParser
from chunker import StructuralChunker, Chunk
from semantic_retriever import SemanticRetriever


def process_one_pdf(pdf_path: Path, chunker: StructuralChunker):
    """处理单篇论文：解析 → 切分 → 保存"""
    name = pdf_path.stem

    # 跳过已经处理过的
    chunks_file = PROCESSED_DIR / f"{name}_chunks.json"
    if chunks_file.exists():
        print(f"  跳过（已处理）: {pdf_path.name}")
        return

    print(f"  正在处理: {pdf_path.name}")

    # 1. 解析
    parser = StructuredPDFParser(pdf_path)
    sections = parser.parse()

    # 保存结构化 JSON
    structured_path = PROCESSED_DIR / f"{name}_structured.json"
    parser.save(sections, structured_path)

    # 2. 切分
    chunks = chunker.process_paper(structured_path)

    # 保存 chunks JSON
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump([{
            "id": c.id, "text": c.text, "paper_name": c.paper_name,
            "section_path": c.section_path, "page_start": c.page_start,
            "page_end": c.page_end,
        } for c in chunks], f, ensure_ascii=False, indent=2)

    print(f"    → {len(sections)} 章节, {len(chunks)} 文本块")


def main():
    # 找所有 PDF
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    print(f"找到 {len(pdf_files)} 篇 PDF\n")

    chunker = StructuralChunker()

    # 批量处理
    for pdf in pdf_files:
        process_one_pdf(pdf, chunker)

    print("\n" + "=" * 60)
    print("全部处理完成！现在重建语义索引...")

    # 加载所有 chunks
    all_chunks = []
    papers_seen = set()
    chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            chunk_dicts = json.load(f)
        for c in chunk_dicts:
            all_chunks.append(Chunk(**c))
            papers_seen.add(c["paper_name"])

    print(f"\n共 {len(papers_seen)} 篇论文，{len(all_chunks)} 个文本块")

    # 建语义索引
    retriever = SemanticRetriever()
    retriever.build_index(all_chunks)

    print("\n论文列表：")
    for p in sorted(papers_seen):
        print(f"  • {p[:60]}")


if __name__ == "__main__":
    main()
