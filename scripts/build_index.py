#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_index.py — 一键重建向量索引

用途
    在新机器 / 新环境上，只要有 data/papers/ 下的 PDF，就能重建出与
    src/app.py「上传论文」流程**完全一致**的 processed/*.json 与 chroma_db 索引，
    因此无需传输几百 MB 的向量库、缓存或 processed 产物。

流程（与 app.py 的 upload_pdf 保持一致）
    1. data/papers/*.pdf          → 解析      → data/processed/<stem>_structured.json
    2. <stem>_structured.json     → 父子分块  → data/processed/<stem>_chunks.json
    3. 全部 chunks                → 写入      → ChromaDB 的 paper_chunks 集合

用法
    python scripts/build_index.py                  # 增量：复用已有解析结果，逐篇入库
    python scripts/build_index.py --rebuild        # 先清空 paper_chunks 集合再全量重建
    python scripts/build_index.py --force-parse    # 忽略已有 structured.json，重新解析 PDF
    python scripts/build_index.py --from-processed # 只处理已有 structured.json 的论文（不解析新 PDF）

说明
    命令行重建**不会**调用 LLM 生成 paper_metadata.json；新增论文的元数据
    （作者 / 年份 / 期刊）会在首次用 Web UI 上传时自动生成。
"""

import argparse
import json
import sys
from pathlib import Path

# 让脚本能 import 到 src/ 下的模块
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from config import PAPERS_DIR, PROCESSED_DIR  # noqa: E402
from pdf_parser import StructuredPDFParser    # noqa: E402
from chunker import StructuralChunker         # noqa: E402
from vector_store import VectorStore          # noqa: E402

COLLECTION = "paper_chunks"


def save_chunks_json(chunks, path: Path) -> None:
    """落盘格式与 src/app.py 的上传流程保持完全一致（供 load_existing_papers 读取）"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{
            "id": c.id, "text": c.text, "paper_name": c.paper_name,
            "section_path": c.section_path, "page_start": c.page_start,
            "page_end": c.page_end,
        } for c in chunks], f, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="一键重建 AI Paper Assistant 向量索引")
    ap.add_argument("--rebuild", action="store_true",
                    help="清空 paper_chunks 集合后全量重建（默认增量）")
    ap.add_argument("--force-parse", action="store_true",
                    help="即使 structured.json 已存在也重新解析 PDF")
    ap.add_argument("--from-processed", action="store_true",
                    help="跳过 PDF 解析，只处理 data/processed/ 下已有 structured.json 的论文")
    args = ap.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[build_index] {PAPERS_DIR} 下没有 PDF 文件，请先放入论文。")
        return 1

    print(f"[build_index] 发现 {len(pdfs)} 个 PDF，开始处理...\n")
    chunker = StructuralChunker()
    vs = VectorStore(collection_name=COLLECTION)

    if args.rebuild:
        print(f"[build_index] --rebuild：清空集合 {COLLECTION} ...")
        try:
            vs.client.delete_collection(COLLECTION)
        except Exception as e:  # 集合不存在时忽略
            print(f"  (忽略) {e}")
        vs.collection = vs.client.create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"})

    total = 0
    processed = 0
    for pdf in pdfs:
        stem = pdf.stem
        structured_path = PROCESSED_DIR / f"{stem}_structured.json"
        chunks_path = PROCESSED_DIR / f"{stem}_chunks.json"

        if args.from_processed and not structured_path.exists():
            print(f"[跳过] {stem}：缺少 structured.json（--from-processed 不解析新 PDF）")
            continue

        if (not structured_path.exists()) or args.force_parse:
            print(f"[1/3] 解析 PDF：{stem}")
            parser = StructuredPDFParser(pdf)
            sections = parser.parse()
            parser.save(sections, structured_path)
        else:
            print(f"[1/3] 复用已有解析结果：{stem}")

        print(f"[2/3] 父子分块：{stem}")
        chunks = chunker.process_paper(structured_path)
        save_chunks_json(chunks, chunks_path)

        print(f"[3/3] 写入向量库：{stem}（{len(chunks)} 个 chunk）")
        if not args.rebuild:
            vs.delete_by_paper(stem)   # 避免与旧数据重复
        vs.add_chunks(chunks)

        total += len(chunks)
        processed += 1
        print()

    print(f"[build_index] 完成：处理 {processed} 篇 / {total} 个 chunk，"
          f"当前集合 {COLLECTION} 共 {vs.count()} 条。")
    print("[build_index] 提示：新增论文的 paper_metadata.json 建议首次用 Web UI 上传时自动生成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
