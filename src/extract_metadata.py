"""
extract_metadata.py — 从已解析的论文中提取元数据（标题、作者、关键词）

原理：
1. 从 Pre-content（论文首页前置内容）中找字号最大的文本块 → 标题
2. 找字号次大的、包含逗号分隔的文本块 → 作者
3. 找包含 "Keywords" 或 "关键词" 的块 → 关键词
4. 保存为 paper_metadata.json，供查询理解模块使用
"""

import json
import re
from pathlib import Path
from typing import List, Dict


PROCESSED_DIR = Path("F:/AI_Project/AI-Paper-Assistant/data/processed")
OUTPUT_FILE = Path("F:/AI_Project/AI-Paper-Assistant/data/paper_metadata.json")


def extract_authors(text: str) -> List[str]:
    """从作者行中提取作者名列表"""
    # 去掉上标标记（a, b, *等）
    cleaned = re.sub(r'[a-z]\s*[,，]?\s*', '', text)  # 去掉单独的上标字母
    cleaned = re.sub(r'\*', '', cleaned)
    cleaned = re.sub(r'\d+', '', cleaned)  # 去掉数字上标

    # 按逗号/分号分割
    authors = re.split(r'[,，;；]', cleaned)
    authors = [a.strip() for a in authors if a.strip() and len(a.strip()) > 1]
    return authors


def extract_metadata_from_paper(structured_path: Path) -> Dict:
    """从单篇论文的结构化JSON中提取元数据"""
    with open(structured_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    paper_name = data["paper_name"]
    sections = data["sections"]

    # 找 Pre-content
    pre_content = None
    for sec in sections:
        if sec.get("section_path") == "Pre-content":
            pre_content = sec
            break

    if not pre_content or not pre_content.get("blocks"):
        # 没有Pre-content，用文件名作为标题
        return {
            "paper_name": paper_name,
            "title": paper_name,
            "authors": [],
            "keywords": [],
            "source": "filename_fallback",
        }

    blocks = pre_content["blocks"]

    # 按字号排序，找最大的几个块
    blocks_by_size = sorted(blocks, key=lambda b: b.get("font_size", 0), reverse=True)

    # 标题：字号最大的、长度合理的文本块
    title = ""
    title_candidates = [b for b in blocks_by_size
                        if b.get("font_size", 0) >= 12
                        and len(b["text"].strip()) > 10
                        and "journal" not in b["text"].lower()
                        and "www." not in b["text"].lower()
                        and "contents" not in b["text"].lower()]
    if title_candidates:
        title = title_candidates[0]["text"].strip()

    # 作者：字号次大的、包含逗号分隔的块
    authors = []
    author_candidates = [b for b in blocks_by_size
                         if 8 <= b.get("font_size", 0) < 13
                         and ("," in b["text"] or "，" in b["text"])
                         and len(b["text"].strip()) < 200
                         and "school" not in b["text"].lower()
                         and "university" not in b["text"].lower()
                         and "department" not in b["text"].lower()
                         and "college" not in b["text"].lower()]
    if author_candidates:
        author_text = author_candidates[0]["text"].strip()
        # 可能作者被分成了多行，合并相邻的作者块
        for b in blocks:
            if b is author_candidates[0]:
                idx = blocks.index(b)
                # 检查下一个块是不是作者的延续
                if idx + 1 < len(blocks):
                    next_b = blocks[idx + 1]
                    if (abs(next_b.get("font_size", 0) - b.get("font_size", 0)) < 1
                            and "," in next_b["text"]
                            and "school" not in next_b["text"].lower()):
                        author_text += " " + next_b["text"].strip()
                break
        authors = extract_authors(author_text)

    # 关键词：找包含 Keywords 或 关键词 的块
    keywords = []
    for b in blocks:
        text = b["text"]
        if "Keywords" in text or "keywords" in text or "关键词" in text:
            # 提取关键词部分
            kw_text = re.sub(r'^(Keywords?|关键词)\s*[:：]\s*', '', text, flags=re.IGNORECASE)
            keywords = [k.strip() for k in re.split(r'[;；,，]', kw_text) if k.strip()]
            break

    return {
        "paper_name": paper_name,
        "title": title,
        "authors": authors,
        "keywords": keywords,
        "source": "pre_content",
    }


def main():
    print("提取论文元数据...\n")

    all_metadata = []
    structured_files = sorted(PROCESSED_DIR.glob("*_structured.json"))

    for sf in structured_files:
        meta = extract_metadata_from_paper(sf)
        all_metadata.append(meta)

        print(f"📄 {meta['paper_name'][:40]}")
        print(f"   标题: {meta['title'][:70]}")
        print(f"   作者: {', '.join(meta['authors'][:5])}")
        print(f"   关键词: {', '.join(meta['keywords'][:5])}")
        print()

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存 {len(all_metadata)} 篇论文元数据到: {OUTPUT_FILE}")

    # 打印作者→论文映射表
    print("\n" + "=" * 60)
    print("作者 → 论文 映射表")
    print("=" * 60)
    author_map = {}
    for meta in all_metadata:
        for author in meta["authors"]:
            if author not in author_map:
                author_map[author] = []
            author_map[author].append(meta["paper_name"])

    for author, papers in sorted(author_map.items()):
        print(f"  {author}: {len(papers)}篇")
        for p in papers:
            print(f"    - {p[:50]}")


if __name__ == "__main__":
    main()
