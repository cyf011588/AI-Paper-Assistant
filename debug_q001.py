"""debug_q001.py — 追踪q001/q004/q008/q009实际匹配过程"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import PROCESSED_DIR
from chunker import Chunk
from smart_retriever import SmartRetriever
import json, re

retriever = SmartRetriever()
all_chunks = []
for cf in Path(PROCESSED_DIR).glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)

questions = {
    "q001": "雷杰鑫大论文《钢框架内填钢板外包混凝土组合剪力墙的滞回性能与抗震设计方法》研究的核心构件是什么？简称是什么？",
    "q004": "雷杰鑫大论文《钢框架内填钢板外包混凝土组合剪力墙的滞回性能与抗震设计方法》的主要研究内容有哪些？",
    "q008": "王鑫涛大论文第四章的研究主题是什么？",
    "q009": "雷杰鑫大论文第五章提出了哪些设计方法与公式？分别位于哪些小节？",
}

for qid, q in questions.items():
    print(f"\n{'='*70}")
    print(f"[{qid}] {q}")
    print(f"{'='*70}")

    # 1. QueryUnderstanding
    qi = retriever.qu.parse(q)
    print(f"target_papers: {qi['target_papers']}")
    print(f"content_query: {qi['content_query']}")

    content_query = qi["content_query"]
    target_papers = qi["target_papers"]

    # 2. section_keywords匹配
    section_keywords = ["结论", "总结", "conclusion", "conclusions",
                       "引言", "绪论", "introduction", "abstract", "摘要",
                       "方法", "试验", "实验", "method", "experiment",
                       "结果", "result", "results", "讨论", "discussion",
                       "设计", "design", "分析", "analysis"]
    query_lower = content_query.lower()
    hit_section_kw = None
    for kw in section_keywords:
        if kw in query_lower:
            hit_section_kw = kw
            break
    print(f"hit_section_kw: '{hit_section_kw}'")

    # 3. 章节号匹配
    hit_section_num = None
    m = re.search(r'第([一二三四五六七八九十\d]+)章', content_query)
    if m:
        cn_num = m.group(1)
        cn_map = {"一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","十":"10"}
        hit_section_num = cn_map.get(cn_num, cn_num)
    if not hit_section_num:
        m = re.search(r'(\d+\.\d+(?:\.\d+)?)', content_query)
        if m:
            hit_section_num = m.group(1)
    print(f"hit_section_num: '{hit_section_num}'")

    # 4. shortcut条件
    is_comparison = False  # 这些题都不是对比题
    shortcut_condition = bool(target_papers and (hit_section_kw or hit_section_num) and not is_comparison)
    print(f"shortcut条件: target_papers={bool(target_papers)} AND (kw={bool(hit_section_kw)} OR num={bool(hit_section_num)}) AND not_comparison=True")
    print(f"  → 进入shortcut: {shortcut_condition}")

    if shortcut_condition:
        # 看实际匹配了哪些chunks
        matched_chunks = []
        for paper_name in target_papers:
            for c in retriever.paper_chunks_map.get(paper_name, []):
                matched = False
                if hit_section_kw:
                    sp = c.section_path.replace(" ","").lower()
                    kw = hit_section_kw.replace(" ","").lower()
                    if kw in sp:
                        matched = True
                if hit_section_num and not matched:
                    if hit_section_num in c.section_path:
                        matched = True
                if matched:
                    matched_chunks.append(c)
        print(f"  匹配到的chunks数: {len(matched_chunks)}")
        for c in matched_chunks[:5]:
            print(f"    - {c.section_path[:60]} | {c.id[-12:]}")
