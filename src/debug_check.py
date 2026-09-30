"""
debug_check.py — 精确定位三个问题：
1. 为什么引用显示的是Introduction而不是Conclusion
2. 为什么中文论文检索不到
3. 中文论文章节识别情况
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from pathlib import Path

# 加载所有chunks
PROCESSED_DIR = Path("F:/AI_Project/AI-Paper-Assistant/data/processed")

print("=" * 60)
print("【问题1】中文论文章节识别情况")
print("=" * 60)

for f in sorted(PROCESSED_DIR.glob("*_chunks.json")):
    with open(f, "r", encoding="utf-8") as fp:
        chunks = json.load(fp)
    sections = set(c["section_path"] for c in chunks)
    print(f"\n📄 {f.stem}:")
    print(f"   chunks数: {len(chunks)}")
    print(f"   章节数: {len(sections)}")
    for s in sorted(sections)[:10]:
        count = sum(1 for c in chunks if c["section_path"] == s)
        print(f"     - {s} ({count} chunks)")

print("\n" + "=" * 60)
print("【问题2】雷杰鑫论文Conclusion章节的chunks")
print("=" * 60)

leijie_file = PROCESSED_DIR / "雷杰鑫小论文_chunks.json"
with open(leijie_file, "r", encoding="utf-8") as fp:
    chunks = json.load(fp)

conclusion_chunks = [c for c in chunks if "Conclusion" in c["section_path"] or "conclusion" in c["section_path"].lower()]
print(f"\nConclusion章节chunks数: {len(conclusion_chunks)}")
for c in conclusion_chunks:
    print(f"\n  chunk_id: {c['id']}")
    print(f"  章节: {c['section_path']}")
    print(f"  页码: {c['page_start']}")
    print(f"  前100字: {c['text'][:100]}...")

print("\n" + "=" * 60)
print("【问题3】检索'main conclusions'实际返回了什么")
print("=" * 60)

from smart_retriever import SmartRetriever
from chunker import Chunk

all_chunks = []
for f in PROCESSED_DIR.glob("*_chunks.json"):
    with open(f, "r", encoding="utf-8") as fp:
        for c in json.load(fp):
            all_chunks.append(Chunk(**c))

retriever = SmartRetriever()
retriever.build_index(all_chunks)

# 测试中文问题
q = "雷杰鑫小论文里主要结论有几条"
print(f"\n问题: {q}")
output = retriever.search(q, top_k=15)
print(f"识别论文: {output['query_info']['target_papers']}")
print(f"内容查询: {output['query_info']['content_query']}")
print(f"\n返回的chunks (按score排序):")
for i, r in enumerate(output["results"][:10]):
    print(f"  [{i+1}] score={r['score']:.3f}")
    print(f"      论文: {r['paper_name']}")
    print(f"      章节: {r['section_path']}")
    print(f"      页码: {r['page_start']}")
    print(f"      前80字: {r['text'][:80]}...")

# 测试李泽辉
print("\n" + "=" * 60)
print("【问题4】李泽辉综述检索测试")
print("=" * 60)
q2 = "李泽辉综述主要讲了什么"
print(f"\n问题: {q2}")
output2 = retriever.search(q2, top_k=10)
print(f"识别论文: {output2['query_info']['target_papers']}")
print(f"内容查询: {output2['query_info']['content_query']}")
print(f"返回chunks数: {len(output2['results'])}")
for i, r in enumerate(output2["results"][:5]):
    print(f"  [{i+1}] score={r['score']:.3f} | {r['paper_name'][:30]} | {r['section_path'][:40]}")
