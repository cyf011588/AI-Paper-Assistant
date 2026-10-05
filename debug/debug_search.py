"""
debug_search.py — 调试检索流程
看看问问题时，每一步都发生了什么
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
from config import PROCESSED_DIR
from chunker import Chunk
from smart_retriever import SmartRetriever

# 初始化
retriever = SmartRetriever()

# 加载所有chunks
all_chunks = []
for cf in PROCESSED_DIR.glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        chunk_dicts = json.load(f)
    for c in chunk_dicts:
        all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)

# 测试问题
question = "雷杰鑫毕业论文第4.3节主要讲了什么"
print(f"\n{'='*70}")
print(f"问题: {question}")
print(f"{'='*70}")

# 第一步：查询理解
query_info = retriever.qu.parse(question)
print(f"\n【1. 查询理解】")
print(f"  内容query: {query_info['content_query']}")
print(f"  目标论文: {query_info['target_papers']}")

# 第二步：语义召回
search_query = query_info["content_query"]
if retriever._is_chinese(search_query):
    print(f"\n  [翻译] 目标论文是中文内容，不翻译")
else:
    translated = retriever._translate_to_english(search_query)
    print(f"\n  [翻译] {search_query[:40]} → {translated[:60]}")
    search_query = translated

print(f"\n【2. 语义召回 top 20】")
search_k = 200
semantic_results = retriever.semantic.search(search_query, top_k=search_k)
for i, r in enumerate(semantic_results[:20]):
    print(f"  [{i+1}] score={r['score']:.3f} | {r['paper_name'][:30]} | {r['section_path'][:50]}")

# 第三步：结构化检索
import re
section_keywords = ["结论", "总结", "conclusion", "conclusions",
                   "引言", "绪论", "introduction", "abstract", "摘要",
                   "方法", "试验", "实验", "method", "experiment",
                   "结果", "result", "results", "讨论", "discussion",
                   "设计", "design", "分析", "analysis"]
content_query = query_info["content_query"]
hit_section_kw = None
for kw in section_keywords:
    if kw in content_query.lower():
        hit_section_kw = kw
        break

hit_section_num = None
m = re.search(r'第([一二三四五六七八九十\d]+)章', content_query)
if m:
    cn_num = m.group(1)
    cn_map = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
              "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}
    hit_section_num = cn_map.get(cn_num, cn_num)
if not hit_section_num:
    m = re.search(r'(\d+\.\d+(?:\.\d+)?)', content_query)
    if m:
        hit_section_num = m.group(1)

print(f"\n【3. 结构化检索】")
print(f"  章节关键词: {hit_section_kw}")
print(f"  章节号: {hit_section_num}")

# 第四步：最终返回给LLM的chunks
output = retriever.search(question, top_k=15)
results = output["results"]
print(f"\n【4. 最终返回给LLM的 chunks (共{len(results)}个)】")
for i, r in enumerate(results):
    print(f"  [{i+1}] {r['paper_name'][:30]} | {r['section_path'][:50]} | 页{r['page_start']}")
    print(f"      内容: {r['text'][:80]}...")
