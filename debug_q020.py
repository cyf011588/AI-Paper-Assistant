"""debug_q020.py — 精确追踪q020每个gold chunk在各阶段的rank"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import PROCESSED_DIR
from chunker import Chunk
from smart_retriever import SmartRetriever
import jieba

retriever = SmartRetriever()
all_chunks = []
for cf in Path(PROCESSED_DIR).glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)

question = "王鑫涛大论文中，λ（耗能梁段与钢板墙竖向剪切屈服承载力之比）如何影响屈服模式？对应的承载力公式有何不"

# gold chunks
golds = [
    "王鑫涛论文终稿_第六章 钢框架两边连接装配式C-PSW/CE 承载力和梁截 / 6.2 钢框架两边连接装配式C-PSW/CE 剪切屈服承载力与刚度计算公式_0133",
    "王鑫涛论文终稿_第六章 钢框架两边连接装配式C-PSW/CE 承载力和梁截 / 6.2 钢框架两边连接装配式C-PSW/CE 剪切屈服承载力与刚度计算公式_0134",
    "王鑫涛论文终稿_第六章 钢框架两边连接装配式C-PSW/CE 承载力和梁截 / 6.2 钢框架两边连接装配式C-PSW/CE 剪切屈服承载力与刚度计算公式_0135",
    "王鑫涛论文终稿_第六章 钢框架两边连接装配式C-PSW/CE 承载力和梁截 / 6.2 钢框架两边连接装配式C-PSW/CE 剪切屈服承载力与刚度计算公式_0138",
]

# 1. 先跑完整search，看debug_info
output = retriever.search(question, top_k=10)
debug = output.get("debug_info", {})
pre_ids = debug.get("pre_expansion_ids", [])
print("=== 实际代码 pre_expansion_ids (reranker后) ===")
for i, cid in enumerate(pre_ids):
    mark = " <<< GOLD" if cid in golds else ""
    print(f"  rank={i+1}: {cid[:70]}{mark}")

# 2. 手动复现RRF阶段
qi = retriever.qu.parse(question)
content_query = qi["content_query"]
target_papers = qi["target_papers"]
print(f"\nquery understanding: target_papers={target_papers}")
print(f"content_query: {content_query}")

# brainstorm扩展
expanded_query = retriever._brainstorm_query(content_query)
print(f"expanded_query: {expanded_query}")

# vector
where = {"paper_name": {"$in": target_papers}} if target_papers else None
vec_results = retriever.vector_db.search(expanded_query, top_k=200, where=where)
vec_ids = [v["id"] for v in vec_results]

# bm25
bm25_tokens = list(jieba.cut(expanded_query))
bm25_scores = retriever.bm25.get_scores(bm25_tokens)
bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
bm25_ids = []
for idx in bm25_ranked[:200]:
    c = retriever.all_chunks[idx]
    if target_papers and c.paper_name not in target_papers:
        continue
    bm25_ids.append(c.id)

# RRF
k = 60
rrf = {}
for rank, vid in enumerate(vec_ids):
    rrf[vid] = rrf.get(vid, 0) + 1.0/(k+rank+1)
for rank, bid in enumerate(bm25_ids):
    rrf[bid] = rrf.get(bid, 0) + 1.0/(k+rank+1)
sorted_rrf = sorted(rrf.items(), key=lambda x: -x[1])

print("\n=== RRF排序后，4个gold的位置 ===")
for gold in golds:
    found = False
    for rank, (cid, score) in enumerate(sorted_rrf):
        if cid == gold:
            print(f"  RRF rank={rank+1}: {gold[:70]}  score={score:.6f}")
            found = True
            break
    if not found:
        print(f"  RRF: NOT FOUND in top200: {gold[:70]}")

# 3. 看reranker的candidates是filtered[:20]
# filtered = RRF排序后 + 章节加权 + 重新排序
# 让我们模拟章节加权
section_keywords = ["结论", "总结", "conclusion", "conclusions",
                   "引言", "绪论", "introduction", "abstract", "摘要",
                   "方法", "试验", "实验", "method", "experiment",
                   "结果", "result", "results", "讨论", "discussion",
                   "设计", "design", "分析", "analysis"]
query_lower = content_query.lower()

# 重建filtered（和smart_retriever一样）
all_results = []
merged = {}
for v in vec_results:
    merged[v["id"]] = v
for i, bid in enumerate(bm25_ids):
    if bid not in merged:
        for c in all_chunks:
            if c.id == bid:
                merged[bid] = {"id": bid, "text": c.text, "paper_name": c.paper_name,
                              "section_path": c.section_path, "page_start": c.page_start,
                              "page_end": c.page_end, "parent_id": c.parent_id,
                              "parent_text": c.parent_text, "score": float(bm25_scores[i])}
                break
    merged[bid]["score"] = rrf[bid]

all_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

# 章节名加权
for r in all_results:
    section_no_space = r["section_path"].replace(" ", "").lower()
    for kw in section_keywords:
        kw_no_space = kw.replace(" ", "")
        if kw in query_lower and kw_no_space in section_no_space:
            r["score"] += 0.2
            break

all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

print("\n=== 章节加权后，4个gold的位置 ===")
for gold in golds:
    for rank, r in enumerate(all_results[:50]):
        if r["id"] == gold:
            print(f"  filtered rank={rank+1}: {gold[:70]}  score={r['score']:.6f}")
            break
    else:
        print(f"  filtered: NOT in top50: {gold[:70]}")

# 4. reranker candidates = filtered[:20]
print(f"\n=== Reranker candidates (filtered[:20]) ===")
candidates = all_results[:20]
for i, r in enumerate(candidates):
    mark = " <<< GOLD" if r["id"] in golds else ""
    print(f"  cand rank={i+1}: {r['id'][:70]}{mark}")

# 5. rerank
if retriever.reranker:
    reranked = retriever.reranker.rerank(content_query, candidates, top_k=10)
    print(f"\n=== Reranker top10 ===")
    for i, r in enumerate(reranked):
        mark = " <<< GOLD" if r["id"] in golds else ""
        print(f"  rerank rank={i+1}: {r['id'][:70]}{mark}")
