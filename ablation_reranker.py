"""ablation_reranker.py — Ablation: RRF Top10 vs Reranker Top10"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import PROCESSED_DIR
from chunker import Chunk
from smart_retriever import SmartRetriever
import jieba

# 加载评测结果
with open("evaluation/eval_v3_result.json", "r", encoding="utf-8") as f:
    eval_results = json.load(f)

# 初始化retriever
retriever = SmartRetriever()
all_chunks = []
for cf in Path(PROCESSED_DIR).glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)

# 只看hybrid_rerank题
hybrid_qs = [r for r in eval_results if r["retrieval_mode"] == "hybrid_rerank"]
print(f"共{len(hybrid_qs)}道hybrid_rerank题\n")

results = []
rrf_helped = 0
rrf_hurt = 0
rrf_neutral = 0

for r in hybrid_qs:
    qid = r["qid"]
    question = r["question"]
    gold_ids = set(r["gold_chunk_ids"])
    reranker_top10 = r["pre_expansion_top10_ids"]  # 这是reranker后的top10

    # 重新跑检索，但在RRF后、reranker前拦截
    qi = retriever.qu.parse(question)
    content_query = qi["content_query"]
    target_papers = qi["target_papers"]

    # 向量检索
    where = {"paper_name": {"$in": target_papers}} if target_papers else None
    vec_results = retriever.vector_db.search(content_query, top_k=200, where=where)
    vec_ids = [v["id"] for v in vec_results]

    # BM25
    bm25_tokens = list(jieba.cut(content_query))
    bm25_scores = retriever.bm25.get_scores(bm25_tokens)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
    bm25_ids = [retriever.all_chunks[i].id for i in bm25_ranked[:200]]
    # 过滤target_papers
    if target_papers:
        bm25_ids = [cid for cid in bm25_ids
                    if any(c.id == cid and c.paper_name in target_papers for c in retriever.all_chunks)]

    # RRF
    k = 60
    rrf = {}
    for rank, vid in enumerate(vec_ids):
        rrf[vid] = rrf.get(vid, 0) + 1.0/(k+rank+1)
    for rank, bid in enumerate(bm25_ids):
        rrf[bid] = rrf.get(bid, 0) + 1.0/(k+rank+1)
    sorted_rrf = sorted(rrf.items(), key=lambda x: -x[1])
    rrf_top10 = [cid for cid, _ in sorted_rrf[:10]]

    # 计算Recall
    rrf_hit = gold_ids & set(rrf_top10)
    rrf_recall = len(rrf_hit) / len(gold_ids) if gold_ids else 0

    reranker_hit = gold_ids & set(reranker_top10)
    reranker_recall = len(reranker_hit) / len(gold_ids) if gold_ids else 0

    # 标记变化
    rrf_only = rrf_hit - reranker_hit  # RRF有但reranker丢了
    reranker_only = reranker_hit - rrf_hit  # reranker新增

    if reranker_recall > rrf_recall:
        rrf_helped += 1
    elif reranker_recall < rrf_recall:
        rrf_hurt += 1
    else:
        rrf_neutral += 1

    results.append({
        "qid": qid,
        "question": question[:50],
        "n_gold": len(gold_ids),
        "rrf_hit": len(rrf_hit),
        "rrf_recall": rrf_recall,
        "reranker_hit": len(reranker_hit),
        "reranker_recall": reranker_recall,
        "rrf_only_gold": list(rrf_only),
        "reranker_only_gold": list(reranker_only),
    })

    change = reranker_recall - rrf_recall
    sign = "+" if change > 0 else ""
    print(f"[{qid}] RRF={rrf_recall:.0%}({len(rrf_hit)}/{len(gold_ids)})  Reranker={reranker_recall:.0%}({len(reranker_hit)}/{len(gold_ids)})  变化={sign}{change:.0%}")
    if rrf_only:
        print(f"    Reranker丢失: {[g[:50] for g in rrf_only]}")
    if reranker_only:
        print(f"    Reranker新增: {[g[:50] for g in reranker_only]}")

# 汇总
print("\n" + "="*70)
print("Ablation: RRF Top10 vs Reranker Top10")
print("="*70)

avg_rrf = sum(r["rrf_recall"] for r in results) / len(results)
avg_reranker = sum(r["reranker_recall"] for r in results) / len(results)
net = avg_reranker - avg_rrf

print(f"\nRRF Top10 Recall@10      = {avg_rrf:.1%}")
print(f"Reranker Top10 Recall@10 = {avg_reranker:.1%}")
print(f"Reranker净变化           = {net:+.1%}")
print(f"\nReranker帮助的题: {rrf_helped}")
print(f"Reranker损害的题: {rrf_hurt}")
print(f"Reranker无影响:   {rrf_neutral}")

# 列出所有丢失和新增
print("\n--- Reranker丢失的gold（RRF有→Reranker无）---")
for r in results:
    if r["rrf_only_gold"]:
        print(f"  [{r['qid']}]")
        for g in r["rrf_only_gold"]:
            print(f"    {g[:70]}")

print("\n--- Reranker新增的gold（RRF无→Reranker有）---")
for r in results:
    if r["reranker_only_gold"]:
        print(f"  [{r['qid']}]")
        for g in r["reranker_only_gold"]:
            print(f"    {g[:70]}")
