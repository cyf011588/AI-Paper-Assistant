"""ablation_real.py — 真实RRF vs Reranker Ablation（从实际代码路径拦截）"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import PROCESSED_DIR, EVAL_DATASET_PATH
from chunker import Chunk
from smart_retriever import SmartRetriever

# 加载评测数据集
with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 初始化retriever
retriever = SmartRetriever()
all_chunks = []
for cf in Path(PROCESSED_DIR).glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)

# 只跑hybrid_rerank题（我们直接跑全部30题，然后筛选）
results = []
for item in dataset:
    qid = item["question_id"]
    question = item["question"]
    gold_ids = set(item["retrieval_gt"]["relevant_chunk_ids"])

    output = retriever.search(question, top_k=10)
    debug = output.get("debug_info", {})
    mode = debug.get("retrieval_mode", "unknown")

    if mode != "hybrid_rerank":
        continue

    pre_rerank = debug.get("pre_rerank_ids", [])  # RRF+章节加权后top20
    post_rerank = debug.get("post_rerank_ids", [])  # reranker后top10

    pre_top10 = pre_rerank[:10]
    pre_hit = gold_ids & set(pre_top10)
    post_hit = gold_ids & set(post_rerank)

    pre_recall = len(pre_hit) / len(gold_ids) if gold_ids else 0
    post_recall = len(post_hit) / len(gold_ids) if gold_ids else 0

    # 每个gold在pre和post中的rank
    pre_ranks = {}
    post_ranks = {}
    for g in gold_ids:
        if g in pre_top10:
            pre_ranks[g] = pre_top10.index(g) + 1
        elif g in pre_rerank:
            pre_ranks[g] = f">{pre_rerank.index(g)+1}"
        else:
            pre_ranks[g] = "—"
        if g in post_rerank:
            post_ranks[g] = post_rerank.index(g) + 1
        else:
            post_ranks[g] = "—"

    if post_recall > pre_recall:
        effect = "Reranker帮助"
    elif post_recall < pre_recall:
        effect = "Reranker损害"
    else:
        effect = "无变化"

    results.append({
        "qid": qid,
        "question": question[:50],
        "n_gold": len(gold_ids),
        "pre_hit": len(pre_hit),
        "post_hit": len(post_hit),
        "pre_recall": pre_recall,
        "post_recall": post_recall,
        "pre_ranks": pre_ranks,
        "post_ranks": post_ranks,
        "effect": effect,
    })

    print(f"[{qid}] pre={pre_recall:.0%}({len(pre_hit)}/{len(gold_ids)})  post={post_recall:.0%}({len(post_hit)}/{len(gold_ids)})  {effect}")

# 汇总
print("\n" + "="*80)
print("真实 Ablation: Pre-reranker (RRF+加权 Top10) vs Post-reranker Top10")
print("="*80)

n = len(results)
avg_pre = sum(r["pre_recall"] for r in results) / n
avg_post = sum(r["post_recall"] for r in results) / n
delta = avg_post - avg_pre

helped = sum(1 for r in results if r["effect"] == "Reranker帮助")
hurt = sum(1 for r in results if r["effect"] == "Reranker损害")
neutral = sum(1 for r in results if r["effect"] == "无变化")

print(f"\n题数: {n}")
print(f"Pre-reranker Recall@10  = {avg_pre:.1%}")
print(f"Post-reranker Recall@10 = {avg_post:.1%}")
print(f"绝对变化               = {delta:+.1%}")
print(f"Reranker帮助: {helped}题, 损害: {hurt}题, 无变化: {neutral}题")

# 详细表格
print(f"\n{'='*80}")
print(f"{'qid':<6} {'gold':<5} {'pre_hit':<8} {'post_hit':<9} {'pre_ranks':<30} {'post_ranks':<30} effect")
print("-"*80)
for r in results:
    pr = ", ".join(f"{k[-12:]}:{v}" for k,v in r["pre_ranks"].items())
    qr = ", ".join(f"{k[-12:]}:{v}" for k,v in r["post_ranks"].items())
    print(f"{r['qid']:<6} {r['n_gold']:<5} {r['pre_hit']}/{r['n_gold']:<6} {r['post_hit']}/{r['n_gold']:<7} {pr:<30} {qr:<30} {r['effect']}")
