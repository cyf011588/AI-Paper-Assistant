"""debug_hybrid.py — 逐题分析hybrid_rerank的13道题，做错误归因"""
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

for r in hybrid_qs:
    qid = r["qid"]
    question = r["question"]
    gold_ids = set(r["gold_chunk_ids"])
    pre_top10 = r["pre_expansion_top10_ids"]
    
    print(f"{'='*70}")
    print(f"[{qid}] {question[:60]}")
    print(f"  gold: {gold_ids}")
    print(f"  pre_top10: {pre_top10}")
    
    # 命中情况
    hit = gold_ids & set(pre_top10)
    miss = gold_ids - set(pre_top10)
    print(f"  命中: {hit}")
    print(f"  未命中: {miss}")
    
    # 对每个未命中的gold，追踪它在vector/bm25/rrf/rerank中的位置
    for gold_id in miss:
        # 找这个chunk
        gold_chunk = None
        for c in all_chunks:
            if c.id == gold_id:
                gold_chunk = c
                break
        if not gold_chunk:
            print(f"    [GOLD NOT FOUND in chunks!] {gold_id}")
            continue
        
        print(f"\n  --- 追踪未命中gold: {gold_id[:60]}...")
        print(f"      section: {gold_chunk.section_path}")
        print(f"      page: {gold_chunk.page_start}")
        
        # 1. Vector检索：在target_papers内搜，看gold在第几
        # 先做query understanding
        qi = retriever.qu.parse(question)
        target_papers = qi["target_papers"]
        content_query = qi["content_query"]
        
        where = {"paper_name": {"$in": target_papers}} if target_papers else None
        vec_results = retriever.vector_db.search(content_query, top_k=200, where=where)
        vec_ids = [v["id"] for v in vec_results]
        vec_rank = vec_ids.index(gold_id) + 1 if gold_id in vec_ids else None
        vec_score = None
        for v in vec_results:
            if v["id"] == gold_id:
                vec_score = v["score"]
                break
        print(f"      Vector: rank={vec_rank}, score={vec_score:.4f}" if vec_rank else f"      Vector: 未进入top200, score={vec_score}")
        
        # 2. BM25检索
        bm25_tokens = list(jieba.cut(content_query))
        bm25_scores = retriever.bm25.get_scores(bm25_tokens)
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
        bm25_idx = None
        for rank, idx in enumerate(bm25_ranked):
            if retriever.all_chunks[idx].id == gold_id:
                bm25_idx = rank
                break
        bm25_score_val = None
        for idx in range(len(retriever.all_chunks)):
            if retriever.all_chunks[idx].id == gold_id:
                bm25_score_val = bm25_scores[idx]
                break
        if bm25_idx is not None:
            print(f"      BM25: rank={bm25_idx+1}, score={bm25_score_val:.4f}")
        else:
            print(f"      BM25: 未进入, score={bm25_score_val:.4f}")
        
        # 3. RRF后rank（模拟：vector top200 + bm25 top200）
        k = 60
        rrf = {}
        for rank, vid in enumerate(vec_ids):
            rrf[vid] = rrf.get(vid, 0) + 1.0/(k+rank+1)
        for rank, idx in enumerate(bm25_ranked[:200]):
            cid = retriever.all_chunks[idx].id
            rrf[cid] = rrf.get(cid, 0) + 1.0/(k+rank+1)
        sorted_rrf = sorted(rrf.items(), key=lambda x: -x[1])
        rrf_rank = None
        for rank, (cid, score) in enumerate(sorted_rrf):
            if cid == gold_id:
                rrf_rank = rank + 1
                break
        print(f"      RRF: rank={rrf_rank}")
    
    print()
