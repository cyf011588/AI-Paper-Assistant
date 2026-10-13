"""q026: Reranker query 对照实验 — 原始 subquery vs expanded query"""
import os, sys, json
from pathlib import Path
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import PROCESSED_DIR
from chunker import Chunk
from smart_retriever import SmartRetriever

sr = SmartRetriever()
all_chunks = []
for cf in Path(PROCESSED_DIR).glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
sr.build_index(all_chunks)

gold = {
    "雷杰鑫毕业论文终稿": [
        "雷杰鑫毕业论文终稿_第二章 钢框架-CPSW 有限元模型_0041",
        "雷杰鑫毕业论文终稿_第二章 钢框架-CPSW 有限元模型 / 2.2 有限元模型验证_0054",
    ],
    "王鑫涛论文终稿": [
        "王鑫涛论文终稿_第二章 钢框架两边连接装配式C-PSW/CE 试验研究_0034",
        "王鑫涛论文终稿_第三章 钢框架两边连接装配式C-PSW/CE 有限元模型_0063",
    ],
}

base_queries = {
    "王鑫涛论文终稿": "王鑫涛论文终稿中使用了哪些研究方法？",
    "雷杰鑫毕业论文终稿": "雷杰鑫毕业论文终稿中使用了哪些研究方法？",
}

# Top 8 terms
top8_terms = {
    "王鑫涛论文终稿": ["低周往复加载试验","拟静力试验","试件设计","加载制度","ABAQUS","有限元模型","材料本构模型","滞回曲线"],
    "雷杰鑫毕业论文终稿": ["低周往复加载试验","试件设计","加载制度","ABAQUS","S4R壳单元","C3D8I实体单元","材料本构模型","滞回曲线对比"],
}

def get_rank(cid, id_list):
    return id_list.index(cid)+1 if cid in id_list else None

print(f"{'论文':<4} {'Cond':<20} {'G1 RRF':>7} {'G1 RR':>6} {'G2 RRF':>7} {'G2 RR':>6} {'R@5':>6} {'R@10':>6}")
print("-" * 75)

for paper in ["王鑫涛论文终稿", "雷杰鑫毕业论文终稿"]:
    gold_ids = gold[paper]
    base_q = base_queries[paper]
    terms = top8_terms[paper]
    expanded_q = base_q + " " + " ".join(terms)
    short = "王" if "王" in paper else "雷"

    # RRF 用 expanded query（两种条件相同）
    raw_results = sr._search_single_paper(expanded_q, paper, top_k=100)
    rrf_ids = [r["id"] for r in raw_results]
    g1_rrf = get_rank(gold_ids[0], rrf_ids)
    g2_rrf = get_rank(gold_ids[1], rrf_ids)

    for cond_name, rerank_q in [("原始 subquery", base_q), ("expanded query", expanded_q)]:
        if sr.reranker:
            reranked = sr.reranker.rerank(rerank_q, raw_results, top_k=10)
        else:
            reranked = raw_results[:10]
        rr_ids = [r["id"] for r in reranked]

        g1_rr = get_rank(gold_ids[0], rr_ids)
        g2_rr = get_rank(gold_ids[1], rr_ids)
        r5 = sum(1 for g in gold_ids if g in rr_ids[:5]) / len(gold_ids)
        r10 = sum(1 for g in gold_ids if g in rr_ids[:10]) / len(gold_ids)

        def fmt(v):
            return str(v) if v else "—"

        print(f"{short:<4} {cond_name:<20} {fmt(g1_rrf):>7} {fmt(g1_rr):>6} {fmt(g2_rrf):>7} {fmt(g2_rr):>6} {r5:>6.0%} {r10:>6.0%}")
    print("-" * 75)
