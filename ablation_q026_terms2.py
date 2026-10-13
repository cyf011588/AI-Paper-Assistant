"""q026: 测试不同 search_terms 数量对 Recall 的影响（修正版，区分 reranked vs RRF rank）"""
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

all_terms = {
    "王鑫涛论文终稿": ["低周往复加载试验","拟静力试验","试件设计","加载制度","ABAQUS","有限元模型","材料本构模型","滞回曲线","骨架曲线","试验与有限元对比","参数分析","理论推导","计算公式"],
    "雷杰鑫毕业论文终稿": ["低周往复加载试验","试件设计","加载制度","ABAQUS","S4R壳单元","C3D8I实体单元","材料本构模型","滞回曲线对比","骨架曲线对比","有限元结果与试验对比","理论推导","参数分析","有限元模型","试验现象","破坏模式"],
}

settings = [3, 5, 8, 10, 999]

print(f"{'论文':<4} {'Terms':>5} {'G1 RRF':>7} {'G1 RR':>6} {'G2 RRF':>7} {'G2 RR':>6} {'R@5':>6} {'R@10':>6}")
print("-" * 65)

for paper in ["王鑫涛论文终稿", "雷杰鑫毕业论文终稿"]:
    gold_ids = gold[paper]
    base_q = base_queries[paper]
    terms_list = all_terms[paper]
    short = "王" if "王" in paper else "雷"

    for n in settings:
        used_terms = terms_list[:n]
        actual_n = len(used_terms)
        expanded = base_q + " " + " ".join(used_terms)

        # RRF top100
        raw_results = sr._search_single_paper(expanded, paper, top_k=100)
        rrf_ids = [r["id"] for r in raw_results]

        # Reranker top10（和正式路径一致：rerank用原始subquery）
        if sr.reranker:
            reranked = sr.reranker.rerank(base_q, raw_results, top_k=10)
        else:
            reranked = raw_results[:10]
        rr_ids = [r["id"] for r in reranked]

        # gold 在 RRF 和 Reranker 中的排名
        def get_rank(cid, id_list):
            return id_list.index(cid)+1 if cid in id_list else None

        g1_rrf = get_rank(gold_ids[0], rrf_ids)
        g1_rr = get_rank(gold_ids[0], rr_ids)
        g2_rrf = get_rank(gold_ids[1], rrf_ids)
        g2_rr = get_rank(gold_ids[1], rr_ids)

        # Recall 基于 Reranker 最终结果
        r5 = sum(1 for g in gold_ids if g in rr_ids[:5]) / len(gold_ids)
        r10 = sum(1 for g in gold_ids if g in rr_ids[:10]) / len(gold_ids)

        def fmt(v):
            return str(v) if v else "—"

        print(f"{short:<4} {actual_n:>5} {fmt(g1_rrf):>7} {fmt(g1_rr):>6} {fmt(g2_rrf):>7} {fmt(g2_rr):>6} {r5:>6.0%} {r10:>6.0%}")
