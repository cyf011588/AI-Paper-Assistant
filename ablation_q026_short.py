"""
q026 C-full vs C-short 对照实验
C-short: 基于gold chunk实际文本精选≤8个术语
"""
import os, sys, json, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, 'src')

from smart_retriever import SmartRetriever, RECALL_K, Chunk
from reranker import Reranker
from pathlib import Path

retriever = SmartRetriever()
all_chunks = []
for cf in Path("data/processed").glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)
reranker = Reranker()

GOLD = {
    "王鑫涛论文终稿": [
        "王鑫涛论文终稿_第二章 钢框架两边连接装配式C-PSW/CE 试验研究_0034",
        "王鑫涛论文终稿_第三章 钢框架两边连接装配式C-PSW/CE 有限元模型_0063",
    ],
    "雷杰鑫毕业论文终稿": [
        "雷杰鑫毕业论文终稿_第二章 钢框架-CPSW 有限元模型_0041",
        "雷杰鑫毕业论文终稿_第二章 钢框架-CPSW 有限元模型 / 2.2 有限元模型验证_0054",
    ],
}

# ---- C-full（上一轮完整搜索词）----
C_FULL = {
    "王鑫涛论文终稿": "有限元模型 单元类型 相互作用与边界条件 材料本构 螺栓连接模拟有效性验证 试验研究 试件尺寸 试验装置 加载制度 数据采集方案 钢材的力学性能 混凝土的力学性能 承载力曲线 试验现象 刚度退化曲线 钢板应变-侧移角曲线 滞回性能 耗能机理 水平剪力-层间侧移角力学模型 剪切屈服承载力与刚度计算公式",
    "雷杰鑫毕业论文终稿": "有限元模型 单元类型 相互作用与边界条件 材料本构 有限元模型验证 有限元算例 滞回曲线 骨架曲线 侧向刚度 耗能能力 塑性耗能 参数分析 钢板厚度 框架梁截面 轴压比",
}

# ---- C-short（基于gold chunk实际文本精选≤8词）----
# 王侧G1文本: "低周往复加载试验""缩尺试件""试验现象""刚度"
# 王侧G2文本: "ABAQUS""有限元模型""初始缺陷""混凝土材料损伤""57个有限元算例""参数影响分析"
# 雷侧G1文本: "ABAQUS""有限元模型""初始缺陷""81个算例""参数影响分析"
# 雷侧G2文本: "试验验证""荷载-位移骨架曲线""滞回曲线""材料本构关系"
C_SHORT = {
    "王鑫涛论文终稿": "低周往复加载试验 ABAQUS 有限元模型 有限元算例 试验现象 材料本构 初始缺陷 参数影响分析",
    "雷杰鑫毕业论文终稿": "ABAQUS 有限元模型 有限元算例 试验验证 骨架曲线 滞回曲线 材料本构 初始缺陷",
}

SUBQUERIES = {
    "王鑫涛论文终稿": "王鑫涛论文终稿采用了哪些研究方法？",
    "雷杰鑫毕业论文终稿": "雷杰鑫毕业论文终稿采用了哪些研究方法？",
}


def run_retrieval(paper_name, query, gold_ids):
    rrf_results = retriever._search_single_paper(query, paper_name, top_k=RECALL_K)
    rrf_ranks = {}
    for rank, r in enumerate(rrf_results):
        rrf_ranks[r["id"]] = rank + 1
    candidates = rrf_results[:20]
    if reranker and candidates:
        reranked = reranker.rerank(query, candidates, top_k=10)
    else:
        reranked = candidates[:10]
    rerank_ranks = {}
    for rank, r in enumerate(reranked):
        rerank_ranks[r["id"]] = rank + 1
    return rrf_ranks, rerank_ranks


print("=" * 80)
print("q026 C-full vs C-short 对照实验")
print("=" * 80)

summary = {}
for paper_name, sub_q in SUBQUERIES.items():
    gold_ids = GOLD[paper_name]
    print(f"\n{'='*60}")
    print(f"论文: {paper_name}")
    print(f"子问题: {sub_q}")

    for label, terms in [("C-full", C_FULL[paper_name]), ("C-short", C_SHORT[paper_name])]:
        query = sub_q + " " + terms
        print(f"\n--- {label} ---")
        print(f"  terms({len(terms.split())}个): {terms}")
        rrf_ranks, rr_ranks = run_retrieval(paper_name, query, gold_ids)
        rrf_hits = sum(1 for g in gold_ids if g in rrf_ranks and rrf_ranks[g] <= 10)
        rr_hits = sum(1 for g in gold_ids if g in rr_ranks and rr_ranks[g] <= 10)
        print(f"  RRF R@10={rrf_hits}/{len(gold_ids)}  RR R@10={rr_hits}/{len(gold_ids)}")
        for g in gold_ids:
            rrf_r = rrf_ranks.get(g, "未进RRF候选")
            rr_r = rr_ranks.get(g, "未进RR Top10")
            print(f"    {g[:55]}: RRF={rrf_r} RR={rr_r}")
        summary[(paper_name, label)] = {
            "rrf_ranks": rrf_ranks, "rr_ranks": rr_ranks,
            "rrf_hits": rrf_hits, "rr_hits": rr_hits, "n_gold": len(gold_ids)
        }

# 汇总
print(f"\n{'='*80}")
print("汇总")
print(f"{'='*80}")
total = sum(len(v) for v in GOLD.values())
for label in ["C-full", "C-short"]:
    rrf_hits = sum(summary[(p, label)]["rrf_hits"] for p in GOLD)
    rr_hits = sum(summary[(p, label)]["rr_hits"] for p in GOLD)
    print(f"{label}: RRF R@10={rrf_hits}/{total}={rrf_hits/total:.0%}  RR R@10={rr_hits}/{total}={rr_hits/total:.0%}")
