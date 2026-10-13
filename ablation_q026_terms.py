"""q026: 测试不同 search_terms 数量对 Recall 的影响"""
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

# 原始 subquery
base_queries = {
    "王鑫涛论文终稿": "王鑫涛论文终稿中使用了哪些研究方法？",
    "雷杰鑫毕业论文终稿": "雷杰鑫毕业论文终稿中使用了哪些研究方法？",
}

# DeepSeek 生成的 search_terms（按顺序）
all_terms = {
    "王鑫涛论文终稿": ["低周往复加载试验","拟静力试验","试件设计","加载制度","ABAQUS","有限元模型","材料本构模型","滞回曲线","骨架曲线","试验与有限元对比","参数分析","理论推导","计算公式"],
    "雷杰鑫毕业论文终稿": ["低周往复加载试验","试件设计","加载制度","ABAQUS","S4R壳单元","C3D8I实体单元","材料本构模型","滞回曲线对比","骨架曲线对比","有限元结果与试验对比","理论推导","参数分析","有限元模型","试验现象","破坏模式"],
}

# 测试的数量档位
settings = [3, 5, 8, 10, 999]  # 999 = 全部

print(f"{'论文':<12} {'Terms':>5} {'G1 rank':>8} {'G2 rank':>8} {'R@5':>6} {'R@10':>6}")
print("-" * 60)

results_summary = []

for paper in ["王鑫涛论文终稿", "雷杰鑫毕业论文终稿"]:
    gold_ids = gold[paper]
    base_q = base_queries[paper]
    terms_list = all_terms[paper]
    
    for n in settings:
        used_terms = terms_list[:n]
        actual_n = len(used_terms)
        # 拼 query：原始 subquery + search_terms
        expanded = base_q + " " + " ".join(used_terms)
        
        # 用 _search_in_paper 的完整路径：brainstorm跳过（手动扩展），直接 search_single + reranker
        raw_results = sr._search_single_paper(expanded, paper, top_k=100)
        # reranker 用原始 subquery（和正式路径一致）
        if sr.reranker:
            reranked = sr.reranker.rerank(base_q, raw_results, top_k=10)
        else:
            reranked = raw_results[:10]
        
        ids = [r["id"] for r in reranked]
        
        g1_rank = ids.index(gold_ids[0])+1 if gold_ids[0] in ids else (
            next((i+1 for i,r in enumerate(raw_results) if r["id"]==gold_ids[0]), "未召回"))
        g2_rank = ids.index(gold_ids[1])+1 if gold_ids[1] in ids else (
            next((i+1 for i,r in enumerate(raw_results) if r["id"]==gold_ids[1]), "未召回"))
        
        r5 = sum(1 for g in gold_ids if g in ids[:5]) / len(gold_ids)
        r10 = sum(1 for g in gold_ids if g in ids[:10]) / len(gold_ids)
        
        short = "王" if "王" in paper else "雷"
        label = "全部" if n == 999 else f"{actual_n}"
        print(f"{short:<12} {actual_n:>5} {str(g1_rank):>8} {str(g2_rank):>8} {r5:>6.0%} {r10:>6.0%}")
        results_summary.append({
            "paper": short, "n": actual_n,
            "g1_rank": g1_rank, "g2_rank": g2_rank,
            "r5": r5, "r10": r10
        })

print("\n\n详细：")
for r in results_summary:
    print(f"  {r['paper']}侧 terms={r['n']:>2d}  G1=#{r['g1_rank']}  G2=#{r['g2_rank']}  R@5={r['r5']:.0%}  R@10={r['r10']:.0%}")
