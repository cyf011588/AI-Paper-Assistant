"""q026 人工 Query Expansion Ablation — 不修改任何系统代码，只做对比实验"""
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

# gold chunks
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

# 当前 brainstorm 后的 query（从 eval 日志拿到的）
current_queries = {
    "王鑫涛论文终稿": "王鑫涛论文终稿中使用了哪些研究方法？ 研究方法 使用方法",
    "雷杰鑫毕业论文终稿": "雷杰鑫毕业论文终稿中使用了哪些研究方法？ 方法论 实验方法 数值模拟",
}

# 人工扩展的 query：把"研究方法"展开成论文章首 chunk 里实际出现的术语
human_queries = {
    "王鑫涛论文终稿": "王鑫涛论文终稿中使用了哪些研究方法？ 试件设计 低周往复加载试验 试验方案 有限元模型建立 模型验证 ABAQUS",
    "雷杰鑫毕业论文终稿": "雷杰鑫毕业论文终稿中使用了哪些研究方法？ ABAQUS 有限元模型建立 单元类型 S4R C3D8I 模型验证 相互作用 边界条件 材料本构",
}

def report(label, query, paper_name, gold_ids):
    print(f"\n{'='*70}")
    print(f"[{label}] {paper_name[:15]}")
    print(f"Query: {query[:80]}...")
    results = sr._search_single_paper(query, paper_name, top_k=100)
    ids = [r["id"] for r in results]
    top10 = results[:10]
    print(f"\nTop 10:")
    for i, r in enumerate(top10):
        hit = " <<< GOLD" if r["id"] in gold_ids else ""
        print(f"  #{i+1:2d}  {r['id'][:70]}{hit}")
    print(f"\nGold 排名:")
    for g in gold_ids:
        if g in ids:
            rank = ids.index(g) + 1
            marker = "✓ TOP10" if rank <= 10 else "✗ 在TOP10外"
            print(f"  {g[:60]} → rank #{rank}  {marker}")
        else:
            print(f"  {g[:60]} → 未召回")
    recall5 = sum(1 for g in gold_ids if g in ids[:5]) / len(gold_ids)
    recall10 = sum(1 for g in gold_ids if g in ids[:10]) / len(gold_ids)
    print(f"\nRecall@5 = {recall5:.0%}  Recall@10 = {recall10:.0%}")
    return recall5, recall10

for paper in ["雷杰鑫毕业论文终稿", "王鑫涛论文终稿"]:
    gold_ids = gold[paper]
    print(f"\n{'#'*70}")
    print(f"# {paper}")
    print(f"# Gold: {gold_ids}")
    print(f"{'#'*70}")
    report("当前 brainstorm", current_queries[paper], paper, gold_ids)
    report("人工扩展", human_queries[paper], paper, gold_ids)
