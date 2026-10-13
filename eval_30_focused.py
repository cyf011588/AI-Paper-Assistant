"""
完整30题验证：只对comparison_decompose 5题跑Conditional-TOC
其余25题 hybrid_rerank/section_shortcut 不受影响（base=toc）
跑3轮，每轮对5题分别跑Baseline和Conditional-TOC
"""
import os, sys, json, re, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, 'src')

from smart_retriever import SmartRetriever, RECALL_K
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, EVAL_DATASET_PATH
from pathlib import Path
from chunker import Chunk

# 初始化
retriever = SmartRetriever()
all_chunks = []
for cf in Path("data/processed").glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)

# 加载评测集和baseline结果
with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)
with open("evaluation/eval_v3_result.json", "r", encoding="utf-8") as f:
    baseline_results = json.load(f)

# comparison_decompose 的5个qid
COMPARE_QIDS = {"q003", "q024", "q025", "q026", "q030"}

SPECIFIC_TERMS = {
    "四边连接", "两边连接", "现浇", "装配", "螺栓", "焊缝", "栓钉",
    "框架梁", "框架柱", "钢板墙", "混凝土板", "墙板", "耗能梁段",
    "ABAQUS", "S4R", "C3D8I", "壳单元", "实体单元", "材料本构",
    "拟静力", "低周往复", "加载制度", "试件", "试验现象",
    "滞回曲线", "骨架曲线", "刚度退化", "塑性耗能", "剪切屈服",
    "轴压比", "宽厚比", "高宽比", "板厚",
    "承载力公式", "设计方法", "力学模型",
}

def get_paper_toc(paper_name):
    seen = set()
    toc = []
    for c in retriever.paper_chunks_map.get(paper_name, []):
        sp = re.sub(r'_\d+$', '', c.section_path)
        sp = sp.replace(paper_name + "_", "")
        if sp not in seen:
            seen.add(sp)
            toc.append(sp)
    return sorted(toc)

toc_cache = {}
def get_toc_terms(sub_q, paper_name):
    key = (sub_q, paper_name)
    if key in toc_cache:
        return toc_cache[key]
    toc = get_paper_toc(paper_name)
    toc_text = "\n".join(toc[:60])
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是论文检索专家。以下是论文章节目录。请根据问题，从目录中提取最多8个搜索词。只从目录标题提取，不要生成泛词。输出JSON：{\"terms\": [\"词1\",...]}"},
            {"role": "user", "content": f"论文：{paper_name}\n问题：{sub_q}\n目录：\n{toc_text}"}
        ],
        temperature=0.1, max_tokens=300,
    )
    terms = json.loads(resp.choices[0].message.content.strip()).get("terms", [])
    toc_cache[key] = terms
    return terms

def count_specific(terms_str):
    return sum(1 for t in terms_str.split() if any(s in t for s in SPECIFIC_TERMS))

original_search_in_paper = retriever._search_in_paper

def patched_search_in_paper(query, paper_name, top_k=10, use_toc=False):
    if not use_toc:
        return original_search_in_paper(query, paper_name, top_k)
    expanded_query = retriever._brainstorm_query(query)
    brainstorm_added = expanded_query.replace(query, "").strip()
    spec_count = count_specific(brainstorm_added)
    if spec_count < 3:
        toc_terms = get_toc_terms(query, paper_name)
        if toc_terms:
            expanded_query = expanded_query + " " + " ".join(toc_terms)
            print(f"    [TOC] +{len(toc_terms)}词: {' '.join(toc_terms[:4])}...")
    results = retriever._search_single_paper(expanded_query, paper_name, top_k=RECALL_K)
    if results and retriever.reranker:
        results = retriever.reranker.rerank(expanded_query, results, top_k=top_k)
    return results[:top_k]


# 跑3轮，只跑comparison_decompose 5题
all_rounds = []
compare_items = [item for item in dataset if item["question_id"] in COMPARE_QIDS]

for round_num in range(1, 4):
    print(f"\n{'#'*60}")
    print(f"# 第{round_num}轮（comparison_decompose 5题）")
    print(f"{'#'*60}")

    round_results = []
    t0 = time.time()

    for item in compare_items:
        qid = item["question_id"]
        question = item["question"]
        gt_ids = set(item["retrieval_gt"]["relevant_chunk_ids"])

        # Baseline
        retriever._search_in_paper = original_search_in_paper
        output_base = retriever.search(question, top_k=10)
        debug_base = output_base.get("debug_info", {})
        pre_base = debug_base.get("pre_expansion_ids", [])[:10]
        post_base = debug_base.get("post_expansion_ids", [])
        base_strict = len(gt_ids & set(pre_base)) / len(gt_ids)
        base_after = len(gt_ids & set(post_base)) / len(gt_ids)

        # Conditional-TOC
        retriever._search_in_paper = lambda q, p, top_k=10: patched_search_in_paper(q, p, top_k, use_toc=True)
        output_toc = retriever.search(question, top_k=10)
        retriever._search_in_paper = original_search_in_paper
        debug_toc = output_toc.get("debug_info", {})
        pre_toc = debug_toc.get("pre_expansion_ids", [])[:10]
        post_toc = debug_toc.get("post_expansion_ids", [])
        toc_strict = len(gt_ids & set(pre_toc)) / len(gt_ids)
        toc_after = len(gt_ids & set(post_toc)) / len(gt_ids)

        round_results.append({
            "qid": qid, "n_gold": len(gt_ids),
            "base_strict": round(base_strict, 2),
            "base_after": round(base_after, 2),
            "toc_strict": round(toc_strict, 2),
            "toc_after": round(toc_after, 2),
        })
        print(f"  {qid}: base={base_strict:.0%} toc={toc_strict:.0%} (gold={len(gt_ids)})")

    elapsed = time.time() - t0
    all_rounds.append({"round": round_num, "results": round_results, "elapsed": elapsed})
    b = sum(r["base_strict"] for r in round_results) / len(round_results)
    t = sum(r["toc_strict"] for r in round_results) / len(round_results)
    print(f"  → 5题平均: base={b:.0%} toc={t:.0%} ({elapsed:.0f}s)")

# ---- 汇总 ----
print(f"\n{'='*60}")
print("comparison_decompose 5题 × 3轮")
print(f"{'='*60}")
for label, key in [("Baseline", "base_strict"), ("Cond-TOC", "toc_strict")]:
    vals = [sum(r[key] for r in rd["results"]) / len(rd["results"]) for rd in all_rounds]
    print(f"{label:12s}: {vals[0]:.0%} / {vals[1]:.0%} / {vals[2]:.0%}  avg={sum(vals)/3:.1%}")

# 合并30题：其他25题用baseline结果，5题用本轮结果
print(f"\n{'='*60}")
print("合并30题完整结果（其他25题 base=toc）")
print(f"{'='*60}")

# 从baseline_results取25题（非comparison）的strict recall
non_compare = [r for r in baseline_results if r["qid"] not in COMPARE_QIDS]
non_compare_avg = sum(r["strict_recall_at_10"] for r in non_compare) / len(non_compare)
print(f"非comparison {len(non_compare)}题 baseline R@10 = {non_compare_avg:.1%}（toc相同）")

# comparison 5题3轮平均
for cond_key, label in [("base_strict", "Baseline"), ("toc_strict", "Cond-TOC")]:
    vals = []
    for rd in all_rounds:
        avg5 = sum(r[cond_key] for r in rd["results"]) / len(rd["results"])
        # 合并：25题 + 5题
        combined = (non_compare_avg * len(non_compare) + avg5 * 5) / 30
        vals.append(combined)
    print(f"全30题 {label:10s}: {vals[0]:.1%} / {vals[1]:.1%} / {vals[2]:.1%}  avg={sum(vals)/3:.1%}")

# 逐题对比（第3轮）
print(f"\ncomparison_decompose 逐题（第3轮）:")
for r in all_rounds[2]["results"]:
    delta = r["toc_strict"] - r["base_strict"]
    mark = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
    print(f"  {r['qid']}: base={r['base_strict']:.0%} → toc={r['toc_strict']:.0%} {mark} ({delta:+.0%})")

# 回归检查
print(f"\n回归检查:")
regression = False
for r in all_rounds[2]["results"]:
    if r["base_strict"] > r["toc_strict"]:
        print(f"  {r['qid']}: base={r['base_strict']:.0%} → toc={r['toc_strict']:.0%} REGRESSION")
        regression = True
if not regression:
    print("  无回归")

with open("evaluation/eval_cond_toc_3rounds.json", "w", encoding="utf-8") as f:
    json.dump({"rounds": all_rounds, "non_compare_avg": non_compare_avg, "non_compare_n": len(non_compare)}, f, ensure_ascii=False, indent=2)
print(f"\n已保存 evaluation/eval_cond_toc_3rounds.json")
