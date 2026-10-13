"""
完整30题验证：Baseline vs Conditional-TOC
只跑检索层，不跑生成层。3轮。
Conditional-TOC: 在_search_in_paper中，brainstorm后如果具体术语<3，追加目录生成的搜索词
"""
import os, sys, json, re, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, 'src')

from smart_retriever import SmartRetriever, RECALL_K, Chunk
from reranker import Reranker
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, require_eval_dataset
from pathlib import Path
from chunker import Chunk as ChunkerChunk

# 初始化
retriever = SmartRetriever()
all_chunks = []
for cf in Path("data/processed").glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)
reranker = Reranker()

# 加载评测集（公开仓库不发布评测集，需本地自备）
try:
    with open(require_eval_dataset(), "r", encoding="utf-8") as f:
        dataset = json.load(f)
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

# ---- Conditional-TOC 工具 ----
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

# 缓存TOC terms，避免重复LLM调用
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

# ---- 包装_search_in_paper以支持Conditional-TOC ----
original_search_in_paper = retriever._search_in_paper

def patched_search_in_paper(query, paper_name, top_k=10, use_toc=False):
    if not use_toc:
        return original_search_in_paper(query, paper_name, top_k)

    # Baseline flow: brainstorm → search → rerank
    expanded_query = retriever._brainstorm_query(query)

    # Conditional-TOC: check if specific terms < 3
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


# ---- 跑3轮 ----
all_rounds = []

for round_num in range(1, 4):
    print(f"\n{'#'*80}")
    print(f"# 第{round_num}轮")
    print(f"{'#'*80}")

    round_results = []
    t0 = time.time()

    for i, item in enumerate(dataset):
        qid = item["question_id"]
        question = item["question"]
        gt_ids = set(item["retrieval_gt"]["relevant_chunk_ids"])

        # Baseline
        retriever._search_in_paper = original_search_in_paper
        output_base = retriever.search(question, top_k=10)
        debug_base = output_base.get("debug_info", {})
        mode = debug_base.get("retrieval_mode", "unknown")
        pre_base = debug_base.get("pre_expansion_ids", [])[:10]
        post_base = debug_base.get("post_expansion_ids", [])

        # Conditional-TOC (only affects comparison_decompose path via _search_in_paper)
        retriever._search_in_paper = lambda q, p, top_k=10: patched_search_in_paper(q, p, top_k, use_toc=True)
        output_toc = retriever.search(question, top_k=10)
        retriever._search_in_paper = original_search_in_paper
        debug_toc = output_toc.get("debug_info", {})
        pre_toc = debug_toc.get("pre_expansion_ids", [])[:10]
        post_toc = debug_toc.get("post_expansion_ids", [])

        # Recall metrics
        base_strict = len(gt_ids & set(pre_base)) / len(gt_ids)
        base_after = len(gt_ids & set(post_base)) / len(gt_ids)
        toc_strict = len(gt_ids & set(pre_toc)) / len(gt_ids)
        toc_after = len(gt_ids & set(post_toc)) / len(gt_ids)

        round_results.append({
            "qid": qid, "mode": mode, "n_gold": len(gt_ids),
            "base_strict": round(base_strict, 2),
            "base_after": round(base_after, 2),
            "toc_strict": round(toc_strict, 2),
            "toc_after": round(toc_after, 2),
        })

        print(f"[{i+1}/30] {qid} [{mode:20s}] base={base_strict:.0%} toc={toc_strict:.0%}")

    elapsed = time.time() - t0
    all_rounds.append({"round": round_num, "results": round_results, "elapsed": elapsed})

    # 本轮汇总
    base_strict_avg = sum(r["base_strict"] for r in round_results) / len(round_results)
    toc_strict_avg = sum(r["toc_strict"] for r in round_results) / len(round_results)
    base_after_avg = sum(r["base_after"] for r in round_results) / len(round_results)
    toc_after_avg = sum(r["toc_after"] for r in round_results) / len(round_results)
    print(f"\n  第{round_num}轮 ({elapsed:.0f}s):")
    print(f"    Baseline: Strict R@10={base_strict_avg:.1%}  After exp={base_after_avg:.1%}")
    print(f"    Cond-TOC: Strict R@10={toc_strict_avg:.1%}  After exp={toc_after_avg:.1%}")

# ---- 3轮汇总 ----
print(f"\n{'='*80}")
print("3轮汇总")
print(f"{'='*80}")

for label, key in [("Baseline", "base_strict"), ("Cond-TOC", "toc_strict")]:
    vals = [sum(r[key] for r in rd["results"]) / len(rd["results"]) for rd in all_rounds]
    print(f"{label:12s} Strict R@10: {vals[0]:.1%} / {vals[1]:.1%} / {vals[2]:.1%}  avg={sum(vals)/3:.1%}")

for label, key in [("Baseline", "base_after"), ("Cond-TOC", "toc_after")]:
    vals = [sum(r[key] for r in rd["results"]) / len(rd["results"]) for rd in all_rounds]
    print(f"{label:12s} After exp:  {vals[0]:.1%} / {vals[1]:.1%} / {vals[2]:.1%}  avg={sum(vals)/3:.1%}")

# 按模式分组（取第3轮）
print(f"\n按检索模式分组（第3轮）:")
by_mode = {}
for r in all_rounds[2]["results"]:
    m = r["mode"]
    if m not in by_mode:
        by_mode[m] = []
    by_mode[m].append(r)
for mode, rs in by_mode.items():
    b = sum(r["base_strict"] for r in rs) / len(rs)
    t = sum(r["toc_strict"] for r in rs) / len(rs)
    print(f"  {mode:25s}: {len(rs)}题  base={b:.0%}  toc={t:.0%}")

# 回归检查（第3轮）
print(f"\n回归检查（第3轮）:")
for r in all_rounds[2]["results"]:
    if r["base_strict"] > r["toc_strict"]:
        print(f"  {r['qid']} [{r['mode']}]: base={r['base_strict']:.0%} → toc={r['toc_strict']:.0%}")

# 保存
with open("evaluation/eval_cond_toc_3rounds.json", "w", encoding="utf-8") as f:
    json.dump(all_rounds, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存 evaluation/eval_cond_toc_3rounds.json")
