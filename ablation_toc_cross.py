"""
TOC-expansion 跨题验证：对所有comparison_decompose题目做Baseline vs TOC-expansion对比
"""
import os, sys, json, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, 'src')

from smart_retriever import SmartRetriever, RECALL_K, Chunk
from reranker import Reranker
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, EVAL_DATASET_PATH
from pathlib import Path

retriever = SmartRetriever()
all_chunks = []
for cf in Path("data/processed").glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)
reranker = Reranker()

# 加载评测集
with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# 找comparison_decompose题目
compare_qs = [q for q in eval_data if q.get("question_type") == "跨论文对比"]
print(f"跨论文对比题目: {[q['question_id'] for q in compare_qs]}")

# 论文名映射
PAPERS = {
    "雷": "雷杰鑫毕业论文终稿",
    "王": "王鑫涛论文终稿",
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


def toc_expand(question, paper_name, toc):
    """基于目录自动生成≤8个搜索词"""
    toc_text = "\n".join(toc[:60])
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": """你是论文检索专家。用户问了一个论文问题。
以下是该论文的章节目录（只有标题）。请根据目录，生成最多8个搜索词，用于检索与该问题相关的章节。
规则：
1. 只从目录标题中提取术语，不要凭空想象。
2. 优先提取目录中与问题相关的章节关键词。
3. 不要生成泛词（如"研究方法""理论分析"）。
4. 输出JSON：{"terms": ["词1",...]}"""},
            {"role": "user", "content": f"论文：{paper_name}\n问题：{question}\n\n目录：\n{toc_text}"}
        ],
        temperature=0.1, max_tokens=300,
    )
    result = json.loads(resp.choices[0].message.content.strip())
    return result.get("terms", [])


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


# 为每道对比题定义子查询
SUBQUERIES = {
    "q003": {
        "雷": "雷杰鑫毕业论文终稿的指导教师是谁？",
        "王": "王鑫涛论文终稿的指导教师是谁？",
    },
    "q024": {
        "雷": "雷杰鑫毕业论文终稿中墙体与钢框架的连接方式是什么？",
        "王": "王鑫涛论文终稿中墙体与钢框架的连接方式是什么？",
    },
    "q025": {
        "雷": "雷杰鑫毕业论文终稿中框架梁在耗能中扮演什么角色？",
        "王": "王鑫涛论文终稿中框架梁在耗能中扮演什么角色？",
    },
    "q026": {
        "雷": "雷杰鑫毕业论文终稿采用了哪些研究方法？",
        "王": "王鑫涛论文终稿采用了哪些研究方法？",
    },
    "q030": {
        "雷": "雷杰鑫毕业论文终稿的核心创新点是什么？",
        "王": "王鑫涛论文终稿的核心创新点是什么？",
    },
}


print("=" * 80)
print("TOC-expansion 跨题验证")
print("=" * 80)

all_results = {}
for q in compare_qs:
    qid = q["question_id"]
    if qid not in SUBQUERIES:
        print(f"\n跳过 {qid}: 未定义子查询")
        continue
    gold_ids = q["retrieval_gt"]["relevant_chunk_ids"]
    print(f"\n{'='*60}")
    print(f"{qid}: {q['question'][:70]}...")
    print(f"Gold({len(gold_ids)}): {gold_ids}")

    all_results[qid] = {"baseline": {}, "toc": {}, "gold": gold_ids}

    for side, sub_q in SUBQUERIES[qid].items():
        paper_name = PAPERS[side]
        paper_gold = [g for g in gold_ids if paper_name in g]
        if not paper_gold:
            continue

        # Baseline: 当前_brainstorm_query
        expanded_base = retriever._brainstorm_query(sub_q)
        rrf_b, rr_b = run_retrieval(paper_name, expanded_base, paper_gold)

        # TOC-expansion
        toc = get_paper_toc(paper_name)
        toc_terms = toc_expand(sub_q, paper_name, toc)
        toc_query = sub_q + " " + " ".join(toc_terms)
        rrf_t, rr_t = run_retrieval(paper_name, toc_query, paper_gold)

        print(f"\n  [{side}] {paper_name}")
        print(f"    sub_q: {sub_q}")
        print(f"    baseline expanded: {expanded_base[-60:]}")
        print(f"    TOC terms: {toc_terms}")
        for g in paper_gold:
            rb = rrf_b.get(g, "-")
            pb = rr_b.get(g, "-")
            rt = rrf_t.get(g, "-")
            pt = rr_t.get(g, "-")
            hit_b = "✓" if isinstance(pb, int) and pb <= 10 else "✗"
            hit_t = "✓" if isinstance(pt, int) and pt <= 10 else "✗"
            print(f"    {g[:50]}: base RRF={rb} RR={pb}{hit_b} | TOC RRF={rt} RR={pt}{hit_t}")

        all_results[qid]["baseline"][side] = {"rrf": rrf_b, "rr": rr_b, "gold": paper_gold}
        all_results[qid]["toc"][side] = {"rrf": rrf_t, "rr": rr_t, "gold": paper_gold, "terms": toc_terms}

# 汇总
print(f"\n{'='*80}")
print("汇总")
print(f"{'='*80}")
total_gold = 0
base_hits = 0
toc_hits = 0
base_rrf_hits = 0
toc_rrf_hits = 0
for qid, r in all_results.items():
    for side in ["雷", "王"]:
        if side not in r["baseline"]:
            continue
        paper_gold = r["baseline"][side]["gold"]
        total_gold += len(paper_gold)
        for g in paper_gold:
            if g in r["baseline"][side]["rr"] and r["baseline"][side]["rr"][g] <= 10:
                base_hits += 1
            if g in r["toc"][side]["rr"] and r["toc"][side]["rr"][g] <= 10:
                toc_hits += 1
            if g in r["baseline"][side]["rrf"] and r["baseline"][side]["rrf"][g] <= 10:
                base_rrf_hits += 1
            if g in r["toc"][side]["rrf"] and r["toc"][side]["rrf"][g] <= 10:
                toc_rrf_hits += 1
    # 每题统计
    q_gold = len(r["gold"])
    q_base = sum(1 for s in r["baseline"] for g in r["baseline"][s]["gold"] if g in r["baseline"][s]["rr"] and r["baseline"][s]["rr"][g] <= 10)
    q_toc = sum(1 for s in r["toc"] for g in r["toc"][s]["gold"] if g in r["toc"][s]["rr"] and r["toc"][s]["rr"][g] <= 10)
    print(f"  {qid}: baseline RR R@10={q_base}/{q_gold}  TOC RR R@10={q_toc}/{q_gold}")

print(f"\n总Gold数: {total_gold}")
print(f"Baseline RRF R@10: {base_rrf_hits}/{total_gold} = {base_rrf_hits/total_gold:.0%}")
print(f"TOC RRF R@10: {toc_rrf_hits}/{total_gold} = {toc_rrf_hits/total_gold:.0%}")
print(f"Baseline RR R@10: {base_hits}/{total_gold} = {base_hits/total_gold:.0%}")
print(f"TOC RR R@10: {toc_hits}/{total_gold} = {toc_hits/total_gold:.0%}")
