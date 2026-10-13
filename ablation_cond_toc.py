"""
Conditional-TOC实验：根据brainstorm输出的具体程度决定是否触发TOC扩展
规则：brainstorm输出中包含≥3个结构工程具体专业术语时不触发，否则触发
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

PAPERS = {"雷": "雷杰鑫毕业论文终稿", "王": "王鑫涛论文终稿"}

SUBQUERIES = {
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
}

# ---- 触发规则 ----
# 具体专业术语词表（结构工程领域的具体对象/方法/参数）
SPECIFIC_TERMS = {
    # 连接方式类
    "四边连接", "两边连接", "现浇", "装配", "螺栓", "焊缝", "栓钉",
    # 构件类
    "框架梁", "框架柱", "钢板墙", "混凝土板", "墙板", "耗能梁段",
    # 材料/单元类
    "ABAQUS", "S4R", "C3D8I", "壳单元", "实体单元", "材料本构",
    # 试验类
    "拟静力", "低周往复", "加载制度", "试件", "试验现象",
    # 力学性能类
    "滞回曲线", "骨架曲线", "刚度退化", "塑性耗能", "剪切屈服",
    # 参数类
    "轴压比", "宽厚比", "高宽比", "板厚",
    # 公式类
    "承载力公式", "设计方法", "力学模型",
}

# 泛化词
GENERIC_WORDS = {
    "方法", "研究方法", "采用的方法", "研究手段", "作用", "角色",
    "关系", "区别", "不同", "特点", "内容", "什么", "哪些",
    "主要", "核心", "关键", "如何", "为什么",
}


def should_truct_toc(brainstorm_terms):
    """判断是否触发TOC扩展
    规则：brainstorm中具体专业术语≥3个 → 不触发；否则触发
    """
    specific_count = 0
    generic_count = 0
    matched_specific = []
    matched_generic = []
    for term in brainstorm_terms:
        for spec in SPECIFIC_TERMS:
            if spec in term:
                specific_count += 1
                matched_specific.append(term)
                break
        else:
            for gen in GENERIC_WORDS:
                if gen in term:
                    generic_count += 1
                    matched_generic.append(term)
                    break
    trigger = specific_count < 3
    return trigger, specific_count, generic_count, matched_specific, matched_generic


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
3. 不要生成泛词。
4. 输出JSON：{"terms": ["词1",...]}"""},
            {"role": "user", "content": f"论文：{paper_name}\n问题：{question}\n\n目录：\n{toc_text}"}
        ],
        temperature=0.1, max_tokens=300,
    )
    result = json.loads(resp.choices[0].message.content.strip())
    return result.get("terms", [])


def run_retrieval(paper_name, query, gold_ids):
    rrf_results = retriever._search_single_paper(query, paper_name, top_k=RECALL_K)
    rrf_ranks = {r["id"]: rank + 1 for rank, r in enumerate(rrf_results)}
    candidates = rrf_results[:20]
    if reranker and candidates:
        reranked = reranker.rerank(query, candidates, top_k=10)
    else:
        reranked = candidates[:10]
    rerank_ranks = {r["id"]: rank + 1 for rank, r in enumerate(reranked)}
    return rrf_ranks, rerank_ranks


# ---- 实验 ----
print("=" * 80)
print("Conditional-TOC 实验")
print("=" * 80)
print("\n触发规则：brainstorm输出中具体专业术语<3个 → 触发TOC；≥3个 → 不触发")
print("理由：具体术语≥3个说明问题本身指向性强，brainstorm已提取精准词；")
print("      <3个说明问题偏抽象，需要目录术语补充。")
print()

results = {}
for qid, subqs in SUBQUERIES.items():
    q_data = next(q for q in eval_data if q["question_id"] == qid)
    gold_ids = q_data["retrieval_gt"]["relevant_chunk_ids"]
    print(f"\n{'='*60}")
    print(f"{qid}: {q_data['question'][:70]}...")
    print(f"Gold({len(gold_ids)}): {[g[:40] for g in gold_ids]}")

    results[qid] = {"gold": gold_ids, "sides": {}}

    for side, sub_q in subqs.items():
        paper_name = PAPERS[side]
        paper_gold = [g for g in gold_ids if paper_name in g]
        if not paper_gold:
            continue

        # Baseline
        expanded_base = retriever._brainstorm_query(sub_q)
        # 提取brainstorm新增的词（去掉原始问题）
        brainstorm_added = expanded_base.replace(sub_q, "").strip()
        brainstorm_terms = brainstorm_added.split()

        trigger, spec_n, gen_n, matched_spec, matched_gen = should_truct_toc(brainstorm_terms)

        print(f"\n  [{side}] {paper_name}")
        print(f"    sub_q: {sub_q}")
        print(f"    brainstorm新增词: {brainstorm_terms}")
        print(f"    具体术语({spec_n}个): {matched_spec}")
        print(f"    泛化词({gen_n}个): {matched_gen}")
        print(f"    → 触发TOC: {trigger}")

        rrf_b, rr_b = run_retrieval(paper_name, expanded_base, paper_gold)

        # Always-TOC
        toc = get_paper_toc(paper_name)
        toc_terms = toc_expand(sub_q, paper_name, toc)
        toc_query = sub_q + " " + " ".join(toc_terms)
        rrf_t, rr_t = run_retrieval(paper_name, toc_query, paper_gold)

        # Conditional-TOC
        if trigger:
            cond_query = toc_query
            cond_label = "TOC"
            rrf_c, rr_c = rrf_t, rr_t
        else:
            cond_query = expanded_base
            cond_label = "baseline"
            rrf_c, rr_c = rrf_b, rr_b

        print(f"    --- 结果 ---")
        for g in paper_gold:
            print(f"    {g[:45]}:")
            print(f"      Baseline:  RRF={rrf_b.get(g,'-'):>3} RR={rr_b.get(g,'-'):>3} {'✓' if g in rr_b and rr_b[g]<=10 else '✗'}")
            print(f"      Always-TOC: RRF={rrf_t.get(g,'-'):>3} RR={rr_t.get(g,'-'):>3} {'✓' if g in rr_t and rr_t[g]<=10 else '✗'}")
            print(f"      Conditional: RRF={rrf_c.get(g,'-'):>3} RR={rr_c.get(g,'-'):>3} {'✓' if g in rr_c and rr_c[g]<=10 else '✗'} ({cond_label})")

        results[qid]["sides"][side] = {
            "trigger": trigger, "paper_gold": paper_gold,
            "base_rrf": rrf_b, "base_rr": rr_b,
            "toc_rrf": rrf_t, "toc_rr": rr_t,
            "cond_rrf": rrf_c, "cond_rr": rr_c,
        }

# 汇总
print(f"\n{'='*80}")
print("汇总")
print(f"{'='*80}")
total = sum(len(r["gold"]) for r in results.values())

for label, key_rrf, key_rr in [("Baseline", "base_rrf", "base_rr"),
                                 ("Always-TOC", "toc_rrf", "toc_rr"),
                                 ("Conditional", "cond_rrf", "cond_rr")]:
    rrf_hits = sum(1 for q in results.values() for s in q["sides"].values()
                   for g in s["paper_gold"] if g in s[key_rrf] and s[key_rrf][g] <= 10)
    rr_hits = sum(1 for q in results.values() for s in q["sides"].values()
                  for g in s["paper_gold"] if g in s[key_rr] and s[key_rr][g] <= 10)
    print(f"{label:15s}: RRF R@10={rrf_hits}/{total}={rrf_hits/total:.0%}  RR R@10={rr_hits}/{total}={rr_hits/total:.0%}")

# 逐题统计
print(f"\n逐题RR R@10:")
for qid, r in results.items():
    for label, key in [("Baseline", "base_rr"), ("Always-TOC", "toc_rr"), ("Conditional", "cond_rr")]:
        hits = sum(1 for s in r["sides"].values() for g in s["paper_gold"]
                   if g in s[key] and s[key][g] <= 10)
        print(f"  {qid} {label:12s}: {hits}/{len(r['gold'])}")
