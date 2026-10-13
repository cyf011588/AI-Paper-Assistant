"""
复核实验：同一环境下重跑Baseline/Always-TOC/Conditional-TOC，核查RRF差异
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

SPECIFIC_TERMS = {
    "四边连接", "两边连接", "现浇", "装配", "螺栓", "焊缝", "栓钉",
    "框架梁", "框架柱", "钢板墙", "混凝土板", "墙板", "耗能梁段",
    "ABAQUS", "S4R", "C3D8I", "壳单元", "实体单元", "材料本构",
    "拟静力", "低周往复", "加载制度", "试件", "试验现象",
    "滞回曲线", "骨架曲线", "刚度退化", "塑性耗能", "剪切屈服",
    "轴压比", "宽厚比", "高宽比", "板厚",
    "承载力公式", "设计方法", "力学模型",
}
GENERIC_WORDS = {"方法", "研究方法", "采用的方法", "研究手段", "作用", "角色",
                 "关系", "区别", "不同", "特点", "内容", "什么", "哪些",
                 "主要", "核心", "关键", "如何", "为什么"}


def should_truct_toc(brainstorm_terms):
    specific_count = sum(1 for t in brainstorm_terms if any(s in t for s in SPECIFIC_TERMS))
    return specific_count < 3, specific_count


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
以下是该论文的章节目录（只有标题）。请根据目录，生成最多8个搜索词。
只从目录标题中提取术语，不要生成泛词。输出JSON：{"terms": ["词1",...]}"""},
            {"role": "user", "content": f"论文：{paper_name}\n问题：{question}\n\n目录：\n{toc_text}"}
        ],
        temperature=0.1, max_tokens=300,
    )
    return json.loads(resp.choices[0].message.content.strip()).get("terms", [])


def run_retrieval(paper_name, query, gold_ids):
    rrf_results = retriever._search_single_paper(query, paper_name, top_k=RECALL_K)
    rrf_ranks = {r["id"]: rank + 1 for rank, r in enumerate(rrf_results)}
    candidates = rrf_results[:20]
    reranked = reranker.rerank(query, candidates, top_k=10) if reranker and candidates else candidates[:10]
    rerank_ranks = {r["id"]: rank + 1 for rank, r in enumerate(reranked)}
    return rrf_ranks, rerank_ranks


# 跑3轮，检查随机性
for round_num in range(1, 4):
    print(f"\n{'#'*80}")
    print(f"# 第{round_num}轮")
    print(f"{'#'*80}")

    total_gold = 0
    base_rrf_hits = base_rr_hits = 0
    always_rrf_hits = always_rr_hits = 0
    cond_rrf_hits = cond_rr_hits = 0

    for qid, subqs in SUBQUERIES.items():
        q_data = next(q for q in eval_data if q["question_id"] == qid)
        gold_ids = q_data["retrieval_gt"]["relevant_chunk_ids"]
        total_gold += len(gold_ids)

        for side, sub_q in subqs.items():
            paper_name = PAPERS[side]
            paper_gold = [g for g in gold_ids if paper_name in g]
            if not paper_gold:
                continue

            # Baseline
            expanded_base = retriever._brainstorm_query(sub_q)
            brainstorm_terms = expanded_base.replace(sub_q, "").strip().split()
            rrf_b, rr_b = run_retrieval(paper_name, expanded_base, paper_gold)

            # Always-TOC
            toc = get_paper_toc(paper_name)
            toc_terms = toc_expand(sub_q, paper_name, toc)
            toc_query = sub_q + " " + " ".join(toc_terms)
            rrf_t, rr_t = run_retrieval(paper_name, toc_query, paper_gold)

            # Conditional
            trigger, spec_n = should_truct_toc(brainstorm_terms)
            if trigger:
                rrf_c, rr_c = rrf_t, rr_t
                cond_src = "TOC"
            else:
                rrf_c, rr_c = rrf_b, rr_b
                cond_src = "base"

            for g in paper_gold:
                base_rrf_hits += 1 if g in rrf_b and rrf_b[g] <= 10 else 0
                base_rr_hits += 1 if g in rr_b and rr_b[g] <= 10 else 0
                always_rrf_hits += 1 if g in rrf_t and rrf_t[g] <= 10 else 0
                always_rr_hits += 1 if g in rr_t and rr_t[g] <= 10 else 0
                cond_rrf_hits += 1 if g in rrf_c and rrf_c[g] <= 10 else 0
                cond_rr_hits += 1 if g in rr_c and rr_c[g] <= 10 else 0

            # 打印q026雷侧的详细信息（重点关注RRF差异）
            if qid == "q026" and side == "雷":
                print(f"\n  [{qid}-{side}] brainstorm={brainstorm_terms} spec_n={spec_n} trigger={trigger} src={cond_src}")
                for g in paper_gold:
                    print(f"    {g[:50]}: base RRF={rrf_b.get(g,'-'):>3} RR={rr_b.get(g,'-'):>3} | "
                          f"TOC RRF={rrf_t.get(g,'-'):>3} RR={rr_t.get(g,'-'):>3} | "
                          f"cond RRF={rrf_c.get(g,'-'):>3} RR={rr_c.get(g,'-'):>3}")

    print(f"\n  第{round_num}轮汇总 (total={total_gold}):")
    print(f"    Baseline:    RRF R@10={base_rrf_hits}/{total_gold}={base_rrf_hits/total_gold:.0%}  RR R@10={base_rr_hits}/{total_gold}={base_rr_hits/total_gold:.0%}")
    print(f"    Always-TOC:  RRF R@10={always_rrf_hits}/{total_gold}={always_rrf_hits/total_gold:.0%}  RR R@10={always_rr_hits}/{total_gold}={always_rr_hits/total_gold:.0%}")
    print(f"    Conditional: RRF R@10={cond_rrf_hits}/{total_gold}={cond_rrf_hits/total_gold:.0%}  RR R@10={cond_rr_hits}/{total_gold}={cond_rr_hits/total_gold:.0%}")
