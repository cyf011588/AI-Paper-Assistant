"""
q026 三组对照实验：A(当前brainstorm) vs B(上轮evidence) vs C(本轮改进evidence)
C组关键改进：先从论文实际章节标题提取上下文，再让LLM基于真实章节生成搜索词。
"""
import os, sys, json, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, 'src')

from smart_retriever import SmartRetriever, RECALL_K, Chunk
from reranker import Reranker
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from pathlib import Path

# ---- 初始化 ----
retriever = SmartRetriever()
all_chunks = []
for cf in Path("data/processed").glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)
reranker = Reranker()

# ---- q026 gold ----
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

SUBQUERIES = {
    "王鑫涛论文终稿": "王鑫涛论文终稿采用了哪些研究方法？",
    "雷杰鑫毕业论文终稿": "雷杰鑫毕业论文终稿采用了哪些研究方法？",
}

# ---- 从论文chunks中提取实际章节标题 ----
def get_paper_sections(paper_name):
    """提取该论文所有chunk的section_path，去重排序"""
    sections = set()
    for c in retriever.paper_chunks_map.get(paper_name, []):
        sections.add(c.section_path)
    return sorted(sections)

# ---- B组：上一轮evidence-oriented prompt ----
def evidence_expand_v2(sub_query, paper_name):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": """你是论文检索专家。用户问了一个论文问题，需要预测论文正文中可能出现的具体证据术语。
不要回答问题，不要生成答案。只输出JSON：
{"search_terms": "论文中实际可能出现的具体术语，用空格分隔"}
要求：
- 必须是论文正文中实际可能出现的具体术语（如"ABAQUS""S4R壳单元""拟静力试验""试件设计""加载制度"）
- 不要生成"研究方法""方法论""实验方法"等抽象同义词
- 根据论文的研究对象生成相关术语，不要机械套用示例
- 生成5-10个具体术语"""},
            {"role": "user", "content": f"论文：{paper_name}\n问题：{sub_query}"}
        ],
        temperature=0.1, max_tokens=300,
    )
    result = json.loads(resp.choices[0].message.content.strip())
    terms = result.get("search_terms", "")
    return sub_query + " " + terms, terms


# ---- C组：本轮改进——基于实际章节标题生成搜索词 ----
def evidence_expand_v3(sub_query, paper_name):
    """先提取论文实际章节标题，让LLM基于真实章节内容生成搜索词"""
    sections = get_paper_sections(paper_name)
    # 只取章节标题（去掉chunk_id后缀）
    section_lines = []
    for s in sections:
        # section_path格式: "论文名_章节 / 子章节_xxxx"，取前面部分
        clean = re.sub(r'_\d+$', '', s)
        # 去掉论文名前缀
        clean = clean.replace(paper_name + "_", "")
        section_lines.append(clean)
    section_text = "\n".join(section_lines[:60])  # 限制长度

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": """你是论文检索专家。用户问了一个论文问题，需要生成检索用的具体搜索词。

以下是该论文的实际章节标题列表。请从中提取与问题相关的具体术语，用于检索对应章节。

规则：
1. 只从给定的章节标题中提取术语，不要凭空生成。
2. 提取章节标题中出现的：软件名（如ABAQUS）、单元类型（如S4R、C3D8I）、试验方法（如拟静力、低周往复）、试件参数、加载制度、验证方法等具体词汇。
3. 如果章节标题中出现了与问题相关的章节（如"试验研究""有限元模型""模型验证"），直接提取这些章节标题中的关键词。
4. 不要输出"研究方法""数值模拟""理论分析""文献综述"等泛词。
5. 只输出JSON：{"search_terms": "空格分隔的具体术语", "source": "这些术语来自哪些章节"}
"""},
            {"role": "user", "content": f"论文章节标题：\n{section_text}\n\n问题：{sub_query}"}
        ],
        temperature=0.1, max_tokens=400,
    )
    result = json.loads(resp.choices[0].message.content.strip())
    terms = result.get("search_terms", "")
    source = result.get("source", "")
    return sub_query + " " + terms, terms, source


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
    return {
        "rrf_ranks": rrf_ranks,
        "rerank_ranks": rerank_ranks,
        "rrf_recall_at5": sum(1 for g in gold_ids if rrf_ranks.get(g, 999) <= 5) / len(gold_ids),
        "rrf_recall_at10": sum(1 for g in gold_ids if rrf_ranks.get(g, 999) <= 10) / len(gold_ids),
        "rerank_recall_at5": sum(1 for g in gold_ids if rerank_ranks.get(g, 999) <= 5) / len(gold_ids),
        "rerank_recall_at10": sum(1 for g in gold_ids if rerank_ranks.get(g, 999) <= 10) / len(gold_ids),
    }


# ---- 实验 ----
print("=" * 80)
print("q026 三组对照实验")
print("=" * 80)

results = {}
for paper_name, sub_q in SUBQUERIES.items():
    gold_ids = GOLD[paper_name]
    print(f"\n{'='*60}")
    print(f"论文: {paper_name}")
    print(f"子问题: {sub_q}")

    # A组
    print(f"\n--- A组：当前_brainstorm_query() ---")
    expanded_a = retriever._brainstorm_query(sub_q)
    print(f"  expanded_query: {expanded_a}")
    res_a = run_retrieval(paper_name, expanded_a, gold_ids)
    print(f"  RRF R@5={res_a['rrf_recall_at5']:.0%} R@10={res_a['rrf_recall_at10']:.0%}")
    print(f"  RR  R@5={res_a['rerank_recall_at5']:.0%} R@10={res_a['rerank_recall_at10']:.0%}")
    for g in gold_ids:
        print(f"    {g[:55]}: RRF={res_a['rrf_ranks'].get(g,'-')} RR={res_a['rerank_ranks'].get(g,'-')}")

    # B组
    print(f"\n--- B组：上轮 evidence-oriented ---")
    expanded_b, terms_b = evidence_expand_v2(sub_q, paper_name)
    print(f"  terms: {terms_b}")
    res_b = run_retrieval(paper_name, expanded_b, gold_ids)
    print(f"  RRF R@5={res_b['rrf_recall_at5']:.0%} R@10={res_b['rrf_recall_at10']:.0%}")
    print(f"  RR  R@5={res_b['rerank_recall_at5']:.0%} R@10={res_b['rerank_recall_at10']:.0%}")
    for g in gold_ids:
        print(f"    {g[:55]}: RRF={res_b['rrf_ranks'].get(g,'-')} RR={res_b['rerank_ranks'].get(g,'-')}")

    # C组
    print(f"\n--- C组：本轮改进（基于实际章节标题） ---")
    expanded_c, terms_c, source_c = evidence_expand_v3(sub_q, paper_name)
    print(f"  terms: {terms_c}")
    print(f"  source: {source_c}")
    res_c = run_retrieval(paper_name, expanded_c, gold_ids)
    print(f"  RRF R@5={res_c['rrf_recall_at5']:.0%} R@10={res_c['rrf_recall_at10']:.0%}")
    print(f"  RR  R@5={res_c['rerank_recall_at5']:.0%} R@10={res_c['rerank_recall_at10']:.0%}")
    for g in gold_ids:
        print(f"    {g[:55]}: RRF={res_c['rrf_ranks'].get(g,'-')} RR={res_c['rerank_ranks'].get(g,'-')}")

    results[paper_name] = {"A": res_a, "B": res_b, "C": res_c,
                           "terms_b": terms_b, "terms_c": terms_c, "source_c": source_c}

# ---- 汇总 ----
print(f"\n{'='*80}")
print("汇总")
print(f"{'='*80}")
total_gold = sum(len(v) for v in GOLD.values())
for label in ["A", "B", "C"]:
    rr_hits = sum(results[p][label]["rerank_recall_at10"] * len(GOLD[p]) for p in GOLD)
    rrf_hits = sum(results[p][label]["rrf_recall_at10"] * len(GOLD[p]) for p in GOLD)
    print(f"{label}组 RRF R@10={rrf_hits}/{total_gold}={rrf_hits/total_gold:.0%}  RR R@10={rr_hits}/{total_gold}={rr_hits/total_gold:.0%}")
