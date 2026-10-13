"""
q026 无泄漏自动生成实验：只给章节标题，不给gold正文。
对比：auto（自动生成）vs C-short（人工参考上限）
"""
import os, sys, json, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, 'src')

from smart_retriever import SmartRetriever, RECALL_K, Chunk
from reranker import Reranker
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
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

SUBQUERIES = {
    "王鑫涛论文终稿": "王鑫涛论文终稿采用了哪些研究方法？",
    "雷杰鑫毕业论文终稿": "雷杰鑫毕业论文终稿采用了哪些研究方法？",
}

# C-short 人工参考（上一轮结果，仅作对比，不作为输入）
C_SHORT_HUMAN = {
    "王鑫涛论文终稿": "低周往复加载试验 ABAQUS 有限元模型 有限元算例 试验现象 材料本构 初始缺陷 参数影响分析",
    "雷杰鑫毕业论文终稿": "ABAQUS 有限元模型 有限元算例 试验验证 骨架曲线 滞回曲线 材料本构 初始缺陷",
}


def get_paper_toc(paper_name):
    """提取论文目录：只返回章节标题路径，不返回正文"""
    seen = set()
    toc = []
    for c in retriever.paper_chunks_map.get(paper_name, []):
        # section_path格式: "论文名_章 / 节 / 小节_xxxx"
        # 去掉chunk_id后缀和论文名前缀
        sp = c.section_path
        sp = re.sub(r'_\d+$', '', sp)
        sp = sp.replace(paper_name + "_", "")
        if sp not in seen:
            seen.add(sp)
            toc.append(sp)
    return sorted(toc)


def auto_generate_terms(question, paper_name, toc):
    """只给章节标题，让DeepSeek生成≤8个搜索词"""
    toc_text = "\n".join(toc)
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": """你是论文检索专家。用户问了一个论文的"研究方法"问题。
以下是该论文的章节目录（只有标题，没有正文）。请根据目录，生成最多8个搜索词，用于检索与"研究方法"相关的章节。

规则：
1. 只从目录标题中提取术语，不要凭空想象论文中可能有什么。
2. 优先提取目录中与"试验""有限元""模型""验证""加载""单元类型""本构"相关的章节关键词。
3. 不要生成"研究方法""方法论""理论分析"等泛词。
4. 每个词必须能在目录标题中找到对应出处。
5. 输出JSON：{"terms": ["词1","词2",...], "sources": {"词1":"对应章节标题",...}}
"""},
            {"role": "user", "content": f"论文：{paper_name}\n问题：{question}\n\n目录：\n{toc_text}"}
        ],
        temperature=0.1, max_tokens=500,
    )
    result = json.loads(resp.choices[0].message.content.strip())
    terms = result.get("terms", [])
    sources = result.get("sources", {})
    return terms, sources


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


# ---- 实验 ----
print("=" * 80)
print("q026 无泄漏自动生成实验")
print("=" * 80)

results = {}
for paper_name, sub_q in SUBQUERIES.items():
    gold_ids = GOLD[paper_name]
    print(f"\n{'='*60}")
    print(f"论文: {paper_name}")

    # 获取目录
    toc = get_paper_toc(paper_name)
    print(f"目录（{len(toc)}个章节标题）:")
    for t in toc[:20]:
        print(f"  {t}")

    # 自动生成
    print(f"\n--- 自动生成（仅基于目录） ---")
    terms, sources = auto_generate_terms(sub_q, paper_name, toc)
    terms_str = " ".join(terms)
    print(f"  terms({len(terms)}个): {terms_str}")
    for t, s in sources.items():
        print(f"    {t} ← {s}")

    query_auto = sub_q + " " + terms_str
    rrf_ranks, rr_ranks = run_retrieval(paper_name, query_auto, gold_ids)
    rrf_hits = sum(1 for g in gold_ids if g in rrf_ranks and rrf_ranks[g] <= 10)
    rr_hits = sum(1 for g in gold_ids if g in rr_ranks and rr_ranks[g] <= 10)
    print(f"  RRF R@10={rrf_hits}/{len(gold_ids)}  RR R@10={rr_hits}/{len(gold_ids)}")
    for g in gold_ids:
        print(f"    {g[:55]}: RRF={rrf_ranks.get(g,'-')} RR={rr_ranks.get(g,'-')}")

    # 人工C-short（对比）
    print(f"\n--- 人工C-short（参考上限） ---")
    human_terms = C_SHORT_HUMAN[paper_name]
    print(f"  terms: {human_terms}")
    query_human = sub_q + " " + human_terms
    rrf_h, rr_h = run_retrieval(paper_name, query_human, gold_ids)
    rrf_hits_h = sum(1 for g in gold_ids if g in rrf_h and rrf_h[g] <= 10)
    rr_hits_h = sum(1 for g in gold_ids if g in rr_h and rr_h[g] <= 10)
    print(f"  RRF R@10={rrf_hits_h}/{len(gold_ids)}  RR R@10={rr_hits_h}/{len(gold_ids)}")
    for g in gold_ids:
        print(f"    {g[:55]}: RRF={rrf_h.get(g,'-')} RR={rr_h.get(g,'-')}")

    results[paper_name] = {
        "auto_rrf": rrf_ranks, "auto_rr": rr_ranks,
        "human_rrf": rrf_h, "human_rr": rr_h,
        "auto_terms": terms, "auto_sources": sources,
    }

# 汇总
print(f"\n{'='*80}")
print("汇总")
print(f"{'='*80}")
total = sum(len(v) for v in GOLD.values())
auto_rrf = sum(1 for p in GOLD for g in GOLD[p] if results[p]["auto_rrf"].get(g, 999) <= 10)
auto_rr = sum(1 for p in GOLD for g in GOLD[p] if results[p]["auto_rr"].get(g, 999) <= 10)
human_rrf = sum(1 for p in GOLD for g in GOLD[p] if results[p]["human_rrf"].get(g, 999) <= 10)
human_rr = sum(1 for p in GOLD for g in GOLD[p] if results[p]["human_rr"].get(g, 999) <= 10)
print(f"自动生成: RRF R@10={auto_rrf}/{total}={auto_rrf/total:.0%}  RR R@10={auto_rr}/{total}={auto_rr/total:.0%}")
print(f"人工参考: RRF R@10={human_rrf}/{total}={human_rrf/total:.0%}  RR R@10={human_rr}/{total}={human_rr/total:.0%}")
