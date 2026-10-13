"""
evaluate_v2.py — RAG分层评估（检索层 + 生成层）
使用专业评测数据集JSON

v3: 修正Recall口径
- strict_recall_at_10: 只看 pre_expansion top10（reranker后，expand前）
- recall_after_expansion: 看 post_expansion 全部
- retrieval_mode: hybrid_rerank / section_shortcut / comparison_decompose / section_list
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import PROCESSED_DIR, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, require_eval_dataset
from chunker import Chunk
from smart_retriever import SmartRetriever
from openai import OpenAI


def main():
    # 公开仓库不发布评测集（含论文原文）——缺失时给出可操作提示而不是直接崩
    try:
        dataset_path = require_eval_dataset()
    except FileNotFoundError as e:
        print(e)
        return 1

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"加载 {len(dataset)} 道测试题\n")

    print("初始化检索器...")
    retriever = SmartRetriever()
    all_chunks = []
    for cf in Path(PROCESSED_DIR).glob("*_chunks.json"):
        with open(cf, "r", encoding="utf-8") as f:
            for c in json.load(f):
                all_chunks.append(Chunk(**c))
    retriever.build_index(all_chunks)

    llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    results = []

    for i, item in enumerate(dataset):
        qid = item["question_id"]
        question = item["question"]
        gt_chunk_ids = set(item["retrieval_gt"]["relevant_chunk_ids"])
        standard_answer = item["generation_gt"]["standard_answer"]
        qtype = item["question_type"]
        difficulty = item["difficulty"]

        print(f"[{i+1}/{len(dataset)}] {qid}: {question[:50]}...")

        # 检索
        output = retriever.search(question, top_k=10)
        debug = output.get("debug_info", {})
        retrieval_mode = debug.get("retrieval_mode", "unknown")

        pre_ids = debug.get("pre_expansion_ids", [])
        post_ids = debug.get("post_expansion_ids", [])

        # strict Recall@10: 只看 pre_expansion 前10个
        pre_top10 = pre_ids[:10]
        strict_recall = len(gt_chunk_ids & set(pre_top10)) / len(gt_chunk_ids) if gt_chunk_ids else 0

        # recall after expansion: 看 post_expansion 全部
        recall_after_exp = len(gt_chunk_ids & set(post_ids)) / len(gt_chunk_ids) if gt_chunk_ids else 0

        # 生成层（暂时不改）
        from generator import AnswerGenerator
        gen = AnswerGenerator()
        gen_result = gen.generate(question, output["results"][:50])
        generated_answer = gen_result["answer"]

        # LLM打分
        try:
            eval_resp = llm.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "对比问题、标准答案和生成答案，给生成答案打分（0-10分）。9-10=完全正确，6-8=基本正确有遗漏，3-5=部分正确，0-2=错误。只输出数字。"},
                    {"role": "user", "content": f"问题：{question}\n标准答案：{standard_answer}\n生成答案：{generated_answer}"}
                ],
                temperature=0.0,
                max_tokens=10,
            )
            score_text = eval_resp.choices[0].message.content.strip()
            gen_score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
        except:
            gen_score = 0

        results.append({
            "qid": qid,
            "question": question,
            "type": qtype,
            "difficulty": difficulty,
            "retrieval_mode": retrieval_mode,
            "pre_expansion_top10_ids": pre_top10,
            "post_expansion_ids": post_ids,
            "gold_chunk_ids": list(gt_chunk_ids),
            "strict_recall_at_10": round(strict_recall, 2),
            "recall_after_expansion": round(recall_after_exp, 2),
            "gen_score": gen_score,
            "generated_answer": generated_answer[:500],
        })
        print(f"  mode={retrieval_mode:20s}  strict_R@10={strict_recall:.0%}  after_exp={recall_after_exp:.0%}  gen={gen_score}/10")
        time.sleep(0.3)

    # ===== 汇总 =====
    print("\n" + "="*70)
    print("RAG分层评估报告（v3 修正口径）")
    print("="*70)

    # 全部
    all_strict = [r["strict_recall_at_10"] for r in results]
    all_after = [r["recall_after_expansion"] for r in results]
    all_gen = [r["gen_score"] for r in results]
    print(f"\n全部30题:")
    print(f"  Strict Recall@10    = {sum(all_strict)/len(all_strict):.1%}")
    print(f"  Recall after expand = {sum(all_after)/len(all_after):.1%}")
    print(f"  生成均分            = {sum(all_gen)/len(all_gen):.1f}/10")

    # 按retrieval_mode分组
    by_mode = {}
    for r in results:
        m = r["retrieval_mode"]
        if m not in by_mode:
            by_mode[m] = []
        by_mode[m].append(r)

    print(f"\n按检索模式分组:")
    for mode, rs in by_mode.items():
        s = sum(r["strict_recall_at_10"] for r in rs) / len(rs)
        a = sum(r["recall_after_expansion"] for r in rs) / len(rs)
        print(f"  {mode:25s}: {len(rs)}题  strict_R@10={s:.0%}  after_exp={a:.0%}")

    # 失败题
    failed = [r for r in results if r["strict_recall_at_10"] < 0.5]
    if failed:
        print(f"\nStrict Recall@10 < 50% 的题 ({len(failed)}题):")
        for r in failed:
            print(f"  [{r['qid']}] mode={r['retrieval_mode']:20s} strict={r['strict_recall_at_10']:.0%} | {r['question'][:50]}")

    # 保存
    out_dir = Path("evaluation")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "eval_v3_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 evaluation/eval_v3_result.json")


if __name__ == "__main__":
    main()
