"""
evaluate_v2.py — RAG分层评估（检索层 + 生成层）
使用专业评测数据集JSON
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import PROCESSED_DIR, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from chunker import Chunk
from smart_retriever import SmartRetriever
from openai import OpenAI


DATASET_PATH = "F:/RAG评估/ai-paper-pinggu/RAG分层评测数据集_雷杰鑫与王鑫涛大论文.json"


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
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

        # 检索层
        output = retriever.search(question, top_k=10)
        retrieved = output["results"]
        retrieved_ids = set(r["id"] for r in retrieved)
        hit_chunks = gt_chunk_ids & retrieved_ids
        recall_at_10 = len(hit_chunks) / len(gt_chunk_ids) if gt_chunk_ids else 0

        # 生成层 — 走和网页上一样的generator.py
        from generator import AnswerGenerator
        gen = AnswerGenerator()
        gen_result = gen.generate(question, retrieved[:50])
        generated_answer = gen_result["answer"]

        # 打分
        try:
            eval_resp = llm.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": """对比问题、标准答案和生成答案，给生成答案打分（0-10分）。
9-10=完全正确，6-8=基本正确有遗漏，3-5=部分正确，0-2=错误。只输出数字。"""},
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
            "qid": qid, "question": question, "type": qtype,
            "difficulty": difficulty,
            "recall_at_10": round(recall_at_10, 2),
            "gen_score": gen_score,
            "generated_answer": generated_answer[:500],
        })
        print(f"  Recall@10={recall_at_10:.0%}  生成={gen_score}/10")
        time.sleep(0.3)

    # 汇总
    print("\n" + "="*60)
    print("RAG分层评估报告")
    print("="*60)
    by_type = {}
    for r in results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = [0, 0, 0, 0]  # count, recall_sum, gen_sum
        by_type[t][0] += 1
        by_type[t][1] += r["recall_at_10"]
        by_type[t][2] += r["gen_score"]
        by_type[t][3] += 1

    print(f"\n{'题型':<10} {'题数':<5} {'Recall@10':<12} {'生成均分':<10}")
    print("-"*45)
    tr = tg = tn = 0
    for t, (n, rs, gs, _) in by_type.items():
        print(f"{t:<10} {n:<5} {rs/n:<12.0%} {gs/n:<10.1f}")
        tr += rs; tg += gs; tn += n
    print("-"*45)
    print(f"{'总计':<10} {tn:<5} {tr/tn:<12.0%} {tg/tn:<10.1f}")

    # 失败题
    failed = [r for r in results if r["recall_at_10"] < 0.5 or r["gen_score"] < 5]
    if failed:
        print(f"\n需要改进 ({len(failed)}题):")
        for r in failed:
            print(f"  [{r['qid']}] R={r['recall_at_10']:.0%} G={r['gen_score']}/10 | {r['question'][:45]}")

    with open("evaluation/eval_v2_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存 evaluation/eval_v2_result.json")


if __name__ == "__main__":
    main()
