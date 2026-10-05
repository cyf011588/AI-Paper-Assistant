import json

with open("evaluation/eval_v2_result.json", "r", encoding="utf-8") as f:
    results = json.load(f)

print("="*80)
print("Recall@10=100% 但生成=0分的题目：")
print("="*80)

for r in results:
    recall = r.get("recall_at_10", 0)
    gen = r.get("gen_score", -1)
    if recall >= 0.99 and gen == 0:
        print(f"\n{'─'*60}")
        print(f"[{r['qid']}] {r['question'][:80]}")
        print(f"  Recall={recall:.0%}  生成分={gen}  类型={r['type']}  难度={r['difficulty']}")
