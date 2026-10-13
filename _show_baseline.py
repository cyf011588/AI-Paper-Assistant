import json
with open('evaluation/eval_v3_result.json','r',encoding='utf-8') as f:
    base = json.load(f)
for r in base:
    print(f"{r['qid']:6s} {r['retrieval_mode']:22s} R@10={r['strict_recall_at_10']:.0%}  {r['question'][:45]}")
