import json
with open('evaluation/eval_v2_result.json','r',encoding='utf-8') as f:
    data = json.load(f)
for r in data:
    if r['recall_at_10'] == 1.0 and r['gen_score'] == 0:
        q = r['question'][:50]
        print(f"{r['qid']}: R=100% G=0 | {q}")
