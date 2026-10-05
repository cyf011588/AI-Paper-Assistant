import json
d = json.load(open('evaluation/eval_v2_result.json','r',encoding='utf-8'))
for item in sorted(d, key=lambda x: x['qid']):
    r = item.get('recall_at_10', 0)
    g = item.get('gen_score', 0)
    qtype = item.get('type', '')
    q = item['question'][:50]
    flag = "⚠️" if g < 5 else ""
    print(f"{flag} [{item['qid']}] {qtype:6s} R={r*100:3.0f}% G={g:4.1f}  {q}")
