import json
d = json.load(open('evaluation/eval_v2_result.json','r',encoding='utf-8'))
for item in d:
    if item.get('type') == '按章节问':
        r = item.get('recall_at_10', 0)
        g = item.get('gen_score', 0)
        print(f"[{item['qid']}] R={r*100:.0f}% G={g}")
        print(f"  Q: {item['question'][:70]}")
        print()
