import json
d = json.load(open('evaluation/eval_v2_result.json','r',encoding='utf-8'))
for item in d:
    if item['qid'] in ['q025', 'q026', 'q022', 'q027', 'q003']:
        print(f"=== {item['qid']} G={item['gen_score']} ===")
        print(f"Q: {item['question'][:80]}")
        print(f"答案: {item.get('generated_answer','')[:200]}")
        print()
