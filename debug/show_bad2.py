import json
d = json.load(open('evaluation/eval_v2_result.json','r',encoding='utf-8'))
for item in d:
    if item['qid'] in ['q025', 'q027', 'q022']:
        print(f"=== {item['qid']} G={item['gen_score']} ===")
        print(f"答案: {item.get('generated_answer','(空)')[:300]}")
        print()
