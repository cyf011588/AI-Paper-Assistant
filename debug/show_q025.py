import json
# 看q025的标准答案
dataset = json.load(open(r'F:\RAG评估\ai-paper-pinggu\RAG分层评测数据集_雷杰鑫与王鑫涛大论文.json','r',encoding='utf-8'))
for item in dataset:
    if item['question_id'] == 'q025':
        print("Q:", item['question'][:100])
        print("\n标准答案:", item['generation_gt']['standard_answer'][:300])
        print("\n正确chunk ids:", item['retrieval_gt']['relevant_chunk_ids'])
        break
