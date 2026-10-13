import json
from pathlib import Path
# 看q025的标准答案
_DATASET = Path(__file__).resolve().parents[1] / "evaluation" / "dataset" / "RAG分层评测数据集_雷杰鑫与王鑫涛大论文.json"
dataset = json.load(open(_DATASET, 'r', encoding='utf-8'))
for item in dataset:
    if item['question_id'] == 'q025':
        print("Q:", item['question'][:100])
        print("\n标准答案:", item['generation_gt']['standard_answer'][:300])
        print("\n正确chunk ids:", item['retrieval_gt']['relevant_chunk_ids'])
        break
