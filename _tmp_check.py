import json
from pathlib import Path
_DATASET = Path(__file__).resolve().parent / "evaluation" / "dataset" / "RAG分层评测数据集_雷杰鑫与王鑫涛大论文.json"
with open(_DATASET, 'r', encoding='utf-8') as f:
    data=json.load(f)
print(f'总题数: {len(data)}')
types={}
for q in data:
    t=q['question_type']
    types[t]=types.get(t,0)+1
print('题型分布:')
for t,n in types.items():
    print(f'  {t}: {n}题')
print()
for q in data:
    gts=q['retrieval_gt']['relevant_chunk_ids']
    qid=q['question_id']
    qtext=q['question'][:45]
    print(f'  {qid}: {len(gts)}个gold | {qtext}')
