import os, json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from config import PROCESSED_DIR
from chunker import Chunk
from smart_retriever import SmartRetriever

all_chunks = []
for cf in Path(PROCESSED_DIR).glob('*_chunks.json'):
    with open(cf, 'r', encoding='utf-8') as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))

r = SmartRetriever()
r.build_index(all_chunks)

question = "王鑫涛大论文第5.5节研究了什么参数的影响？结论是什么？"
print(f"问题: {question}\n")

result = r.search(question, top_k=10)
chunks = result['results']
print(f"共返回 {len(chunks)} 个chunk：\n")

# 加载标准答案chunk ids
dataset = json.load(open(r'F:\RAG评估\ai-paper-pinggu\RAG分层评测数据集_雷杰鑫与王鑫涛大论文.json','r',encoding='utf-8'))
gt_ids = set()
for item in dataset:
    if item['question_id'] == 'q012':
        gt_ids = set(item['retrieval_gt']['relevant_chunk_ids'])
        print(f"标准答案chunk ids: {gt_ids}\n")
        break

for i, c in enumerate(chunks[:30]):
    hit = "✓" if c['id'] in gt_ids else " "
    sec = c['section_path'][:55]
    text = c['text'][:60].replace('\n',' ')
    print(f"  [{i+1:2d}] {hit} {sec} | p.{c['page_start']}")
    print(f"        {text}...")
