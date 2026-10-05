import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, 'src')
from config import PROCESSED_DIR
from chunker import Chunk
from smart_retriever import SmartRetriever
import json
from pathlib import Path

all_chunks = []
for cf in Path(PROCESSED_DIR).glob('*_chunks.json'):
    with open(cf, 'r', encoding='utf-8') as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever = SmartRetriever()
retriever.build_index(all_chunks)

# q021
question = '王鑫涛大论文英文摘要中的"links"指的是哪个构件？其中文名称是什么？'
output = retriever.search(question, top_k=30)
results = output['results']

print(f"检索到 {len(results)} 个chunk:")
for i, r in enumerate(results[:30]):
    print(f"  [{i+1}] {r['section_path'][:60]} | p.{r.get('page_start','?')}")
    print(f"      text前80: {r['text'][:80]}")
