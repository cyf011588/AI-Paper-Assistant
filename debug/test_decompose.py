import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, json
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
result = r.search('雷杰鑫和王鑫涛的两篇大论文在研究方法上有什么共同点和不同点？')
print()
print(f'共{len(result["results"])}个chunk')
for r2 in result['results'][:15]:
    g = r2.get('group_paper', '?')[:20]
    sec = r2['section_path'][:50]
    print(f'  [{g}] {sec} | p.{r2["page_start"]}')
