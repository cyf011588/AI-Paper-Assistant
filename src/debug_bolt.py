import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from pathlib import Path
from smart_retriever import SmartRetriever
from chunker import Chunk

PROCESSED_DIR = Path('F:/AI_Project/AI-Paper-Assistant/data/processed')
all_chunks = []
for f in PROCESSED_DIR.glob('*_chunks.json'):
    with open(f, 'r', encoding='utf-8') as fp:
        for c in json.load(fp):
            all_chunks.append(Chunk(**c))

retriever = SmartRetriever()
retriever.build_index(all_chunks)

q = 'bolt shear force demand formula'
results = retriever.semantic.search(q, top_k=20)
print(f'Query: {q}')
print(f'前20个结果:')
for i, r in enumerate(results):
    print(f'  [{i+1}] score={r["score"]:.3f} | {r["paper_name"][:30]} | {r["section_path"][:40]}')
