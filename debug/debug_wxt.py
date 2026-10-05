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

# 直接用向量库搜索，filter到王鑫涛论文，看"研究方法"能搜到什么
print("=== 王鑫涛论文中搜'研究方法' ===")
results = retriever.vector_db.search("研究方法", top_k=20, where={"paper_name": {"$eq": "王鑫涛论文终稿"}})
for i, r in enumerate(results):
    print(f"  [{i+1}] {r['section_path'][:60]}")
    print(f"      text前80: {r['text'][:80]}")

print("\n=== 王鑫涛第二章的所有chunks ===")
for c in retriever.paper_chunks_map.get("王鑫涛论文终稿", []):
    if "第二章" in c.section_path or "第三章" in c.section_path:
        print(f"  {c.id[:60]} | {c.section_path[:50]}")
