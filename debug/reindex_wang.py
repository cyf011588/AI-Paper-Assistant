"""重新入库王鑫涛论文终稿（修复section_path后）"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, 'src')

from vector_store import VectorStore
from chunker import Chunk
import json

vs = VectorStore()

# 1. 删除旧chunks
vs.collection.delete(where={"paper_name": "王鑫涛论文终稿"})
print("已删除旧chunks")

# 2. 读取修复后的chunks
with open('data/processed/王鑫涛论文终稿_chunks.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)
chunks = [Chunk(**c) for c in raw]
print(f"读取到 {len(chunks)} 个chunks")

# 3. 重新入库
vs.add_chunks(chunks)
