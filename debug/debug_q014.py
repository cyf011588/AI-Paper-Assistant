import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, 'src')
from config import PROCESSED_DIR
from chunker import Chunk
from smart_retriever import SmartRetriever
from generator import AnswerGenerator
import json
from pathlib import Path

all_chunks = []
for cf in Path(PROCESSED_DIR).glob('*_chunks.json'):
    with open(cf, 'r', encoding='utf-8') as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever = SmartRetriever()
retriever.build_index(all_chunks)
gen = AnswerGenerator()

question = "雷杰鑫大论文中，混凝土板在钢框架-CPSW体系中发挥了哪些作用？"
output = retriever.search(question, top_k=15)
results = output['results']

print(f"检索到 {len(results)} 个chunk:")
for i, r in enumerate(results[:15]):
    print(f"  [{i+1}] {r['section_path'][:60]}")
    print(f"      text前100: {r['text'][:100]}")

print("\n" + "="*60)
answer = gen.generate(question, results[:15])
print(f"生成答案:\n{answer}")
