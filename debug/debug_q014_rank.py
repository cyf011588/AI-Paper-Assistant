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

question = "雷杰鑫大论文中，混凝土板在钢框架-CPSW体系中发挥了哪些作用？"
output = retriever.search(question, top_k=30)
results = output['results']

gt = ["雷杰鑫毕业论文终稿_摘  要_0002",
      "雷杰鑫毕业论文终稿_第三章 钢框架-CPSW 的抗侧力与耗能机理 / 3.4 混凝土板受力特征 / 3.4.1 混凝土板截面剪力发展规律_0069",
      "雷杰鑫毕业论文终稿_第三章 钢框架-CPSW 的抗侧力与耗能机理 / 3.7 钢框架-CPSW 的滞回性能和塑性发展 / 3.7.3 钢框架-CPSW 的耗能机理_0081",
      "雷杰鑫毕业论文终稿_第三章 钢框架-CPSW 的抗侧力与耗能机理 / 3.8 本章小节_0082"]

print(f"检索到 {len(results)} 个chunk:")
for i, r in enumerate(results):
    hit = "★" if r["id"] in gt else " "
    print(f"  {hit}[{i+1}] {r['section_path'][:60]}")
