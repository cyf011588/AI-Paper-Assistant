import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, 'src')
from config import PROCESSED_DIR
from chunker import Chunk
from smart_retriever import SmartRetriever
from query_understanding import QueryUnderstanding
import json
from pathlib import Path

# 1. 先看查询理解怎么解析这个问题
qu = QueryUnderstanding()
q = "雷杰鑫和王鑫涛的两篇大论文在研究方法上有什么共同点和不同点？"
result = qu.parse(q)
print("=== 查询理解 ===")
print(f"target_papers: {result['target_papers']}")
print(f"content_query: {result['content_query']}")
print(f"is_paper_specific: {result['is_paper_specific']}")

# 2. 看分解会拆成什么
print("\n=== 加载检索器 ===")
all_chunks = []
for cf in Path(PROCESSED_DIR).glob('*_chunks.json'):
    with open(cf, 'r', encoding='utf-8') as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))

retriever = SmartRetriever()
retriever.build_index(all_chunks)

# 3. 看分解结果
print("\n=== 查询分解 ===")
subs = retriever._decompose_comparison_question(q, result['target_papers'])
if subs:
    for s in subs:
        print(f"  paper: {s['paper']}")
        print(f"  sub_query: {s['sub_query']}")
else:
    print("  没有触发分解")

# 4. 实际检索
print("\n=== 实际检索 ===")
output = retriever.search(q, top_k=5)
results = output['results']
print(f"检索到 {len(results)} 个chunk:")
for i, r in enumerate(results[:10]):
    print(f"  [{i+1}] {r['paper_name'][:25]} | {r['section_path'][:40]}")
