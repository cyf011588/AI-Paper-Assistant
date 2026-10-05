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

# q026
question = "雷杰鑫大论文和王鑫涛大论文在研究方法上有什么共同点和不同点？"
output = retriever.search(question, top_k=15)
results = output['results']

# 标准答案
gt = {
    "雷杰鑫毕业论文终稿_第二章 钢框架-CPSW 有限元模型_0041",
    "雷杰鑫毕业论文终稿_第二章 钢框架-CPSW 有限元模型 / 2.2 有限元模型验证_0054",
    "王鑫涛论文终稿_第二章 钢框架两边连接装配式C-PSW/CE 试验研究_0034",
    "王鑫涛论文终稿_第三章 钢框架两边连接装配式C-PSW/CE 有限元模型_0063",
}

print(f"检索到 {len(results)} 个chunk:")
for i, r in enumerate(results):
    hit = "★" if r["id"] in gt else " "
    print(f"  {hit}[{i+1}] {r['id'][:60]}")
print(f"\n命中: {len(gt & set(r['id'] for r in results))}/{len(gt)}")
