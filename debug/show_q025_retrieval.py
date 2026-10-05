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

question = "雷杰鑫大论文和王鑫涛大论文中，框架梁在结构耗能中的角色和占比有什么异同？"
result = r.search(question, top_k=10)

gt_ids = {
    '雷杰鑫毕业论文终稿_第四章 钢框架-CPSW 抗震性能参数分析 / 4.3 框架梁截面影响 / 4.3.4 各部件塑性耗能_0095',
    '雷杰鑫毕业论文终稿_第四章 钢框架-CPSW 抗震性能参数分析 / 4.7 本章小节_0109',
    '王鑫涛论文终稿_20 所示。从整体上看，结构弹塑性阶段耗能的主要部件是钢板墙和混凝土板；进入塑_0092',
    '王鑫涛论文终稿_第七章 结论与展望 / 7.1 本文结论_0145',
}

print(f"共返回 {len(result['results'])} 个chunk：\n")
for i, c in enumerate(result['results'][:15]):
    hit = "✓" if c['id'] in gt_ids else " "
    paper = c['paper_name'][:15]
    sec = c['section_path'][:50]
    text = c['text'][:50].replace('\n',' ')
    print(f"  [{i+1:2d}] {hit} [{paper}] {sec}")
    print(f"        {text}...")
