"""查看4个gold chunk的实际文本"""
import os, sys, json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, 'src')
from pathlib import Path
from smart_retriever import Chunk

all_chunks = []
for cf in Path("data/processed").glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))

targets = [
    "王鑫涛论文终稿_第二章 钢框架两边连接装配式C-PSW/CE 试验研究_0034",
    "王鑫涛论文终稿_第三章 钢框架两边连接装配式C-PSW/CE 有限元模型_0063",
    "雷杰鑫毕业论文终稿_第二章 钢框架-CPSW 有限元模型_0041",
    "雷杰鑫毕业论文终稿_第二章 钢框架-CPSW 有限元模型 / 2.2 有限元模型验证_0054",
]
for c in all_chunks:
    if c.id in targets:
        print(f"\n{'='*60}")
        print(f"ID: {c.id}")
        print(f"section: {c.section_path}")
        print(f"page: {c.page_start}-{c.page_end}")
        print(f"text (前500字): {c.text[:500]}")
