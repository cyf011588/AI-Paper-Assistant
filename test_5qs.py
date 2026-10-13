"""test_5qs.py — 验证q001/q004/q008/q009/q028"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import PROCESSED_DIR
from chunker import Chunk
from smart_retriever import SmartRetriever
import json

retriever = SmartRetriever()
all_chunks = []
for cf in Path(PROCESSED_DIR).glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)

cases = {
    "q001": ("雷杰鑫大论文《钢框架内填钢板外包混凝土组合剪力墙的滞回性能与抗震设计方法》研究的核心构件是什么？简称是什么？",
             ["雷杰鑫毕业论文终稿_摘  要_0002",
              "雷杰鑫毕业论文终稿_第一章 引  言 / 1.3 目前研究不足和本文研究内容 / 1.3.2 本文研究内容_0040"]),
    "q004": ("雷杰鑫大论文《钢框架内填钢板外包混凝土组合剪力墙的滞回性能与抗震设计方法》的主要研究内容有哪些？",
             ["雷杰鑫毕业论文终稿_摘  要_0002",
              "雷杰鑫毕业论文终稿_第六章 结论与展望 / 6.1 本文结论_0123",
              "雷杰鑫毕业论文终稿_第一章 引  言 / 1.3 目前研究不足和本文研究内容 / 1.3.2 本文研究内容_0040"]),
    "q008": ("王鑫涛大论文第四章的研究主题是什么？",
             ["王鑫涛论文终稿_20 所示。从整体上看，结构弹塑性阶段耗能的主要部件是钢板墙和混凝土板；进入塑 / 4.7.2 滞回性能 / 4.7.3 钢框架两边连接装配式C",
              "王鑫涛论文终稿_第四章 钢框架两边连接装配式C-PSW/CE 受剪机理与耗能机_0076"]),
    "q009": ("雷杰鑫大论文第五章提出了哪些设计方法与公式？分别位于哪些小节？",
             ["雷杰鑫毕业论文终稿_第五章 钢框架-CPSW 承载力和梁截面设计方法 / 5.2 钢框架-CPSW 水平承载力与刚度计算公式_0114",
              "雷杰鑫毕业论文终稿_第五章 钢框架-CPSW 承载力和梁截面设计方法 / 5.3 框架梁截面设计方法与结构设计建议_0121",
              "雷杰鑫毕业论文终稿_第五章 钢框架-CPSW 承载力和梁截面设计方法 / 5.2 钢框架-CPSW 水平承载力与刚度计算公式_0112",
              "雷杰鑫毕业论文终稿_第五章 钢框架-CPSW 承载力和梁截面设计方法 / 5.1 钢框架-CPSW 的侧向力-层间侧移角力学模型_0111"]),
    "q028": ("王鑫涛大论文的试验试件是什么规模和形式？试验中各部件的屈服顺序是什么？",
             ["王鑫涛论文终稿_第二章 钢框架两边连接装配式C-PSW/CE 试验研究 / 2.1 试件信息与试验方案 / 2.1.1 试件尺寸_0035",
              "王鑫涛论文终稿_21 所示。从横梁应变情况看，其受剪切屈服早于弯曲屈服，且如上文分析，横梁两端 / 2.4 本章小结_0062"]),
}

for qid, (q, gold) in cases.items():
    output = retriever.search(q, top_k=10)
    debug = output.get("debug_info", {})
    mode = debug.get("retrieval_mode", "?")
    returned = [r["id"] for r in output["results"]]
    gold_set = set(gold)
    hit = gold_set & set(returned[:10])
    recall = len(hit)/len(gold) if gold else 0

    print(f"\n[{qid}] mode={mode}  recall={recall:.0%}({len(hit)}/{len(gold)})")
    for i, cid in enumerate(returned[:5]):
        mark = " <<< GOLD" if cid in gold_set else ""
        print(f"  {i+1}. {cid[:70]}{mark}")
