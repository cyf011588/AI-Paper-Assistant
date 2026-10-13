"""test_shortcut_fix.py — 回归测试 + 章节匹配单元测试"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from smart_retriever import SmartRetriever

# ========== 单元测试：章节匹配函数 ==========
print("="*60)
print("单元测试：_match_section_number")
print("="*60)

tests = [
    # (section_path, target, expected)
    ("第四章 钢框架-CPSW 抗震性能参数分析 / 4.3 框架梁截面影响 / 4.3.1 滞回曲线", "4", True),
    ("第四章 钢框架-CPSW 抗震性能参数分析 / 4.3 框架梁截面影响 / 4.3.1 滞回曲线", "4.3", True),
    ("第四章 钢框架-CPSW 抗震性能参数分析 / 4.3 框架梁截面影响 / 4.3.1 滞回曲线", "4.3.1", True),
    ("第二章 试验研究 / 2.1 试件信息 / 2.1.4 数据采集方案", "4", False),
    ("第三章 机理分析 / 3.4 混凝土板受力", "4", False),
    ("第五章 承载力设计 / 5.1 力学模型 / 5.2 计算公式", "5", True),
    ("第五章 承载力设计 / 5.1 力学模型", "5.1", True),
    ("第五章 承载力设计 / 5.3 框架梁设计", "5.3", True),
    ("第三章 机理分析 / 3.5 钢框架受力", "5", False),
    ("第四章 参数分析 / 4.5 混凝土板厚度", "5", False),
    ("第四章 参数分析 / 4.6 柱轴压比 / 4.6.1 滞回曲线", "4.6", True),
    ("第四章 参数分析 / 4.6 柱轴压比 / 4.6.1 滞回曲线", "4.6.1", True),
    ("第三章 机理分析 / 3.4.6 某小节", "4.6", False),
    ("第五章 承载力设计 / 5.4.6 某小节", "4.6", False),
]

passed = 0
failed = 0
for sp, target, expected in tests:
    result = SmartRetriever._match_section_number(sp, target)
    ok = result == expected
    sign = "✅" if ok else "❌"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {sign} target={target:6s} | {sp[:55]:55s} | expected={expected} got={result}")

print(f"\n单元测试: {passed} passed, {failed} failed")

# ========== 回归测试：4个失败案例 ==========
print("\n" + "="*60)
print("回归测试：q001/q004/q008/q009")
print("="*60)

from config import PROCESSED_DIR
from chunker import Chunk
import json

retriever = SmartRetriever()
all_chunks = []
for cf in Path(PROCESSED_DIR).glob("*_chunks.json"):
    with open(cf, "r", encoding="utf-8") as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever.build_index(all_chunks)

cases = {
    "q001": "雷杰鑫大论文《钢框架内填钢板外包混凝土组合剪力墙的滞回性能与抗震设计方法》研究的核心构件是什么？简称是什么？",
    "q004": "雷杰鑫大论文《钢框架内填钢板外包混凝土组合剪力墙的滞回性能与抗震设计方法》的主要研究内容有哪些？",
    "q008": "王鑫涛大论文第四章的研究主题是什么？",
    "q009": "雷杰鑫大论文第五章提出了哪些设计方法与公式？分别位于哪些小节？",
}

# gold chunks
gold_map = {
    "q001": ["雷杰鑫毕业论文终稿_摘  要_0002",
             "雷杰鑫毕业论文终稿_第一章 引  言 / 1.3 目前研究不足和本文研究内容 / 1.3.2 本文研究内容_0040"],
    "q004": ["雷杰鑫毕业论文终稿_摘  要_0002",
             "雷杰鑫毕业论文终稿_第六章 结论与展望 / 6.1 本文结论_0123",
             "雷杰鑫毕业论文终稿_第一章 引  言 / 1.3 目前研究不足和本文研究内容 / 1.3.2 本文研究内容_0040"],
    "q008": ["王鑫涛论文终稿_20 所示。从整体上看，结构弹塑性阶段耗能的主要部件是钢板墙和混凝土板；进入塑 / 4.7.2 滞回性能 / 4.7.3 钢框架两边连接装配式C",
             "王鑫涛论文终稿_第四章 钢框架两边连接装配式C-PSW/CE 受剪机理与耗能机_0076"],
    "q009": ["雷杰鑫毕业论文终稿_第五章 钢框架-CPSW 承载力和梁截面设计方法 / 5.2 钢框架-CPSW 水平承载力与刚度计算公式_0114",
             "雷杰鑫毕业论文终稿_第五章 钢框架-CPSW 承载力和梁截面设计方法 / 5.3 框架梁截面设计方法与结构设计建议_0121",
             "雷杰鑫毕业论文终稿_第五章 钢框架-CPSW 承载力和梁截面设计方法 / 5.2 钢框架-CPSW 水平承载力与刚度计算公式_0112",
             "雷杰鑫毕业论文终稿_第五章 钢框架-CPSW 承载力和梁截面设计方法 / 5.1 钢框架-CPSW 的侧向力-层间侧移角力学模型_0111"],
}

for qid, q in cases.items():
    output = retriever.search(q, top_k=10)
    debug = output.get("debug_info", {})
    mode = debug.get("retrieval_mode", "?")
    returned_ids = [r["id"] for r in output["results"]]
    gold = set(gold_map[qid])
    hit = gold & set(returned_ids[:10])

    entered_shortcut = mode == "section_shortcut"
    print(f"\n[{qid}]")
    print(f"  是否进入shortcut: {entered_shortcut} (mode={mode})")
    print(f"  返回前5个chunk:")
    for i, cid in enumerate(returned_ids[:5]):
        mark = " <<< GOLD" if cid in gold else ""
        print(f"    {i+1}. {cid[:65]}{mark}")
    print(f"  gold命中: {len(hit)}/{len(gold)}")
