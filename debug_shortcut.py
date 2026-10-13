"""debug_shortcut.py — 分析section_shortcut和section_list题"""
import json
with open("evaluation/eval_v3_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== section_shortcut / section_list 题 ===")
for r in data:
    if r["retrieval_mode"] in ("section_shortcut", "section_list"):
        print(f"\n{r['qid']} | {r['retrieval_mode']} | gold={len(r['gold_chunk_ids'])} | recall={r['strict_recall_at_10']:.0%}")
        print(f"  Q: {r['question']}")
        print(f"  target_papers: {r.get('target_papers', [])}")
        print(f"  gold chunks:")
        for g in r["gold_chunk_ids"]:
            print(f"    - {g[:80]}")
        print(f"  returned (pre_expansion_top10):")
        for i, cid in enumerate(r["pre_expansion_top10_ids"][:10]):
            mark = " <<< GOLD" if cid in r["gold_chunk_ids"] else ""
            print(f"    {i+1}. {cid[:80]}{mark}")
