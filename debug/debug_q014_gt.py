import json
from pathlib import Path
_DATASET = Path(__file__).resolve().parents[1] / "evaluation" / "dataset" / "RAG分层评测数据集_雷杰鑫与王鑫涛大论文.json"
with open(_DATASET, "r", encoding="utf-8") as f:
    data = json.load(f)

for q in data:
    if q["question_id"] == "q014":
        print(f"问题: {q['question']}")
        print(f"\n标准答案chunks:")
        for cid in q["retrieval_gt"]["relevant_chunk_ids"]:
            print(f"  - {cid}")
        print(f"\n标准答案: {q['generation_gt']['standard_answer'][:300]}")
