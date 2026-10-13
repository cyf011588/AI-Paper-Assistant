"""查看完整数据结构和comparison_decompose题目的gold"""
import json
from pathlib import Path
_DATASET = Path(__file__).resolve().parent / "evaluation" / "dataset" / "RAG分层评测数据集_雷杰鑫与王鑫涛大论文.json"
with open(_DATASET, "r", encoding="utf-8") as f:
    data = json.load(f)

# 看第一题完整结构
print(json.dumps(data[0], ensure_ascii=False, indent=2)[:2000])
print("\n\n===== q024 =====")
print(json.dumps(data[23], ensure_ascii=False, indent=2)[:2000])
