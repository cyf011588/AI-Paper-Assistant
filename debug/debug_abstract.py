import json
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from config import PROCESSED_DIR

# 查雷杰鑫和王鑫涛大论文的英文摘要chunk
for paper in ["雷杰鑫毕业论文终稿", "王鑫涛论文终稿"]:
    cf = Path(PROCESSED_DIR) / f"{paper}_chunks.json"
    if not cf.exists():
        print(f"找不到: {cf}")
        continue
    with open(cf, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"{paper} 的摘要/Abstract相关chunks:")
    print(f"{'='*60}")
    for c in chunks:
        sp = c.get('section_path', '')
        if any(kw in sp for kw in ['Abstract', 'abstract', '摘要', '摘  要', 'Pre-content']):
            print(f"\n  id: {c['id']}")
            print(f"  section: {sp}")
            print(f"  page: {c.get('page_start')}-{c.get('page_end')}")
            print(f"  text前200: {c['text'][:200]}")
            print(f"  text长度: {len(c['text'])}")
