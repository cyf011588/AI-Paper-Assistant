import json
with open('data/processed/王鑫涛论文终稿_structured.json','r',encoding='utf-8') as f:
    data = json.load(f)
bad = 0
for i, sec in enumerate(data['sections']):
    sp = sec['section_path']
    # 检查是否像图注（数字+为/是/的...）
    import re
    m = re.match(r'^(\d+)\s+[为是的了在有和与或]', sp)
    if m:
        bad += 1
        print(f"[{i}] level={sec['section_level']} {sp[:70]}")
print(f"\n共 {bad} 个错误章节（图注被当成标题）")
