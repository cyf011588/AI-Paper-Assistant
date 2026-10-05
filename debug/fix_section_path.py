"""修复王鑫涛大论文的section_path：去掉图注被误识别的标题"""
import json, re
from pathlib import Path

structured_path = Path('data/processed/王鑫涛论文终稿_structured.json')
chunks_path = Path('data/processed/王鑫涛论文终稿_chunks.json')

# 读取structured
with open(structured_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 找到错误标题"14 为TS10..."，把它从section_stack里去掉
# 正确的父章节应该是"第五章 ..."或"5 ..."
bad_prefix = "14 为TS10-TC120-BS1624-CS60016、-BS2032-和-BS2440-中各部件的塑性耗能占比"

# 看看前面正常的章节是什么
for i, sec in enumerate(data['sections']):
    if bad_prefix in sec['section_path']:
        # 找前一个正常章节
        for j in range(i-1, -1, -1):
            if bad_prefix not in data['sections'][j]['section_path']:
                print(f"错误章节[{i}]的前一个正常章节: {data['sections'][j]['section_path'][:60]}")
                break
        break

# 修复：把section_path里的bad_prefix去掉
fixed = 0
for sec in data['sections']:
    if bad_prefix in sec['section_path']:
        # 去掉bad_prefix，保留后面的5.4/5.5等
        parts = sec['section_path'].split(' / ')
        new_parts = [p for p in parts if bad_prefix not in p]
        if new_parts:
            sec['section_path'] = ' / '.join(new_parts)
            fixed += 1
        else:
            sec['section_path'] = '第五章 钢框架两边连接装配式C-PSW/CE抗震性能参数分析'
            fixed += 1

print(f"修复了 {fixed} 个section_path")

# 保存structured
with open(structured_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# 修复chunks
with open(chunks_path, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

fixed_c = 0
for c in chunks:
    if bad_prefix in c['section_path']:
        parts = c['section_path'].split(' / ')
        new_parts = [p for p in parts if bad_prefix not in p]
        if new_parts:
            c['section_path'] = ' / '.join(new_parts)
        else:
            c['section_path'] = '第五章 钢框架两边连接装配式C-PSW/CE抗震性能参数分析'
        fixed_c += 1

print(f"修复了 {fixed_c} 个chunks的section_path")

with open(chunks_path, 'w', encoding='utf-8') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# 验证
print("\n修复后5.5节相关chunks：")
for c in chunks:
    if '5.5' in c['section_path']:
        print(f"  {c['section_path'][:70]}")
