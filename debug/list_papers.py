import json
meta = json.load(open('data/paper_metadata.json','r',encoding='utf-8'))
print(f'论文数: {len(meta)}')
for p in meta:
    name = p.get('paper_name', p.get('name', '?'))
    chunks = p.get('chunk_count', '?')
    print(f'  - {name} ({chunks} chunks)')
