import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, 'src')
from config import PROCESSED_DIR, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from chunker import Chunk
from smart_retriever import SmartRetriever
import json
from pathlib import Path
from openai import OpenAI

# 加载
all_chunks = []
for cf in Path(PROCESSED_DIR).glob('*_chunks.json'):
    with open(cf, 'r', encoding='utf-8') as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))
retriever = SmartRetriever()
retriever.build_index(all_chunks)

# q013: 塑性发展顺序
question = "雷杰鑫大论文中，钢框架-CPSW合理的塑性发展顺序是什么？"
output = retriever.search(question, top_k=10)
contexts = output['results']
print(f"检索到 {len(contexts)} 个chunk:")
for i, r in enumerate(contexts[:5]):
    print(f"  [{i+1}] {r['paper_name'][:20]} | {r['section_path'][:40]}")
    print(f"      text前100字: {r['text'][:100]}")

# 生成答案
context = "\n\n".join([f"[材料]\n{r['text']}" for r in contexts[:5]])
llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
gen = llm.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "根据给定材料回答用户问题。如果材料中没有答案，就说'根据现有材料无法回答'。不要编造。"},
        {"role": "user", "content": f"材料：\n{context}\n\n问题：{question}"}
    ],
    temperature=0.1,
)
answer = gen.choices[0].message.content
print(f"\n生成答案:\n{answer[:500]}")

# 打分
standard = "钢板剪切屈服→梁、柱剪切屈服→梁受弯屈服→钢板塑性屈曲→柱受弯屈服"
score_resp = llm.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "对比问题、标准答案和生成答案，给生成答案打分（0-10分）。9-10=完全正确，6-8=基本正确有遗漏，3-5=部分正确，0-2=错误。只输出数字。"},
        {"role": "user", "content": f"问题：{question}\n标准答案：{standard}\n生成答案：{answer}"}
    ],
    temperature=0.0,
    max_tokens=20,
)
print(f"\n打分原始输出: [{score_resp.choices[0].message.content}]")
