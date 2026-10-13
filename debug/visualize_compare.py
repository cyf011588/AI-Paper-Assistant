"""
跨论文对比题完整过程可视化
展示：问题 → 查询理解 → 检索到的chunks → LLM回答 → 打分结果
"""
import os, json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from config import PROCESSED_DIR, EVAL_DATASET_PATH
from chunker import Chunk
from smart_retriever import SmartRetriever
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

# 1. 加载数据
all_chunks = []
for cf in Path(PROCESSED_DIR).glob('*_chunks.json'):
    with open(cf, 'r', encoding='utf-8') as f:
        for c in json.load(f):
            all_chunks.append(Chunk(**c))

retriever = SmartRetriever()
retriever.build_index(all_chunks)

llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 2. 加载评测数据集
with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# 找跨论文对比题
compare_q = None
for item in dataset:
    if item.get('question_type') == '跨论文对比':
        compare_q = item
        break

if not compare_q:
    print("没找到跨论文对比题")
    sys.exit()

question = compare_q['question']
standard_answer = compare_q['generation_gt']['standard_answer']
gt_ids = set(compare_q['retrieval_gt']['relevant_chunk_ids'])

print("=" * 80)
print("【第1步】用户问题")
print("=" * 80)
print(f"  {question}")

print("\n" + "=" * 80)
print("【第2步】RAG检索过程")
print("=" * 80)

result = retriever.search(question, top_k=10)
chunks = result['results']

print(f"\n  共检索到 {len(chunks)} 个chunk：")
for i, r in enumerate(chunks[:15]):
    g = r.get('group_paper', '全局')[:15]
    hit = "✓" if r['id'] in gt_ids else " "
    text_preview = r['text'][:80].replace('\n', ' ')
    print(f"  [{i+1:2d}] {hit} [{g}] {r['section_path'][:40]} | p.{r['page_start']}")
    print(f"        {text_preview}...")

hit_count = sum(1 for r in chunks if r['id'] in gt_ids)
print(f"\n  命中正确chunk: {hit_count}/{len(gt_ids)} (Recall={hit_count/len(gt_ids)*100:.0f}%)")

print("\n" + "=" * 80)
print("【第3步】喂给LLM的材料")
print("=" * 80)

# 按论文分组
groups = {}
for r in chunks[:30]:
    g = r.get('group_paper', r['paper_name'])
    groups.setdefault(g, []).append(r)

for paper_name, paper_chunks in groups.items():
    print(f"\n  ===== 【{paper_name}】 =====")
    for c in paper_chunks[:5]:
        text = c['text'][:200].replace('\n', ' ')
        print(f"  [{c['section_path'][:30]} p.{c['page_start']}]")
        print(f"  {text}...")

print("\n" + "=" * 80)
print("【第4步】LLM生成的回答")
print("=" * 80)

context = "\n\n".join([f"[材料]\n{r['text']}" for r in chunks[:30]])
resp = llm.chat.completions.create(
    model=DEEPSEEK_MODEL,
    messages=[
        {"role": "system", "content": "根据给定材料回答。如果材料中没有答案，就说无法回答。不要编造。"},
        {"role": "user", "content": f"材料：\n{context}\n\n问题：{question}\n\n注意：这是跨论文对比问题，请分别列出每篇论文的内容，再总结异同。"}
    ],
    temperature=0.1,
    max_tokens=2048,
)
generated = resp.choices[0].message.content
print(f"\n{generated}")

print("\n" + "=" * 80)
print("【第5步】标准答案")
print("=" * 80)
print(f"\n{standard_answer[:500]}...")

print("\n" + "=" * 80)
print("【第6步】打分模型的评价")
print("=" * 80)

score_resp = llm.chat.completions.create(
    model=DEEPSEEK_MODEL,
    messages=[
        {"role": "system", "content": """对比问题、标准答案和生成答案，给生成答案打分（0-10分）。
9-10=完全正确，6-8=基本正确有遗漏，3-5=部分正确，0-2=错误。
先输出数字，再用一句话说明理由。"""},
        {"role": "user", "content": f"问题：{question}\n标准答案：{standard_answer}\n生成答案：{generated}"}
    ],
    temperature=0.0,
    max_tokens=200,
)
print(f"\n{score_resp.choices[0].message.content}")
