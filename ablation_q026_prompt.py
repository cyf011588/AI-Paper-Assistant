"""测试证据导向 Query Expansion Prompt — 调用 DeepSeek，不修改项目代码"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from openai import OpenAI

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPT = """你是论文检索专家。用户问了一个论文问题，你需要帮他扩展检索词。

**不要回答问题本身**。只输出 JSON：
{
  "evidence_types": ["这个问题应该寻找哪些类型的证据/章节"],
  "search_terms": ["论文正文中可能实际出现的具体检索术语"]
}

**硬性要求**：
1. 不要只生成"研究方法、方法论、实验方法"等抽象同义词——这些词论文正文里根本不出现。
2. 必须生成论文正文中可能实际出现的具体术语，例如：
   - 不要写"实验方法"，要写"低周往复加载试验""试件设计""加载制度"
   - 不要写"数值模拟"，要写"ABAQUS""S4R壳单元""C3D8I实体单元""材料本构模型"
   - 不要写"模型验证"，要写"滞回曲线对比""骨架曲线对比""有限元结果与试验对比"
3. search_terms 要具体到能让 BM25 关键词匹配命中正文。
4. 只输出 JSON，不要其他内容。"""

queries = [
    "王鑫涛论文终稿中使用了哪些研究方法？",
    "雷杰鑫毕业论文终稿中使用了哪些研究方法？",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"输入: {q}")
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    raw = resp.choices[0].message.content.strip()
    print(f"\nDeepSeek 原始返回:")
    print(raw)
    try:
        parsed = json.loads(raw)
        print(f"\n解析后 JSON:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except:
        print("\n[解析失败，原样输出]")
