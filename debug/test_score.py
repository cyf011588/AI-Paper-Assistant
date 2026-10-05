"""测试打分解析"""
import sys
sys.path.insert(0, 'src')
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

question = "雷杰鑫论文中混凝土板在体系中发挥了哪些作用？"
standard = "混凝土板提供部分抗侧移刚度，承担部分水平剪力；约束钢板面外屈曲；约束钢框架弯曲变形。"
generated = "根据材料，混凝土板主要约束钢板屈曲，提高延性。"

resp = llm.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "对比问题、标准答案和生成答案，给生成答案打分（0-10分）。9-10=完全正确，6-8=基本正确有遗漏，3-5=部分正确，0-2=错误。只输出数字。"},
        {"role": "user", "content": f"问题：{question}\n标准答案：{standard}\n生成答案：{generated}"}
    ],
    temperature=0.0,
    max_tokens=20,
)
raw = resp.choices[0].message.content
print(f"原始输出: [{raw}]")
import re
nums = re.findall(r'\d+\.?\d*', raw)
print(f"提取到的数字: {nums}")
