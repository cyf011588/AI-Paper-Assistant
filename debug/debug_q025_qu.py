import os, json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, 'src')
from query_understanding import QueryUnderstanding

qu = QueryUnderstanding()
question = "雷杰鑫大论文和王鑫涛大论文中，框架梁在结构耗能中的角色和占比有什么异同？"
result = qu.parse(question)
print("查询理解结果：")
for k, v in result.items():
    print(f"  {k}: {v}")
