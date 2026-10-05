import os, sys, base64
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, 'src')
from openai import OpenAI
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL

image_path = r"C:\Users\齐老师课题组\AppData\Local\Doubao\User Data\ClipboardTemp\36cc00aa-f92c-434e-b596-56c3b758e8f6.png"
with open(image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)

model = "Qwen/Qwen3-VL-30B-A3B-Instruct"
print(f"测试: {model}")
try:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "请详细描述该图表，并提取其中的关键数据点（如峰值、坐标、趋势）。"}
        ]}],
    )
    print("成功!")
    print(resp.choices[0].message.content[:300])
except Exception as e:
    print(f"失败: {str(e)[:200]}")
