"""
vision_extractor.py — 图片特征提取
对不同类型图片用不同Prompt调Qwen-VL，输出结构化JSON
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import base64
import json
from openai import OpenAI
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, VL_MODEL_NAME


PROMPTS = {
    "curve": """请详细分析这张曲线图，提取以下信息并输出JSON：
{
  "chart_type": "曲线类型",
  "x_axis": "X轴含义和范围",
  "y_axis": "Y轴含义和范围",
  "key_points": [
    {"label": "点名称", "x": "X值", "y": "Y值", "description": "该点的物理意义"}
  ],
  "trend": "整体趋势描述",
  "stages": [{"name": "阶段名", "range": "范围", "behavior": "特征"}],
  "notes": "其他关键观察"
}""",

    "contour": """请按以下顺序逐条分析这张应力云图，不允许跳过任何一条：
1. 整体颜色分布：红色（高应力）集中在哪些区域？蓝色（低应力）在哪些区域？
2. 最大应力位置：具体在哪个部件的哪个位置（如"内嵌钢板左下角"、"框架柱脚外翼缘"）？
3. 应力集中区域：是否有明显的应力集中？形状是什么？
4. 屈服区域扩展：与上一荷载步相比，屈服区域是否扩大？向哪个方向扩展？
5. 破坏模式判断：根据应力分布，推断该部位可能发生什么破坏（剪切屈服/弯曲屈曲/局部鼓曲）？
6. 数值：图中能看清的最大应力值是多少MPa？

输出JSON格式：
{
  "color_distribution": "第1条回答",
  "max_stress_location": "第2条回答",
  "stress_concentration": "第3条回答",
  "yield_progression": "第4条回答",
  "failure_mode": "第5条回答",
  "max_stress_value": "第6条回答"
}""",

    "photo": """请详细分析这张试验照片，提取以下信息并输出JSON：
{
  "specimen_state": "试件整体状态",
  "visible_damage": ["可见损伤列表"],
  "crack_description": "裂缝描述（位置、宽度、方向）",
  "buckling": "屈曲描述（如有）",
  "notes": "其他关键观察"
}""",
}


def extract_figure_features(image_path: str, fig_type: str) -> dict:
    """对单张图片提取结构化特征"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".tif": "image/tiff", ".tiff": "image/tiff"}
    mime = mime_map.get(ext, "image/jpeg")

    # TIF需要转成PNG再发给API
    if ext in (".tif", ".tiff"):
        try:
            import cv2
            import numpy as np
            # 用numpy读二进制，避免cv2.imread不支持中文路径
            with open(image_path, "rb") as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise Exception("cv2解码失败")
            # 转RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 缩放
            h, w = img.shape[:2]
            if max(h, w) > 2000:
                scale = 2000 / max(h, w)
                img = cv2.resize(img, (int(w*scale), int(h*scale)))
            buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
            b64 = base64.b64encode(buf.tobytes()).decode()
            mime = "image/png"
        except Exception as e:
            return {"error": f"TIF转换失败: {e}"}
    else:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

    prompt = PROMPTS.get(fig_type, PROMPTS["curve"])

    client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
    resp = client.chat.completions.create(
        model=VL_MODEL_NAME,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}],
    )

    text = resp.choices[0].message.content.strip()
    # 尝试解析JSON
    try:
        # 去掉markdown代码块
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except:
        return {"raw_description": text}


def save_figure_to_db(image_path: str, fig_id: str, specimen: str,
                      load_step: str, fig_type: str):
    """提取特征并存入figure_db"""
    features = extract_figure_features(image_path, fig_type)

    # 文本内容 = 特征JSON
    text = json.dumps(features, ensure_ascii=False, indent=2)

    # 用全局retriever的向量库（避免重复加载模型）
    from vector_store import VectorStore
    vs = VectorStore()
    vs.add_figure(text, {
        "figure_id": fig_id,
        "specimen": specimen,
        "load_step": load_step,
        "type": fig_type,
    })

    print(f"  [figure_db] {fig_id} ({specimen}/{load_step}/{fig_type}) 已入库")
    return features


def search_figures(query: str, top_k: int = 5) -> list:
    """检索figure_db"""
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path="data/chroma_db")
    try:
        col = client.get_collection("figure_db")
    except:
        return []

    model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    q_emb = model.encode([query]).tolist()
    hits = col.query(query_embeddings=q_emb, n_results=top_k)

    results = []
    for doc, meta in zip(hits['documents'][0], hits['metadatas'][0]):
        results.append({**meta, "description": doc[:200]})
    return results
