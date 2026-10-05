"""
multimodal_parser.py — 多模态文件解析器
支持：CSV/Excel → Markdown表格；图片 → Qwen-VL描述
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def parse_excel(file_path: str) -> str:
    """Excel/CSV → Markdown表格"""
    import pandas as pd
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df.to_markdown(index=False)


def parse_image(file_path: str) -> str:
    """图片 → Qwen-VL生成描述"""
    import base64
    from openai import OpenAI
    from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL

    ext = os.path.splitext(file_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
    from config import VL_MODEL_NAME

    resp = client.chat.completions.create(
        model=VL_MODEL_NAME,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": "请详细描述该图表，并提取其中的关键数据点（如峰值、坐标、趋势）。"}
        ]}],
    )
    return resp.choices[0].message.content


def parse_file(file_path: str) -> tuple[str, str]:
    """解析文件，返回 (文本内容, 文件类型)"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return parse_excel(file_path), "excel"
    elif ext == ".csv":
        return parse_excel(file_path), "csv"
    elif ext in (".png", ".jpg", ".jpeg"):
        return parse_image(file_path), "image"
    else:
        raise ValueError(f"不支持的文件类型: {ext}")


def save_to_my_results(text: str, source_name: str, file_type: str):
    """存入my_results集合"""
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path="data/chroma_db")
    try:
        col = client.get_collection("my_results")
    except:
        col = client.create_collection("my_results")

    model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    emb = model.encode([text]).tolist()

    col.add(
        documents=[text],
        ids=[f"my_{source_name}"],
        embeddings=emb,
        metadatas=[{"type": "my_data", "source": "user_upload", "file_type": file_type}]
    )
    print(f"  [my_results] 已存入: {source_name} ({file_type})")


def search_my_results(query: str, top_k: int = 5) -> list:
    """从my_results检索"""
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path="data/chroma_db")
    try:
        col = client.get_collection("my_results")
    except:
        return []

    model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    q_emb = model.encode([query]).tolist()
    hits = col.query(query_embeddings=q_emb, n_results=top_k)
    return hits['documents'][0] if hits['documents'] else []
