"""
图表多模态解析：提取PDF中的图片 → 调Qwen-VL生成描述 → 入库
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, 'src')

import base64, io, json, re
from pathlib import Path
import fitz  # PyMuPDF
from openai import OpenAI
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, DB_DIR, EMBEDDING_MODEL
from vector_store import VectorStore
from chunker import Chunk

# 要处理的论文
PAPERS = [
    ("雷杰鑫毕业论文终稿", "data/papers/雷杰鑫毕业论文终稿.pdf"),
    ("王鑫涛论文终稿", "data/papers/王鑫涛论文终稿.pdf"),
]

def extract_images_from_pdf(pdf_path: str):
    """提取PDF每页的图片，返回 [(page_num, image_bytes, caption_guess)]"""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # 获取页面文本，找图注（图X-X ...）
        text = page.get_text()
        captions = re.findall(r'(图\s*\d+[-–.]\d+[^\n]{0,100})', text)
        
        # 提取图片
        img_list = page.get_images(full=True)
        for img_idx, img in enumerate(img_list):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:  # CMYK
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                
                # 猜测图注：如果这页有图注，用第一个匹配
                caption = captions[0] if captions else f"第{page_num+1}页图{img_idx+1}"
                
                images.append({
                    "page": page_num + 1,
                    "image_bytes": img_bytes,
                    "caption_guess": caption,
                })
                pix = None
            except Exception as e:
                print(f"  跳过图片 p.{page_num+1} img{img_idx}: {e}")
    doc.close()
    return images

def describe_image(client, img_bytes: bytes, caption_guess: str) -> str:
    """调Qwen-VL生成图片描述"""
    b64 = base64.b64encode(img_bytes).decode()
    try:
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}
                        },
                        {
                            "type": "text",
                            "text": f"这是论文中的图片（可能是图：{caption_guess}）。请用中文详细描述：1)这是曲线图/示意图/照片还是表格？2)坐标轴分别是什么？3)展示了什么趋势或现象？4)关键数据点。控制在150字以内。"
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[图片描述失败: {e}]"

# 主流程
client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
vs = VectorStore()

for paper_name, pdf_path in PAPERS:
    if not Path(pdf_path).exists():
        print(f"PDF不存在: {pdf_path}")
        continue
    
    print(f"\n处理: {paper_name}")
    images = extract_images_from_pdf(pdf_path)
    print(f"  提取到 {len(images)} 张图片")
    
    # 去重：同一页的小图标（logo、装饰）跳过
    seen = set()
    unique_images = []
    for img in images:
        key = (img["page"], img["caption_guess"][:20])
        if key not in seen:
            seen.add(key)
            unique_images.append(img)
    print(f"  去重后 {len(unique_images)} 张")
    
    # 生成描述
    new_chunks = []
    for i, img in enumerate(unique_images):
        print(f"  [{i+1}/{len(unique_images)}] p.{img['page']} {img['caption_guess'][:30]}...")
        desc = describe_image(client, img["image_bytes"], img["caption_guess"])
        
        chunk = Chunk(
            id=f"{paper_name}_图表_p{img['page']}_{i:03d}",
            text=f"【图表】{img['caption_guess']}\n描述：{desc}",
            paper_name=paper_name,
            section_path=f"图表 / {img['caption_guess'][:30]}",
            page_start=img["page"],
            page_end=img["page"],
            parent_id="",
            parent_text="",
            block_types=["FIGURE"],
        )
        new_chunks.append(chunk)
    
    # 入库
    if new_chunks:
        vs.add_chunks(new_chunks)
        print(f"  已入库 {len(new_chunks)} 个图表chunk")

print("\n完成！")
