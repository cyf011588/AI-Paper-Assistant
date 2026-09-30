"""调试：看看第一页的文本块"""
import pymupdf
from pathlib import Path

pdf = pymupdf.open("F:/AI_Project/AI-Paper-Assistant/data/papers/pdf1.pdf")
page = pdf[0]
print(f"页面大小: {page.rect.width:.0f} x {page.rect.height:.0f}")
print(f"\n所有文本块（按y排序）:")

blocks = page.get_text("dict")["blocks"]
all_blocks = []
for block in blocks:
    if "lines" not in block:
        continue
    text = ""
    for line in block["lines"]:
        for span in line["spans"]:
            text += span["text"]
    text = text.strip()
    if not text:
        continue
    size = block["lines"][0]["spans"][0]["size"]
    y = block["bbox"][1]
    x = block["bbox"][0]
    all_blocks.append((y, size, x, text[:80]))

all_blocks.sort()
for y, size, x, text in all_blocks:
    print(f"  y={y:.0f} size={size:.1f} x={x:.0f} | {text}")

pdf.close()
