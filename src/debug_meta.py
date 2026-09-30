"""调试元数据提取"""
import pymupdf
import re

pdf = pymupdf.open("F:/AI_Project/AI-Paper-Assistant/data/papers/pdf1.pdf")
page = pdf[0]
blocks = page.get_text("dict")["blocks"]

candidates = []
for block in blocks:
    if "lines" not in block:
        continue
    text = ""
    for line in block["lines"]:
        for span in line["spans"]:
            text += span["text"]
    text = text.strip()
    if not text or len(text) < 2:
        continue
    size = block["lines"][0]["spans"][0]["size"]
    y = block["bbox"][1]
    candidates.append({"text": text, "size": size, "y": y})

print(f"总candidates数: {len(candidates)}")
print(f"\n所有candidates (y>120, 按size降序):")
content = [c for c in candidates if c["y"] > 120]
content.sort(key=lambda c: c["size"], reverse=True)
for c in content[:10]:
    print(f"  y={c['y']:.0f} size={c['size']:.1f} len={len(c['text'])} | {c['text'][:60]}")

title_cands = [c for c in content if len(c["text"]) > 20]
print(f"\n长度>20的candidates数: {len(title_cands)}")
if title_cands:
    print(f"标题候选: {title_cands[0]['text'][:80]}")

pdf.close()
