from pathlib import Path
import pymupdf


# =========================
# 1. 项目根目录
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

pdf_path = BASE_DIR / "data" / "papers" / "pdf1.pdf"


# =========================
# 2. 打开 PDF
# =========================

doc = pymupdf.open(pdf_path)


# =========================
# 3. 查看第 2 页
# =========================

page = doc[1]

print("页面尺寸：")
print("宽度 =", page.rect.width)
print("高度 =", page.rect.height)


# =========================
# 4. 获取页面结构
# =========================

blocks = page.get_text("dict")["blocks"]


for i, block in enumerate(blocks):

    if "lines" not in block:
        continue

    print("\n" + "=" * 70)
    print("BLOCK", i)
    print("=" * 70)

    # block 的位置
    print("bbox =", block["bbox"])

    # block 中的文字
    block_text = ""

    for line in block["lines"]:

        line_text = ""

        for span in line["spans"]:
            line_text += span["text"]

        block_text += line_text

    print("TEXT =", repr(block_text))

    # 只打印第一个 span 的字体信息
    first_span = block["lines"][0]["spans"][0]

    print("size  =", first_span["size"])
    print("font  =", first_span["font"])
    print("flags =", first_span["flags"])