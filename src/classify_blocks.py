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
# 3. 分析第 2 页
# =========================

page = doc[1]

page_width = page.rect.width
page_height = page.rect.height

print("页面宽度：", page_width)
print("页面高度：", page_height)


# =========================
# 4. 获取 Blocks
# =========================

blocks = page.get_text("dict")["blocks"]


# =========================
# 5. 分类
# =========================

for i, block in enumerate(blocks):

    if "lines" not in block:
        continue

    # -------------------------
    # Block 位置
    # -------------------------

    x0, y0, x1, y1 = block["bbox"]

    block_width = x1 - x0
    block_height = y1 - y0


    # -------------------------
    # Block 文字
    # -------------------------

    block_text = ""

    for line in block["lines"]:
        for span in line["spans"]:
            block_text += span["text"]

    block_text = block_text.strip()

    if not block_text:
        continue


    # -------------------------
    # 字体信息
    # -------------------------

    first_span = block["lines"][0]["spans"][0]

    font = first_span["font"]
    size = first_span["size"]
    flags = first_span["flags"]


    # =========================
    # 默认：BODY
    # =========================

    block_type = "BODY"


    # =========================
    # ① Header
    # =========================

    if (
        y0 < page_height * 0.08
        and block_height < 25
        and size < 7
    ):
        block_type = "HEADER"


    # =========================
    # ② Footer
    # =========================

    elif (
        y1 > page_height * 0.92
        and block_height < 25
        and size < 7
    ):
        block_type = "FOOTER"


    # =========================
    # ③ Figure caption
    # =========================

    elif (
        block_text.startswith("Fig.")
        or block_text.startswith("Figure")
    ):
        block_type = "FIGURE_CAPTION"


    # =========================
    # ④ Table caption
    # =========================

    elif (
        block_text.startswith("Table")
    ):
        block_type = "TABLE_CAPTION"


    # =========================
    # ⑤ Heading
    # =========================

    elif (
        len(block_text) < 100
        and block_text[0].isdigit()
        and (
            font.endswith("-Bold")
            or font.endswith("-Italic")
        )
    ):
        block_type = "HEADING"


    # =========================
    # 输出
    # =========================

    print("\n" + "=" * 70)

    print("BLOCK:", i)
    print("TYPE :", block_type)
    print("TEXT :", block_text)
    print("FONT :", font)
    print("SIZE :", size)
    print("BBOX :", block["bbox"])
    #已经能够识别出标题和正文内容了