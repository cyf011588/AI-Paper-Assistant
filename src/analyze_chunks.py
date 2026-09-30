from pathlib import Path
import json
import re


# =========================
# 1. 项目根目录  用来分析文件结构，以方便找取合适的chunk方法
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

json_path = BASE_DIR / "data" / "processed" / "pdf1.json"


# =========================
# 2. 读取 JSON
# =========================

with open(json_path, "r", encoding="utf-8") as f:
    paper = json.load(f)


# =========================
# 3. 基本信息
# =========================

print("数据类型：", type(paper))
print("总页数：", len(paper))


# =========================
# 4. 分析每一页
# =========================

for page in paper:

    page_number = page["page"]
    text = page["text"]

    lines = text.split("\n")

    print("\n" + "=" * 70)
    print(f"PAGE {page_number}")
    print("=" * 70)

    print("字符数：", len(text))
    print("行数：", len(lines))

    # =========================
    # 5. 尝试识别标题
    # =========================

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # 三级标题：2.4.1
        if re.match(r"^\d+\.\d+\.\d+\s+.+$", line):
            print("[SUB-SUBSECTION]", line)

        # 二级标题：2.1
        elif re.match(r"^\d+\.\d+\s+.+$", line):
            print("[SUBSECTION]    ", line)

        # 一级标题：2.
        elif re.match(r"^\d+\.\s+.+$", line):
            print("[SECTION]       ", line)