"""
pdf_parser.py — 结构感知 PDF 解析器（自适应版）

核心改进：
- 不硬编码字号/位置阈值，而是先扫描全文自动校准
- 自适应识别正文字号、页眉页脚、标题层级
- 适配不同期刊的排版差异

工作流程：
1. 第一遍扫描：收集所有页面的块信息，统计字号分布，识别正文字号
2. 跨页分析：找出在多页相同位置重复出现的小块文字 = 页眉/页脚
3. 第二遍解析：用校准后的参数做正式分类和章节跟踪
"""

import re
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import pymupdf


# =========================
# 数据结构
# =========================

@dataclass
class TextBlock:
    """一页中的一个文本块"""
    text: str
    block_type: str          # BODY / HEADING / HEADER / FOOTER / FIGURE_CAPTION / TABLE_CAPTION
    page: int
    bbox: tuple              # (x0, y0, x1, y1)
    font: str = ""
    font_size: float = 0
    is_column_left: bool = True


@dataclass
class PaperSection:
    """一个章节"""
    section_path: str
    section_level: int
    blocks: List[TextBlock] = field(default_factory=list)


# =========================
# 主解析器
# =========================

class StructuredPDFParser:
    """
    自适应结构感知 PDF 解析器

    用法：
        parser = StructuredPDFParser("paper.pdf")
        sections = parser.parse()
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        self.doc = pymupdf.open(self.pdf_path)
        self.paper_name = self.pdf_path.stem

        # 校准参数（运行 _calibrate() 后自动填充）
        self.body_font_size: float = 0           # 正文字号
        self.header_footer_ratio: float = 0.75   # 页眉页脚字号 ≤ 正文 × 此比例
        self.heading_size_ratio: float = 1.15    # 标题字号 ≥ 正文 × 此比例
        self.running_positions: dict = {}        # 跨页重复位置 → HEADER/FOOTER

        # 自动提取的元数据
        self.metadata = {"title": "", "authors": [], "year": None}
        self._extract_metadata()

    def _extract_metadata(self):
        """从PDF第一页自动提取论文标题、作者、年份"""
        page = self.doc[0]
        blocks = page.get_text("dict")["blocks"]

        candidates = []
        all_text = ""
        for block in blocks:
            if "lines" not in block:
                continue
            block_text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"]
            block_text = block_text.strip()
            if not block_text or len(block_text) < 2:
                continue
            all_text += block_text + " "
            first_span = block["lines"][0]["spans"][0]
            candidates.append({
                "text": block_text,
                "size": first_span["size"],
                "y": block["bbox"][1],
                "x": block["bbox"][0],
            })

        if not candidates:
            return

        # 排除页眉区域（y < 120的通常是期刊名/页眉）
        content_candidates = [c for c in candidates if c["y"] > 120]
        if not content_candidates:
            content_candidates = candidates

        # 标题 = 内容区域里字号最大的
        content_candidates.sort(key=lambda c: c["size"], reverse=True)
        title_candidates = [c for c in content_candidates if len(c["text"]) > 20]
        if title_candidates:
            self.metadata["title"] = title_candidates[0]["text"][:200]

        # 作者 = 标题下面字号中等的块
        title_y = title_candidates[0]["y"] if title_candidates else 0
        title_size = title_candidates[0]["size"] if title_candidates else 12
        author_candidates = [
            c for c in candidates
            if c["y"] > title_y and c["y"] < title_y + 100
            and c["size"] < title_size and c["size"] >= title_size * 0.6
        ]
        if author_candidates:
            author_text = " ".join(c["text"] for c in author_candidates)
            authors = re.split(r',|;| and |、', author_text)
            self.metadata["authors"] = [a.strip() for a in authors if a.strip() and len(a.strip()) > 1]

        # 年份 = 从全文找4位数字（1900-2099）
        year_match = re.search(r'\b(19|20)\d{2}\b', all_text)
        if year_match:
            self.metadata["year"] = int(year_match.group())

        print(f"[元数据] 标题: {self.metadata['title'][:60]}...")
        print(f"[元数据] 作者: {self.metadata['authors'][:3]}")
        print(f"[元数据] 年份: {self.metadata['year']}")

    def parse(self) -> List[PaperSection]:
        """完整解析流程：校准 → 逐页解析 → 章节组织"""

        # 第一步：校准自适应参数
        self._calibrate()

        # 第二步：逐页解析
        all_sections = []
        current_section: Optional[PaperSection] = None
        section_stack: List[str] = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height

            is_double_column = self._detect_double_column(page)
            raw_blocks = self._extract_raw_blocks(page)
            ordered_blocks = self._order_blocks(raw_blocks, is_double_column, page_width)

            for block in ordered_blocks:
                # 跳过页眉页脚和非正文声明
                if block.block_type in ("HEADER", "FOOTER", "SKIP"):
                    continue

                # 标题 → 更新章节层级
                if block.block_type == "HEADING":
                    section_num, section_title = self._parse_heading(block.text)
                    if section_num:
                        level = self._count_section_level(section_num)
                        while len(section_stack) >= level:
                            section_stack.pop()
                        section_stack.append(f"{section_num} {section_title}".strip())
                        current_section = PaperSection(
                            section_path=" / ".join(section_stack),
                            section_level=level,
                        )
                        all_sections.append(current_section)
                    else:
                        section_stack = [block.text.strip()]
                        current_section = PaperSection(
                            section_path=block.text.strip(),
                            section_level=1,
                        )
                        all_sections.append(current_section)

                # 正文 → 追加到当前章节
                else:
                    if current_section is None:
                        current_section = PaperSection(
                            section_path="Pre-content",
                            section_level=0,
                        )
                        all_sections.append(current_section)
                    current_section.blocks.append(block)

        self.doc.close()
        return all_sections

    # =========================
    # 第一步：自适应校准
    # =========================

    def _calibrate(self):
        """
        扫描全文，自动校准分类参数。

        做三件事：
        1. 统计所有文字块的字号，找出最常见的 = 正文字号
        2. 收集所有块的位置信息，找出跨页重复出现的位置 = 页眉/页脚
        3. 基于正文字号推算标题和页眉页脚的字号阈值
        """
        all_sizes = []
        position_counter = defaultdict(int)
        position_texts = defaultdict(list)

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            page_height = page.rect.height
            page_width = page.rect.width

            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue

                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                block_text = block_text.strip()
                if not block_text:
                    continue

                first_span = block["lines"][0]["spans"][0]
                size = first_span["size"]
                all_sizes.append(size)

                # 记录归一化位置（精度 5%）
                x0, y0, x1, y1 = block["bbox"]
                x_ratio = round(x0 / page_width * 20) / 20
                y_ratio = round(y0 / page_height * 20) / 20
                position_counter[(x_ratio, y_ratio)] += 1
                position_texts[(x_ratio, y_ratio)].append(block_text[:50])

        # 1. 正文字号：统计出现频率最高的字号
        size_counter = Counter(all_sizes)
        self.body_font_size = size_counter.most_common(1)[0][0] if size_counter else 10.0

        # 2. 页眉/页脚：在超过一半页面都出现的位置
        total_pages = len(self.doc)
        for pos, count in position_counter.items():
            if count >= total_pages * 0.5:
                avg_len = sum(len(t) for t in position_texts[pos]) / len(position_texts[pos])
                if avg_len < 100:  # 文字很短 → 页眉/页脚
                    y_ratio = pos[1]
                    if y_ratio < 0.15:
                        self.running_positions[pos] = "HEADER"
                    elif y_ratio > 0.85:
                        self.running_positions[pos] = "FOOTER"

        print(f"[校准] 正文字号: {self.body_font_size:.1f}pt")
        print(f"[校准] 页眉/页脚位置: {len(self.running_positions)} 处")

    # =========================
    # 第二步：逐页解析
    # =========================

    def _detect_double_column(self, page) -> bool:
        """检测双栏排版"""
        page_width = page.rect.width
        mid_x = page_width / 2
        mid_strip = page_width * 0.05

        blocks = page.get_text("dict")["blocks"]
        crossing = 0
        total = 0

        for block in blocks:
            if "lines" not in block:
                continue
            x0, y0, x1, y1 = block["bbox"]
            total += 1
            if x0 < mid_x - mid_strip and x1 > mid_x + mid_strip:
                crossing += 1

        if total == 0:
            return False
        return crossing / total < 0.2

    def _extract_raw_blocks(self, page) -> List[TextBlock]:
        """提取一页的所有文本块并分类"""
        page_height = page.rect.height
        page_width = page.rect.width
        blocks = page.get_text("dict")["blocks"]
        result = []

        for block in blocks:
            if "lines" not in block:
                continue

            block_text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"]
            block_text = block_text.strip()
            if not block_text:
                continue

            x0, y0, x1, y1 = block["bbox"]
            first_span = block["lines"][0]["spans"][0]
            font = first_span["font"]
            size = first_span["size"]

            block_type = self._classify_block(
                block_text, x0, y0, page_width, page_height, size, font,
            )

            result.append(TextBlock(
                text=block_text,
                block_type=block_type,
                page=page.number + 1,
                bbox=block["bbox"],
                font=font,
                font_size=size,
            ))

        return result

    def _classify_block(self, text, x0, y0, page_width, page_height,
                        font_size, font_name) -> str:
        """
        自适应分类块类型。
        优先级：跨页校准位置 > 相对字号判断 > 文字模式匹配
        """
        # 1. 先查校准出的页眉/页脚位置
        # 但如果字号明显比正文大，或者是黑体标题，说明是标题，不是页眉页脚
        x_ratio = round(x0 / page_width * 20) / 20
        y_ratio = round(y0 / page_height * 20) / 20
        pos_key = (x_ratio, y_ratio)
        is_heiti = "SimHei" in font_name or "HZ" in font_name
        if pos_key in self.running_positions and font_size < self.body_font_size * 1.3 and not is_heiti:
            return self.running_positions[pos_key]

        # 2. 字号判断：明显比正文小 + 在页面边缘 → 页眉页脚
        if font_size < self.body_font_size * self.header_footer_ratio:
            if y_ratio < 0.12 or y_ratio > 0.88:
                return "HEADER" if y_ratio < 0.5 else "FOOTER"

        # 3. 图标题
        if re.match(r'^(Fig\.?|Figure)\s*\d+', text):
            return "FIGURE_CAPTION"

        # 4. 表标题
        if re.match(r'^(Table|Tab\.)\s*\d+', text) and len(text) < 300:
            return "TABLE_CAPTION"

        # 4.5 目录页（有连续点号引导到页码）— 必须在HEADING判断之前
        if re.search(r'\.{3,}|\．{3,}', text):
            return "SKIP"

        # 4.8 公式行（含大量数学符号/括号/上下标，不是标题）— 必须在HEADING判断之前
        # 如："3 项：0.745(Ib/Ic)0.524fyctcwhc。"
        has_math = bool(re.search(r'[=≈≤≥±∑∏∫αβγδθελμπσφψωΔΣΦ]', text))
        has_mixed_formula = bool(re.search(r'\d+\.?\d*\s*[\(（][a-zA-Z/]+[\)）]', text))  # 数字(字母/字母)
        has_formula_vars = bool(re.search(r'[a-zA-Z]{2,}\s*[=＝]', text))  # 变量=
        if (has_math or has_mixed_formula or has_formula_vars) and not re.match(r'^第[一二三四五六七八九十\d]+章', text):
            return "BODY"

        # 5. 章节编号格式标题
        # 支持：1 Introduction、2　承插式...、2.1　模型设计、4.6.1 滞回曲线
        # 一级：X 标题；二级：X.X 标题；三级：X.X.X 标题
        if re.match(r'^\d+(\.\d+){0,2}\s*[.．\s　]+\s*[A-Z一-龥]', text) and len(text) < 150:
            # 排除图注/表注：以"图"/"表"开头
            if not re.match(r'^(图|表|Fig\.?|Table?)\s*\d', text):
                # 排除数字后跟着动词的（如"14 为TS10..."不是标题）
                # 标题一般是名词短语，不会有"为/是/的/了"等动词
                after_num = re.sub(r'^\d+(\.\d+){0,2}\s*[.．\s　]+\s*', '', text).strip()
                if not re.match(r'^(为|是|的|了|在|有|和|与|及|或)', after_num):
                    return "HEADING"

        # 5.2 中文学位论文"第X章"格式
        # 如：第一章 引言、第二章 试验研究
        if re.match(r'^第[一二三四五六七八九十\d]+章\s+[一-龥]', text) and len(text) < 80:
            return "HEADING"

        # 5.5 中文黑体标题
        # SimHei（黑体）或 E-HZ（方正黑体）且字号比正文大
        if ("SimHei" in font_name or "HZ" in font_name) and font_size >= self.body_font_size * 1.15 and len(text) < 80:
            return "HEADING"

        # 5.6 目录、致谢、作者简介等不需要入库的部分（必须在HEADING判断之前）
        if re.match(r'^(目\s*录|图表目录|致\s*谢|作者简介|附\s*录|攻读.*期间)', text.strip()) and len(text) < 20:
            return "SKIP"

        # 6. 无编号标题词（英文 + 中文）
        first_word = text.strip().split()[0] if text.strip() else ""
        known_en = ["Abstract", "ABSTRACT", "Keywords", "KEYWORDS",
                    "References", "REFERENCES", "Introduction",
                    "Conclusion", "CONCLUSION", "Conclusions", "CONCLUSIONS",
                    "Discussion", "Methodology", "Results", "Appendix",
                    "Declaration", "ACKNOWLEDGEMENTS", "Acknowledgements"]
        known_zh = ["摘要", "关键词", "参考文献", "引言", "结论",
                    "试验", "实验", "结果", "分析", "方法", "综述"]
        if first_word in known_en or first_word in known_zh:
            if len(text) < 150:
                return "HEADING"

        # 7. 字号明显大 + 粗体 + 短 → 也可能是标题
        is_larger = font_size >= self.body_font_size * self.heading_size_ratio
        is_bold = "Bold" in font_name or "bold" in font_name.lower()
        if is_larger and is_bold and len(text) < 150:
            return "HEADING"

        # 8. 非正文声明性文字
        if re.match(r'^(Declaration of|Data Availability|Acknowledg|Conflict of)', text, re.I):
            return "SKIP"

        # 8.5 目录、致谢、作者简介等不需要入库的部分
        if re.match(r'^(目\s*录|图表目录|致\s*谢|作者简介|附\s*录|攻读.*期间)', text.strip()) and len(text) < 20:
            return "SKIP"

        # 8.6 公式行（含大量数学符号，不是标题）
        if re.search(r'[=≈≤≥±∑∏∫αβγδθελμπσφψωΔΣΦ]', text) and not re.match(r'^第[一二三四五六七八九十\d]+章', text):
            return "BODY"

        return "BODY"

    def _order_blocks(self, blocks: List[TextBlock],
                      is_double_column: bool,
                      page_width: float) -> List[TextBlock]:
        """按阅读顺序排序：双栏先左后右，单栏从上到下"""
        if not is_double_column:
            return sorted(blocks, key=lambda b: b.bbox[1])

        mid_x = page_width / 2
        left, right = [], []
        for block in blocks:
            center_x = (block.bbox[0] + block.bbox[2]) / 2
            if center_x < mid_x:
                block.is_column_left = True
                left.append(block)
            else:
                block.is_column_left = False
                right.append(block)

        left.sort(key=lambda b: b.bbox[1])
        right.sort(key=lambda b: b.bbox[1])
        return left + right

    def _parse_heading(self, text: str):
        text = text.strip()
        match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+)", text)
        if match:
            return match.group(1), match.group(2)
        return None, text

    def _count_section_level(self, section_num: str) -> int:
        return section_num.count(".") + 1

    def save(self, sections: List[PaperSection], output_path: Path):
        """保存为 JSON"""
        import json
        data = {
            "paper_name": self.paper_name,
            "total_sections": len(sections),
            "calibration": {
                "body_font_size": self.body_font_size,
            },
            "sections": [],
        }
        for sec in sections:
            sec_data = {
                "section_path": sec.section_path,
                "section_level": sec.section_level,
                "block_count": len(sec.blocks),
                "blocks": [],
            }
            for b in sec.blocks:
                sec_data["blocks"].append({
                    "text": b.text,
                    "block_type": b.block_type,
                    "page": b.page,
                    "bbox": list(b.bbox),
                    "font": b.font,
                    "font_size": b.font_size,
                })
            data["sections"].append(sec_data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# 命令行测试
# =========================

if __name__ == "__main__":
    from config import PAPERS_DIR, PROCESSED_DIR

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("没有找到 PDF 文件")
        exit()

    pdf_path = pdf_files[0]
    print(f"解析论文: {pdf_path.name}\n")

    parser = StructuredPDFParser(pdf_path)
    sections = parser.parse()

    print(f"\n解析完成，共 {len(sections)} 个章节:\n")
    for i, sec in enumerate(sections[:20]):
        total_chars = sum(len(b.text) for b in sec.blocks)
        print(f"  [{i+1}] {sec.section_path}")
        print(f"      L{sec.section_level} | {len(sec.blocks)}块 | {total_chars}字")

    output = PROCESSED_DIR / f"{pdf_path.stem}_structured.json"
    parser.save(sections, output)
    print(f"\n结果已保存: {output}")

