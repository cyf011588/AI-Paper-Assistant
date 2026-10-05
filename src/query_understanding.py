"""
query_understanding.py — 查询理解模块

功能：
1. 识别用户问题中提到的论文名/作者名
2. 映射到知识库中的具体论文
3. 提取真正的内容问题（去掉论文/作者引用）

示例：
  输入："雷杰鑫小论文里主要结论有几条？"
  输出：{
    "target_papers": ["雷杰鑫小论文"],
    "content_query": "主要结论有几条？",
    "is_paper_specific": True
  }
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional


METADATA_FILE = Path("F:/AI_Project/AI-Paper-Assistant/data/paper_metadata.json")


class QueryUnderstanding:
    """查询理解：识别论文/作者引用，提取内容问题"""

    def __init__(self):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # 构建别名→论文名列表的映射（一个别名可能对应多篇论文）
        self.alias_map = {}
        self.paper_type_map = {}  # paper_name -> "thesis" 或 "journal"
        for paper in self.metadata:
            paper_name = paper["paper_name"]
            # 判断论文类型
            self.paper_type_map[paper_name] = paper.get("thesis_type", "journal")
            # 论文名本身也是别名
            key = paper_name.lower()
            if key not in self.alias_map:
                self.alias_map[key] = []
            if paper_name not in self.alias_map[key]:
                self.alias_map[key].append(paper_name)
            # 作者别名
            for alias in paper.get("author_aliases", []):
                key = alias.lower()
                if key not in self.alias_map:
                    self.alias_map[key] = []
                if paper_name not in self.alias_map[key]:
                    self.alias_map[key].append(paper_name)
            # 作者英文名
            for author in paper.get("authors", []):
                key = author.lower()
                if key not in self.alias_map:
                    self.alias_map[key] = []
                if paper_name not in self.alias_map[key]:
                    self.alias_map[key].append(paper_name)

    def parse(self, question: str) -> Dict:
        """
        解析用户问题，返回：
        - target_papers: 识别到的目标论文列表
        - content_query: 去掉论文引用后的内容问题
        - is_paper_specific: 是否指定了具体论文
        """
        question_lower = question.lower()
        target_papers = []
        matched_aliases = []

        # 判断用户指的是哪类论文
        wants_thesis = any(kw in question for kw in ["毕业论文", "大论文", "学位论文", "毕业"])
        wants_journal = any(kw in question for kw in ["小论文", "期刊论文", "期刊"])

        # 逐个匹配别名
        for alias, paper_names in self.alias_map.items():
            if len(alias) < 2:
                continue
            if alias in question_lower:
                for pn in paper_names:
                    # 过滤：根据metadata里的thesis_type判断
                    p_type = self.paper_type_map.get(pn, "journal")
                    if wants_thesis and p_type != "thesis":
                        continue
                    if wants_journal and p_type != "journal":
                        continue
                    if pn not in target_papers:
                        target_papers.append(pn)
                matched_aliases.append(alias)

        # 提取内容问题：去掉匹配到的别名和相关词汇
        content_query = question
        for alias in matched_aliases:
            # 去掉别名本身
            content_query = re.sub(re.escape(alias), '', content_query, flags=re.IGNORECASE)

        # 清理常见的引用模式
        patterns_to_remove = [
            r'(这篇|那篇|该)?(论文|文章|小论文|文献)',
            r'(里|中|里面|当中)',
            r'(的|关于)',
            r'(研究|写的|写了)',
        ]
        for pattern in patterns_to_remove:
            content_query = re.sub(pattern, '', content_query)

        # 清理多余空格和标点
        content_query = re.sub(r'\s+', ' ', content_query).strip()
        content_query = re.sub(r'^[，。、,\.]+', '', content_query).strip()

        # 如果清理后内容太少，用原问题
        if len(content_query) < 2:
            content_query = question

        return {
            "original_question": question,
            "target_papers": target_papers,
            "content_query": content_query,
            "is_paper_specific": len(target_papers) > 0,
        }


# =========================
# 测试
# =========================

if __name__ == "__main__":
    qu = QueryUnderstanding()

    test_questions = [
        "雷杰鑫小论文里主要结论有几条，是什么？",
        "雷杰鑫这篇文章有限元模型是怎么建模的？",
        "康童阳的所有文章分别研究的是什么内容？",
        "李慧论文中，试验试件的尺寸，以及加载方式是什么？",
        "王鑫涛小论文摘要？",
        "戎科锦论文提出的具体螺栓需求公式是什么？",
        "双钢板组合剪力墙的抗震性能怎么样？",  # 没有指定论文
        "SF-CPSW的耗能机制是什么？",  # 没有指定论文
    ]

    print("查询理解测试\n")
    for q in test_questions:
        result = qu.parse(q)
        print(f"问题: {q}")
        print(f"  目标论文: {result['target_papers']}")
        print(f"  内容问题: {result['content_query']}")
        print(f"  指定论文: {result['is_paper_specific']}")
        print()
