"""
smart_retriever.py — 智能检索器（优化版）

组合查询理解 + 语义检索：
1. 用户提问 → 查询理解模块识别是否指定了论文
2. 预建全局索引，查询时先全局检索再过滤（不重复建索引）
3. 返回检索结果 + 查询理解信息
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from pathlib import Path
from typing import List, Dict

from config import PROCESSED_DIR, TOP_K, METADATA_PATH
from chunker import Chunk
from semantic_retriever import SemanticRetriever
from query_understanding import QueryUnderstanding


class SmartRetriever:
    """智能检索器：自动识别论文引用并过滤 + 上下文扩展"""

    def __init__(self, neighbor_window: int = 1):
        """
        neighbor_window: 检索到一个chunk后，前后各扩展几个相邻chunk
        比如window=1，检索到chunk_5就把chunk_4、5、6一起返回
        """
        self.qu = QueryUnderstanding()
        self.semantic = SemanticRetriever()
        self.all_chunks: List[Chunk] = []
        self.paper_metadata: List[Dict] = []
        self.is_trained = False
        self.neighbor_window = neighbor_window
        # paper_name -> 按顺序排列的chunks列表
        self.paper_chunks_map: Dict[str, List[Chunk]] = {}
        self._load_metadata()

    def _load_metadata(self):
        """加载论文元数据"""
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.paper_metadata = json.load(f)

    def get_paper_info(self, paper_name: str) -> Dict:
        """根据paper_name获取论文完整信息（用于引用格式化）"""
        for p in self.paper_metadata:
            if p["paper_name"] == paper_name:
                return p
        return {"paper_name": paper_name, "title": paper_name, "authors": [], "year": None, "journal": ""}

    def format_citation(self, paper_name: str) -> str:
        """格式化为：年份 作者. 文章名. 期刊"""
        info = self.get_paper_info(paper_name)
        year = info.get("year") or "n.d."
        authors = info.get("authors", [])
        first_author = authors[0] if authors else "Unknown"
        if len(authors) > 1:
            author_str = f"{first_author} et al."
        else:
            author_str = first_author
        title = info.get("title", paper_name)
        journal = info.get("journal", "")
        if journal:
            return f"{year}, {author_str}. {title}. {journal}"
        return f"{year}, {author_str}. {title}"

    def build_index(self, chunks: List[Chunk]):
        """建立全局语义索引 + 按论文分组"""
        self.all_chunks = chunks
        self.semantic.build_index(chunks)

        # 按论文分组，建立有序chunk列表
        self.paper_chunks_map = {}
        for c in chunks:
            if c.paper_name not in self.paper_chunks_map:
                self.paper_chunks_map[c.paper_name] = []
            self.paper_chunks_map[c.paper_name].append(c)

        self.is_trained = True

    def _expand_neighbors(self, results: List[Dict]) -> List[Dict]:
        """上下文扩展：
        1. 章节级聚合：检索到某章节的chunk → 该章节所有chunks都加进来
        2. 相邻扩展：前后各扩展 neighbor_window 个chunk
        """
        expanded = {}  # key: (paper_name, chunk_id), value: result dict

        # 收集检索到的所有章节
        hit_sections = set()  # (paper_name, section_path)
        for r in results:
            key = (r["paper_name"], r["id"])
            if key not in expanded:
                expanded[key] = r
            hit_sections.add((r["paper_name"], r["section_path"]))

        # 章节级聚合：把命中章节的所有chunks都加进来
        for paper_name, section_path in hit_sections:
            for c in self.paper_chunks_map.get(paper_name, []):
                if c.section_path == section_path:
                    nkey = (c.paper_name, c.id)
                    if nkey not in expanded:
                        # 章节内其他chunk给一个中等分数
                        expanded[nkey] = {
                            "id": c.id,
                            "text": c.text,
                            "paper_name": c.paper_name,
                            "section_path": c.section_path,
                            "page_start": c.page_start,
                            "page_end": c.page_end,
                            "score": 0.5,  # 章节内chunk默认分
                        }

        # 相邻扩展：前后各 neighbor_window 个
        for r in list(expanded.values()):
            paper_chunks = self.paper_chunks_map.get(r["paper_name"], [])
            chunk_idx = None
            for i, c in enumerate(paper_chunks):
                if c.id == r["id"]:
                    chunk_idx = i
                    break
            if chunk_idx is None:
                continue

            for offset in range(-self.neighbor_window, self.neighbor_window + 1):
                if offset == 0:
                    continue
                neighbor_idx = chunk_idx + offset
                if 0 <= neighbor_idx < len(paper_chunks):
                    neighbor = paper_chunks[neighbor_idx]
                    nkey = (neighbor.paper_name, neighbor.id)
                    if nkey not in expanded:
                        expanded[nkey] = {
                            "id": neighbor.id,
                            "text": neighbor.text,
                            "paper_name": neighbor.paper_name,
                            "section_path": neighbor.section_path,
                            "page_start": neighbor.page_start,
                            "page_end": neighbor.page_end,
                            "score": r["score"] * 0.8,
                        }

        # 按score排序
        return sorted(expanded.values(), key=lambda x: x["score"], reverse=True)

    def _is_chinese(self, text: str) -> bool:
        """简单判断文本是否主要是中文"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return chinese_chars > len(text) * 0.3

    def _translate_to_english(self, query: str) -> str:
        """用DeepSeek把中文query翻译成英文，提高跨语言检索效果"""
        from openai import OpenAI
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": """You are a translator specialized in structural engineering academic papers. 
Translate the user's question into English using precise academic terminology.
Context: papers are about steel structures, shear walls, composite plates, seismic engineering.
Only output the English translation, nothing else."""},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    def search(self, question: str, top_k: int = TOP_K) -> Dict:
        """
        智能检索：
        - 先做查询理解
        - 中文query自动翻译成英文再检索
        - 全局检索，过滤指定论文
        - 上下文扩展（章节级聚合 + 相邻扩展）
        """
        if not self.is_trained:
            return {"results": [], "query_info": None}

        # 1. 查询理解
        query_info = self.qu.parse(question)
        content_query = query_info["content_query"]
        target_papers = query_info["target_papers"]

        # 2. 如果是中文query，判断目标论文是中文还是英文内容：
        #    - 中文论文（chunks是中文）→ 不翻译，直接用中文检索
        #    - 英文论文（chunks是英文）→ 翻译成英文检索
        search_query = content_query
        if self._is_chinese(content_query):
            target_is_chinese = False
            if target_papers:
                # 看目标论文的chunks内容是中文还是英文
                for p in target_papers:
                    paper_chunks = self.paper_chunks_map.get(p, [])
                    if paper_chunks:
                        # 看前3个chunk的文字
                        sample_text = " ".join(c.text for c in paper_chunks[:3])
                        chinese_chars = sum(1 for c in sample_text if '\u4e00' <= c <= '\u9fff')
                        if chinese_chars > len(sample_text) * 0.2:
                            target_is_chinese = True
                            break

            if not target_is_chinese:
                try:
                    search_query = self._translate_to_english(content_query)
                    print(f"  [查询翻译] 中文: {content_query[:40]} → 英文: {search_query[:60]}")
                except Exception as e:
                    print(f"  [查询翻译] 失败，用原文检索: {e}")
            else:
                print(f"  [查询翻译] 目标论文是中文内容，不翻译")

        # 3. 全局检索多一些结果
        search_k = max(top_k * 3, 20) if target_papers else top_k
        all_results = self.semantic.search(search_query, top_k=search_k)

        # 3. 如果指定了论文，过滤结果
        if target_papers:
            filtered = [
                r for r in all_results
                if r["paper_name"] in target_papers
            ]
        else:
            filtered = all_results

        # 4. 上下文扩展（章节级聚合 + 相邻扩展）
        expanded = self._expand_neighbors(filtered)

        # 5. 如果指定了论文，把命中章节的所有chunks完整返回（不截断）
        #    全局检索时才截断到top_k
        if target_papers:
            results = expanded
        else:
            results = expanded[:top_k]

        return {
            "results": results,
            "query_info": query_info,
        }


# =========================
# 测试
# =========================

if __name__ == "__main__":
    # 加载所有chunks
    all_chunks = []
    chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            for c in json.load(f):
                all_chunks.append(Chunk(**c))

    print(f"加载了 {len(all_chunks)} 个 chunks\n")

    retriever = SmartRetriever()
    retriever.build_index(all_chunks)

    test_questions = [
        "雷杰鑫小论文里主要结论有几条，是什么？",
        "李慧论文中，试验试件的尺寸，以及加载方式是什么？",
        "戎科锦论文提出的具体螺栓需求公式是什么？",
        "双钢板组合剪力墙的抗震性能怎么样？",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        output = retriever.search(q, top_k=3)
        qi = output["query_info"]
        print(f"  识别论文: {qi['target_papers']}")
        print(f"  内容问题: {qi['content_query']}")
        print(f"  检索结果:")
        for i, r in enumerate(output["results"]):
            cite = retriever.format_citation(r["paper_name"])
            print(f"    [{i+1}] {cite[:60]}... p.{r['page_start']} score={r['score']:.3f}")
