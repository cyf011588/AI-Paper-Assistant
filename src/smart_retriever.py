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
import re
from pathlib import Path
from typing import List, Dict
from rank_bm25 import BM25Okapi
import jieba

from config import PROCESSED_DIR, TOP_K, METADATA_PATH, RECALL_K
from chunker import Chunk
from vector_store import VectorStore
from query_understanding import QueryUnderstanding
from reranker import Reranker


class SmartRetriever:
    """智能检索器：自动识别论文引用并过滤 + 上下文扩展"""

    def __init__(self, neighbor_window: int = 1):
        """
        neighbor_window: 检索到一个chunk后，前后各扩展几个相邻chunk
        比如window=1，检索到chunk_5就把chunk_4、5、6一起返回
        """
        self.qu = QueryUnderstanding()
        self.vector_db = VectorStore()
        self.reranker = Reranker()
        self.bm25 = None
        self.bm25_corpus = []  # 分词后的corpus
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
        """建立索引：ChromaDB已持久化，这里加载paper_chunks_map + 建BM25索引"""
        self.all_chunks = chunks

        # 按论文分组
        self.paper_chunks_map = {}
        for c in chunks:
            if c.paper_name not in self.paper_chunks_map:
                self.paper_chunks_map[c.paper_name] = []
            self.paper_chunks_map[c.paper_name].append(c)

        # 建BM25索引（jieba分词）
        print("正在建BM25关键词索引...")
        self.bm25_corpus = []
        for c in chunks:
            tokens = list(jieba.cut(f"{c.section_path} {c.text}"))
            self.bm25_corpus.append(tokens)
        self.bm25 = BM25Okapi(self.bm25_corpus)
        print(f"BM25索引完成，共 {len(chunks)} 个chunk")

        self.is_trained = True
        print(f"索引就绪，向量库共 {self.vector_db.count()} 个chunk，覆盖 {len(self.paper_chunks_map)} 篇论文")

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

        # 章节级聚合：把命中章节及其子章节的所有chunks都加进来
        for paper_name, section_path in hit_sections:
            for c in self.paper_chunks_map.get(paper_name, []):
                # 前缀匹配：命中"4.3 xxx"，则"4.3 xxx / 4.3.1"也加进来
                if c.section_path == section_path or c.section_path.startswith(section_path + " /"):
                    nkey = (c.paper_name, c.id)
                    if nkey not in expanded:
                        expanded[nkey] = {
                            "id": c.id,
                            "text": c.text,
                            "paper_name": c.paper_name,
                            "section_path": c.section_path,
                            "page_start": c.page_start,
                            "page_end": c.page_end,
                            "score": 0.5,
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

    def _search_single_paper(self, query: str, paper_name: str, top_k: int = 10) -> list:
        """在指定论文内做向量+BM25检索（不走章节匹配/reranker）"""
        # 去掉query里的论文名和作者名，只保留核心问题
        core_query = query
        for pn in self.paper_chunks_map.keys():
            core_query = core_query.replace(pn, "")
        # 去掉"两篇"、"分别"等对比词汇
        for kw in ["两篇", "两篇论文", "分别", "共同点", "不同点", "和", "与", "跟", "对比", "比较", "区别"]:
            core_query = core_query.replace(kw, "")
        core_query = re.sub(r'\s+', ' ', core_query).strip()
        if len(core_query) < 3:
            core_query = query  # 清理后太短就用原query

        # 向量检索，filter到这篇论文
        vector_results = self.vector_db.search(core_query, top_k=RECALL_K, where={"paper_name": {"$eq": paper_name}})
        # BM25检索，filter到这篇论文
        bm25_tokens = list(jieba.cut(core_query))
        bm25_scores = self.bm25.get_scores(bm25_tokens)
        bm25_results = []
        for idx in bm25_scores.argsort()[-RECALL_K:][::-1]:
            c = self.all_chunks[idx]
            if c.paper_name == paper_name:
                bm25_results.append({
                    "id": c.id, "text": c.text, "paper_name": c.paper_name,
                    "section_path": c.section_path, "page_start": c.page_start,
                    "page_end": c.page_end, "parent_id": c.parent_id,
                    "parent_text": c.parent_text, "score": float(bm25_scores[idx]),
                })
        # RRF融合
        k = 60
        rrf = {}
        for rank, r in enumerate(vector_results):
            rrf[r["id"]] = rrf.get(r["id"], 0) + 1.0/(k+rank+1)
        for rank, r in enumerate(bm25_results):
            rrf[r["id"]] = rrf.get(r["id"], 0) + 1.0/(k+rank+1)
        merged = {}
        for r in vector_results + bm25_results:
            merged[r["id"]] = r
        for cid in merged:
            merged[cid]["score"] = rrf[cid]
        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        # 父子块替换
        for r in results:
            if r.get("parent_text"):
                r["text"] = r["parent_text"]
        return results[:top_k]

    @staticmethod
    def _match_section_number(section_path: str, target_num: str) -> bool:
        """结构化章节匹配：
        target="4"   → 匹配 "第四章..."、"4.1..."、"4.7.3..."，不匹配 "2.1.4..."、"3.4..."
        target="4.6" → 匹配 "4.6..."、"4.6.1..."，不匹配 "3.4.6..."、"5.4.6..."
        """
        import re as _re
        cn_map = {"一":"1","二":"2","三":"3","四":"4","五":"5",
                  "六":"6","七":"7","八":"8","九":"9","十":"10"}
        for part in section_path.split(" / "):
            # 匹配 "4.3 xxx" 或 "4.3.1 xxx" 这种以数字章节号开头的段
            m = _re.match(r'^(\d+(?:\.\d+)*)\s', part)
            if m:
                sec_num = m.group(1)
                if sec_num == target_num or sec_num.startswith(target_num + "."):
                    return True
            # 匹配 "第四章 xxx" 这种中文数字章
            m2 = _re.match(r'^第([一二三四五六七八九十\d]+)章', part)
            if m2:
                ch = cn_map.get(m2.group(1), m2.group(1))
                if ch == target_num:
                    return True
        return False

    def _brainstorm_query(self, question: str) -> str:
        """问题头脑风暴：用LLM生成'答案可能在的章节类型'，扩展检索词。"""
        from openai import OpenAI
        from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, TEXT_MODEL_NAME

        if not SILICONFLOW_API_KEY:
            return question

        client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
        try:
            resp = client.chat.completions.create(
                model=TEXT_MODEL_NAME,
                messages=[
                    {"role": "system", "content": """你是论文检索专家。用户问了一个论文问题，请判断答案最可能在论文的哪个章节类型。
返回JSON：{"section_type": "可能的章节关键词", "extra_terms": "额外检索词"}
例如：
- 问"结论" → section_type="结论 小结 总结", extra_terms="主要结论 研究发现"
- 问"连接方式" → section_type="研究内容 本文工作 体系介绍", extra_terms="四边连接 两边连接 现浇 装配"
- 问"有限元模型" → section_type="有限元 模型建立 单元类型", extra_terms="ABAQUS S4R C3D8I"
- 问"摘要" → section_type="摘要 Abstract", extra_terms=""
只输出JSON。"""},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=200,
            )
            result = json.loads(resp.choices[0].message.content.strip())
            extra = result.get("extra_terms", "")
            if extra:
                print(f"  [头脑风暴] +'{extra}'")
                return question + " " + extra
            return question
        except Exception as e:
            print(f"  [头脑风暴] 失败: {e}")
            return question

    def _search_in_paper(self, query: str, paper_name: str, top_k: int = 10) -> list:
        """在指定论文内做完整检索：问题头脑风暴 → 混合检索 → reranker"""
        # 1. 问题头脑风暴：扩展检索词
        expanded_query = self._brainstorm_query(query)

        # 2. 在该论文内混合检索
        results = self._search_single_paper(expanded_query, paper_name, top_k=RECALL_K)

        # 3. Reranker精排（使用与向量/BM25相同的 expanded_query）
        if results and self.reranker:
            results = self.reranker.rerank(expanded_query, results, top_k=top_k)

        return results[:top_k]

    def _decompose_comparison_question(self, question: str, target_papers: List[str]) -> List[Dict]:
        """
        查询分解：把对比问题拆成多个子问题，每个子问题针对一篇论文。
        返回 [{"paper": "论文名", "sub_query": "子问题"}, ...]
        """
        from openai import OpenAI
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

        # 如果只有一篇论文，不需要分解
        if len(target_papers) <= 1:
            return None

        # 检查是否是对比类问题
        comparison_keywords = ["对比", "区别", "不同", "差异", "共同点", "共同", "分别",
                               "比较", "相比", "vs", "和", "与", "跟"]
        is_comparison = any(kw in question for kw in comparison_keywords)
        if not is_comparison:
            return None

        print(f"  [查询分解] 检测到对比问题，拆解子问题...")

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        papers_str = "\n".join(f"- {p}" for p in target_papers)

        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": """你是查询分解专家。用户问了一个对比多篇论文的问题，请把它拆成针对每篇论文的独立子问题。
返回JSON格式：{"sub_queries": [{"paper": "论文名", "sub_query": "该论文的具体问题"}]}
只输出JSON，不要其他内容。"""},
                {"role": "user", "content": f"原始问题：{question}\n\n涉及的论文：\n{papers_str}\n\n请拆成子问题。"}
            ],
            temperature=0.1,
            max_tokens=500,
        )

        try:
            result = json.loads(response.choices[0].message.content.strip())
            sub_queries = result.get("sub_queries", [])
            print(f"  [查询分解] 拆成{len(sub_queries)}个子问题")
            for sq in sub_queries:
                print(f"    → [{sq['paper'][:20]}] {sq['sub_query'][:50]}")
            return sub_queries
        except Exception as e:
            print(f"  [查询分解] 失败: {e}")
            return None

    def search(self, question: str, top_k: int = TOP_K, allow_decompose: bool = True) -> Dict:
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

        # 1.2 对比问题标记
        is_comparison = len(target_papers) >= 2 and any(kw in question for kw in ["对比", "区别", "不同", "差异", "共同点", "分别", "比较", "相比", "异同", "有何异同", "相同点"])
        if is_comparison:
            print(f"  [对比检索] 检测到对比问题")

        # 1.3 查询分解：对比问题时，拆解成子问题，分别检索
        if is_comparison and allow_decompose:
            sub_queries = self._decompose_comparison_question(question, target_papers)
            if sub_queries and len(sub_queries) >= 2:
                # 分别检索每个子问题，只在该论文内检索
                all_grouped = []
                for sq in sub_queries:
                    paper_name = sq["paper"]
                    sub_q = sq["sub_query"]
                    print(f"  [子问题检索] [{paper_name[:20]}] {sub_q[:40]}")
                    # 只搜这一篇论文：强制设置target_papers
                    sub_result = self._search_in_paper(sub_q, paper_name, top_k=10)
                    sub_chunks = sub_result[:5]
                    for r in sub_chunks:
                        r["group_paper"] = paper_name
                    all_grouped.extend(sub_chunks)
                    print(f"    → 命中{len(sub_chunks)}个chunk")

                if all_grouped:
                    print(f"  [查询分解] 共检索到{len(all_grouped)}个chunk（分{len(sub_queries)}组）")
                    return {
                        "results": all_grouped,
                        "query_info": query_info,
                        "debug_info": {
                            "retrieval_mode": "comparison_decompose",
                            "pre_expansion_ids": [r["id"] for r in all_grouped],
                            "post_expansion_ids": [r["id"] for r in all_grouped],
                        }
                    }

        # 1.5 特殊问题：问"有几个章节/目录/结构"时，直接返回章节列表
        if target_papers and any(kw in content_query for kw in ["章节", "目录", "几个章", "有几章"]):
            results = []
            for paper_name in target_papers:
                paper_chunks = self.paper_chunks_map.get(paper_name, [])
                # 收集所有不同的一级和二级章节
                seen_sections = set()
                for c in paper_chunks:
                    # 取到二级章节（如"第五章 / 5.1 xxx"）
                    parts = c.section_path.split(" / ")
                    section_key = " / ".join(parts[:2]) if len(parts) >= 2 else c.section_path
                    if section_key not in seen_sections:
                        seen_sections.add(section_key)
                        results.append({
                            "id": c.id,
                            "text": f"章节列表：{section_key}",
                            "paper_name": c.paper_name,
                            "section_path": c.section_path,
                            "page_start": c.page_start,
                            "page_end": c.page_end,
                            "score": 1.0,
                        })
            if results:
                return {
                    "results": results,
                    "query_info": query_info,
                    "debug_info": {
                        "retrieval_mode": "section_list",
                        "pre_expansion_ids": [r["id"] for r in results],
                        "post_expansion_ids": [r["id"] for r in results],
                    }
                }

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

        # 3. 混合召回：向量检索 + BM25关键词检索，RRF融合
        search_k = RECALL_K * 2 if target_papers else top_k * 2
        # 对比问题时不filter（全局检索，让embedding自由匹配两篇论文的章节）
        where_filter = {"paper_name": {"$in": target_papers}} if (target_papers and not is_comparison) else None

        # 对比问题时，扩展查询词（把抽象词扩展成具体学术词）
        expanded_query = content_query
        if is_comparison:
            expansion_map = {
                "研究方法": "试验 有限元 试件 加载 模型 ABAQUS",
                "方法": "试验 有限元 试件 加载 模型",
                "结论": "结论 总结 本章小结",
                "设计": "设计方法 计算公式 承载力 刚度",
                "创新": "创新点 主要工作 本文贡献",
            }
            for abstract, concrete in expansion_map.items():
                if abstract in content_query:
                    expanded_query = content_query + " " + concrete
                    print(f"  [查询扩展] '{abstract}' → +'{concrete}'")
                    break

        # 问题头脑风暴：用LLM生成"答案可能在的章节关键词"
        if not is_comparison:  # 对比问题已经在子问题里做过头脑风暴了
            expanded_query = self._brainstorm_query(expanded_query)

        # 3a. 向量检索
        vector_results = self.vector_db.search(expanded_query, top_k=search_k, where=where_filter)

        # 3b. BM25关键词检索
        bm25_tokens = list(jieba.cut(expanded_query))
        bm25_scores = self.bm25.get_scores(bm25_tokens)
        # 取BM25 top_k
        bm25_top_idx = bm25_scores.argsort()[-search_k:][::-1]
        bm25_results = []
        for idx in bm25_top_idx:
            c = self.all_chunks[idx]
            # 对比问题时不filter
            if target_papers and not is_comparison and c.paper_name not in target_papers:
                continue
            bm25_results.append({
                "id": c.id,
                "text": c.text,
                "paper_name": c.paper_name,
                "section_path": c.section_path,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "parent_id": c.parent_id,
                "parent_text": c.parent_text,
                "score": float(bm25_scores[idx]),
            })

        # 3c. RRF融合（Reciprocal Rank Fusion）
        # score = sum(1 / (k + rank))，k=60
        k = 60
        rrf_scores = {}
        for rank, r in enumerate(vector_results):
            cid = r["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
            r["rrf_source"] = "vector"
        for rank, r in enumerate(bm25_results):
            cid = r["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
            if cid in rrf_scores:
                r["rrf_source"] = "both"
            else:
                r["rrf_source"] = "bm25"

        # 合并结果，按RRF分数排序
        merged = {}
        for r in vector_results + bm25_results:
            cid = r["id"]
            if cid not in merged:
                merged[cid] = r
            merged[cid]["score"] = rrf_scores[cid]

        all_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        print(f"  [混合检索] 向量{len(vector_results)} + BM25{len(bm25_results)} → 融合后{len(all_results)}个候选")

        # 3.5 结构化检索：query里有章节关键词时，直接从目标论文找对应章节
        #     不依赖语义召回，直接按section_path匹配
        import re
        section_keywords = ["结论", "总结", "conclusion", "conclusions",
                           "引言", "绪论", "introduction", "abstract", "摘要",
                           "方法", "试验", "实验", "method", "experiment",
                           "结果", "result", "results", "讨论", "discussion",
                           "设计", "design", "分析", "analysis"]
        query_lower = content_query.lower()
        hit_section_kw = None
        for kw in section_keywords:
            if kw in query_lower:
                hit_section_kw = kw
                break

        # 只有结构性关键词才允许触发 shortcut（"试验""结论"等真正对应章节名）
        # "方法""设计""分析"等泛化词不触发 shortcut，只用于章节名加权
        shortcut_trigger_keywords = ["摘要", "引言", "绪论", "试验", "结论", "总结"]
        hit_shortcut_kw = None
        for kw in shortcut_trigger_keywords:
            if kw in query_lower:
                hit_shortcut_kw = kw
                break

        # 特殊处理：query提到"英文摘要"时，强制把Abstract章节chunks加入
        if "英文摘要" in content_query or "abstract" in query_lower:
            for paper_name in (target_papers or []):
                for c in self.paper_chunks_map.get(paper_name, []):
                    sp = c.section_path.replace(" ", "").lower()
                    if "abstract" in sp or "abstract" in c.id.lower():
                        already = any(r["id"] == c.id for r in all_results)
                        if not already:
                            all_results.append({
                                "id": c.id,
                                "text": c.text,
                                "paper_name": c.paper_name,
                                "section_path": c.section_path,
                                "page_start": c.page_start,
                                "page_end": c.page_end,
                                "parent_id": c.parent_id,
                                "parent_text": c.parent_text,
                                "score": 0.7,  # 高分优先
                            })
            print(f"  [章节定位] 检测到'英文摘要'，加入Abstract章节chunks")

        # 3.6 章节号匹配：query里有"第X章"、"X.X节"时，直接按章节号匹配
        hit_section_num = None
        # 匹配"第X章"
        m = re.search(r'第([一二三四五六七八九十\d]+)章', content_query)
        if m:
            # 把中文数字转成阿拉伯数字
            cn_num = m.group(1)
            cn_map = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
                      "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}
            if cn_num in cn_map:
                hit_section_num = cn_map[cn_num]
            else:
                hit_section_num = cn_num

        # 匹配"X.X节"或"X.X.X"
        if not hit_section_num:
            m = re.search(r'(\d+\.\d+(?:\.\d+)?)', content_query)
            if m:
                hit_section_num = m.group(1)

        # 3.6 章节号匹配 / 结构性关键词匹配
        #    hit_section_num（章节号）和 hit_shortcut_kw（摘要/试验/结论等）都允许触发
        #    hit_section_kw 中的"方法""设计""分析"等泛化词不触发 shortcut
        if target_papers and (hit_section_num or hit_shortcut_kw) and not is_comparison:
            for paper_name in target_papers:
                for c in self.paper_chunks_map.get(paper_name, []):
                    matched = False
                    if hit_section_num and self._match_section_number(c.section_path, hit_section_num):
                        matched = True
                    if not matched and hit_shortcut_kw:
                        sp = c.section_path.replace(" ", "").lower()
                        if hit_shortcut_kw.replace(" ", "").lower() in sp:
                            matched = True
                    if matched:
                        already = any(r["id"] == c.id for r in all_results)
                        if not already:
                            all_results.append({
                                "id": c.id,
                                "text": c.text,
                                "paper_name": c.paper_name,
                                "section_path": c.section_path,
                                "page_start": c.page_start,
                                "page_end": c.page_end,
                                "score": 0.6,
                            })

        # 4. 如果是对比问题：全局检索后按论文分组，每篇保留top 5
        #    普通问题：按target_papers过滤
        if is_comparison and target_papers:
            # 只保留目标论文的chunks
            filtered = [r for r in all_results if r["paper_name"] in target_papers]
            # 按论文分组，每篇保留top 5
            by_paper = {}
            for r in filtered:
                pn = r["paper_name"]
                if pn not in by_paper:
                    by_paper[pn] = []
                by_paper[pn].append(r)
            grouped = []
            for pn, rs in by_paper.items():
                grouped.extend(rs[:5])
            grouped.sort(key=lambda x: x["score"], reverse=True)
            filtered = grouped
            print(f"  [对比检索] {len(target_papers)}篇论文分组后保留{len(filtered)}个chunk")
        elif target_papers:
            filtered = [
                r for r in all_results
                if r["paper_name"] in target_papers
            ]
        else:
            filtered = all_results

        # 4.5 章节名加权：如果query里有关键词出现在章节名里，给加分
        for r in filtered:
            # 去掉空格后再匹配（"摘  要" vs "摘要"）
            section_no_space = r["section_path"].replace(" ", "").lower()
            # 如果章节名里包含query里的关键词，加0.2分
            for kw in section_keywords:
                kw_no_space = kw.replace(" ", "")
                if kw in query_lower and kw_no_space in section_no_space:
                    r["score"] += 0.2
                    break

        # 按加权后的分数重新排序
        filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)

        # 5. Reranker 精排：从召回的候选里挑出最相关的
        #    如果已通过章节号或结构性关键词精确匹配到了章节，跳过Reranker直接返回
        if target_papers and (hit_section_num or hit_shortcut_kw) and not is_comparison:
            results = []
            for paper_name in target_papers:
                for c in self.paper_chunks_map.get(paper_name, []):
                    matched = False
                    if hit_section_num and self._match_section_number(c.section_path, hit_section_num):
                        matched = True
                    if not matched and hit_shortcut_kw:
                        sp = c.section_path.replace(" ", "").lower()
                        if hit_shortcut_kw.replace(" ", "").lower() in sp:
                            matched = True
                    if matched:
                        results.append({
                            "id": c.id,
                            "text": c.text,
                            "paper_name": c.paper_name,
                            "section_path": c.section_path,
                            "page_start": c.page_start,
                            "page_end": c.page_end,
                            "score": 1.0,
                        })
            # 如果精确匹配到了章节，返回；否则继续走正常检索
            if results:
                return {
                    "results": results,
                    "query_info": query_info,
                    "debug_info": {
                        "retrieval_mode": "section_shortcut",
                        "pre_expansion_ids": [r["id"] for r in results],
                        "post_expansion_ids": [r["id"] for r in results],
                    }
                }

        if filtered and self.reranker:
            # 精排前20个候选（章节名加权后前面的）
            candidates_for_rerank = filtered[:20]
            # ---- debug: 记录 reranker 前的真实 candidates ----
            pre_rerank_ids = [r["id"] for r in candidates_for_rerank]
            filtered = self.reranker.rerank(content_query, candidates_for_rerank, top_k=top_k)
            post_rerank_ids = [r["id"] for r in filtered]
        else:
            pre_rerank_ids = [r["id"] for r in filtered[:20]]
            post_rerank_ids = [r["id"] for r in filtered[:top_k]]

        # query提到"英文摘要"时，先把Abstract chunks加回来
        if "英文摘要" in content_query:
            abstract_in_all = [r for r in all_results if "abstract" in r["section_path"].lower()]
            existing_ids = {r["id"] for r in filtered}
            abstract_new = [r for r in abstract_in_all if r["id"] not in existing_ids]
            filtered = abstract_new + filtered
            print(f"  [强制排序] 从{len(abstract_in_all)}个Abstract chunks中加入{len(abstract_new)}个新的到前面")

        # 6. 上下文扩展（章节级聚合 + 相邻扩展）
        # ---- debug: 记录 expand 前的 ids（reranker 后） ----
        pre_expansion_ids = [r["id"] for r in filtered]
        expanded = self._expand_neighbors(filtered)

        # query提到"英文摘要"时，最后再把Abstract chunks排到最前面
        if "英文摘要" in content_query:
            abstract_exp = [r for r in expanded if "abstract" in r["section_path"].lower()]
            other_exp = [r for r in expanded if "abstract" not in r["section_path"].lower()]
            expanded = abstract_exp + other_exp

        # query提到视觉相关词时，把图表chunk也加进来
        visual_keywords = ["图", "曲线", "长什么样", "示意图", "照片", "可视化", "图形"]
        if any(kw in content_query for kw in visual_keywords):
            fig_chunks = [r for r in all_results if r.get("section_path", "").startswith("图表")]
            if target_papers:
                fig_chunks = [r for r in fig_chunks if r.get("paper_name") in target_papers]
            existing_ids = {r["id"] for r in expanded}
            fig_new = [r for r in fig_chunks if r["id"] not in existing_ids]
            if fig_new:
                expanded = fig_new[:3] + expanded  # 图表chunk放前面
                print(f"  [视觉] 加入{len(fig_new[:3])}个图表chunk")

        # 7. 对比问题时返回更多chunks（两篇论文都要覆盖）
        if target_papers and not is_comparison:
            results = expanded
        elif is_comparison:
            results = expanded[:top_k*3]  # 对比问题多返回
        else:
            results = expanded[:top_k]

        # 8. 父子分块：把返回给LLM的text替换成parent_text（完整父块）
        for r in results:
            if r.get("parent_text"):
                r["original_text"] = r["text"]  # 保留原始子块
                r["text"] = r["parent_text"]     # LLM看到的是完整父块

        return {
            "results": results,
            "query_info": query_info,
            "debug_info": {
                "retrieval_mode": "hybrid_rerank",
                "pre_expansion_ids": pre_expansion_ids,
                "post_expansion_ids": [r["id"] for r in expanded],
                "pre_rerank_ids": pre_rerank_ids,
                "post_rerank_ids": post_rerank_ids,
            }
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
        "张三小论文里主要结论有几条，是什么？",
        "赵六论文中，试验试件的尺寸，以及加载方式是什么？",
        "周八论文提出的具体螺栓需求公式是什么？",
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
