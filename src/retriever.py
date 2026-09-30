"""
retriever.py — 检索模块（双方案）

两种检索方式，方便后面做对比实验：
1. TF-IDF 关键词检索（baseline）—— 用 sklearn，秒出结果
2. 向量语义检索（semantic）—— 用 ChromaDB + embedding 模型

现在先用 TF-IDF 跑通，后面再加语义检索做对比。
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DB_DIR, TOP_K
from chunker import Chunk


class TFIDFRetriever:
    """
    TF-IDF 关键词检索器（基线方案）

    原理：统计每个词在文档中的出现频率，
    用户提问时，把问题也转成向量，跟所有 chunks 算余弦相似度，
    返回最相似的 top_k 个。

    优点：快、不需要下载模型
    缺点：不懂语义，只能匹配字面相同的词
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',  # 去停用词
            max_features=10000,
        )
        self.chunks: List[Chunk] = []
        self.tfidf_matrix = None
        self.is_trained = False

    def build_index(self, chunks: List[Chunk]):
        """把所有 chunks 建成检索索引"""
        self.chunks = chunks
        if not chunks:
            return

        documents = [c.text for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.is_trained = True
        print(f"TF-IDF 索引建立完成，共 {len(chunks)} 个 chunks")

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """检索最相关的 chunks"""
        if not self.is_trained:
            return []

        # 把问题转成 TF-IDF 向量
        query_vec = self.vectorizer.transform([query])

        # 算余弦相似度
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 取 top_k
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "id": chunk.id,
                "text": chunk.text,
                "paper_name": chunk.paper_name,
                "section_path": chunk.section_path,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "score": float(similarities[idx]),
            })

        return results

    def save(self, path: Path):
        """保存索引到磁盘"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'chunks': self.chunks,
                'tfidf_matrix': self.tfidf_matrix,
            }, f)

    def load(self, path: Path):
        """从磁盘加载索引"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.chunks = data['chunks']
        self.tfidf_matrix = data['tfidf_matrix']
        self.is_trained = True
        print(f"TF-IDF 索引加载完成，共 {len(self.chunks)} 个 chunks")


# =========================
# 命令行测试
# =========================

if __name__ == "__main__":
    import sys
    from config import PROCESSED_DIR

    # 加载 chunks
    chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))
    if not chunk_files:
        print("没有找到 chunks 文件，请先运行 chunker.py")
        sys.exit()

    with open(chunk_files[0], "r", encoding="utf-8") as f:
        chunk_dicts = json.load(f)
    chunks = [Chunk(**c) for c in chunk_dicts]
    print(f"加载了 {len(chunks)} 个 chunks\n")

    # 建索引
    retriever = TFIDFRetriever()
    retriever.build_index(chunks)

    # 测试检索
    print("\n" + "=" * 70)
    print("测试 TF-IDF 检索")
    print("=" * 70)

    test_queries = [
        "What is the failure mode of the shear wall?",
        "How does the steel plate contribute to shear resistance?",
        "What material properties were used?",
        "螺栓连接的抗剪承载力",
    ]

    for q in test_queries:
        print(f"\n问题: {q}")
        results = retriever.search(q, top_k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r['score']:.3f} | p.{r['page_start']} | {r['section_path'][:50]}")
            print(f"      {r['text'][:100]}...")
