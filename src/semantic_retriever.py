"""
semantic_retriever.py — 语义检索（Embedding）

原理：
1. 用 embedding 模型把每段文字转成向量（一组数字）
2. 用户提问时，把问题也转成向量
3. 算余弦相似度，找最接近的 top_k 个

和 TF-IDF 的区别：
- TF-IDF 靠字面匹配（搜"shear"才找得到"shear strength"）
- 语义检索靠意思匹配（搜"抗剪"也能找到英文的"shear strength"）
"""

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import TOP_K
from chunker import Chunk


class SemanticRetriever:
    """语义检索器：用 embedding 模型做向量检索"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        加载 embedding 模型。
        第一次运行会自动下载模型（约 100MB），之后就存在本地了。
        """
        print(f"正在加载 Embedding 模型: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.embeddings = None
        self.is_trained = False
        print("模型加载完成！")

    def build_index(self, chunks: List[Chunk]):
        """把所有 chunks 转成向量，建立索引"""
        self.chunks = chunks
        if not chunks:
            return

        # 关键：生成embedding时，把章节路径加到文本前面
        # 这样模型能感知到"这是Conclusion章节的内容"，搜索"主要结论"时能命中
        documents = [f"[{c.section_path}] {c.text}" for c in chunks]
        # 把所有文字转成向量
        print(f"正在生成 {len(chunks)} 个文本块的向量...")
        self.embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            batch_size=32,
        )
        self.is_trained = True
        print(f"语义索引建立完成，共 {len(chunks)} 个文本块")

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """语义检索：输入问题，返回最相关的 top_k 个 chunks"""
        if not self.is_trained:
            return []

        # 把问题转成向量
        query_embedding = self.model.encode([query])

        # 算余弦相似度
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()

        # 取 top_k
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] <= 0.1:  # 相似度太低就不返回了
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


# =========================
# 命令行测试
# =========================

if __name__ == "__main__":
    import sys
    from config import PROCESSED_DIR

    chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))
    if not chunk_files:
        print("没有找到 chunks 文件")
        sys.exit()

    # 加载所有 chunks
    all_chunks = []
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            chunk_dicts = json.load(f)
        for c in chunk_dicts:
            all_chunks.append(Chunk(**c))

    print(f"加载了 {len(all_chunks)} 个 chunks\n")

    # 建语义索引
    retriever = SemanticRetriever()
    retriever.build_index(all_chunks)

    # 测试中英文混合提问
    print("\n" + "=" * 70)
    print("测试语义检索（中文问题搜英文论文）")
    print("=" * 70)

    test_queries = [
        "剪切破坏模式",          # 中文
        "螺栓连接的受力特性",      # 中文
        "What is the failure mode?",  # 英文
    ]

    for q in test_queries:
        print(f"\n问题: {q}")
        results = retriever.search(q, top_k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r['score']:.3f} | p.{r['page_start']} | {r['section_path'][:50]}")
            print(f"      {r['text'][:100]}...")
