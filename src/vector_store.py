"""
vector_store.py — ChromaDB 持久化向量存储（BGE embedding版）

替换原来的内存cosine_similarity方案：
1. 数据持久化在磁盘上，重启不丢失
2. 增量更新：上传新论文只add新数据，不重建索引
3. 子块embedding存入，metadata带父块信息
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, DB_DIR
from chunker import Chunk


class VectorStore:
    """基于ChromaDB的持久化向量库"""

    def __init__(self, collection_name: str = "paper_chunks"):
        DB_DIR.mkdir(parents=True, exist_ok=True)

        print(f"正在连接 ChromaDB 持久化存储: {DB_DIR} ...")
        self.client = chromadb.PersistentClient(
            path=str(DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"ChromaDB 连接成功，当前已有 {self.collection.count()} 个向量")

        # 加载BGE embedding模型
        print(f"加载 Embedding 模型: {EMBEDDING_MODEL} ...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding 模型加载完成！")

    def add_chunks(self, chunks: List[Chunk]):
        """增量添加新chunks到向量库"""
        if not chunks:
            return

        # 生成文档文本（带章节路径前缀）
        documents = [f"[{c.section_path}] {c.text}" for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = []
        for c in chunks:
            metadatas.append({
                "paper_name": c.paper_name,
                "section_path": c.section_path,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "parent_id": c.parent_id,
                "parent_text": c.parent_text[:2000] if c.parent_text else "",
            })

        # 用BGE模型生成embedding
        print(f"正在为 {len(chunks)} 个新chunk生成向量...")
        embeddings = self.model.encode(documents, show_progress_bar=True, batch_size=32).tolist()

        # 插入ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        print(f"已添加 {len(chunks)} 个chunk到向量库，当前共 {self.collection.count()} 个")

    def search(self, query: str, top_k: int = 100,
               where: Optional[Dict] = None) -> List[Dict]:
        """向量检索，返回top_k个结果"""
        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
        )

        output = []
        if not results["ids"] or not results["ids"][0]:
            return output

        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][i] if False else results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            output.append({
                "id": chunk_id,
                "text": results["documents"][0][i],
                "paper_name": meta["paper_name"],
                "section_path": meta["section_path"],
                "page_start": int(meta["page_start"]),
                "page_end": int(meta["page_end"]),
                "parent_id": meta.get("parent_id", ""),
                "parent_text": meta.get("parent_text", ""),
                "score": float(similarity),
            })

        return output

    def delete_by_paper(self, paper_name: str):
        """删除某篇论文的所有chunks"""
        self.collection.delete(where={"paper_name": paper_name})
        print(f"已删除论文 '{paper_name}' 的所有向量")

    def get_paper_names(self) -> List[str]:
        """获取库中所有论文名"""
        results = self.collection.get(include=["metadatas"])
        papers = set()
        for m in results["metadatas"]:
            papers.add(m["paper_name"])
        return sorted(papers)

    def count(self) -> int:
        return self.collection.count()

    def add_figure(self, text: str, metadata: dict):
        """存入figure_db集合"""
        fig_col = self.client.get_or_create_collection("figure_db")
        emb = self.model.encode([text]).tolist()
        fig_col.add(
            documents=[text],
            ids=[f"fig_{metadata.get('figure_id', 'unknown')}_{hash(text) % 10000}"],
            embeddings=emb,
            metadatas=[metadata],
        )
