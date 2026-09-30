"""
vector_store.py — 向量数据库

负责：
1. 把 chunks 转成向量（Embedding）
2. 存入 ChromaDB
3. 提供语义检索接口：输入问题，返回最相关的 chunks

关于 Embedding 模型：
- ChromaDB 默认自带 all-MiniLM-L6-v2（英文效果好，中文一般）
- 后面如果中文论文效果不好，可以换成 bge-small-zh 等中文模型
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

from config import DB_DIR, TOP_K
from chunker import Chunk


class VectorStore:
    """向量数据库管理：入库 + 检索"""

    def __init__(self, collection_name: str = "paper_assistant"):
        """
        初始化向量数据库
        collection_name 就像数据库里的一张表
        """
        # 确保目录存在
        DB_DIR.mkdir(parents=True, exist_ok=True)

        # 连接本地 ChromaDB（持久化存储）
        self.client = chromadb.PersistentClient(
            path=str(DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Research paper knowledge base"},
        )

    def add_chunks(self, chunks: List[Chunk]):
        """
        把一批 chunks 存入向量数据库。

        ChromaDB 会自动：
        1. 调用 embedding 模型把文字转成向量
        2. 存到本地磁盘
        """
        if not chunks:
            print("没有 chunks 需要入库")
            return

        # ChromaDB 要求三个列表：ids, documents, metadatas
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "paper_name": c.paper_name,
                "section_path": c.section_path,
                "page_start": c.page_start,
                "page_end": c.page_end,
            }
            for c in chunks
        ]

        # 分批入库（每批 100 条，避免一次性太大）
        batch_size = 100
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
            )
            print(f"  入库进度: {min(i+batch_size, total)}/{total}")

        print(f"完成，共入库 {total} 个 chunks")

    def search(self, query: str, top_k: int = TOP_K,
               where: Optional[Dict] = None) -> List[Dict]:
        """
        语义检索：输入问题，返回最相关的 top_k 个 chunks。

        返回格式：
        [
            {
                "id": "...",
                "text": "...",
                "paper_name": "...",
                "section_path": "...",
                "page_start": 3,
                "score": 0.85,  # 相似度分数，越高越相关
            },
            ...
        ]
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,  # 可以加过滤条件，比如只搜某篇论文
        )

        # 整理成易读的格式
        hits = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                hit = {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "paper_name": results["metadatas"][0][i].get("paper_name", ""),
                    "section_path": results["metadatas"][0][i].get("section_path", ""),
                    "page_start": results["metadatas"][0][i].get("page_start", 0),
                    "page_end": results["metadatas"][0][i].get("page_end", 0),
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0,
                }
                hits.append(hit)

        return hits

    def count(self) -> int:
        """当前数据库里有多少个 chunks"""
        return self.collection.count()

    def clear(self):
        """清空数据库（重新开始时用）"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"description": "Research paper knowledge base"},
        )
        print("数据库已清空")


# =========================
# 命令行测试
# =========================

if __name__ == "__main__":
    import sys
    from chunker import StructuralChunker
    from config import PROCESSED_DIR

    # 1. 读取 chunks JSON
    chunk_files = list(PROCESSED_DIR.glob("*_chunks.json"))
    if not chunk_files:
        print("没有找到 chunks 文件，请先运行 chunker.py")
        sys.exit()

    chunks_file = chunk_files[0]
    print(f"读取 chunks: {chunks_file.name}")

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunk_dicts = json.load(f)

    chunks = [Chunk(**c) for c in chunk_dicts]
    print(f"共 {len(chunks)} 个 chunks\n")

    # 2. 入库
    store = VectorStore()
    print(f"当前数据库已有: {store.count()} 个 chunks")

    # 如果数据库是空的，才入库（避免重复）
    if store.count() == 0:
        print("开始入库...")
        store.add_chunks(chunks)
    else:
        print("数据库已有数据，跳过入库（如需重新入库请调用 store.clear()）")

    # 3. 测试检索
    print("\n" + "=" * 70)
    print("测试语义检索")
    print("=" * 70)

    test_queries = [
        "What is the failure mode of the shear wall?",
        "How does the steel plate contribute to shear resistance?",
        "What are the material properties used in the test?",
    ]

    for q in test_queries:
        print(f"\n问题: {q}")
        results = store.search(q, top_k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r['score']:.3f} | p.{r['page_start']} | {r['section_path'][:60]}")
            print(f"      {r['text'][:100]}...")
