"""
reranker.py — 重排序（精排）

原理：
1. 语义检索（召回）是粗排，只看"意思大概像不像"
2. Reranker 是精排，看"这个chunk到底有没有回答用户的问题"

为什么需要：
- 召回阶段返回top 30个候选
- 但其中很多是"意思差不多"但"没回答问题"的
- Reranker 模型会把query和chunk放在一起对比，判断真正的相关性
- 然后重新排序，取最相关的top 5
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL


class Reranker:
    """重排序器：用 CrossEncoder 模型做精排"""

    def __init__(self, model_name: str = RERANKER_MODEL):
        """
        加载 reranker 模型。
        第一次运行会自动下载（约 500MB）。
        """
        print(f"正在加载 Reranker 模型: {model_name} ...")
        self.model = CrossEncoder(model_name)
        print("Reranker 模型加载完成！")

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        对候选chunks重新排序。

        参数：
            query: 用户问题
            candidates: 召回阶段返回的候选chunks列表
            top_k: 最终返回多少个

        返回：
            重新排序后的top_k个chunks
        """
        if not candidates:
            return []

        # 把query和每个chunk组成一对，让模型打分
        # 优先用parent_text（完整父块）打分，这样Reranker能看到完整上下文
        pairs = []
        for c in candidates:
            # 有parent_text就用parent_text，没有就用text
            text_for_rerank = c.get("parent_text") or c["text"]
            # 太长的话截断（Reranker有输入长度限制）
            if len(text_for_rerank) > 1000:
                text_for_rerank = text_for_rerank[:1000]
            pairs.append([query, text_for_rerank])
        
        scores = self.model.predict(pairs)

        # 按新分数排序
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        # 排序，取top_k
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
