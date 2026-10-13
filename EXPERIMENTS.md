# 实验登记（EXPERIMENTS）

> 规则：**任何优化都先登记再动手；做完后回填结果。负结果同样登记。**
> 一行一个实验，明细放 `evaluation/`。格式沿用「问题 → 假设 → 改动 → 结果 → 结论」。

---

## 模板

```markdown
### E<n> <一句话标题>
- 日期：
- 方向/负责人：
- 分支：
- 问题：
- 假设：
- 改动（文件 + 关键参数）：
- 结果（`evaluation/<文件>.json`）：
- 结论：
```

---

## 已登记实验

### E1 检索评测口径修正（v2 → v3）
- 日期：2026-10-03
- 方向：检索评测
- 问题：v2 的 `recall_at_10` 把「章节聚合 + 邻居扩展」后的块也算进 Top-10，虚高，且无法区分「真命中」与「靠扩展蹭到」。
- 假设：只统计**扩展前**的 Top-10（即 reranker 精排后、章节聚合前）能反映真实检索能力。
- 改动：`evaluate_v2.py` 增加 `strict_recall_at_10`（只算 `pre_expansion_ids`）与 `recall_after_expansion`（算 `post_expansion_ids`）双口径；输出 `evaluation/eval_v3_result.json`。
- 结果：`evaluation/eval_v2_result.json`（旧口径 73.9%）vs `evaluation/eval_v3_result.json`（strict 78.9% / 扩展后 89.2%）。
- 结论：口径不同**不可直接比**。此后对外一律用 strict 口径，并注明「这是检索指标，不是准确率」。

### E2 Conditional-TOC（对比题补充搜索词）
- 日期：2026-10-11
- 方向：检索 / 查询改写
- 分支：（实验分支，未并入主干）
- 问题：跨论文对比题最弱（strict 58.3%），怀疑是 brainstorm 出的具体术语不足（<3 个），导致检索词覆盖不够。
- 假设：在 `_search_in_paper` 里 brainstorm 后，若具体术语 < 3 个，则追加「目录生成」的搜索词，能提升对比题召回。
- 改动：`ablation_cond_toc.py`、`eval_30_cond_toc.py`（只跑检索层，3 轮）。
- 结果：`evaluation/eval_cond_toc_3rounds.json`
  - 对比题（5 题，3 轮合计 n=15）：Baseline strict **0.750** → Cond-TOC **0.500**（**−25.0 pp**）
  - 非对比题（25 题）不受影响：**79.72%**
  - 全 30 题等效：**74.8%**（较基线 78.9% 约 **−4.1 pp**）
- 结论：**负结果**。追加目录搜索词引入了噪声，反而稀释了精排信号。**不合并**；保留分支与数据作为反例证据。

### E3 q026 检索失败归因
- 日期：2026-10-xx（系列实验）
- 方向：检索 / 查询扩展
- 问题：q026「雷杰鑫大论文和王鑫涛大论文在研究方法上有什么共同点和不同点？」在 `comparison_decompose` 模式下 **strict=0.0、生成 3.0**，是 30 题里最差的样本。
- 假设：对比题被拆成子问题后，每篇论文**只保留 Top-5**，且子问题检索词过短，导致 gold chunk 未进入候选。
- 改动/试过：`ablation_q026.py`、`ablation_q026_prompt.py`（证据导向 prompt）、`ablation_q026_rerank_q.py`（subquery vs expanded query 做 rerank）、`ablation_q026_terms.py`、`ablation_q026_terms2.py`（search_terms 数量扫描）、`ablation_q026_autogen.py`、`ablation_q026_evidence.py`。
- 结果：**打印输出，未落盘 JSON**（见各脚本运行日志）。
- 结论：q026 仍为**未解决问题**，已如实记入作品集的「已知短板」，不粉饰。

### E4 RRF vs Reranker 消融
- 日期：2026-10-xx
- 方向：检索 / 精排
- 问题：RRF 融合与 cross-encoder 精排各自贡献多少？
- 改动：`ablation_reranker.py`（关掉/开启 reranker 对比）、`ablation_real.py`（从真实代码路径拦截，对比 RRF 排名与 reranker 排名）。
- 结果：**打印输出，未落盘 JSON**。
- 结论：reranker 对 Top-10 排序有正向作用（细节见脚本输出）。建议**补一次带 JSON 落盘的复跑**，纳入正式对照。

### E5 TOC 跨章节检索
- 日期：2026-10-xx
- 方向：检索 / 章节感知
- 问题：跨章节问题（结论散落在多章）召回不足。
- 改动：`ablation_toc_cross.py`、`ablation_recheck.py`。
- 结果：**打印输出，未落盘 JSON**。
- 结论：待补正式对照。

---

## 待办（建议优先级）

| 优先级 | 事项 | 理由 |
|---|---|---|
| 高 | 把 E4/E5 的消融补成带 JSON 落盘的正式对照 | 目前只有打印输出，结论不可复查 |
| 高 | 攻「按章节问」（65.3%）与「跨论文对比」（58.3%） | 当前最大短板 |
| 中 | 复盘 E2 为何变差（是否 prompt/噪声导致） | 负结果里可能藏着正解 |
| 中 | q026 单题专项 | 已知最差样本 |
