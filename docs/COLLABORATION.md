# 协作机制（COLLABORATION）

> 目标：多人并行改**论文写作模块**，互不覆盖。
> 三条铁律：**① 只动写作层 ② 检索层已冻结 ③ 数据不进 git。**
>
> ⚠️ **本组当前只开放写作层**：切分 / embedding / 召回 / 精排都不用改，因此也**不跑评测**。
> 下面第三节的评测规范只在「将来重开检索优化」时才适用，现在可以跳过。

---

## 一、谁只碰哪些文件（分工边界）

先看 `docs/INTERFACES.md` 的五层划分。当前只开放 **L4 生成层里的写作模块**：

| 方向 | 只允许改 | 判定标准 |
|---|---|---|
| 写作主链 | `writing_agent.py` 的 6 个节点与 Prompt | 人工看生成效果（见 ONBOARDING 5.1） |
| 看图描述 | `vision_extractor.py` | 描述是否准确 |
| 导师风格库 | `parse_word_revisions.py` + `style_db` 集合 | 改写是否更像导师 |
| 问答 Prompt（可选） | `generator.py` | 引用是否对得上 |

**已冻结（不要改）**：`pdf_parser.py`、`chunker.py`、`multimodal_parser.py`、`vector_store.py`、
`smart_retriever.py`、`reranker.py`、`query_understanding.py`、`app.py`、`session_store.py`。

> 跨层改动（例如同时改 `writing_agent.py` 和 `smart_retriever.py`）是允许的，但**必须在 PR 描述里说明是「契约变更」**，并走 `INTERFACES.md` ⑤ 的流程。

---

## 二、分支策略

```bash
# 主干永远保持「能跑起来」的状态
main / master

# 每个方向一条分支，命名 feat/writing-<方向>
git checkout -b feat/writing-plate-cloudmap-prompt
git checkout -b feat/writing-dedup-threshold
git checkout -b feat/writing-mechanism-prompts
```

规则：
1. **一个方向一条分支**，不要把两个无关改动塞进同一分支（否则说不清是哪个改动起了作用）。
2. 主干只合并**实测有效**的改动；未通过的改动留在分支上——**负结果同样有价值**，但如实标注。
3. **负结果不要删分支**：失败的 Prompt 尝试留在分支上，别人可以少踩一次。

---

## 三、依赖分治

- **通用依赖**只加在 `requirements.txt`（运行核心）里，改动需要说明理由。
- **你自己实验用的临时依赖**只写进自己分支的说明里，**不要污染主干**。
- 想固定整个环境：`pip freeze > requirements-lock.txt`（锁版本，供复现）。
- 任何人都**不要**把 `site-packages`、`chroma_db`、`cache` 提交进 git。

---

## 四、数据不进 git

`.gitignore` 已排除：`data/papers/`、`data/processed/`、`data/chroma_db/`、`data/cache/`、`*.pdf`。

原因：语料是**受版权保护的学位论文**，且索引体积大、可重建。

**新同事怎么拿到数据和索引？**

```bash
# 1) 把 PDF 放进 data/papers/
# 2) 一键重建解析 + 分块 + 向量库
python scripts/build_index.py
```

**确需共享语料时**（例如换学科）：走网盘 / 对象存储，给出**文件清单 + MD5**，不要直接塞 git。

> `data/paper_metadata.json`（论文题录：论文名 / 作者）同样**不进 git**。

---

## 五、提交 / 合并前检查清单

- [ ] 只改了写作层文件（或已声明为契约变更）
- [ ] `python -m py_compile src/writing_agent.py` 通过
- [ ] 在 Web UI「论文写作」Tab 用**同一张图 + 同一条需求**跑通改前 / 改后两版
- [ ] PR 描述含「改了哪个节点 / 哪段 Prompt + 改前改后生成示例」
- [ ] 没有提交 `.env`、`data/`、`*.pdf`
- [ ] 契约若有变更，已更新 `docs/INTERFACES.md`

---

## 六、环境统一（避免「在我机器上能跑」）

统一约定（`scripts/bootstrap.py` 会自检）：

| 项 | 约定 |
|---|---|
| Python | **3.13**（本机为 `F:/anaconda/python.exe`，3.13.9） |
| 依赖 | `pip install -r requirements.txt` |
| 密钥 | 每人自己 `cp .env.example .env` 填 Key；`.env` 不入库 |
| 配置优先级 | **`.env` > 系统环境变量 > 代码默认值** |
| 国内镜像 | `HF_ENDPOINT=https://hf-mirror.com`（已默认） |

> 注意：**`.env` 优先于系统环境变量**是刻意设计的。否则机器上其他项目的同名 `DEEPSEEK_API_KEY` 会悄悄覆盖你的配置。
