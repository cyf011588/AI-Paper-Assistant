# 分发方案（DISTRIBUTION）

> 回答一个问题：**「做完之后，怎么把这套东西给到别人？」**
> 结论先给：**代码走 Git（GitHub/Gitee），环境走脚本，数据走本地重建。不要给安装包。**

---

## 一、先分清「接收方是谁」——这决定了分发方式

| 接收方 | 诉求 | 正确方式 |
|---|---|---|
| **A. 要改代码的队友 / 协作者** | 换机器能跑、能改、能跑评测对比 | ✅ **Git 仓库**（本文主线） |
| **B. 只用不改的人（导师演示、评审、答辩）** | 双击就能看 | ✅ 在线部署链接 / Docker 镜像 / 打包 exe |
| **C. 只想要一份数据或结果** | 看数字、看报告 | ✅ 直接给 `evaluation/*.json` 与作品集 PDF |

**关键判断**：只要对方要「各自方向优化」，就**绝不能给安装包**——见下节。

---

## 二、为什么不推荐「安装包（exe）」给协作者

「打一个安装包，对方下载依赖、再按指示配置」这条路线，对**本项目**是走不通的：

1. **安装包给的是二进制，不是源码。** PyInstaller / cx_Freeze 把 Python 和代码编译成 exe，对方**改不了代码**、也**跑不了评测**——而「各自方向优化」的前提就是能改源码。
2. **体积爆炸。** `torch` + `sentence-transformers` + 模型文件动辄 **几个 GB**，塞进安装包不现实；而 `bge-base-zh`、`bge-reranker-base` 等模型其实是**运行时从 HuggingFace 下载**的，安装包管不到这一步。
3. **环境无法通吃。** 有人有 CUDA、有人只有 CPU；有人 Python 3.11、有人 3.13。安装包只能锁死一种组合。
4. **改一次就要重打重发。** 你优化了检索、队友优化了 Prompt，安装包要重新打包、重新传，且**无法合并**——这正是版本管理的反面。
5. **「下载必要东西 + 按指示配置」这步，本质就是装依赖 + 配密钥。** 那还不如直接用 pip + .env 把这一步**脚本化**，比安装包更透明、可调试。

> 一句话：**安装包适合「交付成品」，不适合「协作开发」。**

---

## 三、推荐路线：Git 仓库 + 脚本自动化

### 3.0 现在就可以做（本仓库已经是 git 仓库）

```bash
cd F:/AI_Project/AI-Paper-Assistant

# 1) 在 GitHub 建一个「私有」仓库（Private），然后关联并推送
git remote add origin https://github.com/<你的账号>/ai-paper-assistant.git
git add -A
git commit -m "chore: 可复现化 + 协作层（依赖/配置/索引/契约/评测集）"
git push -u origin master

# 2) 在仓库 Settings → Collaborators 里把队友加成协作者
```

> 私有仓库是必须的：**语料是受版权保护的学位论文**，公开会有风险。

### 3.1 对方（新机器 / 新同事）拿到之后，只做 4 步

```bash
git clone https://github.com/<你的账号>/ai-paper-assistant.git
cd ai-paper-assistant

python scripts/bootstrap.py            # ① 自检：告诉你缺什么、下一步敲什么
pip install -r requirements.txt        # ② 装依赖（只做检索/评测可用 requirements-experiments.txt）
cp .env.example .env                   # ③ 填自己的 API Key（.env 不入库）
# 把论文 PDF 放进 data/papers/，然后
python scripts/build_index.py          # ④ 一键重建 解析→分块→向量库
python src/app.py                      # 跑起来 → http://127.0.0.1:7860
```

**核心思想：`git clone` 只拿「代码 + 文档」。环境交给脚本；论文、索引、评测集统统一在本地重建。**

---

## 四、什么东西进 git，什么不进

| 类别 | 是否进 git | 原因 / 怎么给对方 |
|---|---|---|
| 源码 `src/`、脚本、文档 | ✅ 进 | 协作的核心 |
| `requirements*.txt`、`.env.example` | ✅ 进 | 环境可复现 |
| 评测集 `evaluation/dataset/*.json` | ⛔ 不进 | 含论文原文与衍生答案。repo 只放格式模板 `example_dataset.json`；**同一份评测集走群内共享（附 MD5）** |
| `evaluation/*.json`（评测产物） | ⛔ 不进 | 含论文衍生答案，本地跑出来即可 |
| `data/paper_metadata.json` | ⛔ 不进 | 论文题录，用 Web UI 上传论文时自动生成 |
| `.env`（真实密钥） | ⛔ 不进 | 已 `.gitignore`；泄露即烧钱 |
| `data/papers/*.pdf` | ⛔ 不进 | 版权 + 体积；走网盘/对象存储给「清单+MD5」 |
| `data/processed/`、`data/chroma_db/`、`data/cache/` | ⛔ 不进 | 体积大、可脚本重建 |
| 模型文件 `*.bin/*.pt/*.pth` | ⛔ 不进 | 运行时自动下载 |

> 给语料的方式：把 PDF 打包放网盘，附一份**文件清单 + 各自的 MD5**，对方校验后放进 `data/papers/`，再 `python scripts/build_index.py`。

---

## 五、三条路线对比（给导师/队友看）

| 方式 | 适合 | 优点 | 缺点 |
|---|---|---|---|
| **Git 源码仓库**（推荐） | 要改代码的队友 | 版本管理、分支、PR、评测可对比、增量同步 | 需要装一次环境（有 bootstrap 脚本后不难） |
| **Conda 环境文件 `environment.yml`** | 环境难装（torch/CUDA）时 | 一次配好，人人一致 | 仍要下载数 GB |
| **Docker 镜像** | 「在我机器上能跑」重灾区 | 环境完全隔离、一致 | 要装 Docker；镜像体积大；GPU 需额外配置 |
| **在线部署（Web 链接）** | 演示 / 评审 / 答辩 | 零安装，点开就用 | 只能看、不能改；适合「接收方 B」 |
| **安装包（exe）** | 交付成品 | 双击即用 | **改不了、跑不了评测、体积大** —— 不适合协作 |

**推荐组合**：
`Git 私有仓库`（协作主线） ＋ `scripts/bootstrap.py`（环境自检） ＋ `可选的 Dockerfile`（给环境脏的人） ＋ `演示时用在线链接`。

---

## 六、常见坑

1. **国内访问 GitHub 慢** → 用 Gitee 私有仓库镜像，或给 git 配代理。
2. **密钥误提交** → 本项目历史上 `src/config.py` 曾是明文密钥。**若该文件曾被推送到远端，请立刻轮换 DeepSeek / 硅基流动的 Key**（改 .env 即可，无需改代码）。
3. **换机后「跑起来是空的」** → 因为 `data/` 没进 git。跑 `python scripts/build_index.py` 即可。
4. **`HF_ENDPOINT` 没设** → 模型下不动。`config.py` 已默认 `https://hf-mirror.com`。
5. **`.env` 被系统环境变量覆盖** → 本项目已把优先级定为 **`.env` > 系统变量 > 默认值**，避免机器上其他项目的同名 Key 干扰。
6. **只给一个压缩包** → 无法协作、无法追溯改动、无法对比评测。不推荐。

---

## 七、对方只是想「用」而不是「改」？

那就走另一条线，与本仓库的协作流程解耦：

```bash
# 方式一：在线部署（最省事），从本机把项目发布成一个链接
# 方式二：Docker（自己打好镜像推到 registry）
# 方式三：PyInstaller 打包成 exe（仅当对方完全不改代码）
```

这三者都只交付**成品**，不承担「各自方向优化」的职责。
