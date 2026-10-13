# 新同事上手指南（从零到跑通，照着做即可）

> 面向：**完全没用过这个项目、甚至不太熟 Python 的人**。
> 假设系统：Windows 10 / 11（Mac 的差异在文中单独标注）。
> 全程大约 30–60 分钟，其中大部分时间是**下载**（依赖约 3–4 GB）。
>
> 如果你只想最快跑起来，直接跳到 **§4 六步走**；前面的 §1–§3 是回答
> 「我电脑要装什么 / 要不要 VSCode / 要不要虚拟环境」这类问题。

---

## 1. 我电脑上需要有什么？

一张表说清楚。**只有前两项是必须的**，其余是"更好用"。

| 需要的东西 | 必需？ | 说明 |
|---|---|---|
| **Python 3.11 / 3.12 / 3.13** | ✅ 必须 | 项目在 **3.13.9** 上验证过；3.10 以下不支持 |
| **Git** | ✅ 必须 | 用来把代码拉下来、以后提交自己的改动 |
| **网络** | ✅ 必须 | 下载依赖（~3–4 GB）、下载模型（首次运行自动拉）、调用大模型 API |
| **磁盘空间 ≥ 10 GB** | ✅ 必须 | 依赖 3–4 GB + 模型 1–2 GB + 语料与索引 1–2 GB |
| 代码编辑器（VSCode / PyCharm） | ⬜ 可选 | **不装也能跑**，装了好改代码。见 §2 |
| 显卡（NVIDIA GPU） | ⬜ 可选 | 有则检索更快；**没有也能跑**（纯 CPU） |
| Anaconda / Miniconda | ⬜ 可选 | 不是必须，只是装 Python 的一种省事方式。见 §3 |

> ⚠️ **网络提示**：本项目默认把 HuggingFace 指向国内镜像 `hf-mirror.com`
> （`config.py` 里自动设置），所以"下载模型卡住"的问题通常不会出现。
> 但 GitHub 在本机可能被代理挡（见 §7 常见问题）。

---

## 2. 需要 VSCode 吗？

**不需要，但强烈建议装。** 分两种人：

### 情况 A：我只想「跑起来看看」，不改代码
**完全不需要 VSCode。** 有一个"命令提示符 / PowerShell"就够了。
本文 §4 的命令全部在命令行里敲，不涉及任何编辑器。

### 情况 B：我要改代码、做实验（大多数人的真实情况）
**建议装 VSCode**，因为：
- 找文件、搜索、看报错行都直观得多；
- 内置终端，省得在两个窗口之间切；
- 装了 **Python 扩展**后能一键选解释器、点按钮运行。

安装（3 分钟）：去 https://code.visualstudio.com 下载 Windows 版，
一路"下一步"即可。装完打开项目文件夹，按提示装 **Python** 扩展。

> **PyCharm** 也完全可以，二选一，不用都装。别装两个折腾。

---

## 3. 如果我电脑里没有 Python / Git，装什么合适？

### 3.1 Python —— 三种装法，推荐第一种

| 方式 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| **① python.org 官方安装包**（推荐） | 干净、可控、和项目验证环境一致 | 需要手动勾一个选项 | ✅ **首推** |
| ② Miniconda / Anaconda | 自带很多科学计算包，管理多环境方便 | 体积大（Anaconda 3 GB+）、有时反而干扰 | 可选 |
| ③ Microsoft Store | 最省事 | 路径特殊、偶发权限怪问题 | 不推荐 |

**用官方安装包的正确姿势**（这一步最容易出错，务必注意）：

1. 打开 https://www.python.org/downloads/windows/
2. 选 **Python 3.13.x** 的 `Windows installer (64-bit)`，下载双击
3. **第一个界面底部，一定勾上 ☑ `Add python.exe to PATH`**
   —— 不勾的话，命令行里敲 `python` 会说"不是内部或外部命令"
4. 点 `Install Now`，等它装完

装完验证 —— 打开 **PowerShell**（开始菜单搜 `PowerShell`）敲：

```powershell
python --version
```

出现 `Python 3.13.x` 就成功了。如果提示找不到命令，就是第 3 步没勾 PATH，
重新运行安装包选 `Modify` → 补勾上。

### 3.2 Git

去 https://git-scm.com/download/win 下载，一路默认下一步即可。
装完在 PowerShell 里验证：

```powershell
git --version
```

> 如果你只是**接收代码、不打算提交**，其实也可以不装 Git ——
> 让对方把仓库打包成 zip 发给你，解压即可（但以后同步更新会麻烦，仍建议装）。

### 3.3 （可选）Anaconda 是什么、要不要用

Anaconda = Python + 一堆科学计算库 + 环境管理工具。
**本项目不需要它**，用官方 Python 就够。如果你已经装了 Anaconda 也无妨，
只要确保 §4 里用的是同一个 Python 解释器即可。

---

## 4. 六步走（复制粘贴即可）

> 以下命令 **Windows PowerShell / CMD** 和 Mac 都能用，
> 只有复制 `.env` 那一步的写法不同（已分别标注）。

### 第 0 步：拿到代码

```powershell
git clone https://github.com/cyf011588/AI-Paper-Assistant.git
cd AI-Paper-Assistant
```

> 如果 GitHub 连不上，让对方把仓库打包发你，解压后 `cd` 进目录同样继续。

### 第 1 步：自检（先看缺什么）

```powershell
python scripts/bootstrap.py
```

它会逐项打印：Python 版本、依赖装没装、密钥配没配、数据有没有、
以及**下一步该敲什么命令**。第一次跑会看到一堆 `[MISS]`，正常。

### 第 2 步：装依赖（耗时最长，主要在这等）

```powershell
python -m pip install -r requirements.txt
```

- 这一步会下载约 **3–4 GB**（主要是 `torch`），**慢是正常的**，去泡杯茶。
- 如果只做**检索 / 评测 / 消融实验**、不跑网页界面和写作工作流，
  可以装更小的版本，省掉一大半体积：
  ```powershell
  python -m pip install -r requirements-experiments.txt
  ```
- 下载慢可以加国内镜像：
  ```powershell
  python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

### 第 3 步：配置你自己的 API 密钥（见 §5 详解）

```powershell
# Windows:
copy .env.example .env
# Mac / Linux:
# cp .env.example .env
```

然后用编辑器打开 `.env`，把两个 Key 填进去。**这是唯一需要你"自己准备"的地方。**

### 第 4 步：放语料 + 重建索引

把你们课题组的论文 PDF 拷进 `data/papers/` 目录，然后：

```powershell
python scripts/build_index.py
```

它会自动完成「解析 PDF → 分块 → 建向量库」，等价于在网页界面里一篇篇上传。
**所以新电脑不需要别人传给你几百 MB 的索引，自己重建即可。**

### 第 5 步：启动

```powershell
python src/app.py
```

看到类似 `Running on local URL: http://127.0.0.1:7860` 后，
浏览器打开 **http://127.0.0.1:7860** 就是问答界面了。

（`127.0.0.1` 只在本机可访问，别人打不开你的页面，安全。）

### 第 6 步：跑评测（改完任何东西都要做）

```powershell
python evaluate_v2.py
```

> ⚠️ 公开仓库**不含评测集**（它里面有论文原文）。你需要把自己的评测集
> JSON 放进 `evaluation/dataset/`，格式见该目录的 `README.md` 和
> `example_dataset.json`。没放的话脚本会提示你怎么做，不会直接崩。

---

## 5. 怎么设置自己的 API Key（重点，详解）

### 5.1 为什么需要它

本项目调用两个大模型服务，**都要你自己的密钥**：

| 用途 | 用谁 | 有没有不要钱的替代 |
|---|---|---|
| 问答生成 / 检索规划 / 评测打分 | **DeepSeek** | 有：换成本地模型，但要额外配置 |
| 问题头脑风暴（Qwen2.5-7B） | **硅基流动** | 同上 |
| 看图（Qwen-VL） | **硅基流动** | 只在上传图片时才用到 |

> 好消息：**只有实际"提问"时才消耗额度**，装环境、建索引不花钱。
> 而且 DeepSeek 与硅基流动注册都有免费额度，小规模测试基本够用。

### 5.2 拿 DeepSeek Key（约 2 分钟）

1. 打开 https://platform.deepseek.com/ ，注册/登录
2. 左侧菜单找 **API Keys**（钥匙图标）
3. 点 **创建 API Key**，起个名字（如 `paper-assistant`）
4. **立刻复制**弹出的那串 `sk-...` —— 关掉就再也看不到了

### 5.3 拿硅基流动 Key（约 2 分钟）

1. 打开 https://cloud.siliconflow.cn/ ，注册/登录
2. 左侧菜单找 **API 密钥**
3. 点 **新建 API 密钥**，复制那串 `sk-...`

### 5.4 填进 `.env`（唯一要动手的地方）

用任意文本编辑器打开项目根目录的 `.env`（VSCode / 记事本都行），
**只改这两行**，其余保持不动：

```ini
DEEPSEEK_API_KEY=sk-你复制的deepseek密钥
SILICONFLOW_API_KEY=sk-你复制的硅基流动密钥
```

保存。就完成了。

**为什么这样做是安全的**：

- `.env` 已在 `.gitignore` 里 → `git status` 永远看不到它，
  **绝不会被提交、绝不会泄露给别人**；
- 优先级是 **`.env` > 系统环境变量 > 代码默认值**：
  你在 `.env` 里写的项一定生效，不会被机器上其他项目的同名变量"顶掉"；
  没写的项才去读系统变量。所以**每人一套 `.env` 互不干扰**。

### 5.5 验证是否配好

```powershell
python scripts/bootstrap.py
```

看到这两行就是成功：

```
[ OK ] DEEPSEEK_API_KEY     = sk-xxxxx...（已配置）
[ OK ] SILICONFLOW_API_KEY  = sk-xxxxx...（已配置）
```

> **踩坑提醒**：`.env` 是由 `python-dotenv` 这个库读取的。
> 如果你**还没装依赖就先配了 .env**，密钥不会生效（因为读它的库还没装）。
> 顺序是：**先 §4 第 2 步装依赖，再配 .env**。现在 `config.py` 若检测到
> 这情况会主动打印黄色警告，见到它照做即可。

---

## 6. 要不要用虚拟环境？（建议：要）

### 6.1 虚拟环境是什么

一句话：**给这个项目单独隔离一个 Python 环境**，
它装的包和系统里其他项目互不影响。

### 6.2 要不要？

| 情况 | 建议 |
|---|---|
| 这台电脑**只有这一个 Python 项目** | 可以不建，直接装（本文 §4 的写法就是这样） |
| 电脑上还有别的项目、或想干净地重来 | **强烈建议建** |
| 多人共用一台机器 / 服务器 | **必须建** |

本项目**不强依赖**虚拟环境（不像有些项目不建就跑不了），
但建了更省心：装坏了删掉 `.venv` 文件夹重来即可，不污染系统。

### 6.3 怎么建（3 条命令）

```powershell
# 1. 在项目根目录创建虚拟环境（会生成一个 .venv 文件夹）
python -m venv .venv

# 2. 激活它
#    Windows PowerShell:
.\.venv\Scripts\Activate.ps1
#    Windows CMD:
# .venv\Scripts\activate.bat
#    Mac / Linux:
# source .venv/bin/activate

# 3. 之后所有 pip / python 命令都作用在这个环境里
python -m pip install -r requirements.txt
python scripts/bootstrap.py
```

激活成功后，命令行前面会多出 `(.venv)` 字样。
**每次新开一个终端窗口，都要重新执行第 2 步激活。**

> PowerShell 若报"禁止运行脚本"，执行一次：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，选 `Y` 即可。

### 6.4 关闭 / 删除

- 关闭：敲 `deactivate`
- 彻底重来：直接删掉项目里的 `.venv` 文件夹，重新执行 6.3

> `.venv` 也已在 `.gitignore` 中，不会被提交，不必担心。

---

## 7. 常见问题速查

| 现象 | 原因 | 解决 |
|---|---|---|
| `python : 不是内部或外部命令` | 装 Python 时没勾 PATH | 重跑安装包 → Modify → 勾上 Add to PATH |
| `pip install` 卡在下载 torch | 网络慢 | 加清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 改了 `.env` 但密钥不生效 | 依赖没装 / 不在项目根目录运行 | 先装依赖（`python-dotenv` 在其中）；确认当前目录是项目根目录 |
| 知识库是空的 / 提问答不上来 | 没放语料或没建索引 | 放 PDF 进 `data/papers/`，跑 `python scripts/build_index.py` |
| 首次运行卡在"下载模型" | 在从 HuggingFace 拉 embedding 模型 | 已默认走国内镜像；耐心等第一次，之后有缓存 |
| `evaluate_v2.py` 报找不到评测集 | 公开仓库不含评测集 | 把自己的评测集放进 `evaluation/dataset/`，见该目录 README |
| `git clone` 连不上 | 网络 / 代理限制 | 让对方打包 zip 发你，或走公司代理 |
| 端口 7860 被占用 | 已有实例在跑 | 换端口：`$env:APP_PORT=7861; python src/app.py` |
| 没有显卡能跑吗 | — | 能。纯 CPU 检索会慢些，功能完整 |

---

## 8. 配好之后，我该改哪里？

环境只是入场券。**改动范围**见 `docs/INTERFACES.md`（五层划分 + 数据契约）
与 `docs/COLLABORATION.md`（分工与评测纪律）。一句话版：

| 我想优化… | 只改这些文件 |
|---|---|
| PDF 解析 / 分块 | `src/pdf_parser.py`、`src/chunker.py` |
| 换 embedding 模型 | `src/vector_store.py` + `.env` 里的 `EMBEDDING_MODEL` |
| 召回 / 精排 / 查询改写 | `src/smart_retriever.py`、`src/reranker.py`、`src/query_understanding.py` |
| 生成回答 / 写作 Agent / 看图 | `src/generator.py`、`src/writing_agent.py`、`src/vision_extractor.py` |
| 网页界面 | `src/app.py` |

**任何优化都必须跑 `python evaluate_v2.py`（同一套评测），否则结果不认。**

---

## 9. 一页速查（贴墙上）

```powershell
# 一次性准备
git clone https://github.com/cyf011588/AI-Paper-Assistant.git
cd AI-Paper-Assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env          # 然后编辑 .env 填两个 Key

# 日常使用
python scripts/bootstrap.py     # 自检，缺什么照它说的做
python scripts/build_index.py   # 放了新 PDF 之后重建索引
python src/app.py               # 启动网页界面
python evaluate_v2.py           # 改完代码跑评测

# 提交自己的改动
git checkout -b feat/我的方向
git add -A
git commit -m "feat: 我做了什么"
git push
```
