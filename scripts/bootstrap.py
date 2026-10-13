#!/usr/bin/env python
"""
scripts/bootstrap.py — 环境自检（新机器 / 新同事第一步）

只读检查，不修改任何东西。逐项告诉你「缺什么、下一步敲什么命令」。

用法：
    python scripts/bootstrap.py

退出码：0 = 核心环境就绪；1 = 尚有关键项缺失。
"""

import sys
import os
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

# --- 让输出在 Windows 控制台也不乱码 ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# 核心运行依赖（package -> 用途）
CORE_DEPS = {
    "pymupdf": "PDF 解析",
    "chromadb": "向量库",
    "sentence-transformers": "本地 embedding / reranker",
    "torch": "深度学习后端",
    "numpy": "数值计算",
    "rank-bm25": "BM25 召回",
    "jieba": "中文分词",
    "pandas": "数据处理",
    "openai": "大模型调用",
    "python-dotenv": "读取 .env",
    "packaging": "版本比较",
}
# 可选（只在跑 Web UI / 写作工作流 / 图像入库时才需要）
UI_DEPS = {
    "gradio": "Web 界面",
    "langgraph": "写作工作流",
    "opencv-python-headless": "TIF 图像解码",
}

OK, MISS, WARN = "[ OK ]", "[MISS]", "[WARN]"


def hr(title=""):
    print("\n" + "=" * 64)
    if title:
        print(title)
        print("=" * 64)


def check_python():
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    print(f"Python 解释器 : {sys.executable}")
    print(f"Python 版本   : {ver}")
    if (v.major, v.minor) < (3, 10):
        print(f"{MISS} 需要 Python >= 3.10（项目在本机 3.13.9 上验证）")
        return False
    if (v.major, v.minor) not in ((3, 11), (3, 12), (3, 13)):
        print(f"{WARN} 非验证版本（项目在 3.13 上验证过），若遇怪问题请换 3.13")
    else:
        print(f"{OK} 版本可用")
    return True


def check_deps():
    missing_core, missing_ui = [], []
    for pkg, why in CORE_DEPS.items():
        try:
            print(f"{OK} {pkg:<26} {version(pkg):<12} ({why})")
        except PackageNotFoundError:
            print(f"{MISS} {pkg:<26} {'--':<12} ({why})")
            missing_core.append(pkg)
    print("-" * 64)
    for pkg, why in UI_DEPS.items():
        try:
            print(f"{OK} {pkg:<26} {version(pkg):<12} ({why})")
        except PackageNotFoundError:
            print(f"{WARN} {pkg:<26} {'--':<12} ({why})  <- 仅 Web UI / 写作需要")
            missing_ui.append(pkg)
    return missing_core, missing_ui


def check_config():
    """导入 config 并检查关键项。config 会被其他模块自动加载。"""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    try:
        import config as c
    except Exception as e:
        print(f"{MISS} 无法导入 src/config.py: {e}")
        return None
    env_file = ROOT / ".env"
    if env_file.exists():
        print(f"{OK} 找到 .env（密钥来源；该文件不入 git）")
    else:
        print(f"{WARN} 没有 .env —— 将使用代码默认值（无密钥时无法调用大模型）")
        print("       解决：复制 .env.example 为 .env，并填入自己的 API Key")

    keys_ok = True
    for name in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY"):
        val = getattr(c, name, "")
        if val:
            print(f"{OK} {name:<20} = {val[:6]}...（已配置）")
        else:
            print(f"{MISS} {name:<20} 未配置")
            keys_ok = False
    return c if keys_ok else None


def check_data(cfg):
    papers = sorted((ROOT / "data" / "papers").glob("*.pdf")) if (ROOT / "data" / "papers").exists() else []
    chunks = sorted((ROOT / "data" / "processed").glob("*_chunks.json")) if (ROOT / "data" / "processed").exists() else []
    chroma = ROOT / "data" / "chroma_db" / "chroma.sqlite3"
    dataset = None
    if cfg is not None:
        dataset = Path(cfg.EVAL_DATASET_PATH)

    print(f"{'[ OK ]' if papers else '[MISS]'} 语料 PDF      : {len(papers)} 篇  (data/papers/)")
    print(f"{'[ OK ]' if chunks else '[MISS]'} 解析分块      : {len(chunks)} 份  (data/processed/*_chunks.json)")
    print(f"{'[ OK ]' if chroma.exists() else '[MISS]'} 向量库        : {'已建立' if chroma.exists() else '未建立'}  (data/chroma_db/)")
    if dataset is not None:
        if dataset.exists():
            print(f"{OK} 评测集(统一裁判): 已就位  {dataset.name}")
        else:
            print("[INFO] 评测集(统一裁判): 尚未自备")
            print("       公开仓库不发布评测集（含论文原文）。把自己的评测集 JSON 放进")
            print("       evaluation/dataset/ 即可，格式见该目录 README.md 与 example_dataset.json")
    return bool(papers), bool(chunks), chroma.exists()


def next_steps(cfg_ok, missing_core, papers_ok, chunks_ok, chroma_ok):
    hr("下一步")
    steps = []
    if missing_core:
        steps.append("1) 安装依赖：        pip install -r requirements.txt")
        steps.append("   （只做检索/评测：  pip install -r requirements-experiments.txt）")
    if not cfg_ok:
        steps.append("2) 配置密钥：        copy .env.example .env  然后用编辑器填入自己的 API Key")
    if not papers_ok:
        steps.append("3) 放语料：          把论文 PDF 拷进 data/papers/")
    if not (chunks_ok and chroma_ok):
        steps.append("4) 重建索引：        python scripts/build_index.py")
    steps.append("5) 启动问答界面：    python src/app.py   （浏览器打开 http://127.0.0.1:7860）")
    steps.append("6) 跑统一评测：      python evaluate_v2.py")
    steps.append("   （需先自备评测集：放进 evaluation/dataset/，见该目录 README.md）")

    for s in steps:
        print("   " + s)

    ready = cfg_ok and not missing_core and papers_ok
    print()
    if ready:
        print(f"{OK} 核心环境就绪。协作规范见 docs/COLLABORATION.md；分发方式见 docs/DISTRIBUTION.md")
        return 0
    print(f"{WARN} 还有关键项缺失，请按上面步骤补齐后再启动。")
    return 1


def main():
    hr("AI Paper Assistant — 环境自检 (bootstrap)")
    print(f"仓库根目录 : {ROOT}")
    print()
    py_ok = check_python()

    hr("依赖")
    missing_core, missing_ui = check_deps()

    hr("配置与密钥")
    cfg = check_config()

    hr("数据与索引")
    papers_ok, chunks_ok, chroma_ok = check_data(cfg)

    rc = next_steps(py_ok and cfg is not None, missing_core, papers_ok, chunks_ok, chroma_ok)
    return rc


if __name__ == "__main__":
    sys.exit(main())
