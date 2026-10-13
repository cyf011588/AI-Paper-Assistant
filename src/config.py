"""
config.py — 全局配置

设计原则（便于「多机器、多人、各自方向」协作）：
1. 密钥与可调参数从 .env / 环境变量读取，不再硬编码在源码里。
   → 每个成员在自己机器上复制 .env.example 为 .env 填自己的 Key（.env 不入 git）。
2. 所有默认值与原版**完全一致**，因此「不配置 .env」时行为不变。
3. 想切换模型 / 检索策略时，只改 .env，不必改代码（不同方向可各自一套 .env）。
"""

import os
from pathlib import Path

# =========================
# 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"          # 放 PDF 文件
PROCESSED_DIR = DATA_DIR / "processed"    # 放解析后的 JSON
DB_DIR = DATA_DIR / "chroma_db"           # 向量数据库存储位置
CACHE_DIR = DATA_DIR / "cache"            # embedding向量缓存
METADATA_PATH = DATA_DIR / "paper_metadata.json"  # 论文元数据
EVAL_DIR = BASE_DIR / "evaluation"        # 评测产物目录
DATASET_PATH = EVAL_DIR / "dataset"       # 评测集目录（公开仓库不含评测集，各方向在本地自建）


def _resolve_eval_dataset() -> Path:
    """定位评测集。

    公开仓库**不发布评测集**（它含论文原文与衍生内容）。每位成员把自己的评测集
    放进 evaluation/dataset/ 即可，文件名随意；格式见 evaluation/dataset/README.md
    与其中的 example_dataset.json。

    优先级：仓库内 evaluation/dataset/*.json（排除 example）  >  历史机器上的绝对路径。
    """
    if DATASET_PATH.exists():
        found = sorted(p for p in DATASET_PATH.glob("*.json")
                       if p.name != "example_dataset.json")
        if found:
            return found[0]
    legacy = Path(r"F:\RAG评估\ai-paper-pinggu\RAG分层评测数据集_雷杰鑫与王鑫涛大论文.json")
    return legacy if legacy.exists() else (DATASET_PATH / "your_dataset.json")


EVAL_DATASET_PATH = _resolve_eval_dataset()
EVAL_DATASET_AVAILABLE = EVAL_DATASET_PATH.exists()


def require_eval_dataset() -> Path:
    """返回评测集路径；缺失时抛出带操作指引的异常。

    评测脚本统一用这个函数取数据集，避免 clone 下来直接 FileNotFoundError。
    """
    if EVAL_DATASET_AVAILABLE:
        return EVAL_DATASET_PATH
    raise FileNotFoundError(
        "\n[评测集缺失] 公开仓库不发布评测集（含论文原文）。\n"
        f"  请把自己的评测集 JSON 放到：{DATASET_PATH}\n"
        "  字段格式见 evaluation/dataset/README.md 与 example_dataset.json\n"
        "  文件名任意（*.json），脚本会自动选取。\n"
    )

# =========================
# 读取 .env（可选）
# 未安装 python-dotenv 或没有 .env 时，自动回退到系统环境变量 / 默认值
#
# 优先级：.env 中的项 > 系统环境变量 > 代码默认值
# （用 override=True，保证「每个人自己的 .env」自洽，不被机器上其他项目的
#   同名环境变量意外遮蔽；若某项没写进 .env，则仍会读系统环境变量）
# =========================
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=True)
except ImportError:
    # 没装 python-dotenv：如果存在 .env 却读不到，明确提示，避免「填了 Key 却不生效」的困惑
    if (BASE_DIR / ".env").exists():
        print("[config] 警告：检测到 .env，但未安装 python-dotenv，配置不会被读取。\n"
              "         请先安装依赖：pip install -r requirements.txt")

# HuggingFace 国内镜像（如需更换，在 .env 里设 HF_ENDPOINT 即可）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _env(name: str, default: str = "") -> str:
    """读环境变量，缺省或空串时取默认值"""
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


# =========================
# 大模型 API 配置（从 .env 读取）
# =========================
# 模型分工：
#   - 问答/检索/规划 → DeepSeek（deepseek-chat）
#   - 问题头脑风暴/文本扩展 → Qwen2.5-7B（便宜快）
#   - 看图/图表描述 → Qwen-VL（视觉理解）
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "deepseek-chat")

SILICONFLOW_API_KEY = _env("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = _env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
TEXT_MODEL_NAME = _env("TEXT_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")      # 文本：问题头脑风暴
VL_MODEL_NAME = _env("VL_MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct")    # 视觉：图表描述

_MISSING_KEYS = [k for k in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY") if not _env(k)]
if _MISSING_KEYS:
    print(f"[config] 提示：未配置 {'、'.join(_MISSING_KEYS)}。"
          f"请复制 .env.example 为 .env 并填入自己的 API Key。")

# =========================
# 解析参数
# =========================
HEADER_HEIGHT_RATIO = _env_float("HEADER_HEIGHT_RATIO", 0.08)   # 页面顶部 8% 以内算页眉
FOOTER_HEIGHT_RATIO = _env_float("FOOTER_HEIGHT_RATIO", 0.92)   # 页面底部 8% 以内算页脚
SMALL_FONT_SIZE = _env_int("SMALL_FONT_SIZE", 7)                # 小于 7pt 算页眉/页脚
MAX_HEADING_LEN = _env_int("MAX_HEADING_LEN", 100)              # 标题文字最长字符数

# =========================
# Chunking 参数
# =========================
CHUNK_SIZE = _env_int("CHUNK_SIZE", 500)       # 每个 chunk 大约多少字符
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 80)  # 相邻 chunk 重叠字符数

# =========================
# 检索参数
# =========================
TOP_K = _env_int("TOP_K", 5)              # 最终返回给LLM的chunk数量
RECALL_K = _env_int("RECALL_K", 100)      # 第一阶段召回候选数（喂给reranker）

EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
RERANKER_MODEL = _env("RERANKER_MODEL", "BAAI/bge-reranker-base")
