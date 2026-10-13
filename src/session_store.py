"""
session_store.py — SQLite 对话持久化

功能：
- 创建/读取对话
- 保存每轮 user/assistant 消息
- 重启应用后恢复最近一次对话
- 会话隔离：不同 conversation_id 的消息互不干扰
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from config import DATA_DIR

DB_PATH = DATA_DIR / "session.db"


def _get_conn():
    """获取数据库连接，启用外键约束"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """初始化数据库表（幂等）"""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '新对话',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                citations_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
    finally:
        conn.close()


def create_conversation(title: str = "新对话") -> str:
    """创建新对话，返回 conversation_id"""
    conv_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now),
        )
        conn.commit()
        return conv_id
    finally:
        conn.close()


def get_latest_conversation():
    """获取最近更新的对话，返回 (id, title) 或 None"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, title FROM conversations ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return (row["id"], row["title"]) if row else None
    finally:
        conn.close()


def get_conversation(conv_id: str):
    """获取单个对话信息"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_messages(conv_id: str):
    """获取某对话的全部消息，按时间正序返回 list[dict]"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT role, content, citations_json, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()
        result = []
        for r in rows:
            result.append({
                "role": r["role"],
                "content": r["content"],
                "citations": json.loads(r["citations_json"]) if r["citations_json"] else None,
                "created_at": r["created_at"],
            })
        return result
    finally:
        conn.close()


def save_message(conv_id: str, role: str, content: str, citations=None):
    """保存一条消息，并更新对话的 updated_at"""
    now = datetime.now().isoformat(timespec="seconds")
    citations_json = json.dumps(citations, ensure_ascii=False) if citations else None
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, citations_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, role, content, citations_json, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conv_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_conversation_title(conv_id: str, title: str):
    """更新对话标题（用第一条用户问题）"""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title[:50], conv_id),
        )
        conn.commit()
    finally:
        conn.close()


def messages_to_chatbot_history(messages: list) -> list:
    """把数据库消息转换成 Gradio Chatbot 的 history 格式。

    项目当前使用 dict 格式：[{"role": "user", "content": "..."}, ...]
    """
    history = []
    for m in messages:
        history.append({"role": m["role"], "content": m["content"]})
    return history
