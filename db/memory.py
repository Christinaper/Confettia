# db/memory.py
# 对话记忆 + 用户偏好存储
# 使用 SQLite 存储对话历史 (单文件数据库, 无需安装额外软件 )

import sqlite3
import threading
from datetime import datetime


class ConversationMemory:
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        # 线程锁: 防止多用户同时操作数据库时发生冲突
        self.lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        """每次获取独立连接(SQLite 线程安全要求)"""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """初始化数据库表结构"""
        with self._get_conn() as conn:
            # 对话历史表
            conn.execute("PRAGMA journal_mode=WAL;")   # ← 加这一行
            conn.execute("PRAGMA synchronous=NORMAL;") # ← 搭配使用，性能更好
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   TEXT NOT NULL,
                    channel   TEXT NOT NULL,
                    role      TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    created   TEXT NOT NULL
                )
            """)
            # 建立索引, 加速按用户+频道查询
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_channel
                ON messages (user_id, channel)
            """)
            # 用户偏好表 (Layer 3 记忆 )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_notes (
                    user_id   TEXT PRIMARY KEY,
                    note      TEXT NOT NULL,
                    updated   TEXT NOT NULL
                )
            """)
            conn.commit()

    # ── 对话历史 ──────────────────────────────────────────────────────────
    def save(self, user_id: str, channel: str, role: str, content: str):
        """保存一条消息记录"""
        with self.lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO messages VALUES (NULL,?,?,?,?,?)",
                    (user_id, channel, role, content, datetime.now().isoformat())
                )
                conn.commit()

    def get_history(self, user_id: str, channel: str, limit: int = 8) -> list:
        """
        获取最近 N 条历史, 返回格式符合 LLM API 要求: 
        [{"role": "user", "parts": ["消息内容"]}, ...]
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                """SELECT role, content FROM messages
                   WHERE user_id=? AND channel=?
                   ORDER BY id DESC LIMIT ?""",
                (user_id, channel, limit)
            )
            rows = cursor.fetchall()
            # 时序反转: 最旧的消息在最前面
        return [{"role": r[0], "parts": [r[1]]} for r in reversed(rows)]

    def clear(self, user_id: str, channel: str):
        """清除指定用户在某频道的所有历史"""
        with self.lock:
            with self._get_conn() as conn:
                conn.execute(
                    "DELETE FROM messages WHERE user_id=? AND channel=?",
                    (user_id, channel)
                )
                conn.commit()

    def count(self, user_id: str, channel: str) -> int:
        """统计消息条数 (用于 /status 命令 )"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE user_id=? AND channel=?",
                (user_id, channel)
            )
            return cursor.fetchone()[0]

    # ── 用户偏好 (Layer 3 )────────────────────────────────────────────────
    def set_user_note(self, user_id: str, note: str):
        """保存用户偏好备注 (会注入 System Prompt )。"""
        with self.lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO user_notes VALUES (?,?,?)
                       ON CONFLICT(user_id) DO UPDATE SET note=?, updated=?""",
                    (user_id, note, datetime.now().isoformat(),
                     note, datetime.now().isoformat())
                )
                conn.commit()

    def get_user_note(self, user_id: str) -> str:
        """获取用户偏好, 不存在时返回空字符串。"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT note FROM user_notes WHERE user_id=?", (user_id,)
            )
            row = cursor.fetchone()
        return row[0] if row else ""

    def clear_user_note(self, user_id: str):
        with self.lock:
            with self._get_conn() as conn:
                conn.execute(
                    "DELETE FROM user_notes WHERE user_id=?", (user_id,)
                )
                conn.commit()