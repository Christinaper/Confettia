# db/memory.py
# 对话记忆 + 用户偏好 + 三层压缩
# 使用 SQLite 存储对话历史 (单文件数据库, 无需安装额外软件 )

import sqlite3
import threading
import logging
from datetime import datetime, timezone

log = logging.getLogger("memory")

# 压缩阈值
RECENT_KEEP    = 10   # 最近 N 条原文保留
COMPRESS_AFTER = 20   # 超过 N 条时触发压缩


class ConversationMemory:
    def __init__(self, guild_id: str = ""):
        from db.guild_config import db_path, bootstrap
        if guild_id:
            bootstrap(guild_id)
            self.db_path = db_path(guild_id)
        else:
            # 兼容模式：未传 guild_id 时用旧路径（迁移过渡期使用）
            self.db_path = "agent_memory.bd"
        # 线程锁: 防止多用户同时操作数据库时发生冲突
        self.lock        = threading.Lock()
        self._init_db()

    def _get_conn(self):
        """每次获取独立连接(SQLite 线程安全要求)"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        """初始化数据库表结构"""
        with self._get_conn() as conn:
            # 对话历史表
            conn.execute("PRAGMA journal_mode=WAL;")   # ← 加这一行
            conn.execute("PRAGMA synchronous=NORMAL;") # ← 搭配使用，性能更好
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    compressed INTEGER DEFAULT 0,
                    created    TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    summary    TEXT NOT NULL,
                    covers_ids TEXT NOT NULL,
                    created    TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_notes (
                    user_id TEXT PRIMARY KEY,
                    note    TEXT NOT NULL,
                    updated TEXT NOT NULL
                )
            """)
            # 索引加速查询
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_user_chan
                ON messages(user_id, channel_id, id)
            """)
            conn.commit()

    # ── 保存消息 ─────────────────────────────────────────────────────────────
    def save(self, user_id: str, channel_id: str, role: str, content: str):
        now = datetime.now(timezone.utc).isoformat()
        with self.lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO messages (user_id,channel_id,role,content,created) "
                    "VALUES (?,?,?,?,?)",
                    (user_id, channel_id, role, content, now)
                )
                conn.commit()

    # ── 读取历史（三层结构）──────────────────────────────────────────────────
    def get_history(self, user_id: str, channel_id: str) -> list[dict]:
        """
        返回供 LLM 使用的消息列表：
        Layer 1: 最近 RECENT_KEEP 条原文
        Layer 2: 更早的内容用摘要替代（如果有）
        """
        with self._get_conn() as conn:
            # 最近 N 条原文
            cur = conn.execute(
                """SELECT role, content FROM messages
                   WHERE user_id=? AND channel_id=? AND compressed=0
                   ORDER BY id DESC LIMIT ?""",
                (user_id, channel_id, RECENT_KEEP)
            )
            recent = [{"role": r[0], "content": r[1]}
                      for r in reversed(cur.fetchall())]

            # 最近一条摘要（如果有）
            cur = conn.execute(
                """SELECT summary FROM summaries
                   WHERE user_id=? AND channel_id=?
                   ORDER BY id DESC LIMIT 1""",
                (user_id, channel_id)
            )
            row = cur.fetchone()

        history = []
        if row:
            # 把摘要作为 system 级别的上下文前置
            history.append({
                "role":    "user",
                "content": f"[之前对话摘要]\n{row[0]}"
            })
            history.append({
                "role":    "assistant",
                "content": "好的，我已了解之前的对话背景。"
            })

        history.extend(recent)
        return history

    def count_uncompressed(self, user_id: str, channel_id: str) -> int:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM messages "
                "WHERE user_id=? AND channel_id=? AND compressed=0",
                (user_id, channel_id)
            )
            return cur.fetchone()[0]

    def get_uncompressed(self, user_id: str, channel_id: str) -> list[dict]:
        """获取所有未压缩的消息（用于生成摘要）。"""
        with self._get_conn() as conn:
            cur = conn.execute(
                """SELECT id, role, content FROM messages
                   WHERE user_id=? AND channel_id=? AND compressed=0
                   ORDER BY id ASC""",
                (user_id, channel_id)
            )
            return [{"id": r[0], "role": r[1], "content": r[2]}
                    for r in cur.fetchall()]

    def compress(self, user_id: str, channel_id: str, summary: str,
                 message_ids: list[int], keep_recent: int = RECENT_KEEP):
        """
        把旧消息标记为 compressed，存入摘要。
        保留最近 keep_recent 条不压缩。
        """
        if not message_ids:
            return
        # 只压缩较早的，保留最近 N 条
        ids_to_compress = message_ids[:-keep_recent] if len(message_ids) > keep_recent \
                          else []
        if not ids_to_compress:
            return

        now = datetime.now(timezone.utc).isoformat()
        covers = ",".join(str(i) for i in ids_to_compress)

        with self.lock:
            with self._get_conn() as conn:
                placeholders = ",".join("?" * len(ids_to_compress))
                conn.execute(
                    f"UPDATE messages SET compressed=1 "
                    f"WHERE id IN ({placeholders})",
                    ids_to_compress
                )
                conn.execute(
                    "INSERT INTO summaries (user_id,channel_id,summary,covers_ids,created) "
                    "VALUES (?,?,?,?,?)",
                    (user_id, channel_id, summary, covers, now)
                )
                conn.commit()
        log.info(f"压缩完成：{len(ids_to_compress)} 条 → 摘要，"
                 f"保留最近 {keep_recent} 条原文")

    def clear(self, user_id: str, channel_id: str):
        with self.lock:
            with self._get_conn() as conn:
                conn.execute(
                    "DELETE FROM messages WHERE user_id=? AND channel_id=?",
                    (user_id, channel_id)
                )
                conn.execute(
                    "DELETE FROM summaries WHERE user_id=? AND channel_id=?",
                    (user_id, channel_id)
                )
                conn.commit()

    # ── 用户偏好（Layer 3）────────────────────────────────────────────────────
    def get_user_note(self, user_id: str) -> str:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT note FROM user_notes WHERE user_id=?", (user_id,)
            )
            row = cur.fetchone()
            return row[0] if row else ""

    def set_user_note(self, user_id: str, note: str):
        now = datetime.now(timezone.utc).isoformat()
        with self.lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO user_notes (user_id,note,updated) VALUES (?,?,?) "
                    "ON CONFLICT(user_id) DO UPDATE SET note=excluded.note, "
                    "updated=excluded.updated",
                    (user_id, note, now)
                )
                conn.commit()

    def clear_user_note(self, user_id: str):
        with self.lock:
            with self._get_conn() as conn:
                conn.execute(
                    "DELETE FROM user_notes WHERE user_id=?", (user_id,)
                )
                conn.commit()