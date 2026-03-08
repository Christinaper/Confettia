# tools/memo.py
# Memo Skill：存储碎片想法，每周整理

import sqlite3
import threading
import logging
from datetime import datetime, timezone

log = logging.getLogger("memo")


class MemoStore:
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memos (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    tags      TEXT DEFAULT '',
                    created   TEXT NOT NULL,
                    sent      INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def save(self, user_id: str, content: str) -> int:
        """存入一条 memo，返回当前用户未发送的总条数。"""
        now = datetime.now(timezone.utc).isoformat()
        with self.lock:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO memos (user_id, content, created) VALUES (?,?,?)",
                    (user_id, content, now)
                )
                conn.commit()
        return self.count_unsent(user_id)

    def get_unsent(self, user_id: str) -> list[dict]:
        """获取该用户所有未发送的 memo。"""
        with self._conn() as conn:
            cur = conn.execute(
                """SELECT id, content, created FROM memos
                   WHERE user_id=? AND sent=0
                   ORDER BY created ASC""",
                (user_id,)
            )
            return [{"id": r[0], "content": r[1], "created": r[2]}
                    for r in cur.fetchall()]

    def mark_sent(self, user_id: str):
        """标记该用户所有 memo 为已发送。"""
        with self.lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE memos SET sent=1 WHERE user_id=? AND sent=0",
                    (user_id,)
                )
                conn.commit()

    def count_unsent(self, user_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM memos WHERE user_id=? AND sent=0",
                (user_id,)
            )
            return cur.fetchone()[0]


# 全局单例
memo_store = MemoStore()


def format_weekly_digest(user_id: str, llm_summary: str, items: list[dict]) -> str:
    """格式化周报消息。"""
    count = len(items)
    date_range = ""
    if items:
        first = items[0]["created"][:10]
        last  = items[-1]["created"][:10]
        date_range = f"{first} ~ {last}"

    lines = [
        f"📓 **本周 Memo 周报** `{date_range}`  共 {count} 条\n",
        f"> {llm_summary}\n",
        "---",
    ]
    for i, item in enumerate(items, 1):
        # 长内容截断显示
        content = item["content"]
        if len(content) > 100:
            content = content[:100] + "…"
        time_str = item["created"][11:16] + " UTC"
        lines.append(f"`{i}.` {content}  _{time_str}_")

    return "\n".join(lines)
    