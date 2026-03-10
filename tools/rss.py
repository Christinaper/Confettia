# tools/rss.py
# RSS 监控工具 — 拉取 AI 行业博客更新
# pip install feedparser aiohttp

import asyncio
import logging
import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("rss")

# ── RSS 源配置 ────────────────────────────────────────────────────────────────
# 可以在这里自由增减，Bot 会自动轮询所有源
RSS_FEEDS = [
    {
        "name": "Anthropic Blog",
        "url":  "https://www.anthropic.com/rss.xml",
        "tag":  "🟠 Anthropic",
    },
    {
        "name": "OpenAI Blog",
        "url":  "https://openai.com/blog/rss.xml",
        "tag":  "🟢 OpenAI",
    },
    {
        "name": "DeepSeek",
        "url":  "https://api.deepseek.com/news/rss",   # 若无效会静默跳过
        "tag":  "🔵 DeepSeek",
    },
    {
        "name": "HuggingFace Blog",
        "url":  "https://huggingface.co/blog/feed.xml",
        "tag":  "🤗 HuggingFace",
    },
    {
        "name": "The Verge AI",
        "url":  "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
        "tag":  "📰 The Verge",
    },
]

MAX_ITEMS_PER_FEED = 3   # 每个源最多取几条新文章


# ── 已推送记录（SQLite，防止重复推送）────────────────────────────────────────
class SeenItems:
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rss_seen (
                    item_hash TEXT PRIMARY KEY,
                    feed_name TEXT,
                    title     TEXT,
                    seen_at   TEXT
                )
            """)
            # 只保留最近 500 条记录，防止无限增长
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS rss_seen_trim
                AFTER INSERT ON rss_seen
                BEGIN
                    DELETE FROM rss_seen
                    WHERE rowid NOT IN (
                        SELECT rowid FROM rss_seen
                        ORDER BY seen_at DESC LIMIT 500
                    );
                END
            """)
            conn.commit()

    def _hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def is_seen(self, url: str) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT 1 FROM rss_seen WHERE item_hash=?", (self._hash(url),)
            )
            return cur.fetchone() is not None

    def mark_seen(self, url: str, feed_name: str, title: str):
        with self.lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO rss_seen VALUES (?,?,?,?)",
                    (self._hash(url), feed_name, title,
                     datetime.now(timezone.utc).isoformat())
                )
                conn.commit()


_seen = SeenItems()


# ── 拉取单个 RSS 源 ───────────────────────────────────────────────────────────
async def _fetch_feed(feed: dict) -> list[dict]:
    """拉取一个 RSS 源，返回未见过的新条目列表。"""
    loop = asyncio.get_event_loop()

    def _parse():
        try:
            import feedparser
        except ImportError:
            raise ImportError("请执行: pip install feedparser")

        parsed = feedparser.parse(feed["url"])
        new_items = []
        for entry in parsed.entries[:MAX_ITEMS_PER_FEED * 2]:  # 多拉几条用于过滤
            url   = entry.get("link", "")
            title = entry.get("title", "无标题").strip()
            summary = entry.get("summary", entry.get("description", "")).strip()

            # 清理 HTML 标签（简单版）
            import re
            summary = re.sub(r"<[^>]+>", "", summary)
            summary = summary[:200].strip()

            if not url or _seen.is_seen(url):
                continue

            new_items.append({
                "feed_name": feed["name"],
                "tag":       feed["tag"],
                "title":     title,
                "url":       url,
                "summary":   summary,
            })

            if len(new_items) >= MAX_ITEMS_PER_FEED:
                break

        return new_items

    try:
        items = await loop.run_in_executor(None, _parse)
        log.info(f"RSS [{feed['name']}]: {len(items)} 条新内容")
        return items
    except Exception as e:
        log.warning(f"RSS [{feed['name']}] 拉取失败: {e}")
        return []


# ── 拉取所有源 ────────────────────────────────────────────────────────────────
async def fetch_all_feeds() -> list[dict]:
    """并行拉取所有 RSS 源，返回所有新条目。"""
    tasks = [_fetch_feed(feed) for feed in RSS_FEEDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items = []
    for result in results:
        if isinstance(result, list):
            all_items.extend(result)

    return all_items


def mark_items_seen(items: list[dict]):
    """批量标记为已推送。"""
    for item in items:
        _seen.mark_seen(item["url"], item["feed_name"], item["title"])


def format_digest(items: list[dict], summary_by_llm: str = "",
                  now_str: str = "") -> str:
    """
    格式化推送消息。
    now_str: 北京时间字符串，由调用方传入保证一致性。
    """
    if not items:
        return ""

    if not now_str:
        now_str = (datetime.now(timezone.utc) +
                   __import__('datetime').timedelta(hours=8)
                   ).strftime("%m/%d %H:%M")

    lines = [f"## 📡 AI 动态速报 `{now_str} BJT`\n"]

    if summary_by_llm:
        lines.append(f"> {summary_by_llm}\n")

    by_feed: dict[str, list] = {}
    for item in items:
        by_feed.setdefault(item["tag"], []).append(item)

    for tag, feed_items in by_feed.items():
        lines.append(f"**{tag}**")
        for item in feed_items:
            lines.append(f"• [{item['title']}](<{item['url']}>)")
            if item["summary"]:
                lines.append(f"  > {item['summary'][:120]}…")
        lines.append("")

    return "\n".join(lines)