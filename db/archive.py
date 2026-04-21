# db/archive.py
# JSONL 全量归档：每条消息永久追加，不压缩不删除
#
# 文件结构：
#   archive/
#   ├── {user_id}_{channel_id}.jsonl   对话归档（按用户+频道）
#   └── system.jsonl                   系统事件（压缩记录、错误等）
#
# 每行格式：
#   {"ts": "ISO8601", "type": "message", "role": "user/assistant",
#    "content": "...", "user_id": "...", "channel_id": "...",
#    "meta": {...}}

import json
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("archive")

ARCHIVE_DIR = Path("archive")


class Archiver:
    def __init__(self, archive_dir: Path = ARCHIVE_DIR):
        self.dir  = Path(archive_dir)
        self.dir.mkdir(exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, filename: str) -> threading.Lock:
        """每个文件独立锁，避免不同频道之间互相阻塞。"""
        with self._global_lock:
            if filename not in self._locks:
                self._locks[filename] = threading.Lock()
            return self._locks[filename]

    def _write(self, filename: str, record: dict):
        """追加一条记录到 JSONL 文件。"""
        path = self.dir / filename
        line = json.dumps(record, ensure_ascii=False) + "\n"
        lock = self._get_lock(filename)
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── 对话消息归档 ──────────────────────────────────────────────────────────
    def log_message(
        self,
        user_id: str,
        channel_id: str,
        role: str,
        content: str,
        meta: dict = None,
    ):
        """
        归档一条对话消息。
        meta 可以传入额外信息，例如：
          {"search_used": True, "rag_used": False, "tokens": 320}
        """
        filename = f"{user_id}_{channel_id}.jsonl"
        record = {
            "ts":         self._now(),
            "type":       "message",
            "role":       role,
            "content":    content,
            "user_id":    user_id,
            "channel_id": channel_id,
            "meta":       meta or {},
        }
        self._write(filename, record)

    # ── 压缩事件归档（最重要）────────────────────────────────────────────────
    def log_compression(
        self,
        user_id: str,
        channel_id: str,
        original_messages: list[dict],
        summary: str,
        compressed_count: int,
    ):
        """
        归档一次压缩事件：原始消息 + 生成的摘要。
        这是测试压缩质量的核心数据。
        """
        filename = f"{user_id}_{channel_id}.jsonl"
        record = {
            "ts":               self._now(),
            "type":             "compression",
            "user_id":          user_id,
            "channel_id":       channel_id,
            "compressed_count": compressed_count,
            "summary":          summary,
            # 原始消息完整保留（这是归档最重要的数据）
            "original_messages": [
                {"role": m["role"], "content": m["content"]}
                for m in original_messages
            ],
        }
        self._write(filename, record)
        log.info(f"归档压缩事件：{compressed_count} 条 → 摘要，"
                 f"文件 archive/{filename}")

    # ── 系统事件归档 ──────────────────────────────────────────────────────────
    def log_system(self, event: str, detail: dict = None):
        """
        归档系统事件：Bot 启动、重启、错误等。
        """
        record = {
            "ts":     self._now(),
            "type":   "system",
            "event":  event,
            "detail": detail or {},
        }
        self._write("system.jsonl", record)

    # ── 读取工具（供测试脚本使用）────────────────────────────────────────────
    def read_file(self, filename: str) -> list[dict]:
        """读取一个 JSONL 文件，返回记录列表。"""
        path = self.dir / filename
        if not path.exists():
            return []
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        log.warning(f"JSONL 解析失败（跳过）：{e}")
        return records

    def list_files(self) -> list[str]:
        """列出所有归档文件。"""
        return [p.name for p in self.dir.glob("*.jsonl")]

    def get_compressions(self, user_id: str, channel_id: str) -> list[dict]:
        """获取某个用户+频道的所有压缩事件。"""
        filename = f"{user_id}_{channel_id}.jsonl"
        records  = self.read_file(filename)
        return [r for r in records if r["type"] == "compression"]

    def stats(self) -> dict:
        """归档统计。"""
        files = self.list_files()
        total_messages    = 0
        total_compressions = 0
        for fname in files:
            if fname == "system.jsonl":
                continue
            for r in self.read_file(fname):
                if r["type"] == "message":
                    total_messages += 1
                elif r["type"] == "compression":
                    total_compressions += 1
        return {
            "files":        len(files),
            "messages":     total_messages,
            "compressions": total_compressions,
        }


# 全局单例
archiver = Archiver()