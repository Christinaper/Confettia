# db/guild_config.py
# 服务器配置管理：初始化、读取、写入

import json
import shutil
import logging
from pathlib import Path

log = logging.getLogger("guild_config")

GUILDS_DIR  = Path("guilds")
GLOBAL_SOUL = Path("config/soul.md")

# 不仅是初始值，也是数据结构的 Schema
DEFAULT_CONFIG = {
    "channels": {
        "chat": [],
        "rss": [], 
        "log": [],
        "topic": [], 
        "memo": [], 
        "review": [], 
        "forum": []
    },
    "rss_feeds": [
        {"name": "Anthropic Blog",
         "url":  "https://www.anthropic.com/rss.xml"},
        {"name": "The Verge AI",
         "url":  "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml"},
        {"name": "HuggingFace Blog",
         "url":  "https://huggingface.co/blog/feed.xml"},
        {"name": "OpenAI Blog",
         "url":  "https://openai.com/blog/rss.xml"},
        {"name": "DeepSeek",
         "url":  "https://api.deepseek.com/rss.xml"},
    ],
    "token_budget":    15000,
    "daily_hour_utc":  1,
    "rag_enabled":     False,
    "initialized":     False,   # /setup 完成后改为 True
}


def guild_dir(guild_id: str) -> Path:
    return GUILDS_DIR / guild_id

def db_path(guild_id: str) -> str:
    """返回该 guild 的数据库文件路径。"""
    return str(guild_dir(guild_id) / "data.db")

def bootstrap(guild_id: str) -> bool:
    """
    确保 guild 目录和最低限度配置存在。
    幂等：多次调用安全，已初始化的目录不会被覆盖。
    返回 True = 成功, False = 失败（调用方应静默跳过）
    """
    gdir = guild_dir(guild_id)
    try:
        gdir.mkdir(parents=True, exist_ok=True)

        # soul.md：复制全局模板（不存在时）
        soul_dst = gdir / "soul.md"
        if not soul_dst.exists():
            if GLOBAL_SOUL.exists():
                shutil.copy(GLOBAL_SOUL, soul_dst)
            else:
                soul_dst.write_text(
                    "你是一个友好的 AI 助手。", encoding="utf-8"
                )

        # config.json：写入默认值（如果不存在）
        cfg_path = gdir / "config.json"
        if not cfg_path.exists():
            cfg_path.write_text(
                json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        return True

    except OSError as e:
        log.error(f"Guild {guild_id} 初始化失败：{e}")
        return False


def get_config(guild_id: str) -> dict:
    """读取 guild 配置，字段不存在时 fallback 到 DEFAULT_CONFIG。"""
    cfg_path = guild_dir(guild_id) / "config.json"
    if not cfg_path.exists():
        bootstrap(guild_id)

    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        # 深度合并：确保新字段有默认值
        merged = {}
        for k, v in DEFAULT_CONFIG.items():
            if k == "channels":
                merged[k] = {**v, **data.get(k, {})}
            else:
                merged[k] = data.get(k, v)
        return merged
    except Exception as e:
        log.error(f"读取 guild {guild_id} config.json 失败：{e}")
        return DEFAULT_CONFIG.copy()


# 局部更新：支持使用解包参数（**kwargs）只修改部分字段。
def set_config(guild_id: str, **kwargs):
    """更新 guild 配置的部分字段。"""
    cfg      = get_config(guild_id)
    cfg_path = guild_dir(guild_id) / "config.json"
    for k, v in kwargs.items():
        if k == "channels" and isinstance(v, dict):
            cfg.setdefault("channels", {}).update(v)
        else:
            cfg[k] = v
    cfg_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def get_soul(guild_id: str) -> str:
    """读取 guild 专属 soul.md, 不存在则 fallback 到全局。"""
    guild_soul = guild_dir(guild_id) / "soul.md"
    if guild_soul.exists():
        return guild_soul.read_text(encoding="utf-8")
    if GLOBAL_SOUL.exists():
        return GLOBAL_SOUL.read_text(encoding="utf-8")
    return "你是一个友好的 AI 助手。"


def get_channel_role(guild_id: str, channel_id: str) -> str:
    """根据 config.json 判断频道角色，默认 mention。"""
    channels = get_config(guild_id).get("channels", {})
    for role, ids in channels.items():
        if str(channel_id) in [str(i) for i in ids]:   # ← 统一转字符串比较
            return role
    return "mention"