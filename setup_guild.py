#!/usr/bin/env python3
# setup_guild.py
# 运行一次，为你的服务器生成 guilds/{id}/config.json
#
# 用法：
#   python setup_guild.py
#
# 运行后会在 guilds/{guild_id}/ 目录生成：
#   config.json  （频道配置）
#   soul.md      （从 config/soul.md 复制）
#   data.db      （首次对话时自动创建）

from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 在这里填入你的服务器和频道 ID
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GUILD_ID = "1299042664557183026"

CHANNELS = {
    # 无需 @，自由对话频道
    "chat":   ["1299042665064829014"],

    # RSS 推送频道（只写推送，忽略用户消息）
    "rss":    ["1479894193340944444"],

    # Bot 状态日志频道
    "log":    ["1479894565875093624"],

    # 含 URL 自动展开总结
    "topic":  [],

    # 静默存储碎片想法
    "memo":   ["1480942784880840795"],

    # 代码/长文自动点评
    "review": [],

    # Discord 论坛频道（可选）
    "forum":  ["1480928796474474516"],
}

# RSS 订阅源（保留默认即可，或自定义）
RSS_FEEDS = [
    {"name": "Anthropic Blog",  "url": "https://www.anthropic.com/rss.xml"},
    {"name": "The Verge AI",    "url": "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml"},
    {"name": "HuggingFace Blog","url": "https://huggingface.co/blog/feed.xml"},
    {"name": "OpenAI Blog",     "url": "https://openai.com/blog/rss.xml"},
]

TOKEN_BUDGET   = 15000
DAILY_HOUR_UTC = 1       # 早报时间，UTC 01:00 = 北京时间 09:00
RAG_ENABLED    = False   # 开启 RAG 需要先跑 python build_index.py

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import sys
import json

def main():
    if GUILD_ID == "在这里填入你的服务器ID":
        print("❌ 请先编辑 setup_guild.py，填入你的服务器 ID 和频道 ID")
        sys.exit(1)

    # 检查是否还有未填的频道 ID
    for role, ids in CHANNELS.items():
        for cid in ids:
            if "填入" in str(cid):
                print(f"❌ {role} 频道 ID 未填入，请编辑 setup_guild.py")
                sys.exit(1)

    from db.guild_config import bootstrap, set_config, guild_dir
    bootstrap(GUILD_ID)

    set_config(
        GUILD_ID,
        channels=CHANNELS,
        rss_feeds=RSS_FEEDS,
        token_budget=TOKEN_BUDGET,
        daily_hour_utc=DAILY_HOUR_UTC,
        rag_enabled=RAG_ENABLED,
        initialized=True,
    )

    cfg_path = guild_dir(GUILD_ID) / "config.json"
    soul_path = guild_dir(GUILD_ID) / "soul.md"

    print(f"✅ Guild {GUILD_ID} 配置完成")
    print(f"   config.json → {cfg_path}")
    print(f"   soul.md     → {soul_path}")
    print()
    print("频道配置：")
    for role, ids in CHANNELS.items():
        if ids:
            print(f"  {role:8s} → {', '.join(str(i) for i in ids)}")
    print()
    print("下一步：")
    print("  1. 把 db/guild_config.py 和本文件复制到项目目录")
    print("  2. sudo systemctl restart discord-bot")
    print("  3. 验证：在 Discord 发送消息，确认频道路由正常")

if __name__ == "__main__":
    main()