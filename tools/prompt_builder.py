# tools/prompt_builder.py
# 三层 System Prompt 组装器
# Layer 1: soul.md (核心人格, 启动时加载一次 )
# Layer 2: 动态上下文 (每次调用生成 )
# Layer 3: 用户记忆摘要 (从 DB 提取, 可选 )

import os
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("prompt_builder")

def _load_soul(guild_id: str = "") -> str:
    """
    按 guild_id 读取, fallback 到全局 soul.md。
    """
    from db.guild_config import get_soul
    if guild_id:
        return get_soul(guild_id)
    # 兼容旧调用
    global_soul = Path(__file__).parent.parent / "config" / "soul.md"
    if global_soul.exists():
        return global_soul.read_text(encoding="utf-8")
    return "你是一个友好的 AI 助手。"


def build_system_prompt(
    user_note: str = "",       # 用户备注 (从 DB 读取的偏好, 可选 )
    has_search: bool = False,  # 本次是否附带搜索结果
    has_rag: bool = False,
    soul_override: str = "",
    guild_id: str = ""
) -> str:
    """
    组装三层 System Prompt。

    Layer 1: soul.md 全文
    Layer 2: 动态上下文 (时间、当前任务模式)
    Layer 3: 用户偏好备注 (可选)
    """
    soul = _load_soul(guild_id)

    # Layer 2: 动态上下文
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    task_hint = (
        "本次附带了实时搜索结果, 请优先基于搜索内容回答, 必要时补充你的知识。"
        if has_search else
        "本次无搜索结果, 基于你的知识回答, 不确定时如实说明。"
    )
    layer2 = f"\n\n## Current Context\n当前时间: {now_utc}\n任务提示: {task_hint}"

    # Layer 3: 用户偏好 (如果有 )
    layer3 = f"\n\n## User Preference\n{user_note}" if user_note else ""

    return soul + layer2 + layer3


def reload_soul(guild_id: str = "") -> str:
    """
    无 cache, 直接重读文件即可, 调用方传 guild_id
    """
    result = _load_soul(guild_id)
    log.info(f"soul.md 已重载(guild={guild_id or '全局'})")
    return result