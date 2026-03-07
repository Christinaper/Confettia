# tools/prompt_builder.py
# 三层 System Prompt 组装器
# Layer 1: soul.md (核心人格, 启动时加载一次 )
# Layer 2: 动态上下文 (每次调用生成 )
# Layer 3: 用户记忆摘要 (从 DB 提取, 可选 )

import os
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("prompt_builder")

SOUL_PATH = Path(__file__).parent.parent / "config" / "soul.md"


@lru_cache(maxsize=1)
def _load_soul() -> str:
    """
    读取 soul.md, 结果缓存在内存里。
    lru_cache 保证文件只读一次, 热更新需重启 (符合预期 )。
    """
    if not SOUL_PATH.exists():
        log.warning(f"soul.md 不存在: {SOUL_PATH}, 使用最小化默认值")
        return "你是一个专业的 Discord 信息助手, 用 Markdown 格式简洁回复。"
    content = SOUL_PATH.read_text(encoding="utf-8")
    log.info(f"soul.md 已加载 ({len(content)} 字符 )")
    return content


def build_system_prompt(
    user_note: str = "",       # 用户备注 (从 DB 读取的偏好, 可选 )
    has_search: bool = False,  # 本次是否附带搜索结果
) -> str:
    """
    组装三层 System Prompt。

    Layer 1: soul.md 全文
    Layer 2: 动态上下文 (时间、当前任务模式)
    Layer 3: 用户偏好备注 (可选)
    """
    soul = _load_soul()

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


def reload_soul() -> str:
    """
    强制重新加载 soul.md (清除 lru_cache)。
    供 /reload 管理员命令调用。
    """
    _load_soul.cache_clear()
    result = _load_soul()
    log.info("soul.md 已热重载")
    return result