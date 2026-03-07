# tools/scheduler.py
# 定时任务管理 — RSS 推送 + 每日早报
# pip install apscheduler

import logging
import os
from datetime import datetime, timezone

log = logging.getLogger("scheduler")

# 推送频道 ID（从 .env 读取）
DIGEST_CHANNEL_ID = int(os.getenv("DIGEST_CHANNEL_ID", "0"))
# 每日早报时间（UTC，北京时间 = UTC+8，早上9点 = UTC 01:00）
DAILY_HOUR_UTC    = int(os.getenv("DAILY_HOUR_UTC", "1"))


def setup_scheduler(bot, call_llm_fn):
    """
    在 on_ready 里调用，注册所有定时任务。
    bot: discord.ext.commands.Bot 实例
    call_llm_fn: tools.llm.call_llm 函数引用
    """
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except ImportError:
        log.error("APScheduler 未安装，请执行: pip install apscheduler")
        return None

    scheduler = AsyncIOScheduler(timezone="UTC")

    # ── 任务 1：每 2 小时检查 RSS 更新（有新内容才推送）──────────────────────
    scheduler.add_job(
        _rss_check_job,
        trigger="interval",
        hours=2,
        args=[bot, call_llm_fn],
        id="rss_check",
        name="RSS 更新检查",
        misfire_grace_time=300,   # 错过触发时间后 5 分钟内仍可执行
    )

    # ── 任务 2：每日早报（固定时间，有内容则发摘要）─────────────────────────
    scheduler.add_job(
        _daily_digest_job,
        trigger="cron",
        hour=DAILY_HOUR_UTC,
        minute=0,
        args=[bot, call_llm_fn],
        id="daily_digest",
        name="每日早报",
        misfire_grace_time=600,
    )

    scheduler.start()
    log.info(
        f"定时任务已启动：RSS 每2小时检查，"
        f"早报每天 {DAILY_HOUR_UTC:02d}:00 UTC（北京时间 {(DAILY_HOUR_UTC+8)%24:02d}:00）"
    )
    return scheduler


# ── RSS 检查任务 ──────────────────────────────────────────────────────────────
async def _rss_check_job(bot, call_llm_fn):
    """拉取 RSS，有新内容则推送，无则静默。"""
    if not DIGEST_CHANNEL_ID:
        return

    channel = bot.get_channel(DIGEST_CHANNEL_ID)
    if not channel:
        log.warning(f"找不到频道 {DIGEST_CHANNEL_ID}，RSS 推送跳过")
        return

    from tools.rss import fetch_all_feeds, mark_items_seen, format_digest
    items = await fetch_all_feeds()

    if not items:
        log.info("RSS 检查：无新内容")
        return

    log.info(f"RSS 检查：{len(items)} 条新内容，准备推送")

    # 用 LLM 生成一句话整体点评（可选，消耗少量 token）
    summary = ""
    try:
        titles = "、".join(i["title"] for i in items[:5])
        summary = await call_llm_fn(
            user_query=f"用一句话（20字以内）点评这些 AI 新闻的整体趋势：{titles}",
            system_prompt="你是简洁的新闻点评员，只输出一句话，不超过20字，语气活泼。",
        )
        summary = summary.strip().replace("\n", "")
    except Exception as e:
        log.warning(f"LLM 摘要生成失败（跳过）：{e}")

    message = format_digest(items, summary)

    # Discord 消息长度限制处理
    for chunk in _split_message(message):
        await channel.send(chunk)

    mark_items_seen(items)


# ── 每日早报任务 ──────────────────────────────────────────────────────────────
async def _daily_digest_job(bot, call_llm_fn):
    """每日固定时间推送，即使没有 RSS 新内容也发一条 AI 摘要。"""
    if not DIGEST_CHANNEL_ID:
        return

    channel = bot.get_channel(DIGEST_CHANNEL_ID)
    if not channel:
        return

    from tools.rss import fetch_all_feeds, mark_items_seen, format_digest
    from tools.search import web_search

    # 先拉 RSS
    rss_items = await fetch_all_feeds()

    # 再搜索补充
    try:
        search_results = await web_search("AI LLM 最新动态 今日")
    except Exception:
        search_results = ""

    # LLM 生成早报摘要
    rss_titles = "\n".join(f"- {i['title']}" for i in rss_items[:5])
    prompt = (
        f"今日 AI 资讯早报，请用 3 条要点总结（每条不超过 40 字）：\n\n"
        f"RSS 头条：\n{rss_titles or '（暂无）'}\n\n"
        f"搜索补充：\n{search_results[:500] or '（暂无）'}"
    )

    try:
        digest_text = await call_llm_fn(
            user_query=prompt,
            system_prompt=(
                "你是 AI 资讯助手，用 Discord Markdown 格式输出每日早报，"
                "3 条要点，每条一行，加 emoji，语气简洁有活力。"
            ),
        )
    except Exception as e:
        log.error(f"每日早报 LLM 调用失败：{e}")
        return

    now_bj = (datetime.now(timezone.utc).hour + 8) % 24
    header = f"🌅 **早安！今日 AI 速递** `{now_bj:02d}:00 北京时间`\n\n"
    full_msg = header + digest_text.strip()

    if rss_items:
        rss_section = format_digest(rss_items[:3])
        full_msg += f"\n\n{rss_section}"
        mark_items_seen(rss_items)

    for chunk in _split_message(full_msg):
        await channel.send(chunk)

    log.info("每日早报已推送")


def _split_message(text: str, limit: int = 1900) -> list[str]:
    if len(text) <= limit:
        return [text]
    return [text[i:i+limit] for i in range(0, len(text), limit)]
