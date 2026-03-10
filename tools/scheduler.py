# tools/scheduler.py
# 定时任务管理 — RSS 推送 + 每日早报 + Memo 周报

import logging
import os
from datetime import datetime, timezone

log = logging.getLogger("scheduler")

# 推送频道 ID（从 .env 读取）
DIGEST_CHANNEL_ID = int(os.getenv("DIGEST_CHANNEL_ID", "0"))
# 每日早报时间（UTC，北京时间 = UTC+8，早上9点 = UTC 01:00）
DAILY_HOUR_UTC    = int(os.getenv("DAILY_HOUR_UTC", "1"))


# ── 工具函数（必须在所有 job 函数之前定义）────────────────────────────────────
def _split_message(text: str, limit: int = 1900) -> list[str]:
    if len(text) <= limit:
        return [text]
    return [text[i:i+limit] for i in range(0, len(text), limit)]


async def _send_to_channel(bot, channel_id: int, text: str) -> bool:
    """发送消息到指定频道，返回是否成功。"""
    if not channel_id:
        log.warning("未配置 DIGEST_CHANNEL_ID，跳过推送")
        return False
    channel = bot.get_channel(channel_id)
    if not channel:
        log.warning(f"找不到频道 {channel_id}，可能 Bot 尚未缓存")
        return False
    for chunk in _split_message(text):
        await channel.send(chunk)
    return True


# ── Scheduler 注册 ────────────────────────────────────────────────────────────
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
        trigger="interval", hours=2,
        args=[bot, call_llm_fn],
        id="rss_check", name="RSS 更新检查",
        misfire_grace_time=300,
    )

    # ── 任务 2：每日早报（固定时间，有内容则发摘要）─────────────────────────
    scheduler.add_job(
        _daily_digest_job,
        trigger="cron", hour=DAILY_HOUR_UTC, minute=0,
        args=[bot, call_llm_fn],
        id="daily_digest", name="每日早报",
        misfire_grace_time=600,
    )

    # ── 任务 3：每周日 memo 周报 ──────────────────────────────────────────────
    scheduler.add_job(
        _memo_weekly_job,
        trigger="cron", day_of_week="sun",
        hour=DAILY_HOUR_UTC, minute=30,
        args=[bot, call_llm_fn],
        id="memo_weekly", name="Memo 周报",
        misfire_grace_time=600,
    )

    scheduler.start()
    log.info(
        f"定时任务已启动：RSS 每2小时检查，"
        f"早报每天 {DAILY_HOUR_UTC:02d}:00 UTC"
        f"（北京时间 {(DAILY_HOUR_UTC+8)%24:02d}:00）"
    )
    return scheduler


# ── RSS 检查任务 ──────────────────────────────────────────────────────────────
async def _rss_check_job(bot, call_llm_fn) -> str:
    """
    返回值供 /digest 命令显示状态：
    'no_new'   → 无新内容
    'ok:N'     → 成功推送 N 条
    'error:msg'→ 推送失败
    """
    from tools.rss import fetch_all_feeds, mark_items_seen, format_digest

    items = await fetch_all_feeds()
    if not items:
        log.info("RSS 检查：无新内容")
        return "no_new"

    log.info(f"RSS 检查：{len(items)} 条新内容，准备推送")

    # LLM 一句话点评（失败不阻塞推送）
    summary = ""
    try:
        titles = "、".join(i["title"] for i in items[:5])
        summary = await call_llm_fn(
            user_query=f"用一句话（20字以内）点评这些 AI 新闻的整体趋势：{titles}",
            system_prompt="你是简洁的新闻点评员，只输出一句话，不超过20字，语气活泼。",
        )
        summary = summary.strip().replace("\n", "")
    except Exception as e:
        log.warning(f"RSS 摘要生成失败（跳过）：{e}")

    message = format_digest(items, summary)

    try:
        ok = await _send_to_channel(bot, DIGEST_CHANNEL_ID, message)
        if not ok:
            return "error:找不到推送频道，请检查 DIGEST_CHANNEL_ID"
        mark_items_seen(items)
        return f"ok:{len(items)}"
    except Exception as e:
        log.error(f"RSS 推送失败：{e}", exc_info=True)
        return f"error:{e}"


# ── 每日早报任务 ──────────────────────────────────────────────────────────────
async def _daily_digest_job(bot, call_llm_fn) -> str:
    from tools.rss import fetch_all_feeds, mark_items_seen, format_digest
    from tools.search import web_search

    rss_items = await fetch_all_feeds()

    # 搜索补充（失败不阻塞）
    search_results = ""
    try:
        search_results = await web_search("AI LLM 最新动态 今日")
    except Exception as e:
        log.warning(f"早报搜索失败（跳过）：{e}")

    # LLM 生成早报摘要
    rss_titles = "\n".join(f"- {i['title']}" for i in rss_items[:5])
    prompt = (
        f"今日 AI 资讯早报，请用 3 条要点总结（每条不超过 40 字）：\n\n"
        f"RSS 头条：\n{rss_titles or '（暂无）'}\n\n"
        f"搜索补充：\n{search_results[:500] or '（暂无）'}\n\n"
        f"重要：只总结上面提供的内容，没有来源的信息不要编造，"
        f"如果内容不足就如实说今日暂无重要动态。"
    )

    try:
        digest_text = await call_llm_fn(
            user_query=prompt,
            system_prompt=(
                "你是 AI 资讯助手，用 Discord Markdown 格式输出每日早报，"
                "3 条要点，每条一行，加 emoji。"
                "严格基于提供的资料，没有来源的内容不要添加。"
            ),
        )
    except Exception as e:
        log.error(f"每日早报 LLM 调用失败：{e}")
        return f"error:LLM 调用失败 {e}"

    now_bj = (datetime.now(timezone.utc).hour + 8) % 24
    header = f"🌅 **早安！今日 AI 速递** `{now_bj:02d}:00 北京时间`\n\n"
    full_msg = header + digest_text.strip()

    if rss_items:
        rss_section = format_digest(rss_items[:3])
        full_msg += f"\n\n{rss_section}"
        mark_items_seen(rss_items)

    try:
        ok = await _send_to_channel(bot, DIGEST_CHANNEL_ID, full_msg)
        if not ok:
            return "error:找不到推送频道"
        log.info("每日早报已推送")
        return "ok:daily"
    except Exception as e:
        log.error(f"早报推送失败：{e}", exc_info=True)
        return f"error:{e}"


# ── Memo 周报任务 ─────────────────────────────────────────────────────────────
async def _memo_weekly_job(bot, call_llm_fn) -> str:
    from tools.memo import memo_store, format_weekly_digest

    # 找所有有未发送 memo 的用户
    # 简化：只处理 OWNER_ID，个人 Bot 场景够用
    owner_id = os.getenv("OWNER_ID", "")
    if not owner_id:
        return "error:未配置 OWNER_ID"

    items = memo_store.get_unsent(owner_id)
    if not items:
        log.info("Memo 周报：无新内容")
        return "no_new"

    # LLM 生成摘要
    all_text = "\n".join(f"- {i['content']}" for i in items)
    try:
        summary = await call_llm_fn(
            user_query=f"以下是我这周随手记的想法，用一句话（30字以内）概括主题趋势：\n{all_text}",
            system_prompt="你是简洁的笔记助手，只输出一句话总结，不加任何前缀。",
        )
        summary = summary.strip()
    except Exception:
        summary = f"共 {len(items)} 条想法等待回顾"

    message = format_weekly_digest(owner_id, summary, items)

    # 发到所有 CHAT_CHANNELS
    chat_ids = os.getenv("CHAT_CHANNELS", "")
    for chan_id in (s.strip() for s in chat_ids.split(",") if s.strip()):
        try:
            channel = bot.get_channel(int(chan_id))
            if channel:
                for chunk in _split_message(message):
                    await channel.send(chunk)
                memo_store.mark_sent(owner_id)
                log.info(f"Memo 周报已发送：{len(items)} 条")
                return f"ok:{len(items)}"
        except Exception as e:
            log.warning(f"Memo 周报发送失败 ({chan_id}): {e}")

    return "error:所有 chat 频道发送失败"