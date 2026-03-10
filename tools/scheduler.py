# tools/scheduler.py
# 定时任务：RSS 智能推送 + 每日早报 + Memo 周报

import logging
import os
from datetime import datetime, timezone, timedelta

log = logging.getLogger("scheduler")

DAILY_HOUR_UTC = int(os.getenv("DAILY_HOUR_UTC", "1"))


# ── 工具函数（必须在所有 job 函数之前定义）────────────────────────────────────
def _split_message(text: str, limit: int = 1900) -> list[str]:
    if len(text) <= limit:
        return [text]
    return [text[i:i+limit] for i in range(0, len(text), limit)]


def _now_bj() -> str:
    """返回当前北京时间字符串，用于消息显示。"""
    bj = datetime.now(timezone.utc) + timedelta(hours=8)
    return bj.strftime("%m/%d %H:%M")


def _get_rss_channel(bot):
    """
    获取 RSS 推送目标频道。
    优先使用 RSS_CHANNELS（第一个），不存在则返回 None。
    统一入口，避免 DIGEST_CHANNEL_ID 和 RSS_CHANNELS 双轨并存。
    """
    raw = os.getenv("RSS_CHANNELS", "")
    ids = [s.strip() for s in raw.split(",") if s.strip()]
    if not ids:
        log.warning("未配置 RSS_CHANNELS，RSS 推送无目标频道")
        return None
    channel = bot.get_channel(int(ids[0]))
    if not channel:
        log.warning(f"RSS_CHANNELS 第一个频道 {ids[0]} 未找到（Bot 未缓存或 ID 有误）")
    return channel


async def _send_to_channel(channel, text: str) -> bool:
    """发送消息到指定频道对象，返回是否成功。"""
    if not channel:
        log.warning(f"找不到频道 {channel_id}，可能 Bot 尚未缓存")
        return False
    try:
        for chunk in _split_message(text):
            await channel.send(chunk)
        return True
    except Exception as e:
        log.error(f"发送失败（{channel.id}）：{e}")
        return False


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

    # RSS 检查：每 2 小时，整点触发
    scheduler.add_job(
        _rss_check_job,
        trigger="cron", minute=0, hour="*/2",
        args=[bot, call_llm_fn],
        id="rss_check", name="RSS 更新检查",
        misfire_grace_time=300,
    )

    # 每日早报：北京时间早上 9 点 = UTC 01:00
    scheduler.add_job(
        _daily_digest_job,
        trigger="cron", hour=DAILY_HOUR_UTC, minute=0,
        args=[bot, call_llm_fn],
        id="daily_digest", name="每日早报",
        misfire_grace_time=600,
    )

    # Memo 周报：每周日早报时间后 30 分钟
    scheduler.add_job(
        _memo_weekly_job,
        trigger="cron", day_of_week="sun",
        hour=DAILY_HOUR_UTC, minute=30,
        args=[bot, call_llm_fn],
        id="memo_weekly", name="Memo 周报",
        misfire_grace_time=600,
    )

    scheduler.start()
    bj_hour = (DAILY_HOUR_UTC + 8) % 24
    log.info(
        f"定时任务已启动："
        f"RSS 每2小时整点检查，"
        f"早报 UTC {DAILY_HOUR_UTC:02d}:00（北京 {bj_hour:02d}:00）"
    )
    return scheduler


# ── RSS 智能过滤 ──────────────────────────────────────────────────────────────
async def _score_items(items: list[dict], call_llm_fn) -> list[dict]:
    """
    让 LLM 对每条新闻打重要性分数（1-5），只推送 >= 3 分的。
    失败时降级：全部推送。
    """
    if not items:
        return items

    titles = "\n".join(f"{i+1}. {item['title']}" for i, item in enumerate(items))
    prompt = (
        f"以下是 AI 行业新闻标题，请为每条打重要性分数（1=普通更新，3=值得关注，5=重大突破）。\n"
        f"只输出 JSON 数组，格式：[分数1, 分数2, ...]，数量必须和标题数量一致。\n\n"
        f"{titles}"
    )
    try:
        import json
        result = await call_llm_fn(
            user_query=prompt,
            system_prompt="你是 AI 资讯评级助手，只输出 JSON 数组，不加任何其他内容。",
        )
        # 清理可能的 markdown 代码块
        result = result.strip().strip("```json").strip("```").strip()
        scores = json.loads(result)
        if len(scores) != len(items):
            raise ValueError("分数数量和新闻数量不一致")

        filtered = [item for item, score in zip(items, scores) if score >= 3]
        skipped  = len(items) - len(filtered)
        log.info(f"RSS 智能过滤：{len(items)} 条 → {len(filtered)} 条（过滤 {skipped} 条低重要性）")
        return filtered

    except Exception as e:
        log.warning(f"RSS 评分失败，降级为全部推送：{e}")
        return items


# ── RSS 检查任务 ──────────────────────────────────────────────────────────────
async def _rss_check_job(bot, call_llm_fn) -> str:
    """
    返回值供 /digest 命令显示状态：
    'no_new'   → 无新内容
    'ok:N'     → 成功推送 N 条
    'error:msg'→ 推送失败
    """
    from tools.rss import fetch_all_feeds, mark_items_seen, format_digest

    channel = _get_rss_channel(bot)
    if not channel:
        return "error:未配置 RSS_CHANNELS 或频道未找到"

    items = await fetch_all_feeds()
    if not items:
        log.info("RSS 检查：无新内容")
        return "no_new"

    log.info(f"RSS 检查：{len(items)} 条新内容，开始评分过滤")

    # 智能过滤
    items = await _score_items(items, call_llm_fn)
    if not items:
        log.info("RSS 检查：评分后无值得推送的内容")
        return "no_new"

    # LLM 一句话点评
    summary = ""
    try:
        titles  = "、".join(i["title"] for i in items[:5])
        summary = await call_llm_fn(
            user_query=f"用一句话（20字以内）点评这些 AI 新闻的整体趋势：{titles}",
            system_prompt="只输出一句话，不超过20字，语气活泼，不加前缀。",
        )
        summary = summary.strip().replace("\n", "")
    except Exception as e:
        log.warning(f"RSS 摘要生成失败：{e}")

    # 时间戳用北京时间，和频道里看到的一致
    now_str = _now_bj()
    message = format_digest(items, summary, now_str)

    ok = await _send_to_channel(channel, message)
    if not ok:
        return "error:消息发送失败"

    # 同步发到论坛频道（每篇文章一个独立帖子）
    forum_raw = os.getenv("FORUM_CHANNELS", "")
    if forum_raw:
        # 延迟导入避免循环
        from bot import post_to_forum
        for item in items:
            body = (
                f"**来源：** {item['tag']}\n"
                f"**链接：** {item['url']}\n\n"
                f"{item.get('summary', '')}"
            )
            await post_to_forum(
                title=item["title"][:100],
                content=body,
                tags=["AI", item["feed_name"][:20]],
            )

    mark_items_seen(items)
    log.info(f"RSS 推送完成：{len(items)} 条 → #{channel.name}")
    return f"ok:{len(items)}"


# ── 每日早报任务 ──────────────────────────────────────────────────────────────
async def _daily_digest_job(bot, call_llm_fn) -> str:
    from tools.rss import fetch_all_feeds, mark_items_seen, format_digest
    from tools.search import web_search

    channel = _get_rss_channel(bot)
    if not channel:
        return "error:未配置 RSS_CHANNELS 或频道未找到"

    rss_items = await fetch_all_feeds()

    search_results = ""
    try:
        search_results = await web_search("AI LLM 最新动态 今日")
    except Exception as e:
        log.warning(f"早报搜索失败：{e}")

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
                "你是 AI 资讯助手，用 Discord Markdown 输出每日早报，"
                "3 条要点，每条一行，加 emoji，严格基于提供资料。"
            ),
        )
    except Exception as e:
        log.error(f"每日早报 LLM 调用失败：{e}")
        return f"error:LLM 调用失败 {e}"

    # 北京时间标签（和定时触发时间一致）
    now_str  = _now_bj()
    header   = f"🌅 **早安！今日 AI 速递** `{now_str} 北京时间`\n\n"
    full_msg = header + digest_text.strip()

    if rss_items:
        full_msg += f"\n\n{format_digest(rss_items[:3], '', now_str)}"
        mark_items_seen(rss_items)

    ok = await _send_to_channel(channel, full_msg)
    if not ok:
        return "error:消息发送失败"

    log.info(f"每日早报已推送 → #{channel.name}")
    return "ok:daily"


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

    all_text = "\n".join(f"- {i['content']}" for i in items)
    try:
        summary = await call_llm_fn(
            user_query=f"以下是我这周随手记的想法，用一句话（30字以内）概括主题趋势：\n{all_text}",
            system_prompt="只输出一句话总结，不加任何前缀。",
        )
        summary = summary.strip()
    except Exception:
        summary = f"共 {len(items)} 条想法等待回顾"

    message = format_weekly_digest(owner_id, summary, items)

    # Memo 周报发到 CHAT_CHANNELS 第一个
    chat_raw = os.getenv("CHAT_CHANNELS", "")
    chat_ids = [s.strip() for s in chat_raw.split(",") if s.strip()]

    for chan_id in chat_ids:
        try:
            channel = bot.get_channel(int(chan_id))
            if channel:
                ok = await _send_to_channel(channel, message)
                if ok:
                    memo_store.mark_sent(owner_id)
                    log.info(f"Memo 周报已发送：{len(items)} 条 → #{channel.name}")
                    return f"ok:{len(items)}"
        except Exception as e:
            log.warning(f"Memo 周报发送失败 ({chan_id}): {e}")

    return "error:所有 CHAT_CHANNELS 发送失败"