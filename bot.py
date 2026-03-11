# bot.py
# 安全版：多层限流 + 输入过滤 + 预算感知 + 频道角色化

import discord
from discord.ext import commands, tasks
from discord import app_commands
import os
import asyncio
import logging
import time
from collections import defaultdict
from dotenv import load_dotenv
from agent import run_agent, memory
from tools.llm import get_usage_summary, is_budget_exceeded, call_llm
from tools.prompt_builder import reload_soul
from tools.scheduler import setup_scheduler, _rss_check_job, _daily_digest_job
from tools.memo import memo_store, format_weekly_digest
from tools.review import should_review, build_review_prompt

load_dotenv()

# ── 日志 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("bot")

# ── 代理 ──────────────────────────────────────────────────────────────────────
PROXY = os.getenv("PROXY", "")

# ── 安全参数（可通过 .env 调整）──────────────────────────────────────────────
MAX_INPUT_LEN    = int(os.getenv("MAX_INPUT_LEN",    "500"))   # 单条消息最大字符
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "30"))    # 用户冷却秒数
COOLDOWN_MAX     = int(os.getenv("COOLDOWN_MAX",     "2"))     # 冷却窗口内最大次数
GLOBAL_MAX_RPM   = int(os.getenv("GLOBAL_MAX_RPM",   "20"))    # 全局每分钟最大请求数

# ── 频道角色配置 ──────────────────────────────────────────────────────────────
# 每个频道 ID 只能属于一个角色，角色决定 Bot 的行为。
# 一个角色可以有多个频道 ID（逗号分隔），但推送类任务只发到第一个。
#
# 角色定义：
#   CHAT_CHANNELS    独聊：无需 @，无 quote，自由对话（猫娘）
#   RSS_CHANNELS     推送：接收 RSS 定时推送和每日早报，忽略用户消息
#                    ⚠️  推送任务只发到第一个 ID，多个 ID 仅作路由识别用
#   LOG_CHANNELS     日志：Bot 状态通知（启动/告警），忽略用户消息
#                    ⚠️  所有 LOG_CHANNELS 都会收到通知（广播模式）
#   TOPIC_CHANNELS   话题：含 URL 自动展开总结
#   MEMO_CHANNELS    备忘：存储碎片想法，加 reaction 不回复
#   REVIEW_CHANNELS  评审：代码/长文自动点评
#
# 多频道防重复规则：
#   - 对话类（CHAT/TOPIC/MEMO/REVIEW）：每条消息只在触发频道内响应，不跨频道
#   - 推送类（RSS）：只推到 RSS_CHANNELS 第一个频道
#   - 日志类（LOG）：广播到所有 LOG_CHANNELS（通常只配一个）
#
# 一个频道 ID 出现在多个角色里时，优先级：
#   chat > rss > log > topic > memo > review > mention（默认）

def _parse_channels(env_key: str) -> set[str]:
    raw = os.getenv(env_key, "")
    return {s.strip() for s in raw.split(",") if s.strip()}

CHAT_CHANNELS   = _parse_channels("CHAT_CHANNELS")
RSS_CHANNELS    = _parse_channels("RSS_CHANNELS")
LOG_CHANNELS    = _parse_channels("LOG_CHANNELS")
TOPIC_CHANNELS  = _parse_channels("TOPIC_CHANNELS")
MEMO_CHANNELS   = _parse_channels("MEMO_CHANNELS")
REVIEW_CHANNELS = _parse_channels("REVIEW_CHANNELS")
FORUM_CHANNELS  = _parse_channels("FORUM_CHANNELS")   # 论坛频道：每条 RSS 开独立帖子

# 兼容旧配置
CHAT_CHANNELS |= _parse_channels("SOLO_CHANNEL_IDS")

def get_channel_role(chan_id: str) -> str:
    if chan_id in CHAT_CHANNELS:   return "chat"
    if chan_id in RSS_CHANNELS:    return "rss"
    if chan_id in LOG_CHANNELS:    return "log"
    if chan_id in TOPIC_CHANNELS:  return "topic"
    if chan_id in MEMO_CHANNELS:   return "memo"
    if chan_id in REVIEW_CHANNELS: return "review"
    if chan_id in FORUM_CHANNELS:  return "forum"
    return "mention"

# ── Bot 初始化 ────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ── 限流状态（内存级）────────────────────────────────────────────────────────
# 用户冷却：{user_id: [(timestamp), ...]}
_user_timestamps: dict[str, list[float]] = defaultdict(list)
# 并发锁：防止同一用户重复处理
_processing: set[str] = set()
# 全局请求计数（滑动窗口）
_global_timestamps: list[float] = []
# 错误计数（恢复机制）
_error_count = 0
MAX_ERRORS = 5


# ── 限流函数 ──────────────────────────────────────────────────────────────────
def _check_user_cooldown(user_id: str) -> tuple[bool, float]:
    """
    检查用户是否在冷却中。
    返回 (是否允许, 还需等待秒数)
    """
    now = time.monotonic()
    window_start = now - COOLDOWN_SECONDS
    # 清理过期记录
    _user_timestamps[user_id] = [
        t for t in _user_timestamps[user_id] if t > window_start
    ]
    if len(_user_timestamps[user_id]) >= COOLDOWN_MAX:
        oldest = _user_timestamps[user_id][0]
        wait = COOLDOWN_SECONDS - (now - oldest)
        return False, round(wait, 1)
    return True, 0.0


def _record_user_request(user_id: str) -> None:
    _user_timestamps[user_id].append(time.monotonic())


def _check_global_rpm() -> bool:
    """全局每分钟请求数限制（防止服务器级别滥用）。"""
    now = time.monotonic()
    window_start = now - 60.0
    # 清理过期
    while _global_timestamps and _global_timestamps[0] < window_start:
        _global_timestamps.pop(0)
    if len(_global_timestamps) >= GLOBAL_MAX_RPM:
        return False
    _global_timestamps.append(now)
    return True


def _sanitize_input(text: str) -> tuple[bool, str]:
    """
    输入清洗：
    - 超长截断
    - 返回 (是否需要警告用户, 清洗后的文本)
    """
    text = text.strip()
    if len(text) > MAX_INPUT_LEN:
        return True, text[:MAX_INPUT_LEN]
    return False, text


# ── 恢复机制 ──────────────────────────────────────────────────────────────────
async def _recovery():
    global _error_count
    log.warning("🔄 触发恢复机制：清空处理队列...")
    _processing.clear()
    _error_count = 0
    await asyncio.sleep(5)
    log.info("🔄 恢复完成。")


# ── 启动事件 ──────────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    global _error_count
    _error_count = 0
    log.info(f"✅ Bot 上线：{bot.user} (ID: {bot.user.id})")
    log.info(f"   服务器：{len(bot.guilds)} 个")
    if PROXY:
        log.info(f"   代理：{PROXY}")
    try:
        await bot.wait_until_ready()
        total = 0
        for guild in bot.guilds:
            synced = await bot.tree.sync(guild=guild)
            total += len(synced)
            log.info(f"   Guild [{guild.name}] 同步：{len(synced)} 个命令")
            
        log.info(f"   Slash 命令同步：{len(synced)} 个（Guild 级别，立即生效）")
        # log.info(f"   Slash 命令同步：{len(synced)} 个")
    except Exception as e:
        log.error(f"   Slash 命令同步失败：{e}")
        await log_to_channel(f"⚠️ **Slash 命令同步失败**\n```{e}```")
    if not health_check.is_running():
        health_check.start()
    setup_scheduler(bot, call_llm)
    # 启动通知
    await log_to_channel(
        f"✅ **Confettia 上线** `{discord.utils.utcnow().strftime('%m/%d %H:%M UTC')}`\n"
        f"服务器：{len(bot.guilds)} 个｜Slash 命令：{len(synced)} 个已同步"
        # f"✅ **Confettia 上线** `{discord.utils.utcnow().strftime('%m/%d %H:%M UTC')}`\n"
        # f"服务器：{len(bot.guilds)} 个｜Slash 命令：已同步"
    )


@bot.event
async def on_disconnect():
    log.warning("⚠️  Bot 断开，等待自动重连...")

@bot.event
async def on_resumed():
    log.info("✅ 重连成功。")


# ── 全局错误处理 ──────────────────────────────────────────────────────────────
@bot.event
async def on_error(event, *args, **kwargs):
    global _error_count
    _error_count += 1
    log.error(f"未捕获异常（{event}，累计 {_error_count}）", exc_info=True)
    if _error_count >= MAX_ERRORS:
        await _recovery()


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    log.error(f"Slash 错误：{error}")
    msg = "❌ 命令执行出错，请稍后重试。"
    if isinstance(error, app_commands.CommandOnCooldown):
        msg = f"⏳ 冷却中，请 {error.retry_after:.0f} 秒后再试。"
    try:
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)
    except Exception:
        pass


# ── 健康检查 ──────────────────────────────────────────────────────────────────
@tasks.loop(minutes=10)
async def health_check():
    if len(_processing) > 10:
        log.warning(f"处理队列积压（{len(_processing)}），强制清空。")
        _processing.clear()
    usage = get_usage_summary()
    log.info(
        f"💓 健康检查｜Token：{usage['total']}/{usage['budget']} "
        f"({usage['pct']}%)｜调用：{usage['calls']}次｜队列：{len(_processing)}"
    )
    # token 用量超过 70% 时推送到 log 频道
    if usage["pct"] >= 70:
        await log_to_channel(
            f"⚠️ **Token 用量告警** {usage['pct']}%\n"
            f"`{usage['total']:,} / {usage['budget']:,}` tokens 已用\n"
            f"剩余：`{usage['remaining']:,}`｜调用：{usage['calls']} 次"
        )

@health_check.before_loop
async def before_health():
    await bot.wait_until_ready()


# ── 核心消息处理（@ 提及）────────────────────────────────────────────────────
@bot.event
async def on_message(message: discord.Message):
    global _error_count

    # Bot 消息一律忽略（防死循环）
    if message.author.bot:
        return

    chan_id  = str(message.channel.id)
    user_id  = str(message.author.id)
    role     = get_channel_role(chan_id)
    mentioned = bot.user in message.mentions

    # ── 按频道角色路由 ────────────────────────────────────────────────────────

    # RSS / LOG 频道：只接收 Bot 写入，用户消息一律忽略
    if role in ("rss", "log"):
        return

    # MEMO 频道：静默存库，不回复
    if role == "memo":
        await _handle_memo(message, user_id)
        return

    # REVIEW 频道：代码/长文自动点评
    if role == "review":
        await _handle_review(message, user_id, chan_id)
        return

    # TOPIC 频道：检测 URL，自动展开总结
    if role == "topic":
        has_url = "http://" in message.content or "https://" in message.content
        if not has_url and not mentioned:
            return
        await _handle_topic(message, user_id, chan_id)
        return

    # CHAT 频道：无需 @，直接对话
    if role == "chat":
        await _handle_chat(message, user_id, chan_id, quote=False)
        return

    # 默认（mention）：必须 @
    if not mentioned:
        await bot.process_commands(message)
        return
    await _handle_chat(message, user_id, chan_id, quote=True)
    await bot.process_commands(message)


def _token_usage_error():
    try:
        from tools.llm import _token_usage
        _token_usage["errors"] += 1
    except Exception:
        pass


# ── 核心对话处理 ──────────────────────────────────────────────────────────────
async def _handle_chat(
    message: discord.Message,
    user_id: str,
    chan_id: str,
    quote: bool,
):
    """chat / mention 频道的通用对话处理。"""
    lock_key = f"{user_id}:{chan_id}"

    if is_budget_exceeded():
        await _send(message, "🔴 **今日预算已达上限**，UTC 00:00 自动重置。")
        return
    if not _check_global_rpm():
        await message.add_reaction("🚫")
        return

    allowed, wait_sec = _check_user_cooldown(user_id)
    if not allowed:
        await _send(message, f"⏳ 请 **{wait_sec}s** 后再试。", delete_after=8)
        return
    if lock_key in _processing:
        await message.add_reaction("⏳")
        return

    raw = (
        message.content
        if not quote                                          # chat 频道：全文
        else message.content.replace(f"<@{bot.user.id}>", "").strip()
    )
    if not raw:
        return

    was_truncated, user_input = _sanitize_input(raw)
    _record_user_request(user_id)
    _processing.add(lock_key)

    try:
        async with message.channel.typing():
            response = await asyncio.wait_for(
                run_agent(user_id, chan_id, user_input),
                timeout=45.0,
            )
        global _error_count
        _error_count = 0
        if was_truncated:
            response = f"> ⚠️ 消息超过 {MAX_INPUT_LEN} 字符，已截取前段。\n\n" + response
        await send_long_message(message, response, quote=quote)

    except asyncio.TimeoutError:
        log.warning(f"超时 45s：uid={user_id} input='{user_input[:50]}'")
        await _send(message, "⏱️ 响应超时，搜索较慢时可能发生，请稍后重试。")
    except Exception as e:
        _error_count += 1
        _token_usage_error()
        log.error(f"对话失败（{_error_count}/{MAX_ERRORS}）：{e}", exc_info=True)
        await _send(message, "❌ 处理出错，请稍后重试。")
        if _error_count >= MAX_ERRORS:
            await _recovery()
    finally:
        _processing.discard(lock_key)


# ── 论坛频道：发帖工具 ────────────────────────────────────────────────────────
async def post_to_forum(title: str, content: str, tags: list[str] = None):
    """
    向所有 FORUM_CHANNELS 发布一个新帖子（Thread）。
    title:   帖子标题（Discord 论坛帖子必须有标题）
    content: 帖子正文
    tags:    论坛标签名列表（需要在 Discord 论坛里预先创建）
    """
    if not FORUM_CHANNELS:
        return

    for chan_id in FORUM_CHANNELS:
        try:
            channel = bot.get_channel(int(chan_id))
            if not channel:
                log.warning(f"论坛频道 {chan_id} 未找到")
                continue

            # 确认是论坛频道类型
            if not isinstance(channel, discord.ForumChannel):
                log.warning(f"频道 {chan_id} 不是 ForumChannel，跳过")
                continue

            # 匹配标签（ForumTag）
            applied_tags = []
            if tags:
                for tag_name in tags:
                    matched = discord.utils.get(channel.available_tags, name=tag_name)
                    if matched:
                        applied_tags.append(matched)

            # 发布帖子
            thread, _ = await channel.create_thread(
                name=title[:100],           # Discord 帖子标题限 100 字符
                content=content[:2000],
                applied_tags=applied_tags,
            )
            log.info(f"论坛帖子已发布：'{title[:30]}' → #{channel.name}/{thread.name}")

        except Exception as e:
            log.error(f"论坛发帖失败 ({chan_id})：{e}", exc_info=True)


# ── Log 频道通知 ──────────────────────────────────────────────────────────────
async def log_to_channel(text: str):
    """向所有 LOG_CHANNELS 发送一条通知，失败静默。"""
    if not LOG_CHANNELS:
        return
    for chan_id in LOG_CHANNELS:
        try:
            channel = bot.get_channel(int(chan_id))
            if channel:
                await channel.send(text)
        except Exception as e:
            log.warning(f"log_to_channel 失败 ({chan_id}): {e}")


# ── Memo 频道：静默存库 ───────────────────────────────────────────────────────
async def _handle_memo(message: discord.Message, user_id: str):
    """
    存入 memo，加 ✅ reaction 表示已收到，不发文字回复。
    保持频道界面干净。
    """
    content = message.content.strip()
    if not content:
        return
    count = memo_store.save(user_id, content)
    try:
        await message.add_reaction("✅")
        # 每存满 10 条，额外加一个提示 reaction
        if count % 10 == 0:
            await message.add_reaction("📓")
    except Exception:
        pass
    log.info(f"Memo 已存：uid={user_id}，当前共 {count} 条未整理")


# ── Review 频道：代码/长文点评 ────────────────────────────────────────────────
async def _handle_review(message: discord.Message, user_id: str, chan_id: str):
    """
    检测消息类型，触发对应的 review prompt。
    不触发条件（短文本、无代码块）则静默忽略。
    """
    triggered, review_type = should_review(message.content)
    if not triggered:
        return

    lock_key = f"{user_id}:{chan_id}"
    if lock_key in _processing:
        await message.add_reaction("⏳")
        return

    if is_budget_exceeded():
        await message.channel.send("🔴 今日预算已达上限，Review 暂停。")
        return

    prompt = build_review_prompt(message.content, review_type)
    icon   = "🔍" if review_type == "code" else "📝"

    _processing.add(lock_key)
    try:
        await message.add_reaction(icon)
        async with message.channel.typing():
            response = await asyncio.wait_for(
                run_agent(user_id, chan_id, prompt, force_search=False),
                timeout=45.0,
            )
        # review 频道不 quote，直接发，界面更干净
        await send_long_message(message, response, quote=False)
    except asyncio.TimeoutError:
        await message.channel.send("⏱️ Review 超时，请稍后重试。")
    except Exception as e:
        log.error(f"review 失败：{e}", exc_info=True)
        await message.channel.send("❌ Review 出错，请稍后重试。")
    finally:
        _processing.discard(lock_key)


# ── Topic 频道：URL 自动展开 ──────────────────────────────────────────────────
async def _handle_topic(message: discord.Message, user_id: str, chan_id: str):
    """
    检测消息里的 URL，抓取页面标题 + 摘要，
    让 LLM 总结并提出 3 个延伸问题。
    """
    import re
    urls = re.findall(r"https?://\S+", message.content)
    user_text = re.sub(r"https?://\S+", "", message.content).strip()
    user_text = user_text.replace(f"<@{bot.user.id}>", "").strip()

    if not urls:
        # 没有 URL 但有 @，当普通对话处理
        await _handle_chat(message, user_id, chan_id, quote=True)
        return

    url = urls[0]   # 只处理第一个 URL
    prompt = (
        f"请帮我总结以下链接的内容，并提出 3 个值得深入思考的延伸问题。\n\n"
        f"链接：{url}\n"
        + (f"用户补充：{user_text}" if user_text else "")
    )

    lock_key = f"{user_id}:{chan_id}"
    if lock_key in _processing:
        await message.add_reaction("⏳")
        return

    _processing.add(lock_key)
    try:
        async with message.channel.typing():
            response = await asyncio.wait_for(
                run_agent(user_id, chan_id, prompt, force_search=False),
                timeout=45.0,
            )
        await send_long_message(message, response, quote=False)
    except asyncio.TimeoutError:
        await message.channel.send("⏱️ 链接处理超时，请稍后重试。")
    except Exception as e:
        log.error(f"topic 处理失败：{e}", exc_info=True)
    finally:
        _processing.discard(lock_key)


# ── Slash：/search ────────────────────────────────────────────────────────────
@bot.tree.command(name="search", description="🔍 搜索最新资讯并用 AI 分析")
@app_commands.describe(query="搜索关键词（不超过 200 字符）")
@app_commands.checks.cooldown(1, 30.0)   # 每用户 30s 冷却
async def slash_search(interaction: discord.Interaction, query: str):
    user_id = str(interaction.user.id)

    if is_budget_exceeded():
        await interaction.response.send_message(
            "🔴 今日预算已达上限，请明天再试。", ephemeral=True
        )
        return

    if not _check_global_rpm():
        await interaction.response.send_message(
            "🚫 服务器请求过于频繁，请稍后再试。", ephemeral=True
        )
        return

    _, clean_query = _sanitize_input(query[:200])

    await interaction.response.defer(thinking=True)
    try:
        response = await asyncio.wait_for(
            run_agent(user_id, str(interaction.channel_id), clean_query, force_search=True),
            timeout=30.0,
        )
        await send_slash_response(interaction, response)
    except asyncio.TimeoutError:
        await interaction.followup.send("⏱️ 搜索超时，请稍后重试。")
    except Exception as e:
        log.error(f"/search 失败：{e}", exc_info=True)
        await interaction.followup.send("❌ 搜索出错，请稍后重试。")


# ── Slash：/note ─────────────────────────────────────────────────────────────
@bot.tree.command(name="note", description="📝 设置你的个人偏好（会影响 Bot 的回复风格）")
@app_commands.describe(preference="描述你的偏好，例如：请用英文回复 / 回答时多给代码示例")
async def slash_note(interaction: discord.Interaction, preference: str):
    """
    用户可以用这个命令告诉 Bot 自己的偏好。
    这段文字会作为 Layer 3 注入到每次的 System Prompt 里。
    """
    if len(preference) > 200:
        await interaction.response.send_message(
            "❌ 偏好描述不能超过 200 字符。", ephemeral=True
        )
        return
    memory.set_user_note(str(interaction.user.id), preference)
    await interaction.response.send_message(
        f"✅ 已保存你的偏好：\n> {preference}\n\n"
        f"从下一条消息开始生效。用 `/note-clear` 可以删除。",
        ephemeral=True,
    )

@bot.tree.command(name="note-clear", description="🗑️ 清除你设置的个人偏好")
async def slash_note_clear(interaction: discord.Interaction):
    memory.clear_user_note(str(interaction.user.id))
    await interaction.response.send_message("✅ 已清除个人偏好设置。", ephemeral=True)


# ── Slash：/reload（仅 Bot 主人可用）────────────────────────────────────────
OWNER_ID = int(os.getenv("OWNER_ID", "0"))  # 在 .env 里填你自己的 Discord User ID

@bot.tree.command(name="reload", description="🔄 热重载 soul.md（仅 Bot 主人）")
async def slash_reload(interaction: discord.Interaction):
    if OWNER_ID and interaction.user.id != OWNER_ID:
        await interaction.response.send_message("❌ 仅 Bot 主人可用。", ephemeral=True)
        return
    new_soul = reload_soul()
    await interaction.response.send_message(
        f"✅ soul.md 已重载（{len(new_soul)} 字符）。\n"
        f"新的人格设定从下一条消息开始生效。",
        ephemeral=True,
    )


# ── Slash：/digest（手动触发推送，仅 Bot 主人）────────────────────────────────
@bot.tree.command(name="digest", description="📡 立即拉取 RSS 并推送到指定频道（仅主人）")
@app_commands.describe(mode="rss=只推 RSS 新内容，daily=完整早报")
@app_commands.choices(mode=[
    app_commands.Choice(name="RSS 新内容", value="rss"),
    app_commands.Choice(name="每日早报",   value="daily"),
])
async def slash_digest(
    interaction: discord.Interaction,
    mode: app_commands.Choice[str] = None,
):
    if OWNER_ID and interaction.user.id != OWNER_ID:
        await interaction.response.send_message("❌ 仅 Bot 主人可用。", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=True)
    selected = mode.value if mode else "rss"

    try:
        if selected == "daily":
            result = await _daily_digest_job(bot, call_llm)
        else:
            result = await _rss_check_job(bot, call_llm)

        # 根据返回状态显示真实结果
        if result == "no_new":
            msg = "ℹ️ 无新内容（所有 RSS 源均无更新）。"
        elif result and result.startswith("ok:"):
            val = result[3:]
            msg = f"✅ 推送完成（{val}）。" if val != "daily" else "✅ 每日早报已推送。"
        elif result and result.startswith("error:"):
            msg = f"❌ 推送失败：{result[6:]}"
        else:
            msg = f"⚠️ 未知状态：{result}"

        await interaction.followup.send(msg, ephemeral=True)

    except Exception as e:
        log.error(f"/digest 失败：{e}", exc_info=True)
        await interaction.followup.send(f"❌ 执行失败：{e}", ephemeral=True)


# ── Slash：/forum-test（论坛频道测试，仅主人）─────────────────────────────────
@bot.tree.command(name="forum-test", description="🧵 向论坛频道发一条测试帖（仅主人）")
async def slash_forum_test(interaction: discord.Interaction):
    if OWNER_ID and interaction.user.id != OWNER_ID:
        await interaction.response.send_message("❌ 仅 Bot 主人可用。", ephemeral=True)
        return

    if not FORUM_CHANNELS:
        await interaction.response.send_message(
            "❌ 未配置 `FORUM_CHANNELS`，请在 `.env` 里添加论坛频道 ID。",
            ephemeral=True,
        )
        return

    await interaction.response.defer(ephemeral=True, thinking=True)

    results = []
    for chan_id in FORUM_CHANNELS:
        channel = bot.get_channel(int(chan_id))
        if not channel:
            results.append(f"❌ 频道 `{chan_id}` 未找到（ID 有误或 Bot 未缓存）")
            continue

        if not isinstance(channel, discord.ForumChannel):
            results.append(
                f"❌ `#{channel.name}` 不是论坛频道（当前类型：{type(channel).__name__}）\n"
                f"  → 请在 Discord 建立**论坛**类型的频道，不是普通文字频道"
            )
            continue

        # 显示已有标签
        available = [t.name for t in channel.available_tags]
        tag_names = ["AI", "Anthropic", "OpenAI"]
        applied   = []
        missing   = []
        for name in tag_names:
            matched = discord.utils.get(channel.available_tags, name=name)
            if matched:
                applied.append(matched)
            else:
                missing.append(name)

        try:
            thread, _ = await channel.create_thread(
                name="🧪 Bot 论坛功能测试帖",
                content=(
                    "这是一条由 Confettia 发布的测试帖 ✅\n\n"
                    "**论坛频道的用法：**\n"
                    "• 每条 RSS 文章 = 一个独立帖子，有标题，可以搜索\n"
                    "• 在帖子里回复 = 针对这篇文章的讨论，不会和其他文章混在一起\n"
                    "• 标签过滤：点击标签名可以筛选同类文章\n\n"
                    f"**已匹配标签：** {', '.join(t.name for t in applied) or '（无）'}\n"
                    f"**频道全部标签：** {', '.join(available) or '（未设置）'}"
                ),
                applied_tags=applied,
            )
            tag_status = ""
            if missing:
                tag_status = f"\n  ⚠️ 标签不存在（需在论坛频道手动创建）：`{'`、`'.join(missing)}`"
            results.append(
                f"✅ `#{channel.name}` 发帖成功\n"
                f"  帖子：**{thread.name}**{tag_status}"
            )
        except discord.Forbidden:
            results.append(
                f"❌ `#{channel.name}` 权限不足\n"
                f"  → 确认 Bot 在该频道有「发送消息」和「管理帖子」权限"
            )
        except Exception as e:
            results.append(f"❌ `#{channel.name}` 发帖失败：{e}")

    await interaction.followup.send("\n\n".join(results), ephemeral=True)


# ── Slash：/clear ─────────────────────────────────────────────────────────────
@bot.tree.command(name="clear", description="🗑️ 清除你在本频道的对话历史")
async def slash_clear(interaction: discord.Interaction):
    memory.clear(str(interaction.user.id), str(interaction.channel_id))
    await interaction.response.send_message("✅ 已清除对话历史。", ephemeral=True)


# ── Slash：/status ────────────────────────────────────────────────────────────
@bot.tree.command(name="status", description="📊 查看 Bot 运行状态和今日用量")
async def slash_status(interaction: discord.Interaction):
    u = get_usage_summary()
    msg_count = memory.count(str(interaction.user.id), str(interaction.channel_id))

    # 预算进度条（10 格）
    filled = round(u["pct"] / 10)
    bar = "█" * filled + "░" * (10 - filled)

    # 预算状态颜色
    status_icon = "🟢" if u["pct"] < 70 else ("🟡" if u["pct"] < 90 else "🔴")

    await interaction.response.send_message(
        f"📊 **Bot 状态** — {u['date']} (UTC)\n"
        f"```\n"
        f"LLM 提供商  : {u['provider'].upper()}\n"
        f"今日用量    : {u['total']:,} / {u['budget']:,} tokens\n"
        f"进度        : {status_icon} [{bar}] {u['pct']}%\n"
        f"剩余        : {u['remaining']:,} tokens\n"
        f"调用次数    : {u['calls']} 次\n"
        f"错误次数    : {u['errors']} 次\n"
        f"```\n"
        f"你的对话记录：`{msg_count}` 条　"
        f"处理队列：`{len(_processing)}` 条",
        ephemeral=True,
    )


# ── 辅助：发送消息（统一处理 quote 逻辑）────────────────────────────────────
async def _send(message: discord.Message, text: str, delete_after: float = None):
    """chat 频道用 channel.send，其他用 reply。"""
    if get_channel_role(str(message.channel.id)) == "chat":
        await message.channel.send(text, delete_after=delete_after)
    else:
        await message.reply(text, delete_after=delete_after)


async def send_long_message(message: discord.Message, text: str, quote: bool = True):
    """
    quote=False（独聊频道）：直接 channel.send，界面干净
    quote=True（普通频道）：reply 保持上下文关联
    """
    chan_id = str(message.channel.id)
    send_fn = message.reply if quote else message.channel.send

    if len(text) <= 1900:
        await send_fn(text)
        return
    chunks = [text[i:i+1900] for i in range(0, len(text), 1900)]
    for i, chunk in enumerate(chunks):
        if i == 0:
            await send_fn(chunk)
        else:
            await message.channel.send(chunk)


async def send_slash_response(interaction: discord.Interaction, text: str):
    if len(text) <= 1900:
        await interaction.followup.send(text)
        return
    for chunk in [text[i:i+1900] for i in range(0, len(text), 1900)]:
        await interaction.followup.send(chunk)


# ── 启动 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        log.critical("❌ 未找到 DISCORD_TOKEN！")
        exit(1)
    log.info("🚀 Bot 启动中...")
    bot.run(token, reconnect=True, log_handler=None)