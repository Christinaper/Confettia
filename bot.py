# bot.py
# 安全版：多层限流 + 输入过滤 + 预算感知状态显示

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

# ── 频道角色配置 ─────────────────────────────────────────────────────────────
# .env 里按角色分别填频道 ID（逗号分隔）：
#   CHAT_CHANNELS=111,222     独聊频道：无需 @，无 quote，自由对话
#   RSS_CHANNELS=333          推送频道：只收推送，不响应用户消息
#   LOG_CHANNELS=444          日志频道：Bot 写状态日志，不响应消息
#   TOPIC_CHANNELS=555        话题频道：检测 URL 自动展开总结
#
# 不在任何角色里的频道：必须 @，正常响应

def _parse_channels(env_key: str) -> set[str]:
    raw = os.getenv(env_key, "")
    return {s.strip() for s in raw.split(",") if s.strip()}

CHAT_CHANNELS  = _parse_channels("CHAT_CHANNELS")
RSS_CHANNELS   = _parse_channels("RSS_CHANNELS")
LOG_CHANNELS   = _parse_channels("LOG_CHANNELS")
TOPIC_CHANNELS = _parse_channels("TOPIC_CHANNELS")

# 兼容旧配置：SOLO_CHANNEL_IDS 等同于 CHAT_CHANNELS
_legacy = _parse_channels("SOLO_CHANNEL_IDS")
CHAT_CHANNELS |= _legacy

def get_channel_role(chan_id: str) -> str:
    """返回频道角色，默认 'mention'（需要 @ 才响应）。"""
    if chan_id in CHAT_CHANNELS:  return "chat"
    if chan_id in RSS_CHANNELS:   return "rss"
    if chan_id in LOG_CHANNELS:   return "log"
    if chan_id in TOPIC_CHANNELS: return "topic"
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
        synced = await bot.tree.sync()
        log.info(f"   Slash 命令同步：{len(synced)} 个")
    except Exception as e:
        log.error(f"   Slash 命令同步失败：{e}")
    if not health_check.is_running():
        health_check.start()
    # 启动定时任务（RSS + 每日早报）
    setup_scheduler(bot, call_llm)


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

@health_check.before_loop
async def before_health():
    await bot.wait_until_ready()


# ── 核心消息处理（@ 提及）────────────────────────────────────────────────────
@bot.event
async def on_message(message: discord.Message):
    global _error_count

    # [安全] Bot 消息一律忽略（防死循环）
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

    # TOPIC 频道：检测 URL，自动展开总结（需 @ 或含 URL）
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

    # [安全] 日预算熔断
    if is_budget_exceeded():
        await _send(message, "🔴 **今日预算已达上限**，UTC 00:00 自动重置。")
        return

    # [安全] 全局 RPM 限制
    if not _check_global_rpm():
        await message.add_reaction("🚫")
        return

    # [安全] 用户冷却
    allowed, wait_sec = _check_user_cooldown(user_id)
    if not allowed:
        await _send(message, f"⏳ 请 **{wait_sec}s** 后再试。", delete_after=8)
        return

    # [安全] 并发锁
    if lock_key in _processing:
        await message.add_reaction("⏳")
        return

    # 清洗输入：not quote频道去掉 @ 标签
    raw = (
        message.content
        if not quote                                          # chat 频道：全文
        else message.content.replace(f"<@{bot.user.id}>", "").strip()
    )
    if not raw:
        return  # 收到空消息（如纯表情）静默忽略

    was_truncated, user_input = _sanitize_input(raw_input)

    # 记录本次请求（冷却窗口计入）
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

        # 如果输入被截断，在回复前加提示
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
            await _daily_digest_job(bot, call_llm)
        else:
            await _rss_check_job(bot, call_llm)
        await interaction.followup.send("✅ 推送完成。", ephemeral=True)
    except Exception as e:
        log.error(f"/digest 失败：{e}", exc_info=True)
        await interaction.followup.send(f"❌ 推送失败：{e}", ephemeral=True)


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