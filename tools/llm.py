# tools/llm.py
# 安全版: 输入截断 + token 计数 + 日预算熔断
# 支持动态 system_prompt 传入 (由 prompt_builder 组装)

import os
import logging
import aiohttp
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("llm")

# ── API 配置 ──────────────────────────────────────────────────────────────────
DEEPSEEK_KEY       = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_KEY         = os.getenv("GEMINI_API_KEY", "")
ACTIVE_PROVIDER    = "deepseek" if DEEPSEEK_KEY else ("gemini" if GEMINI_KEY else "none")
DAILY_TOKEN_BUDGET = int(os.getenv("DAILY_TOKEN_BUDGET", "15000"))
# ── 安全参数 ──────────────────────────────────────────────────────────────────
MAX_OUTPUT_TOKENS  = 800    # 单次最大输出 token
MAX_HISTORY_TURNS  = 6      # 最多保留 6 条历史 (3 轮对话)
MAX_INPUT_CHARS    = 1500   # prompt 最大字符数 (含搜索结果)

log.info(f"LLM Provider: {ACTIVE_PROVIDER.upper()}")

# ── 全局 token 计数器 (内存级, 重启归零 )────────────────────────────────────
_token_usage = {"date": "", "input": 0, "output": 0, "calls": 0, "errors": 0}

def _reset_if_new_day():
    """UTC 日期变更时重置计数器。"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _token_usage["date"] != today:
        log.info(f"重置 token 计数。昨日: in={_token_usage['input']} out={_token_usage['output']}")
        _token_usage.update({"date": today, "input": 0, "output": 0, "calls": 0, "errors": 0})

def _total_today():
    return _token_usage["input"] + _token_usage["output"]

def _add_usage(inp: int, out: int):
    _token_usage["input"]  += inp
    _token_usage["output"] += out
    _token_usage["calls"]  += 1
    log.info(f"Token: +{inp}in +{out}out | 今日 {_total_today()}/{DAILY_TOKEN_BUDGET}")

def get_usage_summary() -> dict:
    """供 /status 命令调用, 返回当日用量摘要。"""
    _reset_if_new_day()
    total = _total_today()
    return {
        "provider":  ACTIVE_PROVIDER,
        "date":      _token_usage["date"],
        "input":     _token_usage["input"],
        "output":    _token_usage["output"],
        "total":     total,
        "budget":    DAILY_TOKEN_BUDGET,
        "remaining": max(0, DAILY_TOKEN_BUDGET - total),
        "calls":     _token_usage["calls"],
        "errors":    _token_usage["errors"],
        "pct":       min(100, round(total / max(1, DAILY_TOKEN_BUDGET) * 100, 1)),
    }

def is_budget_exceeded() -> bool:
    _reset_if_new_day()
    return _total_today() >= DAILY_TOKEN_BUDGET

# ── 输入截断 ──────────────────────────────────────────────────────────────────
def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    log.warning(f"输入截断: {len(text)} → {limit}")
    return text[:limit] + "\n[已截断]"

# ── DeepSeek 调用 ─────────────────────────────────────────────────────────────
async def _call_deepseek(prompt: str, history: list, system_prompt: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-MAX_HISTORY_TURNS:]:
        role = h.get("role", "user").replace("model", "assistant")
        messages.append({"role": role, "content": h["parts"][0]})
    messages.append({"role": "user", "content": prompt})

    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload  = {"model": "deepseek-chat", "messages": messages,
                "max_tokens": MAX_OUTPUT_TOKENS, "temperature": 0.7, "stream": False}

    async with aiohttp.ClientSession() as s:
        async with s.post("https://api.deepseek.com/chat/completions",
                          headers=headers, json=payload,
                          timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 402:
                log.error("DeepSeek 余额不足")
                return "❌ **DeepSeek 余额不足**，请登录 platform.deepseek.com 充值。"
            if resp.status == 429:
                return "⚠️ 请求过频，请稍等片刻。"
            if resp.status != 200:
                txt = await resp.text()
                log.error(f"DeepSeek HTTP {resp.status}: {txt[:200]}")
                _token_usage["errors"] += 1
                return f"❌ API 错误 {resp.status}，请稍后重试。"
            data  = await resp.json()
            usage = data.get("usage", {})
            _add_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            return data["choices"][0]["message"]["content"]

# ── Gemini 调用 (无 DeepSeek Key 时的后备 )───────────────────────────────────
async def _call_gemini(prompt: str, history: list, system_prompt: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "❌ 请执行 `pip install google-genai`。"
    client   = genai.Client(api_key=GEMINI_KEY)
    contents = [
        types.Content(role=h.get("role","user"), parts=[types.Part(text=h["parts"][0])])
        for h in history[-MAX_HISTORY_TURNS:]
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash", contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS, temperature=0.7,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        # Gemini 不返回精确 token 数, 用字符数粗估(1 token ≈ 4 字符)
        _add_usage(len(prompt)//4, len(response.text)//4)
        return response.text
    except Exception as e:
        err = str(e)
        _token_usage["errors"] += 1
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            return "⚠️ **Gemini 配额耗尽**，请在 `.env` 添加 `DEEPSEEK_API_KEY` 后重启。"
        return f"❌ Gemini 错误：{err[:100]}"

# ── 统一入口 ──────────────────────────────────────────────────────────────────
async def call_llm(
    user_query: str,
    search_context: str = "",
    chat_history: list = [],
    system_prompt: str = "",
) -> str:
    # 熔断检查: 日预算超限
    if is_budget_exceeded():
        return "🔴 **今日 API 预算已达上限**, UTC 00:00(北京时间 08:00)自动重置。"

    if not system_prompt:
        system_prompt = "你是专业的 Discord 信息助手, 用 Markdown 格式简洁回复, 不超过800字符。"

    prompt = _truncate(
        f"用户问题：{user_query}\n\n"
        f"参考资料：\n{'─'*28}\n{search_context}\n{'─'*28}\n\n"
        f"要求：严格基于以上参考资料回答。资料中没有的具体数据不要编造，"
        f"如资料不足请直接说明。"
        if search_context else user_query,
        MAX_INPUT_CHARS,
    )

    if ACTIVE_PROVIDER == "deepseek":
        return await _call_deepseek(prompt, chat_history, system_prompt)
    elif ACTIVE_PROVIDER == "gemini":
        return await _call_gemini(prompt, chat_history, system_prompt)
    else:
        return "❌ **未配置任何 LLM API Key**, 请检查 `.env`。"