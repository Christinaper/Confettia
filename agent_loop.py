# agent_loop.py
# 最小 Agent Loop 实现：tool_use 版本
#
# 架构对比（面试用）：
#   pipeline（agent.py）：  你的代码判断 should_search() → 调工具 → LLM
#   agent loop（本文件）：  LLM 判断要不要调工具 → 调工具 → LLM 继续
#
# 这里只有一个工具：web_search
# 复用 tools/search.py 和 tools/llm.py 的配置，不重复造轮子

import asyncio
import json
import logging
import os
import aiohttp
from dotenv import load_dotenv

from tools.search import web_search  # 复用现有搜索工具

load_dotenv()
log = logging.getLogger("agent_loop")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MAX_ITERATIONS = 3  # 防止死循环：最多 LLM→工具→LLM 循环 N 次

# ── 工具定义（告诉 LLM 有哪些工具可用）────────────────────────────────────────
# 这是发给 LLM 的"工具说明书"，格式由 DeepSeek API 规定
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "搜索互联网获取实时信息。"
                "适用场景：最新新闻、当前价格、近期事件、版本更新等需要实时数据的问题。"
                "不适用：解释概念、闲聊、你已经知道答案的问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，5-10个词，精炼简洁",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# ── 单次 LLM 调用（支持 tools 参数）──────────────────────────────────────────
async def _call_deepseek_with_tools(messages: list) -> dict:
    """
    调用 DeepSeek API，传入工具定义。
    返回原始 message 对象（不只是文本），因为需要判断 stop_reason。
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "tools": TOOLS,
        # auto = LLM 自己决定要不要调工具（也可以设 "none" 或 "required"）
        "tool_choice": "auto",
        # 每次最多调1个工具，防止并行乱搜导致 token 浪费和结果混乱
        "parallel_tool_calls": False,
        "max_tokens": 800,
        "temperature": 0.7,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"DeepSeek API 错误 {resp.status}: {text[:200]}")
            data = await resp.json()
            # 返回完整的 message 对象，包含 tool_calls 字段（如果有）
            return data["choices"][0]["message"]


# ── 执行工具调用 ───────────────────────────────────────────────────────────────
async def _execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    根据 LLM 的决定，实际执行工具。
    现在只有 web_search 一个工具；以后加工具在这里扩展。
    """
    if tool_name == "web_search":
        query = tool_args.get("query", "")
        log.info(f"[工具执行] web_search: '{query}'")
        result = await web_search(query)
        return result if result else "搜索无结果，请基于已有知识回答。"
    else:
        return f"未知工具: {tool_name}"


# ── Agent Loop 主函数 ──────────────────────────────────────────────────────────
async def run_agent_loop(user_input: str, system_prompt: str = "") -> str:
    """
    核心 agent loop：
      1. 把用户输入发给 LLM
      2. 如果 LLM 决定调工具 → 执行工具 → 把结果塞回 → 再次调 LLM
      3. 如果 LLM 直接回答 → 返回结果
      4. 最多循环 MAX_ITERATIONS 次防死循环
    """
    if not system_prompt:
        system_prompt = (
            "你是一个智能助手。你有搜索工具可以用。"
            "判断标准：问题涉及实时信息（新闻/价格/最新版本）时主动搜索；"
            "能直接回答的问题不要搜索，避免浪费。"
        )

    # messages 是对话历史，在 loop 中不断追加
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    for iteration in range(MAX_ITERATIONS):
        log.info(f"[Loop] 第 {iteration + 1} 次 LLM 调用")

        # ── LLM 调用 ──
        assistant_message = await _call_deepseek_with_tools(messages)

        # ── 判断 LLM 的决定 ──
        tool_calls = assistant_message.get("tool_calls")

        if not tool_calls:
            # LLM 决定直接回答，loop 结束
            log.info(f"[Loop] LLM 直接回答，结束（共 {iteration + 1} 次调用）")
            return assistant_message.get("content", "")

        # ── LLM 决定调工具 ──
        # 先把 LLM 的"我要调工具"这条消息追加到历史
        messages.append(assistant_message)

        # 执行所有工具调用（通常只有一个，但 API 支持并行多个）
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call["id"]

            log.info(f"[Loop] LLM 选择工具: {tool_name}({tool_args})")
            result = await _execute_tool(tool_name, tool_args)

            # 把工具结果追加到历史，格式固定（DeepSeek API 要求）
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,  # 对应上面的调用 ID
                "content": result,
            })

        # 继续 loop：带着工具结果再次调 LLM

    # 超过最大次数还没结束：让 LLM 用已有信息总结，不再给工具
    log.warning(f"[Loop] 达到最大迭代次数 {MAX_ITERATIONS}，触发兜底总结")
    messages.append({
        "role": "user",
        "content": "请根据你已经搜索到的信息，给出最终回答。不要再搜索了。"
    })
    fallback_payload = {
        "model": "deepseek-chat",
        "messages": messages,
        # 关键：不传 tools，强制 LLM 直接回答
        "max_tokens": 800,
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=fallback_payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
            return data["choices"][0]["message"].get("content", "信息不足，请换个方式提问。")


# ── 命令行测试入口 ─────────────────────────────────────────────────────────────
async def _demo():
    """
    用三个问题演示 agent loop 的行为差异：
      Q1: 闲聊 → LLM 应该直接回答，不调工具
      Q2: 需要实时信息 → LLM 应该调 web_search
      Q3: 技术概念 → LLM 应该直接回答，不调工具
    """
    test_cases = [
        "你好，介绍一下你自己",
        "2026年9月最新的AI模型有哪些值得关注的？",
        "什么是 RAG？用一句话解释",
    ]

    for q in test_cases:
        print(f"\n{'='*50}")
        print(f"问题: {q}")
        print("-" * 50)
        answer = await run_agent_loop(q)
        print(f"回答: {answer}")


if __name__ == "__main__":
    asyncio.run(_demo())