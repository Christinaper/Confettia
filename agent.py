# agent.py
# 接入关键词提炼，优化 input token 消耗

import asyncio
import logging
from tools.search import web_search, should_search, extract_search_query
from tools.llm import call_llm
from tools.prompt_builder import build_system_prompt
from db.memory import ConversationMemory

log = logging.getLogger("agent")
memory = ConversationMemory()


async def run_agent(
    user_id: str,
    channel_id: str,
    user_input: str,
    force_search: bool = False,
) -> str:
    # Step 1: 历史对话
    history = memory.get_history(user_id, channel_id)

    # Step 2: 搜索（含关键词提炼）—— 超时后降级，不中断整个请求
    search_context = ""
    if force_search or should_search(user_input):
        try:
            search_query = await asyncio.wait_for(
                extract_search_query(user_input, call_llm),
                timeout=8.0,
            )
            search_context = await asyncio.wait_for(
                web_search(search_query),
                timeout=12.0,
            )
        except asyncio.TimeoutError:
            log.warning(f"搜索超时，降级为纯 LLM 回答。query='{user_input[:40]}'")
            search_context = ""   # 降级：没有搜索结果，LLM 用自身知识回答
        except Exception as e:
            log.warning(f"搜索异常（降级）：{e}")
            search_context = ""

    # Step 3: 组装三层 Prompt
    user_note     = memory.get_user_note(user_id)
    system_prompt = build_system_prompt(
        user_note=user_note,
        has_search=bool(search_context),
    )

    # Step 4: 调用 LLM
    response = await call_llm(
        user_query=user_input,
        search_context=search_context,
        chat_history=history,
        system_prompt=system_prompt,
    )

    # Step 5: 保存记忆
    memory.save(user_id, channel_id, "user", user_input)
    memory.save(user_id, channel_id, "model", response)

    return response