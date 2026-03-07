# agent.py
# 接入关键词提炼，优化 input token 消耗

from tools.search import web_search, should_search, extract_search_query
from tools.llm import call_llm
from tools.prompt_builder import build_system_prompt
from db.memory import ConversationMemory

memory = ConversationMemory()


async def run_agent(
    user_id: str,
    channel_id: str,
    user_input: str,
    force_search: bool = False,
) -> str:
    # Step 1: 历史对话
    history = memory.get_history(user_id, channel_id)

    # Step 2: 搜索（含关键词提炼）
    search_context = ""
    if force_search or should_search(user_input):
        # 先提炼关键词，再搜索——降低搜索 noise
        search_query   = await extract_search_query(user_input, call_llm)
        search_context = await web_search(search_query)

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