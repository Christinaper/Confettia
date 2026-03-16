# agent.py
# 决策层：搜索 + RAG + LLM + Memory 压缩
# 接入关键词提炼，优化 input token 消耗

import asyncio
import logging
from db.memory import ConversationMemory, COMPRESS_AFTER, RECENT_KEEP
from tools.search import web_search, should_search, extract_search_query
from tools.llm import call_llm
from tools.prompt_builder import build_system_prompt
from tools.rag import async_retrieve, is_rag_available

log    = logging.getLogger("agent")
memory = ConversationMemory()


async def run_agent(
    user_id: str,
    channel_id: str,
    user_input: str,
    force_search: bool = False,
    force_rag: bool = False,
) -> str:

    # Step 1：检查是否需要压缩
    await _maybe_compress(user_id, channel_id)

    # Step 2：读取历史（已含摘要层）
    history = memory.get_history(user_id, channel_id)

    # Step 3：并行搜索 + RAG
    need_search = force_search or should_search(user_input)
    need_rag    = force_rag or (
        is_rag_available() and _should_use_rag(user_input)
    )

    results = await asyncio.gather(
        _run_search(user_input) if need_search else _noop(),
        async_retrieve(user_input) if need_rag else _noop(),
        return_exceptions=True,
    )

    search_context = results[0] if isinstance(results[0], str) else ""
    rag_context    = results[1] if isinstance(results[1], str) else ""

    if rag_context:
        log.info(f"RAG 命中：{len(rag_context)} 字符")

    # Step 4：组装 Prompt
    user_note     = memory.get_user_note(user_id)
    system_prompt = build_system_prompt(
        user_note=user_note,
        has_search=bool(search_context),
        has_rag=bool(rag_context),
    )
    combined = _merge_contexts(search_context, rag_context)

    # Step 5：LLM 调用
    response = await call_llm(
        user_query=user_input,
        search_context=combined,
        chat_history=history,
        system_prompt=system_prompt,
    )

    # Step 6：保存本轮对话
    memory.save(user_id, channel_id, "user", user_input)
    memory.save(user_id, channel_id, "assistant", response)

    return response


async def _maybe_compress(user_id: str, channel_id: str):
    """
    超过阈值时自动触发压缩。
    用 LLM 生成摘要，把旧消息标记为 compressed。
    """
    count = memory.count_uncompressed(user_id, channel_id)
    if count < COMPRESS_AFTER:
        return

    log.info(f"触发 Memory 压缩：{count} 条未压缩消息")
    messages = memory.get_uncompressed(user_id, channel_id)

    # 构建压缩 prompt
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in messages[:-RECENT_KEEP]   # 不压缩最近 N 条
    )
    if not history_text.strip():
        return

    try:
        summary = await asyncio.wait_for(
            call_llm(
                user_query=(
                    f"请将以下对话压缩为简洁摘要（200字以内），"
                    f"保留关键信息、用户偏好和重要结论：\n\n{history_text}"
                ),
                system_prompt=(
                    "你是对话摘要助手。输出一段连贯的中文摘要，"
                    "包含：讨论的主要话题、得出的结论、用户表达的偏好。"
                    "不超过200字。"
                ),
            ),
            timeout=20.0,
        )
        ids = [m["id"] for m in messages]
        memory.compress(user_id, channel_id, summary.strip(), ids)
    except Exception as e:
        log.warning(f"Memory 压缩失败（跳过）：{e}")


async def _run_search(user_input: str) -> str:
    try:
        query  = await asyncio.wait_for(
            extract_search_query(user_input, call_llm), timeout=8.0
        )
        result = await asyncio.wait_for(web_search(query), timeout=12.0)
        return result
    except asyncio.TimeoutError:
        log.warning(f"搜索超时：'{user_input[:40]}'")
        return ""
    except Exception as e:
        log.warning(f"搜索失败：{e}")
        return ""


async def _noop() -> str:
    return ""


def _should_use_rag(user_input: str) -> bool:
    triggers = [
        "我之前", "我记得", "我的笔记", "我写过", "我记录",
        "根据我", "你知道我", "我学过", "我看过", "我研究",
        "my notes", "i wrote", "i mentioned",
    ]
    lower = user_input.lower()
    return any(t in lower for t in triggers)


def _merge_contexts(search: str, rag: str) -> str:
    parts = []
    if rag:
        parts.append(f"【来自你的 Obsidian 笔记】\n{rag}")
    if search:
        parts.append(f"【来自网络搜索】\n{search}")
    return ("\n\n" + "═" * 30 + "\n\n").join(parts) if parts else ""