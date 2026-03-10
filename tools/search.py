# tools/search.py
# 使用新包名 ddgs（原 duckduckgo_search 已更名）
# pip install ddgs

import asyncio
import logging

log = logging.getLogger("search")

SEARCH_TRIGGERS = [
    # 明确要求搜索或查询实时信息
    "搜索", "查一下", "查查", "帮我查",
    # 时效性强的词
    "最新", "最近", "今天", "今日", "现在", "当前", "目前",
    "刚刚", "新闻", "动态", "发布", "上线", "更新",
    # 价格/数据类
    "价格", "多少钱", "汇率", "股价",
    # 明确年份（近年事件）
    "2025", "2026",
    # 英文
    "latest", "recent", "news", "search", "current", "today",
]

# 即使包含触发词也不搜索的场景（闲聊、解释概念等）
SEARCH_EXCLUDES = [
    "你觉得", "你认为", "你喜欢", "你是", "你好",
    "什么意思", "怎么理解", "解释", "帮我写", "帮我改",
    "翻译", "总结一下", "分析一下",
]

MAX_RESULT_CHARS = 800   # 每条搜索结果最大字符数
MAX_RESULTS      = 3     # 最多取几条结果


def should_search(user_input: str) -> bool:
    lower = user_input.lower()
    # 先检查排除词——命中排除词则不搜索
    if any(t in lower for t in SEARCH_EXCLUDES):
        return False
    return any(t in lower for t in SEARCH_TRIGGERS)


def _trim_result(text: str) -> str:
    """截断单条结果，去除 noise。"""
    text = text.strip()
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "…"
    return text


async def web_search(query: str, max_results: int = MAX_RESULTS) -> str:
    """
    异步搜索。使用新包名 ddgs。
    传入 query 应已是精炼后的关键词（由 extract_search_query 处理）。
    """
    loop = asyncio.get_event_loop()

    def _sync_search():
        try:
            from ddgs import DDGS
        except ImportError:
            # 兼容旧包名，给出明确提示
            raise ImportError(
                "请执行: pip uninstall duckduckgo_search -y && pip install ddgs"
            )

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "").strip()
                body  = _trim_result(r.get("body", ""))
                href  = r.get("href", "")
                # 只保留有实质内容的结果
                if body and len(body) > 30:
                    results.append(f"**{title}**\n{body}\n来源: {href}")
        return results

    try:
        results = await loop.run_in_executor(None, _sync_search)
    except ImportError as e:
        return f"⚠️ 搜索工具未安装：{e}"
    except Exception as e:
        log.warning(f"搜索失败：{e}")
        return f"搜索暂时不可用（{str(e)[:60]}），将基于已有知识回答。"

    if not results:
        return ""   # 返回空字符串，让 LLM 基于自身知识回答

    combined = "\n\n---\n\n".join(results[:MAX_RESULTS])
    log.info(f"搜索完成：{len(results)} 条结果，{len(combined)} 字符")
    return combined


async def extract_search_query(user_input: str, call_llm_fn) -> str:
    """
    用 LLM 从用户输入中提炼搜索关键词。
    这一步本身消耗少量 token，但能显著减少搜索 noise，
    从而减少后续主调用的 input token，整体是合算的。

    例：
      输入："你觉得最近有没有什么比 Claude 更厉害的模型出来？"
      输出："2026 最新 LLM 模型 超越 Claude"
    """
    prompt = (
        f"从以下用户问题中提炼出最适合用于网络搜索的关键词（英文或中文均可，"
        f"5–10个词，不要句子，只输出关键词）：\n\n{user_input}"
    )
    try:
        # 直接调用 LLM，不带历史和搜索上下文，极低 token 消耗
        keywords = await call_llm_fn(
            user_query=prompt,
            search_context="",
            chat_history=[],
            system_prompt="你是搜索关键词提炼助手，只输出关键词，不解释，不加标点。",
        )
        keywords = keywords.strip().replace("\n", " ")
        log.info(f"关键词提炼：'{user_input[:30]}…' → '{keywords}'")
        return keywords
    except Exception as e:
        log.warning(f"关键词提炼失败，使用原始输入：{e}")
        return user_input  # 降级到原始输入