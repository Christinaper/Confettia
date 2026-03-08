# tools/review.py
# Review Skill：代码 review 或长文点评

import re
import logging

log = logging.getLogger("review")

# 触发条件
MIN_CHARS_FOR_REVIEW = 200   # 纯文本超过此长度触发 review 模式
CODE_BLOCK_PATTERN   = re.compile(r"```[\s\S]+?```")


def should_review(content: str) -> tuple[bool, str]:
    """
    判断消息是否触发 review 模式。
    返回 (是否触发, 类型: 'code' | 'article' | '')
    """
    if CODE_BLOCK_PATTERN.search(content):
        return True, "code"
    # 去掉 URL 后仍超过字数阈值，视为长文
    text_only = re.sub(r"https?://\S+", "", content).strip()
    if len(text_only) >= MIN_CHARS_FOR_REVIEW:
        return True, "article"
    return False, ""


def build_review_prompt(content: str, review_type: str) -> str:
    """根据内容类型构建 review prompt。"""
    if review_type == "code":
        # 提取所有代码块
        blocks = CODE_BLOCK_PATTERN.findall(content)
        code_section = "\n\n".join(blocks)
        extra = CODE_BLOCK_PATTERN.sub("", content).strip()
        prompt = (
            f"请对以下代码进行 Code Review，关注：\n"
            f"1. 潜在 bug 或逻辑错误\n"
            f"2. 安全问题（如有）\n"
            f"3. 可读性和命名\n"
            f"4. 一个具体的改进建议\n\n"
            f"代码：\n{code_section}"
        )
        if extra:
            prompt += f"\n\n用户补充说明：{extra}"
        return prompt

    else:  # article
        prompt = (
            f"请对以下内容进行点评，包含：\n"
            f"1. 核心观点（一句话）\n"
            f"2. 值得肯定的地方\n"
            f"3. 可以质疑或补充的角度\n"
            f"4. 你的总体评价\n\n"
            f"内容：\n{content}"
        )
        return prompt