#!/usr/bin/env python3
# test_baseline.py
# 验收基线测试
#
# 用法：
#   python test_baseline.py              # 跑所有测试
#   python test_baseline.py --type tool  # 只跑工具触发测试
#   python test_baseline.py --save       # 保存结果到 baseline_results/
#   python test_baseline.py --diff       # 和上次结果对比

import asyncio
import argparse
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path("baseline_results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── 测试用例定义 ──────────────────────────────────────────────────────────────

TEST_CASES = [

    # ── 类型 A：工具触发测试 ──────────────────────────────────────────────────
    {
        "id":       "tool_search_01",
        "type":     "tool",
        "desc":     "时效性问题应触发搜索",
        "input":    "DeepSeek 最新版本是什么",
        "expect": {
            "search_triggered": True,
            "rag_triggered":    False,
        },
    },
    {
        "id":       "tool_search_02",
        "type":     "tool",
        "desc":     "今日新闻应触发搜索",
        "input":    "今天 AI 行业有什么新动态",
        "expect": {
            "search_triggered": True,
        },
    },
    {
        "id":       "tool_no_search_01",
        "type":     "tool",
        "desc":     "闲聊不应触发搜索",
        "input":    "你好啊",
        "expect": {
            "search_triggered": False,
            "rag_triggered":    False,
        },
    },
    {
        "id":       "tool_no_search_02",
        "type":     "tool",
        "desc":     "解释概念不应触发搜索",
        "input":    "帮我解释一下什么是 asyncio",
        "expect": {
            "search_triggered": False,
        },
    },
    {
        "id":       "tool_rag_01",
        "type":     "tool",
        "desc":     "引用笔记触发词应触发 RAG",
        "input":    "我之前记录过关于 RAG 的笔记",
        "expect": {
            "rag_triggered": True,
        },
    },
    {
        "id":       "tool_rag_02",
        "type":     "tool",
        "desc":     "我写过应触发 RAG",
        "input":    "我写过一篇关于 Agent 架构的文章",
        "expect": {
            "rag_triggered": True,
        },
    },

    # ── 类型 B：内容质量测试 ──────────────────────────────────────────────────
    {
        "id":       "quality_no_hallucination_01",
        "type":     "quality",
        "desc":     "无搜索时不应编造具体数据",
        "input":    "2026年3月最新的 LLM 排行榜是什么",
        "expect": {
            "no_hallucination": True,   # LLM 自评：回答有没有无来源的具体数字/日期
        },
    },
    {
        "id":       "quality_honest_01",
        "type":     "quality",
        "desc":     "不确定时应该承认",
        "input":    "上周五 OpenAI 发布了什么",
        "expect": {
            "admits_uncertainty": True,  # 回答里有"不确定"/"建议搜索"等表述
        },
    },
    {
        "id":       "quality_concise_01",
        "type":     "quality",
        "desc":     "简单问题回答应简洁",
        "input":    "Python 的 list 和 tuple 有什么区别",
        "expect": {
            "max_chars": 600,   # 回答不超过 600 字符
        },
    },

    # ── 类型 C：性能测试 ──────────────────────────────────────────────────────
    {
        "id":       "perf_latency_no_search",
        "type":     "perf",
        "desc":     "无搜索响应时间应在 10s 内",
        "input":    "给我讲一个冷笑话",
        "expect": {
            "max_latency_ms": 10000,
        },
    },
    {
        "id":       "perf_token_efficiency",
        "type":     "perf",
        "desc":     "简单问题 token 消耗应低于 500",
        "input":    "1+1等于几",
        "expect": {
            "max_tokens": 500,
        },
    },
]


# ── 执行单个测试 ──────────────────────────────────────────────────────────────

async def run_test(case: dict, call_llm_fn, search_fn, rag_fn) -> dict:
    """执行一个测试用例，返回结果字典。"""
    result = {
        "id":        case["id"],
        "type":      case["type"],
        "desc":      case["desc"],
        "input":     case["input"],
        "passed":    True,
        "failures":  [],
        "metrics":   {},
        "response":  "",
        "ts":        datetime.now(timezone.utc).isoformat(),
    }

    start_ms = time.time() * 1000

    # 记录工具是否触发
    search_triggered = False
    rag_triggered    = False
    tokens_used      = 0

    try:
        from tools.search import should_search
        from tools.rag    import is_rag_available, _should_use_rag_for_test

        search_triggered = should_search(case["input"])
        rag_triggered    = (
            is_rag_available() and _should_use_rag_for_test(case["input"])
        )

        # 实际调用 LLM（不触发真实搜索，只测触发逻辑）
        from tools.llm import call_llm, get_usage_summary
        before = get_usage_summary()["total"]

        response = await asyncio.wait_for(
            call_llm(user_query=case["input"]),
            timeout=30.0,
        )

        after       = get_usage_summary()["total"]
        tokens_used = after - before

    except Exception as e:
        result["passed"] = False
        result["failures"].append(f"执行异常：{e}")
        return result

    end_ms  = time.time() * 1000
    latency = end_ms - start_ms

    result["response"] = response
    result["metrics"]  = {
        "latency_ms":       round(latency),
        "tokens":           tokens_used,
        "search_triggered": search_triggered,
        "rag_triggered":    rag_triggered,
        "response_chars":   len(response),
    }

    # ── 验证期望值 ────────────────────────────────────────────────────────────
    expect = case.get("expect", {})

    if "search_triggered" in expect:
        if search_triggered != expect["search_triggered"]:
            result["passed"] = False
            result["failures"].append(
                f"search_triggered: 期望 {expect['search_triggered']}，"
                f"实际 {search_triggered}"
            )

    if "rag_triggered" in expect:
        if rag_triggered != expect["rag_triggered"]:
            result["passed"] = False
            result["failures"].append(
                f"rag_triggered: 期望 {expect['rag_triggered']}，"
                f"实际 {rag_triggered}"
            )

    if "max_latency_ms" in expect:
        if latency > expect["max_latency_ms"]:
            result["passed"] = False
            result["failures"].append(
                f"latency: {round(latency)}ms > 上限 {expect['max_latency_ms']}ms"
            )

    if "max_tokens" in expect:
        if tokens_used > expect["max_tokens"]:
            result["passed"] = False
            result["failures"].append(
                f"tokens: {tokens_used} > 上限 {expect['max_tokens']}"
            )

    if "max_chars" in expect:
        if len(response) > expect["max_chars"]:
            result["passed"] = False
            result["failures"].append(
                f"response_chars: {len(response)} > 上限 {expect['max_chars']}"
            )

    # ── 质量指标：LLM 自评（只在 quality 类型里跑）─────────────────────────
    if case["type"] == "quality":
        quality_score = await _llm_quality_eval(
            case["input"], response, expect, call_llm_fn
        )
        result["metrics"]["quality_score"] = quality_score

        if "no_hallucination" in expect and quality_score.get("hallucination", 0) > 2:
            result["passed"] = False
            result["failures"].append(
                f"疑似编造：质量评分 hallucination={quality_score['hallucination']}"
            )

        if "admits_uncertainty" in expect:
            if not quality_score.get("admits_uncertainty", False):
                result["passed"] = False
                result["failures"].append("未承认不确定性")

    return result


async def _llm_quality_eval(question: str, response: str,
                             expect: dict, call_llm_fn) -> dict:
    """让 LLM 自评回答质量，返回结构化分数。"""
    prompt = f"""评估以下 AI 回答的质量，只输出 JSON，不加其他内容。

问题：{question}
回答：{response[:500]}

输出格式：
{{
  "hallucination": 1-5,         // 1=无编造 5=严重编造
  "admits_uncertainty": true/false,  // 是否承认了不确定性
  "relevance": 1-5,             // 1=完全不相关 5=高度相关
  "conciseness": 1-5            // 1=极度啰嗦 5=非常简洁
}}"""

    try:
        from tools.llm import call_llm
        result = await call_llm(
            user_query=prompt,
            system_prompt="你是 AI 输出质量评估员，只输出 JSON，不加任何前缀或解释。",
        )
        result = result.strip().strip("```json").strip("```").strip()
        return json.loads(result)
    except Exception:
        return {}


# ── 输出格式 ──────────────────────────────────────────────────────────────────

def _icon(passed: bool) -> str:
    return "✅" if passed else "❌"


def print_results(results: list[dict], verbose: bool = False):
    passed = sum(1 for r in results if r["passed"])
    total  = len(results)

    print(f"\n{'═'*60}")
    print(f"验收基线结果：{passed}/{total} 通过")
    print(f"{'═'*60}\n")

    # 按类型分组显示
    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    type_names = {"tool": "工具触发", "quality": "内容质量", "perf": "性能"}

    for t, cases in by_type.items():
        t_passed = sum(1 for c in cases if c["passed"])
        print(f"【{type_names.get(t, t)}】{t_passed}/{len(cases)} 通过")
        for r in cases:
            icon = _icon(r["passed"])
            m    = r["metrics"]
            metrics_str = (
                f"  {m.get('latency_ms', '?')}ms "
                f"| {m.get('tokens', '?')} tokens "
                f"| search={m.get('search_triggered', '?')} "
                f"| rag={m.get('rag_triggered', '?')}"
            )
            print(f"  {icon} [{r['id']}] {r['desc']}")
            print(f"     {metrics_str}")

            if not r["passed"]:
                for f in r["failures"]:
                    print(f"     ⚠️  {f}")

            if verbose and r.get("response"):
                print(f"     回答：{r['response'][:100]}…")

            if r.get("metrics", {}).get("quality_score"):
                qs = r["metrics"]["quality_score"]
                print(f"     质量：编造={qs.get('hallucination','?')} "
                      f"相关={qs.get('relevance','?')} "
                      f"简洁={qs.get('conciseness','?')}")
        print()


def save_results(results: list[dict]) -> Path:
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"baseline_{ts}.json"
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"📁 结果已保存：{path}")
    return path


def diff_results(results: list[dict]):
    """和上次结果对比，显示退步和进步。"""
    files = sorted(RESULTS_DIR.glob("baseline_*.json"))
    if len(files) < 2:
        print("需要至少两次结果才能对比，先跑两次 --save")
        return

    prev_path = files[-2]
    prev      = json.loads(prev_path.read_text())
    prev_map  = {r["id"]: r for r in prev}
    curr_map  = {r["id"]: r for r in results}

    regressions  = []
    improvements = []

    for test_id, curr in curr_map.items():
        prev_r = prev_map.get(test_id)
        if not prev_r:
            continue
        if prev_r["passed"] and not curr["passed"]:
            regressions.append(test_id)
        elif not prev_r["passed"] and curr["passed"]:
            improvements.append(test_id)

    print(f"\n📊 对比上次结果（{prev_path.name}）")
    if regressions:
        print(f"❌ 退步 {len(regressions)} 个：{', '.join(regressions)}")
    if improvements:
        print(f"✅ 进步 {len(improvements)} 个：{', '.join(improvements)}")
    if not regressions and not improvements:
        print("  → 和上次结果一致，无变化")


# ── 主入口 ────────────────────────────────────────────────────────────────────

async def main(args):
    from tools.llm    import call_llm
    from tools.search import web_search
    from tools.rag    import async_retrieve

    # 过滤测试类型
    cases = TEST_CASES
    if args.type:
        cases = [c for c in TEST_CASES if c["type"] == args.type]
        print(f"只跑 {args.type} 类型，共 {len(cases)} 个用例")

    print(f"开始跑 {len(cases)} 个验收基线测试...\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}...", end=" ", flush=True)
        result = await run_test(case, call_llm, web_search, async_retrieve)
        results.append(result)
        print(_icon(result["passed"]))

    print_results(results, verbose=args.verbose)

    if args.save:
        save_results(results)

    if args.diff:
        # 先保存当前结果再对比
        save_results(results)
        diff_results(results)

    # 有失败用例时退出码非零（方便 CI 检测）
    failed = sum(1 for r in results if not r["passed"])
    sys.exit(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验收基线测试")
    parser.add_argument("--type",    choices=["tool", "quality", "perf"],
                        help="只跑指定类型的测试")
    parser.add_argument("--save",    action="store_true", help="保存结果到文件")
    parser.add_argument("--diff",    action="store_true", help="和上次结果对比")
    parser.add_argument("--verbose", action="store_true", help="显示回答内容")
    args = parser.parse_args()
    asyncio.run(main(args))